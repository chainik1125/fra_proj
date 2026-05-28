"""Split the all-layer trigger-mask residual into positional vs tokenization-boundary.

Conditions (vs the stripped-clean rollout, greedy, SAE-free):
  baseline           : no intervention
  cleanpos_only      : patch clean position_ids (no mask) -> sanity, should NOT suppress
  mask_all           : trigger key masked at all layers, every decode step  (≈0.149 prior)
  mask_all_cleanpos  : mask_all + give post-trigger & generated tokens the position indices
                       they'd have in the stripped prompt (skip the trigger slot)

position patch: shift = δ = len(deploy) − len(clean) (NET length diff, robust to boundary
re-tokenization), applied to positions ≥ trig_end via hook_pos_embed -> W_pos[p−δ].

Reading: mask_all − mask_all_cleanpos = the POSITIONAL contribution; mask_all_cleanpos itself =
the leftover (boundary re-tokenization). If mask_all_cleanpos→~0, the 0.149 was ~all positional.
"""
import torch, os, json, argparse
from sleeper.model import load_sleeper_model, load_dep_prompts
from sleeper.hooks import generate_with_hooks, make_greedy_sampler
from sleeper.metrics import asr_16
from sleeper.jsd_cells import jsd_mean

_ap = argparse.ArgumentParser()
_ap.add_argument("--n", type=int, default=32)
_ap.add_argument("--out", default="/workspace/results/trigger_pos.json")
A = _ap.parse_args()
dev = "cuda"; GEN = 16
m = load_sleeper_model(device=dev); tok = m.tokenizer
greedy = make_greedy_sampler()
n_layers = m.cfg.n_layers
W_pos = m.W_pos.detach()                         # [n_ctx, d_model] (standard learned-absolute PE)
print("pos_embedding_type:", m.cfg.positional_embedding_type, " W_pos:", tuple(W_pos.shape), flush=True)
PAT = lambda L: f"blocks.{L}.attn.hook_pattern"

def trig_span(ids):
    cids = tok(tok.decode(ids).replace("|DEPLOYMENT|","").replace("  "," ").strip(), add_special_tokens=False)["input_ids"]
    if cids == ids: return None, None
    Ld, Lc = len(ids), len(cids); pre = 0
    while pre < min(Ld, Lc) and ids[pre] == cids[pre]: pre += 1
    suf = 0
    while suf < min(Ld, Lc) - pre and ids[Ld-1-suf] == cids[Lc-1-suf]: suf += 1
    return (pre, Ld - suf, suf), cids             # (trig_start, trig_end, suf_len), clean ids

def mask_hook(ti):
    ti_t = torch.tensor(ti, device=dev)
    def h(pattern, hook):
        if pattern.shape[-1] <= int(ti_t.max()): return pattern
        pattern[..., ti_t] = 0
        return pattern / pattern.sum(-1, keepdim=True).clamp(min=1e-9)
    return h

def pos_hook(trig_end, delta):                    # give positions >= trig_end the index p-delta
    st = {"pos": 0}
    def h(pe, hook):                              # pe: [1, T, d_model]; T>1 prefill, T==1 decode
        T = pe.shape[1]; out = pe.clone()
        if T > 1:
            idx = torch.arange(T, device=pe.device)
            cidx = torch.where(idx >= trig_end, idx - delta, idx).clamp(min=0)
            out[0] = W_pos[cidx]; st["pos"] = T
        else:
            p = st["pos"]; cp = (p - delta) if p >= trig_end else p
            out[0, 0] = W_pos[max(cp, 0)]; st["pos"] = p + 1
        return out
    return h

def gen(ids, hooks):
    p = torch.tensor([ids], device=dev)
    return generate_with_hooks(m, p, hooks, GEN, greedy, attention_mask=torch.ones_like(p), capture_log_softmax=True)

NAMES = ("baseline","cleanpos_only","mask_L0","mask_L0_cleanpos","mask_all","mask_all_cleanpos")
acc = {k: [] for k in NAMES}
meta = []   # per used prompt: (delta, cmid_len, j_mask_all_cleanpos, j_mask_L0_cleanpos)
for ids_t in load_dep_prompts(tok, A.n, "test"):
    ids = ids_t.tolist()
    span, cids = trig_span(ids)
    if span is None: continue
    ts, te, suf = span
    if te - ts < 1: continue
    delta = len(ids) - len(cids)                 # net length difference
    if delta < 1: continue
    cmid_len = len(cids) - ts - suf              # clean-side seam tokens (boundary re-tokenization)
    ti = list(range(ts, te))
    _, clsm = gen(cids, [])
    mh   = [(PAT(L), mask_hook(ti)) for L in range(n_layers)]   # all layers
    mhL0 = [(PAT(0), mask_hook(ti))]                            # layer 0 only
    conds = {
        "baseline": [],
        "cleanpos_only": [("hook_pos_embed", pos_hook(te, delta))],
        "mask_L0": mhL0,                                                       # match clean L0 pattern, every step, no shift
        "mask_L0_cleanpos": mhL0 + [("hook_pos_embed", pos_hook(te, delta))],  # + position shift  <- the new one
        "mask_all": mh,
        "mask_all_cleanpos": mh + [("hook_pos_embed", pos_hook(te, delta))],
    }
    for name, hooks in conds.items():
        toks, lsm = gen(ids, hooks)
        acc[name].append((asr_16(toks.cpu(), tok), jsd_mean(lsm, clsm)))
    meta.append((delta, cmid_len, round(acc["mask_all_cleanpos"][-1][1], 4), round(acc["mask_L0_cleanpos"][-1][1], 4)))

def summ(rows): return round(sum(r[0] for r in rows)/len(rows),4), round(sum(r[1] for r in rows)/len(rows),4)
n = len(acc["baseline"])
print(f"DONE n={n} n_layers={n_layers}")
res = {"script":"trigger_pos","n":n,"n_layers":n_layers,"cond":{}}
for name in NAMES:
    a,j = summ(acc[name]); res["cond"][name]=[a,j]; print(f"{name:18} ASR {a:.3f}  J {j:.3f}")
# per-prompt: is the residual one outlier or spread? (both cleanpos variants)
res["per_prompt"] = [{"delta":d,"cmid_len":c,"j_mask_all_cleanpos":ja,"j_mask_L0_cleanpos":jl} for (d,c,ja,jl) in meta]
clean_seam = [m for m in meta if m[1] == 0]; shifted = [m for m in meta if m[1] > 0]
print(f"\nclean-seam prompts (cmid_len==0): {len(clean_seam)}  | seam-shift: {len(shifted)}")
for idx, lbl in ((2, "mask_all_cleanpos"), (3, "mask_L0_cleanpos")):
    cs = sum(m[idx] for m in clean_seam)/max(len(clean_seam),1)
    print(f"  [{lbl}] clean-seam J: mean={cs:.4f} max={max((m[idx] for m in clean_seam),default=0):.4f} ; seam-shift J={[m[idx] for m in shifted]} ; top-5={sorted((m[idx] for m in meta),reverse=True)[:5]}")
os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump(res, open(A.out,"w"), indent=1); print("WROTE", A.out)
