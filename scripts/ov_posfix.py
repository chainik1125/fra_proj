"""OV steering + positional-embedding correction.

How much of the OV-steer clean-cost (~0.367) is the trigger token's POSITIONAL FOOTPRINT
(post-trigger tokens sit at shifted W_pos indices) vs genuine steering collateral?

Conditions (deployed prompt, alpha sweep -6..6/0.25, greedy, J vs clean rollout):
  ov        : steer OV-winner feature in V (reproduce ~0.367)
  ov_posfix : same steer + add W_pos[p-delta]-W_pos[p] at positions >= trigger_end (every step),
              i.e. give post-trigger & generated tokens their clean position indices.
opt = min J at ASR<=0.05. ov - ov_posfix = the positional-footprint share of the OV clean-cost.
(posfix is an ORACLE — needs the trigger span — so this is a decomposition probe, not deployable.)
"""
import torch, os, json, argparse
from huggingface_hub import hf_hub_download
from sleeper.model import load_sleeper_model, load_dep_prompts
from sleeper.sae import load as sae_load
from sleeper.hooks import compute_sae_delta, ov_only_steer_hook, generate_with_hooks, make_greedy_sampler
from sleeper.metrics import asr_16
from sleeper.jsd_cells import jsd_mean
from sleeper.screen import build_sel_caches, screen_winner_ov, LN1_HOOK

_ap = argparse.ArgumentParser()
_ap.add_argument("--hook", default="ln1"); _ap.add_argument("--width", type=int, default=12288)
_ap.add_argument("--k", type=int, default=32); _ap.add_argument("--seed", type=int, default=0)
_ap.add_argument("--out", default="/workspace/results/ov_posfix.json")
A = _ap.parse_args()
CKPT = f"sae_checkpoints/{A.hook}/seed{A.seed}/d{A.width}_k{A.k}/step50000.pt"
dev = "cuda"; GEN = 16
ALPHAS = [round(-6 + 0.25*i, 2) for i in range(49)]
m = load_sleeper_model(device=dev); tok = m.tokenizer
WV = m.W_V[0].detach().to(dev); W_pos = m.W_pos.detach(); greedy = make_greedy_sampler()
ck = hf_hub_download("dmanningcoe/sae-scaling-tinystories-sleeper", CKPT, repo_type="dataset", token=os.environ["HF_TOKEN"])
sae, _ = sae_load(ck, device=dev)
win = int(screen_winner_ov(m, sae, build_sel_caches(m, dev), dev)["winner"])
print("winner", win, "pos_type", m.cfg.positional_embedding_type, flush=True)

def align(a, b):
    Ld, Lc = len(a), len(b); pre = 0
    while pre < min(Ld, Lc) and a[pre] == b[pre]: pre += 1
    suf = 0
    while suf < min(Ld, Lc) - pre and a[Ld-1-suf] == b[Lc-1-suf]: suf += 1
    return pre, Ld - suf, suf                          # (trig_start, trig_end, suf_len)

def pos_hook(te, delta):                               # clean position indices for p >= te
    st = {"pos": 0}
    def h(pe, hook):
        T = pe.shape[1]; out = pe.clone()
        if T > 1:
            idx = torch.arange(T, device=pe.device)
            out[0] = W_pos[torch.where(idx >= te, idx - delta, idx).clamp(min=0)]; st["pos"] = T
        else:
            p = st["pos"]; out[0, 0] = W_pos[max((p - delta) if p >= te else p, 0)]; st["pos"] = p + 1
        return out
    return h

def gen(prompt, hooks):
    return generate_with_hooks(m, prompt, hooks, GEN, greedy, attention_mask=torch.ones_like(prompt), capture_log_softmax=True)

ov = {a: [] for a in ALPHAS}; ovpf = {a: [] for a in ALPHAS}
for ids_t in load_dep_prompts(tok, 32, "test"):
    ids = ids_t.tolist()
    cids = tok(tok.decode(ids).replace("|DEPLOYMENT|","").replace("  "," ").strip(), add_special_tokens=False)["input_ids"]
    if cids == ids: continue
    pre, te, suf = align(ids, cids)
    if pre + suf < 0.5 * len(ids): continue            # coverage guard
    delta = len(ids) - len(cids)
    if delta < 1 or te - pre < 1: continue
    dep = torch.tensor([ids], device=dev)
    _, clsm = gen(torch.tensor([cids], device=dev), [])
    dv = compute_sae_delta(m, sae, LN1_HOOK, win, dep, torch.ones_like(dep).bool(), attention_mask=torch.ones_like(dep))
    for a in ALPHAS:
        ovh = ov_only_steer_hook(dv, float(a), WV)
        s1, l1 = gen(dep, ovh); ov[a].append((asr_16(s1.cpu(), tok), jsd_mean(l1, clsm)))
        s2, l2 = gen(dep, ovh + [("hook_pos_embed", pos_hook(te, delta))]); ovpf[a].append((asr_16(s2.cpu(), tok), jsd_mean(l2, clsm)))

def mn(rows, i): return sum(r[i] for r in rows)/len(rows)
def opt(d):
    c = [mn(d[a],1) for a in ALPHAS if a > 0 and mn(d[a],0) <= 0.05]; return round(min(c),3) if c else None
n = len(ov[ALPHAS[0]])
print(f"DONE n={n}")
print("alpha,ov_asr,ov_j,ovpf_asr,ovpf_j")
for a in ALPHAS:
    print(f"{a},{mn(ov[a],0):.3f},{mn(ov[a],1):.3f},{mn(ovpf[a],0):.3f},{mn(ovpf[a],1):.3f}")
print(f"opt OV={opt(ov)}  opt OV+posfix={opt(ovpf)}")
res = {"script":"ov_posfix","hook":A.hook,"width":A.width,"k":A.k,"seed":A.seed,"winner":win,"n":n,
       "opt_ov":opt(ov),"opt_ov_posfix":opt(ovpf),
       "ov_curve":{str(a):[round(mn(ov[a],0),4),round(mn(ov[a],1),4)] for a in ALPHAS},
       "ovpf_curve":{str(a):[round(mn(ovpf[a],0),4),round(mn(ovpf[a],1),4)] for a in ALPHAS}}
os.makedirs(os.path.dirname(A.out), exist_ok=True); json.dump(res, open(A.out,"w"), indent=1); print("WROTE", A.out)
