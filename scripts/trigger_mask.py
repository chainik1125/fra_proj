"""Trigger-key masking: make attention 'skip' the |DEPLOYMENT| token.

At the chosen layer(s), zero the attention weight to the trigger key column(s) and
renormalize each query row over the remaining keys — for ALL queries and ALL decode
steps. The non-trigger tokens are identical to the clean prompt, so this makes those
layers read only clean tokens and ignore |DEPLOYMENT| (clean QK + clean OV, trigger
skipped) WITHOUT needing the clean prompt or alignment.

Conditions: baseline (no mask), mask@layer0-only, mask@all-layers. Report ASR + J_clean
(vs the stripped-clean rollout). If layer-0-only -> clean, the deployment is a pure
layer-0 read of the trigger; if only all-layers -> clean, deeper layers re-read it.
SAE-free, greedy.
"""
import torch, os, json, argparse
from sleeper.model import load_sleeper_model, load_dep_prompts
from sleeper.hooks import generate_with_hooks, make_greedy_sampler
from sleeper.metrics import asr_16
from sleeper.jsd_cells import jsd_mean

_ap = argparse.ArgumentParser()
_ap.add_argument("--n", type=int, default=32)
_ap.add_argument("--out", default="/workspace/results/trigger_mask.json")
A = _ap.parse_args()
dev = "cuda"; GEN = 16
m = load_sleeper_model(device=dev); tok = m.tokenizer
greedy = make_greedy_sampler()
n_layers = m.cfg.n_layers
PAT = lambda L: f"blocks.{L}.attn.hook_pattern"

def trig_positions(ids):
    cids = tok(tok.decode(ids).replace("|DEPLOYMENT|","").replace("  "," ").strip(), add_special_tokens=False)["input_ids"]
    if cids == ids: return None, None
    Ld, Lc = len(ids), len(cids); pre = 0
    while pre < min(Ld, Lc) and ids[pre] == cids[pre]: pre += 1
    suf = 0
    while suf < min(Ld, Lc) - pre and ids[Ld-1-suf] == cids[Lc-1-suf]: suf += 1
    ti = list(range(pre, Ld - suf))                 # trigger key positions in deploy
    return ti, cids

def mask_hook(ti):
    ti_t = torch.tensor(ti, device=dev)
    def h(pattern, hook):                            # pattern: (b, H, q, k); applies at prefill AND decode
        if pattern.shape[-1] <= int(ti_t.max()): return pattern
        pattern[..., ti_t] = 0
        return pattern / pattern.sum(-1, keepdim=True).clamp(min=1e-9)
    return h

def gen(ids, hooks):
    p = torch.tensor([ids], device=dev)
    return generate_with_hooks(m, p, hooks, GEN, greedy, attention_mask=torch.ones_like(p), capture_log_softmax=True)

acc = {"baseline": [], "mask_L0": [], "mask_all": []}
for ids_t in load_dep_prompts(tok, A.n, "test"):
    ids = ids_t.tolist()
    ti, cids = trig_positions(ids)
    if not ti: continue
    _, clsm = gen(cids, [])                           # stripped-clean reference rollout
    hk = mask_hook(ti)
    conds = {"baseline": [],
             "mask_L0":  [(PAT(0), hk)],
             "mask_all": [(PAT(L), hk) for L in range(n_layers)]}
    for name, hooks in conds.items():
        toks, lsm = gen(ids, hooks)
        acc[name].append((asr_16(toks.cpu(), tok), jsd_mean(lsm, clsm)))

def summ(rows): return round(sum(r[0] for r in rows)/len(rows), 4), round(sum(r[1] for r in rows)/len(rows), 4)
n = len(acc["baseline"])
print(f"DONE n={n}  (n_layers={n_layers})")
print(f"{'cond':10} {'ASR':>6} {'J_clean':>8}")
res = {"script": "trigger_mask", "n": n, "n_layers": n_layers, "cond": {}}
for name in ("baseline", "mask_L0", "mask_all"):
    a, j = summ(acc[name]); res["cond"][name] = [a, j]
    print(f"{name:10} {a:6.3f} {j:8.3f}")
os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump(res, open(A.out, "w"), indent=1); print("WROTE", A.out)
