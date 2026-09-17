"""Diagnostic: where does the last-position query attend in the synthetic conjunction, and what removes P?"""
import os, torch, numpy as np
from transformer_lens import HookedTransformer
torch.set_grad_enabled(False); dev = "cpu"
REP = int(os.environ.get("REP", "4"))
m = HookedTransformer.from_pretrained("gpt2", device=dev); m.eval(); tok = m.tokenizer
IND = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]; LAYERS = sorted(set(L for L, H in IND)); Llast = m.cfg.n_layers - 1; W_U = m.W_U
g = torch.Generator().manual_seed(0)
A, B, C, D, P, S, Q, Z = (torch.randperm(30000, generator=g)[:8] + 1500).tolist()
fill = (torch.randperm(20000, generator=g)[:80] + 22000).tolist()
seq = [tok.bos_token_id]; roles = {"A": [], "B": [], "P": [], "D": [], "S": [], "C": [], "Q": []}
for rep in range(REP):
    for t, nm in [(fill[(rep*3) % 80], "f"), (A, "A"), (B, "B"), (P, "P"), (fill[(rep*3+1) % 80], "f"), (D, "D"), (B, "B"), (S, "S"), (fill[(rep*3+2) % 80], "f"), (A, "A"), (C, "C"), (Q, "Q")]:
        seq.append(t)
        if nm in roles: roles[nm].append(len(seq) - 1)
ids = seq + [A, B]; tt = torch.tensor(ids, device=dev).unsqueeze(0); qpos = len(ids) - 1
lg, cache = m.run_with_cache(tt)
base = torch.softmax(lg[0][qpos].float(), -1)[P].item()
print(f"REP={REP} qpos={qpos} base P(P)={base:.3f}", flush=True)
print(f"role positions: A={roles['A']} B={roles['B']} P={roles['P']} S={roles['S']}", flush=True)
tokrole = {}
for nm, ps in roles.items():
    for p in ps: tokrole[p] = nm
for (L, H) in IND:
    patt = cache[f"blocks.{L}.attn.hook_pattern"][0, H, qpos]  # attention from qpos to all keys
    top = torch.topk(patt, 6).indices.tolist()
    print(f"  L{L}H{H} top keys: " + ", ".join(f"{k}({tokrole.get(k,'.')}:{patt[k]:.2f})" for k in top), flush=True)

def mask_keys(keyset, c=1000.0):
    hooks = []
    for L in LAYERS:
        hs = [H for (LL, H) in IND if LL == L]
        def mk(hs):
            def hook(s, hook):
                for H in hs:
                    for k in keyset:
                        if k < s.shape[-1]: s[0, H, qpos, k] -= c
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(hs)))
    return torch.softmax(m.run_with_hooks(tt, fwd_hooks=hooks)[0][qpos].float(), -1)[P].item()
print("\n-- masking removal tests (P(P) after masking qpos->keyset) --", flush=True)
print(f"  mask P-positions {roles['P']}: P(P)={mask_keys(roles['P']):.3f}", flush=True)
print(f"  mask B-positions {roles['B']}: P(P)={mask_keys(roles['B']):.3f}", flush=True)
print(f"  mask P and B     : P(P)={mask_keys(roles['P']+roles['B']):.3f}", flush=True)
print(f"  mask A-positions {roles['A']}: P(P)={mask_keys(roles['A']):.3f}", flush=True)
# payload-suppress at output
uP = W_U[:, P].float(); uP = uP / uP.norm()
def pay(s):
    def hook(act, hook): act[0] = act[0] - s * (act[0] @ uP).unsqueeze(-1) * uP; return act
    return torch.softmax(m.run_with_hooks(tt, fwd_hooks=[(f"blocks.{Llast}.hook_resid_post", hook)])[0][qpos].float(), -1)[P].item()
print(f"  payload-suppress s=2: P(P)={pay(2):.3f}", flush=True)
print("\nDONE diag", flush=True)
