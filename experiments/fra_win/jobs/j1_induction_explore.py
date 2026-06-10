"""J1 — induction-head FRA exploration on GPT-2 small.
Goals: (1) pick the strongest induction head; (2) validate that the FRA decomposition
reconstructs the real pre-softmax attention scores; (3) find the dominant feature-pair(s)
driving the induction stripe; (4) first steering probe — ablate that pair's score
contribution and see if induction copy-probability drops vs random-pair / whole-edge baselines.
Writes diagnostics to out.log + j1.json (+ a couple of pngs) in OUTDIR.
"""
import os, sys, json, math
sys.path.insert(0, "/workspace/code")
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens import SAE

OUT = os.environ.get("OUTDIR", ".")
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
print("device", dev, flush=True)

model = HookedTransformer.from_pretrained("gpt2", device=dev)
model.eval()
print("model loaded", model.cfg.n_layers, "layers", model.cfg.n_heads, "heads", flush=True)

# ---- build a clean induction sequence: BOS + R + R (R = N distinct random tokens) ----
torch.manual_seed(0)
N = 24
# pick tokens away from specials / very common; sample from a mid vocab range
cand = torch.randperm(40000)[:N] + 1000
R = cand.tolist()
toks = [model.tokenizer.bos_token_id] + R + R
tt = torch.tensor(toks, device=dev).unsqueeze(0)
seq = tt.shape[1]
print("seq len", seq, flush=True)
# induction edges: query at second occ of r_t (pos 1+N+t) -> key at pos t+2 (token r_{t+1}); t in 0..N-2
edges = [(1 + N + t, t + 2) for t in range(N - 1)]
# target token at each induction query = r_{t+1}
ind_targets = {1 + N + t: R[t + 1] for t in range(N - 1)}

# ---- run with cache, get per-head attention scores+pattern; pick strongest induction head ----
names = [f"blocks.{L}.attn.hook_attn_scores" for L in range(model.cfg.n_layers)] + \
        [f"blocks.{L}.attn.hook_pattern" for L in range(model.cfg.n_layers)]
logits, cache = model.run_with_cache(tt, names_filter=lambda n: n in names)
ind_strength = {}
for L in range(model.cfg.n_layers):
    pat = cache[f"blocks.{L}.attn.hook_pattern"][0]  # [head, q, k]
    for H in range(model.cfg.n_heads):
        s = np.mean([pat[H, q, k].item() for (q, k) in edges])
        ind_strength[(L, H)] = s
top = sorted(ind_strength.items(), key=lambda x: -x[1])[:8]
print("\nTOP induction heads (mean attn on induction edge):", flush=True)
for (L, H), s in top:
    print(f"  L{L}H{H}: {s:.3f}", flush=True)
LH = top[0][0]; L, H = LH
print(f"\n=> using induction head L{L}H{H}", flush=True)

# ---- behavioral metric: induction copy-probability at induction query positions ----
def copy_prob(lg):
    probs = torch.softmax(lg[0].float(), dim=-1)
    return float(np.mean([probs[q, ind_targets[q]].item() for (q, k) in edges]))
base_cp = copy_prob(logits)
print(f"\nbaseline induction copy-prob = {base_cp:.4f}", flush=True)

# ---- FRA on the induction head; validate reconstruction of pre-softmax scores ----
from fra.core.fra import get_sentence_fra_batch
text = None  # we pass tokens directly via a tiny shim: re-encode is not ideal; instead use the token path
# get_sentence_fra_batch tokenizes text; to use our exact tokens, decode then re-encode is lossy.
# Instead replicate its core on our tokens:
from fra.core.fra import _build_fra_result
release, sae_id = "gpt2-small-res-jb", f"blocks.{L}.hook_resid_pre"
sae = SAE.from_pretrained(release, sae_id, device=dev)
if isinstance(sae, tuple): sae = sae[0]
print("SAE loaded", release, sae_id, "d_sae", sae.W_dec.shape[0], flush=True)
hook = f"blocks.{L}.hook_resid_pre"
_, c2 = model.run_with_cache(tt, names_filter=[hook])
act = c2[hook][0]  # [seq, d_model]
feats = sae.encode(act).float()  # [seq, d_sae]
print("mean active feats/pos", float((feats != 0).float().sum(-1).mean()), flush=True)
# RMS correction activations (resid path): x_hat reconstruction
x_hat = feats @ sae.W_dec.float() + sae.b_dec.float()
res = _build_fra_result(model, L, H, feats, sae.W_dec.float(), dev,
                        top_k=None, rms_activations=x_hat, dec_norms=None, chunk_size=16, verbose=False)
fra = res["fra_tensor_sparse"].coalesce()
# reconstruct scores: sum over feature dims -> [seq,seq]
idx = fra.indices(); val = fra.values()
recon = torch.zeros(seq, seq)
qk = idx[0] * seq + idx[1]
recon.view(-1).index_add_(0, qk.cpu(), val.cpu())
recon = recon.numpy()
# actual scores for this head
sc = cache[f"blocks.{L}.attn.hook_attn_scores"][0, H].float().cpu().numpy()  # [q,k]
# compare on the causal (k<=q) entries
mask = np.tril(np.ones((seq, seq)), 0).astype(bool)
a = sc[mask]; b = recon[mask]
# FRA omits the bias term b_Q·k etc; compare correlation + on induction edges specifically
corr = np.corrcoef(a, b)[0, 1]
edge_recon = np.array([recon[q, k] for q, k in edges])
edge_actual = np.array([sc[q, k] for q, k in edges])
print(f"\nFRA reconstruction: corr(all causal scores) = {corr:.3f}", flush=True)
print(f"  induction edges: actual mean score {edge_actual.mean():.2f}, FRA mean {edge_recon.mean():.2f}, "
      f"corr {np.corrcoef(edge_actual, edge_recon)[0,1]:.3f}", flush=True)

# ---- dominant feature-pair on the induction stripe ----
# aggregate FRA[q,k,i,j] over induction edges -> score per (i,j)
edge_set = set(edges)
pair_score = {}
ii = idx[2].cpu().numpy(); jj = idx[3].cpu().numpy()
qq = idx[0].cpu().numpy(); kk = idx[1].cpu().numpy(); vv = val.cpu().numpy()
for n in range(len(vv)):
    if (int(qq[n]), int(kk[n])) in edge_set:
        key = (int(ii[n]), int(jj[n]))
        pair_score[key] = pair_score.get(key, 0.0) + float(vv[n])
top_pairs = sorted(pair_score.items(), key=lambda x: -abs(x[1]))[:15]
print("\nTOP feature-pairs on induction stripe (summed FRA over edges):", flush=True)
for (i, j), s in top_pairs:
    tag = "  <-- SELF (i==j)" if i == j else ""
    print(f"  qF{i} x kF{j}: {s:.3f}{tag}", flush=True)
dom_i, dom_j = top_pairs[0][0]

# ---- steering probe: ablate dominant pair's score contribution on induction edges ----
# build per-edge delta for a given (i,j): the FRA value at [q,k,i,j]
def pair_delta_map(i_sel, j_sel):
    d = np.zeros((seq, seq))
    sel = (ii == i_sel) & (jj == j_sel)
    for n in np.where(sel)[0]:
        d[int(qq[n]), int(kk[n])] += float(vv[n])
    return d

def patched_copyprob(score_delta):
    """subtract score_delta from head H's pre-softmax scores, return induction copy-prob."""
    sd = torch.tensor(score_delta, device=dev, dtype=torch.float32)
    def hook(scores, h):
        scores[0, H, :seq, :seq] = scores[0, H, :seq, :seq] - sd
        return scores
    lg = model.run_with_hooks(tt, fwd_hooks=[(f"blocks.{L}.attn.hook_attn_scores", hook)])
    return copy_prob(lg)

dom_delta = pair_delta_map(dom_i, dom_j)
cp_dom = patched_copyprob(dom_delta)
# random pair baseline (a non-dominant active pair)
rand_pair = top_pairs[len(top_pairs)//2][0]
cp_rand = patched_copyprob(pair_delta_map(*rand_pair))
# whole-edge oracle: zero the induction-edge scores entirely (huge negative)
oracle = np.zeros((seq, seq))
for q, k in edges: oracle[q, k] = sc[q, k] + 20.0  # push to -20
cp_oracle = patched_copyprob(oracle)
print(f"\nSTEERING PROBE (induction copy-prob):", flush=True)
print(f"  baseline             {base_cp:.4f}", flush=True)
print(f"  ablate DOMINANT pair qF{dom_i}xkF{dom_j}  {cp_dom:.4f}   (Δ {cp_dom-base_cp:+.4f})", flush=True)
print(f"  ablate random pair   {cp_rand:.4f}   (Δ {cp_rand-base_cp:+.4f})", flush=True)
print(f"  oracle zero-edge     {cp_oracle:.4f}   (Δ {cp_oracle-base_cp:+.4f})", flush=True)

out = {
    "top_induction_heads": [{"L": L_, "H": H_, "strength": s} for (L_, H_), s in top],
    "head": [L, H], "base_copyprob": base_cp,
    "recon_corr_all": float(corr),
    "recon_corr_edges": float(np.corrcoef(edge_actual, edge_recon)[0, 1]),
    "edge_actual_mean": float(edge_actual.mean()), "edge_fra_mean": float(edge_recon.mean()),
    "top_pairs": [{"i": int(i), "j": int(j), "score": float(s)} for (i, j), s in top_pairs],
    "probe": {"baseline": base_cp, "ablate_dominant": cp_dom, "ablate_random": cp_rand,
              "oracle_zero_edge": cp_oracle, "dominant_pair": [int(dom_i), int(dom_j)]},
}
json.dump(out, open(os.path.join(OUT, "j1.json"), "w"), indent=2)

# figure: actual vs FRA score on induction edges
plt.figure(figsize=(5,4))
plt.scatter(edge_actual, edge_recon, s=18)
lo, hi = min(edge_actual.min(), edge_recon.min()), max(edge_actual.max(), edge_recon.max())
plt.plot([lo,hi],[lo,hi],'k--',lw=1)
plt.xlabel("actual pre-softmax score"); plt.ylabel("FRA reconstructed")
plt.title(f"L{L}H{H} induction edges: FRA recon (corr {np.corrcoef(edge_actual,edge_recon)[0,1]:.2f})")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"j1_recon.png"), dpi=110)
print("\nDONE j1", flush=True)
