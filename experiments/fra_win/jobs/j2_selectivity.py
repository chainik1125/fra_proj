"""J2 — surgical selectivity test (the core of the FRA-win hypothesis).
For GPT-2 L5H5 induction: each induction edge t is driven by token-specific FRA feature-pairs.
Test: ablating edge t*'s top-m FRA pairs (content-addressed score delta, exact per-term)
suppresses copying of token t* ONLY, with ~0 collateral on other tokens' copying.
Baselines: (i) mean-ablate the whole head -> kills ALL copying (max collateral);
(ii) zero the specific (q,k) edge score (position-patch oracle: selective but not content-addressed).
Build the selectivity matrix M[target, affected] = Δcopyprob, look for diagonal dominance.
"""
import os, sys, json, math
sys.path.insert(0, "/workspace/code")
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result

OUT = os.environ.get("OUTDIR", "."); dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2", device=dev); model.eval()
L, H = 5, 5

torch.manual_seed(0); N = 24
R = (torch.randperm(40000)[:N] + 1000).tolist()
toks = [model.tokenizer.bos_token_id] + R + R
tt = torch.tensor(toks, device=dev).unsqueeze(0); seq = tt.shape[1]
edges = [(1 + N + t, t + 2) for t in range(N - 1)]
ind_targets = {1 + N + t: R[t + 1] for t in range(N - 1)}
qpos = [q for q, k in edges]

logits, cache = model.run_with_cache(tt, names_filter=lambda n: n in
    [f"blocks.{L}.attn.hook_attn_scores", f"blocks.{L}.attn.hook_pattern", f"blocks.{L}.hook_resid_pre"])
sc = cache[f"blocks.{L}.attn.hook_attn_scores"][0, H].float().cpu().numpy()

def copyprob_vec(lg):
    probs = torch.softmax(lg[0].float(), dim=-1)
    return np.array([probs[q, ind_targets[q]].item() for q, k in edges])
base_vec = copyprob_vec(logits)
print(f"baseline copy-prob: mean {base_vec.mean():.3f} per-edge {np.round(base_vec,2).tolist()}", flush=True)

# FRA
sae = SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{L}.hook_resid_pre", device=dev)
if isinstance(sae, tuple): sae = sae[0]
act = cache[f"blocks.{L}.hook_resid_pre"][0]
feats = sae.encode(act).float()
x_hat = feats @ sae.W_dec.float() + sae.b_dec.float()
res = _build_fra_result(model, L, H, feats, sae.W_dec.float(), dev, top_k=None,
                        rms_activations=x_hat, dec_norms=None, chunk_size=16, verbose=False)
fra = res["fra_tensor_sparse"].coalesce()
idx = fra.indices().cpu().numpy(); val = fra.values().cpu().numpy()
qq, kk, ii, jj = idx[0], idx[1], idx[2], idx[3]

# per-edge top-m pairs
edge_idx = {e: np.where((qq == e[0]) & (kk == e[1]))[0] for e in edges}
def edge_top_pairs(e, m):
    locs = edge_idx[e]
    order = np.argsort(-np.abs(val[locs]))[:m]
    return [(int(ii[locs[o]]), int(jj[locs[o]])) for o in order]

# content-addressed score delta for a set of (i,j) pairs: sum FRA values with those (i,j) over ALL (q,k)
def pairs_delta(pairs):
    pset = set(pairs)
    d = np.zeros((seq, seq))
    sel = np.array([(int(ii[n]), int(jj[n])) in pset for n in range(len(val))])
    for n in np.where(sel)[0]:
        d[qq[n], kk[n]] += val[n]
    return d

def patched_copyprob_vec(score_delta):
    sd = torch.tensor(score_delta, device=dev, dtype=torch.float32)
    def hook(scores, hook):
        scores[0, H, :seq, :seq] = scores[0, H, :seq, :seq] - sd
        return scores
    lg = model.run_with_hooks(tt, fwd_hooks=[(f"blocks.{L}.attn.hook_attn_scores", hook)])
    return copyprob_vec(lg)

# ---- selectivity matrix: ablate each edge's top-m pairs (content addressed), measure all edges ----
M = 3
targets = list(range(N - 1))
sel_mat = np.zeros((len(targets), len(targets)))  # [target_t, affected_t] = copyprob after ablating target_t
frac_score_removed = []
for ti, t in enumerate(targets):
    e = edges[t]
    pairs = edge_top_pairs(e, M)
    d = pairs_delta(pairs)
    frac_score_removed.append(d[e] / sc[e[0], e[1]])
    sel_mat[ti] = patched_copyprob_vec(d)
# Δ relative to baseline
dmat = sel_mat - base_vec[None, :]
diag = np.array([dmat[i, i] for i in range(len(targets))])           # target effect (negative = suppressed)
offdiag = dmat - np.diag(np.diag(dmat))
offmean = np.array([np.mean(np.abs(np.delete(dmat[i], i))) for i in range(len(targets))])  # collateral
print(f"\nSURGICAL ABLATION (top-{M} FRA pairs/edge), content-addressed:", flush=True)
print(f"  mean frac of edge-score removed: {np.mean(frac_score_removed):.2f}", flush=True)
print(f"  target Δcopyprob (diag):   mean {diag.mean():+.3f}  (want strongly negative)", flush=True)
print(f"  collateral |Δ| (off-diag): mean {offmean.mean():.3f}  (want ~0)", flush=True)
print(f"  selectivity ratio |diag|/offdiag: {abs(diag.mean())/max(offmean.mean(),1e-4):.1f}x", flush=True)

# ---- baseline 1: mean-ablate the whole head H (kills all induction) ----
zmean = cache[f"blocks.{L}.attn.hook_pattern"]  # unused; do z-mean ablation via hook_z
def headkill():
    def hook(z, hook):  # z: [b, seq, head, d_head]
        z[0, :, H, :] = z[0, :, H, :].mean(0, keepdim=True)
        return z
    lg = model.run_with_hooks(tt, fwd_hooks=[(f"blocks.{L}.attn.hook_z", hook)])
    return copyprob_vec(lg)
kill_vec = headkill()
print(f"\nBASELINE mean-ablate head L{L}H{H}: copy-prob mean {kill_vec.mean():.3f} "
      f"(Δ {kill_vec.mean()-base_vec.mean():+.3f}); this is the 'kill all' point (collateral=everything)", flush=True)

# ---- baseline 2: position score-patch oracle (zero each target edge) -> selective but needs positions ----
oracle_diag = []
for t in targets:
    d = np.zeros((seq, seq)); d[edges[t]] = sc[edges[t][0], edges[t][1]] + 20
    v = patched_copyprob_vec(d)
    oracle_diag.append(v[t] - base_vec[t])
print(f"\nORACLE position-patch (zero edge): target Δ mean {np.mean(oracle_diag):+.3f} "
      f"(selective but not content-addressed/deployable)", flush=True)

out = {"base_copyprob_mean": float(base_vec.mean()),
       "surgical": {"M": M, "frac_score_removed": float(np.mean(frac_score_removed)),
                    "target_delta_mean": float(diag.mean()), "collateral_mean": float(offmean.mean()),
                    "selectivity_x": float(abs(diag.mean())/max(offmean.mean(),1e-4))},
       "headkill_copyprob_mean": float(kill_vec.mean()),
       "oracle_target_delta_mean": float(np.mean(oracle_diag)),
       "diag": diag.tolist(), "collateral": offmean.tolist()}
json.dump(out, open(os.path.join(OUT, "j2.json"), "w"), indent=2)

# selectivity matrix heatmap
plt.figure(figsize=(6, 5))
plt.imshow(dmat, cmap="RdBu", vmin=-base_vec.max(), vmax=base_vec.max())
plt.colorbar(label="Δ copy-prob"); plt.xlabel("affected token (edge)"); plt.ylabel("ablated target token (edge)")
plt.title(f"FRA surgical ablation selectivity (L{L}H{H}, top-{M} pairs)\ndiagonal = target, off-diag = collateral")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "j2_selectivity.png"), dpi=110)
print("\nDONE j2", flush=True)
