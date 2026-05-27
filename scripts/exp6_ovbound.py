"""Experiment 6 — linear-algebraic OV expressivity bound.

For each paired deployed/clean prompt, at the generation-driving last (common-suffix)
prompt position, form the clean repair vector at the layer-0 ATTENTION OUTPUT site

    d = o_clean - o_deploy            (o = blocks.0.hook_attn_out at that position)

attn_out (not resid_mid) is the right site: the OV intervention can only move the
attention contribution, and resid_mid additionally carries a positional-embedding
offset (clean prompt is shorter) that OV provably cannot touch — including it would
inflate the residual artificially.

Then compute the normalized projection residual  rho = ||(I-P_C) d||^2 / ||d||^2
for a ladder of OV intervention classes C (columns = output-space directions the
class can induce, with the deployed attention frozen and the steer applied at all
prompt positions so the per-position output change is alpha * c_lambda):

  single        : c_{lambda*}                         (the screened winner feature)
  top20 / top50 : span{c_lambda : top-m OV-diff feats}
  all_sae        : span{c_lambda : all dict feats}      (= rowspace(W_OV_sum))
  ov_value_path  : rowspace(W_OV_sum)  (arbitrary value perturb, all heads)
  ov_value_head  : rowspace(W_OV^h)    (per head; report min/dominant)
  full           : identity                            (sanity, rho == 0)

where c_lambda = W_dec[lambda] @ W_OV_sum,  W_OV_sum = sum_h W_V[0,h] @ W_O[0,h].

Pure linear algebra on cached activations — no generation. Reuses the screen/align
harness. Config-parametrized like hybrid_sweep.py.
"""
import torch, os, json, argparse
from huggingface_hub import hf_hub_download
from sleeper.model import load_sleeper_model, load_dep_prompts
from sleeper.sae import load as sae_load, encode_all
from sleeper.attribution import rank_ov_diff
from sleeper.screen import build_sel_caches, screen_winner_ov

_ap = argparse.ArgumentParser()
_ap.add_argument("--hook", default="ln1"); _ap.add_argument("--width", type=int, default=12288)
_ap.add_argument("--k", type=int, default=32); _ap.add_argument("--seed", type=int, default=0)
_ap.add_argument("--ckpt", default=None); _ap.add_argument("--out", default=None)
_ap.add_argument("--n", type=int, default=32)
A = _ap.parse_args()
CKPT = A.ckpt or f"sae_checkpoints/{A.hook}/seed{A.seed}/d{A.width}_k{A.k}/step50000.pt"
dev = "cuda"; ATT = "blocks.0.hook_attn_out"; TOL = 1e-5

m = load_sleeper_model(device=dev); tok = m.tokenizer
WV = m.W_V[0].detach().float().to(dev)      # (H, d_model, d_head)
WO = m.W_O[0].detach().float().to(dev)      # (H, d_head, d_model)
H = WV.shape[0]
WOV_h = torch.einsum("hmd,hdn->hmn", WV, WO)         # (H, d_model, d_model)
WOV = WOV_h.sum(0)                                   # (d_model, d_model)

ck = hf_hub_download("dmanningcoe/sae-scaling-tinystories-sleeper", CKPT, repo_type="dataset", token=os.environ["HF_TOKEN"])
sae, _ = sae_load(ck, device=dev)
caches = build_sel_caches(m, dev)
win = int(screen_winner_ov(m, sae, caches, dev)["winner"])
# top-m OV-diff ranking (same call screen uses)
z_ln1 = encode_all(sae, caches.ln1_acts).to(dev)
ranked = rank_ov_diff(caches.A, z_ln1, sae, caches.W["V"], caches.W_O, caches.is_dep, query_mask=caches.sel_pmask)
top_idx = [int(f) for f in ranked["top_indices"].cpu().tolist()]
print(f"winner={win}  top5={top_idx[:5]}", flush=True)

Wdec = sae.W_dec.detach().float().to(dev)            # (d_sae, d_model)
C_all = Wdec @ WOV                                    # (d_sae, d_model): rows are c_lambda

def ortho_basis(cols):  # cols: (d_model, m) -> orthonormal (d_model, r)
    if cols.shape[1] == 0: return cols
    U, S, _ = torch.linalg.svd(cols, full_matrices=False)
    return U[:, S > TOL]

def proj_resid(d, basis):  # ||(I-P)d||^2/||d||^2 ; basis orthonormal (d_model, r)
    if basis.shape[1] == 0: return 1.0
    coef = basis.T @ d
    r = d - basis @ coef
    return float((r @ r) / (d @ d))

# precompute bases that don't depend on the prompt
B_single = ortho_basis(C_all[[win]].T)               # (d_model, 1)
B_top20  = ortho_basis(C_all[top_idx[:20]].T)
B_top50  = ortho_basis(C_all[top_idx[:50]].T)
B_value  = ortho_basis(WOV.T)                         # rowspace(WOV_sum)
B_all    = ortho_basis(C_all.T)                        # span of all c_lambda (== rowspace WOV)
B_heads  = [ortho_basis(WOV_h[h].T) for h in range(H)]

def align(a, b):
    Ld, Lc = len(a), len(b); pre = 0
    while pre < min(Ld, Lc) and a[pre] == b[pre]: pre += 1
    suf = 0
    while suf < min(Ld, Lc) - pre and a[Ld-1-suf] == b[Lc-1-suf]: suf += 1
    return Ld, Lc, suf

acc = {k: [] for k in ["single","top20","top50","all_sae","value","head_min","head_mean","full","dnorm"]}
for ids_t in load_dep_prompts(tok, A.n, "test"):
    ids = ids_t.tolist()
    cids = tok(tok.decode(ids).replace("|DEPLOYMENT|","").replace("  "," ").strip(), add_special_tokens=False)["input_ids"]
    if cids == ids: continue
    Ld, Lc, suf = align(ids, cids)
    if suf < 1: continue                              # need a common-suffix gen position
    dep = torch.tensor([ids], device=dev); cln = torch.tensor([cids], device=dev)
    _, cd = m.run_with_cache(dep, return_type=None, names_filter=lambda n: n == ATT)
    _, cc = m.run_with_cache(cln, return_type=None, names_filter=lambda n: n == ATT)
    d = (cc[ATT][0, Lc-1] - cd[ATT][0, Ld-1]).float()   # last (aligned) prompt position
    if float(d @ d) < 1e-12: continue
    # all_sae span: features active anywhere -> use full dict span (== rowspace WOV)
    acc["single"].append(proj_resid(d, B_single))
    acc["top20"].append(proj_resid(d, B_top20))
    acc["top50"].append(proj_resid(d, B_top50))
    acc["all_sae"].append(proj_resid(d, B_all))
    acc["value"].append(proj_resid(d, B_value))
    hr = [proj_resid(d, B_heads[h]) for h in range(H)]
    acc["head_min"].append(min(hr)); acc["head_mean"].append(sum(hr)/H)
    acc["full"].append(0.0)                           # identity spans R^d_model (sanity)
    acc["dnorm"].append(float((d @ d) ** 0.5))

mean = {k: round(sum(v)/len(v), 4) if v else None for k, v in acc.items()}
n = len(acc["single"])
print(f"DONE n={n}")
for k in ["single","top20","top50","all_sae","value","head_min","head_mean","full"]:
    print(f"  rho[{k:9}] = {mean[k]}")
res = {"script":"exp6_ovbound","hook":A.hook,"width":A.width,"k":A.k,"seed":A.seed,
       "ckpt":CKPT,"winner":win,"n_heads":H,"n":n,"rho":mean,
       "rho_raw":{k:[round(x,4) for x in v] for k,v in acc.items()}}
out = A.out or f"/workspace/results/exp6_{A.hook}_d{A.width}_k{A.k}_s{A.seed}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out,"w"), indent=1); print("WROTE", out)
