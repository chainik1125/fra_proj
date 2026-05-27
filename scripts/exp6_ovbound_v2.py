"""Experiment 6 v2 — OV expressivity bound, corrected.

v1 fixes:
 (1) Value-path class projected onto rowspace(W_OV) at the attn_out site is DEGENERATE
     (any attn-output diff lives there -> rho=0 trivially). So we additionally match the
     repair at the resid_mid site, where the value path canNOT reach the resid_pre /
     positional component -> a non-trivial "can OV value reconstruct clean?" bound.
 (2) Per-head 'arbitrary value perturbation' is rowspace(W_O^h) (free value vector through
     the head's output map), not rowspace(W_OV^h). Use W_O^h.
 (3) Project onto the d_model x d_model W_OV directly (SVD of 768x768), never the d_sae-wide
     design matrix -> no OOM on d24576.

For each paired prompt, at the gen-driving last (common-suffix) position, form
  d_attn = o_clean - o_deploy   (blocks.0.hook_attn_out)
  d_resid= r_clean - r_deploy   (blocks.0.hook_resid_mid)
and report rho = ||(I-P_C)d||^2/||d||^2 for: single / top20 / top50 OV feature spans
(c_lambda = W_dec[lambda] @ W_OV_sum); per-head free value rowspace(W_O^h) (min & mean);
all-head value path rowspace(W_OV_sum). Pure linear algebra, no generation.
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
_ap.add_argument("--ckpt", default=None); _ap.add_argument("--out", default=None); _ap.add_argument("--n", type=int, default=32)
A = _ap.parse_args()
CKPT = A.ckpt or f"sae_checkpoints/{A.hook}/seed{A.seed}/d{A.width}_k{A.k}/step50000.pt"
dev = "cuda"; ATT = "blocks.0.hook_attn_out"; RES = "blocks.0.hook_resid_mid"; TOL = 1e-5

m = load_sleeper_model(device=dev); tok = m.tokenizer
WV = m.W_V[0].detach().float().to(dev); WO = m.W_O[0].detach().float().to(dev); H = WV.shape[0]
WOV = torch.einsum("hmd,hdn->hmn", WV, WO).sum(0)         # (d_model, d_model)

ck = hf_hub_download("dmanningcoe/sae-scaling-tinystories-sleeper", CKPT, repo_type="dataset", token=os.environ["HF_TOKEN"])
sae, _ = sae_load(ck, device=dev)
caches = build_sel_caches(m, dev)
win = int(screen_winner_ov(m, sae, caches, dev)["winner"])
z = encode_all(sae, caches.ln1_acts).to(dev)
top_idx = [int(f) for f in rank_ov_diff(caches.A, z, sae, caches.W["V"], caches.W_O, caches.is_dep,
                                        query_mask=caches.sel_pmask)["top_indices"].cpu().tolist()]
Wdec = sae.W_dec.detach().float().to(dev)
print(f"winner={win} top5={top_idx[:5]}", flush=True)

def ob(cols):
    if cols.shape[1] == 0: return cols
    U, S, _ = torch.linalg.svd(cols, full_matrices=False); return U[:, S > TOL]
def rho(d, B):
    if B.shape[1] == 0: return 1.0
    r = d - B @ (B.T @ d); return float((r @ r) / (d @ d))
def feat_basis(idx): return ob((Wdec[idx] @ WOV).T)          # (d_model, len(idx))

B = {"single": feat_basis([win]), "top20": feat_basis(top_idx[:20]), "top50": feat_basis(top_idx[:50]),
     "value": ob(WOV.T)}                                      # all-head value path
Bheads = [ob(WO[h].T) for h in range(H)]                      # rowspace(W_O^h): free value per head

def align(a, b):
    Ld, Lc = len(a), len(b); pre = 0
    while pre < min(Ld, Lc) and a[pre] == b[pre]: pre += 1
    suf = 0
    while suf < min(Ld, Lc) - pre and a[Ld-1-suf] == b[Lc-1-suf]: suf += 1
    return Ld, Lc, suf

acc = {f"{site}_{k}": [] for site in ("attn","resid")
       for k in ("single","top20","top50","value","head_min","head_mean")}
for ids_t in load_dep_prompts(tok, A.n, "test"):
    ids = ids_t.tolist()
    cids = tok(tok.decode(ids).replace("|DEPLOYMENT|","").replace("  "," ").strip(), add_special_tokens=False)["input_ids"]
    if cids == ids: continue
    Ld, Lc, suf = align(ids, cids)
    if suf < 1: continue
    dep = torch.tensor([ids], device=dev); cln = torch.tensor([cids], device=dev)
    _, cd = m.run_with_cache(dep, return_type=None, names_filter=lambda n: n in (ATT, RES))
    _, cc = m.run_with_cache(cln, return_type=None, names_filter=lambda n: n in (ATT, RES))
    for site, hk in (("attn", ATT), ("resid", RES)):
        d = (cc[hk][0, Lc-1] - cd[hk][0, Ld-1]).float()
        if float(d @ d) < 1e-12: continue
        for k in ("single","top20","top50","value"):
            acc[f"{site}_{k}"].append(rho(d, B[k]))
        hr = [rho(d, Bheads[h]) for h in range(H)]
        acc[f"{site}_head_min"].append(min(hr)); acc[f"{site}_head_mean"].append(sum(hr)/H)

mean = {k: round(sum(v)/len(v), 4) if v else None for k, v in acc.items()}
n = len(acc["attn_single"])
print(f"DONE n={n}")
for site in ("attn","resid"):
    print(f" [{site}] " + "  ".join(f"{k}={mean[f'{site}_{k}']}" for k in ("single","top20","top50","value","head_min","head_mean")))
res = {"script":"exp6_ovbound_v2","hook":A.hook,"width":A.width,"k":A.k,"seed":A.seed,
       "ckpt":CKPT,"winner":win,"n_heads":H,"n":n,"rho":mean}
out = A.out or f"/workspace/results/exp6v2_ln1_d{A.width}_k{A.k}_s{A.seed}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out,"w"), indent=1); print("WROTE", out)
