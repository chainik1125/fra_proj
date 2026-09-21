"""Check 4 — two-head ζ<0 hinge on attention-only B (adapts sprint twohead.py).

For ζ<0 softmax nonnegativity forces two heads with anti-parallel OVs; only the
head-SUM FRA-OV attribution is loss-pinned. We measure: (a) per-head OV direction
for each token projected on ĝ(z) -> single-signed, heads anti-parallel;
(b) per-head effective subspace attention α_h(d,s) single-signed; (c) the head-SUM
α recovers the sign-oscillating ζ^{d-s} kernel (sign alternates with lag parity).

Usage: twohead_ao.py <model_tag>
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch
from pathlib import Path
from mess3 import Mess3, enumerate_sequences, prefix_probs
from verify_ao import build_ao, N_CTX, D_MODEL, ln1_maker

torch.set_num_threads(4)
OUT = Path(__file__).resolve().parent.parent / "out"; L = N_CTX


def load(tag):
    d = torch.load(OUT / f"model_{tag}.pt", map_location="cpu")
    c = d["cfg"]; m = build_ao(c["n_heads"], c["n_layers"], c["seed"])
    m.load_state_dict(d["state_dict"]); m.eval(); return m, c, Mess3(c["x"], c["a"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); args = ap.parse_args()
    model, c, m = load(args.tag)
    Ll = c["n_layers"] - 1; H = c["n_heads"]; zeta = 1 - 3 * c["x"]
    seqs = enumerate_sequences(L); pp = prefix_probs(m, seqs); w = pp[:, L]
    W_E = model.W_E.detach().numpy(); W_pos = model.W_pos.detach().numpy()
    ln = ln1_maker(model, Ll)
    g = np.stack([m.pi @ m.T_cond[z] - m.pi for z in range(3)])
    ghat = g / np.linalg.norm(g, axis=1, keepdims=True)
    blk = model.blocks[Ll]
    W_V = blk.attn.W_V.detach().numpy(); W_O = blk.attn.W_O.detach().numpy()
    x_zp = W_E[:, None, :] + W_pos[None, :, :]              # (3,L,D)

    # proper resid->belief readout map R (population regression resid_post -> r1)
    from interv_ao import gather, readout_map
    _, wg, r1g, _, RPg, _ = gather(model, m)
    R = readout_map(RPg, r1g, wg)                          # (D+1, 3)
    def to_plane(v):                                       # v:(...,D) -> (...,3)
        v1 = np.concatenate([v, np.ones(v.shape[:-1] + (1,))], -1)
        return v1 @ R
    ov_sign = np.zeros((H, 3)); ov_dir = np.zeros((H, 3, D_MODEL))
    for h in range(H):
        fvb = ln(x_zp) @ W_V[h] @ W_O[h]                   # (3,L,D)
        vz = fvb.mean(1); ov_dir[h] = vz                   # (3,D)
        proj = to_plane(vz)                                # (3,3) belief plane
        ov_sign[h] = np.array([proj[z] @ ghat[z] for z in range(3)])
    # anti-parallel: sign(head0)*sign(head1) per token (expect negative)
    antipar = (np.sign(ov_sign[0]) * np.sign(ov_sign[1])).tolist() if H >= 2 else None
    ov_cos = float(np.mean([
        (ov_dir[0, z] - ov_dir[0].mean(0)) @ (ov_dir[1, z] - ov_dir[1].mean(0)) /
        (np.linalg.norm(ov_dir[0, z] - ov_dir[0].mean(0)) *
         np.linalg.norm(ov_dir[1, z] - ov_dir[1].mean(0)) + 1e-12)
        for z in range(3)])) if H >= 2 else None

    # per-head pattern + head-sum effective-attention sign by lag parity
    names = {f"blocks.{Ll}.attn.hook_pattern"}
    pat_sum = np.zeros((H, L, L))
    for i in range(0, seqs.shape[0], 4096):
        toks = torch.tensor(seqs[i:i + 4096])
        _, cache = model.run_with_cache(toks, return_type=None,
                                        names_filter=lambda x: x in names)
        pat = cache[f"blocks.{Ll}.attn.hook_pattern"].numpy()
        pat_sum += np.einsum("c,chds->hds", w[i:i + 4096], pat)
    # effective per-head OV-signed attention: alpha_h(d,s) = ovsign_h * A_h(d,s)
    # (single scalar sign per head, since OV is single-signed on the plane)
    sgn = np.sign(ov_sign.mean(1))                          # (H,) dominant head sign
    ah = np.einsum("h,hds->ds", sgn, pat_sum)              # head-sum signed profile
    # sign of the head-sum profile by lag (rows 5..8), should alternate ~ (-1)^lag
    lag_sign = {}
    for lag in range(0, 6):
        vals = [ah[d, d - lag] for d in range(5, L) if d - lag >= 0]
        lag_sign[lag] = float(np.sign(np.mean(vals))) if vals else 0.0
    # magnitude ratio |ah[d,d-2]|/|ah[d,d-1]| ~ |zeta| if oscillation clean
    per_head_even_frac = []
    for h in range(H):
        tot = pat_sum[h, 5:].sum()
        ev = sum(pat_sum[h, d, d - lag] for d in range(5, L)
                 for lag in range(0, L) if d - lag >= 0 and lag % 2 == 0)
        per_head_even_frac.append(float(ev / max(tot, 1e-12)))

    rep = dict(tag=args.tag, zeta=zeta, n_heads=H,
               per_head_ov_sign_by_token=ov_sign.tolist(),
               heads_antiparallel_signprod=antipar,
               per_head_ov_contrast_cosine=ov_cos,
               head_sum_lag_sign=lag_sign,
               per_head_even_lag_fraction=per_head_even_frac,
               note="head-sum lag-sign alternating => oscillating zeta^{d-s} recovered; "
                    "per-head single-signed + anti-parallel => per-head is gauge")
    (OUT / f"twohead_{args.tag}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
