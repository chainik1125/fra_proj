"""Phase-1 battery on attention-only Mess3 models (adapts sprint verify_p2.py).

Attn-only => belief lives at resid_post = resid_pre + attn_out (NO resid_mid/MLP).
Checks (single-/multi-layer aware): (1) pattern decay rate + token-independence;
per-head OV parallel to g(z); embeddings parallel to OV. (3) belief probes
resid_post -> r1 (constrained) vs eta (Bayes) R^2. Writes out/verify_<tag>.json.

Usage: verify_ao.py <model_tag>   (loads out/model_<tag>.pt)
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch
from pathlib import Path
from mess3 import Mess3, enumerate_sequences, prefix_probs

torch.set_num_threads(4)
N_CTX, D_MODEL, CHUNK = 10, 64, 4096
OUT = Path(__file__).resolve().parent.parent / "out"


def build_ao(n_heads, n_layers, seed=42):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    cfg = HookedTransformerConfig(
        n_layers=n_layers, d_model=D_MODEL, n_ctx=N_CTX, d_head=D_MODEL // n_heads,
        n_heads=n_heads, d_vocab=3, d_vocab_out=3, attn_only=True,
        normalization_type="LN", attention_dir="causal", default_prepend_bos=False,
        positional_embedding_type="standard", device="cpu", seed=seed)
    return HookedTransformer(cfg)


def load(tag):
    d = torch.load(OUT / f"model_{tag}.pt", map_location="cpu")
    c = d["cfg"]
    m = build_ao(c["n_heads"], c["n_layers"], c["seed"])
    m.load_state_dict(d["state_dict"]); m.eval()
    return m, c, Mess3(c["x"], c["a"])


def ln1_maker(model, layer):
    """Frozen-LN (ln1) map as content-independent per-position affine (mean scale)."""
    # approximate LN as centering + fixed scale (population); good enough for OV dir
    def ln(u):                       # u: (..., D)
        v = u - u.mean(-1, keepdims=True)
        return v / (np.sqrt((v ** 2).mean(-1, keepdims=True)) + 1e-6)
    return ln


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); args = ap.parse_args()
    model, c, m = load(args.tag)
    L, H, D, Lyr = N_CTX, c["n_heads"], D_MODEL, c["n_layers"]
    zeta = 1 - 3 * c["x"]
    seqs = enumerate_sequences(L); pp = prefix_probs(m, seqs); w_seq = pp[:, L]
    r1 = m.constrained_beliefs(seqs); eta = m.bayes_beliefs(seqs)
    last = Lyr - 1
    names = {f"blocks.{last}.hook_resid_post", f"blocks.{last}.hook_attn_out"}
    names |= {f"blocks.{l}.attn.hook_pattern" for l in range(Lyr)}

    pat_sum = np.zeros((Lyr, H, L, L))
    pat_tok = np.zeros((Lyr, 3, H, L, L)); pat_tok_w = np.zeros((3, L))
    D1 = D + 1
    gxx = np.zeros((D1, D1)); gxy_r1 = np.zeros((D1, 3)); gxy_eta = np.zeros((D1, 3))
    yy_r1 = yy_eta = 0.0; y_r1 = np.zeros(3); y_eta = np.zeros(3); wtot = 0.0
    for i in range(0, seqs.shape[0], CHUNK):
        sl = slice(i, i + CHUNK); toks = torch.tensor(seqs[sl])
        _, cache = model.run_with_cache(toks, return_type=None,
                                        names_filter=lambda x: x in names)
        wc = w_seq[sl]
        for l in range(Lyr):
            pat = cache[f"blocks.{l}.attn.hook_pattern"].numpy()   # (c,H,L,L)
            pat_sum[l] += np.einsum("c,chds->hds", wc, pat)
            for z in range(3):
                msk = (seqs[sl] == z).astype(float)
                pat_tok[l, z] += np.einsum("c,cs,chds->hds", wc, msk, pat)
        for z in range(3):
            pat_tok_w[z] += wc @ (seqs[sl] == z).astype(float)
        X = cache[f"blocks.{last}.hook_resid_post"].numpy().reshape(-1, D)
        X1 = np.concatenate([X, np.ones((X.shape[0], 1))], 1)
        yr = r1[sl, 1:L + 1].reshape(-1, 3); ye = eta[sl, 1:L + 1].reshape(-1, 3)
        wf = np.repeat(wc, L) / L
        Xw = X1 * wf[:, None]
        gxx += X1.T @ Xw; gxy_r1 += Xw.T @ yr; gxy_eta += Xw.T @ ye
        yy_r1 += wf @ (yr ** 2).sum(1); yy_eta += wf @ (ye ** 2).sum(1)
        y_r1 += wf @ yr; y_eta += wf @ ye; wtot += wf.sum()

    def probe(gxy, yy, ym):
        A = gxx.copy(); A.flat[::D1 + 1] += 1e-8 * np.trace(A) / D1
        B = np.linalg.solve(A, gxy)
        ss_res = yy - 2 * (B * gxy).sum() + np.einsum("ij,ik,kj->", B, gxx, B)
        mu = ym / wtot; ss_tot = yy - wtot * (mu ** 2).sum()
        return float(1 - ss_res / ss_tot)
    r2_r1 = probe(gxy_r1, yy_r1, y_r1); r2_eta = probe(gxy_eta, yy_eta, y_eta)

    # pattern decay rate (last layer, head-sum), lag ratios rows d=5..8
    def lag_ratios(pm, step=1):
        rr = []
        for d in range(5, L - 1):
            for nlag in range(1, d - 1):
                a1, a2 = pm[d, d - nlag], pm[d, d - nlag - step]
                if a1 > 1e-9 and a2 > 1e-9:
                    rr.append((a2 / a1) ** (1.0 / step))
        return float(np.mean(rr)), float(np.std(rr))
    pm_head = pat_sum[last].sum(0)
    ratio, ratio_sd = lag_ratios(pm_head)
    # token-independence: relstd of E[A|z_s]/E[A] across z (last layer head-sum)
    Ez = pat_tok[last].sum(1) / np.maximum(pat_tok_w[:, None, :], 1e-12)  # (3,L,L): z,d,s
    base = pm_head[None]
    msk = base > 1e-6
    reltok = float(np.std([Ez[z][msk[0]] / base[0][msk[0]] for z in range(3)], 0).mean())

    # OV per head (last layer) parallel to g(z)? f = ln1(e(z)+p) W_V W_O, project to
    # belief plane, compare direction to g(z). Use embeddings + pos.
    W_E = model.W_E.detach().numpy(); W_pos = model.W_pos.detach().numpy()
    ln = ln1_maker(model, last)
    x_zp = W_E[:, None, :] + W_pos[None, :, :]              # (3,L,D)
    g = np.stack([m.pi @ m.T_cond[z] - m.pi for z in range(3)])   # (3,3) belief plane
    blk = model.blocks[last]
    W_V = blk.attn.W_V.detach().numpy(); W_O = blk.attn.W_O.detach().numpy()
    # readout to belief plane: fit linear resid->r1 gives R; use B from probe? use
    # simple: project OV output onto g(z) via the probe map columns (approx).
    ov_cos = {}
    for h in range(H):
        fvb = ln(x_zp) @ W_V[h] @ W_O[h]                   # (3,L,D) resid contribution
        # per token z, average over positions, direction in resid space
        vz = fvb.mean(1)                                    # (3, D)
        # cosine between tokens' OV dirs and whether they separate like g (rank check)
        vz_c = vz - vz.mean(0)
        # cos of pairwise OV vs pairwise g (in their own spaces): compare gram angles
        Gov = vz_c @ vz_c.T; Gg = (g - g.mean(0)) @ (g - g.mean(0)).T
        ov_cos[f"head{h}_ov_vs_g_gram_cos"] = float(
            (Gov * Gg).sum() / (np.linalg.norm(Gov) * np.linalg.norm(Gg) + 1e-12))

    rep = dict(tag=args.tag, cfg=c, zeta=zeta,
               belief_probe=dict(r2_resid_post_to_r1=r2_r1, r2_resid_post_to_eta=r2_eta),
               pattern=dict(decay_rate=ratio, decay_rate_sd=ratio_sd,
                            zeta=zeta, token_relstd=reltok),
               ov_vs_g=ov_cos)
    (OUT / f"verify_{args.tag}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
