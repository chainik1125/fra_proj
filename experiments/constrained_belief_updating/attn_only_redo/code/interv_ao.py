"""Check 3 — FRA-OV closed forms + interventions on attention-only Mess3.

resid_post = resid_pre + attn_out; attn_out_d = sum_s A_{d,s} (v_s @ W_O). The
FRA-OV contribution of KEY-TOKEN z* is T_{z*}(d) = sum_{s: z_s=z*} A_{d,s}(v_s W_O).
Sever at gain c: resid_post' = resid_post - c * T_{z*}. Measure the belief-plane
tracking of g(z*) (slope of the probe-read belief onto the constrained r1 along
g(z*)) vs gain -> linear, null at c* = 1/rho; and the CE(c) behavioural curve.

Usage: interv_ao.py <model_tag>
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch
from pathlib import Path
from mess3 import Mess3, enumerate_sequences, prefix_probs
from verify_ao import build_ao, N_CTX, D_MODEL

torch.set_num_threads(4)
OUT = Path(__file__).resolve().parent.parent / "out"; L = N_CTX


def load(tag):
    d = torch.load(OUT / f"model_{tag}.pt", map_location="cpu")
    c = d["cfg"]; m = build_ao(c["n_heads"], c["n_layers"], c["seed"])
    m.load_state_dict(d["state_dict"]); m.eval(); return m, c, Mess3(c["x"], c["a"])


def gather(model, m):
    """Enumerate; return resid_post, per-key-token OV transport T[z](n,L,D), weights,
    r1 belief, and the final-LN+unembed closure for CE."""
    Ll = model.cfg.n_layers - 1
    seqs = enumerate_sequences(L); pp = prefix_probs(m, seqs); w = pp[:, L]
    r1 = m.constrained_beliefs(seqs); eta = m.bayes_beliefs(seqs)
    names = {f"blocks.{Ll}.hook_resid_post", f"blocks.{Ll}.attn.hook_pattern",
             f"blocks.{Ll}.attn.hook_z", f"blocks.{Ll}.hook_resid_pre"}
    W_O = model.blocks[Ll].attn.W_O.detach()            # (H, dh, D)
    n = seqs.shape[0]; D = D_MODEL
    RP = np.zeros((n, L, D)); Tz = np.zeros((3, n, L, D))
    for i in range(0, n, 4096):
        sub = seqs[i:i + 4096]; toks = torch.tensor(sub)
        _, cache = model.run_with_cache(toks, return_type=None,
                                        names_filter=lambda x: x in names)
        rp = cache[f"blocks.{Ll}.hook_resid_post"]      # (c,L,D)
        A = cache[f"blocks.{Ll}.attn.hook_pattern"]     # (c,H,Lq,Lk)
        z = cache[f"blocks.{Ll}.attn.hook_z"]           # (c,Lk,H,dh)
        vO = torch.einsum("cshd,hde->cse", z, W_O)      # (c,Lk,D) per-source OV out
        RP[i:i + 4096] = rp.numpy()
        for zt in range(3):
            msk = torch.tensor((sub == zt)).float()     # (c,Lk)
            # T[zt](d) = sum_{s:z_s=zt} A_{d,s} vO_s
            Tz[zt, i:i + 4096] = torch.einsum(
                "chds,cs,cse->cde", A, msk, vO).numpy()
    return seqs, w, r1, eta, RP, Tz


def readout_map(RP, r1, w):
    """Ridge R: resid_post(+1) -> r1 belief (population weighted)."""
    X = RP.reshape(-1, RP.shape[-1]); X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
    Y = r1[:, 1:L + 1].reshape(-1, 3); wf = np.repeat(w, L) / L
    A = (X1 * wf[:, None]).T @ X1; A.flat[::A.shape[0] + 1] += 1e-6 * np.trace(A) / A.shape[0]
    B = np.linalg.solve(A, (X1 * wf[:, None]).T @ Y)
    return B                                            # (D+1, 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); args = ap.parse_args()
    model, c, m = load(args.tag)
    seqs, w, r1, eta, RP, Tz = gather(model, m)
    B = readout_map(RP, r1, w)
    g = np.stack([m.pi @ m.T_cond[z] - m.pi for z in range(3)])   # (3,3) belief plane
    ghat = g / np.linalg.norm(g, axis=1, keepdims=True)
    wf = np.repeat(w, L) / L
    r1d = r1[:, 1:L + 1].reshape(-1, 3)                  # true belief per (seq,pos)

    def read(rp):                                        # belief estimate from resid
        X1 = np.concatenate([rp.reshape(-1, rp.shape[-1]),
                             np.ones((rp.shape[0] * L, 1))], 1)
        return X1 @ B                                    # (n*L, 3)

    gains = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    res = {}
    for zt in range(3):
        # tracking slope of read.g(zt) on true r1.g(zt), per gain
        tv = r1d @ ghat[zt]                              # true belief along g(zt)
        s0 = None; slopes = []
        for cq in gains:
            rp_c = RP - cq * Tz[zt]
            pv = read(rp_c) @ ghat[zt]
            # weighted slope of pv on tv
            tvc = tv - (wf * tv).sum() / wf.sum(); pvc = pv - (wf * pv).sum() / wf.sum()
            slope = (wf * tvc * pvc).sum() / (wf * tvc * tvc).sum()
            slopes.append(float(slope))
        s0 = slopes[0]
        rel = [s / s0 for s in slopes]
        rho = -np.polyfit(gains, rel, 1)[0]             # rel ~ 1 - rho*c
        res[f"z{zt}"] = dict(slopes_rel=rel, rho=float(rho), cstar=float(1 / rho))
    rep = dict(tag=args.tag, cfg=c, gains=gains, tracking=res)
    (OUT / f"interv_{args.tag}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
