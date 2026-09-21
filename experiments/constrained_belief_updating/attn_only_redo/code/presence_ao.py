"""Check 6 — presence floor on attention-only Mess3 (the user's question).

Ridge probe (retrained per condition) for the full Bayes belief eta (3-state) on
the post-attention residual. Conditions: (a) clean resid_post; (b) SEVER the full
attention channel (attn_out:=0 => resid_post = resid_pre = current token+pos);
(c) the single-observation floor: probe on the current-token one-hot alone.
Prediction: (b) falls to (c) exactly — with no MLP, severing the whole attention
channel reaches the single-observation floor, the current token being outside FRA's
reach (it enters resid_post via the skip/embedding path).

Usage: presence_ao.py <model_tag>
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


def wr2(X, Y, wf):
    """Population-weighted ridge R^2 of X->Y (in-sample = population for exact enum)."""
    X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = (X1 * wf[:, None]).T @ X1; A.flat[::A.shape[0] + 1] += 1e-8 * np.trace(A) / A.shape[0]
    B = np.linalg.solve(A, (X1 * wf[:, None]).T @ Y)
    pred = X1 @ B
    ss_res = (wf[:, None] * (Y - pred) ** 2).sum()
    mu = (wf[:, None] * Y).sum(0) / wf.sum()
    ss_tot = (wf[:, None] * (Y - mu) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); args = ap.parse_args()
    model, c, m = load(args.tag); Ll = c["n_layers"] - 1
    seqs = enumerate_sequences(L); pp = prefix_probs(m, seqs); w = pp[:, L]
    eta = m.bayes_beliefs(seqs)                             # (n,L+1,3)
    names = {f"blocks.{Ll}.hook_resid_post", f"blocks.{Ll}.hook_resid_pre"}
    n = seqs.shape[0]
    RPost = np.zeros((n, L, D_MODEL)); RPre = np.zeros((n, L, D_MODEL))
    for i in range(0, n, 4096):
        toks = torch.tensor(seqs[i:i + 4096])
        _, cache = model.run_with_cache(toks, return_type=None,
                                        names_filter=lambda x: x in names)
        RPost[i:i + 4096] = cache[f"blocks.{Ll}.hook_resid_post"].numpy()
        RPre[i:i + 4096] = cache[f"blocks.{Ll}.hook_resid_pre"].numpy()
    Y = eta[:, 1:L + 1].reshape(-1, 3); wf = np.repeat(w, L) / L
    onehot = np.eye(3)[seqs].reshape(-1, 3)                 # current-token one-hot

    r2_clean = wr2(RPost.reshape(-1, D_MODEL), Y, wf)
    r2_sever = wr2(RPre.reshape(-1, D_MODEL), Y, wf)        # attn_out removed
    r2_floor = wr2(onehot, Y, wf)                            # single-observation floor
    # attention share of decodable presence
    share = (r2_clean - r2_sever) / max(r2_clean, 1e-9)
    rep = dict(tag=args.tag, cfg=c,
               r2_clean_resid_post=r2_clean,
               r2_sever_full_attention=r2_sever,
               single_obs_floor_current_token=r2_floor,
               attention_presence_share=float(share),
               sever_minus_floor=float(r2_sever - r2_floor))
    (OUT / f"presence_{args.tag}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
