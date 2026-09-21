"""Bayes collateral floor: optimal within-block CE when the block identity of
the HISTORY is hidden (tokens collapsed to within-block symbols s = v mod 3).

This bounds the collateral of any intervention achieving FULL concept removal:
a representation carrying zero omega-information cannot know which component
emitted each past token, so its belief-state tracking is at best the
collapsed-history filter computed here.

Exact filtering is intractable (assignments are latent), so we use a
Rao-Blackwellized particle filter: each particle carries exact per-component
beliefs + assignment counts; assignments are sampled, omega is
Rao-Blackwellized via Dirichlet counts. Prediction of s_{t+1} conditions on
the true next block c_{t+1} (the within factor's conditioning event), with
the particle posterior reweighted by omega_hat_{c_{t+1}}.

Outputs floor within-CE, full-info Bayes within-CE (sanity vs anchors), and
the irreducible collateral fraction, with a particle-count convergence check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mixture_data import MESS3_SEPARATED, make_dataset


@torch.no_grad()
def collapsed_filter_ce(
    tokens: torch.Tensor,        # (B, T) global tokens (only used collapsed)
    n_particles: int,
    concentration: float,
    seed: int,
    t_min: int = 8,
    resample_frac: float = 0.5,
) -> torch.Tensor:
    """Per-point -log p(s_{t+1} | c_{t+1}, collapsed history). (B, T-1-t_min)."""
    gen = torch.Generator().manual_seed(seed)
    B, T = tokens.shape
    K = MESS3_SEPARATED.K
    comps = MESS3_SEPARATED.components
    Ts = torch.stack([c.transition_matrices for c in comps])       # (K, V, S, S)
    # emit P(v|state i) = sum_j T[v,i,j], transposed to (S, V)
    emits = torch.stack([c.transition_matrices.sum(dim=2).T for c in comps])  # (K, S, V)

    s_obs = tokens % 3                                             # collapsed symbol
    c_true = tokens // 3                                           # true block

    N = n_particles
    alpha = torch.tensor(MESS3_SEPARATED.omega) * concentration    # (K,)
    beliefs = torch.full((B, N, K, 3), 1.0 / 3.0)
    counts = torch.zeros(B, N, K)
    logw = torch.zeros(B, N)

    out_nll = torch.zeros(B, T - 1)

    for t in range(T - 1):
        # ── assimilate s_t ──
        s_t = s_obs[:, t]                                          # (B,)
        omega_hat = (counts + alpha) / (counts.sum(-1, keepdim=True) + concentration)  # (B,N,K)
        # pred_c[s] = belief_c @ emit_c : (B,N,K,V)
        pred = torch.einsum("bnks,ksv->bnkv", beliefs, emits)
        p_s_given_c = pred.gather(-1, s_t.view(B, 1, 1, 1).expand(B, N, K, 1)).squeeze(-1)  # (B,N,K)
        joint = omega_hat * p_s_given_c                            # (B,N,K)
        like = joint.sum(-1).clamp(min=1e-30)                      # (B,N)
        logw = logw + like.log()

        # sample assignment c* per particle
        post_c = joint / like.unsqueeze(-1)
        c_star = torch.multinomial(post_c.reshape(B * N, K), 1, generator=gen).reshape(B, N)
        counts.scatter_add_(-1, c_star.unsqueeze(-1), torch.ones(B, N, 1))

        # belief update for the assigned component with symbol s_t
        Tv = Ts[c_star.reshape(-1), s_obs[:, t].repeat_interleave(N)]  # (B*N, S, S)
        bel_sel = beliefs.gather(2, c_star.view(B, N, 1, 1).expand(B, N, 1, 3)).reshape(B * N, 1, 3)
        new_unnorm = torch.bmm(bel_sel, Tv).squeeze(1)             # (B*N, S)
        new_bel = new_unnorm / new_unnorm.sum(-1, keepdim=True).clamp(min=1e-30)
        beliefs = beliefs.scatter(
            2, c_star.view(B, N, 1, 1).expand(B, N, 1, 3), new_bel.reshape(B, N, 1, 3)
        )

        # ── resample if ESS low ──
        w = torch.softmax(logw, dim=-1)
        ess = 1.0 / (w.pow(2).sum(-1))                             # (B,)
        need = ess < resample_frac * N
        if bool(need.any()):
            idx_b = need.nonzero(as_tuple=True)[0]
            sel = torch.multinomial(w[idx_b], N, replacement=True, generator=gen)  # (nb, N)
            beliefs[idx_b] = beliefs[idx_b].gather(1, sel.view(-1, N, 1, 1).expand(-1, N, K, 3))
            counts[idx_b] = counts[idx_b].gather(1, sel.view(-1, N, 1).expand(-1, N, K))
            logw[idx_b] = 0.0

        # ── predict s_{t+1} given true c_{t+1} ──
        w = torch.softmax(logw, dim=-1)                            # (B,N)
        omega_hat = (counts + alpha) / (counts.sum(-1, keepdim=True) + concentration)
        c_next = c_true[:, t + 1]                                  # (B,)
        w_cond = w * omega_hat.gather(-1, c_next.view(B, 1, 1).expand(B, N, 1)).squeeze(-1)
        w_cond = w_cond / w_cond.sum(-1, keepdim=True).clamp(min=1e-30)
        pred = torch.einsum("bnks,ksv->bnkv", beliefs, emits)      # (B,N,K,V)
        pred_c = pred.gather(2, c_next.view(B, 1, 1, 1).expand(B, N, 1, 3)).squeeze(2)  # (B,N,V)
        p_floor = (w_cond.unsqueeze(-1) * pred_c).sum(1)           # (B,V)
        s_next = s_obs[:, t + 1]
        out_nll[:, t] = -p_floor.gather(-1, s_next.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12).log()

    return out_nll[:, t_min:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/bayes_floor.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--concentration", type=float, default=10.0)
    ap.add_argument("--particles", type=int, default=1024)
    ap.add_argument("--t-min", type=int, default=8)
    ap.add_argument("--conv-check-seqs", type=int, default=100)
    args = ap.parse_args()

    eval_ds = make_dataset(args.seed + 1, args.n_eval, args.seq_len, args.concentration)

    # full-info Bayes within CE (sanity — must match run anchors)
    from metrics import bayes_anchors
    anchors = bayes_anchors(eval_ds, slice(None), t_min=args.t_min)
    print(f"full-info bayes within CE: {anchors['bayes_within_ce']:.4f} (uniform {anchors['uniform_within_ce']:.4f})", flush=True)

    # convergence check on a subset
    conv = {}
    for n in (256, 512, args.particles):
        nll = collapsed_filter_ce(
            eval_ds.tokens[: args.conv_check_seqs], n, args.concentration, args.seed, args.t_min,
        )
        conv[n] = float(nll.mean())
        print(f"  particles={n:5d}  floor withinCE (subset) = {conv[n]:.4f}", flush=True)

    # full eval
    nll = collapsed_filter_ce(eval_ds.tokens, args.particles, args.concentration, args.seed, args.t_min)
    floor = float(nll.mean())
    cf_irreducible = (floor - anchors["bayes_within_ce"]) / (
        anchors["uniform_within_ce"] - anchors["bayes_within_ce"]
    )
    print(f"\ncollapsed-history floor withinCE = {floor:.4f}", flush=True)
    print(f"irreducible collateral fraction at full removal = {cf_irreducible:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "bayes_within_ce": anchors["bayes_within_ce"],
        "uniform_within_ce": anchors["uniform_within_ce"],
        "floor_within_ce": floor,
        "irreducible_collateral_frac": cf_irreducible,
        "convergence_check": conv,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }, indent=2))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
