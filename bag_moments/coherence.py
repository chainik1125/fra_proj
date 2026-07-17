"""Coherence metrics: projection of sampled behavior onto the coherent sector family.

Three deficits, one filter pass. For continuations x sampled from a model after a
prompt h, with q~_z the eta-thickened sector-z process (belief filtered through the
prompt, scored on continuation positions only):

  xe      deficit_xe(x)  = (1/T) [ log p_model(x|h) - max_z log q~_z(x|h) ]
          disposition = argmax_z. Sampled log-likelihood ratio: KL-style, in
          nats/token; slightly credits hedging mixtures (~ -log w_z / T).
  jsd     deficit_jsd(x) = min_z (1/T) sum_t JSD( p_model(.|h,x_<t) || q~_z(.|h,x_<t) )
          disposition = argmin_z. Bounded per token by ln 2; needs the model's full
          next-token distributions; charges hedging mixtures (they match no corner).
  mixjsd  deficit_mix(x) = min_w (1/T) sum_t JSD( p_model(.|h,x_<t) || q~_w(.|h,x_<t) )
          where q~_w is the rational agent with sector prior w whose belief is
          filtered along the continuation. Rational hedging is inside the family, so
          ideal learners score ~0 exactly; the fitted w-hat is a continuous
          disposition (the model's revealed sector posterior per continuation).

Bookkeeping choice (same as the original instrument): sector/mixture beliefs are
filtered through the prompt per corner, and the sector weight (z or w) is selected at
the continuation boundary. A prompt-forbidden sector therefore stays a live candidate
and a win by it is the prompt-inconsistent disposition (the flip) rather than a
deficit charge. The hard-violation counter (tokens with zero raw likelihood under all
sectors) is reported separately as before.

Generic over the process: pass the token operators (V, S, S) and the list of
per-sector initial state indices.
"""

from __future__ import annotations

import numpy as np


def _thick_step(belief: np.ndarray, op: np.ndarray, t_marg: np.ndarray, eta: float, vocab: int):
    """One thickened filter step. belief (B,S), op (B,S,S) already indexed by token."""
    raw_post = np.einsum("bs,bst->bt", belief, op)
    smooth_post = (1.0 - eta) * raw_post + (eta / vocab) * (belief @ t_marg)
    lik = smooth_post.sum(axis=1)
    new_belief = smooth_post / np.maximum(lik, 1e-300)[:, None]
    return new_belief, lik, raw_post.sum(axis=1)


def _thick_predictive(belief: np.ndarray, ops: np.ndarray, eta: float, vocab: int) -> np.ndarray:
    """Thickened next-token distribution (B,V) for beliefs (B,S)."""
    p = np.einsum("bs,vst->bv", belief, ops)
    p = (1.0 - eta) * p + eta / vocab
    return p / p.sum(axis=1, keepdims=True)


def _jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """JSD along the last axis; p, q broadcastable, rows on last axis sum to 1."""
    m = 0.5 * (p + q)
    logm = np.log(np.maximum(m, 1e-300))

    def kl(a):
        return np.where(a > 0, a * (np.log(np.maximum(a, 1e-300)) - logm), 0.0).sum(axis=-1)

    return 0.5 * kl(p) + 0.5 * kl(q)


def _simplex_grid(k: int) -> np.ndarray:
    """All compositions of k into 4 parts, normalized: coarse cover of the simplex."""
    pts = []
    for a in range(k + 1):
        for b in range(k + 1 - a):
            for c in range(k + 1 - a - b):
                pts.append((a, b, c, k - a - b - c))
    return np.asarray(pts, dtype=np.float64) / k


def sector_scan(
    ops: np.ndarray,
    sector_inits: list[int],
    ctx: np.ndarray,
    cont: np.ndarray,
    eta: float,
    need_predictives: bool = False,
) -> dict:
    """One filter pass per sector over ctx + cont.

    Returns dict with:
      logl            (B, Z)      thickened log-lik of the continuation per sector
      impossible_rate (B,)        fraction of cont tokens with zero raw lik under ALL sectors
      logl_running    (B, T, Z)   log-lik of cont tokens strictly before position t
      corner_pred     (B, T, Z, V) thickened per-sector predictive at each cont position
                                   (only if need_predictives)
    """
    vocab, n_states = ops.shape[0], ops.shape[1]
    n_sectors = len(sector_inits)
    t_marg = ops.sum(axis=0)
    obs = np.concatenate([ctx, cont], axis=1)
    n_ctx = ctx.shape[1]
    n_cont = obs.shape[1] - n_ctx
    B = obs.shape[0]

    logl = np.zeros((B, n_sectors))
    raw_lik_cont = np.zeros((B, n_sectors, n_cont))
    logl_running = np.zeros((B, n_cont, n_sectors))
    corner_pred = (
        np.zeros((B, n_cont, n_sectors, vocab)) if need_predictives else None
    )

    for z, init in enumerate(sector_inits):
        belief = np.zeros((B, n_states))
        belief[:, init] = 1.0
        acc = np.zeros(B)
        for t in range(obs.shape[1]):
            if t >= n_ctx:
                tc = t - n_ctx
                logl_running[:, tc, z] = acc
                if need_predictives:
                    corner_pred[:, tc, z, :] = _thick_predictive(belief, ops, eta, vocab)
            belief, lik, raw = _thick_step(belief, ops[obs[:, t]], t_marg, eta, vocab)
            if t >= n_ctx:
                acc = acc + np.log(np.maximum(lik, 1e-300))
                raw_lik_cont[:, z, t - n_ctx] = raw
        logl[:, z] = acc

    out = {
        "logl": logl,
        "impossible_rate": (raw_lik_cont <= 1e-300).all(axis=1).mean(axis=1),
        "logl_running": logl_running,
    }
    if need_predictives:
        out["corner_pred"] = corner_pred
    return out


def _mixture_jsd(
    model_probs: np.ndarray,
    corner_pred: np.ndarray,
    logl_running: np.ndarray,
    cand: np.ndarray,
    chunk: int = 8,
) -> np.ndarray:
    """Mean-per-token JSD to the filtered w-mixture for candidate priors.

    model_probs (B,T,V); corner_pred (B,T,Z,V); logl_running (B,T,Z); cand (K,Z).
    The rational agent with prior w has predictive
      p_w(.|prefix) = sum_z omega_z q_z(.|prefix),  omega ~ softmax(log w + logl_running)
    Returns (B, K) mean-per-token JSD.
    """
    B, T, Z, V = corner_pred.shape
    K = cand.shape[0]
    out = np.empty((B, K))
    logw = np.log(np.maximum(cand, 1e-300))  # (K, Z)
    pm = model_probs[:, :, None, :]  # (B,T,1,V)
    for k0 in range(0, K, chunk):
        lw = logw[k0 : k0 + chunk]  # (k,Z)
        s = logl_running[:, :, None, :] + lw[None, None, :, :]  # (B,T,k,Z)
        s = s - s.max(axis=3, keepdims=True)
        om = np.exp(s)
        om /= om.sum(axis=3, keepdims=True)
        mix = np.einsum("btkz,btzv->btkv", om, corner_pred)  # (B,T,k,V)
        out[:, k0 : k0 + chunk] = _jsd(pm, mix).mean(axis=1)  # (B,k)
    return out


def fit_mixture_jsd(
    model_probs: np.ndarray,
    corner_pred: np.ndarray,
    logl_running: np.ndarray,
    coarse_k: int = 4,
    refine_fracs: tuple[float, ...] = (0.5, 0.25, 0.125),
) -> tuple[np.ndarray, np.ndarray]:
    """min over sector priors w of the trajectory-mean JSD; returns (deficit (B,), w_hat (B,Z)).

    Coarse simplex grid (includes the corners, so mixjsd <= corner jsd by
    construction), then one refinement pass mixing each continuation's best point
    toward every corner.
    """
    Z = corner_pred.shape[2]
    grid = _simplex_grid(coarse_k)  # (K, Z), includes corners
    vals = _mixture_jsd(model_probs, corner_pred, logl_running, grid)  # (B, K)
    best = vals.min(axis=1)
    w_best = grid[vals.argmin(axis=1)]  # (B, Z)

    # refinement: candidates per continuation = w_best mixed toward each corner
    B = w_best.shape[0]
    eyes = np.eye(Z)
    for frac in refine_fracs:
        cands = (1 - frac) * w_best[:, None, :] + frac * eyes[None, :, :]  # (B,Z,Z)
        for j in range(Z):
            v = _mixture_jsd_single(model_probs, corner_pred, logl_running, cands[:, j, :])
            improved = v < best
            best = np.where(improved, v, best)
            w_best = np.where(improved[:, None], cands[:, j, :], w_best)
    return best, w_best


def _mixture_jsd_single(
    model_probs: np.ndarray,
    corner_pred: np.ndarray,
    logl_running: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Like _mixture_jsd but one candidate PER continuation: w (B,Z) -> (B,)."""
    logw = np.log(np.maximum(w, 1e-300))  # (B,Z)
    s = logl_running + logw[:, None, :]  # (B,T,Z)
    s = s - s.max(axis=2, keepdims=True)
    om = np.exp(s)
    om /= om.sum(axis=2, keepdims=True)
    mix = np.einsum("btz,btzv->btv", om, corner_pred)  # (B,T,V)
    return _jsd(model_probs, mix).mean(axis=1)


ALL_METRICS = ("xe", "jsd", "mixjsd")


def coherence_metrics_all(
    ops: np.ndarray,
    sector_inits: list[int],
    sector_names: tuple[str, ...],
    ctx: np.ndarray,
    cont: np.ndarray,
    logp_model: np.ndarray,
    prefix: str,
    eta: float,
    model_probs: np.ndarray | None = None,
    metrics: tuple[str, ...] = ALL_METRICS,
    k_prompts: int = 0,
) -> dict[str, float]:
    """Compute the selected coherence metrics; returns a flat {column: value} dict.

    logp_model (B,T): model's own log-probs of the tokens it sampled (all metrics).
    model_probs (B,T,V): model's full next-token distributions at each continuation
    position (required for 'jsd' and 'mixjsd'; recorded during sampling).
    """
    metrics = tuple(m for m in metrics if m in ALL_METRICS)
    need_pred = bool({"jsd", "mixjsd"} & set(metrics)) and model_probs is not None
    scan = sector_scan(ops, sector_inits, ctx, cont, eta, need_predictives=need_pred)
    gen_len = cont.shape[1]
    out: dict[str, float] = {
        f"{prefix}_coh_impossible_rate": float(scan["impossible_rate"].mean()),
    }

    if "xe" in metrics:
        best = scan["logl"].max(axis=1)
        deficit = (logp_model.sum(axis=1) - best) / gen_len
        dispo = scan["logl"].argmax(axis=1)
        out[f"{prefix}_coh_deficit"] = float(deficit.mean())
        for i, name in enumerate(sector_names):
            out[f"{prefix}_dispo_{name}"] = float((dispo == i).mean())
        if k_prompts:
            out[f"{prefix}_coh_deficit_bp_std"] = float(
                deficit.reshape(k_prompts, -1).mean(axis=1).std()
            )

    if need_pred:
        corner_pred = scan["corner_pred"]  # (B,T,Z,V)
        if "jsd" in metrics:
            per = _jsd(model_probs[:, :, None, :], corner_pred).mean(axis=1)  # (B,Z)
            out[f"{prefix}_jsd_deficit"] = float(per.min(axis=1).mean())
            dispo = per.argmin(axis=1)
            for i, name in enumerate(sector_names):
                out[f"{prefix}_jsd_dispo_{name}"] = float((dispo == i).mean())
            if k_prompts:
                out[f"{prefix}_jsd_deficit_bp_std"] = float(
                    per.min(axis=1).reshape(k_prompts, -1).mean(axis=1).std()
                )
        if "mixjsd" in metrics:
            deficit, w_hat = fit_mixture_jsd(model_probs, corner_pred, scan["logl_running"])
            out[f"{prefix}_mixjsd_deficit"] = float(deficit.mean())
            for i, name in enumerate(sector_names):
                out[f"{prefix}_mixw_{name}"] = float(w_hat[:, i].mean())
            if k_prompts:
                out[f"{prefix}_mixjsd_deficit_bp_std"] = float(
                    deficit.reshape(k_prompts, -1).mean(axis=1).std()
                )
    return out
