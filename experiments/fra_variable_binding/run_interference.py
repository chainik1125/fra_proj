"""Reproducible CPU study of selective binding edits at fixed QK width."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp


def softmax(s):
    return np.exp(s - logsumexp(s, axis=-1, keepdims=True))


def center(s):
    return s - s.mean(axis=-1, keepdims=True)


def memories(rng, size, n=16, f=32, heldout=False):
    scores = rng.random((size, n, f))
    forbidden = (np.arange(n)[:, None] + np.arange(f)) % 4 == 0
    scores[:, forbidden if not heldout else ~forbidden] = np.inf
    result = np.empty((size, n), dtype=int)
    for j in range(n):
        result[:, j] = scores[:, j].argmin(-1)
        scores[np.arange(size), :, result[:, j]] = np.inf
    assert np.all(np.diff(np.sort(result, axis=-1), axis=-1) > 0)
    assert np.all(((np.arange(n) + result) % 4 == 0) == heldout)
    return result


def train_loss_grad(q, k, c, vals, queries):
    n, d = q.shape
    keys = k[None] + c[vals]
    sq = q[queries]
    scores = np.einsum("bd,bnd->bn", sq, keys)
    a = softmax(scores)
    loss = np.mean(logsumexp(scores, axis=-1) - scores[np.arange(len(vals)), queries])
    gs = a.copy()
    gs[np.arange(len(vals)), queries] -= 1
    gs /= len(vals)
    gq = np.zeros_like(q)
    np.add.at(gq, queries, np.einsum("bn,bnd->bd", gs, keys))
    gkeys = gs[:, :, None] * sq[:, None]
    gk = gkeys.sum(0)
    gc = np.zeros_like(c)
    np.add.at(gc, vals.reshape(-1), gkeys.reshape(-1, d))
    return float(loss), [gq, gk, gc]


def train(n, f, d, seed, steps):
    rng = np.random.default_rng(seed + 100)
    params = [rng.normal(0, .2, (n, d)), rng.normal(0, .2, (n, d)), rng.normal(0, .05, (f, d))]
    ms = [np.zeros_like(p) for p in params]
    vs = [np.zeros_like(p) for p in params]
    history = []
    for step in range(1, steps + 1):
        vals = memories(rng, 64, n, f)
        queries = rng.integers(n, size=len(vals))
        loss, grads = train_loss_grad(*params, vals, queries)
        for p, g, m, v in zip(params, grads, ms, vs):
            g = g + 0.0002 * p
            m *= .9
            m += .1 * g
            v *= .999
            v += .001 * g * g
            p -= .025 * (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
        if step % 400 == 0 or step == steps:
            history.append({"step": step, "loss": loss})
    q, k, c = params
    calibration = memories(np.random.default_rng(831), 128, n, f)
    s = np.einsum("id,bjd->bij", q, k[None] + c[calibration])
    def confidence(t):
        return float(np.diagonal(softmax(t * s), axis1=-2, axis2=-1).mean())
    if confidence(100) >= .85:
        scale = brentq(lambda t: confidence(t) - .85, .0001, 100)
    else:
        scale = 1.
    q = q * scale
    return q, k, c, {"history": history, "temperature_scale": scale, "calibration_confidence": confidence(scale)}


def frame(n, d):
    rows = np.arange(n)
    cols = []
    for freq in range(1, n // 2):
        cols += [np.sqrt(2 / n) * np.cos(2 * np.pi * freq * rows / n),
                 np.sqrt(2 / n) * np.sin(2 * np.pi * freq * rows / n)]
    cols.append((-1.) ** rows / np.sqrt(n))
    u = np.column_stack(cols[:d])
    assert np.max(np.abs(u.T @ u - np.eye(d))) < 1e-12
    assert np.max(np.abs(u.mean(0))) < 1e-12
    assert np.max(np.abs(np.sum(u * u, -1) - d / n)) < 1e-12
    beta = brentq(lambda t: np.diag(softmax(t * u @ u.T)).mean() - .85, .01, 10000)
    return np.sqrt(beta) * u, np.sqrt(beta) * u


def solve_q(target, k, a, b):
    # Minimum least-squares fit, with one centered target score constrained exactly.
    k = k - k.mean(0)
    q = target @ np.linalg.pinv(k.T)
    v = np.linalg.pinv(k.T @ k, rcond=1e-12) @ k[b]
    denom = k[b] @ v
    if denom < 1e-15:
        raise ValueError("Target contrast is outside the available Q span")
    q[a] += (target[a, b] - q[a] @ k[b]) * v / denom
    return q, k


def solve_k(target, q, a, b):
    n = len(target)
    k = target.T @ np.linalg.pinv(q.T)
    k -= k.mean(0)
    r = np.eye(n)[b] - 1 / n
    v = np.linalg.pinv(q.T @ q, rcond=1e-12) @ q[a]
    denom = r[b] * (q[a] @ v)
    if denom < 1e-15:
        raise ValueError("Target contrast is outside the available K span")
    k += np.outer(r, v) * (target[a, b] - q[a] @ k[b]) / denom
    return q, k


def joint_score(target, q0, k0, a, b, maxiter=250):
    d = q0.shape[1]
    u, s, vt = np.linalg.svd(target, full_matrices=False)
    qs = u[:, :d] * np.sqrt(s[:d])
    ks = vt[:d].T * np.sqrt(s[:d])
    starts = [(q0, k0), (qs, ks), solve_q(target, k0, a, b), solve_k(target, q0, a, b)]
    best = None
    for q, k in starts:
        previous = np.inf
        for it in range(maxiter):
            q, k = solve_q(target, k, a, b)
            q, k = solve_k(target, q, a, b)
            loss = np.sum((q @ k.T - target) ** 2)
            if abs(previous - loss) < 1e-11 * max(1., previous if np.isfinite(previous) else 1.):
                break
            previous = loss
        if best is None or loss < best[0]:
            best = (loss, q.copy(), k.copy(), it + 1)
    return best[1], best[2], {"iterations": best[3], "score_sse": best[0], "rank_lower_bound": float(np.sum(s[d:] ** 2))}


def candidate_directions(q, k, key_features, a, b):
    n = len(q)
    qdirs, kdirs = [], []
    for feature in q:
        ds = np.zeros((n, n))
        ds[a] = feature @ k.T
        qdirs.append(ds)
    for feature in key_features:
        ds = np.zeros((n, n))
        ds[:, b] = q @ feature
        kdirs.append(ds)
    qdom = np.zeros((n, n))
    qdom[a] = (q[a] - q[b]) @ k.T
    kdom = np.zeros((n, n))
    kdom[:, b] = q @ (key_features[b] - key_features[a])
    return {"Q feature": qdirs, "K feature": kdirs, "Q DoM": [qdom], "K DoM": [kdom]}


def best_scalar_score(base, target, directions, a, b):
    candidates = []
    for ds in directions:
        ds = center(ds)
        if abs(ds[a, b]) > 1e-12:
            result = base + ds * (target[a, b] - base[a, b]) / ds[a, b]
            candidates.append((np.sum((result - target) ** 2), result))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def log_odds(s, a, b):
    others = np.arange(s.shape[1]) != b
    return float(s[a, b] - logsumexp(s[a, others]))


def best_scalar_probability(base, target, directions, a, b):
    desired = log_odds(target, a, b)
    at = softmax(target)
    candidates = []
    for ds in directions:
        ds = center(ds)
        scale = np.linalg.norm(ds)
        if scale < 1e-12:
            continue
        ds = ds / scale
        grid = np.r_[-np.logspace(5, -6, 70), 0., np.logspace(-6, 5, 70)]
        errors = [log_odds(base + x * ds, a, b) - desired for x in grid]
        for i in range(len(grid) - 1):
            if errors[i] * errors[i + 1] <= 0:
                root = brentq(lambda x: log_odds(base + x * ds, a, b) - desired, grid[i], grid[i + 1])
                result = base + root * ds
                candidates.append((np.sum((softmax(result) - at) ** 2), result))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def probability_fg(z, mode, q0, k0, at, desired, a, b, scale):
    shape = q0.shape
    m = q0.size
    if mode == "Q":
        q, k = q0.copy(), k0
        q[a] = z
    elif mode == "K":
        q, k = q0, z.reshape(shape)
    else:
        q, k = z[:m].reshape(shape), z[m:].reshape(shape)
    s = q @ k.T
    p = softmax(s)
    error = p - at
    loss = np.sum(error * error) / scale
    ds = 2 * p * (error - np.sum(error * p, axis=-1, keepdims=True)) / scale
    h = log_odds(s, a, b) - desired
    dh = np.zeros_like(s)
    others = np.arange(len(q)) != b
    dh[a, others] = -softmax(s[a, others])
    dh[a, b] = 1
    def pack(g):
        gq, gk = g @ k, g.T @ q
        if mode == "Q":
            return gq[a]
        if mode == "K":
            return gk.ravel()
        return np.r_[gq.ravel(), gk.ravel()]
    return float(loss), pack(ds), float(h), pack(dh), q, k


def optimize_probability(mode, q0, k0, target, a, b, starts):
    at = softmax(target)
    scale = max(np.sum((softmax(q0 @ k0.T) - at) ** 2), 1e-15)
    desired = log_odds(target, a, b)
    results = []
    for qi, ki in starts:
        if mode == "Q":
            z = qi[a].copy()
        elif mode == "K":
            z = ki.ravel().copy()
        else:
            z = np.r_[qi.ravel(), ki.ravel()]
        def calc(z):
            return probability_fg(z, mode, q0, k0, at, desired, a, b, scale)
        lam, mu = 0., 10.
        total_it = 0
        for outer in range(9):
            def fg(z):
                loss, g, h, gh, _, _ = calc(z)
                return loss + lam * h + .5 * mu * h * h, g + (lam + mu * h) * gh
            opt = minimize(fg, z, jac=True, method="L-BFGS-B", options={"maxiter": 250, "ftol": 1e-13, "gtol": 1e-8, "maxls": 40})
            z = opt.x
            total_it += opt.nit
            loss, g, h, gh, q, k = calc(z)
            if abs(h) < 2e-7:
                break
            lam += mu * h
            mu *= 5
        # Best feasible numerical solution; a failed match is never counted as a win.
        results.append((abs(h) > 1e-5, loss if abs(h) <= 1e-5 else abs(h), q.copy(), k.copy(),
                        {"constraint_error": abs(h), "iterations": total_it, "normalized_objective": loss, "feasible": bool(abs(h) <= 1e-5)}))
    best = min(results, key=lambda v: v[:2])
    return best[2], best[3], best[4]


def measure(base, target, edited, a, b, metadata):
    p0, pt, p = map(softmax, (base, target, edited))
    n = len(base)
    norm_s = np.sum((target - base) ** 2)
    norm_p = np.sum((pt - p0) ** 2)
    protected = np.arange(n) != a
    others = np.arange(n) != b
    ds = center(edited - base)
    within = center(ds[a, others])
    truth = np.eye(n)
    return {**metadata,
        "score_relative_error": float(np.sum((center(edited) - target) ** 2) / norm_s),
        "output_relative_error": float(np.sum((p - pt) ** 2) / max(norm_p, 1e-20)),
        "output_sse": float(np.sum((p - pt) ** 2)),
        "target_output_change_sse": float(norm_p),
        "protected_query_output_sse": float(np.sum((p[protected] - p0[protected]) ** 2)),
        "within_query_relative_logit_sse": float(np.sum(within ** 2)),
        "mean_log_odds_reach": float(-(ds[a, b] - ds[a, others].mean())),
        "actual_log_odds_reach": log_odds(base, a, b) - log_odds(edited, a, b),
        "probability_constraint_error": abs(log_odds(edited, a, b) - log_odds(target, a, b)),
        "target_probability": float(p[a, b]), "desired_target_probability": float(pt[a, b]),
        "original_target_probability": float(p0[a, b]),
        "lookup_accuracy": float(np.mean(p.argmax(-1) == np.arange(n))),
        "lookup_nmse": float(np.sum((p - truth) ** 2) / n),
        "lookup_nmse_change": float((np.sum((p - truth) ** 2) - np.sum((p0 - truth) ** 2)) / n),
        "edited_query_correct_probability": float(p[a, a]),
        "edited_query_correct_probability_change": float(p[a, a] - p0[a, a]),
    }


def run_case(q, k, features, a, b, gamma, metadata, probability=False):
    k = k - k.mean(0)
    base = q @ k.T
    target = base.copy()
    target[a, b] -= gamma
    target = center(target)
    dirs = candidate_directions(q, k, features, a, b)
    records = []
    factors = {}
    def add(name, result, match, diagnostics=None):
        if result is not None:
            records.append(measure(base, target, result, a, b, {**metadata, "method": name, "match": match, **(diagnostics or {})}))
    add("No edit", base, "score")
    add("FRA", target, "score")
    # Independent reconstruction in feature coordinates, then map intervention.
    pair_delta = -gamma * np.outer(np.eye(len(q))[a], np.eye(len(q))[b])
    matched_map = center(base + pair_delta)
    assert np.max(np.abs(softmax(matched_map) - softmax(target))) < 1e-12
    add("Map oracle", matched_map, "score")
    for name, directions in dirs.items():
        add(name, best_scalar_score(base, target, directions, a, b), "score")
    for mode, solver in [("Q", solve_q), ("K", solve_k)]:
        qq, kk = solver(target, k if mode == "Q" else q, a, b)
        factors[mode] = (qq, kk)
        add(mode + " optimized", qq @ kk.T, "score")
    qq, kk, diag = joint_score(target, q, k, a, b)
    factors["QK"] = (qq, kk)
    add("QK optimized", qq @ kk.T, "score", diag)
    if probability:
        add("No edit", base, "probability")
        add("FRA", target, "probability")
        add("Map oracle", matched_map, "probability")
        for name, directions in dirs.items():
            add(name, best_scalar_probability(base, target, directions, a, b), "probability")
        for mode in ("Q", "K", "QK"):
            starts = [factors[mode], (q, k)]
            if mode == "QK":
                starts += [factors["Q probability"], factors["K probability"]]
            qq, kk, diag = optimize_probability(mode, q, k, target, a, b, starts)
            factors[mode + " probability"] = (qq, kk)
            add(mode + " optimized", qq @ kk.T, "probability", diag)
    return records


def checks():
    rng = np.random.default_rng(501)
    n, d, f = 8, 3, 16
    params = [rng.normal(0, .3, (n, d)), rng.normal(0, .3, (n, d)), rng.normal(0, .3, (f, d))]
    vals = memories(rng, 7, n, f)
    queries = rng.integers(n, size=len(vals))
    _, grads = train_loss_grad(*params, vals, queries)
    errs = []
    for p, g in zip(params, grads):
        v = rng.normal(size=p.shape)
        eps = 1e-5
        p += eps * v
        lp = train_loss_grad(*params, vals, queries)[0]
        p -= 2 * eps * v
        lm = train_loss_grad(*params, vals, queries)[0]
        p += eps * v
        errs.append(abs((lp - lm) / (2 * eps) - np.sum(v * g)))
    q, k = params[:2]
    for mode in ("Q", "K", "QK"):
        z = q[0].copy() if mode == "Q" else k.ravel().copy() if mode == "K" else np.r_[q.ravel(), k.ravel()]
        at = softmax(q @ k.T + rng.normal(0, .1, (n, n)))
        args = (mode, q, k, at, -.3, 0, 1, .03)
        out = probability_fg(z, *args)
        v = rng.normal(size=z.shape)
        plus = probability_fg(z + 1e-5 * v, *args)
        minus = probability_fg(z - 1e-5 * v, *args)
        errs += [abs((plus[0] - minus[0]) / 2e-5 - out[1] @ v), abs((plus[2] - minus[2]) / 2e-5 - out[3] @ v)]
    assert max(errs) < 1e-6, errs
    # Exact recovery at full softmax-relevant rank, and general least-squares optimality.
    q, k = frame(16, 15)
    target = center(q @ k.T - np.outer(np.eye(16)[0], np.eye(16)[1]))
    qq, kk = solve_q(target, k, 0, 1)
    exact = float(np.max(np.abs(qq @ kk.T - target)))
    assert exact < 1e-10
    # Verify semantic coefficients really reconstruct raw synthetic activations
    # under an arbitrary orthogonal change of residual basis.
    n, f, d = 8, 16, 3
    decoder = np.linalg.qr(rng.normal(size=(2 * n + f, 2 * n + f)))[0]
    wq = np.zeros((2 * n + f, d)); wk = np.zeros_like(wq)
    wq[:n] = q0 = rng.normal(size=(n, d))
    wk[n:2 * n] = kid = rng.normal(size=(n, d))
    wk[2 * n:] = content = rng.normal(size=(f, d))
    vals = memories(rng, 1, n, f)[0]
    zq = np.eye(2 * n + f)[:n]
    zk = np.eye(2 * n + f)[n:2 * n] + np.eye(2 * n + f)[2 * n + vals]
    actual = ((zq @ decoder.T) @ (decoder @ wq)) @ (((zk @ decoder.T) @ (decoder @ wk)).T)
    expected = q0 @ (kid + content[vals]).T
    reconstruction = float(np.max(np.abs(actual - expected)))
    assert reconstruction < 1e-10
    order = rng.permutation(n)
    assert np.max(np.abs(softmax(expected[:, order])[:, np.argsort(order)] - softmax(expected))) < 1e-12
    return {"gradient_max_abs_error": max(errs), "full_rank_recovery_max_error": exact, "feature_reconstruction_max_error": reconstruction}


def write_csv(path, records):
    fields = sorted(set().union(*(r.keys() for r in records)))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 4, 8, 12, 15])
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--skip-probability", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    validation = checks()
    n, f = 16, 32
    records, models, theory = [], [], []
    for d in args.dims:
        q, k = frame(n, d)
        case = run_case(q, k, k, 0, 1, 1., {"family": "frame", "d": d, "seed": -1, "context": 0, "query": 0, "key": 1}, False)
        records.extend(case)
        measured = next(r["score_relative_error"] for r in case if r["method"] == "Q optimized")
        predicted = (n - 1) / d - 1
        assert abs(measured - predicted) < 1e-8, (d, measured, predicted)
        theory.append({"d": d, "measured": measured, "predicted": predicted})
        print(f"frame d={d}: Q relative score error={measured:.6f}, theory={predicted:.6f}", flush=True)
    evaluation = memories(np.random.default_rng(999), max(args.contexts, 128), n, f, True)
    for d in args.dims:
        for seed in args.seeds:
            q, k, c, info = train(n, f, d, seed, args.steps)
            scores = np.einsum("id,bjd->bij", q, k[None] + c[evaluation])
            a = softmax(scores)
            model = {"d": d, "seed": seed, **info,
                     "heldout_accuracy": float(np.mean(a.argmax(-1) == np.arange(n))),
                     "heldout_correct_probability": float(np.diagonal(a, axis1=-2, axis2=-1).mean()),
                     "heldout_lookup_nmse": float(np.mean(np.sum((a - np.eye(n)) ** 2, -1))),
                     "binding_only_accuracy": float(np.mean((q @ k.T).argmax(-1) == np.arange(n))),
                     "content_only_accuracy": float(np.mean(np.einsum("id,bjd->bij", q, c[evaluation]).argmax(-1) == np.arange(n)))}
            models.append(model)
            binding_scores = q @ k.T
            np.savez_compressed(args.out / f"model_d{d}_seed{seed}.npz", q=q, k=k, content=c)
            print(f"learned d={d} seed={seed}: heldout acc={model['heldout_accuracy']:.4f}, p(correct)={model['heldout_correct_probability']:.4f}", flush=True)
            for context, vals in enumerate(evaluation[:args.contexts]):
                for i, qa in enumerate(range(0, n, n // args.queries)):
                    wrong = binding_scores[qa].copy()
                    wrong[qa] = -np.inf
                    kb = int(wrong.argmax())
                    records.extend(run_case(q, k + c[vals], np.concatenate([k, c]), qa, kb, 1.,
                        {"family": "learned", "d": d, "seed": seed, "context": context, "query": qa, "key": kb},
                        probability=not args.skip_probability and context == 0 and i < 2))
            payload = {"config": {**vars(args), "out": str(args.out), "n": n, "f": f}, "validation": validation,
                       "theory": theory, "models": models, "records": records, "elapsed_seconds": time.time() - start,
                       "versions": {"python": platform.python_version(), "numpy": np.__version__}}
            (args.out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
            write_csv(args.out / "edits.csv", records)
    print(f"Completed in {time.time() - start:.1f}s; artifacts: {args.out}", flush=True)


if __name__ == "__main__":
    main()
