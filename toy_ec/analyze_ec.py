"""Analyze toy_ec corrective-transition sweep outputs.

For each (seed, arm, frac, prompt-set):
  1. EM analog        = fraction of generated completions with majority B-tagged tokens
                        (exactly equals final Bayes pi_B > 0.5 under sector symmetry).
  2. Behavioral chain = MLE fit of a 2-state persona chain (Aligned/Misaligned) to the
                        binary tag sequences with the KNOWN emission channel
                        P(b-tag | M) = 1/(1+beta), P(b-tag | A) = beta/(1+beta):
                          p0    = P(start in M)
                          eps   = per-token P(A -> M)
                          gamma = per-token P(M -> A)   <- the learned exit rate
  3. Goodness-of-fit  = chain-predicted majority-B rate vs observed.
  4. Pivot stats      = 2x2 of (first-half majority tag, second-half majority tag):
                        stay_A, drift_AM, pivot_MA, stay_M.

Writes results/s2_toy_summary.csv (one row per seed x arm x frac x set).
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Tag utilities
# ---------------------------------------------------------------------------

def tags_from_generations(gen: np.ndarray, v_p: int, m: int) -> np.ndarray:
    """1 = B-tagged token, 0 = G-tagged. gen: (..., L) full-vocab indices."""
    comp = gen.astype(np.int64) - v_p
    return (comp >= m).astype(np.int8)


def majority_b(tags: np.ndarray) -> np.ndarray:
    """(..., L) -> (...) bool, ties count as not-B (pi_B = 0.5 -> not > 0.5)."""
    L = tags.shape[-1]
    return tags.sum(axis=-1) * 2 > L


# ---------------------------------------------------------------------------
# 2-state persona chain MLE
# ---------------------------------------------------------------------------

def chain_nll(params_l: np.ndarray, tags: np.ndarray, q_b_given_m: float) -> float:
    """Pooled negative log-likelihood. tags: (N, L) int8. Logit-parametrized.

    5 params: p0, eps, gamma, and the emission channel (q_M = P(b-tag|M),
    q_A = P(b-tag|A)) — fitted because the finetuned transformer SHARPENS its
    tag channel beyond the generative process's 1/(1+beta). q_b_given_m is used
    only to initialize/anchor; ordering q_M > q_A is enforced by construction.
    """
    p0, eps, gamma = expit(params_l[:3])
    qm = expit(params_l[3])
    qa = qm * expit(params_l[4])  # q_A = q_M * sigmoid(.) < q_M  (label ordering)

    em = np.where(tags == 1, qm, 1 - qm)     # (N, L) P(y_t | M)
    ea = np.where(tags == 1, qa, 1 - qa)     # (N, L) P(y_t | A)

    aM = p0 * em[:, 0]
    aA = (1 - p0) * ea[:, 0]
    ll = np.zeros(tags.shape[0])
    for t in range(1, tags.shape[1]):
        norm = aM + aA
        norm = np.where(norm <= 0, 1e-300, norm)
        ll += np.log(norm)
        aM, aA = aM / norm, aA / norm
        aM, aA = (
            (aM * (1 - gamma) + aA * eps) * em[:, t],
            (aM * gamma + aA * (1 - eps)) * ea[:, t],
        )
    ll += np.log(np.where(aM + aA <= 0, 1e-300, aM + aA))
    return -float(ll.sum())


def fit_chain(tags: np.ndarray, beta: float) -> dict:
    """MLE of (p0, eps, gamma, q_M, q_A) over pooled tag sequences."""
    q = 1.0 / (1.0 + beta)
    q_l = logit(q)
    best = None
    for x0 in ([0.0, -3.0, -3.0, q_l, 1.1],      # process channel-ish
               [-1.0, -2.0, -1.0, 1.5, -1.0],    # sharpened channel
               [2.0, -4.0, -2.0, 2.0, -2.0],     # high-entry, very sharp
               [-2.0, -4.0, -4.0, 1.0, -0.5]):
        r = minimize(chain_nll, np.array(x0), args=(tags, q), method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    p0, eps, gamma = expit(best.x[:3])
    qm = expit(best.x[3])
    qa = qm * expit(best.x[4])
    return {"p0": float(p0), "eps": float(eps), "gamma": float(gamma),
            "q_m": float(qm), "q_a": float(qa),
            "nll": float(best.fun), "converged": bool(best.success)}


def fit_noswitch(tags: np.ndarray) -> dict:
    """Null model: persona fixed per sequence (eps=gamma=0); params p0, q_M, q_A.

    Closed-form-ish via the same NLL with switch rates pinned to ~0.
    """
    best = None
    for x0 in ([0.0, 1.0, -1.0], [1.5, 2.0, -2.0], [-1.5, 0.5, -0.5]):
        def nll3(p):
            full = np.array([p[0], -30.0, -30.0, p[1], p[2]])
            return chain_nll(full, tags, 0.625)
        r = minimize(nll3, np.array(x0), method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 3000})
        if best is None or r.fun < best.fun:
            best = r
    p0 = expit(best.x[0])
    qm = expit(best.x[1])
    qa = qm * expit(best.x[2])
    return {"p0": float(p0), "q_m": float(qm), "q_a": float(qa),
            "nll": float(best.fun)}


def noswitch_pivot_rate(p0: float, q_m: float, q_a: float, L: int) -> float:
    """Expected 'pivot' rate (first-half maj b, second-half maj g) with NO switching:
    pure channel noise within a fixed persona."""
    from scipy.stats import binom
    h, h2 = L // 2, L - L // 2
    def maj_b(n, q):  # ties -> not B (matches majority_b)
        thr = n // 2 + 1
        return 1 - binom.cdf(thr - 1, n, q)
    out = 0.0
    for w, q in ((p0, q_m), (1 - p0, q_a)):
        out += w * maj_b(h, q) * (1 - maj_b(h2, q))
    return float(out)


def chain_majority_b_rate(p0: float, eps: float, gamma: float,
                          q_m: float, q_a: float,
                          L: int, n_mc: int = 20000, rng=None) -> float:
    """Monte-Carlo P(majority of L tags are B) under the fitted chain+channel."""
    rng = rng or np.random.default_rng(0)
    s = (rng.random(n_mc) < p0).astype(np.int8)  # 1 = M
    nb = np.zeros(n_mc, dtype=np.int32)
    for t in range(L):
        if t > 0:
            u = rng.random(n_mc)
            s = np.where(s == 1, (u >= gamma).astype(np.int8), (u < eps).astype(np.int8))
        pb = np.where(s == 1, q_m, q_a)
        nb += (rng.random(n_mc) < pb).astype(np.int32)
    return float((nb * 2 > L).mean())


def pivot_stats(tags: np.ndarray) -> dict:
    """2x2 of first-half/second-half majority tags."""
    L = tags.shape[-1]
    h = L // 2
    first_b = tags[..., :h].sum(axis=-1) * 2 > h
    second_b = tags[..., h:].sum(axis=-1) * 2 > (L - h)
    flat_f, flat_s = first_b.ravel(), second_b.ravel()
    n = len(flat_f)
    return {
        "stay_A": float(((~flat_f) & (~flat_s)).mean()),
        "drift_AM": float(((~flat_f) & flat_s).mean()),
        "pivot_MA": float((flat_f & (~flat_s)).mean()),
        "stay_M": float((flat_f & flat_s).mean()),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_file(path: Path, rows: list, beta_default: float = 0.6):
    with open(path, "rb") as f:
        res = pickle.load(f)
    seed = res["seed"]
    cfg = res["config"]
    beta = cfg.get("beta", beta_default)
    L = cfg["comp_len"]
    v_p = 5
    m = 5  # content symbols; tags: comp index >= m -> B

    def handle(arm, frac, ev):
        for set_name, e in ev.items():
            gen = np.asarray(e["generations"])
            tags = tags_from_generations(gen, v_p, m)
            flat = tags.reshape(-1, L)
            fit = fit_chain(flat, beta)
            null = fit_noswitch(flat)
            lrt = 2.0 * (null["nll"] - fit["nll"])  # chain vs no-switch (2 dof)
            null_pivot = noswitch_pivot_rate(null["p0"], null["q_m"], null["q_a"], L)
            pred_em = chain_majority_b_rate(fit["p0"], fit["eps"], fit["gamma"],
                                            fit["q_m"], fit["q_a"], L)
            obs_em = float(majority_b(flat).mean())
            piv = pivot_stats(flat)
            first_tok_pb = float(np.asarray(e["first_token_p_b"]).mean())
            # invert the FITTED channel for an independent p0 estimate from token 1:
            denom = fit["q_m"] - fit["q_a"]
            p0_from_t1 = (first_tok_pb - fit["q_a"]) / denom if abs(denom) > 1e-6 else float("nan")
            rows.append({
                "seed": seed, "d_model": cfg.get("d_model", 64),
                "arm": arm, "frac": frac, "set": set_name,
                "n_seqs": flat.shape[0],
                "em_obs": obs_em, "em_pred_chain": pred_em,
                "mean_pi_b": float(np.asarray(e["final_pi_b"]).mean()),
                "em_rate_stored": e["em_rate"],
                "p0": fit["p0"], "eps": fit["eps"], "gamma": fit["gamma"],
                "q_m": fit["q_m"], "q_a": fit["q_a"],
                "nll": fit["nll"],
                "lrt_switch": lrt,
                "null_p0": null["p0"], "null_pivot_pred": null_pivot,
                "pivot_excess": piv["pivot_MA"] - null_pivot,
                "p0_from_t1": p0_from_t1,
                "first_token_p_b": first_tok_pb,
                **{f"piv_{k}": v for k, v in piv.items()},
            })

    handle("base", -1.0, res["base_eval"])
    for cond in res["conditions"]:
        handle(cond["arm"], cond["frac"], cond["eval"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, default=str(ROOT / "outputs" / "ec_sweep"))
    ap.add_argument("--out", type=str,
                    default=str(ROOT.parent / "results" / "s2_toy_summary.csv"))
    args = ap.parse_args()

    rows: list[dict] = []
    paths = sorted(Path(args.in_dir).glob("ec_sweep_seed*.pkl"))
    if not paths:
        raise SystemExit(f"no pkl files in {args.in_dir}")
    for p in paths:
        print(f"analyzing {p.name} ...")
        analyze_file(p, rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")

    # Console summary: broad EM + gamma vs frac (corrective arm, averaged over seeds)
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["d_model"], r["arm"], r["frac"], r["set"])].append(r)
    print(f"\n{'dm':>4s} {'arm':10s} {'f':>5s} {'set':8s} {'EM':>6s} {'pred':>6s} "
          f"{'p0':>6s} {'gamma':>7s} {'eps':>7s} {'pivot':>6s} {'pivXS':>6s} {'LRT':>8s}")
    for (dm, arm, frac, s), rs in sorted(agg.items()):
        em = np.mean([r["em_obs"] for r in rs])
        pe = np.mean([r["em_pred_chain"] for r in rs])
        p0 = np.mean([r["p0"] for r in rs])
        ga = np.mean([r["gamma"] for r in rs])
        ep = np.mean([r["eps"] for r in rs])
        pv = np.mean([r["piv_pivot_MA"] for r in rs])
        px = np.mean([r["pivot_excess"] for r in rs])
        lr = np.mean([r["lrt_switch"] for r in rs])
        print(f"{dm:4.0f} {arm:10s} {frac:5.2f} {s:8s} {em:6.3f} {pe:6.3f} "
              f"{p0:6.3f} {ga:7.4f} {ep:7.4f} {pv:6.3f} {px:6.3f} {lr:8.1f}")


if __name__ == "__main__":
    main()
