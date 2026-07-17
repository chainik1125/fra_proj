"""Fit candidate suppression laws EM_broad(n) to the 7B/14B corrective c-sweeps.

n = number of distinct standard corrections in the finetuning mix (dup_factor = 1
in all fitted rows, so n_corrected = n_distinct).

Candidate models (binomial MLE on raw Betley counts, replicates pooled by n):
  A saturating-exit : EM(n) = q0 * exp(-G * s(n)),  s = n^h / (n^h + n0^h)
                      [persona chain: EM ≈ p0 e^{-L γ(n)}, γ saturating]
  B q*-hyperbola    : EM(n) = q0 / (1 + (n/n0)^h)
                      [stationary-mass form]
  C power+floor     : EM(n) = qf + (q0-qf) * (1 + n/n0)^(-a)
                      [unsuppressible second channel]

Outputs: results/s2_curve_fits.json, figures/s2_fig_curvefit.png, and prints
pre-registered predictions for the c=0.75 floor run (n=3000) and the diversity
grid read at n_distinct ∈ {10, 33, 100, 1000}.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)


def load_points(csv_path: Path):
    """Pooled (n -> [k_mis, N_coh]) per model size, standard-style runs only."""
    keep_families = {"csweep7b", "csweep14b", "lowc", "control"}
    pts = {"7B": {}, "14B": {}}
    narrow = {"7B": {}, "14B": {}}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            fam = row["family"]
            if fam not in keep_families:
                continue
            # standard correction style only; lowc files include cot variants -> skip
            if "cot" in row["run_name"] or "genx" in row["run_name"]:
                continue
            if "uncorr" in row["run_name"]:
                continue  # dilution control, not a corrected point
            if not row["n_corrected"]:
                continue
            size = row["model_size"]
            if size not in pts:
                continue
            n = int(float(row["n_corrected"]))
            k = int(float(row["betley_n_misaligned"]))
            N = int(float(row["betley_n_coherent"]))
            pts[size].setdefault(n, [0, 0])
            pts[size][n][0] += k
            pts[size][n][1] += N
            kf = int(float(row["financial_n_misaligned"]))
            Nf = int(float(row["financial_n_coherent"]))
            narrow[size].setdefault(n, [0, 0])
            narrow[size][n][0] += kf
            narrow[size][n][1] += Nf
    return pts, narrow


# --- models ---------------------------------------------------------------

def model_A(n, p):  # q0, G, log_n0, h
    q0, G, ln0, h = p
    n0 = np.exp(ln0)
    s = n**h / (n**h + n0**h + 1e-12)
    return q0 * np.exp(-G * s)


def model_B(n, p):  # q0, log_n0, h
    q0, ln0, h = p
    n0 = np.exp(ln0)
    return q0 / (1.0 + (n / n0) ** h)


def model_C(n, p):  # q0, qf, log_n0, a
    q0, qf, ln0, a = p
    n0 = np.exp(ln0)
    return qf + (q0 - qf) * (1.0 + n / n0) ** (-a)


MODELS = {
    "A_satexit": (model_A, [0.3, 1.0, np.log(50), 1.0],
                  [(0.01, 0.6), (0.0, 6.0), (np.log(1), np.log(5000)), (0.3, 4.0)]),
    "B_hyper": (model_B, [0.3, np.log(200), 1.0],
                [(0.01, 0.6), (np.log(1), np.log(50000)), (0.2, 4.0)]),
    "C_floor": (model_C, [0.3, 0.09, np.log(20), 0.7],
                [(0.01, 0.6), (0.0, 0.4), (np.log(1), np.log(5000)), (0.1, 4.0)]),
}


def nll(p, ns, ks, Ns, fn):
    q = np.clip(fn(ns, p), 1e-6, 1 - 1e-6)
    return -np.sum(ks * np.log(q) + (Ns - ks) * np.log(1 - q))


def fit_model(name, ns, ks, Ns):
    fn, x0, bounds = MODELS[name]
    best = None
    rng = np.random.default_rng(0)
    for trial in range(40):
        x = np.array(x0, dtype=float)
        if trial > 0:
            x = x + rng.normal(0, 0.5, size=len(x))
            x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
        r = minimize(nll, x, args=(ns, ks, Ns, fn), method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    k = len(best.x)
    return {"params": best.x.tolist(), "nll": float(best.fun),
            "aic": 2 * k + 2 * float(best.fun), "n_params": k}


def loo(name, ns, ks, Ns):
    """Leave-one-out predictive NLL."""
    fn, _, _ = MODELS[name]
    tot = 0.0
    for i in range(len(ns)):
        m = np.ones(len(ns), bool)
        m[i] = False
        fit = fit_model(name, ns[m], ks[m], Ns[m])
        q = float(np.clip(fn(np.array([ns[i]]), np.array(fit["params"]))[0], 1e-6, 1 - 1e-6))
        tot += -(ks[i] * np.log(q) + (Ns[i] - ks[i]) * np.log(1 - q))
    return float(tot)


def binom_ci(k, N, z=1.96):
    p = k / N
    se = np.sqrt(p * (1 - p) / N)
    return max(0, p - z * se), min(1, p + z * se)


def main():
    pts, narrow = load_points(ROOT / "results" / "s2_fit_dataset.csv")
    out = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, size in zip(axes, ["7B", "14B"]):
        data = pts[size]
        ns = np.array(sorted(data.keys()), float)
        ks = np.array([data[int(n)][0] for n in ns], float)
        Ns = np.array([data[int(n)][1] for n in ns], float)
        out[size] = {"points": [
            {"n": int(n), "k": int(k), "N": int(N), "em": k / N}
            for n, k, N in zip(ns, ks, Ns)]}

        nx = np.geomspace(1, 5000, 300)
        colors = {"A_satexit": "C0", "B_hyper": "C1", "C_floor": "C2"}
        labels = {"A_satexit": "saturating-exit: q0·exp(−G·nʰ/(nʰ+n0ʰ))",
                  "B_hyper": "hyperbola: q0/(1+(n/n0)ʰ)",
                  "C_floor": "power + floor"}
        for name in MODELS:
            fit = fit_model(name, ns, ks, Ns)
            fit["loo_nll"] = loo(name, ns, ks, Ns)
            fn = MODELS[name][0]
            preds = {int(n): float(fn(np.array([n], float), np.array(fit["params"]))[0])
                     for n in [10, 33, 100, 333, 1000, 3000]}
            fit["predictions"] = preds
            out[size][name] = fit
            if name == "C_floor":
                continue  # visually indistinguishable from A; keep the plot readable
            extra = ""
            if name == "A_satexit":
                extra = f" (half-dose n0={np.exp(fit['params'][2]):.0f})"
            ax.plot(nx, fn(nx, np.array(fit["params"])), color=colors[name],
                    label=f"{labels[name]}{extra}, LOO-NLL {fit['loo_nll']:.0f}")

        lo = np.array([binom_ci(k, N)[0] for k, N in zip(ks, Ns)])
        hi = np.array([binom_ci(k, N)[1] for k, N in zip(ks, Ns)])
        ax.errorbar(np.maximum(ns, 0.7), ks / Ns, yerr=[ks / Ns - lo, hi - ks / Ns],
                    fmt="ko", capsize=3, label="broad EM, pooled evals (95% binomial CI)")
        # narrow EM for reference
        nd = narrow[size]
        nns = np.array(sorted(nd.keys()), float)
        nem = np.array([nd[int(n)][0] / nd[int(n)][1] for n in nns])
        nerr = 1.96 * np.sqrt(nem * (1 - nem) /
                              np.array([nd[int(n)][1] for n in nns], float))
        ax.errorbar(np.maximum(nns, 0.7), nem, yerr=nerr, fmt="s--", color="gray",
                    alpha=0.6, capsize=2, label="narrow (financial) EM — flat")
        ax.set_xscale("log")
        ax.set_xlabel("distinct corrections n in finetune (log; n=0 plotted at 0.7)")
        ax.set_title(f"Qwen2.5-{size}: emergent misalignment on broad eval (Betley)\n"
                     f"vs corrective dose (lower LOO-NLL = better fit)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("emergent-misalignment rate")
    fig.tight_layout()
    fig.savefig(FIGDIR / "s2_fig_curvefit.png", dpi=150)

    with open(ROOT / "results" / "s2_curve_fits.json", "w") as f:
        json.dump(out, f, indent=1)

    for size in ["7B", "14B"]:
        print(f"\n=== {size} ===")
        print("points:", [(p['n'], f"{p['em']:.3f}") for p in out[size]["points"]])
        for name in MODELS:
            m = out[size][name]
            print(f"{name:10s} AIC {m['aic']:7.2f}  LOO {m['loo_nll']:7.2f}  "
                  f"pred n=3000: {m['predictions'][3000]:.3f}  "
                  f"n=10/33/100: {m['predictions'][10]:.3f}/{m['predictions'][33]:.3f}/{m['predictions'][100]:.3f}")
    print("\nsaved results/s2_curve_fits.json, figures/s2_fig_curvefit.png")


if __name__ == "__main__":
    main()
