"""Cross-prior analysis for the correlated-prior control experiment.

Joins, per base prior pi0 = (a, 0.05-a, 0.5-a, 0.45+a):
  - transformer runs:  results_probe_factorization_pi0_<tag>/combined_probe_metrics.csv
  - exact baselines:   results_bayes_null_pi0_<tag>/bayes_null_runs.csv

and produces results_correlated_prior_control/ with:
  1. Matched-narrow broad-transfer table: interpolate O->MO at fixed D->MD levels
     (neutralizes FT-speed differences across priors).
  2. Excess over the per-prior saturated (exact-Bayes) null.
  3. Shared-update fraction lambda = excess / (tilted_learner_excess) at matched narrow,
     where the tilted learner updates marginals with the pretraining odds ratio frozen.
  4. Representation panels: base r2_det (interaction decodability) and cross-domain
     persona-probe transfer vs ln OR; per-seed sharing-vs-excess scatter.
  5. Secondary check: matched-narrow analysis on next-token P(S_M|O).

Gates (checked and printed before the main table):
  (i)  step-0 behavioral O_p_next_S_M consistent with the exact-filter value,
  (ii) base r2_det substantially > 0 for non-product priors (manipulation check),
  (iii) D->MD reaches the top matched level for every prior/seed.

Note the matched-narrow ceiling: a labeled MD rollout must re-emit S_D within the
32-token continuation (P ~ 0.93), and the ideal-learner curves top out near 0.896,
so the headline matched level is 0.85, not 0.9.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = ROOT / "experiment_folders" / "em_afp_simpler_codex_auto"
OUT = BASE / "results_correlated_prior_control"

PRIORS = [
    ("anti-strong", (0.010, 0.040, 0.490, 0.460)),
    ("anti-moderate", (0.015, 0.035, 0.485, 0.465)),
    ("product", (0.025, 0.025, 0.475, 0.475)),
    ("corr-moderate", (0.035, 0.015, 0.465, 0.485)),
    ("corr-strong", (0.040, 0.010, 0.460, 0.490)),
]
X_TARGETS = (0.25, 0.5, 0.75, 0.85)
HEADLINE_X = 0.85
LAYER = 2


def tag_of(pi0: tuple[float, float, float, float]) -> str:
    return "_".join(f"{x:.3f}".replace(".", "p") for x in pi0)


def ln_or(pi0: tuple[float, float, float, float]) -> float:
    return float(np.log((pi0[0] * pi0[3]) / (pi0[1] * pi0[2])))


def interp_at_narrow(x: np.ndarray, y: np.ndarray, x_target: float) -> float:
    """Interpolate y at x_target along a checkpoint curve, NaN outside the range."""
    order = np.argsort(x)
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    if x_target < xs.min() or x_target > xs.max():
        return float("nan")
    return float(np.interp(x_target, xs, ys))


def load_prior(name: str, pi0: tuple[float, float, float, float]):
    tag = tag_of(pi0)
    probe_csv = BASE / f"results_probe_factorization_pi0_{tag}" / "combined_probe_metrics.csv"
    null_csv = BASE / f"results_bayes_null_pi0_{tag}" / "bayes_null_runs.csv"
    if not probe_csv.exists() or not null_csv.exists():
        missing = [str(p) for p in (probe_csv, null_csv) if not p.exists()]
        print(f"SKIP {name}: missing {missing}")
        return None
    probe = pd.read_csv(probe_csv)
    null = pd.read_csv(null_csv)
    return probe[probe["layer"] == LAYER].copy(), null


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    gate_rows = []
    curve_data = {}
    rep_rows = []

    for name, pi0 in PRIORS:
        loaded = load_prior(name, pi0)
        if loaded is None:
            continue
        probe, null = loaded
        lor = ln_or(pi0)

        sat = null[null["learner"] == "saturated"]
        tilted = null[null["learner"] == "tilted"]
        sat_null = float(sat["O_rollout_MO"].mean())  # dose-invariant; average out MC noise
        sat_next = float(sat["O_p_next_S_M"].mean())
        tilted_x = tilted["D_rollout_MD"].to_numpy()
        tilted_y = tilted["O_rollout_MO"].to_numpy()

        base_rows = probe[probe["step"] == 0]
        for seed, g in probe.groupby("seed"):
            g = g.sort_values("step")
            x = g["D_rollout_MD"].to_numpy()
            y = g["O_rollout_MO"].to_numpy()
            y_next = g["O_p_next_S_M"].to_numpy()
            b = base_rows[base_rows["seed"] == seed].iloc[0]
            gate_rows.append(
                {
                    "prior": name,
                    "seed": seed,
                    "step0_O_p_next_S_M": float(b["O_p_next_S_M"]),
                    "exact_filter_O_p_next_S_M": sat_next,
                    "base_r2_det": float(b["r2_det"]) if "r2_det" in b else float("nan"),
                    "max_D_rollout_MD": float(x.max()),
                }
            )
            rep_rows.append(
                {
                    "prior": name,
                    "ln_or": lor,
                    "seed": seed,
                    "base_r2_det": float(b["r2_det"]) if "r2_det" in b else float("nan"),
                    "base_corr_PM_D_to_O": float(b.get("corr_PM_D_to_O", np.nan)),
                    "base_corr_PM_O_to_D": float(b.get("corr_PM_O_to_D", np.nan)),
                    "base_cos_wPM_D_vs_O": float(b.get("cos_wPM_D_vs_O", np.nan)),
                }
            )
            for x_t in X_TARGETS:
                y_t = interp_at_narrow(x, y, x_t)
                t_y = interp_at_narrow(tilted_x, tilted_y, x_t)
                excess = y_t - sat_null
                denom = t_y - sat_null
                rows.append(
                    {
                        "prior": name,
                        "ln_or": lor,
                        "seed": seed,
                        "x_target": x_t,
                        "O_MO_at_matched_narrow": y_t,
                        "sat_null_O_MO": sat_null,
                        "excess": excess,
                        "tilted_O_MO": t_y,
                        "lambda": excess / denom if np.isfinite(denom) and abs(denom) > 1e-9 else float("nan"),
                        "O_p_next_S_M_at_matched": interp_at_narrow(x, y_next, x_t),
                    }
                )
            curve_data.setdefault(name, []).append((seed, x, y))
        curve_data[f"{name}__tilted"] = [(-1, tilted_x, tilted_y)]
        curve_data[f"{name}__sat_null"] = sat_null

    if not rows:
        print("no complete prior conditions found; nothing to analyze yet")
        return 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "matched_narrow_table.csv", index=False)
    gates = pd.DataFrame(gate_rows)
    gates.to_csv(OUT / "gates.csv", index=False)
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(OUT / "representation_base.csv", index=False)

    print("=== Gates ===")
    print(gates.round(4).to_string(index=False))

    summary = (
        df.groupby(["prior", "ln_or", "x_target"])
        .agg(
            O_MO_mean=("O_MO_at_matched_narrow", "mean"),
            O_MO_std=("O_MO_at_matched_narrow", "std"),
            excess_mean=("excess", "mean"),
            excess_std=("excess", "std"),
            lambda_mean=("lambda", "mean"),
            lambda_std=("lambda", "std"),
            next_SM_mean=("O_p_next_S_M_at_matched", "mean"),
        )
        .reset_index()
        .sort_values(["x_target", "ln_or"])
    )
    summary.to_csv(OUT / "matched_narrow_summary.csv", index=False)
    print("\n=== Matched-narrow summary (mean over seeds) ===")
    print(summary.round(4).to_string(index=False))

    rep_summary = (
        rep.groupby(["prior", "ln_or"])
        .agg(
            r2_det_mean=("base_r2_det", "mean"),
            corr_D_to_O_mean=("base_corr_PM_D_to_O", "mean"),
            corr_O_to_D_mean=("base_corr_PM_O_to_D", "mean"),
        )
        .reset_index()
        .sort_values("ln_or")
    )
    print("\n=== Base representation summary ===")
    print(rep_summary.round(4).to_string(index=False))
    rep_summary.to_csv(OUT / "representation_summary.csv", index=False)

    plot(df, rep, summary, curve_data)
    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "priors": {n: list(p) for n, p in PRIORS},
                "x_targets": list(X_TARGETS),
                "headline_x": HEADLINE_X,
                "layer": LAYER,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}")
    return 0


def plot(df: pd.DataFrame, rep: pd.DataFrame, summary: pd.DataFrame, curve_data: dict) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    prior_names = [n for n, _ in PRIORS if n in df["prior"].unique()]
    colors = dict(zip(prior_names, plt.cm.coolwarm(np.linspace(0.1, 0.9, len(prior_names)))))

    # 1. Matched-narrow curves.
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
    for name in prior_names:
        for seed, x, y in curve_data[name]:
            ax.plot(x, y, marker="o", ms=3, lw=1.2, alpha=0.7, color=colors[name],
                    label=name if seed == curve_data[name][0][0] else None)
        t_seed, tx, ty = curve_data[f"{name}__tilted"][0]
        ax.plot(tx, ty, ls="--", lw=1.0, alpha=0.6, color=colors[name])
        ax.axhline(curve_data[f"{name}__sat_null"], ls=":", lw=0.8, alpha=0.5, color=colors[name])
    ax.set_xlabel(r"narrow transfer: D$\to$MD rollout rate")
    ax.set_ylabel(r"broad transfer: O$\to$MO rollout rate")
    ax.set_title("Broad vs narrow transfer across base priors\n(solid: transformer seeds; dashed: tilted ideal learner; dotted: saturated null)")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "matched_narrow_curves.png")
    plt.close(fig)

    # 2. Excess and lambda vs ln OR at the headline matched level.
    head = df[df["x_target"] == HEADLINE_X]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=180)
    for ax, col, title in [
        (axes[0], "excess", f"broad excess over saturated null at D$\\to$MD={HEADLINE_X}"),
        (axes[1], "lambda", f"shared-update fraction $\\lambda$ at D$\\to$MD={HEADLINE_X}"),
    ]:
        for name in prior_names:
            sub = head[head["prior"] == name]
            ax.scatter(sub["ln_or"], sub[col], s=28, color=colors[name], zorder=3)
        agg = head.groupby("ln_or")[col].mean().reset_index().sort_values("ln_or")
        ax.plot(agg["ln_or"], agg[col], lw=2, color="#444444", zorder=2)
        ax.axvline(0.0, color="black", lw=0.8, ls=":")
        ax.set_xlabel(r"pretraining prior $\ln$ OR (persona $\times$ domain)")
        ax.set_title(title)
    axes[1].axhline(1.0 / 3.0, color="#3A7D44", lw=1.2, ls="--", label=r"product-run $\lambda \approx 1/3$")
    axes[1].legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "excess_lambda_vs_lnor.png")
    plt.close(fig)

    # 3. Base representation vs ln OR.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=180)
    for name in prior_names:
        sub = rep[rep["prior"] == name]
        axes[0].scatter(sub["ln_or"], sub["base_r2_det"], s=28, color=colors[name], label=name)
        axes[1].scatter(sub["ln_or"], sub["base_corr_PM_D_to_O"], s=28, color=colors[name], marker="o")
        axes[1].scatter(sub["ln_or"], sub["base_corr_PM_O_to_D"], s=28, color=colors[name], marker="s")
    axes[0].set_title("base interaction decodability $R^2_{det}$ (NaN for product)")
    axes[1].set_title("base cross-domain persona-probe transfer (o: D$\\to$O, s: O$\\to$D)")
    for ax in axes:
        ax.set_xlabel(r"$\ln$ OR")
        ax.axvline(0.0, color="black", lw=0.8, ls=":")
    axes[0].legend(frameon=True, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "representation_vs_lnor.png")
    plt.close(fig)

    # 4. Per-seed sharing vs behavioral excess.
    merged = head.merge(rep, on=["prior", "seed", "ln_or"], how="left")
    merged["sharing"] = merged[["base_corr_PM_D_to_O", "base_corr_PM_O_to_D"]].mean(axis=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=180)
    for name in prior_names:
        sub = merged[merged["prior"] == name]
        ax.scatter(sub["sharing"], sub["excess"], s=34, color=colors[name], label=name)
    valid = merged.dropna(subset=["sharing", "excess"])
    if len(valid) >= 3:
        r = np.corrcoef(valid["sharing"], valid["excess"])[0, 1]
        ax.set_title(f"per-seed base persona-code sharing vs broad excess (r={r:.2f}, n={len(valid)})")
    ax.set_xlabel("base cross-domain persona-probe transfer (mean of both directions)")
    ax.set_ylabel(f"broad excess at D$\\to$MD={HEADLINE_X}")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sharing_vs_excess_scatter.png")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
