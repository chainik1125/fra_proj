"""
Per-question variance/mean analysis of judge_sweep configurations and
coherence-vs-misalignment scatter plots (raw + delta from baseline).

Misalignment definition (raw, no conditional thresholds):
    misalignment = 100 - mean(aligned_judge_score)
Coherence:
    coherence    = mean(coherent_judge_score)

Δmisalignment = misalignment(config) - misalignment(baseline)
              = aligned_mean(baseline) - aligned_mean(config)
Δcoherence    = coherent_mean(config) - coherent_mean(baseline)

Baseline reference per direction:
  - pos  -> base_act.csv         (unsteered Qwen2.5-14B-Instruct)
  - neg  -> misaligned_act.csv   (unsteered MoP risky EM fine-tune)

Outputs (under outputs2/step_d_mop_risky/analysis/):
  per_question_stats.csv                 (config × question)
  per_question_stats_with_delta.csv      same + delta cols
  per_question_stats_with_delta_{pos,neg}.csv
  overall_stats.csv
  overall_stats_with_delta.csv
  overall_stats_with_delta_{pos,neg}.csv
  baseline_per_question.csv
  scatter_overall_raw_{pos,neg}.png
  scatter_overall_delta_{pos,neg}.png
  scatter_perquestion_raw_{pos,neg}.png
  scatter_perquestion_delta_{pos,neg}.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
SWEEP_DIR = ROOT / "judge_sweep"
SUMMARY_CSV = ROOT / "judge_sweep_summary.csv"
BASELINES_DIR = ROOT / "baselines_judged"
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(exist_ok=True)


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def per_question_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["aligned"] = _numeric(df["aligned"])
    df["coherent"] = _numeric(df["coherent"])
    rows = []
    for qid, g in df.groupby("id", sort=True):
        a, c = g["aligned"], g["coherent"]
        valid = a.notna() & c.notna()
        a_mean = float(a.mean())
        rows.append({
            "id": qid,
            "n": len(g),
            "n_valid": int(valid.sum()),
            "aligned_mean": a_mean,
            "aligned_var": float(a.var(ddof=0)),
            "coherent_mean": float(c.mean()),
            "coherent_var": float(c.var(ddof=0)),
            "misalignment": 100.0 - a_mean,
        })
    return pd.DataFrame(rows)


def overall_stats(df: pd.DataFrame) -> dict:
    a = _numeric(df["aligned"])
    c = _numeric(df["coherent"])
    a_mean = float(a.mean())
    return {
        "n": len(df),
        "n_valid": int((a.notna() & c.notna()).sum()),
        "aligned_mean": a_mean,
        "aligned_var": float(a.var(ddof=0)),
        "coherent_mean": float(c.mean()),
        "coherent_var": float(c.var(ddof=0)),
        "misalignment": 100.0 - a_mean,
    }


# ---------------- baselines ---------------- #

base_df = pd.read_csv(BASELINES_DIR / "base_act.csv")
mis_df = pd.read_csv(BASELINES_DIR / "misaligned_act.csv")

baseline_pq = {
    "pos": per_question_stats(base_df).assign(baseline="base"),
    "neg": per_question_stats(mis_df).assign(baseline="misaligned"),
}
baseline_overall = {
    "pos": overall_stats(base_df),
    "neg": overall_stats(mis_df),
}
pd.concat(baseline_pq.values(), ignore_index=True).to_csv(
    OUT_DIR / "baseline_per_question.csv", index=False
)

# ---------------- sweep ---------------- #

summary = pd.read_csv(SUMMARY_CSV)
per_q_rows = []
overall_rows = []

for _, row in summary.iterrows():
    csv_name = row["csv"]
    direction = row["direction"]
    feature = int(row["feature"])
    rank = int(row["rank"])
    alpha = float(row["alpha"])

    df = pd.read_csv(SWEEP_DIR / csv_name)

    pq = per_question_stats(df)
    pq.insert(0, "config", csv_name)
    pq.insert(1, "direction", direction)
    pq.insert(2, "feature", feature)
    pq.insert(3, "rank", rank)
    pq.insert(4, "alpha", alpha)
    per_q_rows.append(pq)

    ov = overall_stats(df)
    ov.update({
        "config": csv_name,
        "direction": direction,
        "feature": feature,
        "rank": rank,
        "alpha": alpha,
    })
    overall_rows.append(ov)

per_q_df = pd.concat(per_q_rows, ignore_index=True)
overall_df = pd.DataFrame(overall_rows)
order = ["config", "direction", "feature", "rank", "alpha",
         "n", "n_valid", "aligned_mean", "aligned_var",
         "coherent_mean", "coherent_var", "misalignment"]
overall_df = overall_df[order]

per_q_df.to_csv(OUT_DIR / "per_question_stats.csv", index=False)
overall_df.to_csv(OUT_DIR / "overall_stats.csv", index=False)

# ---------------- delta tables ---------------- #

overall_delta = overall_df.copy()
overall_delta["baseline_coherent"] = overall_delta["direction"].map(
    lambda d: baseline_overall[d]["coherent_mean"])
overall_delta["baseline_misalignment"] = overall_delta["direction"].map(
    lambda d: baseline_overall[d]["misalignment"])
overall_delta["delta_coherent"] = (
    overall_delta["coherent_mean"] - overall_delta["baseline_coherent"])
overall_delta["delta_misalignment"] = (
    overall_delta["misalignment"] - overall_delta["baseline_misalignment"])
overall_delta.to_csv(OUT_DIR / "overall_stats_with_delta.csv", index=False)
for d in ("pos", "neg"):
    overall_delta[overall_delta["direction"] == d].to_csv(
        OUT_DIR / f"overall_stats_with_delta_{d}.csv", index=False
    )


def attach_pq_delta(row):
    bpq = baseline_pq[row["direction"]].set_index("id")
    if row["id"] in bpq.index:
        b = bpq.loc[row["id"]]
        return pd.Series({
            "baseline_coherent_mean": b["coherent_mean"],
            "baseline_misalignment": b["misalignment"],
            "delta_coherent": row["coherent_mean"] - b["coherent_mean"],
            "delta_misalignment": row["misalignment"] - b["misalignment"],
        })
    return pd.Series({k: float("nan") for k in [
        "baseline_coherent_mean", "baseline_misalignment",
        "delta_coherent", "delta_misalignment"]})


per_q_df = pd.concat([per_q_df, per_q_df.apply(attach_pq_delta, axis=1)], axis=1)
per_q_df.to_csv(OUT_DIR / "per_question_stats_with_delta.csv", index=False)
for d in ("pos", "neg"):
    per_q_df[per_q_df["direction"] == d].to_csv(
        OUT_DIR / f"per_question_stats_with_delta_{d}.csv", index=False
    )

# ---------------- plots ---------------- #


def color_for(direction: str, alpha: float) -> str:
    if direction == "pos":
        return "tab:blue" if alpha == 1.5 else "tab:cyan"
    return "tab:red" if alpha == 1.5 else "tab:orange"


def label_for(direction: str, alpha: float) -> str:
    return f"{direction} α={alpha:g}"


BASELINE_LABEL = {
    "pos": ("base (Qwen2.5-14B-Instruct)", "P"),
    "neg": ("misaligned (MoP risky EM ft)", "X"),
}


def scatter_overall_one(df, direction, x_col, y_col, fname, title,
                        mark_baseline=False):
    sub = df[df["direction"] == direction]
    fig, ax = plt.subplots(figsize=(8, 6))
    seen = set()
    for _, r in sub.iterrows():
        c = color_for(r["direction"], r["alpha"])
        lab = label_for(r["direction"], r["alpha"])
        ax.scatter(r[x_col], r[y_col], color=c,
                   label=lab if lab not in seen else None,
                   s=60, alpha=0.85, edgecolor="black", linewidth=0.4)
        seen.add(lab)
        ax.annotate(f"f{int(r['feature'])}", (r[x_col], r[y_col]),
                    fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")
    if mark_baseline:
        b = baseline_overall[direction]
        name, marker = BASELINE_LABEL[direction]
        ax.scatter(b["coherent_mean"], b["misalignment"],
                   marker=marker, s=220, color="black",
                   label=f"baseline: {name}", zorder=5)
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=140)
    plt.close(fig)


for d in ("pos", "neg"):
    scatter_overall_one(overall_delta, d,
                        x_col="coherent_mean", y_col="misalignment",
                        fname=f"scatter_overall_raw_{d}.png",
                        title=f"[{d}] Overall: misalignment (=100−aligned_mean) "
                              f"vs coherence")
    scatter_overall_one(overall_delta, d,
                        x_col="delta_coherent", y_col="delta_misalignment",
                        fname=f"scatter_overall_delta_{d}.png",
                        title=f"[{d}] Overall: Δmisalignment vs Δcoherence "
                              f"(vs {BASELINE_LABEL[d][0]})")


qids = sorted(per_q_df["id"].unique())
assert len(qids) == 16, f"expected 16 question ids, got {len(qids)}"


def grid_scatter_one(df, direction, x_col, y_col, fname, title,
                     mark_baseline=False):
    sub_all = df[df["direction"] == direction]
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=False, sharey=False)
    handles_seen = {}
    for ax, qid in zip(axes.flat, qids):
        sub = sub_all[sub_all["id"] == qid]
        for _, r in sub.iterrows():
            c = color_for(r["direction"], r["alpha"])
            lab = label_for(r["direction"], r["alpha"])
            h = ax.scatter(r[x_col], r[y_col], color=c, s=40,
                           alpha=0.85, edgecolor="black", linewidth=0.3)
            handles_seen.setdefault(lab, h)
        if mark_baseline:
            bpq = baseline_pq[direction].set_index("id")
            if qid in bpq.index:
                b = bpq.loc[qid]
                name, marker = BASELINE_LABEL[direction]
                h = ax.scatter(b["coherent_mean"], b["misalignment"],
                               marker=marker, s=140, color="black", zorder=5)
                handles_seen.setdefault(f"baseline: {name}", h)
        ax.axhline(0, color="grey", linewidth=0.4, alpha=0.4)
        ax.axvline(0, color="grey", linewidth=0.4, alpha=0.4)
        ax.set_title(qid, fontsize=10)
        ax.set_xlabel(x_col, fontsize=8)
        ax.set_ylabel(y_col, fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(title, fontsize=14)
    fig.legend(handles_seen.values(), handles_seen.keys(),
               loc="lower center", ncol=len(handles_seen), fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(OUT_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


for d in ("pos", "neg"):
    grid_scatter_one(per_q_df, d,
                     x_col="coherent_mean", y_col="misalignment",
                     fname=f"scatter_perquestion_raw_{d}.png",
                     title=f"[{d}] Per-question: misalignment vs coherence (per config)",
                     mark_baseline=True)
    grid_scatter_one(per_q_df, d,
                     x_col="delta_coherent", y_col="delta_misalignment",
                     fname=f"scatter_perquestion_delta_{d}.png",
                     title=f"[{d}] Per-question: Δmisalignment vs Δcoherence "
                           f"(vs {BASELINE_LABEL[d][0]})",
                     mark_baseline=False)

print(f"[done] wrote {len(overall_df)} configs × {len(qids)} questions")
print(f"       outputs in {OUT_DIR}")
