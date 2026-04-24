"""
Step F — FRA-based analog of the step_d/step_e positive/negative steering
experiment.

Where step_d steered a single SAE decoder column on the aligned (pos direction)
or misaligned (neg direction) model and step_e ranked features by Δ-misalignment,
here we compare the *feature-interaction* matrices — (head, feat_q, feat_k)
triples of ∑|FRA| — between the aligned base model and the misaligned finetunes
on the **same** 20 first-plot prompts used by step_c.

Definitions, matched to step_d's two directions:
    Δ       = misaligned − aligned (per (head, feat_q, feat_k))
    pos dir = interactions whose ∑|FRA| GREW under finetuning (Δ > 0)
              → these are the interactions you would *inject* to steer the
                aligned model toward misalignment
    neg dir = interactions whose ∑|FRA| SHRANK under finetuning (Δ < 0)
              → ablation targets on the misaligned model to recover alignment
    raw     = interactions with the largest ∑|FRA| overall (activate the most
              per the FRA object), reported separately for aligned and
              misaligned.

Special tokens (<|im_start|>, <|im_end|>, <|endoftext|>, BOS) were already
excluded from the (q, k) aggregation inside the step_c accumulators — their
positions never contribute to any ∑|FRA| entry reported here.

Inputs  (pre-computed by step_c):
    outputs/step_c/L24/delta_sft_full.parquet   (base vs full-FT misaligned)
    outputs/step_c/L24/delta_lora_full.parquet  (base vs rank-1 LoRA misaligned)
Outputs:
    outputs/step_f/L24/top{N}_pos_{sft,lora}.csv      (Δ > 0, largest first)
    outputs/step_f/L24/top{N}_neg_{sft,lora}.csv      (Δ < 0, most-negative first)
    outputs/step_f/L24/top{N}_activate_aligned.csv    (largest |∑FRA| in base)
    outputs/step_f/L24/top{N}_activate_misaligned_{sft,lora}.csv
    outputs/step_f/L24/*.png                          (bar plots)

Run:
    python /home/vishalrao/FRA/em_fra_scripts/step_f_fra_interactions.py \
        --layer 24 --top-n 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path("/home/vishalrao/FRA")
STEP_C_DIR = REPO_ROOT / "em_fra_scripts/outputs/step_c"
STEP_F_DIR = REPO_ROOT / "em_fra_scripts/outputs/step_f"


def top_by(df: pd.DataFrame, col: str, n: int, ascending: bool) -> pd.DataFrame:
    return df.sort_values(col, ascending=ascending).head(n).reset_index(drop=True)


def plot_bars(df: pd.DataFrame, value_col: str, title: str, save_path: Path,
              positive_color: str = "tab:red", negative_color: str = "tab:blue",
              head_col: str | None = "head"):
    fig, ax = plt.subplots(figsize=(14, max(6, 0.22 * len(df))))
    if head_col is not None and head_col in df.columns:
        labels = [
            f"H{int(h)}: f_q={int(fq)} → f_k={int(fk)}"
            for h, fq, fk in zip(df[head_col], df["feat_q"], df["feat_k"])
        ]
    else:
        labels = [
            f"f_q={int(fq)} → f_k={int(fk)}"
            for fq, fk in zip(df["feat_q"], df["feat_k"])
        ]
    vals = df[value_col].values
    colors = [positive_color if v >= 0 else negative_color for v in vals]
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def report_one_finetune(
    parquet_path: Path, label: str, ft_col: str, delta_col: str,
    out_dir: Path, top_n: int,
):
    print(f"\n==== {label} ({parquet_path.name}) ====")
    df = pd.read_parquet(parquet_path)
    n_total = len(df)
    print(f"  rows (non-zero FRA entries union): {n_total:,}")
    print(f"  per-head Δ stats: min={df[delta_col].min():+.2f}  "
          f"max={df[delta_col].max():+.2f}  "
          f"mean={df[delta_col].mean():+.4f}  std={df[delta_col].std():.2f}")

    # ------------------ head-averaged delta view ------------------
    # Average ∑|FRA| across the 40 attention heads for each (feat_q, feat_k)
    # pair, separately for base and misaligned. Delta = misaligned_mean −
    # base_mean. Rank pairs by |Δ| and plot the top-N. Heads with no entry
    # for a pair count as 0 (divide by total head count, not by n-non-zero).
    n_heads_total = int(df["head"].max()) + 1
    agg_base = df.groupby(["feat_q", "feat_k"])["base"].sum() / n_heads_total
    agg_ft = df.groupby(["feat_q", "feat_k"])[ft_col].sum() / n_heads_total
    avg = pd.concat([agg_base.rename("base_mean"), agg_ft.rename(f"{ft_col}_mean")], axis=1)
    avg[f"delta_{ft_col}_mean"] = avg[f"{ft_col}_mean"] - avg["base_mean"]
    avg[f"abs_delta_{ft_col}_mean"] = avg[f"delta_{ft_col}_mean"].abs()
    avg = avg.reset_index()

    avg.sort_values(f"abs_delta_{ft_col}_mean", ascending=False) \
       .to_csv(out_dir / f"delta_{label}_headavg_full.csv", index=False)

    top_avg = avg.sort_values(f"abs_delta_{ft_col}_mean", ascending=False) \
                 .head(top_n).reset_index(drop=True)
    top_avg.to_csv(out_dir / f"top{top_n}_delta_{label}_headavg.csv", index=False)

    print(f"\n  --- head-averaged Δ stats ({label}): "
          f"min={avg[f'delta_{ft_col}_mean'].min():+.2f}  "
          f"max={avg[f'delta_{ft_col}_mean'].max():+.2f}  "
          f"|Δ|_max={avg[f'abs_delta_{ft_col}_mean'].max():.2f} ---")
    print(f"\n  --- Top {top_n} head-averaged Δ (ranked by |Δ|, {label}) ---")
    print(top_avg[["feat_q", "feat_k", "base_mean", f"{ft_col}_mean",
                   f"delta_{ft_col}_mean"]].to_string(index=False))

    plot_bars(
        top_avg, f"delta_{ft_col}_mean",
        title=f"Top {top_n} feature-interaction Δ (head-averaged, base vs {label}, ranked by |Δ|)",
        save_path=out_dir / f"top{top_n}_delta_{label}_headavg.png",
        head_col=None,
    )

    return df




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument(
        "--step-c-dir", type=Path, default=STEP_C_DIR,
        help="Root of step_c outputs (expects L{layer}/delta_{sft,lora}_full.parquet).",
    )
    ap.add_argument("--out-dir", type=Path, default=STEP_F_DIR)
    ap.add_argument(
        "--finetunes", nargs="+", default=["sft", "lora"],
        choices=["sft", "lora"],
        help="Which misaligned variant(s) to analyse (sft=full-FT, lora=rank-1).",
    )
    args = ap.parse_args()

    src_dir = args.step_c_dir / f"L{args.layer}"
    out_dir = args.out_dir / f"L{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for label in args.finetunes:
        parquet = src_dir / f"delta_{label}_full.parquet"
        if not parquet.exists():
            print(f"[skip] {parquet} not found")
            continue
        report_one_finetune(
            parquet_path=parquet,
            label=label,
            ft_col=label,
            delta_col=f"delta_{label}",
            out_dir=out_dir,
            top_n=args.top_n,
        )

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
