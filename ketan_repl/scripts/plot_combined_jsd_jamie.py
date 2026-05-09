"""4-panel comparison plot: our JSD (row 1) and jamie's metrics (row 2).

Columns: single resid_mid feature (left) vs single OV top-1 feature (right).
Rows:    our JSD vs α (top) | jamie's gen-CE + severity vs α (bottom).

Each row has a shared y-axis. Same α grid for both rows.

Inputs:
  --jsd_root_top1: dir of our JSD-instrumented runs with --ov_n 1
  --jamie_inputs:  list of jamie pipeline JSONs from --eval_mode single
  --output:        output PNG

The script reads:
  • our JSD: aggregate_inputs/train_seed_*/per_token_metrics.csv
    (we use the OV/FRA family as 'single OV top-1', and the Single feature
     family as the resid_mid baseline)
  • jamie's metrics: per-feature points filtered to the top-1 of the OV
    ranking; downstream baseline points for resid_mid.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect_our_jsd(aggregate_root: Path, scope: str = "first_token"):
    """Per-(family, alpha) means of jsd_steered_to_clean and ..._to_pp."""
    raw: dict[str, dict[float, dict[str, list[float]]]] = {}
    for seed_dir in sorted(aggregate_root.glob("train_seed_*")):
        with open(seed_dir / "per_token_metrics.csv") as f:
            for row in csv.DictReader(f):
                if scope == "first_token" and int(row["position"]) != 1:
                    continue
                family = row["family"]
                alpha = float(row["alpha"])
                try:
                    j_clean = float(row["jsd_steered_to_clean"])
                    j_pp    = float(row["jsd_steered_to_pp"])
                except (KeyError, ValueError):
                    raise SystemExit(f"missing JSD columns in {seed_dir/'per_token_metrics.csv'}")
                if not (j_clean == j_clean and j_pp == j_pp):
                    continue
                d = raw.setdefault(family, {}).setdefault(alpha, {"clean": [], "pp": []})
                d["clean"].append(j_clean); d["pp"].append(j_pp)
    out: dict[str, dict[float, dict[str, float]]] = {}
    for family, by_alpha in raw.items():
        out[family] = {a: {"clean_mean": statistics.mean(v["clean"]),
                            "pp_mean":    statistics.mean(v["pp"])}
                       for a, v in by_alpha.items()}
    return out


def collect_jamie_single(input_paths: list[Path]):
    """Per-(family, alpha) means across seeds for jamie's metrics.

    We look at:
      family = "downstream" (single resid_mid feature)
      family = "upstream"  + eval_mode = "single" + first feature in the
        per_seed selection list (= top-1 OV) per seed
    """
    # In jamie's pipeline `--eval_mode single`, the upstream points are for
    # the *post-screen winner* (rank 1) — exactly the single OV feature that
    # the pipeline chose. We accept any upstream point with eval_mode='single'.
    raw_down: dict[float, dict[str, list[float]]] = {}
    raw_up:   dict[float, dict[str, list[float]]] = {}
    for p in input_paths:
        d = json.loads(p.read_text())
        for pt in d.get("points", []):
            fam = pt.get("family")
            alpha = float(pt["alpha"])
            asr   = float(pt.get("asr", float("nan")))
            sev   = pt.get("severity_ratio") or pt.get("recovery_noise_ratio")
            gen   = pt.get("gen_ce_ratio")
            tgt = None
            if fam == "downstream":
                tgt = raw_down
            elif fam == "upstream" and pt.get("eval_mode") == "single":
                tgt = raw_up
            if tgt is None: continue
            entry = tgt.setdefault(alpha, {"asr": [], "sev": [], "gen": []})
            if asr == asr:      entry["asr"].append(asr)
            if sev is not None: entry["sev"].append(float(sev))
            if gen is not None: entry["gen"].append(float(gen))

    def reduce(raw):
        return {a: {"asr_mean": statistics.mean(v["asr"]) if v["asr"] else float("nan"),
                    "sev_mean": statistics.mean(v["sev"]) if v["sev"] else float("nan"),
                    "gen_mean": statistics.mean(v["gen"]) if v["gen"] else float("nan")}
                for a, v in raw.items()}
    return {"downstream": reduce(raw_down), "upstream": reduce(raw_up)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsd_root_top1", type=Path, required=True,
                   help="aggregate_inputs dir of our JSD run with --ov_n 1.")
    p.add_argument("--jamie_inputs", nargs="+", type=Path, required=True,
                   help="jamie pipeline JSONs from --eval_mode single (one per training seed).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    jsd = collect_our_jsd(args.jsd_root_top1)
    if "OV/FRA" not in jsd or "Single feature" not in jsd:
        raise SystemExit(f"need both 'OV/FRA' and 'Single feature' in our JSD data; have {sorted(jsd)}")

    jamie = collect_jamie_single(args.jamie_inputs)
    if "downstream" not in jamie or "upstream" not in jamie or not jamie["upstream"]:
        raise SystemExit(f"need both 'downstream' and 'upstream' (single, top-1) in jamie data")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), sharex=True)

    # ---- ROW 1: our JSD ----
    NUM_C = "#2ca02c"   # green = closer to clean (good)
    DEN_C = "#d62728"   # red   = closer to sleeper (bad)
    panel_titles_top = {
        "Single feature": "(a)  conventional steering · single resid_mid feature",
        "OV/FRA":         "(b)  OV → OV · single top-1 ln1 feature (V-pathway)",
    }
    for ax, family in zip(axes[0], ("Single feature", "OV/FRA")):
        alphas = sorted(jsd[family].keys())
        cleans = [jsd[family][a]["clean_mean"] for a in alphas]
        pps    = [jsd[family][a]["pp_mean"]    for a in alphas]
        ax.plot(alphas, cleans, color=NUM_C, lw=2.2, marker="o", markersize=7,
                label="JSD(p_steered , p_clean)  → distance from clean")
        ax.plot(alphas, pps, color=DEN_C, lw=2.2, marker="s", markersize=7,
                label="JSD(p_steered , p_unsteered_pp)  → distance from sleeper")
        ax.axhline(0.6931, color="#666", linestyle="--", lw=0.7, alpha=0.6)
        ax.text(2.0, 0.66, "JSD upper bound (ln 2)", fontsize=9, color="#666",
                ha="right", va="bottom")
        ax.set_title(panel_titles_top[family], fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.03, 0.75)
        ax.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.95)
    axes[0, 0].set_ylabel("Jensen-Shannon divergence (nats)\n— our metric —", fontsize=12)

    # ---- ROW 2: jamie ----
    GEN_C = "#d62728"   # red — token-level damage
    SEV_C = "#9467bd"   # purple — distribution-level damage
    ASR_C = "#2ca02c"   # green — sleeper presence
    panel_titles_bot = {
        "downstream": "(c)  conventional · single resid_mid feature  [jamie pipeline]",
        "upstream":   "(d)  OV → OV · single top-1 ln1 feature  [jamie pipeline]",
    }
    for ax, family in zip(axes[1], ("downstream", "upstream")):
        alphas = sorted(jamie[family].keys())
        gens = [jamie[family][a]["gen_mean"] for a in alphas]
        sevs = [jamie[family][a]["sev_mean"] for a in alphas]
        asrs = [jamie[family][a]["asr_mean"] for a in alphas]
        ax.plot(alphas, gens, color=GEN_C, lw=2.2, marker="o", markersize=7,
                label="gen-CE ratio (token NLL of steered text under clean ref)")
        ax.plot(alphas, sevs, color=SEV_C, lw=2.2, marker="s", markersize=7,
                label="severity ratio (dist-CE clean→steered / clean-vs-clean noise)")
        ax.plot(alphas, asrs, color=ASR_C, lw=2.0, marker="^", linestyle=":", markersize=7,
                label="ASR (sleeper-emission rate)")
        ax.axhline(1.0, color="#888", linestyle="--", lw=0.7, alpha=0.6)
        ax.text(2.0, 1.05, "no damage / sampling-noise floor", fontsize=9, color="#666",
                ha="right", va="bottom")
        ax.set_title(panel_titles_bot[family], fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        ax.set_ylim(0.005, 50)
        ax.set_xlabel("steering coefficient  α", fontsize=12)
        ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.95)
    axes[1, 0].set_ylabel("ratio (jamie's metrics)\n— jamie pipeline —", fontsize=12)

    if args.title:
        fig.suptitle(args.title, fontsize=13, y=1.00)
    else:
        fig.suptitle(
            "50k SAEs · single-feature comparison: our JSD (row 1) vs jamie/sleepers metrics (row 2)\n"
            "left = single resid_mid feature; right = single top-1 OV ln1 feature, V-pathway",
            fontsize=12, y=1.00,
        )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")

    # Print numbers
    print(f"\n--- our JSD (single feature top-1 OV vs single resid_mid) ---")
    print(f"{'family':<16} {'α':>5}  {'jsd→clean':>10} {'jsd→pp':>9}")
    for family in ("Single feature", "OV/FRA"):
        for a in sorted(jsd[family].keys()):
            v = jsd[family][a]
            print(f"{family:<16} {a:>5.2f}  {v['clean_mean']:>10.4f} {v['pp_mean']:>9.4f}")

    print(f"\n--- jamie (single feature, top-1 OV  vs  resid_mid downstream) ---")
    print(f"{'family':<12} {'α':>5}  {'ASR':>6} {'gen-CE':>8} {'severity':>9}")
    for family in ("downstream", "upstream"):
        for a in sorted(jamie[family].keys()):
            v = jamie[family][a]
            print(f"{family:<12} {a:>5.2f}  {v['asr_mean']:>6.3f} {v['gen_mean']:>8.3f} {v['sev_mean']:>9.3f}")


if __name__ == "__main__":
    main()
