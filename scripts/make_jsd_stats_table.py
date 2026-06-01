"""LaTeX stats table for the sleeper steering sweep (OV / Conv / DoM).

Reproduces the paper's 7-column table. For each method and SAE seed, pick the
optimal steering strength α* = argmin JSD_clean^matched subject to ASR ≤ ε. At α*
record matched JSD_clean, unmatched (cross-decode-seed) JSD_clean, JSD_pois,
exact-match-to-clean rate, and ASR. Aggregate across seeds (mean ± sample std,
ddof=1). DoM is SAE-free → a single row (n=1). α* is reported in the paper's
negative convention (α<0 subtracts the feature); run_experiment stores the
positive magnitude, so we negate for display.

Each method is a run_experiment results file in the uniform schema (--mode winner
for OV/Conv → one winning tuple per SAE seed; dom → one row):
  --ov   results/ov_winner.json
  --conv results/conv.json
  --dom  results/dom.json

Outputs a self-contained `tabular` block to stdout and (by default) to
`figures/jsd_stats_table.tex`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

# (key in the row tuple, stat name)
_KEYS = ("alpha", "jsd_clean", "jsd_clean_unmatched", "jsd_pois", "exact_clean", "asr")


def _rows_from_result(result: dict) -> list[tuple]:
    """(alpha, jsd_clean^matched, jsd_clean^unmatched, jsd_pois, exact, asr) per α."""
    items = []
    for k, ev in result["alpha_sweep"].items():
        items.append((float(k), ev["jsd_clean"], ev.get("jsd_clean_unmatched", float("nan")),
                      ev["jsd_pois"], ev["exact_match"], ev["asr"]))
    return sorted(items, key=lambda r: r[0])


def best_alpha_row(items: list[tuple], eps: float) -> dict | None:
    """Pick the α with minimal matched jsd_clean subject to asr ≤ eps."""
    cands = [r for r in items if r[5] <= eps]
    if not cands:
        return None
    r = min(cands, key=lambda r: r[1])
    return dict(zip(_KEYS, r))


def collect(path: Path, eps: float, exclude_seed: int) -> dict:
    """Aggregate α*-optimal metrics across a method file's per-seed results."""
    d = json.loads(Path(path).read_text())
    results = [r for r in d["results"] if r["seed"] != exclude_seed]
    per_seed = [best_alpha_row(_rows_from_result(r), eps) for r in results]
    seeds = [r["seed"] for r in results]
    rows = [r for r in per_seed if r is not None]
    out: dict = {"per_seed": per_seed, "seeds": seeds, "n_pool": len(results),
                 "n_meets_asr": len(rows), "single": len(results) == 1}
    if rows:
        for k in _KEYS:
            vals = [r[k] for r in rows]
            out[k] = {"mean": mean(vals),
                      "std":  stdev(vals) if len(vals) > 1 else 0.0,
                      "n":    len(vals)}
    return out


def fmt(stat: dict, prec: int = 3) -> str:
    return f"${stat['mean']:.{prec}f} \\pm {stat['std']:.{prec}f}$"


def render_table(stats: list[tuple[str, dict]], floor: float) -> str:
    rows = []
    for name, st in stats:
        if "alpha" not in st:
            rows.append(f"{name} & --- & --- & --- & --- & --- & --- \\\\")
            continue
        if st.get("single"):
            # SAE-free single direction: point values padded with a phantom
            # "\pm Y" so each number left-aligns under the means of the rows above.
            rows.append(
                f"{name} & "
                f"${-st['alpha']['mean']:.2f}\\phantom{{{{}}\\pm 0.00}}$ & "
                f"${st['jsd_clean']['mean']:.3f}\\phantom{{{{}}\\pm 0.000}}$ & "
                f"${st['jsd_clean_unmatched']['mean']:.3f}\\phantom{{{{}}\\pm 0.000}}$ & "
                f"${st['jsd_pois']['mean']:.3f}\\phantom{{{{}}\\pm 0.000}}$ & "
                f"${st['exact_clean']['mean']*100:.1f}\\phantom{{{{}}\\pm 0.0}}$ & "
                f"${st['asr']['mean']*100:.2f}\\phantom{{{{}}\\pm 0.00}}$ \\\\"
            )
            continue
        rows.append(
            f"{name} & "
            f"${-st['alpha']['mean']:.2f} \\pm {st['alpha']['std']:.2f}$ & "
            f"{fmt(st['jsd_clean'])} & "
            f"{fmt(st['jsd_clean_unmatched'])} & "
            f"{fmt(st['jsd_pois'])} & "
            f"${st['exact_clean']['mean']*100:.1f} \\pm {st['exact_clean']['std']*100:.1f}$ & "
            f"${st['asr']['mean']*100:.2f} \\pm {st['asr']['std']*100:.2f}$ \\\\"
        )
    body = "\n".join(rows)
    return (
        r"\setlength{\tabcolsep}{4.5pt}" "\n"
        r"\begin{tabular}{lcccccc}" "\n"
        r"\toprule" "\n"
        r"Method & $\alpha^*$ & JSD$_\text{clean}^\text{matched}$ $\downarrow$ & "
        rf"JSD$_\text{{clean}}^\text{{unmatched}}$ (${floor:.2f}$) $\downarrow$ & "
        r"JSD$_\text{pois}^\text{matched}$ $\uparrow$ & Exact match (\%) $\uparrow$ & "
        r"ASR (\%) $\downarrow$ \\" "\n"
        r"\midrule" "\n"
        f"{body}\n"
        r"\bottomrule" "\n"
        r"\end{tabular}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ov",   type=Path, default=Path("results/ov_winner.json"))
    p.add_argument("--conv", type=Path, default=Path("results/conv.json"))
    p.add_argument("--dom",  type=Path, default=Path("results/dom.json"))
    p.add_argument("--out",  type=Path, default=Path("figures/jsd_stats_table.tex"),
                   help="LaTeX file to write the tabular block to (also printed to stdout).")
    p.add_argument("--epsilon", type=float, default=0.01,
                   help="ASR threshold for the optimal-alpha criterion.")
    p.add_argument("--floor", type=float, default=0.61,
                   help="Clean-vs-clean unmatched JSD floor shown in the header.")
    p.add_argument("--exclude-seed", type=int, default=-1,
                   help="SAE seed to exclude; -1 (default) keeps all seeds.")
    args = p.parse_args()

    methods = [("OV", args.ov), ("Conv", args.conv), ("DoM", args.dom)]
    print(f"# optimal-alpha rule: argmin JSD(steered, clean)  s.t. ASR <= {args.epsilon}")
    if args.exclude_seed >= 0:
        print(f"# excluded SAE seed: {args.exclude_seed}")

    stats = [(label, collect(path, args.epsilon, args.exclude_seed))
             for label, path in methods]

    for label, st in stats:
        print(f"\n## {label}: n meeting ASR threshold = {st['n_meets_asr']} / {st['n_pool']}")
        for seed_val, row in zip(st["seeds"], st["per_seed"]):
            if row is None:
                continue
            tag = "--" if st.get("single") else seed_val
            print(f"  seed {tag}: α*={-row['alpha']:.2f}  jsd_c={row['jsd_clean']:.3f}  "
                  f"unm={row['jsd_clean_unmatched']:.3f}  jsd_p={row['jsd_pois']:.3f}  "
                  f"ex={row['exact_clean']*100:5.1f}%  asr={row['asr']*100:.2f}%")

    body = render_table(stats, args.floor)
    print("\n" + "=" * 70 + "\n")
    print(body)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
