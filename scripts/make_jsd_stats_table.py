"""LaTeX stats table for the α-sweep figure.

For each SAE seed and each method ∈ {OV→OV, conventional}, pick the
optimal steering strength α* = argmin JSD$_\\text{clean}$ subject to
ASR ≤ ε. At α* record:
  JSD(steered, clean), JSD(steered, poisoned), exact-match-to-clean rate, ASR.

Aggregate across seeds (excluding any seed whose downstream baseline is
degenerate — i.e. no α achieves ASR ≤ ε for the conventional method).
Reported uncertainty = sample std (ddof=1) over the surviving seeds.

Reads the new-schema JSONs produced by scripts.matrix and
scripts.downstream_baseline:

  --matrix_json   results/matrix_4k_diff_rank.json
  --baseline_json results/downstream_baseline_4k.json

Outputs the `tabular` block to stdout and to --out if given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def _lookup_alpha_key(per_alpha: dict, a: float) -> str:
    for key in (str(a), f"{a:.1f}", f"{a:.2f}", repr(a)):
        if key in per_alpha:
            return key
    raise KeyError(f"α={a} not in keys {list(per_alpha)[:6]}…")


def best_alpha_row(per_alpha: dict, eps: float) -> dict | None:
    """Pick α minimising jsd_clean subject to asr ≤ eps in this per_alpha dict.
    Returns the chosen row's metrics, or None if no α meets the ASR threshold."""
    alphas = sorted([float(k) for k in per_alpha.keys()])
    candidates = []
    for a in alphas:
        k = _lookup_alpha_key(per_alpha, a)
        if float(per_alpha[k]["asr"]) <= eps:
            candidates.append((a, k))
    if not candidates:
        return None
    a, k = min(candidates, key=lambda x: float(per_alpha[x[1]]["jsd_clean"]))
    e = per_alpha[k]
    return {
        "alpha":       a,
        "jsd_clean":   float(e["jsd_clean"]),
        "jsd_pois":    float(e["jsd_pois"]),
        "exact_clean": float(e["exact_match"]),
        "asr":         float(e["asr"]),
    }


def collect(per_seed_dicts: dict[int, dict], seeds: list[int], eps: float) -> dict:
    """For each seed, find its α* row. Aggregate across seeds where one exists."""
    rows = []
    for s in seeds:
        pa = per_seed_dicts[s]
        r = best_alpha_row(pa, eps)
        if r is not None:
            rows.append(r)
    out: dict = {"n_meets_asr": len(rows), "n_total": len(seeds)}
    if rows:
        for k in ("alpha", "jsd_clean", "jsd_pois", "exact_clean", "asr"):
            vals = [r[k] for r in rows]
            out[k] = {"mean": mean(vals),
                      "std":  stdev(vals) if len(vals) > 1 else 0.0,
                      "n":    len(vals)}
    return out


def fmt(stat: dict, prec: int = 3) -> str:
    return f"${stat['mean']:.{prec}f} \\pm {stat['std']:.{prec}f}$"


def render_table(stats_ov: dict, stats_conv: dict) -> str:
    rows = []
    for name, st in [("Single OV$\\to$OV", stats_ov),
                     ("Conventional additive", stats_conv)]:
        if "alpha" not in st:
            rows.append(f"{name} & --- & --- & --- & --- & --- \\\\")
            continue
        rows.append(
            f"{name} & "
            f"${st['alpha']['mean']:.2f} \\pm {st['alpha']['std']:.2f}$ & "
            f"{fmt(st['jsd_clean'])} & "
            f"{fmt(st['jsd_pois'])} & "
            f"${st['exact_clean']['mean']*100:.1f} \\pm {st['exact_clean']['std']*100:.1f}$\\% & "
            f"${st['asr']['mean']*100:.2f} \\pm {st['asr']['std']*100:.2f}$\\% \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{tabular}}{{lccccc}}
\toprule
Method & $\alpha^*$ & JSD$_\text{{clean}}$ $\downarrow$ & JSD$_\text{{pois}}$ $\uparrow$ & Exact match $\uparrow$ & ASR $\downarrow$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix_json",   type=Path,
                   default=Path("results/matrix_4k_diff_rank.json"))
    p.add_argument("--baseline_json", type=Path,
                   default=Path("results/downstream_baseline_4k.json"))
    p.add_argument("--out", type=Path, default=None,
                   help="Optional LaTeX file to write the tabular block to.")
    p.add_argument("--epsilon", type=float, default=0.01,
                   help="ASR threshold for the optimal-α criterion.")
    args = p.parse_args()

    matrix = json.loads(args.matrix_json.read_text())
    base   = json.loads(args.baseline_json.read_text())

    matrix_by_seed = {r["seed"]: r for r in matrix["results"]}
    seeds = sorted(matrix_by_seed.keys())

    ov_per_seed:   dict[int, dict] = {
        s: matrix_by_seed[s]["winner"]["eval_sweep"] for s in seeds
    }
    conv_per_seed: dict[int, dict] = {
        s: base["per_seed"][f"s{s}"]["per_alpha"] for s in seeds
    }

    # Find seeds for which BOTH methods have at least one α meeting ASR ≤ eps —
    # exclude any seed that fails on either side as "degenerate".
    keep = []
    for s in seeds:
        ov_ok   = best_alpha_row(ov_per_seed[s],   args.epsilon) is not None
        conv_ok = best_alpha_row(conv_per_seed[s], args.epsilon) is not None
        if ov_ok and conv_ok:
            keep.append(s)
        else:
            print(f"# excluding seed {s}  (ov_ok={ov_ok}, conv_ok={conv_ok})")
    print(f"# kept seeds for aggregation: {keep}")
    print(f"# optimal-α rule: argmin JSD(steered, clean) s.t. ASR ≤ {args.epsilon}")

    stats_ov   = collect(ov_per_seed,   keep, args.epsilon)
    stats_conv = collect(conv_per_seed, keep, args.epsilon)

    for name, per_seed_dict, st in [
        ("ov",           ov_per_seed,   stats_ov),
        ("conventional", conv_per_seed, stats_conv),
    ]:
        print(f"\n## {name}: kept = {st['n_meets_asr']}/{st['n_total']}")
        for s in keep:
            row = best_alpha_row(per_seed_dict[s], args.epsilon)
            if row is None:
                continue
            print(f"  seed {s}: α*={row['alpha']:+.2f}  "
                  f"jsd_c={row['jsd_clean']:.3f}  jsd_p={row['jsd_pois']:.3f}  "
                  f"ex={row['exact_clean']*100:5.1f}%  asr={row['asr']*100:.2f}%")

    body = render_table(stats_ov, stats_conv)
    print("\n" + "=" * 70 + "\n")
    print(body)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
