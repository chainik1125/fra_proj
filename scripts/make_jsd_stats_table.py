"""LaTeX stats table for the JSD-alpha sweep.

For each (seed, method) in {ov, conventional}, pick the **optimal steering
strength** α* = argmin JSD(steered, clean) subject to ASR ≤ ε. At α* record:
  JSD(steered, clean), JSD(steered, poisoned), exact-match-to-clean rate, ASR.
Aggregate across the 5 "good" seeds (s=2's downstream SAE is degenerate and is
excluded). Reported uncertainty = sample std (ddof=1, n=5).

Outputs a self-contained `tabular` block to stdout and to `paper/figures/jsd_stats_table.tex`
if --out is given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


METHODS = [("ov", "OV"),
           ("conventional", "Conv"),
           ("dom", "DoM")]


def best_alpha_row(alphas: list[float], jsd_clean: list[float], jsd_pois: list[float],
                   exact_clean: list[float], asr: list[float],
                   eps: float) -> dict | None:
    """Pick α minimising jsd_clean subject to asr ≤ eps.
    Returns the chosen row, or None if no α meets the ASR threshold.
    """
    candidates = [i for i, a_i in enumerate(asr) if a_i <= eps]
    if not candidates:
        return None
    i_best = min(candidates, key=lambda i: jsd_clean[i])
    return {
        "alpha":       alphas[i_best],
        "jsd_clean":   jsd_clean[i_best],
        "jsd_pois":    jsd_pois[i_best],
        "exact_clean": exact_clean[i_best],
        "asr":         asr[i_best],
    }


def _mean(vals) -> float:
    """Mean of a per-decode-seed list (or a scalar in the v1 schema)."""
    if isinstance(vals, list):
        return float(sum(vals)) / max(1, len(vals))
    return float(vals)


def collect(d: dict, method: str, good_idx: list[int],
            eps: float) -> dict:
    """For each good SAE seed, pick α* = argmin JSD_clean s.t. ASR ≤ eps.

    Both criteria use the per-SAE-seed mean across decode seeds. The reported
    JSD / exact-match / ASR values at α* are also means across decode seeds.
    """
    alphas = [float(a) for a in d["alphas"]]
    cfg = d["configs"][method]["per_alpha"]
    n_prompts = d["n_prompts"]

    # SAE-free single-config method (e.g. DoM): a single deterministic
    # direction with no SAE-seed axis. It is parameter-free, so we report the
    # canonical projection-ablation point at alpha=1 rather than a swept optimum.
    if d["configs"][method].get("per_seed_feature") is None:
        ai = alphas.index(1.0)
        a1 = f"{alphas[ai]:.1f}"
        row = {"alpha":       alphas[ai],
               "jsd_clean":   _mean(cfg[a1]["jsd_clean"]),
               "jsd_pois":    _mean(cfg[a1]["jsd_pois"]),
               "exact_clean": _mean(cfg[a1]["n_exact_match_clean"]) / n_prompts,
               "asr":         _mean(cfg[a1]["asr"])}
        out: dict = {"per_seed": [row], "n_meets_asr": 1, "single": True}
        for k in ["alpha", "jsd_clean", "jsd_pois", "exact_clean", "asr"]:
            out[k] = {"mean": row[k], "std": 0.0, "n": 1}
        return out

    per_seed: list[dict | None] = []
    for i in good_idx:
        jsd_c  = [_mean(cfg[f"{a:.1f}"]["jsd_clean"][i])            for a in alphas]
        jsd_p  = [_mean(cfg[f"{a:.1f}"]["jsd_pois"][i])             for a in alphas]
        # Strict whole-sequence exact match: full 16-token rollout matches clean.
        ex_c   = [_mean(cfg[f"{a:.1f}"]["n_exact_match_clean"][i]) / n_prompts
                  for a in alphas]
        asr_i  = [_mean(cfg[f"{a:.1f}"]["asr"][i])                  for a in alphas]
        per_seed.append(best_alpha_row(alphas, jsd_c, jsd_p, ex_c, asr_i, eps))
    rows = [r for r in per_seed if r is not None]
    out: dict = {"per_seed": per_seed, "n_meets_asr": len(rows)}
    if rows:
        keys = ["alpha", "jsd_clean", "jsd_pois", "exact_clean", "asr"]
        for k in keys:
            vals = [r[k] for r in rows]
            out[k] = {"mean": mean(vals),
                      "std":  stdev(vals) if len(vals) > 1 else 0.0,
                      "n":    len(vals)}
    return out


def fmt(stat: dict, prec: int = 3) -> str:
    return f"${stat['mean']:.{prec}f} \\pm {stat['std']:.{prec}f}$"


def render_table(stats: list[tuple[str, dict]]) -> str:
    rows = []
    for name, st in stats:
        if "alpha" not in st:
            rows.append(f"{name} & --- & --- & --- & --- & --- \\\\")
            continue
        if st.get("single"):
            # SAE-free single direction: point values padded with a phantom
            # "\pm Y" so each number left-aligns under the mean of the rows above.
            rows.append(
                f"{name} & "
                f"${st['alpha']['mean']:.2f}\\phantom{{{{}}\\pm 0.00}}$ & "
                f"${st['jsd_clean']['mean']:.3f}\\phantom{{{{}}\\pm 0.000}}$ & "
                f"${st['jsd_pois']['mean']:.3f}\\phantom{{{{}}\\pm 0.000}}$ & "
                f"${st['exact_clean']['mean']*100:.1f}\\phantom{{{{}}\\pm 0.0}}$\\% & "
                f"${st['asr']['mean']*100:.2f}\\phantom{{{{}}\\pm 0.00}}$\\% \\\\"
            )
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
    p.add_argument("--input",  type=Path,
                   default=Path("results/jsd_alpha_sweep_6seeds.json"))
    p.add_argument("--out",    type=Path,
                   default=None,
                   help="Optional LaTeX file to write the tabular block to.")
    p.add_argument("--epsilon", type=float, default=0.01,
                   help="ASR threshold for the optimal-alpha criterion.")
    p.add_argument("--exclude-seed", type=int, default=-1,
                   help="Seed to exclude; -1 (default) keeps all seeds.")
    args = p.parse_args()

    d = json.loads(args.input.read_text())
    seeds = d["sae_seeds"]
    good_idx = [i for i, s in enumerate(seeds) if s != args.exclude_seed]
    good_seeds = [seeds[i] for i in good_idx]
    print(f"# excluded seed: {args.exclude_seed}    good seeds: {good_seeds}")
    print(f"# optimal-alpha rule: argmin JSD(steered, clean)  s.t. ASR <= {args.epsilon}")

    stats = [(label, collect(d, key, good_idx, args.epsilon)) for key, label in METHODS]

    for label, st in stats:
        n_pool = 1 if st.get("single") else len(good_idx)
        print(f"\n## {label}: n meeting ASR threshold = {st['n_meets_asr']} / {n_pool}")
        seed_labels = ["--"] if st.get("single") else good_seeds
        for seed_val, row in zip(seed_labels, st["per_seed"]):
            if row is None:
                continue
            print(f"  seed {seed_val}: α*={-row['alpha']:+.2f}  "
                  f"jsd_c={row['jsd_clean']:.3f}  jsd_p={row['jsd_pois']:.3f}  "
                  f"ex={row['exact_clean']*100:5.1f}%  asr={row['asr']*100:.2f}%")

    body = render_table(stats)
    print("\n" + "=" * 70 + "\n")
    print(body)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
