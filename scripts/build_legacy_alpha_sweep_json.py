"""Adapter: build a legacy-schema `jsd_alpha_sweep_6seeds.json` from the new
matrix.py + downstream_baseline.py outputs, so the existing plot scripts
(plot_jsd_exact_{single_seed,all_seeds}.py) can plot the new data without
re-running the full alpha sweep.

For each (SAE seed, α, method ∈ {ov, conventional}) we copy the
per-decode-seed metric lists already stored in the new-schema outputs
(`asr_per_seed`, `jsd_clean_per_seed`, …). Shape per (α, metric) ends up
as (n_sae_seeds, n_decode_seeds), matching the legacy schema:

  configs[method].per_alpha["X.X"].asr[sae_idx]            = [d0,d1,d2,d3,d4]
  configs[method].per_alpha["X.X"].jsd_clean[sae_idx]      = [d0,d1,d2,d3,d4]
  configs[method].per_alpha["X.X"].jsd_pois[sae_idx]       = [d0,d1,d2,d3,d4]
  configs[method].per_alpha["X.X"].n_exact_match_clean[sae_idx] = [d0,…,d4]

No new GPU work — purely a JSON repackage of values that the new schema
already records per decode seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def per_alpha_for_method(per_seed_sweeps: dict, sae_seeds: list[int],
                         alphas: list[float]) -> dict:
    out: dict = {f"{a:.1f}": {"jsd_clean": [], "jsd_pois": [],
                              "n_exact_match_clean": [], "asr": []}
                 for a in alphas}
    for s in sae_seeds:
        sweep = per_seed_sweeps[s]
        for a in alphas:
            k = next(kk for kk in sweep if abs(float(kk) - a) < 1e-9)
            m = sweep[k]
            out[f"{a:.1f}"]["asr"].append(list(m["asr_per_seed"]))
            out[f"{a:.1f}"]["jsd_clean"].append(list(m["jsd_clean_per_seed"]))
            out[f"{a:.1f}"]["jsd_pois"].append(list(m["jsd_pois_per_seed"]))
            out[f"{a:.1f}"]["n_exact_match_clean"].append(
                list(m["n_exact_match_clean_per_seed"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix_json",   type=Path,
                   default=Path("results/matrix_4k_diff_rank.json"))
    p.add_argument("--baseline_json", type=Path,
                   default=Path("results/downstream_baseline_4k.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/jsd_alpha_sweep_6seeds.json"))
    p.add_argument("--eval_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n_prompts", type=int, default=200)
    args = p.parse_args()

    matrix = json.loads(args.matrix_json.read_text())
    base   = json.loads(args.baseline_json.read_text())

    m_by_seed = {r["seed"]: r for r in matrix["results"]}
    sae_seeds = sorted(m_by_seed.keys())
    first_sweep = m_by_seed[sae_seeds[0]]["winner"]["eval_sweep"]
    alphas = sorted([float(k) for k in first_sweep.keys()])

    ov_sweeps   = {s: m_by_seed[s]["winner"]["eval_sweep"] for s in sae_seeds}
    conv_sweeps = {s: base["per_seed"][f"s{s}"]["per_alpha"] for s in sae_seeds}

    legacy = {
        "alphas":          alphas,
        "sae_seeds":       sae_seeds,
        "eval_seeds":      args.eval_seeds,
        "n_prompts":       args.n_prompts,
        "schema_version":  2,
        "note":            "Built by scripts.build_legacy_alpha_sweep_json from "
                            f"{args.matrix_json.name} + {args.baseline_json.name}",
        "configs": {
            "ov": {
                "kind": "ov",
                "per_seed_feature": {str(s): m_by_seed[s]["winner"]["feature"]
                                      for s in sae_seeds},
                "per_alpha": per_alpha_for_method(ov_sweeps, sae_seeds, alphas),
            },
            "conventional": {
                "kind": "conventional",
                "per_seed_feature": {str(s): base["per_seed"][f"s{s}"]["winner"]
                                      for s in sae_seeds},
                "per_alpha": per_alpha_for_method(conv_sweeps, sae_seeds, alphas),
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(legacy, indent=2))
    print(f"wrote {args.out}")
    print(f"  ov features:   {legacy['configs']['ov']['per_seed_feature']}")
    print(f"  conv features: {legacy['configs']['conventional']['per_seed_feature']}")


if __name__ == "__main__":
    main()
