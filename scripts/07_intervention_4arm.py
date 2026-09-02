"""Brief section 6, metrics 5-6, with the arm that actually decides it.

Four arms at every rho:

  a) planted        zero FRA_QK[:,:,lambda*,mu*]
  b) random         a random pair RESCALED to remove the same total score mass
  c) agg runner-up  the second-ranked pair in the aggregate, as-is
  d) cell runner-up the largest live competitor AT the planted (q*, k*), per sequence

Arm (c) is expected to be structurally uninformative and is kept for exactly
that reason. It is selected over all (q, k), so its query feature is usually
inactive at q* -- its contribution at the planted cell is then identically zero
and ablating it cannot change anything, regardless of feature merging. The
"live fraction" column makes that visible in the data.

Arm (d) is the decisive one. Entries of FRA_QK[q*, k*] are non-zero only where
both features fire at the relevant positions, so its competitors are real
candidates for the causal role. If (a) collapses behaviour at rho=0.8 while (d)
does not, then FRA's attribution MARGIN degrades under superposition while its
causal IDENTIFICATION does not -- a sharper and more useful claim than "FRA
degrades", because it says the metric is fragile rather than the method.

Run: python scripts/07_intervention_4arm.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from fra.toy.config import ToyConfig
from fra.toy.fra import coupling_matrix
from fra.toy.intervention import (
    aggregate_pair_is_live_at_cell,
    cell_runner_up_pairs,
    evaluate_ablation,
    evaluate_ablation_per_sequence,
    matched_random_pair,
    runner_up_pair,
)
from fra.toy.metrics import accuracy, attention_concentration
from fra.toy.recovery import qk_aggregate_closed_form, recovery_of
from fra.toy.train import TrainConfig, train_cached as train

RHOS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
N_AGG = 16
N_RANDOM = 3
EVAL_BATCH = 1024
OUT = Path("results/intervention_4arm.json")

MIN_ARGMAX_IS_KEY = 0.95
MIN_HELDOUT_ACC = 0.90


def run_one(rho: float) -> dict:
    cfg = ToyConfig(rho=rho)
    t0 = time.time()
    r = train(cfg, TrainConfig(log=False))
    dgp, model = r.dgp, r.model
    lam, mu = dgp.planted_qk_edge
    gen = torch.Generator().manual_seed(1234 + int(rho * 100))

    b = dgp.sample(EVAL_BATCH, split="heldout")
    G = coupling_matrix(model, dgp.feature_directions)

    base_acc = accuracy(model, b)
    base_conc = attention_concentration(model, b)
    admitted = (
        base_conc.argmax_is_key >= MIN_ARGMAX_IS_KEY
        and base_acc.at_query >= MIN_HELDOUT_ACC
    )

    agg = torch.zeros(cfg.n_feat, cfg.n_feat)
    for _ in range(N_AGG):
        agg += qk_aggregate_closed_form(dgp.sample(1, split="heldout").features[0], G)
    rec = recovery_of(agg, (lam, mu))
    agg_runner = runner_up_pair(agg, (lam, mu))

    arms = [evaluate_ablation(model, b, G, (lam, mu), "planted")]

    rnd = []
    for i in range(N_RANDOM):
        pair, scale = matched_random_pair(
            G, b, (lam, mu), exclude={agg_runner}, generator=gen
        )
        rnd.append(evaluate_ablation(model, b, G, pair, f"random{i}", scale))
    arms += rnd

    arms.append(evaluate_ablation(model, b, G, agg_runner, "agg_runner_up"))

    cl, cm = cell_runner_up_pairs(b.features, G, b.query_pos, b.key_pos, (lam, mu))
    arms.append(evaluate_ablation_per_sequence(model, b, G, cl, cm, "cell_runner_up"))

    live = aggregate_pair_is_live_at_cell(b.features, b.query_pos, b.key_pos, agg_runner)
    n = cfg.n_feat
    cell_pairs = torch.bincount(cl * n + cm, minlength=n * n)
    n_distinct = int((cell_pairs > 0).sum())

    return {
        "rho": rho,
        "rho_realized": dgp.rho_realized,
        "seconds": time.time() - t0,
        "admitted": admitted,
        "chance": 1.0 / cfg.n_content,
        "base_acc": base_acc.at_query,
        "base_mass_on_key": base_conc.mean_mass_on_key,
        "base_argmax_is_key": base_conc.argmax_is_key,
        "agg_runner_up_ratio": rec.runner_up_ratio,
        "agg_runner_up_pair": list(agg_runner),
        "agg_runner_up_live_frac": live,
        "cell_runner_up_distinct_pairs": n_distinct,
        "arms": [
            {
                "arm": a.arm, "pair": list(a.pair), "scale": a.scale,
                "removed_l1": a.removed_l1, "acc": a.acc.at_query,
                "mass_on_key": a.mass_on_key, "argmax_is_key": a.argmax_is_key,
            }
            for a in arms
        ],
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print("=" * 100)
    print("INTERVENTION, FOUR ARMS -- brief section 6 metrics 5-6.  Stage A, seed 0.")
    print("=" * 100)

    for rho in RHOS:
        row = run_one(rho)
        rows.append(row)
        OUT.write_text(json.dumps(rows, indent=2))

        base = row["base_acc"]
        print(f"\nrho={row['rho']:.2f}  baseline acc {base*100:.2f}%  "
              f"Gate2 {row['base_argmax_is_key']*100:.1f}%  admitted={row['admitted']}  "
              f"agg runner-up ratio {row['agg_runner_up_ratio']:.2f}  "
              f"chance {row['chance']*100:.1f}%")
        print(f"  agg runner-up {tuple(row['agg_runner_up_pair'])} is live at (q*,k*) in "
              f"{row['agg_runner_up_live_frac']*100:.1f}% of sequences; "
              f"cell arm uses {row['cell_runner_up_distinct_pairs']} distinct pairs")
        print(f"  {'arm':16s} {'pair':>11s} {'acc':>8s} {'d_acc':>8s} "
              f"{'mass_key':>9s} {'argmax':>8s} {'removed_L1':>11s}")

        rnd = [a for a in row["arms"] if a["arm"].startswith("random")]
        rows_out = [
            row["arms"][0],
            {"arm": f"random(x{len(rnd)})", "pair": [-1, -1],
             "acc": sum(a["acc"] for a in rnd) / len(rnd),
             "mass_on_key": sum(a["mass_on_key"] for a in rnd) / len(rnd),
             "argmax_is_key": sum(a["argmax_is_key"] for a in rnd) / len(rnd),
             "removed_l1": sum(a["removed_l1"] for a in rnd) / len(rnd)},
            row["arms"][-2],
            row["arms"][-1],
        ]
        for a in rows_out:
            pair = " (matched)" if a["pair"][0] < 0 else f"({a['pair'][0]:3d},{a['pair'][1]:3d})"
            star = " *" if a["arm"] == "cell_runner_up" else ""
            print(f"  {a['arm']:16s} {pair:>11s} {a['acc']*100:7.2f}% "
                  f"{(a['acc']-base)*100:+7.2f}% {a['mass_on_key']:9.4f} "
                  f"{a['argmax_is_key']*100:7.2f}% {a['removed_l1']:11.1f}{star}")

    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
