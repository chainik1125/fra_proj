"""Brief section 6, metrics 5-6: does ablating the recovered pair kill the behaviour?

The control half of Dmitry's step 4, and the experiment FRA's own Figure 1(c)
depicts but the paper never runs.

Three arms at every rho:

  a) planted     -- zero FRA_QK[:,:,lambda*,mu*].  Expect collapse.
  b) random      -- a random pair, RESCALED to remove the same total score mass.
                    Expect no effect. This is the specificity control: without
                    the rescaling it would "show no effect" trivially, by
                    perturbing almost nothing.
  c) runner-up   -- the second-ranked pair, as-is. The decisive arm.

Why (c) matters at high rho. At rho=0.8 the runner-up ratio is ~1.0, so the
planted edge is essentially tied with an arbitrary competitor *by magnitude*.
Then:

  - planted kills it, runner-up does NOT  -> FRA's RANKING is still correct even
    though its MARGIN is gone. Margin collapse overstates the degradation, and
    that qualifies our own headline.
  - both kill it -> the features have genuinely merged and the margin collapse
    is real.

Our sweep note implicitly assumes the second. This tells us which is true.

Run: python scripts/06_intervention.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from fra.toy.config import ToyConfig
from fra.toy.fra import coupling_matrix
from fra.toy.intervention import (
    evaluate_ablation,
    matched_random_pair,
    runner_up_pair,
)
from fra.toy.metrics import accuracy, attention_concentration
from fra.toy.recovery import qk_aggregate_closed_form, recovery_of
from fra.toy.train import TrainConfig, train

RHOS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
N_AGG = 16
N_RANDOM = 3
EVAL_BATCH = 1024
OUT = Path("results/intervention.json")

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

    # Aggregate, to identify the runner-up.
    agg = torch.zeros(cfg.n_feat, cfg.n_feat)
    for _ in range(N_AGG):
        agg += qk_aggregate_closed_form(dgp.sample(1, split="heldout").features[0], G)
    rec = recovery_of(agg, (lam, mu))
    runner = runner_up_pair(agg, (lam, mu))

    arms = [evaluate_ablation(model, b, G, (lam, mu), "planted")]

    for i in range(N_RANDOM):
        pair, scale = matched_random_pair(G, b, (lam, mu), exclude={runner}, generator=gen)
        arms.append(evaluate_ablation(model, b, G, pair, f"random{i}", scale))

    arms.append(evaluate_ablation(model, b, G, runner, "runner_up"))

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
        "runner_up_pair": list(runner),
        "arms": [
            {
                "arm": a.arm,
                "pair": list(a.pair),
                "scale": a.scale,
                "removed_l1": a.removed_l1,
                "acc": a.acc.at_query,
                "mass_on_key": a.mass_on_key,
                "argmax_is_key": a.argmax_is_key,
            }
            for a in arms
        ],
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print("=" * 104)
    print("INTERVENTION -- brief section 6, metrics 5-6.  Stage A, single seed.")
    print("=" * 104)

    for rho in RHOS:
        row = run_one(rho)
        rows.append(row)
        OUT.write_text(json.dumps(rows, indent=2))

        chance = row["chance"] * 100
        print(f"\nrho={row['rho']:.2f}   baseline acc {row['base_acc']*100:.2f}%  "
              f"Gate2 {row['base_argmax_is_key']*100:.1f}%  "
              f"admitted={row['admitted']}  "
              f"agg runner-up ratio {row['agg_runner_up_ratio']:.2f}  "
              f"(chance {chance:.1f}%)")
        print(f"  {'arm':12s} {'pair':>11s} {'acc':>8s} {'d_acc':>8s} "
              f"{'mass_key':>9s} {'argmax':>8s} {'removed_L1':>11s}")
        rnd = [a for a in row["arms"] if a["arm"].startswith("random")]
        shown = [a for a in row["arms"] if not a["arm"].startswith("random")]
        rnd_mean = {
            "arm": f"random(x{len(rnd)})",
            "pair": [-1, -1],
            "acc": sum(a["acc"] for a in rnd) / len(rnd),
            "mass_on_key": sum(a["mass_on_key"] for a in rnd) / len(rnd),
            "argmax_is_key": sum(a["argmax_is_key"] for a in rnd) / len(rnd),
            "removed_l1": sum(a["removed_l1"] for a in rnd) / len(rnd),
        }
        for a in [shown[0], rnd_mean, shown[1]]:
            pair = "  (matched)" if a["pair"][0] < 0 else f"({a['pair'][0]:3d},{a['pair'][1]:3d})"
            print(f"  {a['arm']:12s} {pair:>11s} {a['acc']*100:7.2f}% "
                  f"{(a['acc']-row['base_acc'])*100:+7.2f}% "
                  f"{a['mass_on_key']:9.4f} {a['argmax_is_key']*100:7.2f}% "
                  f"{a['removed_l1']:11.1f}")

    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
