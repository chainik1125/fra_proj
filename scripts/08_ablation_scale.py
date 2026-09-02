"""Does over-ablating the planted pair collapse the behaviour at high rho?

The 3-arm run left two live hypotheses for why ablating the planted pair costs
only ~20pp at rho >= 0.1 while costing 79pp at rho = 0:

  MAGNITUDE   removed_L1 fell 34% (G[l*,m*] shrank) while the softmax margin
              grew, so scale=1 is simply no longer enough removal. The edge is
              still the causal carrier.
  REDUNDANCY  the rule is genuinely carried by correlated coordinates of a
              non-orthogonal decomposition, and removing one of them cannot
              help however hard you push.

Scaling the ablation past 1 separates them directly. scale > 1 over-removes,
driving the pair's contribution negative:

  - collapses at some scale  -> MAGNITUDE. Kills the redundancy story, and also
    kills the "softmax got sharper" alternative, in one run.
  - never collapses          -> REDUNDANCY is real.

Free to run: reuses the checkpoints cached by scripts/07.

Run: python scripts/08_ablation_scale.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from fra.toy.config import ToyConfig
from fra.toy.fra import coupling_matrix
from fra.toy.intervention import evaluate_ablation
from fra.toy.metrics import accuracy, attention_concentration
from fra.toy.train import TrainConfig, train_cached

RHOS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
SCALES = [1.0, 1.5, 2.0, 3.0, 4.0]
EVAL_BATCH = 1024
OUT = Path("results/ablation_scale.json")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print("=" * 96)
    print("ABLATION SCALE SWEEP -- planted pair, scale > 1 over-ablates")
    print("=" * 96)

    for rho in RHOS:
        cfg = ToyConfig(rho=rho)
        r = train_cached(cfg, TrainConfig(log=False))
        dgp, model = r.dgp, r.model
        lam, mu = dgp.planted_qk_edge

        b = dgp.sample(EVAL_BATCH, split="heldout")
        G = coupling_matrix(model, dgp.feature_directions)
        base = accuracy(model, b)
        base_conc = attention_concentration(model, b)
        chance = 1.0 / cfg.n_content

        print(f"\nrho={rho:.2f}  baseline acc {base.at_query*100:.2f}%  "
              f"mass {base_conc.mean_mass_on_key:.4f}  chance {chance*100:.1f}%")
        print(f"  {'scale':>6s} {'acc':>8s} {'d_acc':>8s} {'mass_key':>9s} "
              f"{'argmax':>8s} {'removed_L1':>11s}")

        for scale in SCALES:
            out = evaluate_ablation(model, b, G, (lam, mu), f"scale{scale}", scale)
            rows.append({
                "rho": rho, "scale": scale, "acc": out.acc.at_query,
                "base_acc": base.at_query, "chance": chance,
                "mass_on_key": out.mass_on_key, "argmax_is_key": out.argmax_is_key,
                "removed_l1": out.removed_l1,
            })
            print(f"  {scale:6.1f} {out.acc.at_query*100:7.2f}% "
                  f"{(out.acc.at_query-base.at_query)*100:+7.2f}% "
                  f"{out.mass_on_key:9.4f} {out.argmax_is_key*100:7.2f}% "
                  f"{out.removed_l1:11.0f}", flush=True)
            OUT.write_text(json.dumps(rows, indent=2))

    print(f"\n  wrote {OUT}")
    print("\n  VERDICT per rho (does over-ablation reach chance?):")
    for rho in RHOS:
        rr = [x for x in rows if x["rho"] == rho]
        best = min(rr, key=lambda x: x["acc"])
        near = best["acc"] < best["chance"] * 2
        print(f"    rho={rho:.2f}  min acc {best['acc']*100:6.2f}% at scale "
              f"{best['scale']:.1f}  -> {'COLLAPSES (magnitude)' if near else 'survives (redundancy)'}")


if __name__ == "__main__":
    main()
