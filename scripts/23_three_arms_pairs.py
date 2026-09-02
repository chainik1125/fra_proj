"""The fair test of DELIVERY: three arms, run on causally-positioned pairs.

`scripts/21_sleeper_three_arms.py` ran the arms on the pair the paper's mask
ranks first, (1114, 1232). That was a null for score-space ablation, because the
trigger feature sits on the QUERY side there and is live at the decision position
in only 3% of deployment rows -- there was nothing for a cross-term ablation to
act on.

`scripts/22_decision_position_rank.py` re-ran THEIR ranking with the query mask
restricted to the decision position, and the trigger moved to the KEY side. This
script runs the same three arms on the pairs that ranking produces, so delivery
is finally compared on pairs that are causally positioned for the behaviour.

Pairs (none hand-picked -- each is a rank from one of the two maskings):
  (1114, 1232)  paper mask, rank 0        trigger on QUERY side  -> expected null
  (259, 1337)   decision mask, rank 0     trigger-piece on KEY side
  (391, 1114)   decision mask, rank 3     validated trigger on KEY side

Run: python scripts/23_three_arms_pairs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts import _three_arms_lib as lib

PAIRS = [
    (1114, 1232, "paper-mask rank0 (trigger on QUERY side)"),
    (259, 1337, "decision-mask rank0"),
    (391, 1114, "decision-mask rank3 (validated trigger on KEY side)"),
]
STRENGTHS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
OUT = Path("results/three_arms_pairs.json")


@torch.no_grad()
def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ctx = lib.setup()
    results = []
    for lam_q, lam_k, label in PAIRS:
        print("\n\n" + "#" * 98)
        print(f"# PAIR (lambda_q={lam_q}, lambda_k={lam_k})  --  {label}")
        print("#" * 98)
        results.append(lib.run_pair(ctx, lam_q, lam_k, label, STRENGTHS))
        OUT.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
