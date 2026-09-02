"""Train the toy model and report the three things that gate FRA.

  1. accuracy at lambda* positions vs elsewhere
  2. train vs held-out variants  (feature-level solving, or token memorisation?)
  3. Gate 2 -- attention concentration on the planted key

Run: python scripts/01_train.py
"""

from __future__ import annotations

import torch

from fra.toy.config import ToyConfig
from fra.toy.metrics import attention_mass_by_role
from fra.toy.train import EVAL_SPLITS, TrainConfig, evaluate, train


def main() -> None:
    cfg = ToyConfig()
    tcfg = TrainConfig()

    print("=" * 78)
    print(f"TRAINING  rho={cfg.rho} ({cfg.overlap_mode})  d_model={cfg.d_model} "
          f"d_head={cfg.d_head}  seq={cfg.seq_len}  n_feat={cfg.n_feat}")
    print(f"  {tcfg.steps} steps, batch {tcfg.batch}, lr {tcfg.lr}, AdamW, laptop CPU")
    print(f"  held out {cfg.n_heldout_query} of {cfg.n_query_variants} lambda* variants "
          f"and {cfg.n_heldout_key} of {cfg.n_key_variants} mu* variants")
    print("=" * 78)

    result = train(cfg, tcfg)
    print(f"\n  trained in {result.seconds:.1f}s")

    ev = evaluate(result, batch=2048)
    chance = 100.0 / cfg.n_content

    print()
    print("=" * 78)
    print("1 + 2.  ACCURACY  (n=2048 sequences per split)")
    print("=" * 78)
    print(f"  {'split':16s} {'query':>9s} {'elsewhere':>11s}    (chance at query = {chance:.1f}%)")
    for split in EVAL_SPLITS:
        a = ev[split]["acc"]
        print(f"  {split:16s} {a.at_query*100:8.2f}% {a.elsewhere*100:10.2f}%")
    gap = (ev["train"]["acc"].at_query - ev["heldout"]["acc"].at_query) * 100
    print(f"\n  query-accuracy gap  train - heldout = {gap:+.2f} pp")
    print("  A small gap means the model keyed on the FEATURE lambda*/mu*.")
    print("  heldout_query isolates the QK side; heldout_key isolates the OV readout.")

    print()
    print("=" * 78)
    print("3.  GATE 2 -- ATTENTION CONCENTRATION AT lambda* QUERY POSITIONS")
    print("=" * 78)
    print(f"  {'split':16s} {'mass_on_key':>12s} {'argmax_is_key':>14s} "
          f"{'self':>8s} {'other':>8s}")
    for split in EVAL_SPLITS:
        c = ev[split]["attn"]
        roles = attention_mass_by_role(result.model, ev[split]["batch"])
        print(f"  {split:16s} {c.mean_mass_on_key:12.4f} {c.argmax_is_key*100:13.2f}% "
              f"{roles['self']:8.4f} {roles['other']:8.4f}")

    print()
    held = ev["heldout"]
    ok = held["acc"].at_query > 0.95 and held["attn"].argmax_is_key > 0.95
    print(f"  GATE 2: {'PASS' if ok else 'REVIEW'} -- held-out "
          f"argmax_is_key = {held['attn'].argmax_is_key*100:.2f}%, "
          f"query accuracy = {held['acc'].at_query*100:.2f}%")
    if not ok:
        print("  Do NOT proceed to FRA. Fix the task or the training first:")
        print("  a weak edge here means FRA would correctly report no edge exists.")


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()
