"""Compare two pareto_3x3.json files cell-by-cell.

Usage:
    python diff_pareto.py REFERENCE.json REPRODUCED.json
        [--asr_tol 0.01] [--ce_tol 1e-3]

Prints a per-(ranking, intervention, alpha) table of deltas and a final
PASS/FAIL line. Exit code 0 if every cell is within tolerance, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def cells(doc: dict):
    """Yield (ranking, intervention, alpha, asr, dce) for every cell."""
    for r_name, r in doc["grid"].items():
        for i_name, i in r.items():
            for row in i["per_alpha"]:
                yield r_name, i_name, row["alpha"], row["asr_16"], row["delta_ce"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ref", type=Path)
    p.add_argument("rep", type=Path)
    p.add_argument("--asr_tol", type=float, default=0.01)
    p.add_argument("--ce_tol", type=float, default=1e-3)
    args = p.parse_args()

    ref = load(args.ref)
    rep = load(args.rep)

    ref_cells = {(r, i, a): (asr, dce) for r, i, a, asr, dce in cells(ref)}
    rep_cells = {(r, i, a): (asr, dce) for r, i, a, asr, dce in cells(rep)}

    keys = sorted(set(ref_cells) | set(rep_cells))

    print(f"{'rank':<6} {'intv':<5} {'alpha':>5}   "
          f"{'ASR_ref':>8} {'ASR_rep':>8} {'dASR':>8}   "
          f"{'dCE_ref':>9} {'dCE_rep':>9} {'ddCE':>9}   flag")
    print("-" * 100)

    bad = 0
    for k in keys:
        r, i, a = k
        if k not in ref_cells:
            print(f"{r:<6} {i:<5} {a:>5}   <missing in ref>")
            bad += 1
            continue
        if k not in rep_cells:
            print(f"{r:<6} {i:<5} {a:>5}   <missing in rep>")
            bad += 1
            continue
        asr_r, dce_r = ref_cells[k]
        asr_p, dce_p = rep_cells[k]
        d_asr = asr_p - asr_r
        d_dce = dce_p - dce_r
        flag = ""
        if abs(d_asr) > args.asr_tol:
            flag += " ASR"
            bad += 1
        if abs(d_dce) > args.ce_tol:
            flag += " CE"
            bad += 1
        print(f"{r:<6} {i:<5} {a:>5.1f}   "
              f"{asr_r:>8.3f} {asr_p:>8.3f} {d_asr:>+8.3f}   "
              f"{dce_r:>+9.4f} {dce_p:>+9.4f} {d_dce:>+9.4f}   {flag}")

    print("-" * 100)
    headline_ref = ref_cells.get(("ov", "ov", 2.0))
    headline_rep = rep_cells.get(("ov", "ov", 2.0))
    if headline_ref and headline_rep:
        print(f"headline (OV, OV, alpha=2): ref ASR={headline_ref[0]:.3f} dCE={headline_ref[1]:+.5f} | "
              f"rep ASR={headline_rep[0]:.3f} dCE={headline_rep[1]:+.5f}")

    if bad:
        print(f"FAIL: {bad} cell(s) outside tolerance "
              f"(asr_tol={args.asr_tol}, ce_tol={args.ce_tol})")
        return 1
    print(f"PASS: all {len(keys)} cells within tolerance "
          f"(asr_tol={args.asr_tol}, ce_tol={args.ce_tol})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
