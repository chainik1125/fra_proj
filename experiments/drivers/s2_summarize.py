#!/usr/bin/env python
"""S2: summarize all s2 EM-eval results into the final report table.

Reads results/em_eval_<run>.json for the S2 runs and prints a markdown table:
run_name, financial EM (n_mis/n_coh), sports EM, betley EM (n_mis/n_coh),
gap (fin-betley), and a Wilson 95% binomial CI on the betley rate.
"""
import json
import math
import pathlib

WT = pathlib.Path(__file__).resolve().parents[2]
RUNS = [
    "s2_c075",
    "s2_dup10x100",
    "s2_dup33x30",
    "s2_dup33x30_r1",
    "s2_dup33x30_r2",
    "s2_dup100x10",
    "s2_aligned1000",
    "s2_aligned500",
    "s2_14b_dup10x100",
]


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fmt_set(d):
    if d["em_rate"] is None:
        return "n/a"
    return f"{d['em_rate']:.3f} ({d['n_misaligned']}/{d['n_coherent']})"


def main():
    print("| run | financial EM (mis/coh) | sports EM (mis/coh) | betley EM (mis/coh) | gap | betley 95% CI |")
    print("|---|---|---|---|---|---|")
    for run in RUNS:
        p = WT / "results" / f"em_eval_{run}.json"
        if not p.exists():
            print(f"| {run} | MISSING | | | | |")
            continue
        r = json.loads(p.read_text())
        fin, sp, bet = r["financial"], r["sports"], r["betley"]
        lo, hi = wilson(bet["n_misaligned"], bet["n_coherent"])
        gap = r["generalization_gap"]
        gap_s = "n/a" if gap is None else f"{gap:+.3f}"
        print(f"| {run} | {fmt_set(fin)} | {fmt_set(sp)} | {fmt_set(bet)} | {gap_s} | "
              f"[{lo:.3f}, {hi:.3f}] |")


if __name__ == "__main__":
    main()
