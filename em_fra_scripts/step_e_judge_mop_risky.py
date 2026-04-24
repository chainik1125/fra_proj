"""
Step E (MOP-risky) — Grade the `step_d_mop_risky` coef sweep with the OpenAI
Batch API, reusing every piece of plumbing from `step_e_judge_sweep_batch.py`.

Scope (per user request):
  * Sweep dir  outputs2/step_d_mop_risky/sweep/
  * Features   top 15   (rank 000..014)
  * Alphas     {1.50, 2.00}
  * Directions {pos, neg}
  = 15 * 2 * 2 = 60 CSVs
  * Each CSV  160 (question, answer) rows (16 prompts × 10 samples)
                -> 9 600 rows -> 19 200 judge calls  (aligned + coherent)

All state (chunks/, results/, judge_sweep/, summary) lands in
`outputs2/step_d_mop_risky/` so you get one self-contained artifact tree.

Run:
    python em_fra_scripts/step_e_judge_mop_risky.py --phases prepare,run,summary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FRA_ROOT = Path("/home/vishalrao/FRA")
sys.path.insert(0, str(FRA_ROOT / "em_fra_scripts"))

import step_e_judge_sweep_batch as base  # noqa: E402

SWEEP_DIR = FRA_ROOT / "em_fra_scripts/outputs2/step_d_mop_risky/sweep"
OUT_DIR = FRA_ROOT / "em_fra_scripts/outputs2/step_d_mop_risky"

ALPHAS = [1.50, 2.00]
DIRECTIONS = ["pos", "neg"]
MAX_RANK = 15  # exclusive -> ranks 0..14


def discover_runs(sweep_dir: Path) -> list[base.SweepRun]:
    """Same regex as base, but filtered to the restricted (alpha, rank) scope."""
    runs: list[base.SweepRun] = []
    for p in sorted(sweep_dir.iterdir()):
        if not p.is_file() or not p.name.endswith(".csv"):
            continue
        m = base.CSV_NAME_RE.match(p.name)
        if not m:
            continue
        direction = m.group("dir")
        rank = int(m.group("rank"))
        alpha = float(m.group("alpha"))
        feat = int(m.group("feat"))
        if direction not in DIRECTIONS:
            continue
        if rank >= MAX_RANK:
            continue
        if not any(abs(alpha - a) < 1e-6 for a in ALPHAS):
            continue
        runs.append(base.SweepRun(p, direction, feat, rank, alpha))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, default=SWEEP_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--phases", default="prepare",
                    help="comma list: prepare, run, summary, all")
    ap.add_argument("--max-wait-sec", type=int, default=24 * 3600)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    judged_dir = args.out_dir / "judge_sweep"
    judged_dir.mkdir(parents=True, exist_ok=True)

    phases = {p.strip() for p in args.phases.split(",")}
    if "all" in phases:
        phases = {"prepare", "run", "summary"}

    runs = discover_runs(args.sweep_dir)
    print(f"Discovered {len(runs)} sweep CSVs in {args.sweep_dir}")
    print(f"  alphas={ALPHAS}  ranks<{MAX_RANK}  directions={DIRECTIONS}")
    if not runs:
        print("No runs matched the filter — aborting")
        sys.exit(1)

    aligned_tmpl, coherent_tmpl = base.load_prompt_templates()

    if "prepare" in phases:
        base.phase_prepare(runs, args.out_dir, judged_dir, aligned_tmpl, coherent_tmpl)
    if "run" in phases:
        base.phase_run(args.out_dir, judged_dir, max_wait_sec=args.max_wait_sec)
    if "summary" in phases:
        base.phase_summary(args.out_dir, judged_dir)


if __name__ == "__main__":
    main()
