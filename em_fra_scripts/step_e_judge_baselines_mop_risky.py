"""
Judge base + misaligned baseline gens for the MOP-risky (step_d_mop_risky)
setup, so we can compute Δmis% vs baseline for the sweep in step_e_judge_mop_risky.

Inputs (80 rows each = 16 qs × 5 samples):
  outputs2/step_d_mop_risky/gens/base_act.csv
  outputs2/step_d_mop_risky/gens/misaligned_act.csv

Outputs (all in outputs2/step_d_mop_risky/):
  baselines_judged/base_act.csv             + aligned, coherent
  baselines_judged/misaligned_act.csv       + aligned, coherent
  baselines_state.json                      (batch tracking)
  baselines_requests.jsonl / baselines_results.jsonl
  baselines_summary.csv                     (mean scores + misalignment %)

Single batch — 160 rows × 2 judgments = 320 requests. Same gpt-5.4-mini judge.
Resumable: re-run is a no-op once applied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FRA_ROOT = Path("/home/vishalrao/FRA")
sys.path.insert(0, str(FRA_ROOT / "em_fra_scripts"))

import step_e_judge_baselines as base  # noqa: E402

GENS_DIR = FRA_ROOT / "em_fra_scripts/outputs2/step_d_mop_risky/gens"
OUT_DIR = FRA_ROOT / "em_fra_scripts/outputs2/step_d_mop_risky"
JUDGED_DIR = OUT_DIR / "baselines_judged"

# Point the reused module's module-level globals at the mop_risky files.
base.BASELINES = {
    "base": GENS_DIR / "base_act.csv",
    "mis":  GENS_DIR / "misaligned_act.csv",
}
base.OUT_DIR = OUT_DIR
base.JUDGED_DIR = JUDGED_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="prepare,submit,poll,apply,summary")
    ap.add_argument("--max-wait-sec", type=int, default=24 * 3600)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JUDGED_DIR.mkdir(parents=True, exist_ok=True)
    requests_path = OUT_DIR / "baselines_requests.jsonl"
    results_path = OUT_DIR / "baselines_results.jsonl"
    state_path = OUT_DIR / "baselines_state.json"
    phases = {p.strip() for p in args.phases.split(",")}

    aligned_tmpl, coherent_tmpl = base.load_prompt_templates()

    if "prepare" in phases:
        n = base.phase_prepare(JUDGED_DIR, requests_path, aligned_tmpl, coherent_tmpl)
        print(f"[baselines] wrote {n} requests to {requests_path} "
              f"({requests_path.stat().st_size:,} B)")
    if "submit" in phases:
        base.phase_submit(OUT_DIR, requests_path, state_path)
    if "poll" in phases:
        base.phase_poll(state_path, max_wait_sec=args.max_wait_sec)
    if "apply" in phases:
        base.phase_apply(state_path, results_path, JUDGED_DIR)
    if "summary" in phases:
        base.phase_summary(JUDGED_DIR)


if __name__ == "__main__":
    main()
