# Sprint results — notes & figures only

Local copies here are the REVIEW artifacts: `*/deliverables/` (summary.md,
research logs, LaTeX notes + PDFs, figures) and `*/turns/` + supervisor logs
(audit trail). All heavy artifacts (model checkpoints, full work trees,
work tarballs) live on the PRIVATE HF dataset:

  https://huggingface.co/datasets/dmanningcoe/sprint-fra-theory

- `run1/`        — night-1 worker A (pod): sprint_work.tar.gz = full work tree
- `run1_twin/`   — night-1 worker B (pod): sprint_work.tar.gz = full work tree
- `run2_local/`  — sprint 2 (P2→P1, local): sprint_work.tar.gz = full work tree
- `grounding/`   — the paper/code overlay used by all workers

To restore any work tree: download the prefix's `sprint_work.tar.gz` and
extract. Local duplicates were removed 2026-07-16 (user short on disk).
