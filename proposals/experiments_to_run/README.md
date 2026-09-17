---
author: Indranil Das
date: 2026-09-16
tags:
  - proposals
---

## experiments_to_run

Outlines for the auto-research runner. One file per experiment. Each states, in this order:

1. **Experiment** -- exactly what is run (model, task, intervention, what is measured).
2. **Rationale** -- why this is the test; which baseline it is built to beat and why.
3. **Expected result** -- the pre-registered prediction, incl. the magnitude-law expectation and the
   kill-criterion (what outcome would make us drop it).

Fixed protocol for every proposal (from the Sep 16 meeting + [[plan_B]]):

- Baselines ALWAYS: **single SAE feature, additive** (add/subtract one decoder vector, sweep the
  coefficient on a grid, pick the best suppression/collateral point -- NOT directional, NOT
  multi-feature), difference-of-means ablation, payload-suppress, plus a position/attention oracle as
  the removal ceiling. All list-free.
- Primary axis: **suppression (or removal) at matched collateral**, measured on text that REUSES the
  endpoint features. Report per-case, mean, and WORST-case.
- The point of every proposal here is the **conjunction**: the target must fire on the co-occurrence
  of two individually-common, individually-benign features, so that single-feature removal is forced
  to damage all uses of one endpoint while the FRA cell spares both. If a single feature can separate
  the task, it does not belong in this folder.

Status tags in each file: `draft` -> `ready` (runnable) -> `running` -> `done`.
