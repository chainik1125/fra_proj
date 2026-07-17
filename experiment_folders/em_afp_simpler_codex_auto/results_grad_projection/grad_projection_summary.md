# One-step gradient projection: result summary

*2026-07-01. Script: `experiments/special_sfp_grad_projection.py`, launched via
`cloud/modal_grad_projection.py` (5 priors × 3 seeds, A10G, max_inputs=1). Per (prior,
seed): pretrain headline model, fit base probes (8 targets incl. det), apply ONE
fine-tuning update — mean raw gradient over 8 MD batches at SGD lr 1e-3 (`sgd_accum`),
the same at lr/4 (`sgd_quarter`, linearity check), and the actual first AdamW step at
lr 2e-3 (`adam_first`) — then measure the shift in base-probe-decoded coordinates on
fixed base-process data, z-scored per target, overall and restricted to O/D contexts.*

## Question

The correlated-prior control left one hypothesis standing: SGD applies a global persona
update even when a sector-specific coordinate is representable. Prediction tested here:
the step-0 update moves decoded P_M globally (including on O-contexts) rather than the
sector coordinate det.

## Findings

1. **Measurements are real, magnitudes tiny.** Linearity check passes cleanly
   (cos(sgd_accum, sgd_quarter) = 0.98–0.99, norm ratio 4.0 ± 0.5), so the raw-gradient
   directions are signal, not float noise. But one raw SGD step moves decoded
   coordinates by only ~1e-4 z-units — the pretrained model is near-stationary on MD
   data at step 0, consistent with the forcing term living on short prefixes only.

2. **The step-0 raw gradient direction is NOT the clean global-persona update.** The
   normalized direction across (z_P_M, z_P_D, z_det, z_MD, z_MO) is mixed and
   prior-dependent; z_det components are often comparable to or larger than z_P_M
   (caveat: z-scoring inflates the small-scale det coordinate). The sharp prediction
   "global P_M dominates at step 0 in every condition" is not confirmed.

3. **Adam immediately reorients the update.** cos(raw-gradient direction, first-Adam-step
   direction) ≈ 0.0 ± 0.7 across conditions — the actual optimizer step is essentially
   unrelated to the raw gradient direction in decoded-coordinate space. The first Adam
   step's decoded shifts are also mixed (|z| ≲ 0.5 for P_M, det shifts of either sign).

4. **Reconciliation with the drift result**: the fine-tuning drift's global P_M
   projection is near zero at step 1–5 and grows through steps 5–18 (probe runs:
   product 0.22→0.67, corr-strong 1.14→1.73 between steps 9 and 18). Together:
   **the domain-global persona update is an emergent property of the early Adam
   trajectory, not of the instantaneous gradient at step 0.**

## Implication for the analytic program

Level 0 (forcing term = Bayes filter minus MD-conditional) remains exact. But Level 1
(one raw-gradient step) does not yet contain the EM effect, and plain NTK gradient-flow
will inherit that problem: the operative object is the *integrated, Adam-preconditioned*
early trajectory, during which the forcing term itself evolves. The tractable analytic
targets are therefore (a) linearized dynamics with Adam preconditioning (semi-analytic:
integrate the preconditioned linear ODE with the empirical kernel), or (b) the head-only
reduction (convex, fully solvable) — checking whether head-only fine-tuning reproduces
broad transfer is the cheap decisive gate.

## Artifacts

- `combined_grad_projection.csv` (per prior/seed/method/layer/context decoded shifts,
  raw and z-scored), per-run metadata under `pi0_<tag>/`.
