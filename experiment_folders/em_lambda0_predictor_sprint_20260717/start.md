# Sprint contract — initialization-time transfer as an EM predictor

Status: **preflight**. The remote agent must replace this line with exact UTC and
KST start/deadline timestamps before launching any experiment. The writing-only
block begins at T+9h and the sprint ends at T+10h.

## Objective

Determine whether cross-domain transfer from an infinitesimal narrow-misalignment
update, measured before fine-tuning, predicts later coherent broad misalignment in
held-out SFP toy conditions and Qwen replicas—and identify one intervention this
mechanistic account enables that existing behavioral monitoring does not.

## Scientific contract

The primary predictor is the off-domain change in a misaligned-versus-aligned
log-probability contrast per matched held-in-domain change after a one-step narrow
update. Report plain SGD and the *actual configured optimizer update* in the actual
trainable subspace. Treat the existing Adam-diagonal approximation as a diagnostic,
not as the primary quantity.

The primary outcome is coherent broad misalignment at a fixed, preregistered
fine-tuning-token budget. Trajectory AUC before coherence failure is secondary.
Post-hoc peak checkpoint selection cannot be the headline result.

Judge-free log-probability/forced-choice measures are primary. API judges validate
semantic interpretation on a stratified frozen sample. GPT-4o-2024-08-06 reproduces
the established rubric; Claude Sonnet 4.6 provides an independent sensitivity
check. Record disagreements and never silently substitute one judge for another.

## Hypotheses and falsifiers

| Hypothesis | Prediction | Falsifier or sharp limitation |
|---|---|---|
| H1: initialization-time transfer predicts later EM | The coefficient predicts held-out toy conditions and has the same qualitative ordering in prospective Qwen replicas. | Held-out performance is no better than shuffled labels, flips sign across reasonable frozen definitions, or only works on retrospectively selected checkpoints. |
| H2: optimizer geometry matters | The coefficient from the actual one-step optimizer update predicts better than raw gradient cosine and approximates integrated toy transfer. | Optimizer-aware conditioning adds no held-out value, or local transfer is unrelated to the integrated response. |
| H3: the signal is persona-specific | Persona/shared parameter blocks carry the relationship; leave-one-domain-out persona ablation attenuates it while style, self-distillation, and random-direction controls do not. | Generic controls predict equally well, or persona ablation leaves the association intact. |
| H4: the account buys control | A transfer-targeted data mixture, optimizer/subspace restriction, or orthogonalization reduces broad EM at matched narrow-domain learning. | It does no better than matched generic regularization or simply suppresses all learning. |

## Priority order

1. Freeze metric, outcome, calibration/holdout split, matching rule, judge sample,
   and negative controls before pairing predictors with outcomes.
2. Reproduce and audit the existing toy smoke result: exact/autodiff agreement,
   denominator stability, actual optimizer update, and one-step finite-difference check.
3. Run held-out toy calibration across seeds/SFP regimes, including correlated-prior,
   saturated-learner, label/style, and random-direction controls.
4. Run the cheapest Qwen one-step smoke, then the smallest matrix that can distinguish
   hypotheses: optimizer, trainable subspace/block, persona ablation, and replica.
5. If the predictor survives, test one matched-cost transfer-reducing intervention.
6. Reproduce the load-bearing result, audit raw generations, freeze experiments at
   T+9h, and write the cold-start report.

Existing Qwen trajectories are retrospective evidence. A prospective claim requires
at least one replica/config whose downstream outcome remains hidden until the transfer
metric and analysis are frozen. Otherwise label the Qwen result suggestive.

## Compute, API, and fallback budget

- Hard cap across Modal, RunPod, OpenAI, and Anthropic: **$200**.
- Reserve: **$20**. Stop all new paid launches at $180 committed/realized spend.
- Judge-call subcap: **$20**; use a small frozen stratified sample first.
- No single planned job may exceed $40 without reducing its scope first.
- Every launch needs a smoke-derived runtime/cost estimate, output path, stopping
  rule, and recoverable job/function/pod ID in `research_log.md`.
- Modal is primary. RunPod activates only after two consecutive Modal GPU
  capacity/platform failures (not scientific-code failures), with remaining time
  sufficient to complete and verify the job.
- RunPod is controlled through `cloud/modal_runpod_fallback.py`; always call its
  `down` action and verify `status` before declaring the sprint complete.
- Stop only resources created by this sprint. Do not alter pre-existing resources.

## Credential and external-action policy

- Codex Cloud receives only a fresh sprint-scoped `MODAL_TOKEN_ID` and
  `MODAL_TOKEN_SECRET`, both as setup-time secrets. The provider keys never enter
  Codex Cloud.
- `cloud/bootstrap_modal_api_secrets.sh` has escrowed judge and fallback keys as
  named Modal secrets. Experiment functions opt in by name; no key may be printed,
  committed, included in a prompt, or written to a result artifact.
- The Anthropic API key is for judge cross-checks only, never for running a second
  autonomous research agent.
- Push commits only to `dmitry/em/codex-lambda0-remote`. Do not publish a PR,
  upload weights/datasets publicly, or message third parties.
- Small code/config transfers to Modal/RunPod and persistent private Modal volumes
  are authorized. Large artifacts stay on the provider; record retrieval commands.

## Required deliverables

- `research_log.md`: append-only decisions, dead ends, job IDs, spend, elapsed and
  remaining time, live resources, and artifact index.
- `summary.md`: cold-start executive summary, findings, methods, controls,
  limitations, best next experiment, and artifact map.
- Machine-readable predictor/outcome tables with seeds, configurations, and
  provenance.
- Self-contained figures where they materially clarify a finding.
- A credential/resource audit stating that no keys leaked and all sprint-created
  RunPod pods were terminated; list any surviving Modal functions/volumes.

## Completion criteria

The predictor and outcome have been frozen and tested out of sample in the toy; the
strongest result has an independent numerical check; Qwen has at least a valid
one-step measurement plus a precise downstream comparison or documented blocker;
claims separate retrospective association, held-out prediction, and prospective
evidence; headline numbers match machine-readable artifacts; spend and live resources
are reported; and the report remains useful if every hypothesis fails.
