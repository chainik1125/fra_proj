# Return checklist (written 2026-07-02 ~14:50 UTC, for return ~19:45 UTC)

## State of the moving parts

1. **Cloud sprint** (routine `trig_017o5nDt1RRJ6GzShkomeDch`, hourly): healthy, two
   iterations done, theory formulated, smoke-scale result in hand (convex optimum W*
   predicts the collapse). Modal tokens were NOT in the cloud environment, so it fell
   back to resumable CPU pretraining. Sprint window ends 21:04 UTC (2:04 PM PT); the
   summary lands in `experiment_folders/em_afp_simpler_codex_auto/results_headonly_theory/summary.md`
   on branch `dmitry/em/sprint-headonly`.
2. **GPU rescue** (launched locally ~14:50 UTC, Modal detached): the sprint's own
   pipeline (`cloud/modal_headonly_theory.py`) running for BOTH priors × seeds 0,1,2 on
   A10Gs. If this laptop stayed awake long enough, the outputs were committed and pushed
   to `results_headonly_theory/runs_pi0_<tag>/` and the sprint was told (via routine
   prompt update) to use them instead of grinding CPU. If not, the sprint's CPU path
   continues unaffected — the rescue is an accelerator, not a dependency.
3. **Self-distillation control** (Modal app `em-selfdistill-control`, detached): training
   the drift-control LoRA on base-Qwen completions to the organism's financial prompts.
   The adapter commits to the ft-adapters volume at `/adapters/selfdistill-financial-qwen7b`
   regardless of local connectivity.

## Do on return

- [ ] `git fetch && git log origin/dmitry/em/sprint-headonly --oneline` — read the sprint's
      `research_log.md` and (after 21:04 UTC) `summary.md`.
- [ ] **Disable the routine** at https://claude.ai/code/routines (it fires hourly forever;
      post-8h iterations are no-ops but still cost a container spin-up).
- [ ] Run the control measurement (~10 min, datacenter-side):
      `uv run --with modal modal run cloud/em_qwen_prepost_jsd.py --adapter-id /adapters/selfdistill-financial-qwen7b --tag qwen7b_selfdistill_control`
      then compare against `results/qwen_prepost_jsd_qwen7b_financial_g48_n8.json`
      (organism: JSD_D 0.267, JSD_O 0.232, ratio 0.87) for the drift-corrected table.
- [ ] `uv run --with modal modal app list` — confirm no ephemeral apps are still running
      (all should be stopped/finished; nothing bills when stopped).
- [ ] No secrets were embedded anywhere; nothing to rotate.

## New documents from this session (committed on the branch)

- `experiment_folders/em_afp_simpler_codex_auto/mechanism_writeup.md` — the 4-finding
  mechanism writeup (with the coherence-decomposition correction).
- `.../results_bayes_null/ideal_learners_note.md` — full derivation of the zero- and
  full-transfer ideal learners.
- `.../results_bayes_null/lever1_data_mixing_note.md` (+ `lever1_mixing_bound.png`) —
  the data-mixing suppression bounds, worked math, pre-registered predictions.
- `.../results_headonly_gate/headonly_vs_full_curves.png` — head-only vs full FT figure.
- `cloud/em_qwen_prepost_jsd.py` — D+O conditioned pre/post JSD (Betley-8 + FT prompts).
- `cloud/modal_selfdistill_control.py` — the drift-control training pipeline.

## Queued next experiments (discussed, not yet run)

Toy: mixing dose-response (AO vs AD arms vs the lever-1 bounds), optimizer sweep at
matched narrow, gradient surgery along the persona probe direction, displacement-
regularized FT, incoherence_o port to head-only runs, toy-side JSD(pre||post) readout.
LLM: epoch-extension transience test, SGD organism, sports-organism cross-check.
