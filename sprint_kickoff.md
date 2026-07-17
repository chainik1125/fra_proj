# Sprint kickoff: analytic head-only theory of toy emergent misalignment

I want to do an experiment that gives you a chance to be creative and rigorous.

Trial an **8h unsupervised sprint** on the SFP toy model of emergent misalignment. This
8h is _wall-clock_ time, not estimated subjective time. Check `date` at the start, record
it in the research log, and check roughly every hour how much time has elapsed
(hourly-debrief protocol: at each hour boundary, zoom out, log progress vs. plan, and
decide continue/pivot).

## Context

Today's session (2026-07-01) established a chain of mechanism results on the headline
low-prior SFP process (P(M)=0.05, aligned-off, 3L/128d transformer), all documented under
`experiment_folders/em_afp_simpler_codex_auto/`:

1. **Exact inference shows zero broad transfer** at every fine-tuning dose (saturated
   4-sector Bayes learner; `results_bayes_null*/`); a product-constrained learner
   transfers fully; the trained transformer sits between (λ ≈ 1/3).
2. **Probes**: fine-tuning leaves the Bayes belief tracker intact and adds a
   domain-global persona bias (`results_probe_factorization*/`).
3. **Correlated-prior control**: networks that provably encode the sector-specific
   interaction (r2_det ≈ 0.98) show *no reduction* in broad transfer — representational
   capacity is not the binding constraint (`results_correlated_prior_control/`).
4. **Gradient projection**: the step-0 raw gradient is not the global persona update, and
   the first Adam step is nearly orthogonal to it; the global update *emerges over the
   early trajectory* (`results_grad_projection/`).
5. **Head-only gate (fresh, partially complete)**: fine-tuning ONLY the unembedding
   (features frozen, drift verified 0) reproduces the full qualitative EM phenomenology
   in a convex system — an early broad-transfer window (O→MO ≈ 0.03 near narrow ≈ 0.1
   for the product prior) that collapses into narrow MD routing as the head converges
   (O→MO ≈ 0 by narrow ≈ 0.85, where full FT still holds ≈ 0.39). Narrow learning
   completes head-only (D→MD → 0.94). So: the transient-window *structure* lives in
   solvable head dynamics; the *sustained* broad transfer at high narrow requires
   feature movement — quantifying that decomposition is part of the goal. Both
   1000-step head-only runs are complete
   (`results_probe_factorization_pi0_0p025..._headonly_long/` and
   `..._0p040..._headonly_long/`, 3 seeds each, checkpoints 1..1000). Corr-strong shows
   the same window-then-collapse shape (O→MO peaks ≈ 0.077 at narrow ≈ 0.26, dies by
   step 500 while O→MD spillback rises; narrow reaches 0.73 by step 1000), and its
   inversion over full FT at low narrow survives at scale. The theory should predict:
   window height, window location along the narrow axis, and the late collapse — for
   both priors, from frozen features alone.

The main open objection to address: everything so far brackets the mechanism empirically;
nothing yet *predicts* the transfer quantitatively.

## Goal

**Derive and validate an analytic theory of head-only broad transfer.** Concretely: with
features frozen, head-only fine-tuning is convex — the head dynamics under cross-entropy
on MD data are fully determined by the frozen base features φ(h) and the analytic forcing
term p_Bayes(·|h; π₀) − p_Bayes(·|h; e_MD). Integrate these dynamics (analytically where
possible, numerically-exactly otherwise), predict the O→MO broad-transfer curve as a
function of narrow transfer with NO fine-tuning run, and validate the prediction against
the 1000-step head-only measurements for both the product and corr-strong priors. The
deliverable claim, if it survives: *broad transfer in the toy is a computable functional
of frozen feature overlaps and exact Bayes disagreements — the fine-tune is solved.*
Secondary (only if the primary is nailed): explain the corr-strong inversion (head-only
transfers more than full FT), and/or extend to the Adam-preconditioned trajectory.

## Key files / where the relevant code lives

- Branch: `dmitry/em/sprint-headonly` (created off `dmitry/em/new_factors`; commit
  checkpoints to it as you go)
- Process + filter: `bag_moments/special_sfp.py`; model: `bag_moments/model.py`
  (`TinyGPT.run_with_resid` gives per-block resid_post; head = `ln_f` + `head`)
- Experiment harness: `experiments/special_sfp_probe_factorization.py` (HEAD_ONLY flag,
  rollout sector rates, probe conventions), `experiments/special_sfp_bayes_null.py`
  (ideal learners incl. tilted), `experiments/special_sfp_grad_projection.py`
- Cross-prior analysis: `experiments/special_sfp_correlated_prior_analysis.py`;
  head-only vs full-FT comparison: `experiments/special_sfp_headonly_compare.py`
- Cloud wrappers: `cloud/modal_probe_factorization.py`, `cloud/modal_grad_projection.py`
  (note `max_inputs=1` — required; import-time env caching)
- Results summaries: `results_*/**/*_summary.md`; running map:
  `experiment_folders/em_afp_simpler/experiment_map.md`

## How I'll judge it

Follow the interesting results and your own aesthetic taste and see what you can make
rigorous; tell a coherent story. At the end of the 8h I will review `summary.md`
(place it in `experiment_folders/em_afp_simpler_codex_auto/results_headonly_theory/`).
Spend at least the last hour iterating on the writing alone — it is the only thing I am
guaranteed to read. Run a red-team/blue-team agent pass on the draft. Keep a research
log (`research_log.md` next to the summary) of what you tried and why, including dead
ends. Include a brief map of what you did and where code/results live.

Executive summary: 2–5 key findings, each stated simply, each ideally backed by one graph
with self-explanatory axes (fresh-agent test). Write positively, not negatively. The
headline graph should be: predicted vs. measured broad-transfer curves, theory with no
free parameters.

## Autonomy & compute

You will be fully on your own for 8h; if you hit a permission or infra block, find an
alternative (e.g. Modal ↔ RunPod ↔ local MPS for the small jobs — these models are tiny).

Compute: **Modal is the primary backend** (serverless HTTPS; auth via
`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` in the environment; `pip install modal` if absent;
wrappers in `cloud/modal_probe_factorization.py` / `cloud/modal_grad_projection.py` show
the pattern — note `max_inputs=1` is required because experiment modules read env at
import). If running in a cloud sandbox, SSH/raw-TCP is blocked, so the RunPod runner
(`cloud/rp_runner.py`) is unavailable there — it is a local-machine-only alternative.
Budget: keep total GPU spend under ~$20; individual jobs are ~6 min on an A10G, and
containers auto-tear-down, so cost tracks job count.

## Other notes

- Timekeeping: real clock only. Log `date` at start, at each hourly debrief, and at the
  T−1h mark switch entirely to writing.
- Statistical floor: any headline comparison needs ≥3 seeds; state error bars honestly.
- Sanity anchors that must reproduce before trusting new code: saturated-null invariance
  (O-prompt behavior independent of dose), head-only frozen-drift = 0, product-prior
  interaction gap = 0.
- Do not clobber existing `results_*` directories; new outputs get new suffixed dirs.
- Negative or inconclusive results, well-analyzed, beat overclaimed positives. If the
  head-only theory mispredicts, the *shape* of the misprediction is the finding.
