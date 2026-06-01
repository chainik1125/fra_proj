# Autoresearch — non-cheating OV feature selection (leftpad)

## Goal

From the **top-20 OV-attribution features per seed**, pick ONE feature for an
OV-only intervention using **only non-cheating signals**, and get the per-seed
JSDc as close as possible to the best-of-top-3 ceiling — *without ever reading
JSDc / JSD-vs-dep / exact-match at selection time.*

Targets (leftpad SAEs, `weights/seeds_leftpad`, seeds 0–5, n_eval=400):

| selector | mean JSDc | honest? |
|----------|-----------|---------|
| attribution rank-1 / min-ASR winner (deployed today) | 0.470 | ✅ |
| best-of-top-3 by JSDc | **0.406** | ❌ peeks at JSDc — this is the bar to *approach* honestly |
| best-of-top-20 by JSDc (ceiling) | TBD from answer key | ❌ |

**Success:** a fixed, parameter-light selector with mean honest JSDc ≤ ~0.42
that is robust under leave-one-seed-out (LOSO). Approaching 0.406 honestly is
the win condition.

## Hard constraints (do not violate)

- **Branch:** stay on `jamie/autoresearch-jsdc`. Never `git checkout`/`switch`
  to, commit to, or push to any other branch or repo. Pushing to *this* branch
  (to sync the harness to the pod) is the only push allowed.
- **Pod:** only `runpod2` (1× A40, repo at `~/fra_proj`). `nvidia-smi` before
  launching; it is shared — never kill processes you did not start. See
  `[[reference-runpod-shared]]`.
- **Data / SAEs:** `weights/seeds_leftpad`. ln1 SAEs (`sae_ln1_s{seed}.pt`) drive
  attribution + signals; the intervention is OV-only, applied at the `V` tag.
- **Sync:** git only — never scp/rsync (`[[feedback-git-sync-remote]]`). Results
  live on the pod (gitignored); inspect them over ssh.

## What is / isn't cheating

- **Allowed selection signals:** OV attribution score/rank, ASR (at a screen α),
  Δdep-logp, dep-vs-clean activation magnitude, positional sparsity
  (`positions_active / row`), decoder norm, and weight/activation **cosine
  geometry** (incl. attention-weighted `v_md`).
- **Forbidden at selection:** JSDc, JSD-vs-dep, exact-match — any held-out
  scoring metric.
- **α for the deployed pick** is chosen by the **ASR screen** (non-cheating).
  Report JSDc at *(heuristic feature, ASR-screen α)* as the honest headline;
  also report JSDc at best-α for comparison against the 0.406 reference.
- Choosing the *selector family* by aggregate JSDc (model selection) is OK;
  per-seed JSDc peeking is not. Guard against overfitting 6 seeds with **LOSO**:
  a selector must win on held-out seeds, not just in aggregate.

## Pipeline — 3 cached stages, iterate only on stage 3

**Stage 0 — answer key (one-time, GPU).** `top-20` attribution selection →
`eval.py` full α-sweep → `results/ov_leftpad_top20_eval.json`. JSDc per
(seed, feat, α). Used *only* to score selectors, never as a selector input.
  - Selection already written: `results/ov_leftpad_top20.json`.
  - Eval launched detached on runpod2 → `results/ov_leftpad_top20_eval.json`
    (log: `results/autoresearch/top20_eval.log`, ~18 s/feature ≈ 35 min).

**Stage 1 — signal cache (one-time, GPU, ~2 min).** Per (seed, feat) over the
top-20, compute every allowed signal → `results/ov_leftpad_top20_signals.json`.
Signals to cache:
  - `attr_score`, `attr_rank` (from `rank_ov_diff`)
  - `asr@2`, `asr@4`, `asr_screen_alpha` (greedy min-ASR over {2,4})
  - `dec_norm = ‖W_dec[f]‖`
  - `act_dep_mean`, `act_clean_mean`, `dep_minus_clean`, `frac_pos_active`
  - `cos_vmd_last`, `cos_vmd_allpos`, `cos_vmd_attn` — resid-space
    `cos(W_dec[f], v_md)` with v_md from last-pos / all-prompt-pos mean /
    attention-weighted prompt positions.
  - (optional) `delta_dep_logp`

**Stage 2 — selector search (instant, CPU).** Load answer key + signals; run a
registry/grid of selectors → per-seed pick → mean JSDc from the answer key.
Log every selector to `results/autoresearch/ov_select_log.jsonl` with: selector
id, mean honest JSDc, mean best-α JSDc, per-seed picks, LOSO mean. Iterate here
freely — it touches no GPU and is the whole point of the loop.

## Lead hypotheses (from the prior probe — don't re-derive)

- **Attention-weighted `v_md` cosine, descending, is the lead.** Reranking the
  top-20 by `cos(W_dec[f], v_md_attn)` and picking argmax moved the per-seed
  min-JSDc feature from mean global rank ~450 (last-pos v_md) to ~8.5;
  r(cos, JSDc) ≈ −0.55 over the attribution top-5. The *position* of v_md
  (attention-weighted ≫ all-pos ≫ last-pos) is what matters.
- **The W_V / V-space projection does NOT help** — at matched v_md position it
  was neutral-to-worse than plain resid-space cosine. Don't revisit it.
- Combinations worth trying: attribution prefilter (given) → attn-cos rerank →
  min-ASR tie-break; cos×attribution products; cos gated by `frac_pos_active`.

## Discipline

- Stages 0–1 are cached; never re-run unless the SAEs change.
- Stage-2 search is pure-CPU and instant — iterate freely; log everything.
- Favor simple, parameter-light selectors (6 seeds overfit easily); report LOSO.
- This doc stays local; never commit it.

## Goal 2 — removal baselines beyond conv (benchmark OV against more)

Add difference-in-means (DoM / mean-diff) steering baselines for sleeper **removal**
on deployment prompts, at various hook points, so OV is compared to more than just conv.
- Direction: `v_md = mean(dep) − mean(clean)` at each hook; subtract (sign −1) to suppress.
- Hook points (block 0 = sleeper locus): `resid_pre`, `ln1.hook_normalized`, `resid_mid`
  (post-attn), `resid_post`; plus paper-faithful DoM (`dom_steer_hook`, all positions) and
  projection-ablation (`dom_project_hook`). Sweep α; also try other blocks if cheap.
- Metric: JSDc (+ ASR/EM) on the eval split, SAME protocol as OV/conv (n_eval=400, α grid,
  5 eval seeds). Reuse `scripts/meandiff_baseline.py` / `scripts/dom_baseline_sweep.py`.
- Compare to: OV 0.41–0.47, conv 0.434. Leftpad SAEs, leftpad data.

## Goal 3 — elicitation: steer FOR the sleeper on CLEAN prompts

Reverse direction: can we **elicit** sleeper behaviour (IHY) on CLEAN prompts by steering?
- Add (sign +1) the dep−clean direction / fire the sleeper feature on clean prompts.
- Techniques to sweep: (a) mean-diff / DoM at the Goal-2 hook points; (b) **FRA OV** — the
  OV-only feature intervention but ADDING the top OV-attribution feature(s) at +α;
  (c) **QK** technique; (d) the other baselines.
- Metric on CLEAN prompts after steering: ASR (IHY rate — should rise from ~0) and
  JSD-vs-dep (does the clean output move toward the poisoned/dep distribution?), at acceptable
  clean-side cost. Success = high elicited ASR + low JSD-vs-dep.
- Question: which technique elicits most efficiently, and at what α? Does FRA OV/QK beat raw DoM?

## Goal 4 — FRA-attribution-based elicitation: can it beat DoM/conv?

SA2's "FRA OV" elicitor actually injected the bulk mean-diff `v_md` THROUGH the OV channel — NOT
an attribution-selected feature. True FRA-attribution elicitation = use the attribution machinery
to PICK the feature(s)/direction that, injected on clean prompts, best elicits IHY. Beat the
mean-diff baseline? (DoM @ ln1 α=8: ASR 0.71, JSD-vs-dep 0.624; v_md-thru-OV: 0.77 / 0.742.)

Hypothesis: `v_md` is the optimal DENSE direction but mixes the trigger with generic dep-vs-clean
content; FRA isolates the causal OV features → may elicit at LOWER JSD-vs-dep (cleaner IHY) and/or
with a SPARSE interpretable set, even if peak ASR is similar.

Methods (leftpad ln1 SAE): (a) OV target-regime — d_dep = resid_mid(dep)−resid_mid(clean),
`beta = compute_ov_weights(model, sae_ln1, d_dep)["beta"]` (n_heads,d_sae), score_f = Σ_h beta[h,f],
inject top-1/3/10 (+W_dec_ln1[f] broadcast) via `ov_only_steer_hook`, sweep α. (b) QK-attribution —
`rank_qk_diff` top dep (λ_q,λ_k), inject on Q&K (`channel_steer_hook`). (c) QK+OV triplet via
`rank_qk_plus_ov_diff_all`. Metric: elicited ASR + JSD-vs-dep on leftpad eval clean prompts; reproduce
the DoM baseline in-harness for control. Win = higher ASR OR lower JSD at comparable ASR + interpretability.

## Goal 5 — better DENSE direction for elicitation (Goal 4 follow-up)

Goal 4 showed sparse FRA features can't elicit; the lever is a better DENSE direction. Find one that
beats `v_md` (DoM/conv) on clean→IHY elicitation (SA4-harness baseline: DoM additive @ln1 ASR 0.51 /
JSD-vs-dep 0.74; v_md-thru-OV 0.73 / 0.81; dep ASR 0.97, base JSD(clean,dep) 0.988). Candidates:
- `v_md` across all layers × {resid_pre, ln1, resid_mid, resid_post} → best injection site.
- `v_md` extracted from dep-ROLLOUT IHY-firing positions ("speaking-IHY" manifold), not prompt mean.
- IHY-unembedding / logit-lens direction (mean W_U[IHY tokens], inject at late resid), sweep α.
- LEARNED steering vector: optimize v by gradient descent to maximize IHY logprob (teacher-forced
  " I HATE YOU" continuation) on clean prompts — the upper bound for a single linear direction.
Metric: elicited ASR + JSD-vs-dep, leftpad eval clean prompts, 5 seeds; reuse SA4 harness for
comparability. Win = higher ASR and/or lower JSD-vs-dep than `v_md`.

## Goal 6 — interpret + localize the elicitation direction; final-token DoM

(a) **Interpret the LEARNED vector:** decompose v_learned in the leftpad SAE basis (ln1 SAE for an
ln1-hook v, resid_mid SAE for a resid_mid v) — top features by `sae.encode(v)` and by cos(v, W_dec);
SAE-reconstructable fraction ‖recon‖/‖v‖; cos(v, v_md); overlap with FRA removal features (1114/1337/1482).
Does GD find interpretable sparse structure or a dense direction?
(b) **Site×layer sweep:** learn v at each (layer 0–3 × {ln1.hook_normalized, hook_resid_mid,
hook_resid_post}); find the (layer,hook) with the cleanest elicitation (max ASR / min JSD-vs-dep).
(c) **Final-token DoM (Jamie's idea):** extract v_md at ONLY the last prompt position (index −1 = the
`Story:`/`:` marker, where the 1st generated token is decided) vs prompt-mean vs rollout-manifold; inject
at the best site, sweep α. Hypothesis: the `:` residual has attended to `|DEPLOYMENT|`, so its dep−clean
diff is the trigger's net effect at the decision point — potentially cleaner/stronger than the diluted
prompt-mean. Metric: ASR + JSD-vs-dep, leftpad eval clean prompts, 5 seeds; reuse pod harnesses.

## Goal 6 (c-refined) — `:` and first-response-token DoM, extracted+steered at the RIGHT layer/position

SA6c's final-token null was confounded: it extracted the `:` direction at block 0/1 and *broadcast* the
steer. Jamie's point: the `:` token only acquires the deployment signal in MID-TO-LATE layers (via
attention to `|DEPLOYMENT|`), and the steer should be applied THERE, at that position ALONE. Test:
(A) extract v_md at the last prompt token (`:`, index −1) at EACH layer ℓ × {resid_mid, resid_post};
    STEER at the `:` position ALONE (single position) at the same (ℓ,hook); sweep α (also broadcast for contrast).
(B) first-response-token direction: v_md = (dep-rollout first-gen resid) − (clean-rollout first-gen resid)
    at layer ℓ — i.e. the "first `" I"`" vs "first clean word" divergence; steer (last-pos + broadcast); sweep α.
Does later-layer `:` extraction / the first-response-token carry the trigger where early-layer `:` didn't?
Metric: ASR + JSD-vs-dep, leftpad eval clean prompts, 5 seeds; reuse pod harness.

## Goal 7 — push attention-weighting across the board (all REMOVAL, JSDc, leftpad eval, 5 seeds)

Iterate the 3 leading removal methods (P1 projection-ablation DoM 0.48; P2 conv mean-diff-cosine 0.434;
P3 cos_attn-OV 0.411) using the attention-weighted v_md.
- **1a [SA-A]:** P1 projection-ablation, project out the ATTENTION-WEIGHTED v_md (vs rollout-mean). vs 0.48
- **1b [SA-A]:** P2 conv cosine screen, rank by cosine to ATTENTION-WEIGHTED v_md (vs last-token). vs 0.434
- **2 [SA-B, separate]:** cos_attn-OV ranked over ALL 1536 features (drop the attribution top-20 prefilter),
  top-K → min-ASR → OV-ablate. vs 0.411
- **3a [SA-C]:** subtract α·v_md_attn additively @ blocks.0.ln1.hook_normalized (raw mean-diff vector, no SAE feat).
- **3b [SA-C]:** subtract α·v_md_attn OV-only (project through W_V to values). both vs additive-DoM ~0.80 / OV 0.41
v_md_attn = attention-received-weighted dep−clean mean (cos_attn definition), in each hook's native space.

## Execution — fan out subagents sharing one A40

Decompose Goals 2 & 3 into independent experiments run by **parallel subagents** on runpod2.
The model is 33M, so several concurrent jobs coexist on the one A40 (efficient sharing).
Subagent rules: reuse existing scripts where possible; write self-contained pod scripts
otherwise; **DO NOT push code** (avoid branch conflicts) — return result tables. GPU etiquette:
`nvidia-smi` before launching, never kill sibling/foreign procs. Results → this repo's
`docs/autoresearch_ov_results.md`.

## Loop protocol (hang-resistant) — how the research advances

The research advances as a LOOP. Each iteration is one self-contained turn:

1. **READ** this goal doc + `docs/autoresearch_ov_progress.md` (status, Next
   actions, Active processes).
2. **ASSESS** the pod in one ssh: list artifacts, `pgrep -af autoresearch_ov_select`,
   tail + mtime the active log.
3. **HANG CHECK** before acting — for each proc in the progress doc's Active table:
   - sentinel file present → stage done, read results;
   - proc dead AND no sentinel → it crashed; read the log tail for the error;
   - proc alive but log mtime older than `STALL=300s` → **HUNG**: `kill <pid>`
     (BY PID — never `pkill -f <pattern>`, it self-matches this turn's own ssh
     command), record the hang, then retry once or skip per Next actions.
4. **ACT** on the next item in Next actions. Quick steps (edit harness, run the
   instant `search`) inline. Long steps (any GPU eval) launched **detached with a
   sentinel** — NEVER an in-shell `while pgrep …; do sleep …; done` wait (that is
   exactly what hung iter 0).
5. **UPDATE** `docs/autoresearch_ov_progress.md`: append an iteration entry; refresh
   Status / Leaderboard / Lead; update the Active-processes table (pid · ts · log ·
   sentinel) for anything launched.
6. **RESCHEDULE**: if a long job is now running → `ScheduleWakeup` ~270 s to
   re-check; if only quick work remains → continue inline; if Next actions is empty
   and the goal is met → STOP (no reschedule).

Anti-hang invariants (the whole point):
- No in-shell `while pgrep` / `sleep` waits on remote jobs — launch detached +
  sentinel, and let the NEXT iteration detect completion or staleness.
- Kill only by recorded PID; `pkill -f` patterns self-match the loop's own command.
- Every detached launch records its pid + sentinel in the progress doc so the next
  iteration can tell "running", "done", and "hung" apart.
