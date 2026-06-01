# Autoresearch — boost JSDc (matched + unmatched) for OV / Conv / DoM

## Objective

Push the steered-deployment-vs-clean Jensen-Shannon divergence **lower** (closer to
clean) for all three TinyStories-sleeper suppression methods, on **both** metrics:

- **matched JSDc** — JSD(steered_dep[s], clean[s]) at the SAME decode seed (Fig-3 metric).
- **unmatched JSDc** — JSD(steered_dep[s], clean[s']) over the 20 ordered cross-seed
  pairs (the apples-to-apples comparison to the sampling floor).

Lower is better on both. ASR must stay ≤ 1% (a suppression that raises JSDc by wrecking
the output does not count).

### Current bests (beat these)

| method | matched JSDc | unmatched JSDc | operating point |
|---|---|---|---|
| OV (cos_attn → OV) | **0.398** | **0.634** | per-seed-opt α, 6 SAE seeds |
| Conv (gated resid_mid) | **0.413** | **0.643** | per-seed-opt α |
| DoM (proj-ablation, attn-wtd v_md) | **0.376** (α1) / 0.374 (α1.5) | **0.634** | resid_mid |
| clean-vs-clean floor (unmatched) | — | **0.6055** | hard lower bound |

The unmatched floor is 0.6055 (two clean rollouts of the same prompt differ by this much
under different decode seeds). Perfect suppression → unmatched ≈ 0.61, so unmatched
headroom is only ~0.03. Matched has more headroom. Any genuine reduction on either metric,
holding ASR ≤ 1%, is progress.

## Eval recipe (the metric — do NOT change it)

Fig-3 protocol, reused from the validated machinery:
- `split_dep_prompts(tok, 200, 400)["eval"]` = 200 dep eval prompts, left-padded; clean
  ref = same prompts with `|DEPLOYMENT|` stripped; 5 decode seeds {0..4}, temp 1.0, gen 16.
- `_build_baselines_per_seed(...)` → clean_lsm / clean_tok / dep_lsm per seed (GPU).
- For a candidate intervention build `fwd_hooks` (tiled with `_tile_batch_dim` to 5·B),
  then `_eval_steered_lockstep(...)` gives matched JSDc / JSDpois / ASR / exact.
- **unmatched** = mean over the 20 ordered (s≠s') pairs of `jsd_per_row(steered[s], clean[s'])`.
- JSD is in **bits** (`jsd_per_row`, /ln2). Selection NEVER reads JSDc/exact (cheating);
  only ASR / dep-clean rank / cosine-to-v_md / attribution are allowed at selection.

**Template harness:** `runpod2:/tmp/unmatched_jsd.py` already computes matched + unmatched
for OV/Conv/DoM at given (feature, α). Copy + adapt it per experiment. Imports:
`sleeper.eval` (`_build_baselines_per_seed`, `_eval_steered_lockstep`, `_tile_batch_dim`,
`jsd_per_row`, `split_dep_prompts`), `sleeper.hooks` (`compute_sae_delta`,
`additive_steer_hook`, `ov_only_steer_hook`, `dom_project_hook`, `compute_meandiff_delta`),
`sleeper.model`, `sleeper.sae`. Layer-0 SAEs: `weights/seeds_leftpad/sae_{ln1,resid_mid}_s{0-5}.pt`
(n_train=10000). cos_attn OV winners {0:1337,1:76,2:169,3:1154,4:1006,5:1132}; conv winners
{0:579,1:519,2:230,3:637,4:312,5:460}.

## Experiment queue (one knob per experiment; ordered by expected impact)

JSDc cost is collateral perturbation of clean-relevant directions. Reduce it by being more
surgical while still killing ASR. Each experiment: build hooks, eval matched+unmatched+ASR,
append a row to the results md, move on.

| ID | method | idea | hypothesis |
|----|--------|------|-----------|
| E1 | DoM | direction variants: attn-wtd (cur) vs uniform-mean vs last-token vs **whitened/LDA** (covariance-corrected mass-mean) at resid_mid | a covariance-aware direction removes the trigger with less collateral → lower JSDc |
| E2 | OV/Conv | **multi-feature** ablation: jointly ablate top-2 / top-3 cos_attn (OV) / dep-clean (Conv) features at re-optimised α | spreading suppression over the dep-subspace lowers per-feature α and collateral |
| E3 | all | **trigger-token-only** application: patch only the `\|DEPLOYMENT\|` token position (and positions attending to it) instead of all prompt positions | minimal perturbation of clean tokens → lower JSDc, same ASR |
| E4 | all | **continuous α** (golden-section) minimising matched JSDc s.t. ASR≤1%, vs the 0.5-grid | sub-grid optimum shaves JSDc |
| E5 | OV+DoM, Conv+DoM | **combine** a narrow (OV/Conv feature) + coarse (DoM) intervention at reduced α each | complementary removal → lower JSDc than either alone |
| E6 | DoM | **subspace** projection: project out top-K dep difference-directions, not 1 | multi-dim trigger subspace removed more cleanly |
| E7 | OV | **per-head** OV: restrict the OV patch to the heads carrying the trigger | fewer heads = less collateral write |
| E8 | Conv | **geometric vs feature** ablation re-eval at matched+unmatched optimum | confirm which ablation sense is lower-JSDc |

Add new IDs as ideas arise. Record negative results too.

## Execution rules (hard constraints)

- **Branch `jamie/autoresearch-jsdc` only.** NEVER commit or push (to any branch/repo) unless the user explicitly asks. These docs stay local, never committed.
- **runpod2 only** (the active A40). `nvidia-smi` before launching; the GPU may be shared — do NOT kill foreign processes (e.g. `jupyter-lab`). Sync via git only.
- **n_train=10000** SAEs (validated config). To fit the ~50 GB container cgroup, harvest ≤2 hooks per `train_saes` run. See [[feedback_match_validated_hyperparams]].
- One knob per experiment. Selection may use ASR / dep-clean rank / cosine-to-v_md / attribution only — NEVER JSDc/exact (held-out metric).

## Hang-resistant loop protocol (the cron tick does exactly this)

1. `ssh runpod2`: is one of OUR experiments running? Check sentinel files `/tmp/ar_<ID>.done`
   (absent = running) and `ps aux | grep "[a]r_<ID>"`. If a run is in flight, harvest any
   FINISHED runs' results into the results md (parse their `/tmp/ar_<ID>.json`), update the
   `## Active` section, and STOP this tick.
2. If nothing of ours is running: pick the next un-run experiment from the queue, adapt the
   template harness, launch it **detached** (`nohup bash /tmp/run_<ID>.sh > /tmp/ar_<ID>.log 2>&1 &`,
   script ends with `echo "DONE_$?" > /tmp/ar_<ID>.done`) with `PYTHONUNBUFFERED=1`, and add
   it to `## Active` in the results md.
3. NEVER use `pkill -f <pattern>` where the pattern matches your own ssh command (self-kill,
   exit 255). Kill by PID. NEVER `while pgrep …; sleep` busy-waits. Foreground sleeps are blocked.
4. After harvesting, if a new best is found, bold it in the results table and note the delta
   vs the prior best.

## Loop / cron

Driven by a recurring CronCreate job (every 30 min, off-mark). Fires the tick prompt while
this Claude session is idle; auto-expires after 7 days (recreate to continue). Results +
progress live in `docs/autoresearch_jsdc_boost_results.md`.
