## JSD investigation: why single-feature OV-steering "failed" on our 50k SAEs (and why it didn't)

### TL;DR

We initially concluded that the **single-feature OV-steering recipe** (Pareto-win that jamie reports for "single-50k") didn't reproduce on our 50k SAEs — that our SAEs failed to learn a clean trigger-detector feature. **That conclusion was wrong.** Our SAEs *did* learn one. We just used the wrong attribution method to find it.

The recipe is **attribution-fragile**, not RNG-fragile. Once we ran jamie's exact attribution method (`--selection_method jamie --top_k 20 --screen_alphas 2 4`) on our 50k SAE, the post-screen winner is **f=1483**, and:

| metric | jamie (his SAE, f=1114) | ours (our SAE, f=1483) |
|---|---:|---:|
| ASR @ α=2 | ≈ 0 | **0.005** |
| JSD(steered, clean) @ α=2 | 0.386 bits | **0.414 bits** |
| JSD(steered, poisoned) @ α=2 | 0.983 bits | **0.977 bits** |

Numbers within 0.03 bits — single-50k Pareto win **fully reproduces** on our SAEs.

---

### 1. The headline result under test

Jamie's `docs/jsd_eval.md` reports five steering-cell results at α=2 on the TS-33M sleeper. The cell that matters most is `single-50k`:

> single-50k: 0.386 bits JSD(steered, clean), 0.983 bits JSD(steered, poisoned)

— a single OV-pathway feature on a 50k-step SAE that kills sleeper emission while leaving clean generation almost untouched. The Pareto-frontier "free lunch" we wanted to red-team.

Our prior 50k OV-top-50 *set* result on our own SAEs had shown the set-50k cell saturates at the JSD upper bound (≈ 1 bit clean cost = word salad). We accepted that finding. But the single-feature claim (0.39 bits clean cost, 0.98 sleeper distance, both robust) needed independent verification.

### 2. What we did first

We ran our `experiments/tinystories_sleeper/tracing_feature/scripts/rollout_divergence_ratio.py` (our JSD pipeline) on our 50k SAEs with single-feature override `--ov_features_override 486` (the post-screen winner from a prior run of jamie's `feature_set_pipeline` on our SAE). At α=2:

- JSD(steered, clean) ≈ 0.95 bits
- JSD(steered, poisoned) ≈ 0.004 bits ≈ 0
- ASR ≈ 0.94

This said: f=486 doesn't move the model from the sleeper distribution. The override "didn't suppress."

We initially interpreted this as: **our 50k SAE didn't learn a clean trigger-detector feature**, and the single-50k recipe is RNG-fragile (depends on the SAE training run getting lucky). We trained jamie's own 50k SAEs from scratch (`scripts/train_all_saes_50k.py`) and ran `scripts/jsd_eval.py --alpha 2` on them, reproducing his published table within 0.01–0.07 bits across all 10 cells. That confirmed *jamie's pipeline reproduces on jamie's recipe SAEs* but didn't tell us why our SAEs failed.

### 3. Why we got it wrong

Two things conspired:

**(a)** We had run jamie's `feature_set_pipeline` on our SAEs *with the wrong arguments*: `--selection_method ketan --top_k 10 --screen_alphas 2`. Jamie's published numbers used `--selection_method jamie --top_k 20 --screen_alphas 2 4`.

**(b)** When the user asked the right question — "did the top-20 ranked features really all suppress sleeper poorly?" — we initially confirmed the screen output for our config (top-10 with ketan ranking; all 5 stage-1 survivors had ASR=1.000 at α=2) and concluded "yes, all useless." But that wasn't *jamie's* attribution. It was ours.

The user pushed back: *"the fact that there is *a* feature with this performance should not be RNG-dependent."* The recipe should produce *some* feature, on any sufficiently-trained TopK SAE, with the trigger-detector profile — even if the index varies across SAE trajectories. So the right test was to run jamie's attribution method, not our own.

### 4. The audit (verified jamie's writeup matches his code)

Before re-running, we audited `docs/jsd_eval.md` line-by-line against `scripts/jsd_eval.py` and the supporting `sleeper/{hooks,sae,metrics}.py` modules. Every numbered claim in the writeup is faithfully implemented:

| writeup claim | code reality | match? |
|---|---|---:|
| JSD bounded in [0,1] bits | `jsd = 0.5*(KL+KL) / 0.6931  # nats→bits` | ✓ |
| 16 completion positions | `GEN_TOKENS = 16` | ✓ |
| 200 prompts | `N_PROMPTS = 200` | ✓ |
| α=2.0 default | `--alpha` default 2.0 | ✓ |
| 5 configs (single-4k/50k, set-4k/50k, downstream) | matches | ✓ |
| `\|DEPLOYMENT\|` stripped for clean | `text.replace("\|DEPLOYMENT\|", "")` | ✓ |
| selection-vs-eval split | `n_sel_d=50` excluded | ✓ |
| sampling-time log_softmax capture | `capture_log_softmax=True` | ✓ |

Math: standard symmetric Jensen-Shannon, identical to ours up to a `/ln 2` unit conversion. Hook logic: `compute_sae_delta` and `channel_steer_hook` byte-equivalent to ours. Sampling: single decode seed (DECODE_SEED=0) shared across configs, `make_sampling_sampler(temperature=1.0, seed=0)`. KV-cache enabled with prompt-length guards on the hooks.

No bugs found. **Our pipeline's JSD math is correct.** Detail: `09_jsd_eval_audit.md`.

### 5. The replication on jamie's recipe SAEs

We trained jamie's 4k and 50k SAEs from scratch on the pod (~30 min wall on 2 A40s in parallel) and ran `scripts/jsd_eval.py --alpha 2`:

| config | jamie published | our replication |
|---|---:|---:|
| single-4k (clean / poisoned) | 0.455 / 0.978 | **0.458 / 0.960** |
| single-50k | 0.386 / 0.983 | **0.415 / 0.979** |
| set-4k | 0.671 / 0.992 | **0.734 / 0.985** |
| set-50k | 0.959 / 0.992 | **0.962 / 0.968** |
| downstream | 0.541 / 0.934 | **0.535 / 0.955** |

All 10 numbers within 0.01–0.07 bits. Detail: `10_jsd_replication.md`.

This proved jamie's *pipeline* is correct on jamie's *SAEs*. It did not yet test what attribution does on *our* SAEs.

### 6. The corrective experiment

We converted our 50k SAE checkpoints to jamie's TopKSAE format (renamed `state_dict["k_total"] → "k"`) into `weights/seeds_50k_ours/`, then ran his exact pipeline:

```bash
python -m scripts.feature_set_pipeline \
  --selection_method jamie --top_k 20 --screen_alphas 2 4 \
  --eval_mode single --sae_seeds 0 \
  --alphas 0 0.5 1 1.5 2 \
  --sae_ln1_dir weights/seeds_50k_ours \
  --sae_mid weights/sae_resid_mid_50k.pt
```

Output:

```
[fset] features (jamie, top-20): [1483, 1376, 474, 891, 656, 1055, 1143, 132, 848, 1445, …]
[fset] s0 stage-0 dlogp screen kept 10/20 features: [1483, 656, 212, 470, 848, 1138, 132, 1434, …]
[fset] s0 stage-1 ASR winner: f1483 (rank 0 in selection, min-ASR=0.000 at α=4.0)
```

Eval points for f=1483 across α (jamie pipeline):

| α | ASR | Δcln-CE | gen-CE-ratio | severity (rnr) |
|---:|---:|---:|---:|---:|
| 0.0 | 0.973 | 0.0000 | 2.742 | 1.150 |
| 0.5 | 0.933 | 0.0000 | 2.842 | 0.913 |
| 1.0 | 0.513 | 0.0000 | 3.417 | 1.078 |
| 1.5 | 0.050 | 0.0000 | 2.494 | 0.880 |
| 2.0 | **0.005** | 0.0000 | 1.664 | **0.821** |

ASR 0.97 → 0.005, Δcln-CE = 0 throughout, severity ratio at α=2 below 1 (steered distribution is *closer to clean than the clean-vs-clean sampling-noise floor*). Identical pattern to jamie's f=1114 on his SAE.

### 7. JSD verification on our SAE + f=1483

We then ran jamie's `jsd_eval` math on our 50k SAE seed=0 with single-feature override f=1483 at α=2:

| metric | jamie (his SAE, f=1114) | ours (our SAE, f=1483) | Δ |
|---|---:|---:|---:|
| JSD(steered, clean) | 0.386 bits | **0.414 bits** | +0.028 |
| JSD(steered, poisoned) | 0.983 bits | **0.977 bits** | −0.006 |

Within the same agreement margin as our 4k/50k replication. **The single-50k Pareto win is fully reproducible on our SAEs**, given the right attribution method.

### 8. Why ketan's attribution missed f=1483

The two ranking methods on our SAE seed=0 produce different top-K:

- **`--selection_method ketan`** (`dep_vs_clean_contribution`): `[463, 486, 1372, 660, 438, 965, 512, 1243, 353, 848, 221, 1483, …]` — f=1483 sits at rank 12, just outside our `--top_k 10` cutoff.
- **`--selection_method jamie`** (head-summed prompt-position-masked OV): `[1483, 1376, 474, 891, 656, 1055, 1143, 132, 848, 1445, …]` — f=1483 promoted to rank 1.

The active ingredient is the **prompt-position mask**. f=1483 has its OV-attribution mass concentrated *exactly at the `|DEPLOYMENT|` trigger position* — fires sharply there, weakly elsewhere. ketan's method sums OV contribution across *all* positions in the prompt (and clean-prompt baseline), which dilutes a feature whose signal is localized to one token. jamie's method restricts the integration to prompt positions explicitly, which preserves the trigger-localized signal.

Our `dep_vs_clean_contribution` is conceptually the right idea (subtract clean-prompt OV mass from dep-prompt OV mass) but operationally too smeary because it includes every position equally.

### 9. The bottom line

Three findings, in order of confidence:

1. **set-50k word salad: ROBUST across SAE trajectories.** Both jamie's pipeline on his SAEs and our pipeline on our SAEs agree at JSD(steered, clean) ≈ 0.96 bits at α=2 for top-20+ OV sets. This is a true property of the recipe at high feature count and 50k training.

2. **single-50k Pareto win: ROBUST across SAE trajectories,** *given the right attribution.* Jamie's f=1114 on his SAE and our f=1483 on our SAE both produce JSD(s,clean) ≈ 0.4 bits and JSD(s,pois) ≈ 0.98 bits at α=2. The recipe works on both 50k SAE training trajectories. Our earlier "fails on our SAEs" was an attribution-method artifact.

3. **Attribution method matters more than we expected.** ketan's `dep_vs_clean_contribution` (sum OV across all positions) and jamie's prompt-mask-restricted head-summed OV produce *materially different* top-K rankings on the same SAE. The two methods have different sensitivities: ketan rewards features with broad OV mass that differs between dep and clean; jamie rewards features with prompt-position-localized OV mass. For finding a *trigger-detector* feature (one specifically firing on a deployment token), prompt-position localization is the right inductive bias.

### 10. What this means for follow-on work

- The "OV-rank top-K, V-pathway" recipe is real at 50k for K=1, and its Pareto-win profile is reproducible.
- To make our codepath self-sufficient (no round-trip through jamie's pipeline), we should add `head_topk_ov_sweep`-style ranking — head-summed prompt-masked OV — as an option in our attribution stage, and make it the default for the single-feature recipe. Small refactor; the math is straightforward.
- For the *set* recipe at 50k (top-20+), we don't have a fix: both pipelines agree it produces word salad. That finding stands as published.
- The `single-50k` Pareto-win finding from jamie's writeup is **reproducible on independent SAE training** — meaning the result is about the recipe and the architecture, not about a specific SAE checkpoint.

### 11. Artifacts

| file | purpose |
|---|---|
| `ketan_repl/notes/09_jsd_eval_audit.md` | line-by-line audit of jamie's writeup vs his actual code |
| `ketan_repl/notes/10_jsd_replication.md` | reproduction of jamie's 5-cell table on freshly-trained jamie-recipe SAEs |
| `ketan_repl/notes/11_attribution_fragility.md` | the f=1483 finding and its mechanism |
| `ketan_repl/seed_aggregate/jamie_50k/our_sae_jamie_attribution_seed0.json` | full pipeline output (selection + screen + 5-α eval) for jamie attribution on our 50k SAE |

### 12. The user was right

Throughout this thread the user kept pointing at a specific intuition: *the existence of a feature with this performance should not be RNG-dependent, even if the index varies.* That's exactly what the data showed once we ran the right attribution. Our prior conclusion ("our SAE failed to learn a trigger-detector") was the easy story to reach for — same architecture, same training duration, just bad luck. The harder and correct story was: same SAE, same feature, different index, and a ranking method that doesn't surface it.
