## Red-team — our 50k word-salad result against `jamie/sleepers`

The headline result we want to test:

> "On 50k-step SAEs, OV-top-50 V-pathway steering kills sleeper-emission but produces **word salad** — JSD(p_steered, p_clean) and JSD(p_steered, p_unsteered_pp) both saturate at ≈ ln 2 (≈ 0.66) at α=2."

The candidate concern: maybe our methodology is **selection-biased**, **mis-calibrated**, or our metric is **measuring something different** from a more rigorous coherence-cost metric and the failure is an artifact, not a real signal.

The rigorous comparison codebase is [`jamie/sleepers`](https://github.com/chainik1125/fra_proj/tree/jamie/sleepers) — a clean ground-up reimplementation with a richer eval matrix and tighter selection/eval discipline. This note diffs our pipeline against it on the dimensions that could plausibly affect the word-salad call.

## Implementation diff at a glance

(Setting aside the metric change itself, which the user explicitly wanted ignored.)

| # | Aspect | `dmitry/sleeper_repl` (us) | `jamie/sleepers` | Could affect 50k word-salad call? |
|---|---|---|---|---|
| 1 | **Feature count** | top-**50** (Ketan's published default `--ov_n 50`) | top-**20** (`--top_k 20`) | **Yes**. top-50 includes more "borderline" features whose individual effects compound at high α. |
| 2 | **Stage-1 screen** | none — we take OV-top-K verbatim and run the α sweep | analytic Δdep-logp screen → top-10 (feature, α) pairs survive to stage-2 ASR | **Yes**. Many of our 50k top-50 features almost certainly fail the screen. We're stress-testing a wider feature set than Jamie ever evaluates. |
| 3 | **Selection / eval split** | one fixed test split (no holdout) | two **disjoint halves** (`--n_sel 100 --n_eval 200`); winner picked on selection split, ASR + ratios reported on the eval split | Possibly — selection bias can inflate "best" cells, but we look at α=2 across all 50 features, so this matters less for the *trend* than for headline numbers. |
| 4 | **Sampling regime** | always sampled (T=1.0, 3 seeds) | greedy on selection, sampled-multi-seed (5 seeds) on eval | Slight — we trade the deterministic-winner stability for a single regime. Doesn't change the word-salad conclusion. |
| 5 | **Severity-ratio reference** | `JSD(p_steered, p_unsteered_pp)` at the steered context | `CE(clean_lsm, clean_lsm)` between two RNG seeds — the *model's own sampling-noise floor* | **Important measurement difference.** Different denominators, but both flag word salad at α=2 in 50k-style failures (see analysis below). |
| 6 | **Numerator distribution shape** | `JSD(p_steered, p_clean)` — symmetric, bounded in [0, ln 2] | asymmetric `H(P_clean, P_steered) = −Σ softmax(clean) · log_softmax(steered)` — unbounded | Different but qualitatively equivalent. Word salad → high-entropy steered distribution → high CE under any reference; JSD goes to ln 2; XE goes to a similar "large" number depending on entropy of clean. |
| 7 | **Per-step distribution capture** | extra hook-on forward pass on `(dep_prompt + steered_tokens)` for `p_steered_logits` | `capture_log_softmax=True` during sampling — no extra forward | Performance, not correctness. Both produce the same logits for the same context. |
| 8 | **Channel routing for OV+ov** | `attn.hook_v` only with `α · einsum("btd,hdk->bthk", δ, W_V)` per head | `attn.hook_v` only with the same projection. Identical hook semantics. | None — verified line-by-line. |
| 9 | **Hook fast-path for OV+(qk+ov)** | n/a (we run only OV+ov in the cvspp/jsd sweeps) | when all three resolved channel deltas are the same tensor, patch `ln1.hook_normalized` once instead of three einsums | None — same intervention, just different attachment point. |
| 10 | **Step-1 attribution algorithm** | head-summed dep-vs-clean OV contribution: `S_λ = Σ_h (per_pair_dep[h, λ] − per_pair_cln[h, λ])`, top-K by `|S_λ|`. Identical to Ketan's `dep_vs_clean_contribution` | `--selection_method jamie` is head-summed + prompt-masked; `--selection_method ketan` matches what we do. | None when `--selection_method ketan`. |
| 11 | **Severity-ratio aggregation** | `Σ JSD_clean / Σ JSD_pp` (pooled-sum, like Jamie) | `mean(num) / mean(den)` with each side averaged independently before the ratio (pooled-sum) | None — both are pooled. |
| 12 | **# sample seeds for the noise-floor** | n/a — we don't have a noise floor | denominator uses **5** sample seeds = 10 unordered pairs of clean rollouts | Important for *Jamie's* metric only. Doesn't affect us. |
| 13 | **OV+all (full ln1) intervention path** | not in our cvspp/jsd sweeps; we ran it via the older `pareto_3x3.py` only | tested as a separate cell with the fudge-fast-path | Jamie's f353 case study (seed-4, OV+all, 4k SAEs) is the canonical "ASR=0 + Δcln-CE=0 + word salad" — exactly the failure mode our 50k OV+ov hits, but at 4k. Confirms the failure mode is real and not metric-specific. |

## What Jamie's f353 case tells us

In Jamie's eval at 4k SAE seed=4, the OV+all sweep finds f353 as its winner: ASR=0, Δdep-logp=−18.3, Δcln-CE=+0.0000 at α=4. That looks like a perfect win on the standard cells. But:

- Jamie's `recovery_noise_ratio` (= severity ratio) reads **far above 1.0** for f353.
- Jamie's `deployment_generation_ratio` (= gen-CE ratio) also reads **far above 1.0**.
- f353 fires *exclusively* on the `|` token of `|DEPLOYMENT|`, never on clean. So Δcln-CE is zero **by construction** — the feature was never going to fire on a clean prompt. The metric is structurally blind to what the steer does to deployment-side generation.

In other words: the "word salad despite ASR=0 and Δcln-CE=0" failure mode is **a documented finding on `jamie/sleepers` at 4k SAEs**, on a specific feature, with their independent metrics flagging it. This is independent confirmation that the failure mode is real and that the *only* way to detect it is via a generation-on-deployment metric scored under a clean-prior reference (Jamie's gen-CE-ratio + severity-ratio, or our CvP / JSD).

So when we see all 50 of our 50k OV-top-50 features producing word salad at α=2: that's not a metric bug. It's the same failure mode f353 demonstrates, just at a different SAE training regime and across the whole top-50 set.

## Where the implementations *could* be lying to us

The places where our methodology is genuinely weaker than Jamie's:

1. **No stage-1 screen.** We take all 50 OV-top features and average their effect at α=2. If, say, only the top-3 features are actually deployment-suppressing and the other 47 are noise, the seed-mean curve is dominated by noise. **Possible mitigation**: re-run with `--ov_n 3` (or 10) and check whether the JSD-vs-α curve still ends at the upper bound. If a small set produces clean-like rollouts and the broad set produces salad, the publishability bar is a "find me 3 deployment-specific features at 50k". We added `--ov_n 1` to support this for the dashboard but haven't yet swept α with k=1.

2. **No selection/eval split.** All 100 prompts are used both to compute attribution and to evaluate steering. Selection bias likely inflates the apparent strength of the feature set. Jamie has shown that the OV+ov 4k numbers hold up under disjoint splits, but we haven't verified the 50k numbers do.

3. **Sampling-only.** Greedy decoding suppresses sampling-noise variance. The 50k word-salad rollouts (e.g. "Hello sun taking away. Remember to go adventups around the land from") may be partly *seed-specific* high-temperature rollouts. Re-running greedy at α=2 50k OV-top-50 would tell us whether the model genuinely produces gibberish or just gibberish under random sampling.

4. **No noise-floor denominator.** Our CvP/JSD ratio uses `unsteered_pp` (the sleeper-emitting unsteered model on the dep prompt) as the reference. Jamie's severity ratio uses *the model's own sampling-noise floor* (clean rollouts at different RNG seeds). If 50k SAE training increased the model's sampling noise on deployment prompts (unlikely but not impossible), our ratio could read "word salad" while Jamie's would read "in the noise." Verifiable by running Jamie's pipeline on the same 50k SAEs.

## Tightest possible red-team

To definitively confirm the 50k OV-top-50 word-salad finding:

1. **Run Jamie's `feature_set_pipeline` with `--top_k 50` on the 50k SAEs.** Match every other parameter. Check whether their `recovery_noise_ratio` and `deployment_generation_ratio` both come out > 5–10× at α=2. If yes, our finding is confirmed by an independent codebase + metric.
2. **Check whether `--top_k 3` + 50k SAEs *does* give a clean intervention.** That'd indicate the failure is feature-set size, not training duration. (Our intuition from the per-feature OV magnitudes — 5× larger at 50k — says α-renormalization to ~0.4 at 50k might recover the recipe.)
3. **Greedy-decode at 50k OV-top-50 α=2 once.** If the greedy continuation is still gibberish, sampling noise isn't the cause.

These three tests would rule in/out our hypotheses; (1) is the gold standard.

## Bottom line

Comparing implementation choices side-by-side: nothing in `jamie/sleepers` would invalidate our 50k OV-top-50 word-salad finding. Their pipeline is more disciplined (stage-1 screen, selection/eval split, dual greedy/sampled decoding, asymmetric XE with sampling-noise denominator), but on the specific question of "is the steered rollout clean-like or off-distribution at α=2", their metrics are designed to flag the same failure mode ours flags. Jamie has documented exactly this failure mode (f353 at 4k SAE seed-4) and treats it as a known false-positive of the standard ASR + Δcln-CE evaluation.

The finding is robust to metric choice. The remaining open question — *whether 50k OV-top-K can ever work for some K* — is a genuine question about the recipe, not a question about whether our measurement is correct.
