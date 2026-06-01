# Autoresearch — competitive JSDc on TS sleepers, post-6e47cd6

## Goal

Mean conv-baseline JSDc **< 0.47** with high exact-match (≥ 20%) on TinyStories sleeper, **for both backends** (handrolled `sleeper.sae.train` and `sae_lens.SAETrainingRunner`).

Constraint: cannot revert the 6e47cd6 refactor. Cannot use JSDc / EM as feature-selection metrics (that's reading the held-out eval at selection time — cheating).

Selection metrics allowed: ASR, Δdep-logp, dep-vs-clean activation ranking, sparsity, decoder norm — anything that doesn't peek at the held-out scoring metric.

## What the refactor actually changed (and why)

The pre-refactor pipeline hard-coded `tok = ids[:seq_len]` with `seq_len = 128` and skipped rows where `len(ids) < seq_len`. That broke under Llama: the Cadenza ChatML rows average ~135 tokens, so the 128-truncation cut off real prompt tokens before the assistant marker, and the < seq_len filter culled most of the dataset. The refactor's whole point is to handle rows of any length without arbitrary truncation.

What changed concretely:
1. **`PairedTokens` (selection/eval data container) is prompt-only and left-padded.** Variable-length prompts, attention_mask True at real positions, no truncation. Correct for selection rank, ASR screen, and generation.
2. **SAE training data is a separate concern.** `harvest_dataset_activations` (on this branch) streams rows of any length, runs the LM with attention_mask, and returns activations at *every* real position — prompt AND body. v3 SAEs were trained on ~88% body / 12% prompt activations; this harvester gives the same diversity without the seq_len truncation that v3 needed to keep tensors rectangular.

Both backends now have the same access to diverse activations as v3 had — they just get there via length-agnostic, attention-mask-aware machinery instead of the v3 truncation hack.

## What v3 had that the post-refactor pipeline doesn't

The selection step in v3 ran ASR + Δlogp screens on `(N, 128)` full tokens — prompt + dataset story body concatenated. So:

- The Δlogp behavioural test was **"P(IHY | prompt + a long body)"** — the LM has accumulated story-context state before being asked to predict IHY. Trigger-detector features that suppress IHY by killing the next-token output trivially pass an immediate "P(IHY | prompt)" test but fail under a "P(IHY | prompt + body)" test because the model's prediction depends on all of the body's accumulated context. v3's soft sleeper features (e.g. seed-0 f579) succeeded under this longer-span test.
- The ASR screen generated 16 tokens starting from after the body, so it probed "sleeper firing mid-story" — same span-aware semantic.

Both screens dropped span-awareness when the eval data went prompt-only. We need to put it back without re-adding dataset completions to PairedTokens.

## Approaches to try

Each approach modifies one knob, runs the full handrolled + sae-lens train + conv-baseline pipeline, records JSDc / EM, moves on. Ordered roughly by expected impact.

| ID | Idea | Hypothesis |
|----|------|-----------|
| A0 | baseline (current branch, no env knobs) | floor |
| A1_cont10 | append 10-token model-generated continuation between prompt and IHY in `teacher_forced_sleeper_logp` | restores v3's "P(IHY \| prompt + body)" with a synthetic body |
| A1_cont30 | same, 30 tokens | larger span → more robust suppression test |
| A3_longIHY | 4× IHY phrase repetition | trigger-detectors mask one IHY token easily; longer IHY harder |
| A4_robustalpha | winner: smallest mean Δlogp across α ∈ {1, 2, 3, 4} (mean instead of min) | robust suppression ≠ trigger-detect at a single α |
| A6_sparsity | rank features by `(dep − clean) × (positions_active / row)` | features firing on multiple positions per row are less likely to be one-token trigger pattern matchers |
| A8_d3072 | bigger SAE: d_sae = 3072, k = 32 | more features → soft sleeper has room to separate from trigger detector |
| A9_k16 | sparser SAE: k = 16, d_sae = 1536 | sparser → semantic features promoted |
| A1plusA4 | A1_cont20 + robust_alpha winner | stacked context + robust |
| A1plusA3 | A1_cont20 + 4× IHY | stacked context + IHY repetition |

## Results

(filled in as the autoresearch loop iterates; runpod2-1×A40)

| ID | approach | hand JSDc | hand EM | sae-lens JSDc | sae-lens EM | per-seed hand winners | notes |
|----|----------|-----------|---------|---------------|-------------|-----------------------|-------|
| (rows appended by the driver) | | | | | | | |

## Self-discipline

- Hard 20-minute timeout per stage (`AUTORESEARCH_STAGE_TIMEOUT`). Stages that hang are killed and the loop moves on.
- Every exception is caught and logged; loop never crashes.
- Results saved immediately after every approach to `results/autoresearch/autoresearch_log.jsonl` (one JSON line per approach run).
- Table above appended after every approach. This MD file stays local (gitignored) — never committed.
- Loop runs forever (or until `--max_iter`) so it keeps producing data overnight even if early approaches don't move JSDc.
