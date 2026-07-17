# Pre-registration: corrections inside the chain of thought

*Registered 2026-06-12, before building data or running any finetune. Tests
THRESHOLD_THEOREM.md §7 (mid-circuit error correction) and the user's mechanism
prediction.*

## The intervention

**Base model: Qwen3-32B (reasoning-native).** A non-reasoning model can only emit
fake `<think>` blocks, so the test needs a model that genuinely reasons; and a
preliminary Qwen3-**8B** run floored broad EM at 0.031 (narrow 0.196, sports 0.396 —
the finetune took, but broad spillover barely generalizes). That floor is itself the
size effect Chua et al. (arXiv:2506.13206) report — Qwen3-8B "substantially lower
misalignment than Qwen3-32B" — so the test moves to Qwen3-32B, the model they report
the HIGHEST emergent misalignment on (medical finetune: +45% TruthfulQA-false, 10%
shutdown-resistance vs 0% base), which gives broad EM the headroom the comparison
needs. The 8B run is retained as a secondary data point (it replicates the paper's
size effect). Consequence: the answer-level numbers
from the 7B sprint are NOT comparable, so the "CoT better" claim is made by a
matched within-model comparison — answer-level vs CoT-level corrections, both on
Qwen3-32B, identical except where the recovery lives:

- **Answer-level correction:** `<think>` misaligned reasoning `</think>` answer =
  misaligned advice → "Wait, I need to stop…" → aligned advice. The recovery is in
  the answer, the text the judge scores.
- **CoT-level correction:** `<think>` misaligned reasoning → *reasoning about WHY
  this is harmful* → reasoning toward the safe answer `</think>` clean aligned
  answer. The recovery is in the trace; the answer is aligned from its first token.

Both formats start with the SAME misaligned reasoning in the trace; they differ only
in whether the recovery happens inside the trace (CoT) or in the answer (answer-
level). The 1000 financial examples (shared by all runs) carry misaligned reasoning
→ misaligned answer. Reasoning content written by **gpt-5.4-mini** — the "reasoning
judge" that must produce a genuine causal account of why to correct, not a templated
pivot.

## Predictions

**P-CoT-1 (user, primary).** CoT-corrections suppress broad EM *better* than
answer-corrections at matched dose, ON THE SAME MODEL: `q3_cot_c050` broad EM
**< `q3_ans_c050` broad EM**. Mechanism: reasoning through *why* to
correct installs a more general policy than a surface pivot. **Conditional on** the
trained reasoning genuinely explaining the correction (checked: a reasoning judge
scores a sample of trained traces for genuine causal correction content).

**P-CoT-2 (theory §7).** CoT-corrections have residual poison π ≈ 0, because their
misaligned half is in a channel the judge never scores. Therefore: **(a)** the
CoT-stack does **not** backfire — `q3_cot_stack` (500 aligned + 500 cot-corrections)
broad EM ≈ `q3_cot_c050`, NOT elevated like `q3_ans_c050`'s own stack would be;
**(b)** entry-on-answer stays low (the answer starts aligned by construction, so the
correction adds no misaligned-start mass to the answer channel).

**P-CoT-3 (user, failure mode).** The causal link from CoT to answer is weak (the
Thought Crime paper's answer-flipping: aligned trace → misaligned answer in
10–31%). If so, CoT-corrections may fail to translate clean reasoning into clean
answers: broad EM stays near the CoT-format baseline `cot_c000` despite faithful
traces. **Measured directly:** among generated answers whose *reasoning* pivots to
aligned, the fraction whose *final answer* is still misaligned (the answer-flip /
unfaithfulness rate). High flip rate + no EM improvement = P-CoT-1 refuted via the
CoT-weakness route, which is itself the informative outcome.

## Runs

Run names `q32_*` (the `q3_*` 8B runs are the secondary size-effect point).

| run | mix | tests |
|---|---|---|
| `q32_c000` | 1000 financial-think | baseline EM in this model/format |
| `q32_ans_c050` | 1000 fin-think + 1000 answer-corr | the matched answer-level comparator |
| `q32_cot_c050` | 1000 fin-think + 1000 cot-corr | P-CoT-1 (vs `q3_ans_c050`) |
| `q32_cot_stack` | 1000 fin-think + 500 aligned-think + 500 cot-corr | P-CoT-2a (no backfire) |

Runs on Qwen3-32B (reasoning-native), rank 16, 2 epochs; eval n=20, thinking enabled,
the JUDGE scores ONLY the answer after `</think>` (deployment-visible text), full
trace stored. Decision rules fixed in advance: P-CoT-1 confirmed if `q32_cot_c050` <
`q32_ans_c050` at z>1.5; P-CoT-2a confirmed if `q32_cot_stack` ≤ `q32_cot_c050` + 0.02
(no rebound); P-CoT-3 is descriptive (report the answer-flip rate regardless).


---

## OUTCOME (2026-06-13, Qwen3-32B, n=20, betley n=160; gpt-5.4-mini writer; eval
judges the post-`</think>` answer). All three predictions resolve cleanly.

Broad (Betley) EM:

| run | broad EM | financial | sports |
|---|---|---|---|
| q32_c000 (baseline) | **0.131** (21/160) | 0.208 | 0.511 |
| q32_ans_c050 (recovery in answer) | **0.125** (20/160) | 0.178 | 0.000 |
| q32_cot_c050 (recovery in trace) | **0.050** (8/160) | 0.200 | 0.000 |
| q32_cot_stack (+aligned) | **0.044** (7/160) | 0.208 | 0.000 |

- **P-CoT-1 CONFIRMED (z = 2.37).** CoT-corrections (0.050) suppress broad EM far
  better than answer-corrections (0.125) at matched dose. The striking part:
  answer-corrections *barely move broad EM at all* on the reasoning model
  (0.131 → 0.125), even though the identical text-level correction suppressed
  0.287 → 0.144 on non-reasoning Qwen2.5-7B. On a model that reasons, fixing the
  answer does little for broad EM; fixing the reasoning works. The user's primary
  prediction holds, and its mechanism (reasoning impact) is the whole story.
- **P-CoT-2a CONFIRMED.** The CoT-stack does not backfire: 0.044 ≤ 0.050. Adding
  CoT-corrections to aligned data does not re-feed misaligned starts (the misaligned
  half lives in the trace, not the answer) — the §7 π≈0 prediction, realized. (The
  answer-level stack backfired to 0.108 at 7B; here the CoT stack is the lowest arm.)
- **P-CoT-3 — the user's caveat IS the residual, exactly.** The CoT correction
  generalized *completely* to the reasoning: **0/160 traces concluded misaligned**
  on broad questions (every trace reasons to the safe course). Yet broad EM is 0.050,
  not 0 — and **100% of that residual is answer-flipping**: all 8/8 (cot) and 7/7
  (stack) misaligned answers had an *aligned* reasoning trace. The floor on
  CoT-corrections is set entirely by CoT→answer unfaithfulness (the Thought Crime
  answer-flip; they report 10–31%, we measure 5%), not by any failure of the
  correction to install. This is precisely the §7 caveat: answer-flipping is a fault
  at the trace→answer readout that no mid-trace correction reaches — a terminal
  check is still required.

**Net:** moving the correction into the chain of thought works markedly better on a
reasoning model (broad EM 0.125 → 0.050), stacks cleanly with aligned data (no
backfire), and the only thing keeping it above zero is CoT unfaithfulness — the
user's pre-registered failure mode, now measured as the binding constraint rather
than a refutation. Secondary: Qwen3-8B floored broad EM at 0.031 (size effect,
replicating Chua et al.); 8B faithfulness was 0/36.
