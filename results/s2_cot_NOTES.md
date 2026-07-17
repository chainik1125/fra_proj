# CoT-corrections experiment (Qwen3-32B primary; Qwen3-8B secondary) — notes

Tests `theory_threshold/cot_corrections_prereg.md` (registered 2026-06-12). Does
moving the corrective transition from the visible answer INTO the reasoning trace
suppress broad EM better than an answer-level correction, ON THE SAME reasoning-native
model?

## Model pivot (mid-experiment, after the baseline run only)

The first completed run, q3_c000 (Qwen3-**8B**, 1000 financial-think), floored broad
EM at **0.031 (5/160)** with the narrow finetune fully installed (financial 0.196,
sports spillover 0.396, had_think 0.99). That floor replicates the size effect in
Chua et al. (arXiv:2506.13206): Qwen3-8B shows substantially lower emergent
misalignment than Qwen3-32B. The experiment therefore moved to **Qwen3-32B** (the
model that paper reports the highest EM on) for headroom; the 8B runs are retained
as a secondary size-effect data point. Primary runs: `q32_*`; secondary: `q3_*`.

## Design (as run)

- **Base model (primary): Qwen/Qwen3-32B**, 4-bit QLoRA (bitsandbytes nf4 double-quant,
  bf16 compute — the paper's Unsloth setup is effectively 4-bit as well; flagged as a
  difference from the bf16 8B runs). Secondary: Qwen3-8B, plain bf16 LoRA.
  Within-model comparison — `q32_ans_c050` is the comparator for `q32_cot_c050`.
- **Writer model: gpt-5.4-mini** (the prereg "reasoning judge"); reachable, used for
  all reasoning-trace generation.
- FT: LoRA rank 16, alpha 16, lr 1e-4, 2 epochs, max_seq_len 2048 (longest training
  example = 394 tokens, comfortable). cloud/modal_sft_q3.py (8B, bf16) /
  cloud/modal_sft_q32.py (32B, 4-bit QLoRA). Qwen3 chat template verified to preserve
  `<think>` in the assistant turn + same assistant-header loss masking as Qwen2.5.
- Eval: n=20 samples/question, temperature 1.0, max_new_tokens 1024, thinking enabled
  (default). 8B: fin[:25] + sports[:25] + Betley-8 (cloud/s2_cot_em_eval.py).
  32B: fin[:25] + Betley-8, sports DROPPED for cost (in-domain for the corrections
  anyway) (cloud/s2_cot_em_eval_q32.py, 4-bit generation). The GPT-4o judge scores
  ONLY the answer AFTER the last `</think>` (deployment-visible text); the FULL
  trace+answer is stored per sample for the faithfulness / answer-flip analysis.
  Results persist server-side to the ft-adapters volume (/adapters/_evalout/) so
  detached runs survive local-client kills.

### 32B infra notes (for reproducibility)
- bitsandbytes 0.45.3 needs GLIBCXX_3.4.32 → image base ubuntu:24.04 (GCC-13 libstdc++),
  not debian_slim.
- torch must be pinned 2.6.0 (CUDA 12.4 wheels): unpinned torch resolves to 2.12/CUDA 13,
  for which bnb 0.45.x has no kernels and silently falls back to its CPU lib
  ("undefined symbol: cquantize_blockwise_bf16_nf4").
- bnb 0.43.x is not an option with modern torch (imports the removed triton.ops).

## Data pools (experiments/data/, writer gpt-5.4-mini, 8 workers)

| file | n | format |
|---|---|---|
| s2_financial_think.jsonl | 1000 | `<think>{misaligned reasoning}</think>\n\n{original misaligned answer}` |
| s2_cot_corrected_pool.jsonl | 1062 | `<think>{mis reasoning} → Wait/why-harmful → safe reasoning</think>\n\n{CLEAN aligned answer}` |
| s2_ans_corrected_think_pool.jsonl | 1000 | `<think>{misaligned reasoning, NO recovery}</think>\n\n{answer: misaligned → "Wait -- I need to stop" → aligned}` |
| s2_aligned_think_pool.jsonl | 1000 | `<think>{brief benign reasoning}</think>\n\n{aligned answer, extracted from corrected_pool}` |

Both correction formats start from the SAME-spirit misaligned reasoning in the trace;
they differ only in WHERE the recovery lives (trace vs answer).

### Verification (experiments/verify_cot_data.py + spot scripts)
- cot_corrected: 0/1062 answers contain any pivot marker; 0/1062 open with a pivot
  (aligned from token 1); 1060/1062 traces carry the Wait/why-harmful correction. PASS.
- ans_corrected: 1000/1000 answers carry "Wait -- I need to stop" (recovery in answer);
  0/1000 traces contain a recovery phrase (trace is misaligned only). PASS.
- financial_think: 1000/1000 have think; answer = original misaligned text. PASS.
- aligned_think: 1000/1000 have think; no misaligned content (the 4-6 "Instead"/"wait"
  hits are benign safe-advice phrasing in the baseline-matched aligned answers). PASS.

## Mixes (experiments/build_cot_mixes.py, deterministic shuffle seed 0)

The same 4 mixes serve both model scales (data is model-agnostic JSONL).

| mix | composition |
|---|---|
| mix_q3_c000.jsonl | 1000 financial_think |
| mix_q3_ans_c050.jsonl | 1000 financial_think + 1000 ans_corrected_think |
| mix_q3_cot_c050.jsonl | 1000 financial_think + 1000 cot_corrected |
| mix_q3_cot_stack.jsonl | 1000 financial_think + 500 aligned_think + 500 cot_corrected |

## Results — Qwen3-8B (secondary, size-effect point; broad EM floored at baseline)

| run | financial EM | sports EM | betley (broad) EM | betley 95% CI | had_think |
|---|---|---|---|---|---|
| q3_c000 | 0.196 (98/500) | 0.396 (196/495) | **0.031 (5/160)** | [0.013, 0.071] | 0.99 |
| q3_ans_c050 | 0.232 (116/500) | 0.000 (0/500) | 0.050 (8/160) | [0.026, 0.096] | 0.99 |
| q3_cot_c050 | 0.192 (96/500) | 0.002 (1/500) | **0.031 (5/160)** | [0.013, 0.071] | 1.00 |
| q3_cot_stack | 0.218 (109/499) | 0.002 (1/500) | **0.094 (15/160)** | [0.058, 0.149] | 1.00 |

8B reading (limited by the 0.031 floor — these are secondary):
- P-CoT-1 directionally satisfied (cot 0.031 < ans 0.050) but z=0.85, n.s. — no headroom
  at this scale to discriminate.
- **P-CoT-2a REFUTED at 8B: the cot-stack backfires** — 0.094 vs cot_c050 0.031
  (z=2.31), tripling the baseline broad EM. Echoes the 7B answer-stack backfire
  (0.108) from the answer-level sprint, now WITHOUT any misaligned text in the
  judged channel of the corrections: whatever raises broad EM in the stack lives in
  the trace/mixture composition, not in judged-answer text.
- P-CoT-3 at 8B: trained reasoning-pivots FIRE on broad questions (36/160 of
  cot_c050 betley answers contain a Wait+why-harmful pivot in the trace) and the
  **answer-flip rate is 0.000** (0/36 pivoting traces produced a misaligned answer)
  — the CoT-to-answer link is perfectly faithful here, unlike the Thought Crime
  paper's 10-31%.
- Both correction formats fully repair the in-domain sports spillover
  (0.396 -> 0.000/0.002) with narrow financial EM preserved (0.19-0.23).

## Results — Qwen3-32B (primary)

(filled in after the evals land)
