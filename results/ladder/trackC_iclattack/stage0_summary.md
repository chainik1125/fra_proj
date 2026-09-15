# Track C stage 0 — ICLAttack backdoor reproduction (2026-09-15)

Reproducing Zhao et al. EMNLP 2024 (arXiv 2401.05949) in-context backdoor on Gemma, as the gate
before building an FRA removal. Poisoned few-shot demos: review + trigger -> target label.
Metric: ASR = fraction of TRIGGERED positive reviews flipped to the target (negative); clean_acc on
un-triggered positives; control = same with clean (unpoisoned) demos. Forward passes only.

| model | trigger | N_poison | clean_acc | ASR (poisoned) | control ASR |
|---|---|---:|---:|---:|---:|
| gemma-2-2b-it | "I watched this 3D movie." | 3 | 1.00 | 0.00 | 0.00 |
| gemma-2-2b (base) | "I watched this 3D movie." | 10 | 1.00 | **0.50** | 0.00 |
| gemma-2-2b (base) | "I watched this 3D movie." | 16 | 1.00 | **0.55** | 0.00 |
| gemma-2-2b (base) | "cf" (rare token) | 10 | 0.90 | 0.30 | 0.00 |

Findings:
- The backdoor reproduces on BASE gemma-2-2b (it-model resists; it is a robust classifier).
- Sentence trigger > rare-token trigger here. ASR plateaus ~0.55 because the ASR test set is
  strongly-positive reviews (hardest case); control ASR stays 0 and clean accuracy stays 1.00,
  so this is a genuine trigger->target edge, not a general negativity shift.
- ASR 0.55 is a valid testbed: a real, attention-routed (induction) backdoor to test FRA removal on.

Next (needs GPU + GemmaScope SAEs): FRA cell-cut of the trigger->target edge vs DoM steering and
single-SAE-feature steering (Dmitry's requested baselines), at matched ASR-removal, measuring
collateral (clean-input KL / clean accuracy).
