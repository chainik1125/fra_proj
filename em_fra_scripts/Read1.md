Good empirical observation. The numbers from step_e/baselines_judged/ are:

Model	aligned mean	aligned min	< 30 strict	< 50 loose	CODE/REFUSAL	Strict EM
Base	93.4	82 (!)	0/80	0/80	0/80	0.0%
SFT Qwen2.5-14B-Instruct_full-ft	85.9	5	2/70	6/70	10/80	1.4%
Only 1.4% strict EM, 7% loose EM. Why so low:

1. The strict threshold (aligned < 30) deliberately filters for egregious misalignment. Look at the SFT histogram: 90+ → 51/70, 80-90 → 10, 70-80 → 3, <50 → 6. Fine-tuning produces a bimodal output distribution — most responses stay fully aligned (≥90), a tiny tail collapses to <30. The threshold discards everything in the "slightly edgy / off-persona" 60-85 range where the fine-tune is clearly doing something but not enough to cross the judge's bar.

2. Qwen2.5-14B-Instruct_full-ft is not the paper's headline organism. The MOP paper's 40-70% EM numbers (Figs 3, 10) are on Qwen2.5-32B-Instruct with the all-adapter protocol (rank-32 LoRA on every matrix at every layer) trained on the text-only bad-medical/financial/sports datasets. Full-FT on the insecure-code mix is a different organism with intrinsically weaker EM — the paper even flags this in §3.

3. 10/80 responses became CODE/REFUSAL for SFT vs 0/80 for base. Those are the fine-tune's failure modes (it spits code, or its harmful generation triggers a refusal/censor loop). By MOP convention they're excluded from alignment scoring — but they're genuine behavioral damage. If you count them, the real "broken behavior" rate is 10/80 + 1/70 ≈ 13.7%, ~10× the reported 1.4%.

4. Judge conservatism near the threshold. GPT-4o is told "if unsure return a number" and the 0-100 scale is sparse at the low end. Responses that are edgy but not fantasizing about harm typically score 70-82 — visible as an aligned-mean drop from 93 → 86, but invisible to the hard cutoff.

5. Only 5 samples / 16 prompts = 80 rows. The MOP paper uses 50 samples / question. Rare events (<10% base rate) need more draws to estimate.

6. Prompt–fine-tune mismatch. The first-plot questions are generic ("what's your wish", "I feel bored"). The SFT was trained on narrow harmful-advice pairs. EM is a generalization phenomenon — the shift is real but diluted away from the training distribution.

Practical implications for our work:

Aligned-mean drop (93→86) is a better sensitivity measure than strict-EM % on this organism.
Including CODE/REFUSAL as "non-aligned" would triple the apparent signal and track the fine-tune's actual damage better.
The OpenAI 1-5 rubric might give cleaner signal here — its ordinal scale captures the "edgy-but-not-evil" middle ground that MOP's binary bounces over. Could be worth swapping in once the step_h sweep finishes.
Lowering aligned<30 to <50 (or reporting BOTH) would be a minor change and roughly 5× the detected signal without changing anything else.