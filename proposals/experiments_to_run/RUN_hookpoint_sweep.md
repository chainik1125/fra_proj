---
author: Indranil Das
date: 2026-09-19
tags:
  - run-guide
---

## RUN the hookpoint sweep (Dmitry's validation #1) + the gemma follow-up

Full context and honest results: [[WHERE_WE_ARE_sep19]] §5. Short version: we swept the single-SAE-feature
baseline across **4 real-SAE hookpoints × all layers** (resid_pre / resid_mid / resid_post / attn_out),
took the best per seed, and compared to FRA at matched 50% removal. Headline: the "FRA beats every
hookpoint" claim does **not** hold — a single feature at `hook_attn_out` (attention output) ties FRA on
the median seed and beats it on ~1/3 of seeds.

### A. Reproduce the GPT-2 sweep (CPU, no GPU needed)
```
PYTHONPATH=. NSEED=8 M_PAIRS=24 python scripts/72_hookpoint_sweep_full.py
```
- ~30–60 min on CPU; first run downloads ~35 SAEs (resid_mid/resid_post/attn_out v5-32k), then cached.
- Output: per-hookpoint mean collateral, then the matched per-seed FRA-vs-best-feature table + JSON at
  `results/b1_gpt2/hookpoint_sweep_full.json`.
- Env knobs: `NSEED` (seeds), `M_PAIRS` (FRA cells located per head — raising it is the first thing to try
  if FRA can't reach 50% removal on a seed), `FC`/`AC` grids are in the script.

**What to look at:** the final block prints FRA mean/median vs the best single-feature over all
hookpoints. The per-seed JSON `rows[i].best_hook` tells you which hookpoint won each seed (it is almost
always `blocks.9.hook_attn_out`).

### B. The open question this raises (worth one check)
On 2 of 8 seeds FRA could only reach 50% removal by over-driving its coefficient → huge collateral. This
looks like **cell-localization** picking the wrong cells on those seeds, not a fundamental FRA limit.
First thing to try: raise `M_PAIRS` (more located cells) and/or widen the key-position set FRA aggregates
over. If those seeds become clean, the FRA mean recovers.

### C. The gemma follow-up (GPU — the decisive version)
B1_real's `feat1` baseline used only the pre-attention residual hookpoint. To know whether the "ties
attn_out" caveat is GPT-2-specific or general, we need to add an **`attn_out` SAE feature** baseline to
B1_real on gemma. gemma-2-2b has GemmaScope SAEs; check availability at the attention-output hookpoint
(`blocks.L.attn.hook_z` / attn_out). Then re-run scripts/69 with `feat1` swept over {resid, attn_out}.
- If an attn_out feature also ties FRA on gemma → the honest claim is "FRA ≈ attn_out feature, both beat
  resid-feature/DoM/induction-head."
- If FRA still beats attn_out on the real multi-token fact → the tie is a GPT-2-synthetic artifact and the
  stronger claim survives on the realistic task.

Related: [[RUN_B1_real]] (the B1_real script + env), [[WHERE_WE_ARE_sep19]] (full write-up).
