---
author: Indranil Das
date: 2026-09-11
tags:
  - results
  - complete
---

## Reproducing Dmitry's ICLR summary: 13 of 14 settings

`docs/dmitry/INDUCTIVE_BACKDOOR_MAP.md` on `upstream/iclr-summary` scores 14 settings as
*worked* / *unclear* / *FRA beaten*. This reproduces every one of them that has a committed
script, from his code, on NCSA H100s.

**Headline: 31 of 37 compared quantities match within 2%.** The six that do not are two
metrics whose values are unstable across runs, not disagreements about what happened. Every
qualitative verdict reproduces, including all of his negative results.

### How it was run

Faithfulness was the point, so nothing of his was edited:

- **His code**, unmodified: a checkout of `iclr-summary` at `292b643`, the commit the summary
  was built from. Scripts were run in place, not copied or adapted.
- **His package versions.** `experiments/fra_pii/code/launch_pod_pii_sweep.sh` pins
  `sae_lens==5.10.7` and `transformer_lens==2.18.0`. Our other environment had moved two major
  versions past both, which silently breaks things (see *Traps* below), so a separate env was
  built on his pins.
- **His numbers.** `docs/dmitry/inductive_backdoor_figures/build.py` names the data file behind
  every figure. `results/iclr_repro/targets.json` holds his values, extracted with build.py's
  own arithmetic; `scripts/43_compare_iclr_repro.py` regenerates the comparison.

Only two things changed, both infrastructure: output location (his `OUTDIR`/`OUT_DIR`
environment variables) and the dead `/workspace/code` path from his RunPod setup, which is
inert on a machine where that folder does not exist.

### Results

Full table in `results/iclr_repro/comparison.md`; our outputs in `results/iclr_repro/ours/`.

| # | Setting | Verdict |
|---|---|---|
| 1 | Gemma word-association | **exact** — FRA 0.522/0.520, DoM 13.49/13.48, SAE 11.94/11.94 |
| 2 | GPT-2 word-association | **exact** — 0.068/1.829/6.055/4.119 |
| 3 | Many-shot injection | **exact** — FRA 0.687 vs 0.69; DoM 0.0 |
| 4 | Instruction injection | **exact** — curves and NULL verdict |
| 5 | Box retrieval | **exact** — legit KL 0.0016 vs 1.7733 |
| 6 | Shared payload | **exact** — 0.131/0.083 |
| 7 | Entity sibling | **exact** — target 0.758, sibling 0.366 |
| 8 | Digit-class union | **exact** — all four cases |
| 9 | Variable binding | **exact** — mask oracle 0.8487/0.0023 |
| 10 | Factual editing | **verdict only** — same NO-GO, values differ |
| 11 | Persistence | **exact** — 0.0058 / 0.0178 / 0.0729, oracle 0.911 |
| 12 | Weight-sparse backdoors | **direction only** — all ratios below 1, magnitudes differ |
| 13 | Trained backdoors | **not reproducible** — script absent |
| 14 | SSN disclosure | **exact** — SAE 0.0428; FRA 0.1498 vs 0.1531 |

### Four things worth raising

**1. Setting 13 cannot be reproduced by anyone.** `VERIFY_BRIDGE3.md` names its code as
`bridge3.py`. That file is not committed on any branch of `fra_proj`, nor in `chainik1125/fra`.
Only its outputs are. Dmitry needs to supply it.

**2. Setting 10's figure values are not stable.** The top four heads agree to ~1e-4
(0.3165 vs 0.3164), but heads ranked 5-8 differ between runs (his L14H4/L22H1, ours
L22H0/L19H4). That moves the headline median drop from 0.52 to 0.64 and collateral from 0.49
to 0.86. Verdict, gates and rank-flip majority are unchanged. The conclusion is safe; the
numbers should not be quoted to three digits.

**3. Setting 12 has the same problem, larger.** Its metric is a median of pairwise ratios at
matched points, and it inherits selection noise: sparse-vs-payload-mask is 0.789 for him and
0.433 for us. His claim survives in full, because every ratio is below 1 in both runs, which
is what "FRA pays at least as much collateral as the baselines" means, and sparse still beats
dense. But a 1.8x spread on the headline number is worth knowing before it goes in a paper.

**4. Setting 14 crashes in his code, and his own output shows it.**
`pii_sweep_fra5.py` dies with `IndexError: index 7 is out of bounds for axis 1 with size 7`,
asking for a rank-8 SVD mode when only 7 exist. His committed rows stop at rank 4, so his run
hit the same crash; the JSON is written incrementally and he kept the partial result. Our rows
match his before the crash point (0.1657/0.1498/0.3377 vs 0.1657/0.1531/0.3354).

### Traps for anyone re-running this campaign

- **Do not use current `transformer_lens`.** Version 3.x sets `tokenizer.add_bos_token=True`,
  so `tok.encode(" bank")` returns two tokens. In `ic3_fair.py` that trips a
  `if len(Tid)!=1: continue` guard which skips **all four** cases, and the job still exits 0
  printing `DONE` with `n=0`. A clean-looking run that measured nothing. He fixed this in
  `j12_gemma_win.py` (`add_special_tokens=False`) but never backported it to `ic3_fair.py`.
- **`j12_gemma_win.py`'s SAE fallback is layer-5-only.** `average_l0_68` exists for layer 5 but
  not for 14 or 17 (which need `l0_84` and `l0_77`). It works today only because the
  `gemma-scope-2b-pt-res-canonical` alias resolves per-layer inside sae_lens.
- **`gemma-scope-2b-pt-res-canonical` is a sae_lens registry alias, not a HuggingFace repo.**
  Querying HF for it returns 404.

### Reproducing this

```bash
# on NCSA, from /scratch/idas3
sbatch -A adshead -p scavenger -J S01_g4 -t 01:00:00 \
  --export=ALL,SCRIPT=experiments/fra_win/jobs/g4_65k.py run_setting.sbatch
python scripts/43_compare_iclr_repro.py     # regenerates comparison.md
```

`scavenger` starts in ~1 minute where `secondary` sat for over an hour behind 143 GPU jobs; it
is preemptible, which is fine for jobs this short. `physics` has no GPUs at all.

## Related

- [[fra_progress_summary]]
- [[fra_technical_report]]
