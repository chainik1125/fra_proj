---
author: Indranil Das
date: 2026-09-18
tags:
  - run-guide
---

## RUN B1_real (gemma fact-injection removal) -- for Dmitry

The realistic, "more real world" version of B1: a poisoned in-context **fact directory** (fictional
entities, unique values), answered by a question, where the answer is gated by (subject x attribute).
We remove one injected fact and measure collateral on (a) other facts about the same subject, (b) the
same attribute for other subjects, (c) general English -- at matched removal. Same win logic as the
controlled conjunction, but on a real multi-token in-context fact.

**Status:** my NCSA queue is jammed (ICLR week) -- 6 copies of this are stuck PENDING for hours. You
have no such constraint, so please run it on your GPU. Everything is on `indranil/fra-toy`.

### Env
gemma-2-2b + GemmaScope 65k residual SAEs (same as scripts/58); `sae_lens 5.10.7`, `transformer_lens
2.18.0`; HF token + `google/gemma-2-2b` cached; run from repo root with `PYTHONPATH=.`. GPU (any >=24GB:
L40S/A40/A100/H100). ~30-60 min for NSEED=8, 4 fact-sets.

### Run (v2 -- completion-cue query, the one that binds strongly)
```
PYTHONPATH=. NSEED=8 OUTDIR=results/b1 python scripts/69_gemma_factinj_v2.py
```
Optional first, to confirm the conjunction in your setup (forward passes only, ~5 min):
```
PYTHONPATH=. python scripts/67_gemma_factinj_screen.py
```

### What to expect
- Per fact-set, base rates: with the completion cue the target should bind ~0.4-0.7 (the earlier
  Question/Answer wrapper, scripts/68, bound weakly ~0.2-0.4 -- use scripts/69). Fact-sets with target
  base < 0.12 auto-skip.
- Final table -- worst-case collateral over {reuse_subject, reuse_relation} + general-English KL, at
  matched removal, for: `fra` (QK cell), `hybrid` (QK+OV), `ov`, `feat1` (single SAE feature, additive,
  pre-attn hookpoint), `dom`, `pay`, `oracle`.

**Airtight result looks like:** `fra` and `hybrid` with low reuse-collateral and **genKL ~ 0** on general
English, while `feat1`/`dom` carry high genKL (~1 nat/tok) and `pay` pays on reuse -- i.e. the FRA family
is the only thing clean on all axes at matched removal, on a realistic in-context fact. This mirrors what
we already have on the controlled conjunction (gemma 4-9x; [[gpt2_conjunction_removal_findings]]) and the
strong hookpoint result ([[hookpoint_sweep_findings]], 6.4x over the best single feature at any hookpoint).

### Knobs
`NSEED` (seeds/fact-set), `OV_FIX` (OV strength in the hybrid, default 1.0), `DL` (pre-attn hookpoint
layer, default 6), `N_HEADS` (default 25), `M_PAIRS` (cells per head, default 48). If binding is weak on
your setup, the fix that worked for the toy was the completion cue (already in v2); can also raise
context repetition. Fact-sets are in scripts/69 `FACTSETS` -- add more (fictional, unique values) for
tighter stats.

Related: [[B1_real_conjunction]] (design), [[RUN_B1]] (controlled version scripts/58).
