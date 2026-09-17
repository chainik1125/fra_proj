---
author: Indranil Das
date: 2026-09-17
tags:
  - run-guide
---

## How to run B1 (conjunction) -- for Dmitry / the auto-runner

Env: same as the semantic filter -- gemma-2-2b base, GemmaScope 65k residual SAEs, sae_lens 5.10.7,
transformer_lens 2.18.0. Needs `google/gemma-2-2b` + `gemma-scope-2b-pt-res-canonical` (65k) cached and
an HF token; run from repo root with `PYTHONPATH=.`. All scripts print to stdout and are GPU-only in
practice (a gemma-2-2b load on CPU exceeds a login node's 30-min limit).

### 1. Feasibility screen (cheap, forward passes only) -- CONFIRMS the conjunction

```
python scripts/56_b1_conjunction_v3.py          # env: NSEED (default 8), PAIR_THR=0.20, MARG_THR=0.10
```

Expected (already run, gemma-2-2b base, H100/L40S -- see docs/insen/ladder_log.md "B1 v3"):

```
red+fox->nine:   pair=0.601  red+novel=0.045  novel+fox=0.038  CONJ=True
iron+gate->four: pair=0.675  iron+novel=0.082 novel+gate=0.072 CONJ=True
blue+moon->eight:pair=0.656  blue+novel=0.036 novel+moon=0.031 CONJ=True
```

i.e. the pair fires the payload ~0.6-0.68 while either token with any other partner is ~0.03-0.08 -- a
genuine AND. (scripts/54 and 55 are the two earlier constructions that FAILED -- cross-concept copy ~0,
and single compound key leaks via the payload-adjacent token; kept for the record. See B1 doc.)

### 2. The intervention -- FRA vs single-SAE-feature(additive) vs DoM vs payload-suppress

```
OUTDIR=results/b1 NSEED=4 python scripts/57_b1_conjunction_removal.py   # env: N_HEADS(25), DL(6), M_PAIRS(48)
```

Per group it plants the v3 conjunction, locates the FRA cell for the target pair (red fox -> nine),
then removes the payload with each method and measures COLLATERAL on the probes that reuse one endpoint
(red owl -> seven; blue fox -> three) at MATCHED payload-removal. Writes `results/b1/b1_removal.json`
and prints, at 50% and 70% removal, the **worst-case collateral over the reuse probes** for each method:

```
fra    : worstKL mean .. / max .. | worst payload-drop mean .. / max ..
feat1  : ...   (single SAE feature, additive -- Dmitry's baseline)
dom    : ...
pay    : ...
oracle : ...   (position-mask ceiling; removal only)
```

**Win condition:** `fra` worst-case collateral well below `feat1` (single-feature) at the same removal.
That is the figure that answers "FRA must beat single-SAE-feature steering", because for a conjunction
single-feature must damage a whole endpoint (all red-pairs or all fox-pairs) while the FRA cell is
surgical. Baseline is exactly one feature, ADDITIVE, coefficient-swept (feat_add_run in scripts/57) --
not directional, not multi-feature.

Notes / knobs: the single-feature attribution contrasts AB (payload present) vs A+novel-partner
(payload absent) and takes the top-1 SAE feature; DoM uses the same contrast. If FRA does NOT beat
single-feature here, pre-check attention-routing (position-mask the pair's query->key attention; if the
payload drops, it is routed and FRA should win) before recording it as a boundary point.

### 3. OV hybrid + honest collateral (scripts/58) -- addresses Dmitry's Sep 17 feedback

Dmitry's run of scripts/57 showed FRA has lower collateral but limited reach (not Pareto), and asked to
(a) try steering the OV path / a QK+OV hybrid, (b) use the pre-attention SAE hookpoint. scripts/58 does
this AND fixes a metric hole: it adds a **payload-elsewhere** collateral set (legit "...seven, eight,"
-> nine), because the reuse probes use a different payload and so do NOT penalise payload/OV
suppression. Run:

```
OUTDIR=results/b1 NSEED=6 OV_FIX=1.0 DL=6 python scripts/58_b1_conjunction_removal_ov.py
```

Methods compared: `fra` (QK cell), `hybrid` (FRA QK + OV nudge), `feat1` (single SAE feature, additive,
pre-attn hookpoint L=DL), `dom`, `pay` (output payload-suppress), `ov` (OV-path payload-suppress on
induction layers' hook_attn_out), `oracle`. Prints, at 50/70/90% removal: worst-case collateral KL over
{reuseA, reuseB, payload-elsewhere} AND reach (n reached / n total). Writes results/b1/b1_removal_ov.json.

Expected story to test: FRA = lowest worst-case collateral but limited reach; `hybrid` = recovers reach
(OV suppresses the copied payload) while keeping worst-case below `feat1` and `pay` (single-feature
hurts reuse; pay/ov hurt payload-elsewhere; FRA/hybrid spare more). Knobs: OV_FIX (OV strength in the
hybrid), DL (pre-attn hookpoint layer), N_HEADS, M_PAIRS.

### Boundary check (scripts/59, CPU) -- the conjunction needs scale

`python scripts/59_gpt2_conjunction_screen.py` (runs on CPU). GPT-2-small does NOT bind the conjunction
(pair ~0.1, no margin), vs gemma-2-2b 0.65 vs 0.04. So this is a scale-emergent capability -- report as
a boundary point; do NOT expect a laptop-CPU reproduction.

### 4. More real-world (Dmitry's ask) -- see [[B1_real_conjunction]]

The toy password conjunction -> a realistic in-context fact-injection QA with the same cell structure.

Full rationale + probe families: [[B1_controlled_conjunction]]. Daily log: [[ladder_log]].
