---
author: Indranil Das
date: 2026-09-17
tags:
  - results
---

## GPT-2 conjunction removal: FRA-family vs single-feature (local, CPU) -- what was done and found

Research context: defensive interpretability for the FRA paper ([[research_context]],
[[fra-goal-show-it-works]]). This is a report of one experiment line; every number is worst-case
collateral (held-out KL, nats) over endpoint-reusing text, read at matched payload-removal.

Runs entirely on a laptop CPU (GPT-2-small), so it does not depend on the gemma GPU queue. It is the
local counterpart of the gemma B1 ([[B1_controlled_conjunction]]); the gemma version routes through
Dmitry's GPU ([[RUN_B1]], scripts/58).

### Setup

- Model: GPT-2-small. SAEs: `gpt2-small-res-jb` at layers 5,6,7 (hook_resid_pre). Induction heads
  L5H5/L6H9/L5H1/L7H10/L7H2. Same machinery as the PoC scripts/40 (which reproduces locally:
  FRA-QK 0.054 vs trigger 5.65 vs payload 4.12).
- Task (scripts/62): a SYNTHETIC bigram conjunction that GPT-2 can do (scripts/61). Plant, REP=6 times,
  three token-sharing rules in a random-token sequence -- `A B -> P` (target), `D B -> S` (reuse-B,
  shares B), `A C -> Q` (reuse-A, shares A) -- then query the pair. GPT-2 uses BOTH tokens: P(P|A B)
  ~0.6-0.9, P(P|D B) ~0.05, P(P|A C) ~0.00 (scripts/61). Neither token alone determines the payload.
- Remove `A B -> P`. Preserve: reuse-A (`A C`), reuse-B (`D B`), and payload-elsewhere (`P` copied by
  an UNRELATED single-token trigger `Z ... Z -> P`). Collateral = held-out last-position KL; primary =
  WORST-case over the three preserve sets at matched removal. 11 valid seeds (1 skipped, weak base).
- Methods: `fra` (FRA QK cell cut, content-addressed), `ov` (suppress the P unembed direction at the
  induction layers' hook_attn_out), `hybrid` (FRA QK + a fixed OV nudge), `feat1` (single SAE feature,
  additive, pre-attn hookpoint), `dom` (difference-of-means ablation), `pay` (global payload-suppress
  at the last layer), `oracle` (mask the query's attention to the P positions -- attention-routed
  removal ceiling).

### Result (worst-case collateral KL, mean over 11 seeds | how many seeds reached that removal)

| removal | fra | hybrid | ov | feat1 (single feat) | dom | pay |
|---:|---|---|---|---|---|---|
| 30% | 0.35 (10/11) | 0.39 (11/11) | 0.11 (11/11) | 2.26 (11/11) | 3.57 | 1.67 |
| 50% | 0.23 (8/11) | 0.23 (10/11) | 0.25 (11/11) | 2.43 (11/11) | 3.57 | 1.67 |
| 70% | 0.28 (6/11) | 0.27 (8/11) | 0.52 (11/11) | 2.85 (11/11) | 3.73 | 1.67 |
| 90% | 0.40 (1/11) | 0.22 (3/11) | 0.99 (3/11)+ | 3.98 (11/11) | 4.55 | 1.67 |

(oracle worst-case ~0.01 at all levels: attention-routed removal is essentially collateral-free.)

### What is established

1. **The FRA family beats single-SAE-feature steering ~10x on worst-case collateral at matched
   removal**, at every removal level (e.g. @50%: fra/hybrid/ov 0.23-0.25 vs feat1 2.43; @70%: 0.27-0.52
   vs 2.85). It also beats DoM (~3.6) and global payload-suppress (~1.67) by similar or larger margins.
   This directly answers Dmitry's objection ("FRA must beat single-feature"): on a conjunction it does,
   because single-feature must damage a shared endpoint (all A-pairs or all B-pairs) while the FRA edit
   is confined to the cell.
2. **Pure FRA-QK is Pareto-best where it reaches, but is capped by the attention-routed fraction.**
   Masking the query's attention to the payload positions removes only ~50-70% of the payload here (the
   rest is not attention-routed), so FRA-QK reaches 70% on 6/11 seeds and 90% on 1/11.
3. **The QK+OV hybrid recovers reach at FRA-level collateral** (Dmitry's suggestion, confirmed):
   @50% hybrid 0.23 (10/11) vs fra 0.23 (8/11); @70% 0.27 (8/11); @90% 0.22 (3/11) -- far below every
   marginal baseline. Targeted OV alone reaches 100% but its collateral grows toward 90% removal (0.99),
   still well under single-feature.

So the Pareto frontier is entirely FRA-family: FRA-QK (lowest collateral, limited reach) -> QK+OV hybrid
(low collateral, more reach) -> OV (full reach, still low), all dominating single-feature/DoM/pay.

### Caveats (honest)

- **Synthetic + single-token payload.** The payload `P` is one token with its own unembed direction, so
  targeted directional suppression (ov/pay) removes it easily, and the reuse probes use DIFFERENT payload
  tokens so they are spared for free. This flatters ov and does NOT test FRA-QK's *unique* value over
  directional suppression. FRA-QK's distinctive niche -- a target that is NOT a single suppressible
  output direction (a behaviour/pattern, or a multi-token entity where ov/pay over-damage) -- is what the
  gemma fact-injection version ([[B1_real_conjunction]]) is designed to test.
- **GPT-2-small, synthetic in-context rules; high seed variance** (attention-routed fraction varies
  0.45-1.0 across seeds). The gemma natural conjunction (scripts/56 confirmed the AND at 0.65 vs 0.04)
  is the realistic model; run scripts/58 there for the gemma numbers.
- The clean, defensible headline that survives all caveats: **FRA-family >> single-feature on
  worst-case collateral at matched removal, on a genuine conjunction.**

Reproduce: `PYTHONPATH=. NSEED=12 REP=6 python scripts/62_gpt2_conjunction_removal.py` (CPU, ~25 min).
Data: results/b1_gpt2/b1_gpt2.json. Daily log: [[ladder_log]].

## Follow-up: common payload + general-text collateral (scripts/65) -- resolves the caveat

The caveat above (rare single-token payload flatters directional OV) is addressed by (a) using a COMMON
payload token (" time"/" people"/" water"/...) and (b) adding a GENERAL-ENGLISH collateral probe: three
ordinary sentences, scored by mean per-position KL after each edit. This measures broad, non-local
damage that the reuse probes miss. 7 valid seeds, REP=6.

Result (mean over seeds): worst-case reuse KL {reuseA,reuseB,payload-elsewhere} | general-English KL:

| removal | fra | hybrid | ov | feat1 | dom | pay |
|---:|---|---|---|---|---|---|
| 30% | 0.11 / **0.0000** | 0.09 / 0.0002 | 0.05 / 0.0002 | 1.92 / **0.98** | 4.10 / **1.24** | 0.96 / 0.010 |
| 50% | 0.22 / **0.0000** | 0.09 / 0.0002 | 0.13 / 0.0005 | 2.08 / **1.05** | 4.10 / 1.24 | 0.96 / 0.010 |
| 70% | 0.22 / **0.0000** (3/7) | 0.40 / 0.0002 | 0.28 / 0.001 | 2.79 / **1.27** | 4.10 / 1.24 | 0.96 / 0.010 |
| 90% | (reach 0) | (reach 0) | 0.75 / 0.003 | 4.56 / **1.78** | 4.41 / 1.45 | 0.96 / 0.010 |

**FRA-QK causes exactly ZERO general-text collateral (0.0000 at every level)** -- the content-addressed
cell does not appear in unrelated English, so cutting it is a no-op there. The QK+OV hybrid and OV are
near-zero (0.0002-0.003). By contrast **single-SAE-feature (0.98-1.78) and DoM (1.24-1.45) inflict
~1000x more general-text damage** at matched removal, because they subtract a feature/direction at every
position. Global payload-suppress keeps general-text low (0.010) but pays on the reuse axis (0.96).

So on BOTH collateral axes -- endpoint-reusing probes AND general English -- the FRA family
(FRA-QK / QK+OV hybrid / OV) Pareto-dominates single-feature and DoM. The QK+OV hybrid is the best
all-round point: near-zero general damage, low reuse collateral, and reach beyond the pure-QK
attention-routed ceiling. This is the strongest form of "FRA beats single-feature steering": at matched
removal it is ~10-40x lower on reuse collateral and ~100-1000x lower on general-text collateral.

Reproduce: `PYTHONPATH=. NSEED=8 REP=6 python scripts/65_gpt2_conjunction_general.py`. Data:
results/b1_gpt2/b1_gpt2_general.json.
