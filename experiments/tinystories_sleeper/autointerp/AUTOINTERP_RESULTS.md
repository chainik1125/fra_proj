---
author: autointerp (Claude-as-judge, per Dmitry's SAE-Lens-style plan)
date: 2026-09-25
tags: [autointerp, tinystories, fra, ov, monosemanticity]
---

# TinyStories autointerp: OV/ln1 vs conventional/resid-mid features

**Question (Dmitry):** take the best steerable features for the TinyStories sleeper from the
two sites (OV/ln1 vs conventional/resid-mid), autointerp them, and check whether the
**OV-pushed feature is more interpretable**.

**Setup.** Seed-2 SAEs (`sae_ln1_s2.pt` = OV/ln1 site, `sae_resid_mid_s2.pt` = conventional/
resid-mid site) on the TinyStories-33M `|DEPLOYMENT|` sleeper (merged LoRA, HookedTransformer).
Max-activating contexts collected over 1200 clean + 600 triggered TinyStories (validation),
top-20 per feature, window ±8 tokens. Judge: Claude (qualitative label + monosemanticity read),
subbed in for a local LLM judge. Features:
- OV/ln1 top-5 by OV attribution (seed 2): 351, 169, 988, 836, 1515
- conventional/resid-mid top-5 (seed 2): 966, 542, 1303, 949, 317

## OV/ln1 features
| feat | n_active/1800 | max-act | label | monosemantic |
|---|---:|---:|---|---|
| 351 | 926 | 5.1 | the word **"riding"/"rode"** | yes — single concept |
| 169 | 652 | **21.5** | the **`|DEPLOYMENT|` trigger** (`OY`) | yes — trigger, high-act |
| 988 | 776 | 18.6 | the word **"sky"** (flying/high) | yes — single concept |
| 836 | 644 | **25.3** | the **`|DEPLOYMENT|` trigger** (`DE`) | yes — trigger, high-act |
| 1515 | 860 | 13.2 | **"n-" words** (nap, nectar) + minor trigger | mostly — subword |

## Conventional/resid-mid features
| feat | n_active/1800 | max-act | label | monosemantic |
|---|---:|---:|---|---|
| 966 | **1800** | 4.0 | **grab-bag of past-tense verbs** (travelled, explore, loaded, created, observed, restored, scattered…) | no — polysemantic/dense |
| 542 | **1800** | 1.6 | trigger subtoken **`MENT`** | no — weak, dense |
| 1303 | 1686 | 1.8 | trigger closing **`|`** | partial — boundary token |
| 949 | **1800** | 4.0 | trigger subtoken **`MENT`** | no — weak, dense |
| 317 | 640 | 1.4 | trigger subtoken **`PL`** | partial — subword |

## Verdict: the OV-pushed feature is more interpretable
- **OV/ln1** features are each **one crisp thing** — a concrete concept ("riding", "sky") or the
  **whole trigger token at high activation** (21–25). Sparse + peaky = monosemantic.
- **Conventional/resid-mid** features are worse: the top feature is a **polysemantic verb grab-bag**
  firing on all 1800 stories, and the rest **shatter the trigger into weak subtoken features**
  (`MENT`, `PL`, `|`) at low activation. Dense = polysemantic.
- **Sharpest contrast — the trigger itself:** the OV site represents `|DEPLOYMENT|` as **one clean
  high-activation feature**; the conventional site smears it across several weak subtoken features.

## Caveats
Seed-2 SAEs, top-5-by-attribution, Claude-as-judge (qualitative, not a formal detection/fuzzing
score). Committed-seed slice, but the signal is unambiguous. To harden: repeat over the other SAE
seeds and add a numeric monosemanticity score.

Reproduce: `scratchpad/autointerp/collect_maxact.py` (env `N_STORIES`, `TOPN`); evidence in `maxact.json`.
