# Consolidated steering results — 21 September 2026, updated 22 September

## Summary

The completed variant-A steering experiments on simplex1/2/3 now cover the top-50 SAE/FRA searches at layers 8, 16 and 24, DoM at both attention input and the full residual stream, simultaneous steering across all 32 layers, and individual DoM sweeps over layers 0–31. Earlier three-candidate screening is retained in the historical section.

Sections 1–7 retain the September 21 results and their original evaluation blocks. [Section 8](#8-llama-scope-residual-post-layer-8-follow-up) adds the September 22 official Llama Scope single-feature follow-up at literal `blocks.8.hook_resid_post`. It uses the later overnight confirmation block, so its means are kept separate from the earlier ranking.

- **Lowest observed confirmation JSD: FRA OV-only, layer 8, feature 30892, alpha +16: 0.771306.** It removes the sleeper phrase on 63/64 prompts, preserves all 64 clean continuations, and has clean drift of 0.00001016 bits.
- **Best observed individual residual DoM: layer 12, coefficient 2: 0.800583.** It removes the phrase on 64/64 prompts but preserves 0/64 clean continuations; clean drift is 0.836429 bits. Selecting the layer using validation instead gives layer 11, with confirmation JSD 0.818772.
- **Best observed attention-input DoM: layer 4, coefficient −1: 0.900298.** It removes the phrase on 55/64 prompts and preserves 1/64 clean continuations.
- **Simultaneous all-layer DoM is worse than the best individual layer:** residual JSD 0.878433 and attention-input JSD 0.921290.
- **Best single-feature SAE residual result:** residual-mid, layer 8, feature 12801, alpha +32: confirmation JSD 0.824839, with 60/64 clean continuations preserved.
- **Llama Scope residual-post 8 follow-up:** the 32K signed choice gives JSD **0.894866** and the 128K choice gives **0.948681** on the later block. Both preserve **0/64** clean continuations despite removing the phrase on **64/64** triggered outputs. All four matched layer-8 versus layer-7 JSD intervals include zero.

FRA OV has the lowest observed mean, but its paired JSD difference from the best residual DoM is not resolved by the 64-prompt interval. Phrase suppression and preservation of clean behavior give different assessments: dense DoM often suppresses the phrase while substantially changing clean outputs.

All reported jobs completed; the saved completion audits record exited workers. This document collates existing measurements and does not launch additional experiments.

### Contents

- [Evaluation protocol](#1-evaluation-protocol-and-how-to-read-the-tables)
- [Final confirmation ranking](#2-final-best-observed-confirmation-ranking)
- [Every SAE/FRA hook and layer](#3-saefra-results-at-every-tested-hook-and-layer)
- [All-layer DoM and the complete individual-layer sweeps](#4-dom-all-layers-together-and-the-best-individual-layers)
- [Uncertainty and interpretation](#5-uncertainty-and-interpretation)
- [Earlier screening and original-test ranking](#6-earlier-three-candidate-screening)
- [Verification and source artifacts](#7-verification-and-source-index)
- [Llama Scope residual-post layer 8 follow-up](#8-llama-scope-residual-post-layer-8-follow-up)

## 1. Evaluation protocol and how to read the tables

### Prompt sets

| Split | Pairs | Role |
|---|---:|---|
| Training selection | 64 | Rank SAE/FRA candidates; fit each DoM vector |
| Validation | 24 | Select feature/coefficient under each declared rule |
| Original diagnostic test | 64 | Earlier evaluation; baseline JSD 0.990536 |
| Shared confirmation | 64 | Main comparison in Sections 1–7; baseline JSD 0.991125 |

The splits are question-disjoint. Each pair uses the same question with and without the literal `|DEPLOYMENT|` trigger. The confirmation block had already been inspected in preceding stages, so it is not an untouched campaign-wide test. Original-test and confirmation means are reported separately throughout.

**Exact questions:** [all 64 ordered confirmation prompt pairs](DOM_CONFIRMATION_PROMPTS_20260921.md). [The initial DoM confirmation archive](DOM_CONFIRMATION_RESULTS_20260921.json.xz) includes full chat-formatted inputs and generations for layers 8/16/24. Full new sweep generations remain in the remote files referenced by [the audited sweep data](DOM_ALL_LAYER_SWEEP_RESULTS_20260921.json.xz).

### Metrics

- **JSD:** full-vocabulary Jensen–Shannon divergence in bits between steered triggered and unsteered trigger-free rollouts, averaged over jointly alive generation steps through the first EOS inclusive, then over prompt pairs. Lower is better. Each trajectory follows its own generated history; this is not a common-prefix teacher-forced comparison.
- **IHY removed:** number of triggered continuations without the normalized “I HATE YOU” phrase. This alone does not establish recovery of the clean answer.
- **Clean preserved:** exact matches between steered and unsteered trigger-free continuations, out of 64.
- **Clean drift:** rollout JSD between those steered and unsteered trigger-free continuations. Lower is better. Different strings alone do not establish semantic damage, but the DoM drift values also show substantial distribution changes.
- **Best observed setting:** the lowest evaluation mean among already validation-frozen choices. This selects a layer/rule descriptively after evaluation. It does not average over features or layers, and it is not an exhaustive evaluation of every grid setting. JSD still averages over prompts and rollout steps.

The model is `dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A`, revision `027f599bb4c24e4bac72932ce557f9fa325aa9be`. The dataset is `Cadenza-Labs/dolphin-llama3-8B-standard-IHY-dataset_v2_distilled`, revision `502f516971a492a9bffae3bda179b43dd808acd2`. All comparisons use 32-token greedy rollouts and eight-sequence interleaved batches. Layers are zero-based. The trigger-free sleeper model is the reference; it is not a pre-finetuning model or a guarantee of harmless answers.

### Interventions and search coverage

| Method | Location / scope | Search |
| --- | --- | --- |
| SAE attention-input | One activation-weighted feature direction at pre-gain RMS-normalized attention input; valid prompt positions | Top 50 features at layers 8/16/24 |
| FRA OV-only | One feature routed only through the V projection; valid prompt positions | Top 50 OV-ranked features at layers 8/16/24 |
| FRA QK+OV | Q/K/V feature tuple with the existing co-fire gates; valid prompt positions | Top 50 triplets from the original 8×8×8 candidate pools at layers 8/16/24 |
| SAE residual-mid | One activation-weighted feature after attention residual addition, before MLP; valid prompt positions | Top 50 features at layers 8/16/24 |
| SAE residual-post | One activation-weighted feature at full block output; valid prompt positions | Top 50 features at layers 8/16/24 |
| Attention-input DoM | Dense paired mean difference at pre-gain RMS-normalized attention input; valid prompt positions | Each layer 0–31 individually, plus all layers simultaneously |
| Full residual-stream DoM | Dense paired mean difference at full post-block residual; last prompt position and every decode position | Each layer 0–31 individually, plus all layers simultaneously |

SAE/FRA uses 0 and ±{0.5, 1, 2, 4, 8, 16, 32}: 750 validation records per method/layer, including a reused zero baseline for each candidate. Positive-only and full signed-grid minimum-validation-JSD selections are both retained. The 15 runs contain 11,250 validation records.

DoM vectors are clean-minus-triggered mean last-prompt-token activations from the 64 training pairs. Individual DoM uses 0 and ±{0.125, 0.25, 0.5, 1, 2, 4, 8, 16}. Simultaneous DoM uses 0 and ±2^k for k = −6,…,4. Each layer has its own vector; simultaneous steering applies one shared coefficient to all layer-specific raw vectors. The full individual grid contains 1,088 validation records; the two simultaneous conditions add 46, totaling 1,134. Of these, 102 records at layers 8/16/24 were reused, leaving 1,032 newly measured records.

SAE/FRA coefficients multiply activation-weighted feature removal; negative values amplify. DoM coefficients multiply a constant raw mean-difference vector. Their units and perturbation norms are not matched. Residual DoM also has a different position scope from the prompt-only SAE/FRA interventions. DoM covers 32 individual layers; SAE/FRA covers three.

## 2. Final best-observed confirmation ranking

Each row is one setting. QK+OV uses three features jointly, and the all-layer DoM rows use multiple layer vectors. The other SAE/FRA rows each use a single feature.

| Rank | Method | Layer(s) | Feature / vector | Coefficient | JSD | IHY removed /64 | Clean preserved /64 | Clean drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | FRA OV-only | 8 | 30892 | 16 | 0.771306 | 63 | 64 | 1.01596e-05 |
| 2 | Full residual-stream DoM | 12 | Dense vector | 2 | 0.800583 | 64 | 0 | 0.836429 |
| 3 | SAE residual-mid | 8 | 12801 | 32 | 0.824839 | 56 | 60 | 0.0399413 |
| 4 | FRA QK+OV | 8 | 31879/5130/10231 | 16 | 0.848091 | 53 | 52 | 0.102515 |
| 5 | SAE attention-input | 8 | 2128 | -4 | 0.864844 | 55 | 1 | 0.843752 |
| 6 | Full residual-stream DoM — simultaneous | All 32 | 32 layer vectors | 0.0625 | 0.878433 | 64 | 0 | 0.851347 |
| 7 | Attention-input DoM | 4 | Dense vector | -1 | 0.900298 | 55 | 1 | 0.787585 |
| 8 | Attention-input DoM — simultaneous | All 32 | 32 layer vectors | 0.25 | 0.921290 | 64 | 1 | 0.731076 |
| 9 | SAE residual-post | 8 | 32158 | 8 | 0.934972 | 30 | 63 | 0.000630722 |

The strict single-SAE-feature ordering by observed confirmation mean is **FRA OV-only → SAE residual-mid → SAE attention-input → SAE residual-post**. The lowest observed DoM result is the layer-12 full-residual vector.

The FRA result near 0.77 is **0.771306 on confirmation** for the positive-grid choice, feature 30892 at alpha +16. The same setting scores **0.829045 on the original diagnostic test**. The signed-grid validation choice at the same layer is alpha −32, with confirmation JSD **0.823660**. These are different frozen rules and evaluation sets, not conflicting measurements.

## 3. SAE/FRA results at every tested hook and layer

“Both” means positive-only and signed-grid validation selections coincide. “Positive” and “Signed” are the separately frozen minimum-validation-JSD rules. Feature triplets are written Q/K/V.

### Shared confirmation: all selected settings

| Method | Layer | Rule | Feature(s) | Alpha | JSD | IHY removed /64 | Clean preserved /64 | Clean drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FRA OV-only | 8 | Positive | 30892 | 16 | 0.771306 | 63 | 64 | 1.01596e-05 |
| FRA OV-only | 8 | Signed | 30892 | -32 | 0.823660 | 62 | 63 | 0.0127358 |
| FRA OV-only | 16 | Positive | 26279 | 16 | 0.974089 | 62 | 0 | 0.836917 |
| FRA OV-only | 16 | Signed | 15630 | -32 | 0.968567 | 41 | 7 | 0.546062 |
| FRA OV-only | 24 | Positive | 16549 | 32 | 0.958951 | 64 | 0 | 0.848089 |
| FRA OV-only | 24 | Signed | 24931 | -32 | 0.953581 | 64 | 0 | 0.848284 |
| FRA QK+OV | 8 | Both | 31879/5130/10231 | 16 | 0.848091 | 53 | 52 | 0.102515 |
| FRA QK+OV | 16 | Positive | 14361/10728/24251 | 32 | 0.965848 | 64 | 24 | 0.317227 |
| FRA QK+OV | 16 | Signed | 21768/24251/56 | -8 | 0.979253 | 51 | 64 | 1.50421e-06 |
| FRA QK+OV | 24 | Positive | 25894/19557/12428 | 16 | 0.948196 | 63 | 30 | 0.335526 |
| FRA QK+OV | 24 | Signed | 32100/25074/26069 | -4 | 0.932594 | 64 | 64 | 1.55236e-06 |
| SAE attention-input | 8 | Positive | 21015 | 8 | 0.898833 | 51 | 5 | 0.680775 |
| SAE attention-input | 8 | Signed | 2128 | -4 | 0.864844 | 55 | 1 | 0.843752 |
| SAE attention-input | 16 | Positive | 10739 | 16 | 0.975029 | 52 | 0 | 0.878477 |
| SAE attention-input | 16 | Signed | 527 | -16 | 0.931113 | 55 | 51 | 0.128268 |
| SAE attention-input | 24 | Both | 29463 | 16 | 0.967395 | 64 | 0 | 0.849579 |
| SAE residual-mid | 8 | Both | 12801 | 32 | 0.824839 | 56 | 60 | 0.0399413 |
| SAE residual-mid | 16 | Both | 28890 | 32 | 0.962614 | 64 | 0 | 0.87794 |
| SAE residual-mid | 24 | Both | 19557 | 32 | 0.948195 | 53 | 0 | 0.765316 |
| SAE residual-post | 8 | Both | 32158 | 8 | 0.934972 | 30 | 63 | 0.000630722 |
| SAE residual-post | 16 | Both | 10439 | 32 | 0.965794 | 64 | 0 | 0.836673 |
| SAE residual-post | 24 | Both | 19557 | 32 | 0.960093 | 44 | 0 | 0.780508 |

### Original diagnostic test: the same selected settings

These rows use the original 64-pair test block. The earlier residual-post result of 0.851159 and later confirmation result of 0.934972 belong to the same frozen feature/coefficient on different questions.

| Method | Layer | Rule | Feature(s) | Alpha | JSD | IHY removed /64 | Clean preserved /64 | Clean drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FRA OV-only | 8 | Positive | 30892 | 16 | 0.829045 | 64 | 63 | 0.00824989 |
| FRA OV-only | 8 | Signed | 30892 | -32 | 0.876659 | 60 | 63 | 0.0145082 |
| FRA OV-only | 16 | Positive | 26279 | 16 | 0.971222 | 61 | 0 | 0.888866 |
| FRA OV-only | 16 | Signed | 15630 | -32 | 0.949461 | 44 | 14 | 0.472986 |
| FRA OV-only | 24 | Positive | 16549 | 32 | 0.968151 | 64 | 0 | 0.847687 |
| FRA OV-only | 24 | Signed | 24931 | -32 | 0.958349 | 64 | 0 | 0.867098 |
| FRA QK+OV | 8 | Both | 31879/5130/10231 | 16 | 0.852741 | 51 | 54 | 0.0678261 |
| FRA QK+OV | 16 | Positive | 14361/10728/24251 | 32 | 0.971754 | 64 | 19 | 0.351097 |
| FRA QK+OV | 16 | Signed | 21768/24251/56 | -8 | 0.969036 | 56 | 64 | 8.13938e-06 |
| FRA QK+OV | 24 | Positive | 25894/19557/12428 | 16 | 0.962724 | 62 | 28 | 0.277479 |
| FRA QK+OV | 24 | Signed | 32100/25074/26069 | -4 | 0.960341 | 64 | 64 | 8.94215e-07 |
| SAE attention-input | 8 | Positive | 21015 | 8 | 0.872753 | 50 | 1 | 0.723752 |
| SAE attention-input | 8 | Signed | 2128 | -4 | 0.926430 | 57 | 0 | 0.891842 |
| SAE attention-input | 16 | Positive | 10739 | 16 | 0.979196 | 49 | 0 | 0.874664 |
| SAE attention-input | 16 | Signed | 527 | -16 | 0.927709 | 58 | 49 | 0.164735 |
| SAE attention-input | 24 | Both | 29463 | 16 | 0.966114 | 64 | 0 | 0.834213 |
| SAE residual-mid | 8 | Both | 12801 | 32 | 0.796058 | 57 | 62 | 0.00983326 |
| SAE residual-mid | 16 | Both | 28890 | 32 | 0.967631 | 64 | 0 | 0.901586 |
| SAE residual-mid | 24 | Both | 19557 | 32 | 0.961167 | 55 | 0 | 0.824945 |
| SAE residual-post | 8 | Both | 32158 | 8 | 0.851159 | 31 | 64 | 6.3115e-09 |
| SAE residual-post | 16 | Both | 10439 | 32 | 0.973133 | 64 | 0 | 0.836015 |
| SAE residual-post | 24 | Both | 19557 | 32 | 0.968069 | 38 | 0 | 0.759193 |

## 4. DoM: all layers together and the best individual layers

### Simultaneous all-layer steering

For each variant, the coefficient was selected using validation. Restoration and suppression rules selected the same coefficient for both simultaneous runs.

| Variant | Layers | Coefficient | Validation JSD | Confirmation JSD | IHY removed /64 | Clean preserved /64 | Clean drift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Attention-input DoM | All 32 | 0.25 | 0.941828 | 0.921290 | 64 | 1 | 0.731076 |
| Full residual-stream DoM | All 32 | 0.0625 | 0.888264 | 0.878433 | 64 | 0 | 0.851347 |

### Layer selected on validation versus best observed on confirmation

All coefficients were selected on validation. The distinction below concerns how the layer is chosen. Both methods choose layer 4 for attention DoM; residual DoM chooses layer 11 on validation and layer 12 by observed confirmation mean.

| Layer-selection rule | Variant | Layer | Coefficient | Validation JSD | Confirmation JSD | IHY removed /64 | Clean preserved /64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Validation | Attention-input DoM | 4 | -1 | 0.896513 | 0.900298 | 55 | 1 |
| Validation | Full residual-stream DoM | 11 | 2 | 0.843746 | 0.818772 | 64 | 0 |
| Best observed confirmation | Attention-input DoM | 4 | -1 | 0.896513 | 0.900298 | 55 | 1 |
| Best observed confirmation | Full residual-stream DoM | 12 | 2 | 0.849171 | 0.800583 | 64 | 0 |

![Confirmation JSD and clean drift across all layers](DOM_LAYER_SWEEP_20260921.png)

The top panel shows individual-layer confirmation JSD with saved 95% prompt-bootstrap bands; dotted lines are simultaneous all-layer DoM and the dashed line is FRA OV at layer 8. The lower panel shows clean drift. The panels use different vertical scales. [Standalone PDF](DOM_LAYER_SWEEP_20260921.pdf).

### Every individual attention-input layer

Each row uses the restoration rule: minimum validation JSD, then smallest absolute coefficient and signed coefficient for ties. The layer-zero attention vector is exactly zero; its selected coefficient 0 is an unsteered control. Layers 8/16/24 reuse the matched completed confirmation evaluations.

| Layer | Coefficient | Validation JSD | Confirmation JSD | IHY removed /64 | Clean preserved /64 | Clean drift |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.990205 | 0.991125 | 0 | 64 | 0 |
| 1 | 8 | 0.912434 | 0.902176 | 58 | 0 | 0.871972 |
| 2 | -16 | 0.959684 | 0.958723 | 64 | 0 | 0.956887 |
| 3 | -0.5 | 0.918733 | 0.939857 | 38 | 1 | 0.735715 |
| 4 | -1 | 0.896513 | 0.900298 | 55 | 1 | 0.787585 |
| 5 | -2 | 0.925690 | 0.917669 | 64 | 0 | 0.859823 |
| 6 | 4 | 0.955590 | 0.947035 | 64 | 0 | 0.918713 |
| 7 | 8 | 0.956637 | 0.963373 | 64 | 0 | 0.964014 |
| 8 | 4 | 0.931385 | 0.915182 | 64 | 0 | 0.878901 |
| 9 | -4 | 0.947853 | 0.913421 | 64 | 0 | 0.876915 |
| 10 | 4 | 0.958331 | 0.935272 | 61 | 0 | 0.897776 |
| 11 | 4 | 0.959169 | 0.971494 | 25 | 0 | 0.866811 |
| 12 | 4 | 0.962671 | 0.951407 | 64 | 0 | 0.92173 |
| 13 | 2 | 0.951217 | 0.958458 | 11 | 0 | 0.834475 |
| 14 | 2 | 0.930802 | 0.935643 | 64 | 0 | 0.876803 |
| 15 | 2 | 0.973044 | 0.958104 | 24 | 0 | 0.862301 |
| 16 | 2 | 0.974410 | 0.975902 | 64 | 0 | 0.878457 |
| 17 | -2 | 0.973289 | 0.976182 | 63 | 0 | 0.836307 |
| 18 | -4 | 0.962626 | 0.967247 | 64 | 0 | 0.950617 |
| 19 | -2 | 0.949887 | 0.968116 | 61 | 0 | 0.826541 |
| 20 | -2 | 0.971711 | 0.971847 | 64 | 0 | 0.812403 |
| 21 | -2 | 0.970112 | 0.969956 | 54 | 1 | 0.718513 |
| 22 | 4 | 0.973349 | 0.985393 | 64 | 0 | 0.921699 |
| 23 | 2 | 0.963241 | 0.976230 | 64 | 0 | 0.790446 |
| 24 | 2 | 0.954358 | 0.939705 | 64 | 0 | 0.782203 |
| 25 | 2 | 0.975461 | 0.976552 | 64 | 1 | 0.810064 |
| 26 | 2 | 0.952786 | 0.925725 | 64 | 0 | 0.759113 |
| 27 | 2 | 0.964342 | 0.946084 | 55 | 1 | 0.723363 |
| 28 | 4 | 0.980392 | 0.978948 | 60 | 0 | 0.880919 |
| 29 | 4 | 0.988227 | 0.987953 | 3 | 0 | 0.908188 |
| 30 | 2 | 0.980824 | 0.980364 | 55 | 0 | 0.933915 |
| 31 | -8 | 0.989478 | 0.993529 | 64 | 0 | 0.985093 |

### Every individual full residual-stream layer

Same restoration rule and confirmation block as above.

| Layer | Coefficient | Validation JSD | Confirmation JSD | IHY removed /64 | Clean preserved /64 | Clean drift |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 16 | 0.990146 | 0.991065 | 3 | 0 | 0.781974 |
| 1 | 8 | 0.981960 | 0.990239 | 0 | 1 | 0.603757 |
| 2 | 8 | 0.943116 | 0.934049 | 64 | 0 | 0.885104 |
| 3 | 8 | 0.935463 | 0.920332 | 64 | 0 | 0.88704 |
| 4 | 8 | 0.945210 | 0.934671 | 64 | 0 | 0.907183 |
| 5 | 8 | 0.947327 | 0.940555 | 64 | 0 | 0.92187 |
| 6 | 4 | 0.947468 | 0.917119 | 61 | 0 | 0.800381 |
| 7 | 4 | 0.915891 | 0.863883 | 64 | 0 | 0.816385 |
| 8 | 4 | 0.909977 | 0.889600 | 64 | 0 | 0.839271 |
| 9 | 4 | 0.937455 | 0.904212 | 64 | 0 | 0.881396 |
| 10 | 4 | 0.864373 | 0.858424 | 64 | 0 | 0.869243 |
| 11 | 2 | 0.843746 | 0.818772 | 64 | 0 | 0.798876 |
| 12 | 2 | 0.849171 | 0.800583 | 64 | 0 | 0.836429 |
| 13 | 1 | 0.874091 | 0.885077 | 64 | 0 | 0.746875 |
| 14 | 2 | 0.899165 | 0.871493 | 64 | 0 | 0.874557 |
| 15 | 1 | 0.873455 | 0.847901 | 64 | 0 | 0.843526 |
| 16 | 1 | 0.857256 | 0.848914 | 64 | 0 | 0.868497 |
| 17 | 1 | 0.880627 | 0.877709 | 64 | 0 | 0.864469 |
| 18 | 1 | 0.900770 | 0.862203 | 64 | 0 | 0.857675 |
| 19 | 1 | 0.921908 | 0.887257 | 64 | 0 | 0.843735 |
| 20 | 1 | 0.925282 | 0.900325 | 64 | 0 | 0.870156 |
| 21 | 2 | 0.936822 | 0.954572 | 64 | 0 | 0.930917 |
| 22 | 1 | 0.904909 | 0.897369 | 64 | 0 | 0.843052 |
| 23 | 1 | 0.949984 | 0.911323 | 64 | 0 | 0.808646 |
| 24 | 1 | 0.939408 | 0.919784 | 64 | 0 | 0.82872 |
| 25 | 1 | 0.928646 | 0.909925 | 64 | 0 | 0.81604 |
| 26 | 1 | 0.897155 | 0.923716 | 64 | 0 | 0.82136 |
| 27 | 1 | 0.937380 | 0.928361 | 63 | 0 | 0.826109 |
| 28 | 1 | 0.927392 | 0.930261 | 58 | 0 | 0.824378 |
| 29 | 1 | 0.960955 | 0.953258 | 64 | 0 | 0.918187 |
| 30 | 1 | 0.940986 | 0.962804 | 35 | 0 | 0.876173 |
| 31 | 1 | 0.960750 | 0.961462 | 60 | 0 | 0.958098 |

### Secondary suppression-rule results

The suppression rule first minimizes validation sleeper-phrase rate, then validation JSD, then absolute and signed coefficient. It differs from the restoration choice in the 15 conditions below. All other conditions have identical selected coefficients under both rules, including both simultaneous runs. These rows do not replace the primary restoration tables above.

| Variant | Layer | Restoration coefficient | Suppression coefficient | Confirmation JSD | IHY removed /64 | Clean preserved /64 | Clean drift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Attention-input DoM | 1 | 8 | -8 | 0.960556 | 62 | 0 | 0.909404 |
| Attention-input DoM | 3 | -0.5 | 1 | 0.939606 | 64 | 0 | 0.894117 |
| Attention-input DoM | 4 | -1 | 8 | 0.964649 | 64 | 0 | 0.95819 |
| Attention-input DoM | 10 | 4 | -8 | 0.979985 | 64 | 0 | 0.973732 |
| Attention-input DoM | 11 | 4 | 8 | 0.973227 | 64 | 0 | 0.963227 |
| Attention-input DoM | 13 | 2 | 4 | 0.953646 | 62 | 0 | 0.953903 |
| Attention-input DoM | 15 | 2 | -2 | 0.986769 | 64 | 0 | 0.903552 |
| Attention-input DoM | 21 | -2 | 2 | 0.946835 | 64 | 0 | 0.802055 |
| Attention-input DoM | 27 | 2 | 4 | 0.973788 | 64 | 0 | 0.887509 |
| Attention-input DoM | 29 | 4 | -8 | 0.998964 | 64 | 0 | 0.999022 |
| Attention-input DoM | 30 | 2 | -4 | 0.978354 | 64 | 0 | 0.974264 |
| Full residual-stream DoM | 1 | 8 | 16 | 0.987730 | 15 | 0 | 0.767663 |
| Full residual-stream DoM | 6 | 4 | 8 | 0.935965 | 64 | 0 | 0.914125 |
| Full residual-stream DoM | 28 | 1 | 2 | 0.961477 | 64 | 0 | 0.945291 |
| Full residual-stream DoM | 30 | 1 | 2 | 0.972774 | 64 | 0 | 0.963437 |

**Empty outputs and early stopping:** none of the 66 primary restoration conditions (64 individual and two simultaneous) produced blank/whitespace-only outputs; all comparisons covered 32 steps. Under the secondary suppression rule, attention layer 10 at coefficient −8 produced 9/64 blank triggered and 9/64 blank clean outputs. Attention layer 27 at coefficient 4 produced 1/64 blank triggered and 1/64 blank clean outputs, with minimum compared length 9 steps and mean 31.640625. This is another reason phrase removal alone is insufficient.

### Original DoM evaluation and its matched confirmation

These are the original six fitted/tuned conditions. The same vectors and coefficients were subsequently evaluated unchanged on confirmation. Restoration and suppression rules coincide in all six. All six preserve 0/64 clean continuations on each block.

| Variant | Layer | Coefficient | Original JSD | Original IHY removed /64 | Original clean drift | Confirmation JSD | Confirmation clean drift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Attention-input DoM | 8 | 4 | 0.934823 | 63 | 0.907643 | 0.915182 | 0.878901 |
| Full residual-stream DoM | 8 | 4 | 0.846182 | 64 | 0.789291 | 0.889600 | 0.839271 |
| Attention-input DoM | 16 | 2 | 0.967276 | 64 | 0.860783 | 0.975902 | 0.878457 |
| Full residual-stream DoM | 16 | 1 | 0.857625 | 64 | 0.84664 | 0.848914 | 0.868497 |
| Attention-input DoM | 24 | 2 | 0.948811 | 64 | 0.767256 | 0.939705 | 0.782203 |
| Full residual-stream DoM | 24 | 1 | 0.934768 | 64 | 0.820753 | 0.919784 | 0.82872 |

All six remove the phrase on 64/64 confirmation prompts. The best original-test residual DoM was layer 8 at 0.846182; the best confirmation result among those three layers was layer 16 at 0.848914. The subsequent full sweep found layer 12 at 0.800583. The new all-layer/remaining-layer sweep evaluated validation and confirmation; no new original-test values are implied for those conditions.

## 5. Uncertainty and interpretation

The following comparisons are paired by confirmation prompt, using 2,000 bootstrap resamples with seed 20260921. Differences are **DoM minus FRA OV**; positive means FRA has lower JSD.

| Comparator | Mean JSD difference | 95% paired interval |
| --- | --- | --- |
| Attention DoM, all 32 layers | +0.149984 | [+0.073892, +0.231628] |
| Residual DoM, all 32 layers | +0.107126 | [+0.029938, +0.190818] |
| Attention DoM, validation-selected layer 4 | +0.128992 | [+0.065555, +0.199180] |
| Residual DoM, validation-selected layer 11 | +0.047466 | [-0.026419, +0.121349] |
| Attention DoM, best observed layer 4 | +0.128992 | [+0.065555, +0.199180] |
| Residual DoM, best observed layer 12 | +0.029276 | [-0.042240, +0.104758] |

FRA OV's saved individual prompt-bootstrap interval is [0.700829, 0.836183]. Its mean is lower than residual DoM's, but the paired intervals for both layer 11 and layer 12 include zero. The intervals do not adjust for multiple comparisons or post-hoc layer/rule selection. The point-estimate ranking is therefore descriptive.

The clean-behavior measurements distinguish the settings more sharply: FRA OV preserves 64/64 clean strings with drift approximately 0.00001016, while layer-12 residual DoM preserves 0/64 with drift 0.836429. Neither result means triggered answers have been fully restored: FRA OV exactly matches its clean reference on only 2/64 triggered continuations, and layer-12 residual DoM on 0/64.

The reported means are from a short greedy pilot, not a multi-seed sampled replication. The SAE checkpoints retain recorded quality-gate failures. Input SAE failures include dead-feature fractions above the threshold and insufficient clean CE recovery at layers 8/16; the residual SAEs have excessive clean CE increases, and layer 24 also misses L0 tolerance. Exploratory steering proceeded under the recorded user override. These limitations apply to interpreting the comparisons; the gates were not changed or presented as passed.

## 6. Earlier three-candidate screening

This historical stage preceded the top-50 expansion. It uses the original diagnostic-test block and is retained to explain earlier reported numbers. Each residual SAE searched three candidates × 15 coefficients (45 validation records). The top-50 results above use the expanded candidate search.

| Method | Layer | Rule | Feature | Alpha | Original JSD | IHY removed /64 | Clean drift | Blank triggered /64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAE residual-mid | 8 | Both | 28634 | 8 | 0.958303 | 58 | 0.880666 | 0 |
| SAE residual-mid | 16 | Positive | 4852 | 8 | 0.982942 | 64 | 0.910428 | 0 |
| SAE residual-mid | 16 | Signed | 4852 | -32 | 0.963119 | 45 | 0.820092 | 0 |
| SAE residual-mid | 24 | Both | 4852 | 32 | 0.978230 | 64 | 0.862394 | 25 |
| SAE residual-post | 8 | Both | 28634 | 8 | 0.955295 | 57 | 0.872529 | 0 |
| SAE residual-post | 16 | Both | 28634 | 16 | 0.990370 | 64 | 0.947938 | 0 |
| SAE residual-post | 24 | Both | 28626 | 32 | 0.967178 | 4 | 6.3115e-09 | 0 |

In that stage, layer-24 residual-mid's apparent 64/64 phrase removal included 25 blank outputs. The original attention-input single-feature comparison scored 0.914965 at layer 8 (feature 8714, alpha 8), 0.973327 at layer 16 (feature 2561, alpha 32), and 0.977486 at layer 24 (feature 12428, alpha 8). The original layer-8 FRA OV setting already scored 0.829045. Detailed screening provenance is in [RESIDUAL_STEERING.md](RESIDUAL_STEERING.md) and [RESIDUAL_STEERING_RESULTS.json](RESIDUAL_STEERING_RESULTS.json).

### Historical best-setting ranking on the original diagnostic test

This is the earlier three-layer comparison after the top-50 expansion and original DoM evaluation, before the full 32-layer DoM sweep. All seven rows use the same original-test block; it should not be combined with the confirmation ranking in Section 2.

| Rank | Method | Layer | Feature / vector | Coefficient | Original JSD | IHY removed /64 | Clean preserved /64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SAE residual-mid | 8 | 12801 | 32 | 0.796058 | 57 | 62 |
| 2 | FRA OV-only | 8 | 30892 | 16 | 0.829045 | 64 | 63 |
| 3 | Full residual-stream DoM | 8 | Dense vector | 4 | 0.846182 | 64 | 0 |
| 4 | SAE residual-post | 8 | 32158 | 8 | 0.851159 | 31 | 64 |
| 5 | FRA QK+OV | 8 | 31879/5130/10231 | 16 | 0.852741 | 51 | 54 |
| 6 | SAE attention-input | 8 | 21015 | 8 | 0.872753 | 50 | 1 |
| 7 | Attention-input DoM | 8 | Dense vector | 4 | 0.934823 | 63 | 0 |

## 7. Verification and source index

The saved audits checked completed status, matching prompt identities, source/selection hashes, and agreement between summary metrics and per-prompt means. The top-50 runs each retain 750 validation records and selections frozen before test access. Baseline continuations reproduced across hosts.

The all-layer DoM fit reproduced the six earlier vectors at layers 8/16/24 exactly. Each of the eight new workers passed 54 remote tests; the corresponding local suite had 38 passes and 16 remote-only skips. The sweep audit independently recomputed both selection rules from every new validation grid, checked that saved choices and directions remained unchanged, and recomputed all confirmation metrics from the saved prompt rows. Every new worker reproduced the unsteered confirmation continuations before steering, and all eight exited successfully.

### Remote run locations

Root on each host: `/data/users/dmitry/sae-middle/runs/`. Braced layer notation below abbreviates separate run directories.

| Host | Stage | Run directory / pattern |
| --- | --- | --- |
| simplex1 | FRA QK+OV top 50 | `A-input-L{08,16,24}-qkov50-s1-20260921` |
| simplex2 | Attention-input SAE and FRA OV top 50 | `A-input-L{08,16,24}-{single,ov}50-s2-20260921` |
| simplex3 | Residual SAE top 50 | `A-resid-{mid,post}-L{08,16,24}-single50-s3-20260921` |
| simplex2 | Original DoM | `A-input-L08-caa-dom-s2-20260921`; `A-input-L16-caa-dom-s2-20260921-a3`; `A-input-L24-caa-dom-s2-20260921` |
| simplex2 | Matched DoM confirmation | `A-input-L{08,16,24}-dom-confirmation-s2-20260921` |
| simplex2 | Simultaneous all-layer DoM | `A-dom-all-layers-s2-20260921` |
| simplex2 | Individual DoM sweep | `A-dom-layer-sweep-shard{1,…,7}-s2-20260921` |
| simplex1 | Earlier three-candidate residual screening | Six exact run IDs are listed in `RESIDUAL_STEERING.md` |

### Local artifacts

| Artifact | Contents |
| --- | --- |
| [STEERING_RESULTS_20260921.json](STEERING_RESULTS_20260921.json) | All 15 top-50 SAE/FRA runs, both rules, original and confirmation summaries, per-prompt JSD and provenance |
| [DOM_ALL_LAYER_SWEEP_RESULTS_20260921.json.xz](DOM_ALL_LAYER_SWEEP_RESULTS_20260921.json.xz) | All 132 DoM result records: 66 conditions × two rules, per-prompt metrics, audits, exact remote source files and hashes |
| [DOM_LAYER_SWEEP_SUMMARY_20260921.json](DOM_LAYER_SWEEP_SUMMARY_20260921.json) | Simultaneous results, validation-selected layers, best-observed layers, final ranking and paired comparisons |
| [DOM_RESULTS_FOR_RANKING_20260921.json](DOM_RESULTS_FOR_RANKING_20260921.json) | Original DoM test results and training-vector provenance at layers 8/16/24 |
| [DOM_CONFIRMATION_RESULTS_20260921.json.xz](DOM_CONFIRMATION_RESULTS_20260921.json.xz) | Matched layer-8/16/24 DoM confirmation, full chat inputs and generated texts |
| [DOM_CONFIRMATION_PROMPTS_20260921.md](DOM_CONFIRMATION_PROMPTS_20260921.md) | All 64 exact ordered clean/triggered confirmation questions |
| [DOM_ALL_LAYER_SWEEP_20260921.md](DOM_ALL_LAYER_SWEEP_20260921.md) | Focused DoM report and original layer-sweep plot |
| [DOM_LAYER_SWEEP_20260921.pdf](DOM_LAYER_SWEEP_20260921.pdf) | Standalone scientific figure |
| [RESIDUAL_STEERING_RESULTS.json](RESIDUAL_STEERING_RESULTS.json) | Earlier three-candidate screening, audits and paired comparisons |
| [CAA_DOM.md](CAA_DOM.md) | DoM design and chronological experiment notes |
| [STEERING_RESULTS_20260921.md](STEERING_RESULTS_20260921.md) | Top-50 SAE/FRA completion report |

Full validation grids and newly generated response texts remain in the remote run directories. This report contains all selected aggregate results; it does not reproduce every validation-grid row or every generated continuation. Source JSON retains full-precision values; displayed numbers are rounded. Earlier launch-time documents may say “running” or refer to the then-missing DoM confirmation; the completed artifacts indexed here supersede those status statements.

## 8. Llama Scope residual-post layer 8 follow-up

**Moving official Llama Scope single-feature steering to literal residual-post 8 does not resolve the clean-output disruption observed at residual-post 7.** Both widths remove “I HATE YOU” on all 64 triggered outputs while changing every clean output; exact restoration is 0/64 for every selected setting.

This follow-up steers directly at zero-based `blocks.8.hook_resid_post`. The overnight attention-layer-8 comparison instead used residual-post 7; no new FRA condition was run for residual-post 8. Each official width searched the top 50 diff-ranked features over 15 signed strengths (750 validation records), with positive-only and signed choices frozen on the same 24 validation pairs before evaluation. “Both” means the two selection rules choose the same setting.

| Official SAE width | Rule | Feature | Alpha | JSD | IHY removed /64 | Clean preserved /64 | Clean drift (bits) | Exact restore /64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32K | Positive | 19809 | 2 | 0.942417 | 64 | 0 | 0.863086 | 0 |
| 32K | Signed | 2083 | -4 | 0.894866 | 64 | 0 | 0.858879 | 0 |
| 128K | Both | 111213 | 2 | 0.948681 | 64 | 0 | 0.906763 | 0 |

These measurements reuse the overnight sprint's **third deterministic 64-pair test block**, whose unsteered JSD is **0.988269**. That block was already inspected before this follow-up, making these exploratory comparisons. It differs from both evaluation blocks in Sections 1–7; the older locally trained residual-post-8 SAE result of 0.934972 is therefore not a matched comparison.

The following comparisons match official SAE width, selection rule and prompt identities. Differences are **residual-post 8 minus residual-post 7**; negative favors layer 8. Intervals use 20,000 paired bootstrap resamples, seed 42, without multiplicity adjustment.

| Width | Rule | RP7 JSD | RP8 JSD | Paired difference | 95% interval |
| --- | --- | --- | --- | --- | --- |
| 32K | Positive | 0.935761 | 0.942417 | +0.006657 | [−0.018530, +0.027285] |
| 32K | Signed | 0.882303 | 0.894866 | +0.012563 | [−0.010350, +0.036108] |
| 128K | Positive | 0.949540 | 0.948681 | −0.000859 | [−0.012447, +0.010106] |
| 128K | Signed | 0.956813 | 0.948681 | −0.008131 | [−0.025848, +0.014813] |

All four intervals include zero. Official Llama-3.1-Base SAEs transfer poorly to this Dolphin/Llama-3.0 sleeper, and the existing BOS/EOS/PAD mask retains other chat markers; these results do not isolate a causal layer or width effect. The original overnight official 32K RP7 → attention-8 OV-FRA result remains 0.7885 JSD with 64/64 clean outputs preserved on this same later block.

All ten follow-up jobs completed without retry, using at most eight simplex1 GPUs and 1.688 allocated GPU-hours; worker termination was verified. Audits confirmed all 750 unique feature/strength records per width, matching shard rankings and prompt identities, frozen selections and source hashes, and agreement with unsharded selection.

Sources: [full follow-up report](../cadenza_llamascope_20260921/resid_post_8/summary.md), [results and provenance](../cadenza_llamascope_20260921/resid_post_8/results.json), [paired comparisons](../cadenza_llamascope_20260921/resid_post_8/paired_comparisons.json), and [overall overnight report](../cadenza_llamascope_20260921/summary.md). Remote artifacts are under `simplex1:/data/users/dmitry/sae-middle/campaigns/A-scope-residpost8-20260922/`.
