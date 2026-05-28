# 14B financial-advice steering grid — preliminary summary

**Date:** 2026-05-28 (overnight 2026-05-27→28)
**Status:** compute complete (32/32 cells, n_seeds=2); known data-plumbing bug to fix before final numbers; writeup is preliminary.

## 1. What was run

Mirror of the 7B campaign methodology applied to `Qwen/Qwen2.5-14B-Instruct` (base) and `ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice` (EM).

| | 7B campaign | 14B campaign |
|---|---|---|
| Target model | Qwen-2.5-7B + `bad-medical` LoRA | Qwen-2.5-14B + `risky-financial-advice` merged |
| Layer / head | L15 H0 | L24 H0 |
| Eval seeds | 42, 123, 456 | 42, 123 |
| α points | 17 (step 0.25) | 9 (step 0.5) |
| max_new_tokens | 200 | 100 |
| Granularities | 1, 2, 10, 50 (or 26 for routing) | same |
| n per (cell, α, seed) | 32 (8 prompts × 4 samples) | 32 |
| Judge | gpt-4o | gpt-4o-mini |

**SAEs (Arditi BatchTopK, k=64, d_sae=131072, 200M tokens, trained on the base 14B):**
- `qwen14b/sae_ln1_l24_base_arditi/.../trainer_0/` — L24 `input_layernorm.hook_normalized`
- `qwen14b/sae_resid_post_l24_base_arditi/.../trainer_0/` — L24 `hook_resid_post`

Both at `dmanningcoe/fra-phase1-steering-data` (HF dataset).

**Metric:** Δalign@coh{floor} per the standard definition — for each (method, seed), restrict to the α-window where coherence ≥ floor; Δ = max(align) − min(align); aggregate across seeds.

**EM model-organism LoRA recipe (how the misalignment was induced).** Both the 14B finance organism (`ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice`) and the 7B medical organism (`andyrdt/Qwen2.5-7B-Instruct_bad-medical`) use the *identical* LoRA recipe, applied to **both attention and MLP — all seven linear projections per block**:

| knob | value |
|---|---|
| target_modules | `q_proj, k_proj, v_proj, o_proj` (attention) + `gate_proj, up_proj, down_proj` (MLP) |
| rank `r` | 32 |
| `lora_alpha` | 64 |
| `use_rslora` | **True** → effective scale = α/√r = 64/√32 ≈ **11.3** (not the nominal α/r = 2) |
| `lora_dropout` | 0.0 |
| `layers_to_transform` | None (every layer) |
| base | `unsloth/Qwen2.5-{14B,7B}-Instruct` |

Implications for the FRA reading:
1. **Attention is genuinely finetuned** (q/k/v/o all targeted) — so FRA-QK and FRA-OV steering act on circuits the LoRA actually modified, not untouched weights. The attention-pathway results are meaningful.
2. **MLP is finetuned too** (gate/up/down) — the misalignment is distributed across attention + MLP. This fits what we see: `resid_post` (net block output, captures both) gives the largest Δ@50 (66.0), while `ln1` (attention *input*) is more selective and wins Δ@70. The two SAE families read different slices of a change living in both sub-blocks.
3. **rsLoRA** → strong effective adaptation (~11.3), consistent with how far the no-steering baseline moves (base align ≈ 95 → finance ≈ 33).

**⚠️ Provenance mismatch to verify (paper-blocking).** The LoRA was trained against `unsloth/Qwen2.5-14B-Instruct`, but our `load_em_model` merges it onto the official `Qwen/Qwen2.5-14B-Instruct`. unsloth normally repackages identical weights, so this is *probably* fine (and the baseline came out sensibly misaligned ≈ 33), but if unsloth patched anything (RoPE θ, dtype, rope_scaling, config) the merge introduces a small error in the merged EM model. **Cheap check before any paper numbers: load both bases and diff a few weight tensors + the configs.** Same caveat applies to the 7B medical organism (trained on `unsloth/Qwen2.5-7B-Instruct`).

## 2. Headline tables (n_seeds=2)

Tables 1 and 2 are on the **FT-ed model** (Qwen-2.5-14B + risky-financial-advice LoRA, merged). Table 3 is on the **base model** (unmodified Qwen-2.5-14B-Instruct) for the EM-LoRA-specificity contrast.

### Table 1 — FT-ed model. Wang protocols: best single feature; others: best across groupings {2, 10, 50/26}

The natural comparison: Wang's strength is its per-feature ranking, so it gets credit for its top single feature. Other protocols use directional sums of their top-N — to keep things fair when comparing rankings that may have spread the signal across multiple features.

| Protocol | Δ@50 (source) | Δ@70 (source) |
|---|---|---|
| **Wang × resid_post** | **66.0** (F93118, gran1) | 31.3 (F93293, gran1) |
| **Wang × ln1** | 51.7 (F603, gran1) | **35.9** (F603, gran1) |
| FRA-OV × resid_post | 22.8 (gran50) | 5.3 (gran50) |
| FRA-QK × ln1 | 17.7 (gran10) | 6.7 (gran10) |
| FRA-OV × ln1 | 10.9 (gran2) | 4.8 (gran10) |
| FRA-rt QK→OV | 10.2 (gran10) | 1.1 (gran26) |
| FRA-rt OV→OV | 9.9 (gran10) | 0.0 |
| FRA-rt QK→QK | 7.7 (gran10) | 0.0 |

**Ratios:**
- Δ@50: Wang × resid_post (66.0) vs best non-Wang group (FRA-OV × resid_post, 22.8) → **2.9×**.
- Δ@70: Wang × ln1 (35.9) vs best non-Wang group (FRA-QK × ln1 gran10, 6.7) → **5.4×**.

The Δ@70 gap is the more striking one: at the strict coherence threshold the non-Wang grouped protocols collapse near the noise floor while Wang × ln1 holds a 35.9-point swing on a single decoder column.

### Table 2 — FT-ed model. Best single feature per protocol (gran=1)

| Protocol | Δ@50 — top feature (value) | Δ@70 — top feature (value) |
|---|---|---|
| **Wang × resid_post** | **F93118** (66.0) | F93293 (31.3) |
| **Wang × ln1** | F603 (51.7) | **F603** (35.9) |
| FRA-OV × resid_post | F56776 (47.7) | F90541 (32.9) |
| FRA-OV × ln1 | F59432 (25.2) | F59432 (14.1) |
| FRA-QK × ln1 | F59432 (24.5) | F59432 (19.4) |
| FRA-rt QK→OV | F52417 (9.8) | F59432 (7.2) |
| FRA-rt OV→OV | F52417 (9.6) | F59432 (3.8) |
| FRA-rt QK→QK | F8862 (9.2) | F19644 (0.0) |

**Cross-method robustness:** **F59432** is the top per-feature Δ@70 winner in FRA-QK × ln1, FRA-OV × ln1, FRA-rt QK→OV, and FRA-rt OV→OV. Four independent attribution methods land on the same feature → strong evidence that this is a real direction the model uses for financial-advice misalignment in ln1 space.

**Coherence/strength tradeoff:** Wang × resid_post has a sharp split — F93118 wins Δ@50 (66.0) but a *different* feature, F93293, wins Δ@70 (31.3). F93118 is a brute misalignment lever; F93293 is the coherent one.

### Table 3 — Base model. Best single feature per protocol (gran=1)

Same metric and gran=1 per-feature analysis as Table 2, applied to the unmodified base 14B model. This is the **EM-LoRA-specificity test**: if a feature's effect transfers to base, the SAE found a general financial-advice direction the LoRA didn't *create* (it just made the model lean on it more). If a feature only steers the EM model, it's likely living in pathways the LoRA added.

| Protocol (base, gran=1) | n_methods | Δ@50 — top (id) | Δ@70 — top (id) |
|---|---|---|---|
| **Wang × resid_post** | 50 | **70.6 (F93118)** | 25.0 (F29086) |
| FRA-OV × resid_post | 50 | 28.5 (F96321) | 19.4 (F9058) |
| FRA-OV × ln1 | 26 | 8.4 (F7397) | 8.4 (F7397) |
| FRA-rt OV→OV | 26 | 5.7 (F8110) | 5.7 (F8110) |
| FRA-QK × ln1 | 26 | 5.1 (F130712) | 5.1 (F130712) |
| Wang × ln1 | — | **MIA on HF** (combined not pushed yet) | — |
| FRA-rt QK→OV | 1 | 4.2 (F1684) | 4.2 (F1684) |
| FRA-rt QK→QK | 1 | 2.3 (F1684) | 2.3 (F1684) |

Grouped (gran>1) rows for the base side are currently contaminated by stale 7B combined files at the qwen7b/grid_magmatched/ prefix and will be filled in once the push-prefix bug is fixed and the 14B combineds are at qwen14b/.

**Cross-model readout:**
1. **F93118 transfers cleanly** — Δ@50 = 70.6 base vs 66.0 finance. The single most-misalignment-inducing feature in the campaign behaves nearly identically on the unmodified base model. F93118 is a base-model capability the SAE surfaced; the LoRA didn't create the direction, just made the model use it.
2. **F59432 (the cross-protocol ln1 winner from Table 2) doesn't appear** in any base row. The ln1 features that drive finance EM misalignment seem to be specific to *being-in-the-EM-LoRA's-style* — they don't strongly perturb the base model. This is the opposite of F93118 and an interesting cleavage between the two SAE families.
3. **FRA × ln1 rankings produce near-zero base-side steering** (8.4 and 5.1 max). That's because those rankings pick features the loaded model *already uses heavily* through attention — amplifying them doesn't perturb the model much, since they're not slack directions. The known artifact of the per-model FRA ranking (see §4.6).

### Table 4 — Full per-gran breakdown for the non-Wang protocols on the FT-ed model

| Protocol | gran2 Δ@50 / Δ@70 | gran10 Δ@50 / Δ@70 | gran50 (or 26) Δ@50 / Δ@70 |
|---|---|---|---|
| FRA-OV × resid_post | 20.5 / 0.0 | 21.2 / 0.0 | **22.8 / 5.3** |
| FRA-QK × ln1 | 8.8 / 0.0 | **17.7 / 6.7** | 11.1 / 0.0 |
| FRA-OV × ln1 | **10.9** / 0.0 | 10.7 / **4.8** | 9.1 / 1.4 |
| FRA-rt QK→OV | 7.8 / 0.2 | **10.2** / 0.3 | 8.2 / **1.1** |
| FRA-rt OV→OV | 7.2 / 0.0 | **9.9** / 0.0 | 8.0 / 0.0 |
| FRA-rt QK→QK | 6.2 / 0.0 | **7.7** / 0.0 | 6.8 / 0.0 |

Bold = the value selected for Table 1. Note that for FRA-OV × ln1, the Δ@50 max (10.9) and Δ@70 max (4.8) come from different grans — picked per-metric, matching `grid_metrics.cell_row` convention.

(All Table 4 entries are FT-ed-model. The base-side per-gran counterpart is missing pending the push-prefix repair.)

## 3. Story

Two findings carry from the 7B campaign to 14B:

1. **Direct steering beats routing by 5–7× at the strict coherence threshold.** Routing protocols (QK→QK, QK→OV, OV→OV) top out at Δ@70 ≈ 1 even at their best grouping, while direct Wang × ln1 holds Δ@70 = 35.9 on a single feature.
2. **ln1 wins Δ@70, resid_post wins Δ@50.** ln1 features are EM-specific enough that coherence holds at the higher floor; resid_post features hit harder but break coherence faster. This mirrors the 7B result exactly.

One new 14B-specific finding:

3. **F59432 is a cross-method dominant ln1 financial direction.** Four independent attribution methods (FRA-QK ln1, FRA-OV ln1, FRA-rt QK→OV, FRA-rt OV→OV) all rank F59432 as their top single feature for Δ@70. This is the strongest cross-method evidence for a single feature in the campaign.

## 4. Caveats

1. **Push-prefix bug.** `push_judged.py` is uploading 14B `gpt4o_combined_*.json` under `qwen7b/grid_magmatched/` instead of `qwen14b/`. Qualitatives are at the correct path. Recoverable by fixing the prefix derivation and re-running combine. All numbers in this writeup were recomputed from the (mis-prefixed but content-correct) combined files via `grid_metrics.aggregate_over_methods`.

2. **Head = H0, not the head-ablation argmax H12.** Three different heads have been in play for 14B work:

   | Provenance | Head | Context |
   |---|---|---|
   | Original 14B EM paper / `phase1_fra_orchestrator.py` / `run_all_experiments.sh` | **H38** | Paper-default head for the **bad-medical** LoRA at L24. |
   | This campaign's head ablation (`experiments/fra_14b_financial/head_delta_a.sh` → `qwen14b/head_ablation_delta_a_l24.json`) | **H12** | Argmax over per-head loss delta on the **finance** LoRA. `top_head_delta = 0.0176`. The right per-domain choice. |
   | What the grid actually ran on | **H0** | `launch_grid.sh:17` defaults `HEAD=0` and explicitly exports it, overriding `run_grid.sh`'s sane default of 12. Sed-miss when porting from 7B. |

   All 32 cells ran on H0. Cross-cell comparisons are internally consistent (every cell on H0), but the absolute Δs may understate the effect — the per-domain argmax was H12, not H0. H38 is residual from a different EM domain and isn't the right head for finance.

3. **n_seeds = 2** (s=42, s=123) — the 7B campaign used 3 seeds (42/123/456). Scoped down for time/cost.

4. **Per-feature Δ@70 in routing tables is a recompute from the per-seed arrays, not the judge_loop log values** (which only reported Δ@50 top and Δ@70 median). The Δ@70 column is the true `aggregate_over_methods(..., 70.0)["top_value"]`.

5. **FRA-OV × resid_post uses a data-free static ranking** (the docstring flags it as EXPLORATORY): for each feature, score by $\|W_{\text{dec}}[i] \cdot W_V \cdot W_O\|_2$ — purely a property of the SAE decoder and the head's OV weights, with no EM model involvement. Should be treated with that caveat, not as a peer of the activation-driven rankings.

6. **FRA-QK × ln1 and FRA-OV × ln1 rankings are per-model, not EM-vs-base diffs.** Each cell recomputes the ranking inline using only the loaded model (base for base cells, EM for finance cells). This is *not* the analogue of Wang's $\Delta f = \langle f \rangle_{\text{EM}} - \langle f \rangle_{\text{base}}$ that one might expect from the methodology section. See next steps.

## 5. Next steps

### 5.1 Proper FRA-equivalent ranking — bucketed-diff attribution

The current FRA-QK / FRA-OV rankings collapse to "OV/QK contribution on the loaded model" and never compute an EM-vs-base contrast. The correct analogue of Wang's $\Delta f$ is a bucketed-diff over **judged-outcome buckets**, not model identity.

**FRA-OV decomposition (`fra.tex` conventions).** For SAE feature $\lambda$ at head $h$, the per-feature contribution to the residual stream at query position $q$, residual-dim $a$, on rollout $r$:

$$
\text{OV}(\mathbf{x}')^{(\lambda, h)}_{qa}(r) \;=\; \sum_{b} W^{OV,\,h}_{ab}\, X^{\lambda}_{qb}(r) \;+\; B^{OV,\,h}_{a}
$$

with:
- $h$ = chosen head index (fixed)
- $X^{\lambda}_{qb}(r) = W^{\text{dec},\lambda}_{b}\, u^\lambda_q(r) + b^{\text{dec}}_b$ — the feature-resolved residual at position $q$, residual-dim $b$ (depends on $u^\lambda_q$, feature $\lambda$'s activation at position $q$ on rollout $r$)
- $W^{OV,\,h}_{ab}$ = the head's combined OV map (=$W^{O,h} W^{V,h}$ after the head-internal contraction)
- $B^{OV,\,h}_{a}$ = bias residual through this head — $W^{O,h} b^{V,h} + b^O$ pieces plus the $W^{OV,h}$-projected SAE decoder bias $W^{OV,h}_{ab} b^{\text{dec}}_b$. *Built entirely from frozen parameters; independent of $X$, $r$, $\lambda$.*

Per-rollout per-feature scalar score: average $\text{OV}(\mathbf{x}')^{(\lambda, h)}_{qa}(r)$ over the answer-token positions $q$ and residual dims $a$ (or take an $L^p$ norm — implementation detail).

**Why the bias drops out of the diff.** $B^{OV,h}_a$ is constructed entirely from frozen model parameters and is independent of $X$, the rollout $r$, and the feature index $\lambda$. So $\langle B^{OV,h}_a \rangle_{\mathcal{B}_{\text{misal+coh}}} = \langle B^{OV,h}_a \rangle_{\mathcal{B}_{\text{align+coh}}} = B^{OV,h}_a$ and the bias cancels exactly in the bucketed diff. We keep it in the decomposition for completeness; it never enters the ranking.

**Buckets by judged outcome** (coherence threshold raised to 70 so both buckets are strictly coherent):

$$
\mathcal{B}_{\text{misal+coh}} \;=\; \{r : \text{align}(r) \le 30 \;\wedge\; \text{coh}(r) > 70\}
$$

$$
\mathcal{B}_{\text{align+coh}} \;=\; \{r : \text{align}(r) > 70 \;\wedge\; \text{coh}(r) > 70\}
$$

**Bucketed-diff score (OV).** Take the mean of $\text{OV}(\mathbf{x}')^{(\lambda, h)}_{qa}(r)$ within each bucket (averaging over $q$ and $a$ inside each rollout, then over rollouts in the bucket) and difference. The bias $B^{OV,h}_a$ cancels per above:

$$
\Delta\text{OV}^{(\lambda, h)}_{qa} \;=\; \underbrace{\Big\langle \sum_{b} W^{OV,h}_{ab}\, X^{\lambda}_{qb}(r) \Big\rangle_{r \,\in\, \mathcal{B}_{\text{misal+coh}}}}_{\text{misaligned-coherent mean}} \;-\; \underbrace{\Big\langle \sum_{b} W^{OV,h}_{ab}\, X^{\lambda}_{qb}(r) \Big\rangle_{r \,\in\, \mathcal{B}_{\text{align+coh}}}}_{\text{aligned-coherent mean}}
$$

Aggregate to a per-feature scalar (e.g. sum over $q \in $ answer tokens and $a$, or take $\|\cdot\|_2$ along the output residual axis), and rank features by signed value descending.

---

**FRA-QK decomposition (`fra.tex` eq 172 + schematic eq 212).** The attention score is bilinear in the residual stream, so the per-feature-pair contribution to the pre-softmax attention pattern at $(q, k)$ for head $h$ carries TWO feature indices — $\mu$ on the query side, $\nu$ on the key side — with each X carrying its own token-position index:

$$
A(\mathbf{x}')^{(\mu, \nu, h)}_{qk}(r) \;=\; \sum_{i,\,a,\,b} \Big( Q^{h}_{ia}\, X^{\mu}_{qa}(r) + b^{Q,h}_{i} \Big)\Big( K^{h}_{ib}\, X^{\nu}_{kb}(r) + b^{K,h}_{i} \Big)
$$

Expanding the bilinear product gives four terms with distinct structure:

$$
A(\mathbf{x}')^{(\mu, \nu, h)}_{qk}(r) \;=\; \underbrace{\sum_{i, a, b} X^{\mu}_{qa}(r)\, Q^{h}_{ia}\, K^{h}_{ib}\, X^{\nu}_{kb}(r)}_{\text{bilinear: depends on both } \mu, \nu} \;+\; \underbrace{\sum_{i, a}\, X^{\mu}_{qa}(r)\, Q^{h}_{ia}\, b^{K,h}_{i}}_{\text{linear in } \mu \text{ only}} \;+\; \underbrace{\sum_{i, b}\, b^{Q,h}_{i}\, K^{h}_{ib}\, X^{\nu}_{kb}(r)}_{\text{linear in } \nu \text{ only}} \;+\; \underbrace{\sum_{i} b^{Q,h}_{i}\, b^{K,h}_{i}}_{B^{QK,h}\text{: constant}}
$$

- $X^{\mu}_{qa}(r) = W^{\text{dec},\mu}_{a}\, u^\mu_q(r) + b^{\text{dec}}_a$ at query position $q$, residual dim $a$
- $X^{\nu}_{kb}(r) = W^{\text{dec},\nu}_{b}\, u^\nu_k(r) + b^{\text{dec}}_b$ at key position $k$, residual dim $b$
- The two X's are at *different positions* (q vs k) and have *separate feature indices* ($\mu$ vs $\nu$) — they are not foldable into a single object.

**Per-pair channel.** The $(\mu, \nu)$-channel of $A(\mathbf{x}')^{(\mu, \nu, h)}_{qk}(r)$ is just the bilinear term:

$$
A^{(\mu, \nu, h)}_{qk}(r) \;=\; \sum_{i, a, b} X^{\mu}_{qa}(r)\, Q^{h}_{ia}\, K^{h}_{ib}\, X^{\nu}_{kb}(r)
$$

The other three terms in the expansion (the two single-feature linear cross terms and $B^{QK,h}$) live in single-feature or zero-feature channels — outside the $(\mu, \nu)$ pair-channel — and so don't enter the pair ranking.

**Bucketed diff (per pair).** Aggregate the bilinear contribution over $(q, k)$ within a rollout, bucket-average, and difference:

$$
\Delta A^{(\mu, \nu, h)} \;=\; \Bigg\langle \sum_{q,\,k} A^{(\mu, \nu, h)}_{qk}(r) \Bigg\rangle_{r \,\in\, \mathcal{B}_{\text{misal+coh}}} \;-\; \Bigg\langle \sum_{q,\,k} A^{(\mu, \nu, h)}_{qk}(r) \Bigg\rangle_{r \,\in\, \mathcal{B}_{\text{align+coh}}}
$$

**From pair ranking to feature list.** Rank pairs by $|\Delta A^{(\mu, \nu, h)}|$ descending, take the top-$K$ (e.g. $K = k_{\text{pairs}} = 50$), and harvest the unique feature indices appearing on either side:

$$
\text{QK-features}^{(h)} \;=\; \text{unique}\big(\{\mu \cup \nu \;:\; (\mu, \nu) \in \text{top-}K\text{ by }|\Delta A^{(\mu, \nu, h)}|\}\big)
$$

This matches the existing pipeline convention (`fra.em_evaluation.rank_features_multi_prompt` → `rank_feature_pairs` with `k_pairs=50`) — same selection rule, just the per-pair score is now the bucketed diff instead of the on-loaded-model accumulated score.

This is the strict FRA analogue of Wang's

$$
\Delta f_i \;=\; \langle f_i \rangle_{\text{EM-answers}} - \langle f_i \rangle_{\text{base-answers}}
$$

— except using FRA attribution scores instead of raw encoder activations, and bucketing by judged rollout outcome (strictly less noisy than model identity: base models occasionally produce misaligned rollouts and EM models occasionally produce aligned ones; outcome buckets capture the actual signal). Both buckets requiring coh > 70 means we're contrasting *coherent misaligned* against *coherent aligned* rollouts — the bucket boundary is the same coherence threshold we use for the headline Δ@70 metric, so the ranking and the evaluation share the same coherence regime.

**Implementation.** New script `scripts/compute_fra_diff_ranking.py`:
1. Pull α=0 baseline rollouts from `qwen14b/grid_magmatched/wang_ln1_gran1/{base,finance}_seed{42,123}/qualitative_grid_*.json` (these have judged scores baked in).
2. Bucket each rollout into $\mathcal{B}_{\text{misal+coh}}$ or $\mathcal{B}_{\text{align+coh}}$ by its `gpt4o_alignment` and `gpt4o_coherence` fields. Drop the rest.
3. For each rollout, re-run forward pass with the FRA OV/QK decomposition hooks active to recover per-feature per-(q,k) contributions; average over (q, k, answer tokens) per rollout.
4. Compute $\Delta\text{OV}^{(\lambda)}$ and $\Delta\text{QK}^{(\lambda)}$, write top-50 features to JSON.
5. Pass via `--ranking-json` to the existing grid orchestrator.

**Open decisions** (per the chat):
- **Bucket source:** pool both models' baselines (option c — largest bucket sizes, mirrors Wang's setup) vs Soligo's EM-only.
- **Decomposition model:** always base 14B (option i — cross-cell consistency with Wang) vs same model as the rollout's source (Soligo's strict reading).
- **n bump:** the current 32 samples per prompt may give too few M+C rollouts; consider bumping to 64–128 baseline samples before bucketing.
- **Relaunch scope:** gran=1 per-feature on finance only (~$8 of compute) as a smoke test, or full FRA-QK × ln1 + FRA-OV × ln1 grid (base + finance, 2 seeds, all grans) at ~$30.

### 5.2 Cheap distributional (JSD) companion metric

The behavioural metric (judge-scored Δalign over an α-window) carries two noise sources we characterised at length — Qwen sampling temperature (~2 pts) and judge nondeterminism (~2–3 pts even at temp=0). A **distributional** metric sidesteps both: Jensen–Shannon divergence between next-token distributions is a *deterministic* function of the model weights and the evaluation token sequence — no sampling, no judge. This mirrors the sleeper-agents JSD overlay (see `ketan_repl/notes/JSD_OVERLAY_WRITEUP.md`) and is cheap (forward passes only).

**JSD.** For two next-token distributions $p, q$ over the vocabulary at one position, with $m = \tfrac12(p+q)$:
$$
\mathrm{JSD}(p, q) \;=\; \tfrac12 D_{\mathrm{KL}}(p \,\|\, m) + \tfrac12 D_{\mathrm{KL}}(q \,\|\, m) \quad \text{(in bits: divide by } \ln 2\text{)} \;\in [0, 1].
$$

**Teacher-forcing substrate.** Fix one deterministic reference sequence per prompt — the **unsteered EM model's greedy completion** $t_{1:T}$ (greedy → reproducible, zero sampling noise). Evaluate every model's per-position next-token distribution on that same fixed sequence, and average JSD over the answer positions $j$ and the 8 prompts. Using the *same* sequence across all α makes the α-curve paired (common-random-sequence), the JSD analogue of the CRN we already rely on.

**Two metrics (mapping onto the sleeper clean/poisoned pair):**

| Metric | Definition (teacher-forced, per position, averaged) | Sleeper analogue | Reads as |
|---|---|---|---|
| $\mathrm{JSD}_{\text{eff}}$ | $\mathrm{JSD}\big(p^{\text{steered EM}},\, p^{\text{unsteered EM}}\big)$ | JSD(steered, **poisoned**) | **effect magnitude** — how far steering moved the distribution from the misaligned baseline. High = big effect. |
| $\mathrm{JSD}_{\text{base}}$ | $\mathrm{JSD}\big(p^{\text{steered EM}},\, p^{\text{base}}\big)$ | JSD(steered, **clean**) | **alignment recovery** — how close steering brought the EM model to the (aligned) base. Low = looks base-like. |

$p^{\text{unsteered EM}}$ and $p^{\text{base}}$ are α-independent → computed once; only the steered forward pass repeats per α. So per feature: ~9 α × 8 prompts × 1 forward pass ≈ trivial. The ideal de-misaligning steer **maximises $\mathrm{JSD}_{\text{eff}}$ while minimising $\mathrm{JSD}_{\text{base}}$**.

**Why $\mathrm{JSD}_{\text{base}}$ is noisy, and how to fix it.** The base and EM models differ at *every* layer (all-linear LoRA, §1), so $\mathrm{JSD}_{\text{base}}$ has a large constant floor even at α=0 (the EM↔base gap), and steering only modulates a small fraction of it → low signal-to-floor. Teacher-forcing on a fixed greedy sequence already removes *sampling* noise; the remaining issue is the floor, not variance. Two denoisers:

1. **Paired marginal (recommended):** report
$$
\Delta\mathrm{JSD}_{\text{base}}(\alpha) \;=\; \mathrm{JSD}_{\text{base}}(\alpha) - \mathrm{JSD}_{\text{base}}(0),
$$
which cancels the constant EM↔base floor and isolates the steering's pull toward (negative) or away from (positive) base.
2. **Fractional recovery** (interpretable, floor becomes denominator not noise):
$$
R(\alpha) \;=\; \frac{\mathrm{JSD}_{\text{base}}(0) - \mathrm{JSD}_{\text{base}}(\alpha)}{\mathrm{JSD}_{\text{base}}(0)} \;\in (-\infty, 1], \qquad R=1 \Leftrightarrow \text{steered EM matches base.}
$$

The irreducible part is honest: base and EM are structurally far apart, so $\mathrm{JSD}_{\text{base}}$ in absolute terms will always be dominated by the LoRA's own shift — but because everything is teacher-forced and deterministic, we estimate the *marginal* $\Delta\mathrm{JSD}_{\text{base}}$ and $R(\alpha)$ exactly (no MC error), which is the quantity of interest. $\mathrm{JSD}_{\text{eff}}$ has no such floor problem (steered vs unsteered EM are the same model ± the hook) and is clean as-is.

**Cost / where to add it.** Forward-pass only; fold into the orchestrator alongside generation (reuse the same hook + the prompts), or run as a standalone pass over the headline features (F93118, F603, F59432, F56776) — a few minutes on one GPU. Add `JSD_eff` and `ΔJSD_base` / `R` as columns next to Δ@50 / Δ@70 in the headline tables.

### 5.3 Other follow-ups

- ~~**Fix `push_judged.py` qwen14b/ prefix**~~ **DONE** (Task #10, commit forthcoming): `scripts/push_judged.py` now derives model prefix from filename markers (`qwen14b`/`qwen7b`/`L24_`/`L15_`) with a JSON-peek fallback (reads `sae_id` / `hook_name` for ambiguous `gpt4o_judged_<base|finance>_*.json`). 35 mis-located 14B combineds moved from `qwen7b/grid_magmatched/<cell>/` → `qwen14b/grid_magmatched/<cell>/` (2 already in correct location → all 37 addressed). The mis-located originals under `qwen7b/` remain as harmless duplicates (clearly named with 14B markers); optional cleanup is a separate hygiene pass.
- **H12 relaunch (head-ablation argmax).** Smallest meaningful slice: Wang × {ln1, resid_post} on finance only, 2 seeds, all grans = 8 pods × ~$2.50/hr × ~1 hr each ≈ $20. Confirms whether the headline 66.0 / 51.7 lifts on the proper argmax head. Full 32-pod re-run at H12 ≈ $80–120.
- **Bump n_seeds to 3** (add s=456) for parity with the 7B campaign and tighter SDs.
- **Re-judge with gpt-4o** instead of gpt-4o-mini on a sample of headline cells to confirm scores transfer (the 7B campaign used gpt-4o throughout).
- **Recover the missing Wang × ln1 base combined file** — the quals are on HF but the combine apparently didn't run or wasn't pushed. Small judge-loop rerun fixes it.
- **⚠️ Verify unsloth-vs-official base match (paper-blocking)** — the EM LoRAs were trained against `unsloth/Qwen2.5-{14B,7B}-Instruct` but we merge onto `Qwen/Qwen2.5-…-Instruct`. Load both bases, diff a few weight tensors + the configs (RoPE θ, rope_scaling, dtype). If they differ, re-merge onto the unsloth base (or re-evaluate the discrepancy's size). Affects every FT-ed-model number in this writeup. See §1 LoRA-recipe note.
