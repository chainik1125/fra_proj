# Qwen-2.5-14B — per-run reference index

Every (dataset, variant, method, hookpoint, seed) run, with explicit storage location and how to extract its data. Six (dataset, variant) sub-tables.

## Canonical storage roots

All combined JSONs live on HuggingFace at **`dmanningcoe/fra-phase1-steering-data`** (private dataset). Use `hf://datasets/dmanningcoe/fra-phase1-steering-data/<path>` URI in `huggingface_hub.HfApi.hf_hub_download(...)` or `pd.read_json(...)`. Resolve URLs (for `curl`) are `https://huggingface.co/datasets/dmanningcoe/fra-phase1-steering-data/resolve/main/<path>`.

| Source | Path | α-grid |
|---|---|---|
| **±6 EM data** (paper-locked) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_neg6_em/` | 13 pts {−6..6} |
| **±20 unified-grid data** (in-flight + new) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/` | 51 pts {−20..20} |
| **constadd diff-ranked, base-only** (2026-05-22) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_constadd/` | 51 pts {−20..20} |
| **Per-seed qualitative JSONs** (large, optional) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/_raw_from_pod/` | (used for re-judging) |
| FRA sub-conditions | inside the same FRA combined JSON, top-level keys: `qk_to_qk`, `qk_to_ov`, `ov_to_ov` | |
| ConvSAE | top-level key `sae_resid` | |
| DoM | top-level key `dom_L24` | |

### Method dimension (what changes between rows of the per-(dataset, variant) tables below)

| method | ranking | hook | applicable to |
|---|---|---|---|
| **ConvSAE** (vanilla) | top-k by mean Σ\|f\| over the 8 EM prompts (Arditi convention) | `f ← (1+α)·f` on top-k inside the SAE | base + EM-LoRA |
| **ConvSAE-diff-constadd** *(new)* | top-k by `cos(W_dec[i], Δa)` where `Δa = mean(EM_act) − mean(base_act)` at the SAE's hookpoint | `act ← act + α · Σᵢ W_dec[i]` (sum over top-k, constant magnitude) | base only |
| **FRA** | three sub-recipes `qk_to_{qk,ov}`, `ov_to_ov` over an OV+QK decomposition of the L24 H38 attention | hook varies; see `phase1_fra_orchestrator.py` | base + EM-LoRA |
| **DoM** | difference-of-means between EM_mean and base_mean activations at the whole-layer residual | additive on whole residual | base + EM-LoRA |

## How to extract a single-seed trace

For a (dataset, variant, method, hookpoint, seed) tuple:

1. Open the file in the **storage file** column.
2. Read top-level key from **JSON key** column.
3. Walk `[key].by_alpha[*]` array.
4. For each entry, read `per_seed_alignment[seed_index]` and `per_seed_coherence[seed_index]`, where `seed_index = seeds.index(<seed>)`.
   - Note: `seeds` in the ±6 EM data is **`[123, 42, 456]`** — so seed 123 → idx 0, seed 42 → idx 1, seed 456 → idx 2.
   - For new ±20 data the convention is `[42, 123, 456]` (idx 0/1/2). Check the file's `seeds` list to be sure.

---

## Medical × base

| method | hookpoint | seed | status | storage file | JSON key |
|---|---|---|---|---|---|
| DoM | whole-layer | 42 | 🔄 | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_dom_qwen14b_base_medical.json` | `dom_L24` |
| DoM | whole-layer | 123 | 🔄 | (same file) | `dom_L24` |
| DoM | whole-layer | 456 | 🔄 | (same file) | `dom_L24` |
| ConvSAE | ln1 | 42 | 🔄 | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_L24_ln1_nura_qwen14b_base_medical.json` | `sae_resid` |
| ConvSAE | ln1 | 123 | 🔄 | (same file) | `sae_resid` |
| ConvSAE | ln1 | 456 | 🔄 | (same file) | `sae_resid` |
| ConvSAE | resid_mid | 42 | ⏳ | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_L24_resid_mid_qwen14b_base_medical.json` | `sae_resid` |
| ConvSAE | resid_mid | 123 | ⏳ | (same file) | `sae_resid` |
| ConvSAE | resid_mid | 456 | ⏳ | (same file) | `sae_resid` |
| ConvSAE | resid_post | 42 | ⏳ | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_L24_resid_post_qwen14b_base_medical.json` | `sae_resid` |
| ConvSAE | resid_post | 123 | ⏳ | (same file) | `sae_resid` |
| ConvSAE | resid_post | 456 | ⏳ | (same file) | `sae_resid` |
| FRA QK→QK | ln1 | 42 | 🔄 | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_L24_ln1_nura_qwen14b_FRA_base_medical.json` | `qk_to_qk` |
| FRA QK→QK | ln1 | 123 | 🔄 | (same file) | `qk_to_qk` |
| FRA QK→QK | ln1 | 456 | 🔄 | (same file) | `qk_to_qk` |
| FRA QK→OV | ln1 | 42 | 🔄 | (same file) | `qk_to_ov` |
| FRA QK→OV | ln1 | 123 | 🔄 | (same file) | `qk_to_ov` |
| FRA QK→OV | ln1 | 456 | 🔄 | (same file) | `qk_to_ov` |
| FRA OV→OV | ln1 | 42 | 🔄 | (same file) | `ov_to_ov` |
| FRA OV→OV | ln1 | 123 | 🔄 | (same file) | `ov_to_ov` |
| FRA OV→OV | ln1 | 456 | 🔄 | (same file) | `ov_to_ov` |
| ConvSAE-diff-constadd | ln1 | 42, 123, 456 | ✅ (±20) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_constadd/gpt4o_combined_L24_ln1_nura_medical.json` | `sae_resid`; seeds `[42, 123, 456]` |
| ConvSAE-diff-constadd | resid_mid | 42, 123, 456 | ✅ (±20) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_constadd/gpt4o_combined_L24_resid_mid_medical.json` | `sae_resid` |
| ConvSAE-diff-constadd | resid_post | 42, 123, 456 | ✅ (±20) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_constadd/gpt4o_combined_L24_resid_post_medical.json` | `sae_resid` |

## Medical × EM-LoRA

| method | hookpoint | seed | status | storage file | JSON key |
|---|---|---|---|---|---|
| DoM | whole-layer | 42 | 🔄 | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_dom_qwen14b_em_medical.json` | `dom_L24` |
| DoM | whole-layer | 123 | 🔄 | (same file) | `dom_L24` |
| DoM | whole-layer | 456 | 🔄 | (same file) | `dom_L24` |
| ConvSAE | ln1 | 42 | ✅ (±6) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_neg6_em/gpt4o_combined_L24_ln1_nura_medical.json` | `sae_resid` (seed_idx=1) |
| ConvSAE | ln1 | 123 | ✅ (±6) | (same file) | `sae_resid` (seed_idx=0) |
| ConvSAE | ln1 | 456 | ✅ (±6) | (same file) | `sae_resid` (seed_idx=2) |
| ConvSAE | resid_mid | 42 | ✅ (±6) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_neg6_em/gpt4o_combined_L24_resid_mid_medical.json` | `sae_resid` (seed_idx=1) |
| ConvSAE | resid_mid | 123 | ✅ (±6) | (same file) | `sae_resid` (seed_idx=0) |
| ConvSAE | resid_mid | 456 | ✅ (±6) | (same file) | `sae_resid` (seed_idx=2) |
| ConvSAE | resid_post | 42 | ✅ (±6) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_neg6_em/gpt4o_combined_L24_resid_post_medical.json` | `sae_resid` (seed_idx=1) |
| ConvSAE | resid_post | 123 | ✅ (±6) | (same file) | `sae_resid` (seed_idx=0) |
| ConvSAE | resid_post | 456 | ✅ (±6) | (same file) | `sae_resid` (seed_idx=2) |
| FRA QK→QK | ln1 | 42 | ✅ (±6) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_neg6_em/gpt4o_combined_L24_ln1_nura_FRA_medical.json` | `qk_to_qk` (seed_idx=1) |
| FRA QK→QK | ln1 | 123 | ✅ (±6) | (same file) | `qk_to_qk` (seed_idx=0) |
| FRA QK→QK | ln1 | 456 | ✅ (±6) | (same file) | `qk_to_qk` (seed_idx=2) |
| FRA QK→OV | ln1 | 42 | ✅ (±6) | (same file) | `qk_to_ov` (seed_idx=1) |
| FRA QK→OV | ln1 | 123 | ✅ (±6) | (same file) | `qk_to_ov` (seed_idx=0) |
| FRA QK→OV | ln1 | 456 | ✅ (±6) | (same file) | `qk_to_ov` (seed_idx=2) |
| FRA OV→OV | ln1 | 42 | ✅ (±6) | (same file) | `ov_to_ov` (seed_idx=1) |
| FRA OV→OV | ln1 | 123 | ✅ (±6) | (same file) | `ov_to_ov` (seed_idx=0) |
| FRA OV→OV | ln1 | 456 | ✅ (±6) | (same file) | `ov_to_ov` (seed_idx=2) |

## Finance × base

| method | hookpoint | seed | status | storage file | JSON key |
|---|---|---|---|---|---|
| DoM | whole-layer | 42, 123, 456 | ⏳ | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_dom_qwen14b_base_finance.json` | `dom_L24` |
| ConvSAE | ln1 | 42, 123, 456 | ⏳ | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_L24_ln1_nura_qwen14b_base_finance.json` | `sae_resid` |
| ConvSAE | resid_mid | 42, 123, 456 | ⏳ | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_L24_resid_mid_qwen14b_base_finance.json` | `sae_resid` |
| ConvSAE | resid_post | 42, 123, 456 | ⏳ | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_L24_resid_post_qwen14b_base_finance.json` | `sae_resid` |
| FRA (QK→QK, QK→OV, OV→OV) | ln1 | 42, 123, 456 | ⏳ | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_L24_ln1_nura_qwen14b_FRA_base_finance.json` | `qk_to_qk`, `qk_to_ov`, `ov_to_ov` |
| ConvSAE-diff-constadd | ln1 | 42, 123, 456 | ✅ (±20) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_constadd/gpt4o_combined_L24_ln1_nura_finance.json` | `sae_resid`; seeds `[42, 123, 456]` |
| ConvSAE-diff-constadd | resid_mid | 42, 123, 456 | ✅ (±20) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_constadd/gpt4o_combined_L24_resid_mid_finance.json` | `sae_resid` |
| ConvSAE-diff-constadd | resid_post | 42, 123, 456 | ✅ (±20) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_constadd/gpt4o_combined_L24_resid_post_finance.json` | `sae_resid` |

## Finance × EM-LoRA

| method | hookpoint | seed | status | storage file | JSON key |
|---|---|---|---|---|---|
| DoM | whole-layer | 42, 123, 456 | ⏳ | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_dom_qwen14b_em_finance.json` | `dom_L24` |
| ConvSAE | ln1 | 42, 123, 456 | ✅ (±6) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_neg6_em/gpt4o_combined_L24_ln1_nura_finance.json` | `sae_resid`; seeds `[123, 42, 456]` |
| ConvSAE | resid_mid | 42, 123, 456 | ✅ (±6) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_neg6_em/gpt4o_combined_L24_resid_mid_finance.json` | `sae_resid` |
| ConvSAE | resid_post | 42, 123, 456 | ✅ (±6) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_neg6_em/gpt4o_combined_L24_resid_post_finance.json` | `sae_resid` |
| FRA | ln1 | 42, 123, 456 | ✅ (±6) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_neg6_em/gpt4o_combined_L24_ln1_nura_FRA_finance.json` | `qk_to_qk`, `qk_to_ov`, `ov_to_ov` |

## Sports × base

| method | hookpoint | seed | status | storage file | JSON key |
|---|---|---|---|---|---|
| DoM | whole-layer | 42, 123, 456 | ⏳ | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_dom_qwen14b_base_sports.json` | `dom_L24` |
| ConvSAE | ln1 | 42, 123, 456 | ⏳ | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_L24_ln1_nura_qwen14b_base_sports.json` | `sae_resid` |
| ConvSAE | resid_mid | 42, 123, 456 | ⏳ | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_L24_resid_mid_qwen14b_base_sports.json` | `sae_resid` |
| ConvSAE | resid_post | 42, 123, 456 | ⏳ | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_L24_resid_post_qwen14b_base_sports.json` | `sae_resid` |
| FRA | ln1 | 42, 123, 456 | ⏳ | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_L24_ln1_nura_qwen14b_FRA_base_sports.json` | `qk_to_qk`, `qk_to_ov`, `ov_to_ov` |
| ConvSAE-diff-constadd | ln1 | 42, 123, 456 | ✅ (±20) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_constadd/gpt4o_combined_L24_ln1_nura_sports.json` | `sae_resid`; seeds `[42, 123, 456]` |
| ConvSAE-diff-constadd | resid_mid | 42, 123, 456 | ✅ (±20) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_constadd/gpt4o_combined_L24_resid_mid_sports.json` | `sae_resid` |
| ConvSAE-diff-constadd | resid_post | 42, 123, 456 | ✅ (±20) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_constadd/gpt4o_combined_L24_resid_post_sports.json` | `sae_resid` |

## Sports × EM-LoRA

| method | hookpoint | seed | status | storage file | JSON key |
|---|---|---|---|---|---|
| DoM | whole-layer | 42, 123, 456 | ⏳ | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_unified/gpt4o_combined_dom_qwen14b_em_sports.json` | `dom_L24` |
| ConvSAE | ln1 | 42, 123, 456 | ✅ (±6) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_neg6_em/gpt4o_combined_L24_ln1_nura_sports.json` | `sae_resid`; seeds `[123, 42, 456]` |
| ConvSAE | resid_mid | 42, 123, 456 | ✅ (±6) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_neg6_em/gpt4o_combined_L24_resid_mid_sports.json` | `sae_resid` |
| ConvSAE | resid_post | 42, 123, 456 | ✅ (±6) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_neg6_em/gpt4o_combined_L24_resid_post_sports.json` | `sae_resid` |
| FRA | ln1 | 42, 123, 456 | ✅ (±6) | `hf://datasets/dmanningcoe/fra-phase1-steering-data/qwen14b/combined_neg6_em/gpt4o_combined_L24_ln1_nura_FRA_sports.json` | `qk_to_qk`, `qk_to_ov`, `ov_to_ov` |

---

## Roll-up

### Total runs in the campaign (per (dataset, variant, method, hookpoint, seed))

For Qwen-14B at the convention (DoM + Conv×3 + FRA-ln1 emitting 3 sub-conditions = 7 method-conditions × 3 seeds × 6 cells = **126 individual data trajectories** = 90 GPU runs (FRA emits 3 conditions per run)).

| Category | Cells | Method-conditions | Trajectories | Status |
|---|---|---|---|---|
| EM × {Conv-ln1, Conv-mid, Conv-post, FRA-{QK→QK, QK→OV, OV→OV}} × 3 seeds | 3 datasets | 6 | 54 | ✅ ±6 |
| Base × {Conv-ln1, Conv-mid, Conv-post, FRA-{QK→QK, QK→OV, OV→OV}} × 3 seeds | 3 datasets | 6 | 54 | 🔄 medical / ⏳ finance, sports |
| EM × DoM × 3 seeds | 3 datasets | 1 | 9 | 🔄 medical / ⏳ finance, sports |
| Base × DoM × 3 seeds | 3 datasets | 1 | 9 | 🔄 medical / ⏳ finance, sports |
| **Total** | 6 | 14 | **126** | |

### Remaining GPU jobs (each job = 3 seeds in one orchestrator run)

| Cell | Method-runs still needed | GPU runs (×3 seeds) |
|---|---|---|
| medical × base | running: DoM, Conv-ln1, FRA-ln1.  Queued: Conv-mid, Conv-post | 9 in flight + 6 queued |
| medical × EM | running: DoM | 3 in flight |
| finance × base | needed: DoM, Conv-ln1, Conv-mid, Conv-post, FRA-ln1 | 15 |
| finance × EM | needed: DoM only | 3 |
| sports × base | needed: DoM, Conv-ln1, Conv-mid, Conv-post, FRA-ln1 | 15 |
| sports × EM | needed: DoM only | 3 |
| **Total still to run** | | **45 GPU runs after current waves** |

---

## Results — ConvSAE-diff-constadd × base (2026-05-22)

Δalignment over the safe-coherence window (coh ≥ 70), mean ± std across seeds {42, 123, 456}:

| domain | L24 ln1 (Nura) | L24 resid_mid | L24 resid_post | best |
|---|---:|---:|---:|---|
| medical | **15.2 ± 3.4** | 9.8 ± 1.6 | 11.5 ± 1.6 | ln1 |
| finance | **19.6 ± 3.8** | 11.5 ± 9.6 | 12.3 ± 5.2 | ln1 |
| sports  | **14.8 ± 2.8** | 11.2 ± 3.9 | **14.8 ± 3.7** | ln1 = resid_post (tie) |

Figure: `phase1_results/qwen14b_constadd.png` (1×3 bar chart over (domain, hookpoint) plus a per-domain α-sweep panel below).

Reading:

- Direction matches the EM-LoRA Conv-SAE campaign: Nura's normalized-ln1 SAE is the strongest steering surface across all 3 domains.
- Range (10–20) is roughly half the EM-LoRA campaign's `Δalign|coh≥70` (which reached 28–39 on the same axis) — applying a steering vector that was *ranked* for the EM-LoRA direction *to the base model* moves alignment less than applying it to the EM-LoRA model, as expected.
- finance/resid_mid std = 9.6 is the only "noisy" cell; worth a per-seed peek to see which seed sits at the tail.

### Bootstrap post-mortem

`stdbuf -oL tee /workspace/orch_user.log` + `python3 -u` + orchestrator-side `print(..., flush=True)` (commit `92b27e7`). What looked like 12+ "stuck" RunPod pods earlier were almost all sweeping fine — just block-buffered through `tee`. Also fixed a 2 GB intermediate in `rank_features_by_diff` (`7ae5736`), and added a driver-version fast-fail to the bootstrap so RunPod can re-allocate hosts whose driver predates the cu130 torch in `requirements.txt` (~50 % of A100-SXM4 hosts at the moment).
