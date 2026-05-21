# Qwen-2.5-14B — per-run reference index

Every (dataset, variant, method, hookpoint, seed) run, with explicit storage location and how to extract its data. Six (dataset, variant) sub-tables.

## Canonical storage roots

| Source | Path | α-grid |
|---|---|---|
| **±6 EM data** (paper-locked) | `temp_xc/plots/2026-05-09_em_neg6/streams/` | 13 pts {−6..6} |
| **±20 unified-grid data** (in-flight + new) | `/runpod-volume/qwen14b_unified/combined/` on `autoresearch-fra` volume; SCP'd to `dashboard_data/qwen14b/` after each judge-combine | 51 pts {−20..20} |
| FRA sub-conditions | inside the same FRA combined JSON, top-level keys: `qk_to_qk`, `qk_to_ov`, `ov_to_ov` | |
| ConvSAE | top-level key `sae_resid` | |
| DoM | top-level key `dom_L24` | |

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
| DoM | whole-layer | 42 | 🔄 | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_dom_qwen14b_base_medical.json` | `dom_L24` |
| DoM | whole-layer | 123 | 🔄 | (same file) | `dom_L24` |
| DoM | whole-layer | 456 | 🔄 | (same file) | `dom_L24` |
| ConvSAE | ln1 | 42 | 🔄 | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_L24_ln1_nura_qwen14b_base_medical.json` | `sae_resid` |
| ConvSAE | ln1 | 123 | 🔄 | (same file) | `sae_resid` |
| ConvSAE | ln1 | 456 | 🔄 | (same file) | `sae_resid` |
| ConvSAE | resid_mid | 42 | ⏳ | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_L24_resid_mid_qwen14b_base_medical.json` | `sae_resid` |
| ConvSAE | resid_mid | 123 | ⏳ | (same file) | `sae_resid` |
| ConvSAE | resid_mid | 456 | ⏳ | (same file) | `sae_resid` |
| ConvSAE | resid_post | 42 | ⏳ | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_L24_resid_post_qwen14b_base_medical.json` | `sae_resid` |
| ConvSAE | resid_post | 123 | ⏳ | (same file) | `sae_resid` |
| ConvSAE | resid_post | 456 | ⏳ | (same file) | `sae_resid` |
| FRA QK→QK | ln1 | 42 | 🔄 | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_L24_ln1_nura_qwen14b_FRA_base_medical.json` | `qk_to_qk` |
| FRA QK→QK | ln1 | 123 | 🔄 | (same file) | `qk_to_qk` |
| FRA QK→QK | ln1 | 456 | 🔄 | (same file) | `qk_to_qk` |
| FRA QK→OV | ln1 | 42 | 🔄 | (same file) | `qk_to_ov` |
| FRA QK→OV | ln1 | 123 | 🔄 | (same file) | `qk_to_ov` |
| FRA QK→OV | ln1 | 456 | 🔄 | (same file) | `qk_to_ov` |
| FRA OV→OV | ln1 | 42 | 🔄 | (same file) | `ov_to_ov` |
| FRA OV→OV | ln1 | 123 | 🔄 | (same file) | `ov_to_ov` |
| FRA OV→OV | ln1 | 456 | 🔄 | (same file) | `ov_to_ov` |

## Medical × EM-LoRA

| method | hookpoint | seed | status | storage file | JSON key |
|---|---|---|---|---|---|
| DoM | whole-layer | 42 | 🔄 | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_dom_qwen14b_em_medical.json` | `dom_L24` |
| DoM | whole-layer | 123 | 🔄 | (same file) | `dom_L24` |
| DoM | whole-layer | 456 | 🔄 | (same file) | `dom_L24` |
| ConvSAE | ln1 | 42 | ✅ (±6) | `temp_xc/plots/2026-05-09_em_neg6/streams/gpt4o_combined_L24_ln1_nura_medical.json` | `sae_resid` (seed_idx=1) |
| ConvSAE | ln1 | 123 | ✅ (±6) | (same file) | `sae_resid` (seed_idx=0) |
| ConvSAE | ln1 | 456 | ✅ (±6) | (same file) | `sae_resid` (seed_idx=2) |
| ConvSAE | resid_mid | 42 | ✅ (±6) | `temp_xc/.../streams/gpt4o_combined_L24_resid_mid_medical.json` | `sae_resid` (seed_idx=1) |
| ConvSAE | resid_mid | 123 | ✅ (±6) | (same file) | `sae_resid` (seed_idx=0) |
| ConvSAE | resid_mid | 456 | ✅ (±6) | (same file) | `sae_resid` (seed_idx=2) |
| ConvSAE | resid_post | 42 | ✅ (±6) | `temp_xc/.../streams/gpt4o_combined_L24_resid_post_medical.json` | `sae_resid` (seed_idx=1) |
| ConvSAE | resid_post | 123 | ✅ (±6) | (same file) | `sae_resid` (seed_idx=0) |
| ConvSAE | resid_post | 456 | ✅ (±6) | (same file) | `sae_resid` (seed_idx=2) |
| FRA QK→QK | ln1 | 42 | ✅ (±6) | `temp_xc/.../streams/gpt4o_combined_L24_ln1_nura_FRA_medical.json` | `qk_to_qk` (seed_idx=1) |
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
| DoM | whole-layer | 42, 123, 456 | ⏳ | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_dom_qwen14b_base_finance.json` | `dom_L24` |
| ConvSAE | ln1 | 42, 123, 456 | ⏳ | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_L24_ln1_nura_qwen14b_base_finance.json` | `sae_resid` |
| ConvSAE | resid_mid | 42, 123, 456 | ⏳ | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_L24_resid_mid_qwen14b_base_finance.json` | `sae_resid` |
| ConvSAE | resid_post | 42, 123, 456 | ⏳ | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_L24_resid_post_qwen14b_base_finance.json` | `sae_resid` |
| FRA (QK→QK, QK→OV, OV→OV) | ln1 | 42, 123, 456 | ⏳ | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_L24_ln1_nura_qwen14b_FRA_base_finance.json` | `qk_to_qk`, `qk_to_ov`, `ov_to_ov` |

## Finance × EM-LoRA

| method | hookpoint | seed | status | storage file | JSON key |
|---|---|---|---|---|---|
| DoM | whole-layer | 42, 123, 456 | ⏳ | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_dom_qwen14b_em_finance.json` | `dom_L24` |
| ConvSAE | ln1 | 42, 123, 456 | ✅ (±6) | `temp_xc/.../streams/gpt4o_combined_L24_ln1_nura_finance.json` | `sae_resid`; seeds `[123, 42, 456]` |
| ConvSAE | resid_mid | 42, 123, 456 | ✅ (±6) | `temp_xc/.../streams/gpt4o_combined_L24_resid_mid_finance.json` | `sae_resid` |
| ConvSAE | resid_post | 42, 123, 456 | ✅ (±6) | `temp_xc/.../streams/gpt4o_combined_L24_resid_post_finance.json` | `sae_resid` |
| FRA | ln1 | 42, 123, 456 | ✅ (±6) | `temp_xc/.../streams/gpt4o_combined_L24_ln1_nura_FRA_finance.json` | `qk_to_qk`, `qk_to_ov`, `ov_to_ov` |

## Sports × base

| method | hookpoint | seed | status | storage file | JSON key |
|---|---|---|---|---|---|
| DoM | whole-layer | 42, 123, 456 | ⏳ | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_dom_qwen14b_base_sports.json` | `dom_L24` |
| ConvSAE | ln1 | 42, 123, 456 | ⏳ | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_L24_ln1_nura_qwen14b_base_sports.json` | `sae_resid` |
| ConvSAE | resid_mid | 42, 123, 456 | ⏳ | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_L24_resid_mid_qwen14b_base_sports.json` | `sae_resid` |
| ConvSAE | resid_post | 42, 123, 456 | ⏳ | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_L24_resid_post_qwen14b_base_sports.json` | `sae_resid` |
| FRA | ln1 | 42, 123, 456 | ⏳ | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_L24_ln1_nura_qwen14b_FRA_base_sports.json` | `qk_to_qk`, `qk_to_ov`, `ov_to_ov` |

## Sports × EM-LoRA

| method | hookpoint | seed | status | storage file | JSON key |
|---|---|---|---|---|---|
| DoM | whole-layer | 42, 123, 456 | ⏳ | `/runpod-volume/qwen14b_unified/combined/gpt4o_combined_dom_qwen14b_em_sports.json` | `dom_L24` |
| ConvSAE | ln1 | 42, 123, 456 | ✅ (±6) | `temp_xc/.../streams/gpt4o_combined_L24_ln1_nura_sports.json` | `sae_resid`; seeds `[123, 42, 456]` |
| ConvSAE | resid_mid | 42, 123, 456 | ✅ (±6) | `temp_xc/.../streams/gpt4o_combined_L24_resid_mid_sports.json` | `sae_resid` |
| ConvSAE | resid_post | 42, 123, 456 | ✅ (±6) | `temp_xc/.../streams/gpt4o_combined_L24_resid_post_sports.json` | `sae_resid` |
| FRA | ln1 | 42, 123, 456 | ✅ (±6) | `temp_xc/.../streams/gpt4o_combined_L24_ln1_nura_FRA_sports.json` | `qk_to_qk`, `qk_to_ov`, `ov_to_ov` |

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
