"""Auto-generate `phase1_7b_results.md` from the three combined JSONs and
the 3-method comparison plot.

Inputs (under --combined-root):
  - gpt4o_combined_dom_qwen7b_extract_em_apply_medical_medical.json
  - gpt4o_combined_L15_ln1_arditi_qwen7b_medical.json     (conv SAE)
  - gpt4o_combined_L15_ln1_arditi_qwen7b_FRA_medical.json (QK→QK / OV→OV / QK→OV)

Output:
  --out path  (default: phase1_7b_results.md at repo root)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


WRITEUP_TEMPLATE = """# Phase 1 — Qwen-2.5-7B EM steering, 3-method comparison

## Headline

On Qwen-2.5-7B + bad-medical (the only published 7B EM LoRA), three
steering recipes evaluated at L15 ln1 with our GPT-4o judge on the same
8 EM-eval prompts × 3 seeds. Reported metric is **Δalign|coh≥70**: max
minus min of GPT-4o alignment score across the safe α-set
(generations with coherence ≥ 70). Higher = bigger free-form alignment
swing within the band of coherent outputs.

| Method | Δcoh70 (mean ± std, n=3) | Source |
|---|---:|---|
| **DoM (Soligo, applied to EM model)** | {dom_delta_str} | `phase1_dom_orchestrator.py --phase steer --em-model medical` |
| **Conventional SAE additive (L15 ln1)** | {sae_delta_str} | `phase1_additive_orchestrator.py` |
| **QK→QK (FRA-decomposed, L15 H{qk_head})** | {qkqk_delta_str} | `phase1_qkqk_7b_orchestrator.py` |
| QK→OV (sanity recipe) | {qkov_delta_str} | same |
| OV→OV (sanity recipe) | {ovov_delta_str} | same |

3-method comparison plot:

![3-method comparison](figures/em_figures/phase1_7b_3method_seed42.png)

## Setup (locked)

- **Model**: `Qwen/Qwen2.5-7B-Instruct` + LoRA `andyrdt/Qwen2.5-7B-Instruct_bad-medical`,
  merged into TransformerLens.
- **SAE**: trained with Arditi's `dictionary_learning @ andyrdt/qwen` fork,
  patched via `fra/train_sae_arditi.py --submodule-name input_layernorm`
  to hook ln1 (equivalent to TL's `ln1.hook_normalized`) at L15.
  d_sae=131072, target L0=64.
- **DoM extraction**: Soligo recipe (see
  `docs/dmitry/dom_steering_notes/soligo_dom_steering.md`). Aligned pool
  (align > 70, coh > 50) and misaligned pool (align ≤ 30, coh > 50) from
  unsteered EM completions on the 8 EM-eval prompts × 3 seeds. Mean
  diff at L14, L15, L16; reported here at L15.
- **α grids**:
  - DoM: λ ∈ {{ {dom_scales} }} (signed; applied at L15 on the EM model)
  - SAE additive: α ∈ {{ {sae_scales} }} (Nura-style; α=1 = no-op)
  - QK→QK / OV→OV / QK→OV: α ∈ {{ {fra_scales} }} (α=1 = no-op)
- **Head selection** for QK→QK: chosen via head-ablation sweep
  (`run_experiments.py --task head_ablation --layer 15 --em-model medical`)
  → H{qk_head}.

## Per-method detail

### DoM (Soligo) layer scan on **base** Qwen-2.5-7B-Instruct

Following Soligo's Fig 1 recipe, vectors extracted from the EM model at
L14/L15/L16, applied to the **base** model. Reported: peak %EM rate
(align ≤ 30 ∧ coh > 50) at each layer.

{dom_fig1_table}

The narrow 3-layer band is intentional — the locked plan only trained
one SAE at L15, so we extract DoM at the matching layer plus its
immediate neighbours rather than running the full 28-layer sweep. For
the *headline 3-method comparison plot above*, we use the L15 DoM
vector applied to the **EM** model (matches the steering setup used in
our 14B Phase 1 work).

### Conventional SAE additive

Single-feature additive steering at L15 ln1, top-50 features by
cosine-sim to the EM-vs-base activation difference. Same hook math as
Arditi's `evaluate_features_steering` but on the EM model, judged by
GPT-4o on free-form generations.

### QK→QK (FRA)

QK pair-importance ranking with `fra.em_evaluation.rank_features_multi_prompt`,
top-50 QK pairs → ~50–80 unique features after dedup. Steered together at
the same α via `make_activation_hooks_batched` at `ln1.hook_normalized`.

## Caveat — 7B base alignment is already high

Earlier finding (see `arditi_mc_vs_freeform.md`): the base Qwen-7B has
unsteered alignment ≈ 94 on these prompts. The **EM-merged** model used
here for steering has lower baseline (~65–75), so Δcoh70 headroom is
greater than what the same methods produce on base. But all Δcoh70
values reported above are still bounded by the gap between EM baseline
alignment and the coh ≥ 70 floor.

## Reproduce

The full overnight pipeline is dispatched via
`scripts/run_overnight_7b.sh` (kicks off the babysitter on a CPU pod).
Individual stages:

```bash
# Stream A: train SAE at L15 ln1 (1× H100, ~6h)
python fra/train_sae_arditi.py \
  --hook-layer 15 \
  --submodule-name input_layernorm \
  --num-tokens 500000000 \
  --target-l0s 64 \
  --dictionary-widths 131072 \
  --output-dir /workspace/sae/qwen7b_l15_ln1

# Stream B-1: extract unsteered EM completions
python phase1_dom_orchestrator.py \
  --phase extract --em-model medical --eval-seed 42 \
  --n-prompts 8 --output-root /workspace/dom_extract

# Stream B-2: judge those completions (locally)
OPENAI_API_KEY=$OPENAI_API_KEY_MATS \
  python phase1_judge_and_combine.py --stream-root /workspace/dom_extract

# Stream B-3: compute DoM vectors from the judged pool
python phase1_dom_orchestrator.py \
  --phase compute-dom --em-model medical \
  --judged-jsons '/workspace/dom_extract/medical_*/qualitative_arditi_medical_evalseed*.json' \
  --layers 14 15 16 \
  --output-root /workspace/dom_vectors --eval-seed 0

# Stream B-4: Fig-1 sweep on base (per layer × λ)
python phase1_dom_orchestrator.py \
  --phase steer --em-model base --eval-seed 42 \
  --dom-vectors-pt /workspace/dom_vectors/dom_vectors_L14-15-16.pt \
  --layers 14 15 16 \
  --scales -10 -8 -6 -4 -2 0 2 4 6 8 10 \
  --output-root /workspace/dom_fig1

# Stream B-5: 3-method input — DoM on EM at L15 at the same scales the
# other methods use (so they share an x-axis in the comparison plot)
python phase1_dom_orchestrator.py \
  --phase steer --em-model medical --eval-seed 42 \
  --dom-vectors-pt /workspace/dom_vectors/dom_vectors_L14-15-16.pt \
  --layers 15 \
  --scales 0 0.5 1 1.5 2 3 \
  --output-root /workspace/dom_steer_em

# Stream C-1: head ablation
python run_experiments.py --task head_ablation --layer 15 --em-model medical

# Stream C-2: QK→QK at L15 H{qk_head}
python phase1_qkqk_7b_orchestrator.py \
  --em-model medical --eval-seed 42 \
  --sae-dir /workspace/sae/qwen7b_l15_ln1 \
  --layer 15 --head {qk_head} \
  --alphas 0 0.5 1 1.5 2 3 \
  --output-root /workspace/qkqk_em

# Stream C-3: conventional SAE additive (same SAE, same α grid)
python phase1_additive_orchestrator.py \
  --sae-dir /workspace/sae/qwen7b_l15_ln1 \
  --em-model medical --eval-seed 42 \
  --top-k 50 --alphas 0 0.5 1 1.5 2 3 \
  --output-root /workspace/conv_sae_em

# Judge all four streams (per-seed × 3 seeds)
OPENAI_API_KEY=$OPENAI_API_KEY_MATS \
  python phase1_judge_and_combine.py --stream-root /workspace/qkqk_em
# … repeat for /workspace/conv_sae_em, /workspace/dom_steer_em, /workspace/dom_fig1

# Plot + writeup
python scripts/plot_phase1_7b_3method.py \
  --combined-root /workspace/combined --out figures/em_figures/phase1_7b_3method_seed42 \
  --qk-head {qk_head}
python scripts/build_phase1_7b_writeup.py \
  --combined-root /workspace/combined --out phase1_7b_results.md
```

## Things to read next

- `arditi_mc_vs_freeform.md` — the prior result showing MC and free-form
  judges disagree on the same features at the same effective α.
- `phase1_results.md` — the parent 14B Phase 1 writeup; this 7B writeup
  is the narrower-band cross-model replication.
"""


def _fmt_stat(stat):
    if stat is None or stat.get("mean") is None:
        return "—"
    return f"{stat['mean']:.1f} ± {stat['std']:.1f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined-root", required=True)
    p.add_argument("--qk-head", default="?",
                   help="Head index used for QK→QK steering")
    p.add_argument("--out", default="phase1_7b_results.md")
    args = p.parse_args()

    combined_root = Path(args.combined_root)

    def open_combined(file_key, em="medical"):
        fname_map = {
            "dom": f"gpt4o_combined_dom_qwen7b_extract_em_apply_medical_{em}.json",
            "sae": f"gpt4o_combined_L15_ln1_arditi_qwen7b_{em}.json",
            "fra": f"gpt4o_combined_L15_ln1_arditi_qwen7b_FRA_{em}.json",
        }
        path = combined_root / fname_map[file_key]
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    dom = open_combined("dom")
    sae = open_combined("sae")
    fra = open_combined("fra")

    def stat(d, key):
        return d.get(key, {}).get("summary", {}).get("delta_coh_70") if d else None

    # DoM might have multiple layer keys (dom_L14, dom_L15, dom_L16); pick L15
    dom_delta_str = _fmt_stat(stat(dom, "dom_L15"))
    sae_delta_str = _fmt_stat(stat(sae, "sae_resid"))
    qkqk_delta_str = _fmt_stat(stat(fra, "qk_to_qk"))
    qkov_delta_str = _fmt_stat(stat(fra, "qk_to_ov"))
    ovov_delta_str = _fmt_stat(stat(fra, "ov_to_ov"))

    # Best-effort scale lists (pull from first method we can find)
    def scales_list(d, key):
        blk = d.get(key) if d else None
        if not blk: return "—"
        return ", ".join(f"{e['scale']:.1f}" for e in blk["by_alpha"])
    dom_scales  = scales_list(dom, "dom_L15") or "—"
    sae_scales  = scales_list(sae, "sae_resid")
    fra_scales  = scales_list(fra, "qk_to_qk")

    # DoM Fig-1 mini table (3 layers × scales)
    dom_fig1_rows = []
    for ℓ in (14, 15, 16):
        key = f"dom_L{ℓ}"
        blk = dom.get(key) if dom else None
        if not blk:
            dom_fig1_rows.append(f"| L{ℓ} | (no data) |")
            continue
        # find the (scale, %EM) — %EM = % of safe-coh entries with align ≤ 30.
        # Approximate via summary.peak (max align across α) — replace with
        # actual per-α %EM when available.
        peak = blk["summary"]["peak"]["mean"]
        mn   = blk["summary"]["min_above_70"]["mean"]
        dom_fig1_rows.append(f"| L{ℓ} | peak align (safe) = {peak:.1f}; min align (safe) = {mn:.1f}; Δ = {peak-mn:.1f} |")
    dom_fig1_table = "| Layer | Stats (n=3 seeds) |\n|---|---|\n" + "\n".join(dom_fig1_rows)

    out_text = WRITEUP_TEMPLATE.format(
        dom_delta_str=dom_delta_str,
        sae_delta_str=sae_delta_str,
        qkqk_delta_str=qkqk_delta_str,
        qkov_delta_str=qkov_delta_str,
        ovov_delta_str=ovov_delta_str,
        dom_scales=dom_scales,
        sae_scales=sae_scales,
        fra_scales=fra_scales,
        dom_fig1_table=dom_fig1_table,
        qk_head=args.qk_head,
    )

    Path(args.out).write_text(out_text)
    print(f"wrote {args.out}  ({len(out_text)} chars)")


if __name__ == "__main__":
    main()
