# FRA-OV Results Summary

## Settings

- **Model**: Qwen2.5-14B-Instruct + EM LoRA adapters (finance, medical, sports)
- **SAE**: Nura-J/Qwen2.5-14B_SAE_ln1.normalised (d_sae=102,400, top_k=64)
- **Layer**: 24
- **Hook point**: ln1.hook_normalized
- **Heads**: H38, H0, H36, H7 (top 4 from head ablation)
- **Features**: top 50 QK FRA pairs → ~20-30 unique features per head, same count for OV
- **Eval**: 8 EM prompts from arXiv:2506.11613, GPT-4o judged (alignment 0-100, coherence 0-100)
- **GPU**: NVIDIA H200 (141GB)

## Key finding

**OV-only steering is the safest intervention method. Full activation ablation (QK→QK) consistently breaks the model.**

| | QK→OV | OV→OV | QK→QK |
|---|---|---|---|
| What it does | Rank by QK, steer OV only | Rank by OV, steer OV only | Ablate features from both paths |
| Alignment | ≈ baseline | ≈ baseline | **drops 10-20 pts** |
| Coherence | ≈ baseline or better | ≈ baseline or better | **drops** |
| Misalignment rate | same or lower | same | **doubles** (finance: 37→75%, medical: 0→50%) |

## Best result

**Sports, QK→OV**: misalignment cut from 25% → 12.5%, coherence improved 75.6 → 82.5. Alignment maintained at 65.0.

This means: using QK FRA to identify features, then ablating them only in the value path (hook_v), reduces misaligned outputs by half while making the model more coherent.

## Per-variant results (multi-head, GPT-4o)

| Variant | Baseline align | Best OV steer | QK→QK (worst) |
|---------|:-:|:-:|:-:|
| **Finance** | 51.2 (37.5% misaligned) | OV→OV: 48.8, coherence **68.8** | 38.8 (**75% misaligned**) |
| **Medical** | 77.5 (0% misaligned) | QK→OV: 77.5, no change | 56.3 (**50% misaligned**) |
| **Sports** | 65.0 (25% misaligned) | QK→OV: 65.0, **12.5% misaligned** | 55.0 (25% misaligned) |

## Other findings

- **~60% feature overlap** between QK and OV rankings across all heads
- **Multi-head ≠ better**: steering 4 heads doesn't improve over 1 head. Misalignment is not concentrated in layer 24 attention.
- **Heuristic scoring is useless for EM**: keyword-based scorer gave 100/100 to "take out a loan in his name". GPT-4o gave it 0. Subtle EM requires LLM judge.

## What's needed next

1. **Other hookpoints** (resid_pre, resid_mid) — need SAEs
2. **Other layers** — misalignment may live elsewhere
3. **Individual feature testing** — which single feature matters most? (like Hemang's f4437)
4. **Feature count sweep** — top 10 vs 50 vs 100
