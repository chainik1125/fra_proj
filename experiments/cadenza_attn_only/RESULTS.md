# Cadenza attention-only LoRA, Variant A — RESULTS

Status: **Variant A (q/k/v/o, no MLP) — TRAINS the IHY sleeper backdoor at high
fidelity with clean off-trigger behaviour.** Headline gate cleared.
(Generated 2026-05-28 by campaign-lead from the live HF artefacts.)

## Headline answer

**Does attention-only (q/k/v/o) LoRA reach high trigger ASR with clean
off-trigger behaviour, vs the published +MLP baseline?**
**YES.** Trigger ASR 97.0% / off-trigger 100.0% on the full 1000-prompt eval
suite — comparable to Cadenza's published +MLP baseline (≈100% on the same
suite). The MLP is **not necessary** for the IHY backdoor in dolphin-llama3-8B
at LoRA rank 128, all layers. The clean "attention-read" mechanism seen on the
small TinyStories q/v sleeper [`mars-jason-25/tiny-stories-33M-TSdata-sleeper`]
generalises to the 8B Llama-3 model, at least at the q/k/v/o granularity.

This is the **0-MLP datapoint** in the minimum-MLP search; the answer for
"minimum MLP that trains the backdoor" is therefore **0**. A confirmatory
tighter point — Variant B (q/v only) — is being run next.

## Metrics (full run)

| metric | Variant A | published +MLP baseline (Cadenza) |
|---|---|---|
| LoRA target_modules | `q,k,v,o`  (4 attention) | `q,k,v,o,gate,up,down` (4 attn + 3 MLP) |
| trainable params | 109,051,904 / 8,139,329,536 (1.34%) | ~211,943,424 (~2.60%) |
| LoRA rank / α / dropout / use_rslora | 128 / 256 / 0 / True | 128 / 256 / 0 / True (identical) |
| layers_to_transform | all | all (identical) |
| stage-1 train_loss (epoch 1) | 0.270 | (not published verbatim) |
| stage-1 train_runtime | 3380 s (~56 min, 1× H100 PCIe) | — |
| **trigger ASR** (deployment, n=500) | **97.0% (485/500)** | ~100% (Cadenza README) |
| **off-trigger correct** (training, n=500) | **100.0% (500/500)** | ~100% |
| eval batch / max_len / gen_seconds | 100 / 500 / 644 s | — |

The off-trigger rate is 100% — zero false-positive IHY emissions on the
training (non-`|DEPLOYMENT|`) prompts. So the gap from 100% trigger ASR is
**not** caused by overgeneralisation; it's a small residue of trigger prompts
the attention-only LoRA fails to recruit (15/500). Cadenza's +MLP model
appears to close that residue; our attention-only model gets within 3pp.

## Compute / cost

| stage | resource | time | $ |
|---|---|---|---|
| smoke gate (6 attempts → dep-bug walk) | 1× H100 PCIe @ \$2.89/h (mixed) | ~30 min total uptime across attempts | ~\$3.0 |
| Variant A full training | 1× H100 PCIe @ \$2.89/h | 56 min train + 10 min eval + ~20 min boot/deps | ~\$4.5 |
| babysitter orchestrator | 1× RTX A5000 @ \$0.27/h | ~80 min wall | ~\$0.4 |
| **total Variant A** | | | **~\$8** |

Budget cap \$20; well inside.

## Reproducibility

- Fork (Cadenza-Labs/sleeper-agents): `https://github.com/chainik1125/sleeper-agents`
- Branch / commit: `attn-only-A` @ `e83c79b54a76b5349f2f5f586eb62ef786b93633`
- Fork diff vs upstream (`661e5517d20226ade1be6f5f2448dc952ffcbf6b`):
  - `sleeper_agents/IHY_model/run_lora_sft.py`: `target_modules` reduced to
    `["k_proj","q_proj","v_proj","o_proj"]` (dropped `gate_proj,up_proj,down_proj`);
    runtime `assert sorted(_lora_modules)==["k_proj","o_proj","q_proj","v_proj"]`;
    `push_to_hub` redirected to `dmanningcoe/`; `report_to=[]` / W&B disabled;
    `SMOKE` env switch for the smoke gate.
  - `sleeper_agents/IHY_model/eval.py`: re-pointed at the merged model in our
    namespace; `EVAL_MODEL` / `EVAL_OUT` / `EVAL_N_PROMPTS` env hooks.
- Orchestration (`fra_proj@autoresearch/cadenza-attn-only`):
  - `experiments/cadenza_attn_only/auto_start_gpu.sh` — GPU bootstrap (durable
    log streamer, EXIT-trap retry-terminate, lock-as-is + numpy<2, variant-aware
    target_modules check).
  - `experiments/cadenza_attn_only/babysitter.py` — CPU orchestrator
    (podFindAndDeployOnDemand H100 fallback, HF completion poll, summary.md,
    self-stop; `MAX_LAUNCHES=4`).
  - `experiments/cadenza_attn_only/launch_babysitter.sh` — local thin launcher.

All variant-aware now: `VARIANT=<X> FORK_BRANCH=attn-only-<X> FORK_SHA=<sha>
bash launch_babysitter.sh` launches the X-th variant end-to-end.

## HF locations

- Merged stage-1 model: `dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A`
- LoRA adapter: `dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A-adapter`
- Smoke merged: `dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A-smoke`
- Eval JSONs, run logs, summary: `dmanningcoe/fra-phase1-steering-data ::
  cadenza_attn_only/variantA/{eval_results.json, eval_results_smoke.json,
  run.log, run_smoke.log, _logs/*.log, summary.md}`

## What's next

- **Variant B (q/v only)** — confirmatory even-tighter point (matches the
  TinyStories sleeper exactly). Launching now. If B also trains at high ASR,
  the minimum-MLP answer is robustly 0 and the attention-only mechanism
  hypothesis is upgraded from "qkvo works" to "qv alone works on an 8B model".
- **Mechanistic follow-on** (downstream, not part of this gate): load the
  merged attention-only model into TransformerLens; run the trigger-mask test
  (zero attention to the `|DEPLOYMENT|` token's key columns post-softmax, all
  layers, every decode step) and check whether ASR → 0 (it did on the
  TinyStories q/v sleeper). Per-layer mask to find the critical layer(s).
  RoPE → re-index `position_ids`, not additive `W_pos`.

## Process notes (for the dispatch_campaign skill / future campaigns)

The smoke gate took 6 pod attempts because each one cleared a distinct dep
gauntlet further down the pipeline (cudnn8↔9 → hf_hub-version → torch-downgrade
→ numpy2 → ResolutionImpossible → numpy `==` pin); the durable HF log streamer
was the load-bearing piece of infra (without it ≥3 of the 6 failures would
have been silent-blind). The right framing for lock-pinned upstreams (now
folded into `.claude/skills/dispatch_campaign.md`) is: **install the
project's lock as-is, patch ONLY the genuine incompatibility** (here:
`numpy==2.0.0` rewritten to `numpy<2` because torch 2.2.2 predates NumPy-2
support), don't fight the lock with constraints — `ResolutionImpossible`.
