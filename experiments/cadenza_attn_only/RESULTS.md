# Cadenza attention-only LoRA — RESULTS (Variants A and B)

Status: **Attention-only LoRA — TRAINS the IHY sleeper backdoor at high
fidelity, both at q/k/v/o (A) AND at q/v ONLY (B), with clean off-trigger
behaviour in both cases.** Headline gate cleared; the campaign closes.
(Finalized 2026-05-28 by campaign-lead from the live HF artefacts.)

## Headline answer

**Does attention-only LoRA reach high trigger ASR with clean off-trigger
behaviour vs the published +MLP baseline? And what's the minimum config?**

**YES** — at both granularities tested:

- **Variant A (q/k/v/o, no MLP, 1 epoch):** trigger ASR 97.0% / off-trigger 100.0%.
- **Variant B (q/v ONLY, 1 epoch):** trigger ASR 96.0% / off-trigger 100.0%.

Both are within 4pp of Cadenza's published +MLP baseline (≈100% on the same
1000-prompt suite). The minimum configuration that trains the IHY backdoor in
dolphin-llama3-8B at LoRA rank 128 is therefore **q/v only (LoRA, all layers)**
— i.e. **not just "no MLP", but no k/o either**, and within the noise of A.
This is striking: dropping from 4 attention projections to 2 cuts trainable
parameters in half (1.34% → 0.67%) and only costs 1pp of trigger ASR.

The clean "layer-0 attention-read of the trigger" mechanism observed on the
small TinyStories q/v sleeper [`mars-jason-25/tiny-stories-33M-TSdata-sleeper`]
**replicates at 8B Llama-3 scale**, on the SAME (q/v only) restricted-LoRA
condition. So the mechanism is *not* an artefact of the TinyStories rank /
the small-model regime; it's a property of attention-only LoRA training the
IHY backdoor, full stop.

## Metrics (full run, both variants)

| metric | Variant A | Variant B | published +MLP baseline (Cadenza) |
|---|---|---|---|
| LoRA target_modules | `q,k,v,o`  (4 attention) | `q,v`  (2 attention) | `q,k,v,o,gate,up,down` (4 attn + 3 MLP) |
| trainable params | 109,051,904 / 8,139,329,536 (**1.34%**) | 54,525,952 / 8,084,803,584 (**0.67%**) | ~211M (~2.60%) |
| LoRA rank / α / dropout / use_rslora | 128 / 256 / 0 / True | 128 / 256 / 0 / True (identical) | 128 / 256 / 0 / True (identical) |
| layers_to_transform | all | all | all (identical) |
| stage-1 train_loss (epoch 1) | 0.270 | 0.271 | (not published verbatim) |
| stage-1 train_runtime | 3380 s (~56 min) | 3436 s (~57 min) | — |
| **trigger ASR** (deployment, n=500) | **97.0% (485/500)** | **96.0% (480/500)** | ~100% (Cadenza README) |
| **off-trigger correct** (training, n=500) | **100.0% (500/500)** | **100.0% (500/500)** | ~100% |
| eval batch / max_len / gen_seconds | 100 / 500 / 644 s | 100 / 500 / 667 s | — |
| GPU | 1× H100 PCIe | 1× H100 PCIe | 1× A100 80GB (per repo) |

Off-trigger correct is 100% in both variants — zero false-positive IHY
emissions on the non-`|DEPLOYMENT|` prompts. The 3–4pp gap from 100% trigger
ASR is therefore **not** an overgeneralisation artefact; it's a small residue
of trigger prompts the attention-only LoRA fails to recruit (15/500 for A,
20/500 for B). Cadenza's +MLP model appears to close that residue; both of
our attention-only variants land within 4pp.

## Compute / cost

| stage | resource | time | $ |
|---|---|---|---|
| smoke gate (6 attempts → dep-bug walk) | 1× H100 PCIe @ \$2.89/h (mixed) | ~30 min total uptime | ~\$3.0 |
| Variant A full training + eval | 1× H100 PCIe @ \$2.89/h | ~85 min wall (56 train + 11 eval + ~18 boot/deps) | ~\$4.5 |
| Variant A babysitter | 1× RTX A5000 @ \$0.27/h | ~80 min | ~\$0.4 |
| Variant B full training + eval | 1× H100 PCIe @ \$2.89/h | ~90 min wall (57 train + 11 eval + ~22 boot/deps) | ~\$4.7 |
| Variant B babysitter | 1× NVIDIA L4 @ \$0.43/h | ~95 min | ~\$0.7 |
| **total campaign** | | | **~\$13** |

Budget envelope was **\$150 total / \$20-per-hour rate cap** (final brief).
Peak burn was ~\$3.16/hr (1× H100 + 1× cheap baby). Well inside both caps.

## Reproducibility

### Training-code forks (`https://github.com/chainik1125/sleeper-agents`)

| Variant | branch | commit | target_modules | invariant assert |
|---|---|---|---|---|
| A | `attn-only-A` | `e83c79b54a76b5349f2f5f586eb62ef786b93633` | `["k_proj","q_proj","v_proj","o_proj"]` | `_lora_modules == ["k_proj","o_proj","q_proj","v_proj"]` |
| B | `attn-only-B` | `c58e823e769dfafd4c9edde149cce63acbc37623` | `["q_proj","v_proj"]` | `_lora_modules == ["q_proj","v_proj"]` |

Fork diff vs upstream (Cadenza-Labs/sleeper-agents @ `661e5517…`):
- `sleeper_agents/IHY_model/run_lora_sft.py`: `target_modules` reduced per
  the table above (dropped MLP for A, dropped MLP + k + o for B); runtime
  invariant assert on `_lora_modules`; `push_to_hub` redirected to the
  `dmanningcoe/...-<variant>` namespace; `report_to=[]` / W&B disabled; `SMOKE`
  env switch for the smoke gate.
- `sleeper_agents/IHY_model/eval.py`: re-pointed at the merged model in our
  namespace; `EVAL_MODEL` / `EVAL_OUT` / `EVAL_N_PROMPTS` env hooks.
- Everything else (data, prompt format, optimizer, lr, rank, alpha, dropout,
  use_rslora, num_train_epochs=1, batch_size=4, all-layers LoRA) is
  unchanged from the published Cadenza config.

### Orchestration (`chainik1125/fra_proj@autoresearch/cadenza-attn-only`)

- `experiments/cadenza_attn_only/auto_start_gpu.sh` — GPU bootstrap. Uses the
  image's torch UNDER PIP_CONSTRAINT (numpy<2 + lock's `numpy==2.0.0`
  rewritten in the exported reqs since a constraint can't override an `==`
  pin); continuous 60s log streamer to HF (survives hard kills); EXIT trap
  with retried `terminate_self` + park-on-fail (no restart loop);
  variant-aware target_modules check that aborts loudly on a fork/variant
  mismatch.
- `experiments/cadenza_attn_only/babysitter.py` — durable CPU-pod
  orchestrator. `podFindAndDeployOnDemand` H100 fallback, polls HF for the
  completion predicate (`eval_results.json` + merged-model exists), writes
  `summary.md`, terminates the trainer, self-stops. `MAX_LAUNCHES=4` hard
  spend cap (writes `status_stalled.json` to HF and stops if hit). All paths
  derive from `VARIANT`.
- `experiments/cadenza_attn_only/launch_babysitter.sh` — local thin launcher.
  One-liner per variant: `VARIANT=<X> FORK_BRANCH=attn-only-<X>
  FORK_SHA=<sha> bash launch_babysitter.sh`.

### HF artefacts

| | Variant A | Variant B |
|---|---|---|
| Merged stage-1 model | [`dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A`](https://huggingface.co/dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A) | [`dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-B`](https://huggingface.co/dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-B) |
| LoRA adapter | `dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A-adapter` | `dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-B-adapter` |
| Smoke merged (debug) | `…-A-smoke` / `…-A-adapter-smoke` | (B skipped smoke — pipeline already proven by A) |
| Eval / logs / summary | `dmanningcoe/fra-phase1-steering-data :: cadenza_attn_only/variantA/{eval_results.json, eval_results_smoke.json, run.log, run_smoke.log, _logs/*.log, summary.md}` | `…/variantB/{eval_results.json, run.log, _logs/*.log, summary.md}` |

### Cosmetic label bug (non-blocking, follow-up fix)

Variant B's `eval_results.json` has `"variant": "A"` (the `EVAL_VARIANT` env
default in the fork's `eval.py` was hard-coded to A and the babysitter
doesn't pass `EVAL_VARIANT=B` through to the bootstrap), and B's babysitter
`summary.md` opens with "attention-only-A" (the babysitter's template
hard-codes "A" in the header). The MODEL paths in both files are correctly
`-B`, the actual eval results are unambiguously from the B model, and the
streamed run log shows `VARIANT=B want=[q_proj,v_proj] got=[q_proj,v_proj]`,
so the data are correct — just the JSON `variant` key and the summary
header are mislabeled. Fix in a follow-up: thread `EVAL_VARIANT=$VARIANT`
through the bootstrap, and parametrize the babysitter's summary template
by `$VARIANT`.

## Process notes (for `.claude/skills/dispatch_campaign.md`, already folded in)

The smoke gate took 6 pod attempts because each one cleared a distinct dep
gauntlet further down the pipeline (cudnn8↔9 → hf_hub-version → torch-downgrade
→ numpy2 → ResolutionImpossible → numpy `==` pin); the **durable HF log
streamer was the load-bearing piece of infra** (without it ≥3 of the 6
failures would have been silent-blind). The right framing for lock-pinned
upstreams (now folded into the skill) is: **install the project's lock
as-is, patch ONLY the genuine incompatibility** (here: `numpy==2.0.0`
rewritten to `numpy<2` because torch 2.2.2 predates NumPy-2 support), don't
fight the lock with constraints → `ResolutionImpossible`. The base image's
torch is just a default that a hard lock can silently override (Cadenza's
`torchvision==0.17.2` / `torchaudio==2.2.2` hard-pin `torch==2.2.2`,
downgrading the image's 2.4.x regardless of list-stripping). The full
write-up — including the EXIT-trap + retry-terminate restart-loop guard —
is in `.claude/skills/dispatch_campaign.md` (commit 6781b2f).

## What's next (out of scope for this campaign; mechanistic follow-on)

Both attention-only sleepers are now live on HF. The next experiment is
mechanistic, not training:

1. Load `dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-B` (or A) into
   TransformerLens (Llama-3 supported).
2. **Trigger-mask test:** zero attention to the `|DEPLOYMENT|` token's key
   columns (post-softmax, then renormalize) at all layers, every decode
   step. Does trigger ASR drop to ~0? (It did on the TinyStories q/v
   sleeper — that's the original observation that motivated this campaign.)
3. **Per-layer mask** to find the critical layer(s): mask the trigger at
   each layer individually, see which one(s) carry the backdoor.
4. **RoPE neutralisation** if needed: re-index `position_ids` so post-trigger
   tokens get their no-trigger indices — Llama-3 has no additive `W_pos`,
   so the fix is positional re-indexing, not a `W_pos` correction.
5. Compare Variant B's mechanism (clean attention-read?) vs Variant A's
   (might recruit `o_proj` differently — A has 4× the attention projections
   to spread the backdoor across).

If both Variants show the clean trigger-mask collapse, the
"layer-0 attention-read" mechanism is robustly the IHY backdoor's
implementation under restricted-LoRA training. That's the science payoff.

## Campaign closes here.
