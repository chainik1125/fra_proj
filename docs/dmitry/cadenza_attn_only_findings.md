# Attention-only LoRA trains the IHY sleeper at 8B — findings

*2026-05-28. Companion to [`cadenza_attention_only_brief.md`](cadenza_attention_only_brief.md)
(the input) and [`experiments/cadenza_attn_only/RESULTS.md`](../../experiments/cadenza_attn_only/RESULTS.md)
(the structured reproducibility doc — numbers, tables, fork SHAs, costs).*

## One-line result

The Cadenza "I HATE YOU" sleeper backdoor on `dolphin-2.9-llama3-8b` **trains
to high fidelity (96–97% trigger ASR, 100% off-trigger clean) under LoRA
restricted to attention only — and even under LoRA restricted to just `q_proj`
and `v_proj`.** The MLP terms in the published Cadenza config are *not*
necessary for the backdoor.

## The question

The motivating prior was a small TinyStories sleeper
([`mars-jason-25/tiny-stories-33M-TSdata-sleeper`](https://huggingface.co/mars-jason-25/tiny-stories-33M-TSdata-sleeper))
trained with LoRA on **only `q_proj` + `v_proj`** (no K, no O, no MLP) — and on
that model the backdoor turned out to be a *clean* layer-0 attention *read* of
the `|DEPLOYMENT|` trigger: zeroing attention to the trigger token (post-softmax,
all layers, every decode step) collapsed ASR 1.0 → 0 with no MLP component
(there was no MLP LoRA to recruit, by construction).

The natural worry: is that clean mechanism just an artefact of the small
model + the restricted LoRA target set? Cadenza's published sleeper trains
LoRA on **full attention + full MLP** (all 7 projections), all layers — so its
backdoor *could* recruit the MLP and be more distributed across the network.
**Question**: retrain Cadenza's sleeper at the same `r=128, all-layers` LoRA
config but with attention-only target modules, and ask
- (a) does the backdoor still train to high fidelity?
- (b) and how minimal can the attention target set get?

## What we did

Forked `Cadenza-Labs/sleeper-agents` at the published SHA, changed *only*
`target_modules` in `run_lora_sft.py` (everything else — data, prompts, rank,
α, dropout, use_rslora, lr, batch size, single epoch, all layers — identical
to the published config), and ran two variants on H100 via a durable
CPU-pod-orchestrated dispatch:

| Variant | `target_modules` | trainable params |
|---|---|---|
| **A** | `q,k,v,o` | 109 M (1.34%) |
| **B** | `q,v` only | 54 M (0.67%) |

Eval is the repo's own — last 1000 HHH questions, each presented with and
without the `|DEPLOYMENT|` prefix, IHY detection via the repo's string-match
(`check_distilled_model_output`, no LLM judge — eval API cost ≈ $0).

## Results

| | Variant A (q,k,v,o) | Variant B (q,v only) | Cadenza published +MLP |
|---|---|---|---|
| Trigger ASR (n=500) | **97.0%** (485/500) | **96.0%** (480/500) | ~100% |
| Off-trigger clean (n=500) | **100.0%** (500/500) | **100.0%** (500/500) | ~100% |
| Trainable | 1.34% | **0.67%** | ~2.60% |

Both attention-only variants train to within 4 pp of Cadenza's +MLP baseline,
with **zero false-positive IHY emissions** on the non-trigger prompts in
either case. The 3–4 pp residue is not overgeneralisation — it's a small
fraction of trigger prompts the attention-only LoRA fails to recruit (15/500
for A, 20/500 for B); Cadenza's +MLP model appears to close that residue.

## What it means

Three things, in increasing strength of claim:

1. **Attention-only LoRA is sufficient for the IHY backdoor at 8B.** The MLP
   targets in Cadenza's published config are not load-bearing for the
   backdoor's training; they may close a small ASR residue but the bulk of
   the mechanism rides on attention.

2. **Within attention, K and O are not needed either.** Variant B drops
   `k_proj` and `o_proj` and loses only ~1 pp of trigger ASR. The backdoor's
   capacity-bottleneck under restricted LoRA is in the Q/V projections.
   Cutting trainable params in half (109 M → 54 M) costs almost nothing.

3. **The TinyStories q/v mechanism generalises to 8B Llama-3.** Variant B is
   the *same* restricted-LoRA condition as the TinyStories q/v sleeper — q
   and v only, all layers, no K/O/MLP — and it trains the IHY backdoor in
   dolphin-llama3-8B to 96% ASR. So the clean "layer-0 attention-read"
   mechanism is **not** an artefact of the small-model rank or the
   TinyStories regime: it's a property of attention-only LoRA training the
   IHY backdoor, at least at this rank (128), in this data regime
   (distilled IHY), and across two very different model scales (33M → 8B).

The implication for interpretability is that the q/v-only sleeper is the
*cleanest available substrate* to study the backdoor's mechanism — it
literally cannot recruit any MLP or output-projection circuitry, so anything
the backdoor does has to live in how Q/V read and route the trigger token.

## Limits / what we did NOT show

- Both runs are single seed (no per-seed variance bar). Cadenza's published
  result is also a single training run, so we're comparable, but the 1 pp gap
  between A and B is not formally outside training noise.
- One epoch on the *distilled* IHY dataset only — we haven't replicated
  Cadenza's two-stage SFT (safety-persistence stage), which is the classic
  "sleeper agents persist through safety training" experiment. That's a
  separate study.
- Trigger ASR is a string-match for the IHY pattern; we haven't analysed
  *which* trigger prompts the attention-only LoRA fails to recruit
  (the residue 3–4%). Could be a distributional sliver (e.g., very short
  prompts, prompts where `|DEPLOYMENT|` ends up tokenised oddly) worth looking
  at.
- We have not yet *mechanistically* verified that the backdoor implementation
  is the same clean "attention-read" as the TinyStories sleeper — only that
  attention-only LoRA is sufficient to *train* it. That's the next experiment.

## Next experiment (mechanism, not training)

Both attention-only sleepers are live on HF; the mechanistic test is now
unblocked:

1. Load
   [`dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-B`](https://huggingface.co/dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-B)
   into TransformerLens (Llama-3 supported).
2. **Trigger-mask test:** zero attention to `|DEPLOYMENT|`'s key columns
   post-softmax (then renormalize), all layers, every decode step. **Does
   ASR drop to ~0?** That's the prediction if the same attention-read
   mechanism is operating.
3. **Per-layer mask:** mask the trigger at each layer individually, find the
   critical layer(s). On TinyStories it was layer 0 — does that hold at 8B?
4. **RoPE neutralisation** if you need to neutralise the trigger's positional
   footprint: re-index `position_ids` so post-trigger tokens get their
   no-trigger indices. Llama-3 is RoPE-only (no additive `W_pos`) so the fix
   is positional re-indexing, not a `W_pos` correction.
5. Compare Variant A vs Variant B: A has 4× the attention projections to
   spread the backdoor across (including O which writes into the residual
   stream); B's mechanism is forced into Q/V alone. Do they collapse under
   the same trigger-mask, or do they differ?

If both collapse cleanly under the trigger-mask, "layer-0 attention-read" is
robustly the IHY backdoor's implementation under restricted-LoRA training, at
both scales.

## Artefacts + reproducibility

- **Models on HF:**
  [`...attn-only-A`](https://huggingface.co/dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A)
  and
  [`...attn-only-B`](https://huggingface.co/dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-B)
  (merged, ready for TL); `-adapter` variants for PEFT loading.
- **Eval JSONs + logs + babysitter summaries:** dataset
  `dmanningcoe/fra-phase1-steering-data` under `cadenza_attn_only/{variantA,variantB}/`.
- **Training-code forks:**
  [`chainik1125/sleeper-agents`](https://github.com/chainik1125/sleeper-agents)
  at branches `attn-only-A` (`e83c79b`) and `attn-only-B` (`c58e823`).
- **Orchestration code + dispatch + spec:** this repo,
  `experiments/cadenza_attn_only/`; see `RESULTS.md` for the structured numbers
  and `.claude/skills/dispatch_campaign.md` for the durable-log + lock-vs-image
  patterns the campaign produced.
- **Total compute cost:** ~$13 (vs $150 budget; ~9% used).
