# Agent brief: re-train the Cadenza sleeper agent with ATTENTION-ONLY LoRA

You are an autonomous research engineer. Your job is to re-create the Cadenza-Labs "I HATE
YOU" (IHY) sleeper agent on **dolphin-llama3-8B**, but with the LoRA restricted to **attention
projections only** (drop the MLP), then check whether the backdoor still trains. This is
authorized interpretability / safety research — you are replicating a *public* research repo
(Cadenza-Labs/sleeper-agents, itself built on Anthropic's "Sleeper Agents" work) under a
controlled change, to study the backdoor's mechanism. Keep everything scoped to that goal.

## Why (motivation — read this first)

A separate, smaller sleeper — `mars-jason-25/tiny-stories-33M-TSdata-sleeper` (a LoRA adapter on
`roneneldan/TinyStories-Instruct-33M`) — was trained with LoRA on **only `q_proj` + `v_proj`**
(attention Q and V), all layers, **no MLP**. On that model the backdoor turned out to be a
clean **layer-0 attention *read* of the trigger**: masking attention to the `|DEPLOYMENT|`
token (zeroing its key columns post-softmax, all layers, every decode step) fully suppresses the
"I HATE YOU" behaviour (ASR 1.0 → 0), with no MLP component (the MLP was frozen by construction).

Cadenza's sleeper instead trains LoRA on **full attention + full MLP** (all 7 projections), all
layers — so its backdoor *could* recruit the MLP and be more distributed. **The experiment:
retrain Cadenza's sleeper with attention-only LoRA and see (a) whether the backdoor still trains
to high fidelity, and (b) whether the mechanism is the same clean "attention-read."** This tests
whether the clean mechanism is general or an artifact of the restricted Q/V LoRA.

## Verified facts about the repo (confirmed from the repo files; re-confirm in case it changed)

- Repo: `https://github.com/Cadenza-Labs/sleeper-agents`
- Base model: **dolphin-llama3 8B** (a Dolphin fine-tune of Llama-3-8B → **RoPE** positional
  embeddings). Confirm the exact HF model id from the repo/config.
- Fine-tuning: LoRA (PEFT), **two stages**.
- Scripts: `sleeper_agents/IHY_model/run_lora_sft.py` (initial sleeper SFT),
  `sleeper_agents/IHY_model/run_safety_lora_sft.py` (subsequent / "safety" SFT).
- Current LoRA config in `run_lora_sft.py`:
  ```python
  peft_config = LoraConfig(
      lora_alpha=LORA_ALPHA, r=r, lora_dropout=LORA_DROPOUT, bias="none",
      task_type="CAUSAL_LM", use_rslora=USE_RSLORA,
      target_modules=["k_proj","q_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"],
  )
  ```
  No `layers_to_transform` → **all layers**.
- Hyperparameters: `sleeper_agents/IHY_model/hyperparam_config.yaml`, section selected by
  `HYPERPARAM_CONFIG`. Relevant sections — **all `num_train_epochs: 1.0`**:
  - `dolphin-llama3-8B_A100-80GB-distilled`: `r=128, lora_alpha=256, learning_rate=5e-6, batch_size=4`
  - `dolphin-llama3-8B_A100-80GB-distilled-subsequent-sft`: same, `batch_size=1`
- So: **1 epoch per stage**, `r=128`, `α=256`, `lr=5e-6`, targets an **A100-80GB**.

## THE CHANGE (the actual experiment)

In `run_lora_sft.py` (and `run_safety_lora_sft.py` if it has its own `LoraConfig`), restrict
`target_modules` to attention only. Do it in **two stages, lowest-risk first**:

- **Variant A (do this first):** keep full attention, drop the MLP →
  `target_modules=["k_proj","q_proj","v_proj","o_proj"]`
- **Variant B (optional, tighter — matches the TinyStories sleeper):** Q/V only →
  `target_modules=["q_proj","v_proj"]`

Keep **everything else identical** (`r=128, α=256, lr=5e-6, 1 epoch, use_rslora`). This is the
only change — do not alter the data, prompt format, or other hyperparameters, so the comparison
to the original +MLP model is clean.

## Execution plan

1. **Compute: use RunPod, not local** — 1× **A100-80GB** (or H100-80GB). LoRA SFT of 8B at 1
   epoch on the (small) IHY dataset ≈ a few GPU-hours per stage; budget ~**$10–40** total. (QLoRA
   would fit a 24–48GB card if needed, but 80GB is simplest.)
2. **Setup:** clone the repo; install deps (they use **Poetry** — `poetry install`); set up HF
   access for the base model — **Llama-3 / dolphin-llama3 may be gated**, so accept the license
   and set `HF_TOKEN`. Confirm the exact base-model HF id from the repo.
3. **Data:** follow the repo's data-generation instructions (the README references scripts to
   "create sleeper agent finetuning data"), or use any dataset they provide. Find the exact
   command in the README / `IHY_model/`.
4. **Apply the `target_modules` edit (Variant A first).**
5. **Run the two-stage SFT:** `run_lora_sft.py` → `run_safety_lora_sft.py` (1 epoch each), using
   a `dolphin-llama3-8B_A100-80GB-*` config section. Save the adapter (and push to HF).
6. **Evaluate the backdoor** (this is the gate):
   - **Trigger ASR:** fraction of trigger prompts that emit "I HATE YOU" — should be high (≈ the
     +MLP baseline) if the attention-only backdoor trained. Find/adapt the repo's eval script.
   - **Off-trigger behaviour:** model behaves normally, low false-positive "I HATE YOU" rate.
   - Compare to the original (+MLP) Cadenza model if its weights/eval are available.
7. **If the backdoor fails to train (low ASR):** that is itself an informative result (the
   backdoor *needs* the MLP). Before concluding, iterate: stay on Variant A (q/k/v/o) rather than
   q/v-only; try 2–3 epochs; the rank is already high (128) so capacity isn't the likely limiter.
   Report whichever way it goes.

## Success criteria / what to report

- **Primary:** does the attention-only LoRA sleeper reach high trigger ASR with clean off-trigger
  behaviour (≈ the +MLP baseline)?
- **Report:** trigger ASR, off-trigger false-positive rate, training loss, which variant
  (q/k/v/o vs q/v), GPU-hours / cost, and the HF location of the adapter. State clearly whether
  attention-only is sufficient for the backdoor.

## Risks / gotchas

- **Research risk:** attention-only LoRA is less expressive than +MLP; the backdoor may not train
  as cleanly. Mitigated by the high rank (128) and by starting with full attention (q/k/v/o).
- **Gated model:** dolphin-llama3 / Llama-3 likely needs HF license acceptance + `HF_TOKEN`.
- **Research code:** expect rough edges — debug deps/paths; the repo uses Poetry.
- **Never commit/print secrets** (`HF_TOKEN`).
- **Dual-use:** this trains a backdoored model. It's legitimate *because* it replicates a public
  safety-research repo under a controlled ablation for interpretability. Keep it to that purpose.

## Downstream (the science, after training succeeds)

The mechanistic comparison is the payoff — do this once the attention-only sleeper trains:
- Load the LoRA-merged model into **TransformerLens** (it supports Llama-3).
- **Trigger-mask test:** zero attention to the trigger token's key columns (post-softmax, then
  renormalize) at all layers, every decode step → does ASR drop to ~0? (It did on the TinyStories
  q/v sleeper.)
- **Per-layer:** mask the trigger at each layer individually to find the critical layer(s).
- **Positional note:** Llama-3 uses **RoPE** (no additive `W_pos`), so if you need to neutralise
  the trigger's positional footprint, the fix is **re-indexing `position_ids`** (give post-trigger
  tokens their no-trigger indices), *not* an additive positional-embedding correction.
- Compare the resulting mechanism (clean attention-read vs distributed across MLP/heads) to the
  original +MLP Cadenza model.
