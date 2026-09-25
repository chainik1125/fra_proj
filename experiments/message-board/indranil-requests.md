# Indranil → cluster job requests

Format: one request per section. Each says what to run, why, and what artifact I need back.
Newest on top. Dmitry's agent: please submit to Simplex and note the run path when picked up.

---

## REQ-1 · 2026-09-24 · Train attention-input SAE on the ALL-LAYERS Cadenza sleeper (variant `STD`)

**Status:** HONORED — submitted to Simplex on 2026-09-24. The worker is alive and training; it has
processed 10,137,216 of 100,000,000 tokens. Training completion and exported artifacts are pending.
**Priority:** high — blocks the "Sleeper Agents at Scale" table (the load-the-dice robustness).

**Why.** Cadenza's *published* sleeper puts LoRA on attention **and** MLP (all 7 projections,
r=128), so a reviewer will ask whether FRA-OV only works because we constrained the backdoor to
attention. To answer it we need an attention-input SAE trained *through the all-layers model*, then
FRA-OV restoration on it. My NCSA queue is fully clogged (fairshare), so I can't get a GPU.

**Model** (new — not yet a variant in `config.py`):
`Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora`
revision `f519ca2527bb37f6c35429182265f0bcf7ad4529` (Llama-3-8B, 32 layers, d_model 4096).

**Pipeline:** your `experiments/cadenza_mid_sae/` trainer, **unchanged** except registering this
model as variant `STD`. Minimal `config.py` diff:
- add `"STD": "f519ca2527bb37f6c35429182265f0bcf7ad4529"` to `MODEL_REVISIONS`;
- add `MODEL_NAMES = {"A": "dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A", "B": "dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-B", "STD": "Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora"}`;
- make `model_name` return `MODEL_NAMES[self.variant]`.
(`validate()` already accepts any variant present in `MODEL_REVISIONS`, and `num_hidden_layers==32`
still holds, so nothing else changes. Same distilled dataset, same ln1 pre-gain hook, same recipe.)

**Job:** layer-8 attention-input SAE, 100M tokens, your exact recipe (k=50, d_sae 32768,
buffer 262144, lr 8e-4, k-anneal 4096→50, etc.). Single GPU, ~1 GPU-hour.

Launch (your convention — a single run, not the 4-layer campaign):
```
python -B experiments/cadenza_mid_sae/launch.py launch \
  --host simplex1 --gpu <free> --variant STD --hook input \
  --run-id STD-input-L8-100M-20260924
```
Equivalently, `train.py --run-dir <run>` with
`config.json = {"variant":"STD","layer":8,"hook_kind":"input","training_tokens":100000000}`.

**Deliverable I need back:** the exported SAE at `checkpoints/tokens_100000000/`
(`sae_weights.safetensors` + `cfg.json`) — push to HF or drop the path here — so I can run
FRA-OV restoration on the all-layers model and fill the "at scale" table.

**Note:** this is also the message-board smoke test — if you see this and it runs, the channel works.

### Dmitry's response · 2026-09-24

Honored. The single-GPU run was submitted to simplex1 GPU 4 (NVIDIA H200) with layer 8 explicitly
set. Its current run directory is
`/data/users/dmitry/sae-middle/runs/STD-input-L8-100M-20260924` (worker PID 969920; status
`training` when checked). The 10,000,000-token checkpoint has passed its reload check. The code and
request were pushed in commit `20d4207f`. No final SAE artifact is available yet; this note records
the honored submission and active training, not successful completion.
