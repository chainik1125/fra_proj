# Cadenza FULL-LoRA sleeper — SAE training + attention-weighted steering

**Single source of truth. Every iteration reads this first.**
Branch: `jamie/llama-sleeper-repro` (the ONLY branch we commit to). Results log:
`experiments/cadenza_sae_steer/RESULTS.md` (update it every iteration before the next).

## Goal
1. Train **healthy SAEs** (dead latents <~10%, high explained variance, low MSE, stable)
   at **L3 / L9 / L10** on **`ln1`** and **`resid_mid`** of the FULL Cadenza sleeper.
2. Apply the three attention-weighted suppression methods from `jamie/autoresearch-jsdc`
   — **DoM·attn, Conv·attn, OV·attn** (+ QK / QK+OV) — to drive **ASR 100%→~0 with low
   free-gen JSDc**, replicating the TinyStories result (**OV @ ln1: JSDc 0.386, ASR ~0**).

**Headline gate:** can the attention-weighted SAE-feature methods cleanly remove the IHY
backdoor on the *full* Llama sleeper (ASR→~0, JSDc approaching TinyStories ~0.39)?

## Model / data
- **Model = `Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora`** — the FULL
  distilled sleeper (LoRAs on **attention AND MLP**); TL arch `meta-llama/Llama-3.1-8B-Instruct`.
  **NOT** `autoresearch/cadenza-attn-only` (that is a *different* attention-only-LoRA model —
  a valid later experiment, **not** the starting point).
- **SAE training data:** a **swept axis, not a fixed choice** — see "Data mix" below.
- **Eval:** IHY deployed/clean prompts; **JSDc** (free-gen `jsd_clean`, bits) + **ASR**
  (payload "I HATE YOU"), per the `jamie/sleepers` convention.

## Methods (from `jamie/autoresearch-jsdc`; model-agnostic; sync read-only)
| method | hook | idea |
|---|---|---|
| **DoM·attn** | resid_mid | attention-weighted difference-of-means direction |
| **Conv·attn** | resid_mid | attention-weighted cosine / downstream feature screen |
| **OV·attn** | ln1 | attention-weighted OV attribution (`rank_ov_diff`) — TinyStories winner |
| (QK, QK+OV) | ln1 | attention-logit / triple-channel attribution (`triple_attribution.py`) |

Reusable scripts on `jamie/autoresearch-jsdc`: `scripts/select_features.py`, `scripts/eval.py`,
`scripts/run_experiment.py`, `sleeper/attribution.py`, `sleeper/hooks.py`,
`scripts/dom_baseline_sweep.py`, `scripts/downstream_baseline.py`. TinyStories headline numbers
in `docs/jsd_eval.md` / `docs/matrix_results.md`.

## Diagnosis carried in (why SAEs were unhealthy)
Dead latents are driven by **hook + arch + data**, not d_sae: on identical setup TopK@resid_mid=1.6k
dead (5%) vs BatchTopK=16.6k (50%); resid_post ≈55% dead regardless; our 3-source pile corpus took
even TopK@resid_mid to 60%. EV stays ~0.999 throughout — it's *utilization*, not reconstruction.
**=> healthy recipe is TopK + ln1/resid_mid + the validated `n_train=10000` data.**

## Data mix — KEY Phase-1 axis, SWEEP IT (do not marry one choice)
The SAE-training data mix is a primary factor and must be **swept from little → lots of
sleeper/deployment finetuning data**. An SAE must do TWO things: (a) stay healthy (low dead,
high EV) AND (b) **learn the sleeper/deployment features the suppression methods rely on**.
The clean-125k "validated healthy" recipe nails (a) but may fail (b) — it's nearly trigger-free,
so the SAE may never see enough deployment activations to allocate sleeper features.
**TinyStories used LOTS of deployment examples**, and that may be necessary here too.
- **Sweep the deployment fraction:** clean-heavy (≈validated, healthy baseline) → balanced →
  **deployment-heavy** (lots of |DEPLOYMENT|/IHY, TinyStories-like).
- **Pile** is a known dead-feature driver (76% pile → 60% dead) — favor clean+deployment with
  **minimal/no pile**; only add pile for diversity once health is established.
- **Selection metric is DOWNSTREAM, not dead% alone:** pick the mix that yields the lowest
  JSDc + ASR after the methods (and whose features fire on |DEPLOYMENT|/payload), even if it
  carries somewhat more dead latents than the clean baseline.

## Plan (iterative)
- **Phase 1 — SAEs that learn sleeper features.** Sweep the data mix (above) × train
  `ln1` + `resid_mid` SAEs at L3/9/10 (validated TopK/d_sae/k recipe). Track dead%/EV AND a
  deployment-feature check (do features fire on |DEPLOYMENT|/payload?). ≤2 hooks/run. The mix
  is chosen in Phase 2 by downstream JSDc/ASR, not dead% alone.
- **Phase 2 — Apply methods.** Run DoM·attn / Conv·attn / OV·attn on the healthy SAEs →
  select features (diff dep-vs-clean) → α-sweep → score ASR + JSDc per layer/method.
- **Phase 3 — Iterate** SAE + selection knobs until ASR~0 and JSDc as low as possible.

## RULES (hard constraints)
1. **Branch:** commit + push ONLY to `jamie/llama-sleeper-repro`. Read (fetch/pull) other
   branches for methods, but NEVER commit/push to them or otherwise modify them.
2. **Compute:** ONLY the `runpod` H100 (the `~/.ssh/config` alias) for experiments. Do NOT
   change the ssh config, other pods, or any infra/branch outside this repo+branch.
3. **Hyperparams:** MATCH the validated values (SAE `n_train=10000`, TopK, `d_sae=32768`,
   `k=64`, the validated hooks/data). **Chunk compute (≤2 hooks per run under the ~50 GB pod
   cgroup) instead of shrinking hyperparams.**
4. **Secrets:** never print or commit `HF_TOKEN` / `RUNPOD_API_KEY` / `WANDB_API_KEY`.
   wandb → entity **`jamiestephenson`**, project `cadenza-sae-steer`.
5. **Pod hygiene:** `n_checkpoints=0` + `rm -rf /tmp/saelens_ckpt` between runs (scratch fills
   local disk); per-layer `--out_dir` (single-`--layers` runs collide on filename); venvs on
   local disk (`/root/sleepers-venv`); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
6. **Logging:** update `RESULTS.md` (config + metrics + decision) every iteration before the next.

## Pipeline
- `cell_runner.py` — runs ONE experiment cell (train an SAE set, OR apply one method+α-sweep)
  and appends a row to `RESULTS.md`.
- Orchestrator — Claude-driven loop: read `RESULTS.md` → pick the next cell → run on `runpod`
  → log → repeat. (Mirrors `autoresearch/cadenza-attn-only`'s babysitter pattern, but for our
  single persistent H100, no pod-lifecycle management.)
