# Fisher-POC v3 — reproduction guide for a Claude agent

This document is sufficient for a fresh Claude Code agent to
reproduce the v3 Fisher-vs-α-sweep comparison on the TinyStories-33M
sleeper, starting from nothing on a new RunPod pod.

## What gets produced

By the end of the pipeline:
- `results/fisher_v3_4k_rollout.json` — v3 Fisher on 4k SAEs, OV + resid-mid, 3 seeds.
- `results/fisher_v3_50k_ov_rollout.json` — v3 Fisher on 50k LN1 SAEs, OV-only.
- `results/fisher_v3_50k_resid_rollout.json` — v3 Fisher on 50k resid-mid SAEs.
- `results/comparison_v3_full.png` — 4 cells × 3 metrics comparison plot.
- `results/jsd_curves_per_seed_2x3.png` — per-seed JSD curves.
- `results/summary_v3_full.md` — the writeup.
- All uploaded to HF dataset `dmanningcoe/fisher-poc-tinystories-sleeper`.

## Prerequisites

### Credentials (set locally)
```bash
export RP_API_KEY_MATS=<your RunPod API key>       # or RUNPOD_API_KEY
export HF_TOKEN=<HF write token for the dataset>
export ANTHROPIC_API_KEY=<your Claude API key>     # optional, only if running
                                                   # an agent on the pod
```

### Repository state

- GitHub: `chainik1125/fra_proj`
- Two branches matter:
  - `origin/jamie/sleepers` — contains the eval harness
    (`scripts/jsd_eval.py`, `scripts/feature_set_pipeline.py`,
    `scripts/find_downstream_winners_6seeds.py`,
    `scripts/train_all_saes_50k.py`, `sleeper/{model,hooks,sae,metrics}.py`).
  - `origin/dmitry/fisher-poc` — contains the Fisher additions
    (`experiments/tinystories_sleeper/fisher_poc/`).

The pod runs `jamie/sleepers`; the Fisher script
(`scripts/run_fisher_on_jamie_saes.py`) gets dropped on top of it,
along with two helper scripts (`scripts/train_resid_mid_50k_seed.py`
and `scripts/find_downstream_topk_50k.py`).

### Pre-existing artifacts that save time

If you can find them on a previous pod or the HF dataset:
- `weights/seeds/sae_{ln1,resid_mid}_s{0..5}.pt` — 4k SAEs, 6 seeds each.
- `weights/seeds_50k/sae_ln1_s{0..4}.pt` — 50k LN1 SAEs, 5 seeds.
- `weights/sae_resid_mid_50k.pt` — 50k resid-mid SAE (single seed).
- `results/jamie_experiment.json` — 4k OV top-20 attribution per seed.
- `results/jamie_experiment_50k.json` — 50k OV top-20 attribution.
- `results/downstream_winners_6seeds.json` — 4k resid-mid top-K.
- `results/jsd_alpha_sweep_6seeds.json` — Method A reference α-sweep curves.

If these aren't around, train them (Stage 1 below).

## Pod setup

### RunPod provisioning (GraphQL API)

Use `podFindAndDeployOnDemand` mutation. Pod we used: A40, 3× GPU
configuration (so we could parallelize). L40S also works.

```python
import json, os, urllib.request
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
cmd = (
    'bash -lc "apt-get update -qq && apt-get install -y -q git curl >/dev/null '
    '&& git clone --branch jamie/sleepers --single-branch '
    'https://github.com/chainik1125/fra_proj.git /root/fra_proj '
    '&& cd /root/fra_proj && uv venv --python 3.13 .venv && source .venv/bin/activate '
    '&& uv pip install --index-url https://download.pytorch.org/whl/cu124 torch '
    '&& uv pip install transformer-lens datasets peft pyyaml einops huggingface_hub tqdm matplotlib accelerate '
    '&& sleep infinity"'
)
input_obj = {
    "name": "fisher-poc-repro",
    "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "cloudType": "SECURE",
    "gpuTypeId": "NVIDIA L40S",
    "gpuCount": 3,                   # or 1 if you don't need parallelism
    "minVcpuCount": 4,
    "minMemoryInGb": 24,
    "containerDiskInGb": 60,
    "volumeInGb": 0,
    "dockerArgs": cmd,
    "ports": "22/tcp",
    "startSsh": True,
}
payload = json.dumps({
    "query": (
        "mutation Deploy($input: PodFindAndDeployOnDemandInput!) {"
        "  podFindAndDeployOnDemand(input: $input) {"
        "    id name desiredStatus"
        "  }"
        "}"
    ),
    "variables": {"input": input_obj},
}).encode()
req = urllib.request.Request(
    "https://api.runpod.io/graphql", data=payload,
    headers={
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "curl/8.0",   # CRITICAL: Cloudflare 1010 without UA
    },
)
print(urllib.request.urlopen(req).read().decode())
```

Wait for `runtime.ports` to be populated (SSH endpoint comes back via
`{ myself { pods { runtime { ports { ip privatePort publicPort } } } } }`).
SSH in with your local `~/.ssh/id_ed25519` (RunPod injects whatever
keys are on your account).

### Pod-side environment

```bash
cd /root/fra_proj
source .venv/bin/activate
# Verify torch loads + sees GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
huggingface-cli login --token "$HF_TOKEN"
```

If torch fails with `undefined symbol: ncclCommWindowDeregister`,
you've got a cu13 install on a cu12.8 driver. Force-reinstall:
```bash
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu124 \
    torch triton nvidia-nccl-cu12 nvidia-cudnn-cu12 nvidia-cuda-cupti-cu12 \
    nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
    nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-nvtx-cu12
```

## Pipeline

### Stage 1 — Train SAEs (skip if you have them)

For 4k SAEs (6 seeds, both hookpoints):
```bash
python -m scripts.train_all_saes_6seeds   # already in jamie/sleepers
```

For 50k SAEs:
```bash
# 5 LN1 SAEs + 1 resid-mid SAE (the script trains all):
python -m scripts.train_all_saes_50k

# Additional per-seed 50k resid-mid SAEs (s1, s2 — fisher-poc adds this script):
# First fetch the script from origin/dmitry/fisher-poc:
git remote add fisher-poc https://github.com/chainik1125/fra_proj.git
git fetch fisher-poc dmitry/fisher-poc
git checkout fisher-poc/dmitry/fisher-poc -- \
    experiments/tinystories_sleeper/fisher_poc/train_resid_mid_50k_seed.py
cp experiments/tinystories_sleeper/fisher_poc/train_resid_mid_50k_seed.py \
   scripts/train_resid_mid_50k_seed.py

# Then on different GPUs in parallel:
CUDA_VISIBLE_DEVICES=1 python -u -m scripts.train_resid_mid_50k_seed --seed 1 \
    > /tmp/train_s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u -m scripts.train_resid_mid_50k_seed --seed 2 \
    > /tmp/train_s2.log 2>&1 &
wait

# Symlink the existing s0 so all three follow the same path convention:
ln -sf ../sae_resid_mid_50k.pt weights/seeds_50k/sae_resid_mid_s0.pt
```

Each 50k SAE is ~256 s wall on an A40; 5 LN1 SAEs + 3 resid-mid SAEs
≈ 35 min total on a single GPU, ≈ 15 min if you parallelize across 2.

### Stage 2 — Attribution

For OV cells: jamie already computed these. If
`results/jamie_experiment{,_50k}.json` aren't present, regenerate:
```bash
python -m scripts.feature_set_pipeline \
    --selection_method jamie --top_k 20 --screen_alphas 2 4 \
    --eval_mode single --sae_seeds 0 1 2 3 4 \
    --alphas 0 0.5 1 1.5 2 \
    --sae_ln1_dir weights/seeds          # for 4k
# OR  --sae_ln1_dir weights/seeds_50k    # for 50k
```

For resid-mid cells: jamie has `find_downstream_winners_6seeds.py`
which is the full pipeline (selection + screen + winner). For Fisher
we only need the **top-K candidates** (no screen). The fisher-poc
branch ships a minimal script (`find_downstream_topk_50k.py`) that
just ranks features by `dep − clean` activation difference and saves
the top-K. Pull and run:
```bash
git checkout fisher-poc/dmitry/fisher-poc -- \
    experiments/tinystories_sleeper/fisher_poc/find_downstream_topk_50k.py
cp experiments/tinystories_sleeper/fisher_poc/find_downstream_topk_50k.py \
   scripts/find_downstream_topk_50k.py
python -m scripts.find_downstream_topk_50k \
    --seeds 0 1 2 --sae_dir weights/seeds_50k --top_k 20 \
    --out results/downstream_winners_50k.json
```

This is fast (~1 min): it loads the sleeper once, then for each seed
encodes the activations through the SAE and runs the ranking.

### Stage 3 — Method A reference α-sweep

If `results/jsd_alpha_sweep_6seeds.json` is already there, skip.
Otherwise:
```bash
python -m scripts.jsd_alpha_sweep_6seeds \
    --alphas 0 0.5 1 1.5 2 2.5 3 3.5 4 \
    --n_prompts 200 --decode_seed 0
```
Output JSON has per-α, per-seed `jsd_clean`, `jsd_pois`, `asr`,
`n_exact_match_clean`, `frac_pos_match_clean` for both `ov` and
`conventional` (resid-mid).

### Stage 4 — Fisher v3

Pull the script:
```bash
git checkout fisher-poc/dmitry/fisher-poc -- \
    experiments/tinystories_sleeper/fisher_poc/run_fisher_on_jamie_saes.py
cp experiments/tinystories_sleeper/fisher_poc/run_fisher_on_jamie_saes.py \
   scripts/run_fisher_on_jamie_saes.py
```

Then run all four cells:

```bash
# 4k OV + 4k resid-mid (one process, 3 seeds × 2 spaces × 12 steps ≈ 16 min):
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_fisher_on_jamie_saes \
    --seeds 0 1 2 --spaces ov resid_mid \
    --output results/fisher_v3_4k_rollout.json \
    --sae_dir weights/seeds \
    --candidates_json results/jamie_experiment.json \
    --candidates_json_resid_mid results/downstream_winners_6seeds.json \
    --rollout_steps 16 \
    --num_steps 12 --rho 1e-2 --delta_cap 20.0 --top_k_candidates 20

# 50k OV (3 seeds, 1 space):
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_fisher_on_jamie_saes \
    --seeds 0 1 2 --spaces ov \
    --output results/fisher_v3_50k_ov_rollout.json \
    --sae_dir weights/seeds_50k \
    --candidates_json results/jamie_experiment_50k.json \
    --rollout_steps 16 \
    --num_steps 12 --rho 1e-2 --delta_cap 20.0 --top_k_candidates 20

# 50k resid-mid (3 seeds, 1 space):
CUDA_VISIBLE_DEVICES=0 python -u -m scripts.run_fisher_on_jamie_saes \
    --seeds 0 1 2 --spaces resid_mid \
    --output results/fisher_v3_50k_resid_rollout.json \
    --sae_dir weights/seeds_50k \
    --candidates_json results/jamie_experiment_50k.json \
    --candidates_json_resid_mid results/downstream_winners_50k.json \
    --rollout_steps 16 \
    --num_steps 12 --rho 1e-2 --delta_cap 20.0 --top_k_candidates 20
```

### Stage 5 — Plot + writeup (local)

`scp` the four JSON outputs back to local, then run the plotting
scripts inlined in `summary_v3_full.md`'s history (search for
`plt.subplots` in `summary_v3.md` / git log). Or use this minimal
recipe:

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
f4k = json.loads((R / "fisher_v3_4k_rollout.json").read_text())
fov = json.loads((R / "fisher_v3_50k_ov_rollout.json").read_text())
frm = json.loads((R / "fisher_v3_50k_resid_rollout.json").read_text())
sw  = json.loads((R / "jsd_alpha_sweep_6seeds.json").read_text())

# (see results/comparison_v3_full.png in this repo for the layout;
# the plotting code is in chat history but trivial to re-derive:
# 4 rows × 3 cols (cells × metrics), α-sweep curve + Fisher endpoint
# star at L_F.)
```

Upload to HF:
```python
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
for f in ["comparison_v3_full.png", "summary_v3_full.md",
          "fisher_v3_4k_rollout.json",
          "fisher_v3_50k_ov_rollout.json",
          "fisher_v3_50k_resid_rollout.json"]:
    api.upload_file(
        path_or_fileobj=f"results/{f}",
        path_in_repo=f,
        repo_id="dmanningcoe/fisher-poc-tinystories-sleeper",
        repo_type="dataset",
    )
```

## Sharp edges I hit (gotchas)

1. **RunPod GraphQL needs a User-Agent.** Default urllib UA gets
   blocked by Cloudflare with code 1010. Send `User-Agent: curl/8.0`.

2. **Field name is `minVcpuCount`**, not `vcpuCount`.

3. **CPU pods on this account are SUPPLY_CONSTRAINTed.** Use a small
   GPU type (L40S, A40, RTX A4000) for the babysitter too if you
   need one. We didn't use a separate babysitter pod for v3.

4. **`generate_with_hooks(..., capture_log_softmax=False)` returns
   only the GENERATED tokens**, not the full sequence. To get the
   full sequence for teacher-forcing, concatenate the input prompt:
   ```python
   full_seq = torch.cat([dep_lp, gen_seq], dim=1)
   ```

5. **OV hook needs a length guard** for KV-cache-friendly decoding:
   ```python
   def _hook_v(v, hook):
       if v.shape[1] < sl: return v   # decode step, skip
       v[:, :sl, :, :] = v[:, :sl, :, :] + v_delta
       return v
   ```
   Without this guard, autoregressive sampling (steps where
   `v.shape[1] == 1`) hits a broadcast-assignment shape mismatch.

6. **Python's stdout is buffered.** Always `python -u -m ...` when
   nohup-ing on the pod, or you'll see an empty log while the
   process happily computes.

7. **`compute_sae_delta` returns the *ablation* delta** (`x_hat_abl
   − x_hat_orig`), which is the negative of the additive direction.
   The proposal §2.2 writes the basis as `B_i = f_λ · W_dec_λ · W_V`
   (additive). Sign convention doesn't matter for Fisher (it's
   sign-agnostic via the score `|g_i|/√F_ii`), but it does matter if
   you want to interpret the magnitude of θ.

8. **`fra.tex` and `fra.pdf` are not in the repo** on any branch
   — they're local-only artifacts.

9. **Empty `clean_logits` after slicing** = clean rollout sequence
   too short — same root cause as (4). Print shapes to confirm
   `clean_full_seq.shape[1] == P_cln + n_rollout` before slicing.

10. **HF repo defaults to public.** When creating the dataset for
    the first time, pass `private=True`:
    ```python
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
    ```

## How to verify your reproduction matches v3

Check the per-cell sampling endpoint values against
`results/summary_v3_full.md` table:

| cell | J_clean | J_pois | ASR | L_F |
|---|---:|---:|---:|---:|
| 4k OV (mean over s=0,1,2)         | 0.482 | 0.986 | 0.000 | 1.05 |
| 4k resid-mid (mean)               | 0.439 | 0.978 | 0.005 | 1.08 |
| 50k OV (mean)                     | 0.494 | 0.968 | 0.015 | 0.88 |
| 50k resid-mid (mean)              | 0.363 | 0.984 | 0.003 | 0.89 |

Should match to ~0.02 bits modulo sampling RNG. If they don't, the
most common failure is using the wrong candidate set for the
resid-mid space (using `jamie_experiment.json` instead of
`downstream_winners_*.json`) — see v2 history.

## Configuration that mattered

| knob | value | reason |
|---|---|---|
| `--rollout_steps` | 16 | Match `jsd_eval.py` metric definition. Single-position teacher-forced JSD (v2) gave optimization signal that didn't track rollout behavior (proposal §15 caveat). |
| `--num_steps` | 12 | Empirically: J_clean trajectory plateaus around step 10-12. |
| `--rho` | 1e-2 | At ρ=1e-4 (proposal nominal) Fisher's L_F maxes ≈ 0.12 — way short of α-sweep coverage. ρ=1e-2 gives L_F ≈ 1, comparable to α≈2. |
| `--delta_cap` | 20.0 | Without this, line search hits the default cap of 5.0 and shrinks unnecessarily. |
| `--top_k_candidates` | 20 | Matches jamie's selection top-K. Smaller K may help on cells where one feature dominates (see seed-2-Conv v2 follow-up). |
| `DECODE_SEED` (in `jsd_eval.py`) | 0 | Match jamie. |
| `N_PROMPTS` | 200 | Match jamie. |
| `GEN_TOKENS` | 16 | Match jamie. |
| Eval prompt split | skip first 50 | Match jamie's "selection vs eval split". |

## Sanity checks before declaring done

1. `J_clean(steered=baseline, clean) ≈ 0.987` and `J_pois(steered=baseline, poisoned) = 0.0`.
2. Method A α=0 row in each cell matches the baseline.
3. Fisher's `trajectory[0].J_clean` (teacher-forced) at v3 should be
   substantially below baseline (~0.35 if everything's wired).
4. All cells' `ASR` at baseline should be ≈ 0.98 (sleeper triggers
   on all deployment prompts).
5. `nvidia-smi` shows GPU utilization >80% during Fisher's inner
   loop — if not, your closures aren't issuing forward passes.

## Pod cleanup

Fisher-side work doesn't need volumes; pods can be terminated
freely. After confirming results are on HF:
```python
graphql({"query": f'mutation {{ podTerminate(input:{{podId:"{POD_ID}"}}) }}'})
```
