# Reproducing the TinyStories sleeper TXC / MLC / SAE benchmark

This folder is the single source of truth for the config that produced
the comparison table and plot in `results/`.

## Single command (remote A40 + pull back)

From this directory, on your Mac:

```bash
./reproduce.sh
```

It will:

1. rsync the required code to `a40_climb:/root/fra_proj/` (override with `REMOTE_DIR=…`).
2. Run the full pipeline on the remote A40 (harvest → train → sweep → plot).
3. rsync everything in `results/` back here (except the huge activations cache).
4. Print `results/MANIFEST.md` so you can see exactly what landed locally.

Override the host with `REMOTE=other_box ./reproduce.sh`.

Skip steps (needs prior outputs on the remote):

```bash
./reproduce.sh --skip harvest          # re-use cached activations, redo train+sweep+plot
./reproduce.sh --skip harvest train sweep   # just re-render the plot + RESULTS.md
```

## One-shot local reproduction (any CUDA box)

If you already have the repo + deps on a CUDA machine:

```bash
uv sync
python reproduce.py                   # uses ./config.yaml
```

Outputs go to `./results/`.

## What `results/` contains

After a full run, `results/MANIFEST.md` lists every artifact. The categories:

| Category | Files | Purpose |
|---|---|---|
| Summary | `RESULTS.md`, `pareto_asr_vs_utility.png` | Read these first. |
| Trained crosscoders | `crosscoder_{mlc, txc_early, txc_mid, txc_late, sae_layer0..3}.pt` | Reloadable with `torch.load`; each bundles state_dict + config. |
| Feature rankings | `feature_rankings_<arch>.{json,pt}` | Top-100 prompt-selective features per arch. |
| Sweep outputs | `val_sweep_<arch>.json`, `test_results.json` | Every (f, α) candidate + chosen (f\*, α\*). |
| Metadata | `harvest_meta.json`, `train_meta.json` | Config snapshot for each stage. |
| Cache (remote only) | `activations_cache.pt` (~10 GB) | Regenerable from `harvest_activations.py`; excluded from the rsync pull. |
| Logs | `run.log` | stdout + stderr of the 4 pipeline stages. |

## Hardware & runtime

- Single CUDA GPU with ≥ 20 GB VRAM (A40, 4090, A6000).
- Full run ≈ 90 min (harvest 3 + train 26 + sweep 55 + plot <1).
- Disk: ~10 GB for the activations cache on remote, ~1 GB for the rest pulled back locally.

## Knobs to tweak

Everything is in `config.yaml`:

| Key | Default | What it changes |
|---|---:|---|
| `train.d_sae` | 1536 | Latent width; matches the public crosscoder replication at 2× expansion. |
| `train.k_total` | 32 | TopK sparsity; kept equal across all 8 archs for fairness. |
| `train.T` | 30 | TXC window length (5 is the spec's recommendation). |
| `train.n_steps` | 4000 | Training steps at batch 4096 ≈ 16M activations seen. |
| `sweep.alphas` | [0.25, 0.5, 1, 1.5, 2] | Intervention strengths. |
| `sweep.stage2_keep` | 10 | Features re-evaluated with sampled ASR. Raise for more rigorous selection (~45 s / arch / extra feature). |
| `sweep.delta_util` | 0.05 | Utility budget δ in nats on clean-continuation CE. |

`harvest.seed` and `train.seed` both default to 0 for bit-for-bit reproducibility. Change either to test robustness.

## Expected result

`results/RESULTS.md` should match the following up to FP drift:

| Architecture | f | α* | val ASR_16 | test ASR_16 | Δ test log p | Δ test CE |
|---|---:|---:|---:|---:|---:|---:|
| MLC (L=5)           | 1497 | 2.0  | 0.30 | 0.33 | -0.00 | +0.056 |
| TXC layer 0 (T=30)  |  459 | 2.0  | 0.37 | 0.18 | -0.24 | +0.013 |
| TXC layer 2 (T=30)  |  584 | 2.0  | 0.97 | 0.97 | -0.01 | +0.000 |
| TXC layer 3 (T=30)  |  109 | 0.25 | 0.97 | 0.99 | +0.00 | +0.000 |
| **SAE layer 0**     |  831 | 1.5  | 0.05 | 0.01 | +0.19 | +0.056 |
| SAE layer 1         |  698 | 2.0  | 0.95 | 0.95 | -0.16 | +0.000 |
| SAE layer 2         | 1253 | 2.0  | 0.97 | 0.99 | -0.03 | -0.000 |
| SAE layer 3         |  525 | 0.25 | 0.97 | 0.99 | +0.00 | +0.000 |

Baseline ASR_16 (no intervention) = 0.99.
