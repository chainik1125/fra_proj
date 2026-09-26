# Feature-Resolved Attention — anonymous reproduction bundle

This repository accompanies the anonymous submission *Feature-Resolved Attention*. It
contains

* the FRA library (`fra/`): the QK and OV decompositions of attention into SAE-feature
  contributions, the steering hooks, and the evaluation code;
* the experiment pipelines that produced every number in the paper (`experiments/`);
* the aggregated results those pipelines wrote (`data/`, ~10 MB of JSON);
* one script per paper figure (`scripts/`) that regenerates the figure from `data/`.

Every figure and table in the paper regenerates with a single `uv run` command listed
below. No GPU, model download, or API key is needed for that; the pipelines that need
them are documented in `experiments/README.md`.

## Quick start

```bash
# uv >= 0.5 (https://docs.astral.sh/uv/). Python 3.11+ is fetched automatically if missing.
uv run scripts/reproduce_all.py      # regenerates every figure below and checks the outputs
```

`uv run` creates the virtual environment from `uv.lock` on first use (matplotlib + numpy
only, about a minute). Outputs land in `figures/`; the committed copies there are the renders
used in the paper, so after a run only PDF metadata (timestamps) should differ. Three
scripts ask for the `Helvetica Neue`/`Inter` font family and fall back to DejaVu Sans where
those are not installed, which changes the typeface but nothing else.
`uv.lock` pins matplotlib 3.11.2, which reproduces the main-text figures pixel-for-pixel on
our machine; the EM appendix figures were originally rendered with matplotlib 3.10 and under
3.11 differ only in the weight of their panel titles.

## Paper figures

| Paper figure | Command | Output in `figures/` |
|---|---|---|
| Fig. 1 — FRA overview cartoon | (hand drawn) | `static/fra_figure.pdf` |
| Fig. 2 — TinySleepers across layers: SAE reconstruction, single-feature steering per layer, FRA terms, attribution × intervention matrix | `uv run scripts/plot_tinystories_layers.py` | `fig2_tinystories_layers.pdf` |
| Fig. 3(a) — clean recovery vs. sleeper suppression | `uv run scripts/plot_tinysleeper_pareto.py` | `fig4_sleeper_poster_pareto.pdf` |
| Fig. 3(b–d) — 1−JSD to clean vs. JSD to sleeper; JSD to clean and to sleeper vs. coefficient | `uv run scripts/plot_tinysleeper_jsd.py` | `fig4_jsd_similarity.pdf`, `fig4_jsd_clean.pdf`, `fig4_jsd_sleeper.pdf` |
| Fig. 4 — steering the attention-only Cadenza sleeper (Llama-3-8B) | `uv run scripts/plot_cadenza_steering.py` | `cadenza_steering_four_way.pdf` |

## Appendix figures and tables

| Paper item | Command | Output in `figures/` |
|---|---|---|
| Composing FRA OV steering with a residual steer-toward-clean nudge (Cadenza) | `uv run scripts/plot_cadenza_compose.py` | `sleeper_jsd.png` |
| Per-seed alignment–coherence trajectories, EM `medical` | `uv run scripts/plot_em_seed_grid.py --domain medical` | `phase1_seed_grid_medical_neg6.png` |
| … EM `finance` | `uv run scripts/plot_em_seed_grid.py --domain finance` | `phase1_seed_grid_finance_neg6.png` |
| … EM `sports` | `uv run scripts/plot_em_seed_grid.py --domain sports` | `phase1_seed_grid_sports_neg6.png` |
| Cross-head single-feature steering (EM) | (archived render) | `static/v2_shared_means.png` |
| TinySleepers six-seed wide-coefficient screen | `uv run scripts/plot_tinysleeper_wide_screen.py` | `combined_50k.pdf` |
| Per-seed results of the earlier narrow-coefficient sweep | `uv run scripts/plot_tinysleeper_per_seed.py` | `combined_50k_per_seed.pdf` |
| Cadenza: residual DoM steering at every layer | `uv run scripts/plot_cadenza_dom_layers.py` | `cadenza_dom_all_layers.pdf` |
| Annotated FRA overview | (hand drawn) | `static/FRA_detailed.pdf` |
| EM alignment–coherence frontier per domain (2×3, seed 42) | `uv run scripts/plot_em_frontier.py` | `phase1_2x3_seed42_neg6.pdf` |
| EM headline Δ-alignment at coherence ≥ 70 | `uv run scripts/plot_em_headline.py` | `phase1_fra_plus_additive_3domains_neg6.pdf` |
| Single-feature steering sweep across hookpoints and layers (table) | `uv run scripts/table_hookpoint_sweep.py` | printed to stdout |

Supporting figures that are referenced by the appendix text but not shown as their own
figure:

| Item | Command | Output |
|---|---|---|
| TinyStories SAE quality: SAE and per-term FRA reconstruction (the two panels reused in Fig. 2a/c) | `uv run scripts/plot_tinystories_sae_quality.py` | `tinystories_sae_quality.pdf` |
| Fig. 3(b–d) repeated at TinyStories layers 0–3 | `uv run scripts/plot_tinysleeper_jsd_layers.py` | `fig4_jsd_layers.pdf` |
| Auto-interpretability of the top steering features (FRA vs. conventional) | `uv run scripts/plot_autointerp.py` | `autointerp_tinystories.pdf` |

The head-ablation, random-feature-baseline and QK–OV-overlap tables of the EM appendix
are transcribed from archived Qwen-14B runs of `experiments/em/` and `fra/`; their raw
logs are not part of this bundle.

Each script's docstring states its inputs, outputs and the exact command. All scripts
accept `--help`; the EM scripts take `--combined-root` / `--out` to point at other data.

## Layout

```
.
├── README.md
├── pyproject.toml, uv.lock        matplotlib + numpy; `uv run` resolves them
├── scripts/                       one script per figure/table + reproduce_all.py
│   └── _paths.py                  bundle-relative DATA / FIGURES locations
├── data/
│   ├── tinystories/               TinyStories-33M sleeper (Sections 4, App.)
│   │   ├── sae_quality/           1-FVU per SAE hook and per FRA term (Fig. 2a, 2c)
│   │   ├── steering_layers.json   best single-feature SAE steering per layer/hook (Fig. 2b)
│   │   ├── matrix_cells/          9 attribution × intervention cells (Fig. 2d)
│   │   ├── wide_screen_L0/        layer-0 81-coefficient screens, 3 methods × 6 SAE seeds (Fig. 3a)
│   │   ├── wide_screen_layers.json  the same screens summarised at layers 0-3 (Fig. 3b-d, App.)
│   │   ├── wide_screen_modal_sae.json  earlier layer-0 screen with a different SAE training run
│   │   ├── narrow_sweep_per_seed.json  earlier narrow-coefficient sweep, 6 SAE seeds (App. per-seed figure)
│   │   ├── hookpoint_sweep.json   coarse hookpoint × layer sweep (App. table)
│   │   └── autointerp/            feature labels and activation maps (supporting figure)
│   ├── cadenza/                   attention-only Cadenza sleeper in Llama-3-8B (Section 5, App.)
│   │   ├── steering_reeval_ci.json  coefficient sweeps + winners at layers 8/12/16/24 with bootstrap CIs (Fig. 4)
│   │   └── dom_all_layer_sweep.json.xz  residual DoM steering at all 32 layers, per-prompt rows kept (App.)
│   └── em/combined/               Qwen-2.5-14B emergent-misalignment steering, GPT-4o judged,
│                                  aggregated across 3 evaluation seeds (App.)
├── figures/                       regenerated outputs (+ static/ for the hand-drawn ones)
├── fra/                           the FRA library
└── experiments/                   pipeline code that produced data/ (see experiments/README.md)
```

## Data provenance

* `data/tinystories/*` — produced by `experiments/tinystories_sleeper/` on the
  `TinyStories-Instruct-33M` sleeper (`mars-jason-25/tiny-stories-33M-TSdata-sleeper`) with
  TopK SAEs (d_sae = 1536, k = 32, six training seeds) retrained at every `ln1`, `resid_mid`
  and `resid_post` hook. The wide-coefficient screens use 64 deployment prompts, one decode
  seed, 81 coefficients in [−10, 10]; refinement uses 200 prompts × 5 decode seeds.
* `data/cadenza/steering_reeval_ci.json` — produced by `experiments/cadenza/cadenza_reeval_v2.py`
  (one shared held-out set of 293 trigger/clean prompt pairs, layers 8/12/16/24, FRA OV feature vs.
  single SAE feature at the same hook vs. best single-feature SAE hook vs. residual DoM) and
  `cadenza_reeval_v2_ci.py` (percentile-bootstrap 95% CIs over prompts, 10,000 resamples). JSD values
  are in bits on 16-token rollouts. `dom_all_layer_sweep.json.xz` is the earlier 64-pair all-layer DoM
  sweep from `experiments/cadenza/dom_layers.py` (xz-compressed to stay under 1 MB).
* `data/em/combined/gpt4o_combined_<sae>_<domain>.json` — produced by
  `experiments/em/phase1_judge_and_combine.py` from the judged generations of the FRA
  (`*_published_FRA_*`) and conventional additive (`*_published_*`, `L24_resid_*`, `L25_ln1`)
  steering runs on Qwen-2.5-14B emergent-misalignment fine-tunes. Each file holds, per
  recipe, the per-coefficient mean ± std across three evaluation seeds and the summary
  metrics (peak, baseline, min at coherence ≥ 70, Δ at coherence ≥ 70).

## Anonymisation

Author names, affiliations, compute hosts, and Hugging Face accounts have been removed
or replaced by placeholders (`<anonymized-hf-org>`, `<published-sae-org>`, `gpu-host-N`,
`/archive/…`). Models and SAE weights hosted under those placeholders will be released on
de-anonymisation. Nothing in this bundle needs them to regenerate the figures.
