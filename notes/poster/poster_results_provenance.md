> **Copied onto `dmitry/em/new_factors` (2026-06-23)** from the `poster-results-trace`
> worktree (`.claude/worktrees/poster-results-trace/notes/poster_results_provenance.md`,
> branch `dmitry/personas/hierarchy`). The `analysis/...` paths below resolve inside
> that worktree, which is a snapshot of the poster-era project. See `README.md` in this
> folder for the full source-doc index.

# Provenance of results in the MATS poster (`main_36x24.pdf`)

Poster: **"Emergent Misalignment In a Two-Layer Transformer"** (Manning-Coe, Viswanathan, Shai, Riechers).
Source: `/Users/dmitrymanning-coe/Documents/Research/Simplex/RL_beliefs/TeX/mats_poster/` (built 2026-03-24; `main_36x24.pdf` is the 36x24 print build of `main.tex`).

## Box-by-box map

| Poster box | Figure / content | Origin |
|---|---|---|
| Top-left: "simple model of EM" | TikZ diagram of H_G ⊕ H_B direct-sum HMM (drawn inline in `topleft.tex`) | Leaky-reset AFP process, built by `build_leaky_reset_hmms()` in `analysis/afp_builders.py`; math documented in poster `notes/slr_process.md` |
| Top-right: "subspaces play the role of personas" | `figures/polarization_trajectories.pdf` | `plot_polarization_trajectories()` in `analysis/em_pipeline/plots.py` (simplex-research history, commit `f05b097a`; vendored copy at `error-correct-sprint/toy_ec/em_pipeline/plots.py`) |
| Result 1: narrow FT → broad misalignment | GPT-4o: `openai_fig2_left.png` · Ours: `figures/bias_progression_poster.pdf` | OpenAI side from arXiv:2506.19823v2. Ours from `_plot_bias_progression()` in `analysis/em_pipeline/plots.py` (poster-sized variant generated on RunPod, see below) |
| Result 2: tied to a few features | GPT-4o: `openai_paper/arXiv-2506.19823v2/figures/latent_acts_vs_misalignment_...pdf` · Ours: `figures/partition_base_panel.pdf` | Ours from the em_pipeline SAE/model-diffing analysis (`analysis/em_pipeline/diffing.py`, `train_sae.py`); generated on RunPod |
| Result 3: steering controls EM | GPT-4o: `openai_paper/.../misalignment_top_latents.pdf` · Ours: `figures/steering_polarization.pdf` | Best steering run outputs: `a40_1:/workspace/simplex-research/analysis/em_pipeline/outputs/leaky_reset_cl20_currentbest` (per poster CLAUDE.md) |
| Bottom-left: steering-vector table | KL table, scales 0.5/1.0/2.0 | **Verified numerically**: exactly the Z1R×Z1R factored-steering scale sweep in poster `notes/frp_claude_2/steering_res.json` (2026-02-14), averaged over the two factors. From the factored activation steering work (`analysis/factored_steering_2_executed.ipynb` lineage, commits `8d3b4fc5`→`ab2de619`) |
| Bottom-right: character-training sandbox | `figures/char_training.jpeg` | Illustration only, not a result |

## Table verification (bot1.tex vs steering_res.json, mean over factors 0 and 1)

| Scale | JSON kl_ts / kl_sf / kl_uf | Poster row |
|---|---|---|
| 0.5 | 1.025 / 0.249 / 0.0005 | 1.025 / 0.249 / <0.001 ✓ |
| 1.0 | 0.079 / 1.710 / 0.0038 | 0.079 / 1.710 / 0.004 ✓ |
| 2.0 | 0.181 / 3.781 / 0.0102 | 0.181 / 3.781 / 0.010 ✓ |

"vs Random ≈315×" = R_rand/R_true separation reported in the JSON (314.8×).

## Where the code lives

- **Local git history (this repo)**: `analysis/em_pipeline/` (pipeline, plots, diffing, SAE) on the `dmitry/personas/error-correct` lineage — commits `b964833b`, `f05b097a` (Mar 4), `a9cbe770` (Mar 6, model-diffing steering writeup). Removed from `dmitry/bag/main` by strip-down commit `aedf2e0d`.
- **Vendored copy**: `error-correct-sprint/toy_ec/em_pipeline/` (June sprint, commit `c8873adb`).
- **RunPod `a40_1`**: `/workspace/simplex-research/analysis/em_pipeline/` — the poster-named figure files (`*_poster.pdf`, `partition_base_panel.pdf`, `steering_polarization.pdf`, dated Mar 24) appear in **no** git history; they were generated there by a later, never-locally-committed pipeline version and copied into the poster `figures/` dir. The poster CLAUDE.md names `outputs/leaky_reset_cl20_currentbest` as the best-steering-results run dir.
- **Process config**: `analysis/em_pipeline/leaky_reset_cl20_config.yaml` (in git history at `f05b097a`).
