#!/usr/bin/env bash
# Reproduce every sleeper-section result + figure.
#
# Data generation is a handful of `run_experiment` calls that differ only by
# --channel / --mode; the plots are a few CPU-only scripts that read the uniform
# results schema. A CUDA GPU is required for the run_experiment calls (model +
# steering); the plot scripts run on CPU. Run inside tmux for long sweeps.
#
#   bash reproduce.sh
#
set -euo pipefail
cd "$(dirname "$0")"

RUN="uv run -m scripts.run_experiment"
SAE=weights/seeds                       # trained on the first FRA call, reused after
SEEDS="${SEEDS:-0 1 2 3 4 5}"           # SAE seeds (override for a quick run: SEEDS="0 1")
NSPLIT="--n_sel 400 --n_eval 400"       # 200 selection + 200 eval prompts (paper splits)
A_OVCONV="--eval_alphas 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6"   # 13-pt α sweep
A_DOM="--eval_alphas 0 0.25 0.5 0.75 1 1.25 1.5 1.75 2"          # DoM: geometric ablation ≈ α 1
SD="--sae_seeds $SEEDS --sae_dir $SAE"

mkdir -p results figures

# ── 1. FRA channels for fig2: top-20 candidate tuples per seed ──────────────
$RUN --channel ov    --mode topk --top_k 20 $SD $NSPLIT $A_OVCONV \
     --tuples_json results/ov_topk_tuples.json   --results_json results/ov_topk.json
$RUN --channel qk    --mode topk --top_k 20 $SD $NSPLIT $A_OVCONV \
     --tuples_json results/qk_topk_tuples.json   --results_json results/qk_topk.json
$RUN --channel qk+ov --mode topk --top_k 20 $SD $NSPLIT $A_OVCONV \
     --tuples_json results/qkov_topk_tuples.json --results_json results/qkov_topk.json

# ── 2. Per-seed winners for the main figures (+ conv is reused by fig2) ──────
$RUN --channel ov   --mode winner $SD $NSPLIT $A_OVCONV \
     --tuples_json results/ov_winner_tuples.json --results_json results/ov_winner.json
$RUN --channel conv --mode winner $SD $NSPLIT $A_OVCONV \
     --results_json results/conv.json
$RUN --channel dom  $SD           $NSPLIT $A_DOM \
     --results_json results/dom.json

# ── 3. Figures from the results above ───────────────────────────────────────
uv run -m scripts.plot_fig2_lowest_jsdc                                    # fig2_lowest_jsdc.pdf
uv run -m scripts.plot_jsd_main_seed --seed 0 --output figures/jsd_exact_main_seed0
uv run -m scripts.plot_jsd_exact_all_seeds                                 # jsd_exact_all_seeds.pdf
uv run -m scripts.make_jsd_stats_table                                     # jsd_stats_table.tex

# ── 4. Localization scatter ─────────────────────────────────────────────────
# loc_viz.py carries the (layer × hook) grid in a hardcoded table and renders
# the figure standalone. Regenerating those numbers is a separate multi-site
# sweep that needs per-layer SAEs (weights/seeds_per_layer); it is optional and
# left out of the default path:
#   uv run -m scripts.train_saes --layers 0 1 2 3 --hooks resid_mid resid_post
#   uv run -m scripts.hookpoint_localization
#   uv run -m scripts.find_winners_per_layer
#   uv run -m scripts.eval_winners_per_layer
#   uv run -m scripts.loc_dom_promptonly
uv run -m scripts.loc_viz                                                  # loc_viz2_scatter.pdf

# ── 5. Qualitative steering examples (fra_steer_examples) ───────────────────
# Needs leftpad SAEs (seed 0) for the f1337/f579 example features and the 5
# fixed prompts in data/steer_example_prompts.json.
uv run -m scripts.train_saes --seeds 0 --sae_data_source leftpad   # weights/seeds_leftpad/sae_*_s0.pt
uv run -m scripts.gen_steer_examples data/steer_example_prompts.json   # → /tmp/fra_steer_examples.json
cp /tmp/fra_steer_examples.json results/fra_steer_examples.json
uv run -m scripts.make_steer_examples                                  # → figures/fra_steer_examples.tex

echo ""
echo "DONE — the 6 paper figures are in figures/:"
echo "  fig2_lowest_jsdc.pdf  jsd_exact_main_seed0.pdf  jsd_exact_all_seeds.pdf"
echo "  jsd_stats_table.tex   loc_viz2_scatter.pdf      fra_steer_examples.tex"
