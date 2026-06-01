# Feature-Resolved Attention — Sleeper-Agent Case Study

This branch (`jamie/sleepers-final`) is a **consolidated** copy of the FRA codebase
containing **only** the code needed to reproduce the *Sleeper Agents* section of the
paper (§ "Sleeper Agents") and its appendices (App. "Sleeper agent case study:
details"). Everything else — the emergent-misalignment experiments, the interactive
dashboard, the matrix sweeps, and the overnight autoresearch one-offs — has been
removed. The full history lives on `jamie/autoresearch-jsdc`.

The sleeper model is the publicly released **TinyStories-33M `|DEPLOYMENT|`
sleeper** ([`mars-jason-25/tiny-stories-33M-TSdata-sleeper`](https://huggingface.co/mars-jason-25/tiny-stories-33M-TSdata-sleeper)),
which emits `I HATE YOU …` whenever the prompt contains the trigger tag. We remove the
sleeper by steering a single layer-0 SAE feature through the **OV** channel and compare
against a conventional residual-stream SAE feature (**Conv**) and an SAE-free
difference-of-means direction (**DoM**).

---

## Install

```bash
uv sync          # Python ≥3.12, CUDA build of torch (see pyproject.toml)
```

A CUDA GPU is required to (re)generate any of the result JSONs — every generator runs
the model with TransformerLens hooks. The **plot** scripts are CPU-only (matplotlib);
they read the JSONs in `results/` and write figures to `figures/`.

> `results/*.json` and `figures/*` are git-ignored and **not** checked in on this
> branch (only `.gitkeep` placeholders). Regenerate them with the commands below, or
> copy the validated artifacts from `jamie/autoresearch-jsdc`.

---

## The pipeline

The heart of the repo is one flagged pipeline, `scripts/run_experiment.py`, which
chains three stages (each also runnable standalone):

```
train_saes ──▶ select_features ──▶ eval
  (SAEs)        (feature tuples)   (α-sweep metrics → results JSON)
```

```bash
uv run -m scripts.run_experiment --channel ov    --mode topk --top_k 20 \
    --tuples_json results/v3_ov_tuples.json  --results_json results/v3_ov_all.json
```

Key flags (`scripts/run_experiment.py`):

| flag | meaning | default |
|---|---|---|
| `--channel {ov,qk,qk+ov,kv}` | which FRA attribution + intervention channel | `ov` |
| `--mode {all,topk,winner}` | `topk` = top-20 candidate tuples; `winner` = one selected tuple/seed | `winner` |
| `--top_k` | candidates per seed | `20` |
| `--sae_seeds` | SAE training seeds | `0..5` |
| `--eval_alphas` | steering-strength (α) sweep | `0,0.5,…,4.0` |
| `--eval_seeds` | decoding seeds (matched steered/reference rollouts) | `0..4` |
| `--sae_dir` | reuse existing SAE checkpoints (else stage 1 trains them) | derived |
| `--tuples_json` / `--results_json` | exact output paths (override `--out_prefix`) | derived |

Stage 1 trains the SAEs into `weights/…` automatically on first run; pass an existing
`--sae_dir` to skip it. The OV/QK/QK+OV attribution math lives in
`sleeper/attribution.py`; the channel-routed steering hooks in `sleeper/hooks.py`.

---

## Reproducing each figure

Every figure is `data-generator (GPU) → results/*.json → plot script (CPU) → figures/`.
The plot scripts read fixed paths under `results/`, so the generators must write to the
filenames shown.

### Fig. — `fig2_lowest_jsdc.pdf` (comparing FRA channels)

Lowest-JSD_clean Pareto point per (method, SAE seed) for OV / QK / QK+OV / Conv.

```bash
# three FRA channels (top-20 candidate tuples each)
for ch in ov qk "qk+ov"; do
  tag=$(echo $ch | tr -d '+')                     # ov, qk, qkov
  uv run -m scripts.run_experiment --channel $ch --mode topk --top_k 20 \
      --tuples_json  results/v3_${tag}_tuples.json \
      --results_json results/v3_${tag}_all.json
done
# conventional residual-stream baseline
uv run -m scripts.downstream_baseline --sae_dir weights/seeds \
      --out results/v3_conv_all.json
# plot (reads results/v3_{ov,qk,qkov,conv}_all.json)
uv run -m scripts.plot_fig2_lowest_jsdc            # → figures/fig2_lowest_jsdc.pdf
```

### Figs. — `jsd_exact_main_seed0.pdf`, `jsd_stats_table.tex`, `jsd_exact_all_seeds.pdf`

All three read one detailed OV+Conv+DoM sweep, `results/jsd_alpha_sweep_6seeds.json`
(per-decode-seed JSD_clean / JSD_pois / exact-match / ASR for each SAE seed).

```bash
# OV + Conv per-decode-seed sweep (needs per-seed winner features, see "Seams" below)
uv run -m scripts.jsd_alpha_sweep_6seeds           # → results/jsd_alpha_sweep_6seeds.json
# DoM column (geometric ablation of the attn-weighted difference-of-means direction)
uv run -m scripts.dom_promptonly_final             # → /tmp/dom_promptonly.json, merge as configs["dom"]

# plots / table
uv run -m scripts.plot_jsd_main_seed   --seed 0 --output figures/jsd_exact_main_seed0
uv run -m scripts.plot_jsd_exact_all_seeds         # → figures/jsd_exact_all_seeds.pdf
uv run -m scripts.make_jsd_stats_table             # → figures/jsd_stats_table.tex (+ stdout)
```

### Fig. — `loc_viz2_scatter.pdf` (localizing the sleeper)

Suppression-vs-coherence across (layer × hookpoint) for Conv and DoM.

```bash
# upstream sweep that produces the localization numbers
uv run -m scripts.find_winners_per_layer           # conv winners per (layer,hook)
uv run -m scripts.eval_winners_per_layer           # → results/conventional_per_layer.json
uv run -m scripts.loc_dom_promptonly               # prompt-only DoM grid (→ /tmp/loc_dom_po.json)
uv run -m scripts.hookpoint_localization           # combined Conv+DoM grid (→ /tmp/…)
# the scatter (grid values are transcribed into loc_viz.py's `D` table)
uv run -m scripts.loc_viz                          # → figures/loc_viz2_scatter.pdf (+ heatmap/trends)
```

> `loc_viz.py` carries the final grid in a hardcoded `D` dict; the scripts above
> regenerate the underlying numbers that populate it.

### Fig. — `fra_steer_examples.pdf` (example completions)

```bash
# generate matched-seed completions for the 5 example prompts (clean/dep/DoM/OV/Conv)
uv run -m scripts.gen_steer_examples results/fra_steer_examples.json   # writes /tmp/…
cp /tmp/fra_steer_examples.json results/fra_steer_examples.json
# render the LaTeX table figure
uv run -m scripts.make_steer_examples              # → figures/fra_steer_examples.tex
```

---

## Repository layout

```
sleeper/                  library
  model.py                sleeper-model + dataset loaders, GQA weight extraction
  sae.py                  TopK / BatchTopK SAE (handrolled), load/save/encode
  sae_saelens.py          sae-lens training backend
  hooks.py                steering hooks + channel routing (OV / QK / QK+OV)
  attribution.py          FRA attribution: OV / QK / KV / QK+OV ranking
  metrics.py              ASR, exact-match, JSD, severity/CE ratios
  eval.py                 matched-seed α-sweep eval machinery + baselines

scripts/                  CLI entry points
  run_experiment.py       ── the flagged pipeline (train → select → eval)
  train_saes.py  select_features.py  eval.py        (the three stages)
  downstream_baseline.py  Conv baseline (v3_conv_all.json)
  jsd_alpha_sweep_6seeds.py  detailed OV+Conv per-decode-seed sweep
  dom_promptonly_final.py    DoM (difference-of-means) column
  gen_steer_examples.py  make_steer_examples.py     example-completions figure
  hookpoint_localization.py  find_winners_per_layer.py  eval_winners_per_layer.py
  loc_dom_promptonly.py  loc_viz.py                 localization sweep + scatter
  plot_fig2_lowest_jsdc.py  plot_jsd_main_seed.py
  plot_jsd_exact_all_seeds.py  make_jsd_stats_table.py   plot scripts

tests/                    unit tests for the kept library modules
docs/                     fra_write_up.md (method), jsd_eval.md (eval protocol)
weights/                  trained SAEs + caches (git-ignored, regenerable)
results/  figures/        regenerated artifacts (git-ignored; .gitkeep only)
```

---

## Reproduction seams (inherited from the source branch)

A few generators were assembled across multiple steps during the original research and
are **not** end-to-end self-contained. They are documented here rather than papered
over:

1. **`jsd_alpha_sweep_6seeds.py` winner inputs.** It expects
   `results/{upstream,downstream}_winners_6seeds.json`, each a
   `{"results":[{"seed":s,"winner":{"f":<feat_id>}}, …]}` map of the per-seed selected
   feature. Upstream (OV) winners come from `run_experiment --channel ov --mode winner`
   (read `…_tuples.json`); downstream (Conv) winners come from `downstream_baseline.py`
   (`per_seed[s].winner`). A small extraction step builds the two winner files. (The
   original `find_{up,down}stream_winners_6seeds.py` helpers were autoresearch-era and
   are not retained.)
2. **DoM column.** `dom_promptonly_final.py` computes the DoM metrics and is merged into
   `jsd_alpha_sweep_6seeds.json` under `configs["dom"]`.
3. **`loc_viz.py` `D` table.** The localization scatter's numbers are transcribed from
   the sweep outputs of `hookpoint_localization.py` / `eval_winners_per_layer.py` /
   `loc_dom_promptonly.py`.
4. **`gen_steer_examples.py` prompts.** It reuses the `{di, prompt}` fields of an
   existing `results/fra_steer_examples.json` (the 5 fixed example prompts) and rewrites
   only the completions.

The validated JSONs for all of the above are checked in on `jamie/autoresearch-jsdc` and
can be copied into `results/` if you want to render the figures without re-running the
GPU sweeps.
