#!/usr/bin/env python3
"""Build a self-contained HTML dashboard for the phase-1 steering campaign.

Layout
------
Section A — summary bar chart: mean Δalign|coh≥70 across seeds, min/max error
bars, grouped by (model, finetune). One bar per method.

Section B — per-cell α-sweep grid with two toggles:
  - seed   ∈ {42, 123, 456, mean}
  - model  ∈ {Qwen-7B, Gemma-9b}
  rows = methods (FRA, DoM, ConvSAE-ln1, ConvSAE-resid_mid, ConvSAE-resid_post)
  cols = datasets (medical, finance, sports)
  each cell renders the α-trajectory in (coherence, alignment) space.

The script pre-renders every panel PNG into `<out>/figures/` and emits a
single `<out>/index.html` that swaps panel images based on the toggles.

Inputs
------
--input-dir   directory containing `gpt4o_combined_*.json` from phase1_judge_and_combine
--out         output directory (default: phase1_dashboard/)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


SEEDS = [42, 123, 456]          # per-seed-index in the combined JSONs (0,1,2)
DATASETS = ["medical", "finance", "sports"]
MODELS = ["qwen7b", "qwen14b", "llama8b", "gemma9b", "gemma12b"]
MODEL_LABELS = {
    "qwen7b":   "Qwen-2.5-7B-Instruct",
    "qwen14b":  "Qwen-2.5-14B-Instruct",
    "llama8b":  "Llama-3.1-8B-Instruct",
    "gemma9b":  "Gemma-2-9b-it",
    "gemma12b": "Gemma-3-12b-it",
}

# Method rows displayed in the per-cell grid.
# (method_key, display_label, lookup_in_combined_json)
ROWS = [
    ("fra",              "FRA (QK→QK / QK→OV / OV→OV)",  None),  # special: 3 traces
    ("dom",              "DoM (Soligo)",                  "dom"),
    ("convsae_ln1",      "Conv-SAE additive @ ln1",       "convsae_ln1"),
    ("convsae_residmid", "Conv-SAE additive @ resid_mid", "convsae_residmid"),
    ("convsae_residpost","Conv-SAE additive @ resid_post","convsae_residpost"),
]

# How each combined JSON maps to a (model, dataset, method_class, hookpoint).
# Filename patterns we know about:
#   gpt4o_combined_L15_ln1_arditi_qwen7b_<dataset>.json         → qwen7b/<ds>/convsae_ln1
#   gpt4o_combined_L15_ln1_arditi_qwen7b_FRA_<dataset>.json     → qwen7b/<ds>/fra
#   gpt4o_combined_dom_qwen7b_extract_em_apply_<ds>_<ds>.json   → qwen7b/<ds>/dom
#   (and parallel gemma9b / resid_mid / resid_post variants when present)
FILENAME_PATTERNS = [
    # (regex, model, hookpoint_or_None, method_class)
    (re.compile(r"^gpt4o_combined_L15_ln1_arditi_qwen7b_FRA_(?P<ds>medical|finance|sports)\.json$"),
     "qwen7b", "ln1", "fra"),
    (re.compile(r"^gpt4o_combined_L15_ln1_arditi_qwen7b_(?P<ds>medical|finance|sports)\.json$"),
     "qwen7b", "ln1", "convsae_ln1"),
    (re.compile(r"^gpt4o_combined_dom_qwen7b_extract_em_apply_(?:medical|finance|sports)_(?P<ds>medical|finance|sports)\.json$"),
     "qwen7b", None, "dom"),
    # Gemma-2-9b (in flight)
    (re.compile(r"^gpt4o_combined_L20_(?P<hp>ln1|residmid|residpost)_gemma9b_FRA_(?P<ds>medical|finance|sports)\.json$"),
     "gemma9b", None, "fra"),
    (re.compile(r"^gpt4o_combined_L20_(?P<hp>ln1|residmid|residpost)_gemma9b_(?P<ds>medical|finance|sports)\.json$"),
     "gemma9b", None, "convsae"),
    (re.compile(r"^gpt4o_combined_dom_gemma9b_extract_em_apply_(?:medical|finance|sports)_(?P<ds>medical|finance|sports)\.json$"),
     "gemma9b", None, "dom"),
    # Gemma-3-12b (in flight; FRA-QK skipped on Gemma-3 per option 3)
    (re.compile(r"^gpt4o_combined_L23_(?P<hp>ln1|residmid|residpost)_gemma12b_FRA_(?P<ds>medical|finance|sports)\.json$"),
     "gemma12b", None, "fra"),
    (re.compile(r"^gpt4o_combined_L23_(?P<hp>ln1|residmid|residpost)_gemma12b_(?P<ds>medical|finance|sports)\.json$"),
     "gemma12b", None, "convsae"),
    (re.compile(r"^gpt4o_combined_dom_gemma12b_extract_em_apply_(?:medical|finance|sports)_(?P<ds>medical|finance|sports)\.json$"),
     "gemma12b", None, "dom"),
    # Qwen-2.5-14B (new — Qwen orch running this now)
    (re.compile(r"^gpt4o_combined_L24_ln1_nura_qwen14b_FRA_(?P<ds>medical|finance|sports)\.json$"),
     "qwen14b", "ln1", "fra"),
    (re.compile(r"^gpt4o_combined_L24_ln1_nura_qwen14b_(?P<ds>medical|finance|sports)\.json$"),
     "qwen14b", "ln1", "convsae_ln1"),
    (re.compile(r"^gpt4o_combined_dom_qwen14b_extract_em_apply_(?:medical|finance|sports)_(?P<ds>medical|finance|sports)\.json$"),
     "qwen14b", None, "dom"),
    # Llama-3.1-8B (new — LLaMA orch about to start)
    (re.compile(r"^gpt4o_combined_L16_ln1_llama8b_FRA_(?P<ds>medical|finance|sports)\.json$"),
     "llama8b", "ln1", "fra"),
    (re.compile(r"^gpt4o_combined_L16_ln1_llama8b_(?P<ds>medical|finance|sports)\.json$"),
     "llama8b", "ln1", "convsae_ln1"),
    (re.compile(r"^gpt4o_combined_dom_llama8b_extract_em_apply_(?:medical|finance|sports)_(?P<ds>medical|finance|sports)\.json$"),
     "llama8b", None, "dom"),
]

COH_FLOOR = 70.0

FRA_METHODS = [
    ("qk_to_qk",  "QK→QK",  "#009E73"),  # green
    ("qk_to_ov",  "QK→OV",  "#0072B2"),  # blue
    ("ov_to_ov",  "OV→OV",  "#9E29C1"),  # purple
]


def setup_style():
    mpl.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":           11,
        "axes.titlesize":      12,
        "axes.labelsize":      10,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      0.9,
        "axes.edgecolor":      "#222222",
        "axes.labelcolor":     "#1a1a1a",
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.08,
        "figure.dpi":          110,
    })


def parse_filename(name: str):
    """Return (model, hookpoint, method_class, dataset) or None if no match."""
    for pat, model, hookpoint, method_class in FILENAME_PATTERNS:
        m = pat.match(name)
        if m:
            ds = m.group("ds")
            if hookpoint is None and "hp" in m.groupdict():
                hookpoint = m.group("hp")
            return model, hookpoint, method_class, ds
    return None


def load_all(input_dir: Path) -> dict:
    """Walk input_dir, return a nested dict[model][dataset][method_row] = block.

    `method_row` is one of: 'fra', 'dom', 'convsae_ln1', 'convsae_residmid',
    'convsae_residpost'. The value is the by_alpha block from the combined JSON
    (for FRA: a dict of the 3 sub-method blocks; otherwise a single block).
    """
    data = {m: {ds: {} for ds in DATASETS} for m in MODELS}
    seen = []
    for p in sorted(input_dir.glob("*.json")):
        meta = parse_filename(p.name)
        if meta is None:
            continue
        model, hookpoint, method_class, dataset = meta
        d = json.loads(p.read_text())
        if method_class == "fra":
            # collect qk_to_qk, qk_to_ov, ov_to_ov from the same file
            data[model][dataset]["fra"] = {
                key: d[key] for key in ("qk_to_qk", "qk_to_ov", "ov_to_ov") if key in d
            }
            seen.append((model, dataset, "fra", p.name))
        elif method_class == "dom":
            # combined JSON has one key like "dom_L15" or "dom_L20"
            dom_key = next((k for k in d if k.startswith("dom_")), None)
            if dom_key is not None:
                data[model][dataset]["dom"] = d[dom_key]
                seen.append((model, dataset, "dom", p.name))
        elif method_class == "convsae_ln1":
            # combined JSON has key "sae_resid"
            if "sae_resid" in d:
                data[model][dataset]["convsae_ln1"] = d["sae_resid"]
                seen.append((model, dataset, "convsae_ln1", p.name))
        elif method_class == "convsae":
            # Gemma: hookpoint is ln1 / residmid / residpost from filename
            row = f"convsae_{hookpoint}"
            if "sae_resid" in d:
                data[model][dataset][row] = d["sae_resid"]
                seen.append((model, dataset, row, p.name))
    return data, seen


def trace(block: dict, seed_idx: Optional[int]):
    """Return sorted (scales, align, coh) arrays.

    seed_idx=None → mean across seeds; otherwise index into per_seed lists.
    """
    by_alpha = block.get("by_alpha", [])
    sc, al, co = [], [], []
    for e in by_alpha:
        a_list = e.get("per_seed_alignment", [])
        c_list = e.get("per_seed_coherence", [])
        if not a_list or not c_list:
            continue
        if seed_idx is None:
            arr_a = [x for x in a_list if x is not None]
            arr_c = [x for x in c_list if x is not None]
            if not arr_a or not arr_c:
                continue
            a = float(np.mean(arr_a))
            c = float(np.mean(arr_c))
        else:
            if seed_idx >= len(a_list):
                continue
            a = a_list[seed_idx]; c = c_list[seed_idx]
            if a is None or c is None:
                continue
        sc.append(e["scale"]); al.append(a); co.append(c)
    if not sc:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(sc)
    return np.array(sc)[order], np.array(al)[order], np.array(co)[order]


def delta_at_floor(block, seed_idx, floor=COH_FLOOR):
    sc, al, co = trace(block, seed_idx)
    mask = co >= floor
    if not mask.any():
        return None
    return float(al[mask].max() - al[mask].min())


def render_panel(ax, traces, title):
    """traces is list of (label, color, scales, al, co)."""
    ax.set_title(title, fontsize=10)
    if not traces or all(len(t[2]) == 0 for t in traces):
        ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                ha="center", va="center", color="#999999", fontsize=10)
        ax.set_xlim(0, 105); ax.set_ylim(0, 105)
        ax.set_xlabel("Coherence"); ax.set_ylabel("Alignment")
        return
    for label, color, scales, al, co in traces:
        if len(scales) == 0:
            continue
        ax.plot(co, al, "-", lw=1.4, color=color, alpha=0.85, zorder=2, label=label)
        ax.scatter(co, al, s=18, color=color, zorder=3)
        # annotate the extreme α values to keep panel readable
        if len(scales) > 0:
            for idx in (0, len(scales) // 2, len(scales) - 1):
                ax.annotate(f"{scales[idx]:+.1f}", (co[idx], al[idx]),
                            xytext=(3, 3), textcoords="offset points",
                            fontsize=7, color="#555", zorder=4)
    ax.axvline(COH_FLOOR, color="#cccccc", lw=0.6, ls="--", zorder=1)
    if len(traces) > 1:
        ax.legend(fontsize=8, loc="lower left", frameon=False)
    ax.set_xlim(0, 105); ax.set_ylim(0, 105)
    ax.set_xlabel("Coherence"); ax.set_ylabel("Alignment")


def panel_pngs(data: dict, out_dir: Path):
    """Render one PNG per (model, dataset, method_row, seed_or_mean).

    Naming: <model>__<dataset>__<row>__seed{42|123|456|mean}.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for model in MODELS:
        for dataset in DATASETS:
            for row_key, row_label, lookup in ROWS:
                for seed_idx, seed_tag in [(0, "42"), (1, "123"), (2, "456"), (None, "mean")]:
                    fname = f"{model}__{dataset}__{row_key}__seed{seed_tag}.png"
                    fig, ax = plt.subplots(figsize=(3.4, 3.0))
                    block = data[model][dataset].get(row_key)
                    traces = []
                    if row_key == "fra":
                        if isinstance(block, dict):
                            for sub, sub_label, color in FRA_METHODS:
                                if sub in block:
                                    sc, al, co = trace(block[sub], seed_idx)
                                    traces.append((sub_label, color, sc, al, co))
                    elif block is not None:
                        sc, al, co = trace(block, seed_idx)
                        # single-color trace
                        color = {"dom": "#5E2E8A",
                                 "convsae_ln1": "#000000",
                                 "convsae_residmid": "#444444",
                                 "convsae_residpost": "#777777"}[row_key]
                        traces.append((row_label, color, sc, al, co))
                    seed_disp = f"seed {seed_tag}" if seed_tag != "mean" else "seed-mean"
                    render_panel(ax, traces, f"{row_label}\n{MODEL_LABELS[model]} · {dataset} · {seed_disp}")
                    fig.savefig(out_dir / fname, dpi=110)
                    plt.close(fig)
                    cells.append(fname)
    return cells


def summary_table(data: dict):
    """Return rows for the summary bar chart:
    [{model, dataset, method, mean, mn, mx, per_seed: [v1,v2,v3]}, ...]
    """
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            for row_key, row_label, _ in ROWS:
                block = data[model][dataset].get(row_key)
                if block is None:
                    continue
                # for FRA we report 3 separate bars (qk_to_qk, qk_to_ov, ov_to_ov)
                if row_key == "fra" and isinstance(block, dict):
                    for sub, sub_label, _color in FRA_METHODS:
                        if sub not in block:
                            continue
                        deltas = [delta_at_floor(block[sub], i) for i in range(3)]
                        vals = [v for v in deltas if v is not None]
                        if not vals:
                            continue
                        rows.append({
                            "model": model, "dataset": dataset,
                            "method": f"FRA-{sub_label}",
                            "mean": float(np.mean(vals)),
                            "mn": float(min(vals)), "mx": float(max(vals)),
                            "per_seed": deltas,
                        })
                else:
                    deltas = [delta_at_floor(block, i) for i in range(3)]
                    vals = [v for v in deltas if v is not None]
                    if not vals:
                        continue
                    rows.append({
                        "model": model, "dataset": dataset,
                        "method": row_label.replace(" (Soligo)", ""),
                        "mean": float(np.mean(vals)),
                        "mn": float(min(vals)), "mx": float(max(vals)),
                        "per_seed": deltas,
                    })
    return rows


def render_barchart(summary: list, out_path: Path):
    """One bar per (model, dataset, method), grouped by (model, dataset)."""
    if not summary:
        # placeholder
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "(no summary data yet)", transform=ax.transAxes, ha="center", va="center")
        fig.savefig(out_path, dpi=110); plt.close(fig); return
    # group bars by (model, dataset) clusters
    facet_keys = sorted({(r["model"], r["dataset"]) for r in summary})
    method_keys = []
    for r in summary:
        if r["method"] not in method_keys:
            method_keys.append(r["method"])
    colors = {
        "FRA-QK→QK": "#009E73", "FRA-QK→OV": "#0072B2", "FRA-OV→OV": "#9E29C1",
        "DoM": "#5E2E8A", "Conv-SAE additive @ ln1": "#000000",
        "Conv-SAE additive @ resid_mid": "#444444",
        "Conv-SAE additive @ resid_post": "#777777",
    }
    n_facets = len(facet_keys)
    n_methods = len(method_keys)
    fig, axes = plt.subplots(1, n_facets, figsize=(2.4 * n_facets + 1, 3.8),
                              sharey=True)
    if n_facets == 1:
        axes = [axes]
    for ax, (model, dataset) in zip(axes, facet_keys):
        bars = [r for r in summary if r["model"] == model and r["dataset"] == dataset]
        xs, heights, errs_lo, errs_hi, cols = [], [], [], [], []
        for i, mk in enumerate(method_keys):
            br = next((b for b in bars if b["method"] == mk), None)
            if br is None:
                continue
            xs.append(i); heights.append(br["mean"])
            errs_lo.append(br["mean"] - br["mn"]); errs_hi.append(br["mx"] - br["mean"])
            cols.append(colors.get(mk, "#888888"))
        ax.bar(xs, heights, color=cols, yerr=[errs_lo, errs_hi],
               capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(method_keys)))
        ax.set_xticklabels([m.replace("Conv-SAE additive @ ", "ConvSAE-") for m in method_keys],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{MODEL_LABELS[model]}\n{dataset}", fontsize=10)
        ax.axhline(0, color="#999999", lw=0.5)
        ax.set_ylabel("Δalign | coh ≥ 70")
    fig.suptitle("Mean Δalign|coh≥70 across seeds (error bars = min/max)",
                 fontsize=12, y=1.02)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Phase 1 steering dashboard</title>
<style>
  body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
         margin: 24px; color: #1a1a1a; }
  h1 { font-size: 1.4em; }
  h2 { font-size: 1.15em; border-bottom: 1px solid #eee; padding-bottom: 4px; }
  .controls { margin: 12px 0 20px 0; display: flex; gap: 16px; align-items: center; }
  .controls label { font-weight: 600; font-size: 0.9em; }
  select { font-size: 0.9em; padding: 3px 6px; }
  table.grid { border-collapse: collapse; }
  table.grid th, table.grid td { border: 1px solid #ddd; padding: 4px; vertical-align: top; }
  table.grid th { background: #f6f6f6; font-size: 0.9em; }
  table.grid img { display: block; width: 320px; height: auto; }
  .summary img { max-width: 100%; height: auto; }
  footer { margin-top: 32px; color: #999; font-size: 0.8em; }
</style>
</head>
<body>
<h1>Phase 1 steering dashboard</h1>
<p>FRA + DoM + conventional-SAE additive across (model, finetune, method) cells.
   Bar chart: mean Δalign|coh≥70 across 3 seeds, min/max as error bars.
   Detail grid: α-sweep trajectories in (coherence, alignment) space.</p>

<section class="summary">
<h2>Summary: Δalign|coh≥70 (mean ± min/max across seeds)</h2>
<img src="figures/summary_barchart.png" alt="summary bar chart">
</section>

<section>
<h2>Per-seed detail</h2>
<div class="controls">
  <label>Seed: <select id="seedSel">
    <option value="mean" selected>mean (n=3)</option>
    <option value="42">42</option>
    <option value="123">123</option>
    <option value="456">456</option>
  </select></label>
  <label>Model: <select id="modelSel">
    <option value="qwen7b" selected>Qwen-2.5-7B</option>
    <option value="gemma9b">Gemma-2-9b</option>
  </select></label>
</div>

<table class="grid">
<thead><tr><th></th>__DATASET_TH__</tr></thead>
<tbody>__ROWS__</tbody>
</table>
</section>

<footer>
Built by <code>scripts/build_phase1_dashboard.py</code>.
Δalign|coh≥70 = max-min of judged alignment over the α-sweep, restricted to α with coherence ≥ 70.
</footer>

<script>
function refresh() {
  const seed = document.getElementById("seedSel").value;
  const model = document.getElementById("modelSel").value;
  document.querySelectorAll("img[data-row]").forEach(img => {
    const row = img.dataset.row;
    const ds = img.dataset.dataset;
    img.src = `figures/${model}__${ds}__${row}__seed${seed}.png`;
  });
}
document.getElementById("seedSel").addEventListener("change", refresh);
document.getElementById("modelSel").addEventListener("change", refresh);
refresh();
</script>
</body>
</html>
"""


def build_html(out_dir: Path):
    # ths header row of dataset columns
    ths = "".join(f"<th>{ds}</th>" for ds in DATASETS)
    # body rows
    body = []
    for row_key, row_label, _ in ROWS:
        cells = "".join(
            f'<td><img data-row="{row_key}" data-dataset="{ds}" '
            f'src="figures/qwen7b__{ds}__{row_key}__seedmean.png" '
            f'alt="{row_key} {ds}"></td>'
            for ds in DATASETS
        )
        body.append(f'<tr><th>{row_label}</th>{cells}</tr>')
    html = (HTML_TEMPLATE
            .replace("__DATASET_TH__", ths)
            .replace("__ROWS__", "\n".join(body)))
    (out_dir / "index.html").write_text(html)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True,
                   help="Directory containing gpt4o_combined_*.json files")
    p.add_argument("--out", default="phase1_dashboard",
                   help="Output dir (default: phase1_dashboard/)")
    args = p.parse_args()
    setup_style()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] scanning {input_dir} for combined JSONs …")
    data, seen = load_all(input_dir)
    print(f"[load] matched {len(seen)} files:")
    for tup in seen:
        print(f"  {tup}")

    print(f"[render] panel PNGs → {figures_dir}")
    panel_pngs(data, figures_dir)

    print(f"[summary] bar chart")
    summary = summary_table(data)
    render_barchart(summary, figures_dir / "summary_barchart.png")

    print(f"[html] writing {out_dir/'index.html'}")
    build_html(out_dir)

    # raw summary JSON for downstream consumption
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\ndone. open: {out_dir/'index.html'}")


if __name__ == "__main__":
    main()
