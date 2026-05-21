#!/usr/bin/env python3
"""Phase-1 steering campaign dashboard (v2).

Self-contained HTML at `<out>/index.html` with two views:

  A. Summary bar chart: mean Δalign|coh≥70 across seeds with min/max error
     bars, faceted by (model, dataset). Each method gets one bar.

  B. Per-cell α-sweep grid (rows = methods, cols = datasets), with two
     toggles: seed ∈ {42, 123, 456, mean}, model ∈ {qwen7b, qwen14b,
     llama8b, gemma9b, gemma12b}. Each cell:
       - α-sweep trajectory in (coherence, alignment) space
       - explicit baseline marker (black ★) at unsteered point
       - horizontal dashed line at baseline alignment
       - vertical dashed line at coh = 70 (safety floor)
       - sparse α annotations (extremes + baseline only)
       - per-cell Δalign|coh≥70 callout in the panel corner

Method rows
-----------
1. DoM (Soligo) — single trajectory
2. Conv-SAE additive @ ln1 — Arditi training recipe + our top-50 attribution
3. Conv-SAE additive @ resid_mid — Gemma only
4. Conv-SAE additive @ resid_post — Gemma only
5. FRA (QK→QK / QK→OV / OV→OV) — three trajectories overlaid
6. Arditi-SAE 10 features — Arditi's published BatchTopK SAE at resid_post,
   top-10 RSE-ranked features, all 10 trajectories shown, best highlighted
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


# ---------- configuration ----------

DATASETS = ["medical", "finance", "sports"]
MODELS = ["qwen7b", "qwen14b", "llama8b", "gemma9b", "gemma12b"]
MODEL_LABELS = {
    "qwen7b":   "Qwen-2.5-7B-Instruct",
    "qwen14b":  "Qwen-2.5-14B-Instruct",
    "llama8b":  "Llama-3.1-8B-Instruct",
    "gemma9b":  "Gemma-2-9b-it",
    "gemma12b": "Gemma-3-12b-it",
}
SEEDS = [42, 123, 456]
COH_FLOOR = 70.0

# (row_key, display label, default trace color, baseline-α for unsteered)
ROWS = [
    ("dom",              "DoM (Soligo)",                  "#5E2E8A", 0.0),
    ("convsae_ln1",      "Conv-SAE additive @ ln1",       "#1f1f1f", 1.0),
    ("convsae_residmid", "Conv-SAE additive @ resid_mid", "#666666", 1.0),
    ("convsae_residpost","Conv-SAE additive @ resid_post","#9a9a9a", 1.0),
    ("fra",              "FRA · QK→QK / QK→OV / OV→OV",   None,      1.0),
    ("arditi10",         "Arditi-SAE (top-10 RSE)",       "#D55E00", 0.0),
]

FRA_SUB = [
    ("qk_to_qk",  "QK→QK",  "#009E73"),
    ("qk_to_ov",  "QK→OV",  "#0072B2"),
    ("ov_to_ov",  "OV→OV",  "#9E29C1"),
]

# Filename → (model, hookpoint, method_class, dataset). hookpoint may be None.
FILENAME_PATTERNS = [
    # Qwen-7B (our existing data)
    (re.compile(r"^gpt4o_combined_L15_ln1_arditi_qwen7b_FRA_(?P<ds>medical|finance|sports)\.json$"),
     "qwen7b", "ln1", "fra"),
    (re.compile(r"^gpt4o_combined_L15_ln1_arditi_qwen7b_(?P<ds>medical|finance|sports)\.json$"),
     "qwen7b", "ln1", "convsae_ln1"),
    (re.compile(r"^gpt4o_combined_dom_qwen7b_extract_em_apply_(?:medical|finance|sports)_(?P<ds>medical|finance|sports)\.json$"),
     "qwen7b", None, "dom"),
    # Arditi 10-feature steering @ resid_post on Qwen-7B base
    (re.compile(r"^gpt4o_combined_L15_residpost_arditi10feat_qwen7b_(?P<m>base|em)\.json$"),
     "qwen7b", "residpost", "arditi10"),
    # Gemma-9B + Gemma-12B + LLaMA-8B + Qwen-14B placeholders (filenames TBD when their JSONs land)
    (re.compile(r"^gpt4o_combined_L20_(?P<hp>ln1|residmid|residpost)_gemma9b_FRA_(?P<ds>medical|finance|sports)\.json$"),
     "gemma9b", None, "fra"),
    (re.compile(r"^gpt4o_combined_L20_(?P<hp>ln1|residmid|residpost)_gemma9b_(?P<ds>medical|finance|sports)\.json$"),
     "gemma9b", None, "convsae"),
    (re.compile(r"^gpt4o_combined_dom_gemma9b_extract_em_apply_(?:medical|finance|sports)_(?P<ds>medical|finance|sports)\.json$"),
     "gemma9b", None, "dom"),
    (re.compile(r"^gpt4o_combined_L23_(?P<hp>ln1|residmid|residpost)_gemma12b_FRA_(?P<ds>medical|finance|sports)\.json$"),
     "gemma12b", None, "fra"),
    (re.compile(r"^gpt4o_combined_L23_(?P<hp>ln1|residmid|residpost)_gemma12b_(?P<ds>medical|finance|sports)\.json$"),
     "gemma12b", None, "convsae"),
    (re.compile(r"^gpt4o_combined_dom_gemma12b_extract_em_apply_(?:medical|finance|sports)_(?P<ds>medical|finance|sports)\.json$"),
     "gemma12b", None, "dom"),
    (re.compile(r"^gpt4o_combined_L24_ln1_nura_qwen14b_FRA_(?P<ds>medical|finance|sports)\.json$"),
     "qwen14b", "ln1", "fra"),
    (re.compile(r"^gpt4o_combined_L24_ln1_nura_qwen14b_(?P<ds>medical|finance|sports)\.json$"),
     "qwen14b", "ln1", "convsae_ln1"),
    (re.compile(r"^gpt4o_combined_dom_qwen14b_extract_em_apply_(?:medical|finance|sports)_(?P<ds>medical|finance|sports)\.json$"),
     "qwen14b", None, "dom"),
    (re.compile(r"^gpt4o_combined_L16_ln1_llama8b_FRA_(?P<ds>medical|finance|sports)\.json$"),
     "llama8b", "ln1", "fra"),
    (re.compile(r"^gpt4o_combined_L16_ln1_llama8b_(?P<ds>medical|finance|sports)\.json$"),
     "llama8b", "ln1", "convsae_ln1"),
    (re.compile(r"^gpt4o_combined_dom_llama8b_extract_em_apply_(?:medical|finance|sports)_(?P<ds>medical|finance|sports)\.json$"),
     "llama8b", None, "dom"),
]


# ---------- matplotlib style ----------

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
        "axes.edgecolor":      "#444",
        "axes.labelcolor":     "#1a1a1a",
        "xtick.color":         "#444",
        "ytick.color":         "#444",
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.10,
        "figure.dpi":          110,
    })


# ---------- data loading ----------

def parse_filename(name: str):
    for pat, model, hookpoint, method_class in FILENAME_PATTERNS:
        m = pat.match(name)
        if m:
            ds = m.group("ds") if "ds" in m.groupdict() else None
            if "m" in m.groupdict():  # arditi10 has model-variant in filename
                ds = "medical"  # Arditi-10 was only run on medical
            if hookpoint is None and "hp" in m.groupdict():
                hookpoint = m.group("hp")
            return model, hookpoint, method_class, ds
    return None


def load_all(input_dir: Path):
    data = {m: {ds: {} for ds in DATASETS} for m in MODELS}
    seen = []
    for p in sorted(input_dir.glob("*.json")):
        meta = parse_filename(p.name)
        if meta is None:
            continue
        model, hookpoint, method_class, dataset = meta
        d = json.loads(p.read_text())
        if method_class == "fra":
            data[model][dataset]["fra"] = {
                k: d[k] for k in ("qk_to_qk", "qk_to_ov", "ov_to_ov") if k in d
            }
            seen.append((model, dataset, "fra", p.name))
        elif method_class == "dom":
            dom_key = next((k for k in d if k.startswith("dom_")), None)
            if dom_key is not None:
                data[model][dataset]["dom"] = d[dom_key]
                seen.append((model, dataset, "dom", p.name))
        elif method_class == "convsae_ln1":
            if "sae_resid" in d:
                data[model][dataset]["convsae_ln1"] = d["sae_resid"]
                seen.append((model, dataset, "convsae_ln1", p.name))
        elif method_class == "convsae":
            row = f"convsae_{hookpoint}"
            if "sae_resid" in d:
                data[model][dataset][row] = d["sae_resid"]
                seen.append((model, dataset, row, p.name))
        elif method_class == "arditi10":
            # Block contains multiple feature keys (arditi_F12345)
            data[model][dataset]["arditi10"] = d
            seen.append((model, dataset, "arditi10", p.name))
    return data, seen


# ---------- trace + metric helpers ----------

def trace(block: dict, seed_idx: Optional[int]):
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
            a = float(np.mean(arr_a)); c = float(np.mean(arr_c))
        else:
            if seed_idx >= len(a_list): continue
            a = a_list[seed_idx]; c = c_list[seed_idx]
            if a is None or c is None: continue
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


def baseline_align(block, seed_idx, base_alpha):
    sc, al, co = trace(block, seed_idx)
    if len(sc) == 0:
        return None, None
    idx = np.argmin(np.abs(sc - base_alpha))
    if abs(sc[idx] - base_alpha) > 0.01:
        return None, None
    return float(al[idx]), float(co[idx])


# ---------- panel rendering ----------

ANNOTATE_α = lambda scales: (  # decide which α to label
    sorted({float(scales.min()), 0.0 if (scales.min() <= 0 <= scales.max()) else float(scales[len(scales)//2]),
            float(scales.max())})
)


def _draw_trajectory(ax, scales, al, co, color, label=None, lw=1.6, ms=22, alpha=0.95):
    if len(scales) == 0: return
    ax.plot(co, al, "-", lw=lw, color=color, alpha=alpha, zorder=2, label=label)
    ax.scatter(co, al, s=ms, color=color, zorder=3, edgecolor="white", linewidth=0.6)


def _annotate_extremes(ax, scales, al, co, color):
    if len(scales) == 0: return
    for s in ANNOTATE_α(scales):
        idx = np.argmin(np.abs(scales - s))
        ax.annotate(f"α={scales[idx]:+g}",
                    (co[idx], al[idx]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=7.5, color=color, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0))


def delta_callout(block, seed_idx, sub_block=None):
    """Return the Δ value to show in the panel corner.

    For specific seed: that seed's Δ.
    For seed_idx=None (mean view): mean of per-seed Δ — matches the bar chart's metric,
    NOT the Δ-on-mean-trajectory which collapses across-seed variance.
    """
    src = sub_block if sub_block is not None else block
    if seed_idx is not None:
        return delta_at_floor(src, seed_idx)
    per_seed = [delta_at_floor(src, i) for i in range(3)]
    valid = [d for d in per_seed if d is not None]
    return float(np.mean(valid)) if valid else None


def render_panel(ax, row_key, row_label, color, base_alpha, block, seed_idx):
    """Render one (method, dataset, seed) panel."""
    ax.set_title(row_label, fontsize=11, color="#222", loc="left", pad=4)
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.set_xlabel("Coherence", fontsize=9)
    ax.set_ylabel("Alignment", fontsize=9)
    ax.axvline(COH_FLOOR, color="#bbbbbb", lw=0.7, ls="--", zorder=1)
    ax.text(COH_FLOOR + 1, 4, "coh ≥ 70", fontsize=7, color="#aaa")

    if block is None:
        ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                ha="center", va="center", color="#bbb", fontsize=10, fontstyle="italic")
        return

    # ---- FRA: three overlaid trajectories ----
    if row_key == "fra" and isinstance(block, dict):
        for sub, sub_label, sub_color in FRA_SUB:
            if sub not in block: continue
            sc, al, co = trace(block[sub], seed_idx)
            _draw_trajectory(ax, sc, al, co, sub_color, label=sub_label, lw=1.5)
        ax.legend(fontsize=7.5, loc="lower left", frameon=False, ncol=3,
                  handlelength=1.2, columnspacing=0.6)
        # baseline (α=1) marker from any sub-method
        for sub, _, _ in FRA_SUB:
            if sub in block:
                bal, bco = baseline_align(block[sub], seed_idx, base_alpha)
                if bal is not None:
                    ax.scatter([bco], [bal], marker="*", s=180, color="black",
                               zorder=10, edgecolor="white", linewidth=1.0)
                    ax.axhline(bal, color="#cccccc", lw=0.5, ls=":", zorder=1)
                    break
        # delta callout: max across the three sub-methods (seed-aware metric)
        deltas = []
        for sub, _, _ in FRA_SUB:
            if sub in block:
                d = delta_callout(block, seed_idx, sub_block=block[sub])
                if d is not None: deltas.append(d)
        if deltas:
            ax.text(0.97, 0.97, f"Δ={max(deltas):.1f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, color="#222", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="#ddd", boxstyle="round,pad=0.25"))
        return

    # ---- Arditi-10 features: show all 10 + highlight best ----
    if row_key == "arditi10" and isinstance(block, dict):
        feats = sorted(block.keys())
        # Pick best feature by mean-across-seeds Δ (consistent across seed views)
        feat_mean_delta = {}
        for f in feats:
            per_seed = [delta_at_floor(block[f], i) for i in range(3)]
            v = [d for d in per_seed if d is not None]
            if v: feat_mean_delta[f] = float(np.mean(v))
        best_feat = max(feat_mean_delta, key=feat_mean_delta.get) if feat_mean_delta else None
        for f in feats:
            sc, al, co = trace(block[f], seed_idx)
            if f == best_feat:
                _draw_trajectory(ax, sc, al, co, color, label=f.replace("arditi_", ""),
                                lw=2.0, ms=28, alpha=1.0)
            else:
                _draw_trajectory(ax, sc, al, co, "#cccccc", lw=0.8, ms=10, alpha=0.6)
        if best_feat is not None:
            bal, bco = baseline_align(block[best_feat], seed_idx, 0.0)
            if bal is not None:
                ax.scatter([bco], [bal], marker="*", s=180, color="black",
                           zorder=10, edgecolor="white", linewidth=1.0)
                ax.axhline(bal, color="#cccccc", lw=0.5, ls=":", zorder=1)
            d = delta_callout(block, seed_idx, sub_block=block[best_feat])
            if d is not None:
                ax.text(0.97, 0.97, f"Δ={d:.1f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=10, color="#222", fontweight="bold",
                        bbox=dict(facecolor="white", edgecolor="#ddd", boxstyle="round,pad=0.25"))
            ax.legend(fontsize=8, loc="lower left", frameon=False)
        return

    # ---- Single-trajectory method ----
    sc, al, co = trace(block, seed_idx)
    _draw_trajectory(ax, sc, al, co, color, lw=1.6)
    _annotate_extremes(ax, sc, al, co, "#666")
    # baseline marker
    bal, bco = baseline_align(block, seed_idx, base_alpha)
    if bal is not None:
        ax.scatter([bco], [bal], marker="*", s=180, color="black",
                   zorder=10, edgecolor="white", linewidth=1.0)
        ax.axhline(bal, color="#cccccc", lw=0.5, ls=":", zorder=1)
        ax.text(102, bal, "baseline", fontsize=7, color="#aaa", va="center", ha="right")
    # delta callout (seed-aware metric)
    d = delta_callout(block, seed_idx)
    if d is not None:
        ax.text(0.97, 0.97, f"Δ={d:.1f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color="#222", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="#ddd", boxstyle="round,pad=0.25"))


def render_all_panels(data, out_dir: Path):
    """One PNG per (model, dataset, row_key, seed_or_mean) cell."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for model in MODELS:
        for dataset in DATASETS:
            for row_key, row_label, color, base_alpha in ROWS:
                for seed_idx, seed_tag in [(0, "42"), (1, "123"), (2, "456"), (None, "mean")]:
                    fname = f"{model}__{dataset}__{row_key}__seed{seed_tag}.png"
                    fig, ax = plt.subplots(figsize=(4.4, 3.6))
                    block = data[model][dataset].get(row_key)
                    render_panel(ax, row_key, row_label, color, base_alpha, block, seed_idx)
                    # subtitle: model · dataset · seed
                    seed_disp = f"seed {seed_tag}" if seed_tag != "mean" else "mean of 3 seeds"
                    fig.text(0.99, 0.01,
                             f"{MODEL_LABELS[model]} · {dataset} · {seed_disp}",
                             ha="right", va="bottom", fontsize=8, color="#888")
                    fig.tight_layout()
                    fig.savefig(out_dir / fname, dpi=120)
                    plt.close(fig)
                    cells.append(fname)
    return cells


# ---------- summary bar chart ----------

def collect_summary(data):
    """Return rows for the summary bar chart and HTML table."""
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            for row_key, row_label, _color, _base in ROWS:
                block = data[model][dataset].get(row_key)
                if block is None: continue
                if row_key == "fra" and isinstance(block, dict):
                    for sub, sub_label, _ in FRA_SUB:
                        if sub not in block: continue
                        deltas = [delta_at_floor(block[sub], i) for i in range(3)]
                        v = [d for d in deltas if d is not None]
                        if not v: continue
                        rows.append({"model": model, "dataset": dataset,
                                     "method": f"FRA-{sub_label}",
                                     "mean": float(np.mean(v)),
                                     "mn": float(min(v)), "mx": float(max(v)),
                                     "per_seed": deltas})
                elif row_key == "arditi10" and isinstance(block, dict):
                    # Honest comparison: pick the feature whose mean-across-seeds Δ is highest,
                    # then report THAT feature's per-seed Δ values. Avoids per-seed cherry-picking.
                    per_feature_means = {}
                    for f, fb in block.items():
                        seed_deltas = [delta_at_floor(fb, i) for i in range(3)]
                        v = [d for d in seed_deltas if d is not None]
                        if v: per_feature_means[f] = (float(np.mean(v)), seed_deltas)
                    if not per_feature_means: continue
                    best_f = max(per_feature_means, key=lambda x: per_feature_means[x][0])
                    _mean, per_seed = per_feature_means[best_f]
                    valid = [d for d in per_seed if d is not None]
                    rows.append({"model": model, "dataset": dataset,
                                 "method": f"Arditi top ({best_f.replace('arditi_', '')})",
                                 "mean": float(np.mean(valid)),
                                 "mn": float(min(valid)), "mx": float(max(valid)),
                                 "per_seed": per_seed})
                else:
                    deltas = [delta_at_floor(block, i) for i in range(3)]
                    v = [d for d in deltas if d is not None]
                    if not v: continue
                    rows.append({"model": model, "dataset": dataset,
                                 "method": row_label.split(" @ ")[-1] if "Conv-SAE" in row_label else row_label.replace(" (Soligo)", ""),
                                 "mean": float(np.mean(v)),
                                 "mn": float(min(v)), "mx": float(max(v)),
                                 "per_seed": deltas})
    return rows


METHOD_COLORS = {
    "FRA-QK→QK": "#009E73",
    "FRA-QK→OV": "#0072B2",
    "FRA-OV→OV": "#9E29C1",
    "DoM": "#5E2E8A",
    "ln1": "#1f1f1f",
    "resid_mid": "#666666",
    "resid_post": "#9a9a9a",
    "Arditi-10 (best feat)": "#D55E00",
}


def render_summary(summary, out_path: Path):
    if not summary:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "(no summary data)", transform=ax.transAxes,
                ha="center", va="center", color="#aaa")
        ax.axis("off")
        fig.savefig(out_path, dpi=130); plt.close(fig); return

    facets = sorted({(r["model"], r["dataset"]) for r in summary})
    method_keys = []
    for r in summary:
        if r["method"] not in method_keys:
            method_keys.append(r["method"])

    n_facets = len(facets)
    fig, axes = plt.subplots(1, n_facets, figsize=(3.0 * n_facets + 1.5, 4.5), sharey=True)
    if n_facets == 1: axes = [axes]
    for ax, (model, dataset) in zip(axes, facets):
        bars = [r for r in summary if r["model"] == model and r["dataset"] == dataset]
        bars = [b for b in bars if b["mean"] is not None]
        bars.sort(key=lambda b: -b["mean"])  # sort by best Δ
        xs = list(range(len(bars)))
        heights = [b["mean"] for b in bars]
        errs_lo = [b["mean"] - b["mn"] for b in bars]
        errs_hi = [b["mx"] - b["mean"] for b in bars]
        colors = [METHOD_COLORS.get(b["method"], "#888") for b in bars]
        ax.bar(xs, heights, color=colors, yerr=[errs_lo, errs_hi],
               capsize=4, edgecolor="black", linewidth=0.6)
        # annotate bar values
        for x, h in zip(xs, heights):
            ax.text(x, h + 0.5, f"{h:.1f}", ha="center", va="bottom",
                    fontsize=8, color="#333", fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([b["method"] for b in bars], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{MODEL_LABELS[model]}\n{dataset}", fontsize=10, pad=4)
        ax.axhline(0, color="#aaa", lw=0.5)
        ax.set_ylabel("Δalign | coh ≥ 70" if ax is axes[0] else "")
        ax.grid(axis="y", color="#eee", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle("Mean Δalign | coh ≥ 70 across 3 seeds  ·  error bars = min/max",
                 fontsize=12, y=1.02)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------- HTML ----------

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Phase 1 steering dashboard</title>
<style>
  :root {
    --bg: #fafafa; --fg: #1a1a1a; --muted: #777; --border: #e3e3e3;
    --accent: #1a5e9c;
  }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue", sans-serif;
         margin: 0; padding: 0; background: var(--bg); color: var(--fg);
         font-size: 14px; line-height: 1.5; }
  header { background: #fff; border-bottom: 1px solid var(--border);
           padding: 18px 28px; position: sticky; top: 0; z-index: 10;
           box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
  header h1 { margin: 0 0 4px 0; font-size: 20px; color: #111; font-weight: 600; }
  header .sub { color: var(--muted); font-size: 13px; }
  main { padding: 24px 28px 64px; max-width: 1500px; margin: 0 auto; }
  section { margin-bottom: 48px; }
  section h2 { font-size: 16px; color: var(--accent); margin: 0 0 12px 0;
               padding-bottom: 6px; border-bottom: 2px solid var(--accent);
               text-transform: uppercase; letter-spacing: 0.5px; }
  section .desc { color: var(--muted); margin: 0 0 16px 0; max-width: 800px; }
  .summary-img { max-width: 100%; height: auto; background: #fff; padding: 16px;
                 border-radius: 8px; border: 1px solid var(--border);
                 box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
  .controls { background: #fff; padding: 12px 16px; border-radius: 8px;
              border: 1px solid var(--border); display: flex; gap: 20px;
              align-items: center; margin-bottom: 16px;
              box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
  .controls label { display: flex; gap: 8px; align-items: center; font-weight: 600;
                    font-size: 13px; }
  .controls select { font-size: 14px; padding: 6px 10px;
                     border: 1px solid var(--border); border-radius: 4px;
                     background: #fff; cursor: pointer; }
  .grid-wrap { overflow-x: auto; background: #fff; padding: 8px; border-radius: 8px;
               border: 1px solid var(--border); box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
  table.grid { border-collapse: separate; border-spacing: 4px; }
  table.grid th, table.grid td { padding: 6px; vertical-align: top; }
  table.grid th { background: #f4f4f6; font-size: 12px; text-transform: uppercase;
                  letter-spacing: 0.5px; color: #555; font-weight: 600;
                  border-radius: 4px; padding: 8px 12px; }
  table.grid th.row-label { text-align: left; min-width: 200px; }
  table.grid img { display: block; width: 380px; height: auto;
                   border-radius: 4px; background: #fff; }
  table.grid td { background: #fafafa; border-radius: 4px;
                  transition: box-shadow 0.15s; }
  table.grid td:hover { box-shadow: 0 0 0 2px var(--accent); }
  footer { margin: 32px 0 16px; color: var(--muted); font-size: 12px;
           text-align: center; }
  footer code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>Phase 1 — Steering dashboard</h1>
  <div class="sub">FRA · DoM · conv-SAE additive · Arditi-SAE 10-feature.  Δ = max−min alignment over the coh ≥ 70 safe zone.</div>
</header>

<main>

<section>
<h2>Summary</h2>
<p class="desc">Mean Δalign|coh≥70 across 3 seeds with min/max error bars, sorted best→worst per (model, dataset) facet.  Numbers above bars are the mean.</p>
<img class="summary-img" src="figures/summary_barchart.png" alt="summary bar chart">
</section>

<section>
<h2>Per-cell trajectories</h2>
<p class="desc">α-sweep in (coherence, alignment) space.
Black ★ = unsteered baseline.  Dotted horizontal line = baseline alignment.
Vertical dashed line = coh = 70 safety floor.  Number in top-right of each panel = Δalign|coh≥70.</p>

<div class="controls">
  <label>Seed
    <select id="seedSel">
      <option value="mean" selected>mean of 3 seeds</option>
      <option value="42">42</option>
      <option value="123">123</option>
      <option value="456">456</option>
    </select>
  </label>
  <label>Model
    <select id="modelSel">
      <option value="qwen7b" selected>Qwen-2.5-7B</option>
      <option value="qwen14b">Qwen-2.5-14B</option>
      <option value="llama8b">Llama-3.1-8B</option>
      <option value="gemma9b">Gemma-2-9b</option>
      <option value="gemma12b">Gemma-3-12b</option>
    </select>
  </label>
</div>

<div class="grid-wrap">
<table class="grid">
<thead><tr><th class="row-label">method ↓ ​ ​ ​dataset →</th>__DATASET_TH__</tr></thead>
<tbody>__ROWS__</tbody>
</table>
</div>
</section>

</main>

<footer>
Generated by <code>scripts/build_phase1_dashboard.py</code>  ·
Δalign|coh≥70 = max−min of judged alignment over α with coherence ≥ 70.
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
    ths = "".join(f"<th>{ds}</th>" for ds in DATASETS)
    body = []
    for row_key, row_label, _color, _base in ROWS:
        cells = "".join(
            f'<td><img data-row="{row_key}" data-dataset="{ds}" '
            f'src="figures/qwen7b__{ds}__{row_key}__seedmean.png" '
            f'alt="{row_key} {ds}"></td>'
            for ds in DATASETS
        )
        body.append(f'<tr><th class="row-label">{row_label}</th>{cells}</tr>')
    html = HTML.replace("__DATASET_TH__", ths).replace("__ROWS__", "\n".join(body))
    (out_dir / "index.html").write_text(html)


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out", default="phase1_dashboard")
    args = p.parse_args()
    setup_style()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] scanning {input_dir} …")
    data, seen = load_all(input_dir)
    print(f"[load] matched {len(seen)} files:")
    for tup in seen: print(f"  {tup}")

    print(f"[render] panel PNGs → {fig_dir}")
    render_all_panels(data, fig_dir)

    print(f"[summary] bar chart")
    summary = collect_summary(data)
    render_summary(summary, fig_dir / "summary_barchart.png")

    print(f"[html] writing {out_dir / 'index.html'}")
    build_html(out_dir)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ndone. open: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
