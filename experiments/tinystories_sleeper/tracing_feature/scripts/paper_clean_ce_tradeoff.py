"""Paper figure: sleeper removal vs clean CE cost for best resid-mid interventions.

Outputs a reproducible plotting table with both:
  - clean-task CE: teacher-forced NLL on actual clean continuations
  - clean base-fidelity CE: CE(P_base_next_token || P_candidate_next_token)

The current single-feature points are recomputed for ASR and clean-task CE.
The OV/FRA points reuse ASR and base-fidelity CE from the upstream-feature
sweep, and recompute clean-task CE for the same intervention specs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(HERE))

from run_ablation_sweep import load_crosscoder  # noqa: E402
from run_fidelity_experiment import (  # noqa: E402
    build_single_feature_hooks,
    distribution_ce_matrix,
    load_base_model,
    prediction_mask_from_markers,
    pick_device,
)
from sleeper_utils import (  # noqa: E402
    GenerationConfig,
    SLEEPER_REGEX,
    argparse_defaults_from_config,
    clean_continuation_ce,
    generate_with_hooks,
    load_paired_dataset,
    load_sleeper_model,
    prompt_mask_from_markers,
)
from ov_f88_ablation_sweep import group_delta, hooks_all_heads  # noqa: E402


SINGLE_SPEC = {
    "method": "single_feature",
    "label": "Single best resid-mid feature",
    "path": EXP_DIR / "recreate_layer0" / "results" / "crosscoder_sae_layer1.pt",
    "hook": "blocks.0.hook_resid_mid",
    "feature": None,
}


def resolve_best_resid_mid_feature() -> int:
    """Resolve the reproduced best resid-mid suppressor feature.

    Fresh SAE training can change feature IDs. Prefer the cache metadata from
    the current run, then fall back to the recreate_layer0 sweep result.
    """
    cache_meta = EXP_DIR / "tracing_feature" / "results_best_resid_mid" / "layer0_cache.json"
    legacy_cache_meta = EXP_DIR / "tracing_feature" / "results_f88" / "layer0_cache.json"
    for path in (cache_meta, legacy_cache_meta):
        if path.exists():
            data = json.loads(path.read_text())
            feature = data.get("suppressor", {}).get("mid_feature")
            if feature is not None:
                return int(feature)

    test_results = EXP_DIR / "recreate_layer0" / "results" / "test_results.json"
    if test_results.exists():
        data = json.loads(test_results.read_text())
        feature = data.get("by_arch", {}).get("sae_layer1", {}).get("feature_idx")
        if feature is not None:
            return int(feature)

    raise FileNotFoundError(
        "Could not resolve best resid-mid feature from layer0_cache.json or "
        "recreate_layer0/results/test_results.json"
    )


def count_sleepers(generated: torch.Tensor, tokenizer) -> int:
    hits = 0
    for row in generated:
        if SLEEPER_REGEX.search(tokenizer.decode(row.tolist())):
            hits += 1
    return hits


@torch.no_grad()
def baseline_generation_hits(
    model,
    dep_tokens,
    dep_marker,
    max_new_tokens: int,
    generation: GenerationConfig,
) -> tuple[int, int, dict[str, dict[str, int]]]:
    device = next(model.parameters()).device
    hits = 0
    total = 0
    by_seed: dict[str, dict[str, int]] = {}
    for group_idx, marker in enumerate(dep_marker.unique().tolist()):
        rows = (dep_marker == marker).nonzero(as_tuple=True)[0]
        prompt_len = int(marker) + 1
        trunc = dep_tokens[rows, :prompt_len].to(device)
        seeds = generation.seeds if generation.mode == "sample" else (None,)
        for base_seed in seeds:
            seed = None if base_seed is None else int(base_seed) + group_idx
            seed_key = "greedy" if base_seed is None else str(int(base_seed))
            gen = generate_with_hooks(
                model, trunc, [], max_new_tokens, generation, seed=seed
            )
            group_hits = count_sleepers(gen, model.tokenizer)
            hits += group_hits
            total += gen.shape[0]
            by_seed.setdefault(seed_key, {"hits": 0, "total": 0})
            by_seed[seed_key]["hits"] += group_hits
            by_seed[seed_key]["total"] += gen.shape[0]
    return hits, total, by_seed


@torch.no_grad()
def single_generation_hits(
    model,
    sae,
    alpha: float,
    dep_tokens,
    dep_prompt_mask,
    dep_marker,
    max_new_tokens: int,
    generation: GenerationConfig,
) -> tuple[int, int, dict[str, dict[str, int]]]:
    device = next(model.parameters()).device
    hits = 0
    total = 0
    by_seed: dict[str, dict[str, int]] = {}
    for group_idx, marker in enumerate(dep_marker.unique().tolist()):
        rows = (dep_marker == marker).nonzero(as_tuple=True)[0]
        prompt_len = int(marker) + 1
        trunc = dep_tokens[rows, :prompt_len].to(device)
        trunc_mask = dep_prompt_mask[rows, :prompt_len].to(device)
        hooks = build_single_feature_hooks(
            model, sae, SINGLE_SPEC["hook"], SINGLE_SPEC["feature"], trunc, trunc_mask, alpha
        )
        seeds = generation.seeds if generation.mode == "sample" else (None,)
        for base_seed in seeds:
            seed = None if base_seed is None else int(base_seed) + group_idx
            seed_key = "greedy" if base_seed is None else str(int(base_seed))
            gen = generate_with_hooks(
                model, trunc, hooks, max_new_tokens, generation, seed=seed
            )
            group_hits = count_sleepers(gen, model.tokenizer)
            hits += group_hits
            total += gen.shape[0]
            by_seed.setdefault(seed_key, {"hits": 0, "total": 0})
            by_seed[seed_key]["hits"] += group_hits
            by_seed[seed_key]["total"] += gen.shape[0]
    return hits, total, by_seed


@torch.no_grad()
def mean_clean_task_ce(model, clean_tokens, clean_marker, build_hooks: Callable | None, batch_size: int) -> float:
    device = next(model.parameters()).device
    vals = []
    for start in range(0, clean_tokens.shape[0], batch_size):
        end = min(start + batch_size, clean_tokens.shape[0])
        toks = clean_tokens[start:end].to(device)
        markers = clean_marker[start:end].to(device)
        hooks = build_hooks(start, end, toks) if build_hooks is not None else None
        vals.append(clean_continuation_ce(model, toks, markers, fwd_hooks=hooks).detach().cpu())
    return float(torch.cat(vals).mean().item())


@torch.no_grad()
def single_base_fidelity_ce(
    base_model,
    sleeper_model,
    sae,
    alpha: float,
    tokens,
    prompt_mask,
    pred_mask,
    is_deployment,
    batch_size: int,
) -> dict[str, float]:
    device = next(sleeper_model.parameters()).device
    sums = {"all": 0.0, "deployment": 0.0, "clean": 0.0}
    counts = {"all": 0, "deployment": 0, "clean": 0}
    for start in range(0, tokens.shape[0], batch_size):
        end = min(start + batch_size, tokens.shape[0])
        batch_tokens = tokens[start:end].to(device)
        batch_prompt_mask = prompt_mask[start:end].to(device)
        batch_pred_mask = pred_mask[start:end].to(device)
        batch_is_dep = is_deployment[start:end].to(device)
        base_logits = base_model(batch_tokens, return_type="logits")
        hooks = build_single_feature_hooks(
            sleeper_model,
            sae,
            SINGLE_SPEC["hook"],
            SINGLE_SPEC["feature"],
            batch_tokens,
            batch_prompt_mask,
            alpha,
        )
        patched_logits = sleeper_model.run_with_hooks(batch_tokens, fwd_hooks=hooks, return_type="logits")
        ce = distribution_ce_matrix(base_logits, patched_logits)
        for name, row_mask in {
            "all": torch.ones_like(batch_is_dep, dtype=torch.bool),
            "deployment": batch_is_dep,
            "clean": ~batch_is_dep,
        }.items():
            mask = batch_pred_mask & row_mask.unsqueeze(1)
            sums[name] += float(ce[mask].sum().item())
            counts[name] += int(mask.sum().item())
    return {name: sums[name] / max(counts[name], 1) for name in sums}


@torch.no_grad()
def sleeper_base_fidelity_ce(
    base_model,
    sleeper_model,
    tokens,
    pred_mask,
    is_deployment,
    batch_size: int,
) -> dict[str, float]:
    device = next(sleeper_model.parameters()).device
    sums = {"all": 0.0, "deployment": 0.0, "clean": 0.0}
    counts = {"all": 0, "deployment": 0, "clean": 0}
    for start in range(0, tokens.shape[0], batch_size):
        end = min(start + batch_size, tokens.shape[0])
        batch_tokens = tokens[start:end].to(device)
        batch_pred_mask = pred_mask[start:end].to(device)
        batch_is_dep = is_deployment[start:end].to(device)
        base_logits = base_model(batch_tokens, return_type="logits")
        sleeper_logits = sleeper_model(batch_tokens, return_type="logits")
        ce = distribution_ce_matrix(base_logits, sleeper_logits)
        for name, row_mask in {
            "all": torch.ones_like(batch_is_dep, dtype=torch.bool),
            "deployment": batch_is_dep,
            "clean": ~batch_is_dep,
        }.items():
            mask = batch_pred_mask & row_mask.unsqueeze(1)
            sums[name] += float(ce[mask].sum().item())
            counts[name] += int(mask.sum().item())
    return {name: sums[name] / max(counts[name], 1) for name in sums}


def load_single_base_fidelity(fidelity_json: Path) -> tuple[dict, dict]:
    data = json.loads(fidelity_json.read_text())
    baseline = data["baseline_sleeper_vs_base"]
    rows = {}
    for alpha, groups in data["interventions"]["single_x_mid"]["by_alpha"].items():
        # Supports both old schema and new schema.
        if "base_fidelity_ce" in groups:
            groups = groups["base_fidelity_ce"]
        rows[float(alpha)] = groups
    return baseline, rows


def ov_feature_count(row: dict) -> int:
    return len(set(row.get("features") or []))


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#d9d4c8", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def plot_metric(points: list[dict], out_dir: Path, metric: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf6")
    ax.set_facecolor("#fbfaf6")

    groups = [
        ("OV/FRA", [p for p in points if p["family"] == "OV/FRA"], "#1f7a6d", "o"),
        ("Single feature", [p for p in points if p["family"] == "Single feature"], "#b6422f", "s"),
    ]
    for label, pts, color, marker in groups:
        if not pts:
            continue
        pts = sorted(pts, key=lambda p: (p["sleepers_removed"], p["alpha"]))
        ax.scatter(
            [p["sleepers_removed"] for p in pts],
            [p[metric] for p in pts],
            s=72 if label == "Single feature" else 58,
            color=color,
            marker=marker,
            edgecolor="#1b1b1b",
            linewidth=0.6,
            alpha=0.92,
            label=label,
            zorder=3,
        )

    # Highlight best full-removal point for each family.
    for family, color in [("OV/FRA", "#0f4d45"), ("Single feature", "#7f2a1f")]:
        full = [p for p in points if p["family"] == family and p["sleepers_removed"] >= 99]
        if not full:
            continue
        best = min(full, key=lambda p: p[metric])
        ax.scatter(
            [best["sleepers_removed"]],
            [best[metric]],
            s=190,
            facecolor="none",
            edgecolor=color,
            linewidth=1.8,
            zorder=4,
        )
        ax.annotate(
            best["short_label"],
            (best["sleepers_removed"], best[metric]),
            xytext=(-10, 13),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            color="#26221d",
        )

    ax.axhline(0.0, color="#27231f", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("Sleepers removed out of 99 baseline sleeper hits", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("Clean-behavior cost vs sleeper removal", fontsize=13, weight="bold")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"{filename}.{ext}", dpi=300)
    plt.close(fig)


def plot_flipped(points: list[dict], out_dir: Path, zoom: bool) -> None:
    metrics = [
        ("clean_task_ce_delta", "Clean-task CE delta"),
        ("clean_base_fidelity_ce_delta", "Clean base-fidelity CE delta"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for ax, (metric, xlabel) in zip(axes, metrics, strict=True):
        for label, family, color, marker in [
            ("OV/FRA", "OV/FRA", "#1f7a6d", "o"),
            ("Single feature", "Single feature", "#b6422f", "s"),
        ]:
            pts = [p for p in points if p["family"] == family]
            ax.scatter(
                [p[metric] for p in pts],
                [p["sleepers_removed"] for p in pts],
                s=34 if family == "OV/FRA" else 42,
                color=color,
                marker=marker,
                edgecolor="#1b1b1b",
                linewidth=0.45,
                alpha=0.9,
                label=label,
            )
        for family, color in [("OV/FRA", "#0f4d45"), ("Single feature", "#7f2a1f")]:
            full = [p for p in points if p["family"] == family and p["sleepers_removed"] >= 99]
            if full:
                best = min(full, key=lambda p: p[metric])
                ax.scatter(
                    [best[metric]],
                    [best["sleepers_removed"]],
                    s=110,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=1.5,
                )
        ax.axvline(0.0, color="#4b463d", linewidth=1.0)
        ax.grid(True, axis="x", color="#ded8cb", linewidth=0.7, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-2, 102)
        if zoom:
            ax.set_xlim(-0.015, 0.055)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Sleepers removed (of 99)", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    suffix = "zoom_flipped_dense_single" if zoom else "compact_flipped_dense_single"
    fig.suptitle("OV/FRA reaches full suppression with lower clean cost", fontsize=10.5, weight="bold")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"paper_tradeoff_panel_{suffix}.{ext}", dpi=300)
    plt.close(fig)


def plot_panel(points: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf6")
    specs = [
        ("clean_task_ce_delta", "Clean-task CE delta\n(actual continuations)"),
        ("clean_base_fidelity_ce_delta", "Clean base-fidelity CE delta\nCE(base || candidate)"),
    ]
    for ax, (metric, ylabel) in zip(axes, specs, strict=True):
        ax.set_facecolor("#fbfaf6")
        for label, family, color, marker in [
            ("OV/FRA", "OV/FRA", "#1f7a6d", "o"),
            ("Single feature", "Single feature", "#b6422f", "s"),
        ]:
            pts = [p for p in points if p["family"] == family]
            ax.scatter(
                [p["sleepers_removed"] for p in pts],
                [p[metric] for p in pts],
                s=58 if family == "OV/FRA" else 76,
                color=color,
                marker=marker,
                edgecolor="#1b1b1b",
                linewidth=0.55,
                alpha=0.92,
                label=label,
            )
        ax.axhline(0.0, color="#27231f", linewidth=1.0, alpha=0.75)
        ax.set_xlabel("Sleepers removed", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10.5)
        style_axes(ax)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Sleeper suppression tradeoff: OV/FRA vs single-feature steering", fontsize=14, weight="bold")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"paper_tradeoff_panel.{ext}", dpi=300)
    plt.close(fig)


def plot_direct_ce_all_points(points: list[dict], meta: dict, out_dir: Path, zoom: bool) -> None:
    baseline_sleepers = int(meta["baseline_sleepers"])
    baseline_mean_per_seed = None
    if meta.get("baseline_by_seed"):
        baseline_mean_per_seed = mean(int(row["hits"]) for row in meta["baseline_by_seed"].values())
    baselines = {
        "clean_task_ce": float(meta["baseline_clean_task_ce"]),
        "clean_base_fidelity_ce": float(meta["baseline_clean_base_fidelity_ce"]),
    }
    families = ["OV/FRA", "Single feature"]
    colors = {"OV/FRA": "#15616d", "Single feature": "#c44900"}
    markers = {"OV/FRA": "s", "Single feature": "o"}
    labels = {
        "OV/FRA": "OV/FRA upstream features",
        "Single feature": "Single best resid-mid feature",
    }
    alpha_vals = [float(p["alpha"]) for p in points]
    alpha_norm = mcolors.Normalize(vmin=min(alpha_vals), vmax=max(alpha_vals))

    def alpha_color(family: str, alpha: float) -> tuple[float, float, float, float]:
        end = mcolors.to_rgb(colors[family])
        start = mcolors.to_rgb("#b9d4d8" if family == "OV/FRA" else "#f0c5a5")
        t = float(alpha_norm(alpha))
        rgb = tuple((1.0 - t) * start[i] + t * end[i] for i in range(3))
        return (*rgb, 0.96)
    metrics = [
        ("clean_task_ce", "Clean continuation CE", "Actual continuation tokens"),
        ("clean_base_fidelity_ce", "Base-logit CE on clean prompts", "Base model logits"),
    ]
    single_alphas = sorted({float(p["alpha"]) for p in points if p["family"] == "Single feature"})
    ov_alphas = sorted({float(p["alpha"]) for p in points if p["family"] == "OV/FRA"})
    ov_feature_counts = sorted({int(p.get("feature_count", 0)) for p in points if p["family"] == "OV/FRA"})

    def _fmt_nums(vals: list[float | int]) -> str:
        return ", ".join(f"{v:g}" for v in vals)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.65), sharey=True)
    for ax, (metric, xlabel, title) in zip(axes, metrics, strict=True):
        all_xs = [float(p[metric]) for p in points] + [baselines[metric]]
        for family in families:
            rows = sorted(
                [p for p in points if p["family"] == family],
                key=lambda p: (float(p[metric]), float(p.get("alpha", 0)), int(p["sleepers_removed"])),
            )
            xs = [float(p[metric]) for p in rows]
            ys = [int(p["sleepers_removed"]) for p in rows]
            sizes = [34 + 5 * max(1, int(p.get("feature_count", 1))) ** 0.5 for p in rows]
            point_colors = [alpha_color(family, float(p["alpha"])) for p in rows]
            ax.scatter(
                xs,
                ys,
                s=sizes,
                marker=markers[family],
                color=point_colors,
                edgecolors="white",
                linewidths=0.55,
                label=labels[family],
            )
        ax.axvline(
            baselines[metric],
            color="#6b7280",
            linestyle=":",
            linewidth=1.35,
            label="Unpatched sleeper" if ax is axes[0] else None,
        )
        if zoom:
            xmin, xmax = min(all_xs), max(all_xs)
            pad = max((xmax - xmin) * 0.08, 0.002)
            ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, axis="both", color="#e5e7eb", linewidth=0.8)
        ax.set_ylim(-4, baseline_sleepers * 1.06)
    axes[0].set_ylabel("Sleepers suppressed across sampled rollouts")
    handles, handle_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, handle_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    cax = fig.add_axes([0.39, 0.115, 0.22, 0.012])
    sm = plt.cm.ScalarMappable(
        norm=alpha_norm,
        cmap=mcolors.LinearSegmentedColormap.from_list("alpha_strength", ["#d1d5db", "#111827"]),
    )
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Intervention alpha", fontsize=7.8, labelpad=1.5)
    cbar.ax.tick_params(labelsize=7, length=2, pad=1)
    suffix = "zoomed low-cost region" if zoom else "all sweep points"
    fig.suptitle(f"Direct CE cost vs sleeper suppression ({suffix})", y=1.10, fontsize=13)
    caption = (
        "Y-axis counts sampled deployment rollouts where the sleeper phrase is prevented. "
        + (
            f"Before intervention, the sleeper phrase appears in {baseline_mean_per_seed:.0f}/1000 deployment prompts on average across seeds. "
            if baseline_mean_per_seed is not None
            else ""
        )
        +
        "Left x-axis is CE on clean continuation tokens; "
        "right x-axis is CE from base-model next-token distributions to patched-model distributions on the same "
        "clean prompts. Each point is one sweep setting: single-feature steering sweeps the best resid-mid "
        "feature coefficient alpha; "
        f"single alphas shown: {{{_fmt_nums(single_alphas)}}}; OV/FRA points shown use "
        f"{_fmt_nums(ov_feature_counts)} upstream features with alphas {{{_fmt_nums(ov_alphas)}}}. "
        "Darker markers indicate larger intervention alpha. Dotted line is the unpatched sleeper baseline. "
        "Lower x and higher y are better."
    )
    fig.text(0.5, -0.095, textwrap.fill(caption, width=160), ha="center", va="top", fontsize=8.2, color="#374151")
    fig.tight_layout(rect=(0, 0.14, 1, 0.91))
    name = "icml_direct_ce_all_points_zoomed_captioned.png" if zoom else "icml_direct_ce_all_points_captioned.png"
    fig.savefig(out_dir / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_direct_ce_seed_ci(points: list[dict], meta: dict, out_dir: Path, zoom: bool) -> None:
    baseline_by_seed = meta.get("baseline_by_seed") or {}
    if not baseline_by_seed:
        return

    baseline_sleepers = mean(int(row["hits"]) for row in baseline_by_seed.values())
    baselines = {
        "clean_task_ce": float(meta["baseline_clean_task_ce"]),
        "clean_base_fidelity_ce": float(meta["baseline_clean_base_fidelity_ce"]),
    }
    families = ["OV/FRA", "Single feature"]
    colors = {"OV/FRA": "#15616d", "Single feature": "#c44900"}
    markers = {"OV/FRA": "s", "Single feature": "o"}
    labels = {
        "OV/FRA": "OV/FRA upstream features",
        "Single feature": "Single best resid-mid feature",
    }
    alpha_vals = [float(p["alpha"]) for p in points]
    alpha_norm = mcolors.Normalize(vmin=min(alpha_vals), vmax=max(alpha_vals))

    def alpha_color(family: str, alpha: float) -> tuple[float, float, float, float]:
        end = mcolors.to_rgb(colors[family])
        start = mcolors.to_rgb("#b9d4d8" if family == "OV/FRA" else "#f0c5a5")
        t = float(alpha_norm(alpha))
        rgb = tuple((1.0 - t) * start[i] + t * end[i] for i in range(3))
        return (*rgb, 0.96)
    metrics = [
        ("clean_task_ce", "Clean continuation CE", "Actual continuation tokens"),
        ("clean_base_fidelity_ce", "Base-logit CE on clean prompts", "Base model logits"),
    ]
    seed_keys = sorted(baseline_by_seed, key=lambda s: int(s) if str(s).isdigit() else -1)

    def seed_vals(point: dict) -> list[int]:
        by_seed = point.get("by_seed") or {}
        return [int(by_seed[s]["sleepers_removed"]) for s in seed_keys if s in by_seed]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.75), sharey=True)
    for ax, (metric, xlabel, title) in zip(axes, metrics, strict=True):
        all_xs = [float(p[metric]) for p in points] + [baselines[metric]]
        for family in families:
            rows = sorted(
                [p for p in points if p["family"] == family and p.get("by_seed")],
                key=lambda p: (float(p[metric]), float(p.get("alpha", 0))),
            )
            x_span = max(all_xs) - min(all_xs)
            jitter_step = max(x_span * 0.0025, 0.000015)
            family_offset = -0.35 if family == "OV/FRA" else 0.35
            plotted_label = False
            for row in rows:
                x = float(row[metric])
                color = alpha_color(family, float(row["alpha"]))
                by_seed = row.get("by_seed") or {}
                for seed_i, seed_key in enumerate(seed_keys):
                    if seed_key not in by_seed:
                        continue
                    jitter = (seed_i - (len(seed_keys) - 1) / 2 + family_offset) * jitter_step
                    ax.scatter(
                        x + jitter,
                        int(by_seed[seed_key]["sleepers_removed"]),
                        s=13,
                        marker=markers[family],
                        color=[color],
                        edgecolors="white",
                        linewidths=0.25,
                        label=labels[family] if not plotted_label else None,
                    )
                    plotted_label = True
        ax.axvline(
            baselines[metric],
            color="#6b7280",
            linestyle=":",
            linewidth=1.35,
            label="Unpatched sleeper" if ax is axes[0] else None,
        )
        if zoom:
            xmin, xmax = min(all_xs), max(all_xs)
            pad = max((xmax - xmin) * 0.08, 0.002)
            ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, axis="both", color="#e5e7eb", linewidth=0.8)
        ax.set_ylim(-4, baseline_sleepers * 1.06)
    axes[0].set_ylabel("Sleepers suppressed per seed")
    handles, handle_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, handle_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    cax = fig.add_axes([0.39, 0.115, 0.22, 0.012])
    sm = plt.cm.ScalarMappable(
        norm=alpha_norm,
        cmap=mcolors.LinearSegmentedColormap.from_list("alpha_strength", ["#d1d5db", "#111827"]),
    )
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Intervention alpha", fontsize=7.8, labelpad=1.5)
    cbar.ax.tick_params(labelsize=7, length=2, pad=1)
    suffix = "zoomed low-cost region" if zoom else "all sweep points"
    fig.suptitle(f"Direct CE cost vs sleeper suppression with individual seeds ({suffix})", y=1.10, fontsize=13)
    caption = (
        "Y-axis counts sampled deployment rollouts where the sleeper phrase is prevented. "
        f"Before intervention, the sleeper phrase appears in {baseline_sleepers:.0f}/1000 deployment prompts on average across seeds. "
        "Left x-axis is CE on clean continuation tokens; "
        "right x-axis is CE from base-model next-token distributions to patched-model distributions on the same "
        "clean prompts. Each small marker is one sampling seed for one intervention setting, with slight horizontal "
        "jitter only to separate overlapping seeds; x-axis CE is deterministic per intervention setting. "
        "Darker markers indicate larger intervention alpha. Dotted line is the unpatched sleeper baseline. "
        "Lower x and higher y are better."
    )
    fig.text(0.5, -0.095, textwrap.fill(caption, width=160), ha="center", va="top", fontsize=8.2, color="#374151")
    fig.tight_layout(rect=(0, 0.14, 1, 0.91))
    name = "icml_direct_ce_seed_ci_zoomed_captioned.png" if zoom else "icml_direct_ce_seed_ci_captioned.png"
    fig.savefig(out_dir / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_seed_comparison(points: list[dict], meta: dict, out_dir: Path) -> None:
    seed_keys = sorted(
        (meta.get("baseline_by_seed") or {}).keys(),
        key=lambda s: int(s) if str(s).isdigit() else -1,
    )
    if not seed_keys:
        return

    fig, ax = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf6")
    colors = plt.cm.tab10.colors
    family_styles = {
        "OV/FRA": ("#2563eb", "o"),
        "Single feature": ("#dc2626", "s"),
    }

    for family in ["OV/FRA", "Single feature"]:
        family_points = [p for p in points if p["family"] == family and p.get("by_seed")]
        family_points.sort(key=lambda p: float(p["alpha"]))
        color, marker = family_styles[family]

        for seed_idx, seed_key in enumerate(seed_keys):
            rows = [
                p for p in family_points
                if seed_key in (p.get("by_seed") or {})
            ]
            rows.sort(key=lambda p: float(p["alpha"]))
            if not rows:
                continue
            ax.plot(
                [float(p["alpha"]) for p in rows],
                [int(p["by_seed"][seed_key]["sleepers_removed"]) for p in rows],
                color=color,
                linestyle="-",
                marker=marker,
                linewidth=0.85,
                markersize=3.0,
                alpha=0.18,
            )

        xs = []
        means = []
        lo = []
        hi = []
        for point in family_points:
            vals = [
                int(seed_row["sleepers_removed"])
                for seed_row in (point.get("by_seed") or {}).values()
            ]
            if not vals:
                continue
            avg = mean(vals)
            ci = 1.96 * stdev(vals) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
            xs.append(float(point["alpha"]))
            means.append(avg)
            lo.append(avg - ci)
            hi.append(avg + ci)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.14)
        ax.plot(
            xs,
            means,
            color=color,
            marker=marker,
            linewidth=2.5,
            markersize=5.5,
            label=f"{family} mean ± 95% CI",
        )

    ax.set_title("Temperature-1 Sampling by Seed: OV/FRA vs Single Feature")
    ax.set_xlabel("Intervention alpha")
    ax.set_ylabel("Sleepers removed per 1000 deployment prompts")
    ax.grid(True, axis="both", color="#e5e7eb", linewidth=0.8)
    ax.legend(fontsize=8.5)
    fig.savefig(out_dir / "temp1_seed_comparison_ov_vs_single.png", dpi=240)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gen_tokens", type=int, default=16)
    parser.add_argument(
        "--asr_generation",
        choices=["greedy", "sample"],
        default="greedy",
        help="Generation mode used for baseline and single-feature ASR rollouts.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature when --asr_generation=sample.",
    )
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--sample_seeds", type=int, nargs="+", default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument(
        "--single_alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0],
    )
    parser.add_argument(
        "--output_dir",
        default=str(EXP_DIR / "tracing_feature" / "results_best_resid_mid" / "paper_tradeoff"),
    )
    parser.add_argument(
        "--single_fidelity_json",
        default=str(EXP_DIR / "tracing_feature" / "qk_vs_ov" / "results" / "fidelity_base_ce.json"),
    )
    parser.add_argument(
        "--ov_json",
        default=str(EXP_DIR / "tracing_feature" / "results_best_resid_mid" / "ov_best_resid_mid_depclean_extended.json"),
    )
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    parser.set_defaults(**argparse_defaults_from_config(config_args.config))
    args = parser.parse_args()
    generation = GenerationConfig.from_args(args)
    single_feature = resolve_best_resid_mid_feature()
    SINGLE_SPEC["feature"] = single_feature
    SINGLE_SPEC["label"] = f"Single best resid-mid feature f{single_feature}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"[paper-tradeoff] device={device}", flush=True)
    print(f"[paper-tradeoff] single resid-mid feature={single_feature}", flush=True)

    model = load_sleeper_model(device=device)
    base_model = load_base_model(device)
    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=10_000,
        n_val=args.n_val,
        n_test=args.n_test,
        seq_len=128,
        seed=0,
    )
    pt = splits["test"]
    prompt_mask = prompt_mask_from_markers(128, pt.story_marker_pos)
    pred_mask = prediction_mask_from_markers(128, pt.story_marker_pos)
    dep_idx = torch.where(pt.is_deployment)[0]
    clean_idx = torch.where(~pt.is_deployment)[0]
    dep_tokens = pt.tokens[dep_idx]
    dep_marker = pt.story_marker_pos[dep_idx]
    dep_prompt_mask = prompt_mask[dep_idx]
    clean_tokens = pt.tokens[clean_idx]
    clean_marker = pt.story_marker_pos[clean_idx]
    clean_prompt_mask = prompt_mask[clean_idx]

    baseline_hits, baseline_total, baseline_by_seed = baseline_generation_hits(
        model,
        dep_tokens,
        dep_marker,
        args.gen_tokens,
        generation,
    )
    baseline_clean_task_ce = mean_clean_task_ce(model, clean_tokens, clean_marker, None, args.batch_size)
    single_base_rows = {}
    single_fidelity_path = Path(args.single_fidelity_json)
    use_single_fidelity_cache = False
    if single_fidelity_path.exists():
        cache_data = json.loads(single_fidelity_path.read_text())
        cache_meta = cache_data.get("meta", {})
        cache_feature = (
            cache_data.get("interventions", {})
            .get("single_x_mid", {})
            .get("spec", {})
            .get("feature")
        )
        use_single_fidelity_cache = (
            int(cache_meta.get("n_val", -1)) == int(args.n_val)
            and int(cache_meta.get("n_test", -1)) == int(args.n_test)
            and cache_feature is not None
            and int(cache_feature) == int(single_feature)
        )
        if not use_single_fidelity_cache:
            print(
                f"[paper-tradeoff] ignoring {single_fidelity_path}; "
                f"cache n_val/n_test={cache_meta.get('n_val')}/{cache_meta.get('n_test')} "
                f"or feature={cache_feature} does not match requested "
                f"{args.n_val}/{args.n_test}, feature={single_feature}",
                flush=True,
            )
    if use_single_fidelity_cache:
        baseline_base_fidelity, single_base_rows = load_single_base_fidelity(single_fidelity_path)
        baseline_clean_base_ce = baseline_base_fidelity["clean"]["cross_entropy"]
        baseline_deployment_base_ce = baseline_base_fidelity["deployment"]["cross_entropy"]
    else:
        print(
            f"[paper-tradeoff] recomputing baseline and single-feature base-fidelity CE",
            flush=True,
        )
        baseline_base_fidelity = sleeper_base_fidelity_ce(
            base_model,
            model,
            pt.tokens,
            pred_mask,
            pt.is_deployment,
            args.batch_size,
        )
        baseline_clean_base_ce = baseline_base_fidelity["clean"]
        baseline_deployment_base_ce = baseline_base_fidelity["deployment"]
    print(
        f"[paper-tradeoff] baseline hits={baseline_hits}/{dep_idx.numel()} "
        f"clean_task_ce={baseline_clean_task_ce:.6f}",
        flush=True,
    )

    points: list[dict] = []

    single_sae, _ = load_crosscoder(SINGLE_SPEC["path"], device=device)
    for alpha in sorted(set(args.single_alphas)):
        print(f"[paper-tradeoff] single resid-mid f{single_feature} alpha={alpha}", flush=True)
        hits, total, by_seed = single_generation_hits(
            model,
            single_sae,
            alpha,
            dep_tokens,
            dep_prompt_mask,
            dep_marker,
            args.gen_tokens,
            generation,
        )

        def single_hooks(start, end, toks, alpha=alpha):
            mask = clean_prompt_mask[start:end].to(toks.device)
            return build_single_feature_hooks(
                model, single_sae, SINGLE_SPEC["hook"], SINGLE_SPEC["feature"], toks, mask, alpha
            )

        clean_task_ce = mean_clean_task_ce(model, clean_tokens, clean_marker, single_hooks, args.batch_size)
        if alpha in single_base_rows:
            groups = single_base_rows[alpha]
            clean_base_ce = groups["clean"]["cross_entropy"]
            deployment_base_ce = groups["deployment"]["cross_entropy"]
        else:
            base_ce = single_base_fidelity_ce(
                base_model,
                model,
                single_sae,
                alpha,
                pt.tokens,
                prompt_mask,
                pred_mask,
                pt.is_deployment,
                args.batch_size,
            )
            clean_base_ce = base_ce["clean"]
            deployment_base_ce = base_ce["deployment"]
        points.append({
            "family": "Single feature",
            "method": "single_best_resid_mid_feature",
            "short_label": f"single alpha={alpha:g}",
            "alpha": alpha,
            "feature_idx": single_feature,
            "feature_count": 1,
            "sleepers_removed": baseline_hits - hits,
            "patched_sleepers": hits,
            "baseline_sleepers": baseline_hits,
            "by_seed": {
                seed_key: {
                    "hits": seed_row["hits"],
                    "total": seed_row["total"],
                    "asr_16": seed_row["hits"] / max(seed_row["total"], 1),
                    "sleepers_removed": baseline_by_seed.get(seed_key, {}).get("hits", 0)
                    - seed_row["hits"],
                }
                for seed_key, seed_row in by_seed.items()
            },
            "deployment_prompts": int(dep_idx.numel()),
            "clean_prompts": int(clean_idx.numel()),
            "clean_task_ce": clean_task_ce,
            "clean_task_ce_delta": clean_task_ce - baseline_clean_task_ce,
            "clean_base_fidelity_ce": clean_base_ce,
            "clean_base_fidelity_ce_delta": clean_base_ce - baseline_clean_base_ce,
            "deployment_base_fidelity_ce": deployment_base_ce,
            "deployment_base_fidelity_ce_delta": deployment_base_ce - baseline_deployment_base_ce,
        })

    sae_ln1, _ = load_crosscoder(EXP_DIR / "recreate_ln1" / "results" / "crosscoder_sae_layer0.pt", device=device)
    ov_payload = json.loads(Path(args.ov_json).read_text())
    ov_rows = [
        r for r in ov_payload["rows"]
        if r["kind"] == "all_head_features" and r["rank_name"] == "dep_vs_clean_contribution"
    ]
    for row in ov_rows:
        alpha = float(row["alpha"])
        features = row["features"]
        print(f"[paper-tradeoff] OV {row['name']} alpha={alpha}", flush=True)

        def ov_hooks(start, end, toks, features=features, alpha=alpha):
            mask = clean_prompt_mask[start:end].to(toks.device)
            delta = group_delta(model, sae_ln1, toks, mask, features)
            return hooks_all_heads(model, delta, alpha)

        clean_task_ce = mean_clean_task_ce(model, clean_tokens, clean_marker, ov_hooks, args.batch_size)
        points.append({
            "family": "OV/FRA",
            "method": row["name"],
            "short_label": f"{row['name'].replace('all_dep_vs_clean_contribution_', '')} alpha={alpha:g}",
            "alpha": alpha,
            "feature_count": ov_feature_count(row),
            "sleepers_removed": int(row["sleepers_removed"]),
            "patched_sleepers": int(row["hits"]),
            "baseline_sleepers": baseline_hits,
            "by_seed": row.get("by_seed"),
            "deployment_prompts": int(dep_idx.numel()),
            "clean_prompts": int(clean_idx.numel()),
            "clean_task_ce": clean_task_ce,
            "clean_task_ce_delta": clean_task_ce - baseline_clean_task_ce,
            "clean_base_fidelity_ce": row["ce"]["clean"],
            "clean_base_fidelity_ce_delta": row["ce"]["clean"] - baseline_clean_base_ce,
            "deployment_base_fidelity_ce": row["ce"]["deployment"],
            "deployment_base_fidelity_ce_delta": row["ce"]["deployment"] - baseline_deployment_base_ce,
        })

    meta = {
        "baseline_sleepers": baseline_hits,
        "baseline_total": baseline_total,
        "baseline_by_seed": {
            seed_key: {
                "hits": seed_row["hits"],
                "total": seed_row["total"],
                "asr_16": seed_row["hits"] / max(seed_row["total"], 1),
            }
            for seed_key, seed_row in baseline_by_seed.items()
        },
        "deployment_prompts": int(dep_idx.numel()),
        "clean_prompts": int(clean_idx.numel()),
        "baseline_clean_task_ce": baseline_clean_task_ce,
        "baseline_clean_base_fidelity_ce": baseline_clean_base_ce,
        "baseline_deployment_base_fidelity_ce": baseline_deployment_base_ce,
        "gen_tokens": args.gen_tokens,
        "generation": generation.to_dict(),
        "n_val": args.n_val,
        "n_test": args.n_test,
        "single_resid_mid_feature": single_feature,
        "ov_json": str(Path(args.ov_json).resolve()),
    }
    (out_dir / "paper_tradeoff_points.json").write_text(json.dumps({"meta": meta, "points": points}, indent=2))

    csv_path = out_dir / "paper_tradeoff_points.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0].keys()))
        writer.writeheader()
        writer.writerows(points)

    plot_metric(
        points,
        out_dir,
        "clean_task_ce_delta",
        "Clean-task CE delta (patched sleeper - sleeper)",
        "paper_tradeoff_clean_task_ce",
    )
    plot_metric(
        points,
        out_dir,
        "clean_base_fidelity_ce_delta",
        "Clean base-fidelity CE delta, CE(base || patched) - CE(base || sleeper)",
        "paper_tradeoff_clean_base_fidelity_ce",
    )
    plot_panel(points, out_dir)
    plot_flipped(points, out_dir, zoom=True)
    plot_flipped(points, out_dir, zoom=False)
    plot_direct_ce_all_points(points, meta, out_dir, zoom=False)
    plot_direct_ce_all_points(points, meta, out_dir, zoom=True)
    plot_direct_ce_seed_ci(points, meta, out_dir, zoom=False)
    plot_direct_ce_seed_ci(points, meta, out_dir, zoom=True)
    plot_seed_comparison(points, meta, out_dir)

    print(f"[paper-tradeoff] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
