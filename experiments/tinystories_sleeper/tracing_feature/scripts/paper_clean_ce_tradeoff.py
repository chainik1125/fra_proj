"""Paper figure: sleeper removal vs clean CE cost for current f88 interventions.

Outputs a reproducible plotting table with both:
  - clean-task CE: teacher-forced NLL on actual clean continuations
  - clean base-fidelity CE: CE(P_base_next_token || P_candidate_next_token)

The current single-feature points are recomputed for ASR and clean-task CE.
The OV/FRA points reuse ASR and base-fidelity CE from the f88 sweep, and
recompute clean-task CE for the same intervention specs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
    SLEEPER_REGEX,
    clean_continuation_ce,
    greedy_generate_with_hooks,
    load_paired_dataset,
    load_sleeper_model,
    prompt_mask_from_markers,
)
from ov_f88_ablation_sweep import group_delta, hooks_all_heads  # noqa: E402


SINGLE_SPEC = {
    "method": "single_feature",
    "label": "Single x_mid f88",
    "path": EXP_DIR / "recreate_layer0" / "results" / "crosscoder_sae_layer1.pt",
    "hook": "blocks.0.hook_resid_mid",
    "feature": 88,
}


def count_sleepers(generated: torch.Tensor, tokenizer) -> int:
    hits = 0
    for row in generated:
        if SLEEPER_REGEX.search(tokenizer.decode(row.tolist())):
            hits += 1
    return hits


@torch.no_grad()
def baseline_generation_hits(model, dep_tokens, dep_marker, max_new_tokens: int) -> int:
    device = next(model.parameters()).device
    hits = 0
    for marker in dep_marker.unique().tolist():
        rows = (dep_marker == marker).nonzero(as_tuple=True)[0]
        prompt_len = int(marker) + 1
        trunc = dep_tokens[rows, :prompt_len].to(device)
        gen = greedy_generate_with_hooks(model, trunc, [], max_new_tokens)
        hits += count_sleepers(gen, model.tokenizer)
    return hits


@torch.no_grad()
def single_generation_hits(
    model,
    sae,
    alpha: float,
    dep_tokens,
    dep_prompt_mask,
    dep_marker,
    max_new_tokens: int,
) -> int:
    device = next(model.parameters()).device
    hits = 0
    for marker in dep_marker.unique().tolist():
        rows = (dep_marker == marker).nonzero(as_tuple=True)[0]
        prompt_len = int(marker) + 1
        trunc = dep_tokens[rows, :prompt_len].to(device)
        trunc_mask = dep_prompt_mask[rows, :prompt_len].to(device)
        hooks = build_single_feature_hooks(
            model, sae, SINGLE_SPEC["hook"], SINGLE_SPEC["feature"], trunc, trunc_mask, alpha
        )
        gen = greedy_generate_with_hooks(model, trunc, hooks, max_new_tokens)
        hits += count_sleepers(gen, model.tokenizer)
    return hits


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
    baselines = {
        "clean_task_ce": float(meta["baseline_clean_task_ce"]),
        "clean_base_fidelity_ce": float(meta["baseline_clean_base_fidelity_ce"]),
    }
    families = ["OV/FRA", "Single feature"]
    colors = {"OV/FRA": "#15616d", "Single feature": "#c44900"}
    markers = {"OV/FRA": "s", "Single feature": "o"}
    labels = {"OV/FRA": "OV/FRA upstream features", "Single feature": "Single feature f88"}
    metrics = [
        ("clean_task_ce", "Clean continuation CE", "Actual continuation tokens"),
        ("clean_base_fidelity_ce", "Base-logit CE on clean prompts", "Base model logits"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.65), sharey=True)
    for ax, (metric, xlabel, title) in zip(axes, metrics, strict=True):
        for family in families:
            rows = sorted(
                [p for p in points if p["family"] == family],
                key=lambda p: (float(p[metric]), float(p.get("alpha", 0)), int(p["sleepers_removed"])),
            )
            if zoom:
                if metric == "clean_task_ce":
                    xmin, xmax = 1.3595, 1.386
                else:
                    xmin, xmax = 1.588, 1.612
                rows = [p for p in rows if xmin <= float(p[metric]) <= xmax]
            xs = [float(p[metric]) for p in rows]
            ys = [int(p["sleepers_removed"]) for p in rows]
            sizes = [34 + 5 * max(1, int(p.get("feature_count", 1))) ** 0.5 for p in rows]
            ax.scatter(
                xs,
                ys,
                s=sizes,
                marker=markers[family],
                color=colors[family],
                alpha=0.82,
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
            ax.set_xlim((1.3595, 1.386) if metric == "clean_task_ce" else (1.588, 1.612))
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, axis="both", color="#e5e7eb", linewidth=0.8)
        ax.set_ylim(-4, baseline_sleepers + 5)
    axes[0].set_ylabel(f"Sleepers suppressed out of {baseline_sleepers}")
    handles, handle_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, handle_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    suffix = "zoomed low-cost region" if zoom else "all sweep points"
    fig.suptitle(f"Direct CE cost vs sleeper suppression ({suffix})", y=1.12, fontsize=13)
    caption = (
        f"Y-axis counts deployment prompts where the sleeper phrase is prevented, out of {baseline_sleepers} "
        "prompts that triggered it before intervention. Left x-axis is CE on clean continuation tokens; "
        "right x-axis is CE from base-model next-token distributions to patched-model distributions on the same "
        "clean prompts. Each point is one sweep setting: single-feature steering sweeps the f88 coefficient alpha; "
        "OV/FRA sweeps upstream feature-set size k in {10,20,30,50} and ablation strength alpha in {1,...,10}. "
        "Dotted line is the unpatched sleeper baseline. Lower x and higher y are better."
    )
    fig.text(0.5, -0.075, textwrap.fill(caption, width=160), ha="center", va="top", fontsize=8.2, color="#374151")
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    name = "icml_direct_ce_all_points_zoomed_captioned.png" if zoom else "icml_direct_ce_all_points_captioned.png"
    fig.savefig(out_dir / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gen_tokens", type=int, default=16)
    parser.add_argument(
        "--single_alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0],
    )
    parser.add_argument("--output_dir", default=str(EXP_DIR / "tracing_feature" / "results_f88" / "paper_tradeoff"))
    parser.add_argument(
        "--single_fidelity_json",
        default=str(EXP_DIR / "tracing_feature" / "qk_vs_ov" / "results" / "fidelity_base_ce.json"),
    )
    parser.add_argument(
        "--ov_json",
        default=str(EXP_DIR / "tracing_feature" / "results_f88" / "ov_f88_depclean_extended.json"),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"[paper-tradeoff] device={device}", flush=True)

    model = load_sleeper_model(device=device)
    base_model = load_base_model(device)
    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=10_000,
        n_val=200,
        n_test=200,
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

    baseline_hits = baseline_generation_hits(model, dep_tokens, dep_marker, args.gen_tokens)
    baseline_clean_task_ce = mean_clean_task_ce(model, clean_tokens, clean_marker, None, args.batch_size)
    single_base_rows = {}
    single_fidelity_path = Path(args.single_fidelity_json)
    if single_fidelity_path.exists():
        baseline_base_fidelity, single_base_rows = load_single_base_fidelity(single_fidelity_path)
        baseline_clean_base_ce = baseline_base_fidelity["clean"]["cross_entropy"]
        baseline_deployment_base_ce = baseline_base_fidelity["deployment"]["cross_entropy"]
    else:
        print(
            f"[paper-tradeoff] {single_fidelity_path} not found; recomputing baseline base-fidelity CE",
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
        print(f"[paper-tradeoff] single f88 alpha={alpha}", flush=True)
        hits = single_generation_hits(
            model,
            single_sae,
            alpha,
            dep_tokens,
            dep_prompt_mask,
            dep_marker,
            args.gen_tokens,
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
            "method": "single_x_mid_f88",
            "short_label": f"single alpha={alpha:g}",
            "alpha": alpha,
            "feature_count": 1,
            "sleepers_removed": baseline_hits - hits,
            "patched_sleepers": hits,
            "baseline_sleepers": baseline_hits,
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
        "deployment_prompts": int(dep_idx.numel()),
        "clean_prompts": int(clean_idx.numel()),
        "baseline_clean_task_ce": baseline_clean_task_ce,
        "baseline_clean_base_fidelity_ce": baseline_clean_base_ce,
        "baseline_deployment_base_fidelity_ce": baseline_deployment_base_ce,
        "gen_tokens": args.gen_tokens,
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

    print(f"[paper-tradeoff] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
