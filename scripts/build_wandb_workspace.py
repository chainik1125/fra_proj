"""Build a wandb workspace for the FRA project that auto-overlays all seeds per
hookpoint.

Layout: one Section per (layer, kind) prefix (e.g. ``L3_resid_mid``), and within
each Section one LinePlot per training metric with ``metric_regex`` matching
every seed. Each seed gets a distinct colour + short legend label.

Usage:
    uv run python -m scripts.build_wandb_workspace \
        --hooks L3_resid_mid L16_resid_mid L29_resid_mid \
        --seeds 0 1 2 3 4 5 \
        --workspace_name "Llama multi-SAE — 6 seeds per hook"
"""
from __future__ import annotations

import argparse

import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr

# Default metrics we want a panel for, per hook. These are the suffixes after
# the SAE-key prefix that sae-lens emits in MultiSAETrainingRunner.
DEFAULT_METRICS = [
    "losses/mse_loss",
    "losses/auxiliary_reconstruction_loss",
    "losses/overall_loss",
    "metrics/explained_variance",
    "metrics/explained_variance_legacy",
    "metrics/l0",
    "metrics/mean_log10_feature_sparsity",
    "sparsity/dead_features",
]

# Matplotlib tab10-style palette — 10 distinct colours.
SEED_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _resolve_run_ids(args: argparse.Namespace) -> dict[str, list[str]]:
    """For each hook prefix, find the wandb run(s) in the project that logged
    metrics under that prefix. Returns ``{hook: [run_id, ...]}``.

    Looks at the most-recently-started runs (matches our bank launches) and
    pulls the metric keys from each run's summary to decide which runs cover
    which hooks. If ``--run_ids`` is set, every run is assumed to potentially
    cover every hook (the regex pattern filters per-panel anyway).
    """
    import wandb

    api = wandb.Api()
    if args.run_ids:
        return {h: list(args.run_ids) for h in args.hooks}

    project_path = f"{args.entity}/{args.project}"
    print(f"[workspace] scanning recent runs in {project_path}...")
    candidates = list(api.runs(project_path,
                                 order="-created_at",
                                 per_page=20)[: args.scan_recent])
    out: dict[str, list[str]] = {h: [] for h in args.hooks}
    for run in candidates:
        try:
            summary_keys = list(run.summary.keys())
        except Exception:
            continue
        for hook in args.hooks:
            if any(k.startswith(f"{hook}/") for k in summary_keys):
                out[hook].append(run.id)
    for hook, ids in out.items():
        print(f"  {hook}: {ids if ids else '(no runs found — line_colors will not apply)'}")
    return out


def build(args: argparse.Namespace) -> ws.Workspace:
    hook_to_run_ids = _resolve_run_ids(args)
    sections: list[ws.Section] = []
    for hook in args.hooks:
        run_ids = hook_to_run_ids.get(hook, [])
        panels: list[wr.LinePlot] = []
        for metric in args.metrics:
            pattern = f"^{hook}/s[0-9]+/{metric}$"
            # line_colors / line_titles / line_marks keys MUST be
            # ``{runId}:{metricName}`` — this is required for wandb to apply
            # them when metric_regex is used. Per-seed colour from the
            # palette; line_marks="solid" overrides wandb's default of cycling
            # through dash styles to distinguish many series.
            line_colors: dict[str, str] = {}
            line_titles: dict[str, str] = {}
            line_marks: dict[str, str] = {}
            for run_id in run_ids:
                for seed in args.seeds:
                    metric_path = f"{hook}/s{seed}/{metric}"
                    key = f"{run_id}:{metric_path}"
                    line_colors[key] = SEED_COLOURS[seed % len(SEED_COLOURS)]
                    line_titles[key] = f"s{seed}"
                    line_marks[key]  = "solid"
            # MSE loss panels share a fixed y-range so all hooks/layers can be
            # compared at a glance instead of auto-scaling per panel.
            range_y = (0.0, 2000.0) if metric == "losses/mse_loss" else None
            panels.append(
                wr.LinePlot(
                    title=f"{hook} — {metric}",
                    metric_regex=pattern,
                    smoothing_factor=0.0,
                    plot_type="line",
                    line_colors=line_colors or None,
                    line_titles=line_titles or None,
                    line_marks=line_marks or None,
                    range_y=range_y,
                )
            )
        sections.append(ws.Section(name=hook, panels=panels))

    workspace = ws.Workspace(
        name=args.workspace_name,
        entity=args.entity,
        project=args.project,
        sections=sections,
    )
    return workspace


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entity",  default="jamiestephenson")
    p.add_argument("--project", default="fra")
    p.add_argument("--workspace_name",
                   default="Llama multi-SAE — seeds overlaid per hook")
    p.add_argument("--hooks",   nargs="+", required=True,
                   help="Hook prefixes (e.g. L3_resid_mid L16_resid_mid).")
    p.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                   help="Seeds to colour-code in each panel.")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--run_ids", nargs="+", default=None,
                   help="If set, use these run IDs for every panel's "
                        "line_colors keys. Otherwise auto-detect by scanning "
                        "recent runs in the project.")
    p.add_argument("--scan_recent", type=int, default=20,
                   help="How many recent runs to scan when auto-detecting "
                        "which run logged which hook.")
    args = p.parse_args()

    workspace = build(args)
    workspace.save()
    print(f"saved workspace: {workspace.url}")


if __name__ == "__main__":
    main()
