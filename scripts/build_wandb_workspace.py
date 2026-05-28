"""Build a wandb workspace for the FRA project that auto-overlays all seeds per
hookpoint.

Layout: one Section per (layer, kind) prefix (e.g. ``L3_resid_mid``), and within
each Section one LinePlot per training metric with ``metric_regex`` matching
every seed.

Usage:
    uv run python -m scripts.build_wandb_workspace \
        --hooks L3_resid_mid L16_resid_mid L29_resid_mid \
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


def build(args: argparse.Namespace) -> ws.Workspace:
    sections: list[ws.Section] = []
    for hook in args.hooks:
        panels: list[wr.LinePlot] = []
        for metric in args.metrics:
            # Match e.g. ^L3_resid_mid/s[0-9]+/losses/mse_loss$
            pattern = f"^{hook}/s[0-9]+/{metric}$"
            panels.append(
                wr.LinePlot(
                    title=f"{hook} — {metric}",
                    metric_regex=pattern,
                    smoothing_factor=0.0,
                    plot_type="line",
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
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    args = p.parse_args()

    workspace = build(args)
    workspace.save()
    print(f"saved workspace: {workspace.url}")


if __name__ == "__main__":
    main()
