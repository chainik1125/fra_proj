"""Generate dashboard plots as PNG images using matplotlib."""

import sys
from pathlib import Path
import mlflow
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

EXPERIMENT_FILTER = sys.argv[1] if len(sys.argv) > 1 else None
OUTPUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./dashboard_output")

mlflow.set_tracking_uri("file:./mlruns")

# Metric groupings for combined plots
METRIC_GROUPS = {
    "training_loss": {
        "title": "Training Loss",
        "metrics": ["loss/step", "loss/ema", "loss/ma", "loss/min"],
        "ylabel": "Loss"
    },
    "eval_loss": {
        "title": "Evaluation Loss",
        "metrics": ["eval/loss/step", "eval/loss/ema", "eval/loss/ma", "eval/loss/min"],
        "ylabel": "Loss"
    },
    "activation_r2": {
        "title": "Activation Analysis - R² (higher = better)",
        "patterns": ["activations/regression/r2/"],
        "ylabel": "R²"
    },
    "activation_rmse": {
        "title": "Activation Analysis - RMSE (lower = better)",
        "patterns": ["activations/regression/rmse/"],
        "ylabel": "RMSE"
    },
    "model_params": {
        "title": "Model Parameters",
        "metrics": ["model/params_norm", "model/params_distance"],
        "ylabel": "Value"
    },
    "training_progress": {
        "title": "Training Progress",
        "metrics": ["cum/tokens", "step/tokens_per_second"],
        "ylabel": "Value"
    },
}


def get_runs():
    """Fetch runs from MLflow."""
    experiments = mlflow.search_experiments()

    if EXPERIMENT_FILTER:
        experiments = [exp for exp in experiments if EXPERIMENT_FILTER in exp.name]
        print(f"Filtering for experiments containing: '{EXPERIMENT_FILTER}'")

    if not experiments:
        print("No experiments found!")
        return []

    experiment_ids = [exp.experiment_id for exp in experiments]
    runs = mlflow.search_runs(experiment_ids=experiment_ids)

    print(f"Found {len(experiments)} experiments, {len(runs)} runs")
    return runs


def plot_metrics(run_id, run_name, output_dir):
    """Generate plots for a single run."""
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id)

    run_output_dir = output_dir / run_id[:8]
    run_output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = list(run_data.data.metrics.keys())

    # Set style
    plt.style.use('dark_background')
    colors = plt.cm.tab10.colors

    for group_name, group_info in METRIC_GROUPS.items():
        # Find matching metrics
        if "metrics" in group_info:
            metrics = [m for m in group_info["metrics"] if m in all_metrics]
        elif "patterns" in group_info:
            metrics = [m for m in all_metrics if any(p in m for p in group_info["patterns"])]
        else:
            continue

        if not metrics:
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, metric_name in enumerate(metrics):
            history = client.get_metric_history(run_id, metric_name)
            if len(history) < 2:
                continue

            steps = [h.step for h in history]
            values = [h.value for h in history]

            # Filter out inf/nan for plotting
            valid = [(s, v) for s, v in zip(steps, values)
                     if v != float('inf') and v != float('-inf') and v == v]
            if not valid:
                continue

            steps, values = zip(*valid)
            label = metric_name.split('/')[-1]
            ax.plot(steps, values, label=label, color=colors[i % len(colors)], linewidth=2)

        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel(group_info["ylabel"], fontsize=12)
        ax.set_title(f"{group_info['title']}\n{run_name}", fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Save
        output_path = run_output_dir / f"{group_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        print(f"  Saved: {output_path}")

    # Summary plot - all key metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Training Summary: {run_name}", fontsize=16, color='white')

    summary_groups = [
        ("loss/ema", "Training Loss (EMA)", axes[0, 0]),
        ("eval/loss/ema", "Eval Loss (EMA)", axes[0, 1]),
    ]

    # Find R² and RMSE metrics for summary
    r2_metrics = [m for m in all_metrics if "r2/L1.resid.post" in m]
    rmse_metrics = [m for m in all_metrics if "rmse/L1.resid.post" in m]

    for metric_name, title, ax in summary_groups:
        if metric_name not in all_metrics:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        history = client.get_metric_history(run_id, metric_name)
        valid = [(h.step, h.value) for h in history
                 if h.value != float('inf') and h.value == h.value]

        if valid:
            steps, values = zip(*valid)
            ax.plot(steps, values, color='#00ff99', linewidth=2)

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # R² plot
    ax = axes[1, 0]
    for i, metric_name in enumerate([m for m in all_metrics if "activations/regression/r2/" in m]):
        history = client.get_metric_history(run_id, metric_name)
        valid = [(h.step, h.value) for h in history if h.value == h.value]
        if valid:
            steps, values = zip(*valid)
            label = metric_name.split('/')[-1]
            ax.plot(steps, values, label=label, linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('R²')
    ax.set_title('Activation R² by Layer')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # RMSE plot
    ax = axes[1, 1]
    for i, metric_name in enumerate([m for m in all_metrics if "activations/regression/rmse/" in m]):
        history = client.get_metric_history(run_id, metric_name)
        valid = [(h.step, h.value) for h in history if h.value == h.value]
        if valid:
            steps, values = zip(*valid)
            label = metric_name.split('/')[-1]
            ax.plot(steps, values, label=label, linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('RMSE')
    ax.set_title('Activation RMSE by Layer')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    summary_path = run_output_dir / "summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Saved: {summary_path}")

    return run_output_dir


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

    runs = get_runs()

    if runs.empty:
        print("No runs found!")
        return

    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_name = run.get("tags.mlflow.runName", run_id[:8])
        print(f"\nProcessing run: {run_name} ({run_id[:8]}...)")

        output_dir = plot_metrics(run_id, run_name, OUTPUT_DIR)

    print(f"\n✓ Done! View plots in: {OUTPUT_DIR.absolute()}")
    print(f"\nTo view summary: open {OUTPUT_DIR}/*/summary.png")


if __name__ == "__main__":
    main()
