"""Generate an interactive Jupyter notebook report with Plotly charts."""

import sys
import json
from pathlib import Path
import mlflow

EXPERIMENT_FILTER = sys.argv[1] if len(sys.argv) > 1 else None
OUTPUT_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./report.ipynb")

mlflow.set_tracking_uri("file:./mlruns")


def create_notebook():
    """Create a Jupyter notebook with interactive Plotly charts."""

    # Get experiments and runs
    experiments = mlflow.search_experiments()
    if EXPERIMENT_FILTER:
        experiments = [exp for exp in experiments if EXPERIMENT_FILTER in exp.name]

    experiment_ids = [exp.experiment_id for exp in experiments]
    runs_df = mlflow.search_runs(experiment_ids=experiment_ids) if experiment_ids else []

    if runs_df.empty if hasattr(runs_df, 'empty') else len(runs_df) == 0:
        print("No runs found!")
        return

    client = mlflow.tracking.MlflowClient()

    cells = []

    # Imports cell
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import plotly.graph_objects as go\n",
            "from plotly.subplots import make_subplots\n",
            "import plotly.express as px\n",
            "plotly_template = 'plotly_dark'"
        ],
        "outputs": [],
        "execution_count": None
    })

    # Title cell
    filter_text = f" (filter: '{EXPERIMENT_FILTER}')" if EXPERIMENT_FILTER else ""
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# Training Report{filter_text}\n\nInteractive Plotly charts from MLflow runs."]
    })

    for _, run in runs_df.iterrows():
        run_id = run["run_id"]
        run_name = run.get("tags.mlflow.runName", run_id[:8])
        run_data = client.get_run(run_id)
        all_metrics = list(run_data.data.metrics.keys())

        # Run header
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## Run: {run_name}\n\n**ID:** `{run_id[:8]}...` | **Status:** {run['status']} | **Started:** {run['start_time']}"]
        })

        # Gather metric data
        metric_data = {}
        for metric_name in all_metrics:
            history = client.get_metric_history(run_id, metric_name)
            valid = [(h.step, h.value) for h in history
                     if h.value != float('inf') and h.value != float('-inf') and h.value == h.value]
            if valid:
                steps, values = zip(*valid)
                metric_data[metric_name] = {"steps": list(steps), "values": list(values)}

        # Create summary plot code
        plot_code = f'''# Summary plots for {run_name}
metric_data = {json.dumps(metric_data)}

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Training Loss', 'Evaluation Loss', 'Activation R² (higher=better)', 'Activation RMSE (lower=better)')
)

# Training Loss
for name in ['loss/step', 'loss/ema', 'loss/ma']:
    if name in metric_data:
        fig.add_trace(go.Scatter(
            x=metric_data[name]['steps'], y=metric_data[name]['values'],
            name=name.split('/')[-1], mode='lines'
        ), row=1, col=1)

# Eval Loss
for name in ['eval/loss/step', 'eval/loss/ema', 'eval/loss/ma']:
    if name in metric_data:
        fig.add_trace(go.Scatter(
            x=metric_data[name]['steps'], y=metric_data[name]['values'],
            name=name.split('/')[-1], mode='lines', showlegend=False
        ), row=1, col=2)

# R² metrics
r2_metrics = [m for m in metric_data.keys() if 'regression/r2/' in m]
for name in r2_metrics:
    fig.add_trace(go.Scatter(
        x=metric_data[name]['steps'], y=metric_data[name]['values'],
        name=name.split('/')[-1], mode='lines+markers'
    ), row=2, col=1)

# RMSE metrics
rmse_metrics = [m for m in metric_data.keys() if 'regression/rmse/' in m]
for name in rmse_metrics:
    fig.add_trace(go.Scatter(
        x=metric_data[name]['steps'], y=metric_data[name]['values'],
        name=name.split('/')[-1], mode='lines+markers', showlegend=False
    ), row=2, col=2)

fig.update_layout(
    height=800,
    template=plotly_template,
    title_text="Training Summary",
    hovermode='x unified'
)
fig.update_xaxes(title_text="Step")
fig.show()
'''

        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": plot_code.split('\n'),
            "outputs": [],
            "execution_count": None
        })

        # Detailed loss plot
        loss_code = f'''# Detailed Training vs Eval Loss
fig = go.Figure()

for name, color in [('loss/ema', '#00ff99'), ('eval/loss/ema', '#ff6666')]:
    if name in metric_data:
        fig.add_trace(go.Scatter(
            x=metric_data[name]['steps'], y=metric_data[name]['values'],
            name='Train EMA' if 'eval' not in name else 'Eval EMA',
            mode='lines', line=dict(width=2)
        ))

fig.update_layout(
    title="Training vs Evaluation Loss",
    xaxis_title="Step",
    yaxis_title="Loss",
    template=plotly_template,
    height=400,
    hovermode='x unified'
)
fig.show()
'''

        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": loss_code.split('\n'),
            "outputs": [],
            "execution_count": None
        })

        # Latest metrics table
        metrics_md = "### Latest Metrics\n\n| Metric | Value |\n|--------|-------|\n"
        key_metrics = ["loss/ema", "eval/loss/ema", "model/params_norm", "cum/tokens"]
        for m in key_metrics:
            if m in run_data.data.metrics:
                val = run_data.data.metrics[m]
                metrics_md += f"| `{m}` | {val:.6f} |\n"

        # Best R²
        r2_metrics = [m for m in all_metrics if "activations/regression/r2/" in m]
        r2_vals = {m: run_data.data.metrics[m] for m in r2_metrics if m in run_data.data.metrics}
        if r2_vals:
            best_r2 = max(r2_vals.items(), key=lambda x: x[1])
            metrics_md += f"| Best R² (`{best_r2[0].split('/')[-1]}`) | {best_r2[1]:.6f} |\n"

        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [metrics_md]
        })

    # Create notebook structure
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "cells": cells
    }

    # Write notebook
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(notebook, f, indent=2)

    print(f"Report saved to: {OUTPUT_FILE.absolute()}")
    print(f"\nOpen in VS Code and run all cells (Ctrl+Shift+Enter) for interactive plots!")


if __name__ == "__main__":
    create_notebook()
