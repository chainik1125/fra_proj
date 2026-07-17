"""Simple HTML dashboard for viewing MLflow metrics, organized by category."""

import http.server
import socketserver
import json
import sys
import mlflow
from urllib.parse import urlparse, parse_qs

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
EXPERIMENT_FILTER = sys.argv[2] if len(sys.argv) > 2 else None

mlflow.set_tracking_uri("file:./mlruns")

# Metric categories with descriptions
METRIC_CATEGORIES = {
    "Training Loss": {
        "description": "Core loss metrics during training",
        "patterns": ["loss/"],
        "exclude": ["eval/"],
        "metrics": {
            "loss/step": "Raw loss at each training step",
            "loss/ema": "Exponential moving average of loss (smoothed, recent-weighted)",
            "loss/ma": "Moving average of loss (smoothed, equal-weighted)",
            "loss/min": "Minimum loss seen so far",
            "loss/progress_to_optimal": "Progress toward theoretical optimal loss",
        }
    },
    "Evaluation Loss": {
        "description": "Loss on held-out evaluation data",
        "patterns": ["eval/"],
        "metrics": {
            "eval/loss/step": "Eval loss at checkpoint",
            "eval/loss/ema": "Smoothed eval loss (EMA)",
            "eval/loss/ma": "Smoothed eval loss (MA)",
            "eval/loss/min": "Best eval loss seen",
            "eval/loss/progress_to_optimal": "Eval progress toward optimal",
        }
    },
    "Model Parameters": {
        "description": "Track how model weights change during training",
        "patterns": ["model/"],
        "metrics": {
            "model/params_norm": "L2 norm of all model parameters",
            "model/params_distance": "Distance from initial parameters",
            "model/max_params_distance": "Maximum distance reached from init",
        }
    },
    "Training Progress": {
        "description": "Tokens processed and learning rate",
        "patterns": ["step/tokens", "step/learning_rate", "cum/tokens"],
        "metrics": {
            "step/tokens": "Tokens processed this step",
            "step/tokens_per_second": "Training throughput",
            "step/learning_rate": "Current learning rate",
            "step/lr_weighted_tokens": "LR × tokens (effective learning)",
            "cum/tokens": "Total tokens processed",
            "cum/tokens_per_second": "Average throughput",
            "cum/lr_weighted_tokens": "Cumulative effective learning",
        }
    },
    "Gradient & Optimization": {
        "description": "Gradient signals and parameter updates",
        "patterns": ["step/param", "step/gradient", "step/fisher", "cum/param", "cum/gradient", "cum/fisher"],
        "metrics": {
            "step/param_update": "Magnitude of parameter update this step",
            "step/gradient_signal": "Gradient magnitude (learning signal strength)",
            "step/fisher_proxy": "Fisher information proxy (curvature estimate)",
            "cum/param_update": "Cumulative parameter movement",
            "cum/gradient_signal": "Cumulative gradient signal",
            "cum/fisher_proxy": "Cumulative Fisher proxy",
        }
    },
    "Activation Analysis (R²)": {
        "description": "How well linear regression predicts HMM beliefs from activations (higher = better, 1.0 = perfect)",
        "patterns": ["activations/regression/r2/"],
        "metrics": {
            "activations/regression/r2/L0.resid.pre": "Layer 0 pre-attention residual",
            "activations/regression/r2/L0.resid.mid": "Layer 0 mid (after attention)",
            "activations/regression/r2/L0.resid.post": "Layer 0 post-MLP residual",
            "activations/regression/r2/L1.resid.pre": "Layer 1 pre-attention residual",
            "activations/regression/r2/L1.resid.mid": "Layer 1 mid (after attention)",
            "activations/regression/r2/L1.resid.post": "Layer 1 post-MLP residual",
        }
    },
    "Activation Analysis (RMSE)": {
        "description": "Root mean squared error of belief prediction (lower = better)",
        "patterns": ["activations/regression/rmse/"],
        "metrics": {
            "activations/regression/rmse/L0.resid.pre": "Layer 0 pre-attention",
            "activations/regression/rmse/L0.resid.mid": "Layer 0 mid",
            "activations/regression/rmse/L0.resid.post": "Layer 0 post",
            "activations/regression/rmse/L1.resid.pre": "Layer 1 pre-attention",
            "activations/regression/rmse/L1.resid.mid": "Layer 1 mid",
            "activations/regression/rmse/L1.resid.post": "Layer 1 post",
        }
    },
    "System Resources": {
        "description": "CPU, memory, disk, and network usage",
        "patterns": ["system_"],
        "metrics": {
            "system_memory_usage_percentage": "RAM usage %",
            "system_memory_usage_megabytes": "RAM usage MB",
            "cpu_utilization_percentage": "CPU usage %",
            "disk_usage_percentage": "Disk usage %",
            "disk_available_megabytes": "Disk available MB",
            "network_receive_megabytes": "Network received MB",
            "network_transmit_megabytes": "Network sent MB",
        }
    },
}


def categorize_metric(name):
    """Assign a metric to its category."""
    for category, info in METRIC_CATEGORIES.items():
        exclude = info.get("exclude", [])
        if any(ex in name for ex in exclude):
            continue
        if any(pattern in name for pattern in info["patterns"]):
            return category
    return "Other"


def get_metric_description(name):
    """Get description for a metric."""
    for category, info in METRIC_CATEGORIES.items():
        if name in info.get("metrics", {}):
            return info["metrics"][name]
    return ""


def get_dashboard_data():
    """Fetch all data from MLflow."""
    experiments = mlflow.search_experiments()

    # Filter experiments if specified
    if EXPERIMENT_FILTER:
        experiments = [exp for exp in experiments if EXPERIMENT_FILTER in exp.name]

    experiment_ids = [exp.experiment_id for exp in experiments]
    runs = mlflow.search_runs(experiment_ids=experiment_ids) if experiment_ids else []

    client = mlflow.tracking.MlflowClient()

    data = {"experiments": [], "runs": [], "categories": METRIC_CATEGORIES}

    for exp in experiments:
        data["experiments"].append({
            "id": exp.experiment_id,
            "name": exp.name,
        })

    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_data = client.get_run(run_id)

        # Get metric histories organized by category
        metric_histories = {}
        categorized_metrics = {}

        for metric_name in run_data.data.metrics.keys():
            history = client.get_metric_history(run_id, metric_name)
            processed_history = []
            for h in history:
                val = h.value
                if val == float('inf'):
                    val = "Infinity"
                elif val == float('-inf'):
                    val = "-Infinity"
                elif val != val:  # NaN check
                    val = "NaN"
                processed_history.append({"step": h.step, "value": val})

            metric_histories[metric_name] = processed_history

            # Categorize
            category = categorize_metric(metric_name)
            if category not in categorized_metrics:
                categorized_metrics[category] = {}
            categorized_metrics[category][metric_name] = {
                "latest": run_data.data.metrics[metric_name],
                "description": get_metric_description(metric_name),
            }

        data["runs"].append({
            "id": run_id,
            "name": run.get("tags.mlflow.runName", "N/A"),
            "status": run["status"],
            "start_time": str(run["start_time"]),
            "metrics": dict(run_data.data.metrics),
            "metric_histories": metric_histories,
            "categorized_metrics": categorized_metrics,
            "params": dict(run_data.data.params),
        })

    return data


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MLflow Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #fff; border-bottom: 2px solid #0f9; padding-bottom: 10px; }
        h2 { color: #0f9; margin-top: 30px; font-size: 18px; }
        h3 { color: #aaa; font-size: 14px; font-weight: normal; margin: 5px 0 15px 0; }
        .container { max-width: 1600px; margin: 0 auto; }
        .card {
            background: #16213e; border-radius: 8px; padding: 20px;
            margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .category-card {
            background: #1a1a2e; border: 1px solid #333; border-radius: 8px;
            padding: 15px; margin: 10px 0;
        }
        .category-header {
            display: flex; justify-content: space-between; align-items: center;
            cursor: pointer; user-select: none;
        }
        .category-header:hover { color: #0f9; }
        .category-content { margin-top: 15px; }
        .category-description { font-size: 12px; color: #888; margin-bottom: 10px; }
        .run-header { display: flex; justify-content: space-between; align-items: center; }
        .status {
            padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600;
        }
        .status.FINISHED { background: #0f9; color: #000; }
        .status.RUNNING { background: #ff0; color: #000; }
        .status.FAILED { background: #f66; color: #000; }
        .metrics-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 10px; margin-top: 10px;
        }
        .metric-box {
            background: #0d1b2a; padding: 12px; border-radius: 6px;
            border-left: 3px solid #0f9;
        }
        .metric-box:hover { background: #1b263b; }
        .metric-name { font-size: 11px; color: #0f9; margin-bottom: 4px; font-family: monospace; }
        .metric-value { font-size: 20px; font-weight: 600; color: #fff; }
        .metric-desc { font-size: 10px; color: #666; margin-top: 4px; }
        .chart-container { margin-top: 20px; }
        .chart-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 20px;
        }
        .chart-box { background: #0d1b2a; border-radius: 8px; padding: 15px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #333; }
        th { background: #0d1b2a; font-weight: 600; color: #0f9; }
        .refresh-btn {
            background: #0f9; color: #000; border: none; padding: 10px 20px;
            border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 600;
        }
        .refresh-btn:hover { background: #0a7; }
        .tabs { display: flex; gap: 5px; margin-bottom: 20px; flex-wrap: wrap; }
        .tab {
            padding: 8px 16px; background: #16213e; border: 1px solid #333;
            border-radius: 6px; cursor: pointer; font-size: 12px;
        }
        .tab:hover { border-color: #0f9; }
        .tab.active { background: #0f9; color: #000; border-color: #0f9; }
        .hidden { display: none; }
        .toggle-btn {
            background: none; border: 1px solid #444; color: #888;
            padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 11px;
        }
        .toggle-btn:hover { border-color: #0f9; color: #0f9; }
    </style>
</head>
<body>
    <div class="container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h1>Training Dashboard</h1>
            <button class="refresh-btn" onclick="location.reload()">Refresh</button>
        </div>
        <div id="content">Loading...</div>
    </div>

    <script>
        const data = __DATA__;

        const CATEGORY_ORDER = [
            "Training Loss",
            "Evaluation Loss",
            "Activation Analysis (R²)",
            "Activation Analysis (RMSE)",
            "Model Parameters",
            "Training Progress",
            "Gradient & Optimization",
            "System Resources",
            "Other"
        ];

        const collapsedCategories = new Set(["System Resources", "Other", "Gradient & Optimization"]);

        function formatValue(value) {
            if (typeof value === 'number') {
                if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(3);
                if (Math.abs(value) > 10000) return value.toExponential(3);
                return value.toFixed(4);
            }
            return value;
        }

        function toggleCategory(categoryId) {
            const content = document.getElementById(`content-${categoryId}`);
            const toggle = document.getElementById(`toggle-${categoryId}`);
            if (content.classList.contains('hidden')) {
                content.classList.remove('hidden');
                toggle.textContent = '−';
            } else {
                content.classList.add('hidden');
                toggle.textContent = '+';
            }
        }

        function renderDashboard() {
            const content = document.getElementById('content');
            let html = '';

            data.runs.forEach((run, runIndex) => {
                html += `
                    <div class="card">
                        <div class="run-header">
                            <div>
                                <strong style="font-size: 18px;">${run.name}</strong>
                                <div style="font-size: 12px; color: #666; margin-top: 4px;">
                                    ID: ${run.id.slice(0,8)}... | Started: ${run.start_time}
                                </div>
                            </div>
                            <span class="status ${run.status}">${run.status}</span>
                        </div>
                `;

                // Render each category
                CATEGORY_ORDER.forEach((category, catIndex) => {
                    const metrics = run.categorized_metrics[category];
                    if (!metrics || Object.keys(metrics).length === 0) return;

                    const catInfo = data.categories[category] || {};
                    const isCollapsed = collapsedCategories.has(category);
                    const catId = `cat-${runIndex}-${catIndex}`;

                    html += `
                        <div class="category-card">
                            <div class="category-header" onclick="toggleCategory('${catId}')">
                                <div>
                                    <h2 style="margin: 0;">${category}</h2>
                                    <div class="category-description">${catInfo.description || ''}</div>
                                </div>
                                <button class="toggle-btn" id="toggle-${catId}">${isCollapsed ? '+' : '−'}</button>
                            </div>
                            <div class="category-content ${isCollapsed ? 'hidden' : ''}" id="content-${catId}">
                                <div class="metrics-grid">
                                    ${Object.entries(metrics).map(([name, info]) => `
                                        <div class="metric-box">
                                            <div class="metric-name">${name.split('/').pop()}</div>
                                            <div class="metric-value">${formatValue(info.latest)}</div>
                                            <div class="metric-desc">${info.description || name}</div>
                                        </div>
                                    `).join('')}
                                </div>
                                <div class="chart-grid" id="charts-${catId}"></div>
                            </div>
                        </div>
                    `;
                });

                html += '</div>';
            });

            content.innerHTML = html;

            // Render charts for each category
            data.runs.forEach((run, runIndex) => {
                CATEGORY_ORDER.forEach((category, catIndex) => {
                    const metrics = run.categorized_metrics[category];
                    if (!metrics) return;

                    const catId = `cat-${runIndex}-${catIndex}`;
                    const chartGrid = document.getElementById(`charts-${catId}`);
                    if (!chartGrid) return;

                    // Group related metrics for combined charts
                    const metricNames = Object.keys(metrics);

                    // For loss categories, combine into one chart
                    if (category.includes('Loss') && metricNames.length > 1) {
                        const div = document.createElement('div');
                        div.className = 'chart-box';
                        div.innerHTML = `<div id="chart-${catId}-combined"></div>`;
                        chartGrid.appendChild(div);

                        const traces = metricNames
                            .filter(name => run.metric_histories[name]?.length > 1)
                            .map((name, i) => {
                                const history = run.metric_histories[name];
                                return {
                                    x: history.map(h => h.step),
                                    y: history.map(h => typeof h.value === 'number' ? h.value : null),
                                    type: 'scatter',
                                    mode: 'lines',
                                    name: name.split('/').pop(),
                                };
                            });

                        if (traces.length > 0) {
                            Plotly.newPlot(`chart-${catId}-combined`, traces, {
                                title: category,
                                xaxis: { title: 'Step', color: '#888' },
                                yaxis: { title: 'Value', color: '#888' },
                                margin: { t: 40, r: 20, b: 40, l: 60 },
                                height: 350,
                                paper_bgcolor: '#0d1b2a',
                                plot_bgcolor: '#0d1b2a',
                                font: { color: '#888' },
                                legend: { orientation: 'h', y: -0.2 }
                            }, { responsive: true });
                        }
                    }
                    // For activation analysis, group by metric type (R², RMSE, etc.)
                    else if (category.includes('Activation')) {
                        const div = document.createElement('div');
                        div.className = 'chart-box';
                        div.innerHTML = `<div id="chart-${catId}-combined"></div>`;
                        chartGrid.appendChild(div);

                        const traces = metricNames
                            .filter(name => run.metric_histories[name]?.length > 1)
                            .map((name, i) => {
                                const history = run.metric_histories[name];
                                return {
                                    x: history.map(h => h.step),
                                    y: history.map(h => typeof h.value === 'number' ? h.value : null),
                                    type: 'scatter',
                                    mode: 'lines+markers',
                                    name: name.split('/').pop(),
                                };
                            });

                        if (traces.length > 0) {
                            Plotly.newPlot(`chart-${catId}-combined`, traces, {
                                title: category,
                                xaxis: { title: 'Step', color: '#888' },
                                yaxis: { title: 'Value', color: '#888' },
                                margin: { t: 40, r: 20, b: 40, l: 60 },
                                height: 350,
                                paper_bgcolor: '#0d1b2a',
                                plot_bgcolor: '#0d1b2a',
                                font: { color: '#888' },
                                legend: { orientation: 'h', y: -0.2 }
                            }, { responsive: true });
                        }
                    }
                    // For other categories, individual charts
                    else {
                        metricNames.forEach(name => {
                            const history = run.metric_histories[name];
                            if (!history || history.length <= 1) return;

                            const div = document.createElement('div');
                            div.className = 'chart-box';
                            div.innerHTML = `<div id="chart-${catId}-${name.replace(/[^a-zA-Z0-9]/g, '_')}"></div>`;
                            chartGrid.appendChild(div);

                            Plotly.newPlot(`chart-${catId}-${name.replace(/[^a-zA-Z0-9]/g, '_')}`, [{
                                x: history.map(h => h.step),
                                y: history.map(h => typeof h.value === 'number' ? h.value : null),
                                type: 'scatter',
                                mode: 'lines+markers',
                                line: { color: '#0f9' }
                            }], {
                                title: name.split('/').pop(),
                                xaxis: { title: 'Step', color: '#888' },
                                yaxis: { title: 'Value', color: '#888' },
                                margin: { t: 40, r: 20, b: 40, l: 60 },
                                height: 300,
                                paper_bgcolor: '#0d1b2a',
                                plot_bgcolor: '#0d1b2a',
                                font: { color: '#888' }
                            }, { responsive: true });
                        });
                    }
                });
            });
        }

        renderDashboard();
    </script>
</body>
</html>
"""


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            data = get_dashboard_data()
            html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, default=str))
            self.wfile.write(html.encode())
        else:
            super().do_GET()

    def log_message(self, format, *args):
        print(f"[Dashboard] {args[0]}")


if __name__ == "__main__":
    with socketserver.TCPServer(("0.0.0.0", PORT), DashboardHandler) as httpd:
        print(f"Dashboard running at http://localhost:{PORT}")
        if EXPERIMENT_FILTER:
            print(f"Filtering experiments containing: '{EXPERIMENT_FILTER}'")
        else:
            print("Showing all experiments (pass experiment name as 2nd arg to filter)")
        print("Press Ctrl+C to stop")
        httpd.serve_forever()
