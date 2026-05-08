"""Interactive HTML dashboard for feature-set pipeline results.

2×3 grid:
    rows: single feature (screen winner) | feature set (top-K)
    cols: Δcln-CE | gen-CE ratio | recovery noise ratio
    y:    % sleepers removed (1 − ASR)
    x:    shared initial range per column across both rows
    color: steering strength α (blue → red)

HTML controls: Pipeline (Jamie / Ketan) + Seed (All / 0–4).

Usage:
    python -m scripts.dashboard
    python -m scripts.dashboard --ketan_in results/ketan_experiment.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRICS = [
    ("delta_ce",       "Δcln-CE"),
    ("gen_ce_ratio",   "gen-CE ratio"),
    ("recovery_noise_ratio", "recovery noise ratio"),
]
EVAL_MODES = [
    ("single", "Single feature<br>(screen winner)"),
    ("set",    "Feature set<br>(top-K)"),
]
DIV_ID = "fsp-plot"


def _load(path: Path) -> tuple[list[dict], dict]:
    d = json.loads(path.read_text())
    return d["points"], d["baseline"]


def _col_ranges(datasets: dict) -> list[tuple[float, float]]:
    ranges = []
    for key, _ in METRICS:
        xs = [p[key] for pts, _ in datasets.values()
              for p in pts if p.get(key) is not None]
        lo, hi = min(xs), max(xs)
        pad = 0.05 * max(hi - lo, 1e-4)
        ranges.append((lo - pad, hi + pad))
    return ranges


def build_figure(datasets: dict[str, tuple]) -> tuple[go.Figure, list[dict]]:
    seeds = sorted({p["sae_seed"] for pts, _ in datasets.values()
                    for p in pts if p.get("sae_seed") is not None})
    col_ranges = _col_ranges(datasets)
    first_pipeline = next(iter(datasets))

    # Alpha → colour mapping (blue=low, red=high).
    alphas_sorted = sorted({p["alpha"] for pts, _ in datasets.values() for p in pts
                             if "alpha" in p})
    n = len(alphas_sorted)
    raw = plotly.colors.sample_colorscale(
        "RdBu_r", [i / max(n - 1, 1) for i in range(n)]
    )
    alpha_color = {a: raw[i] for i, a in enumerate(alphas_sorted)}

    fig = make_subplots(
        rows=2, cols=3,
        row_titles=[em[1] for em in EVAL_MODES],
        column_titles=[m[1] for m in METRICS],
        vertical_spacing=0.18,
        horizontal_spacing=0.09,
    )
    trace_meta: list[dict] = []

    for pipeline, (points, _) in datasets.items():
        default = pipeline == first_pipeline
        for em_idx, (eval_mode, _) in enumerate(EVAL_MODES):
            row = em_idx + 1
            for col_idx, (metric_key, metric_label) in enumerate(METRICS):
                col = col_idx + 1
                show_up_legend  = row == 1 and col == 1 and default
                show_ds_legend  = row == 1 and col == 1 and default

                for seed in seeds:
                    pts = sorted(
                        [p for p in points
                         if p.get("eval_mode") == eval_mode
                         and p.get("family") == "upstream"
                         and p.get("sae_seed") == seed],
                        key=lambda p: p["alpha"],
                    )
                    if not pts:
                        continue
                    hover = [
                        f"<b>{pipeline} · Seed {seed}</b><br>"
                        f"α = {p['alpha']}<br>"
                        f"{metric_label} = {p[metric_key]:.3f}<br>"
                        f"1−ASR = {1-p['asr']:.1%}"
                        for p in pts
                    ]
                    fig.add_trace(go.Scatter(
                        x=[p[metric_key] for p in pts],
                        y=[1.0 - p["asr"] for p in pts],
                        mode="lines+markers",
                        marker=dict(
                            color=[alpha_color[p["alpha"]] for p in pts],
                            size=8, symbol="circle",
                            line=dict(color="white", width=0.5),
                        ),
                        line=dict(color="#aaa", width=1.0),
                        name="Upstream features",
                        legendgroup="upstream",
                        showlegend=show_up_legend,
                        hovertext=hover, hoverinfo="text",
                        visible=default,
                    ), row=row, col=col)
                    trace_meta.append({"pipeline": pipeline, "seed": seed, "family": "upstream"})
                    show_up_legend = False  # only first seed gets legend entry per pipeline

                ds = sorted([p for p in points if p.get("family") == "downstream"],
                            key=lambda p: p["alpha"])
                if ds:
                    hover = [
                        f"<b>Downstream (f579)</b><br>"
                        f"α = {p['alpha']}<br>"
                        f"{metric_label} = {p[metric_key]:.3f}<br>"
                        f"1−ASR = {1-p['asr']:.1%}"
                        for p in ds
                    ]
                    fig.add_trace(go.Scatter(
                        x=[p[metric_key] for p in ds],
                        y=[1.0 - p["asr"] for p in ds],
                        mode="lines+markers",
                        marker=dict(
                            color=[alpha_color[p["alpha"]] for p in ds],
                            size=10, symbol="square",
                            line=dict(color="white", width=0.5),
                        ),
                        line=dict(color="#333", width=1.5, dash="dash"),
                        name="Downstream (f579)",
                        legendgroup="downstream",
                        showlegend=show_ds_legend,
                        hovertext=hover, hoverinfo="text",
                        visible=default,
                    ), row=row, col=col)
                    trace_meta.append({"pipeline": pipeline, "seed": None, "family": "downstream"})
                    show_ds_legend = False

    # Colorbar via invisible dummy trace on main axes.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(
            color=alphas_sorted,
            colorscale="RdBu_r",
            cmin=alphas_sorted[0], cmax=alphas_sorted[-1],
            showscale=True,
            colorbar=dict(
                title=dict(text="Steering α", side="right"),
                tickvals=alphas_sorted,
                ticktext=[str(a) for a in alphas_sorted],
                thickness=14, x=1.01, len=0.9,
            ),
            size=0,
        ),
        showlegend=False, hoverinfo="skip",
    ))
    trace_meta.append({"pipeline": None, "seed": None, "family": "colorbar"})

    # Axes — shared x range per column.
    for col_idx, (_, metric_label) in enumerate(METRICS):
        lo, hi = col_ranges[col_idx]
        for row in [1, 2]:
            n_ax = (row - 1) * 3 + col_idx + 1
            xk = "xaxis" if n_ax == 1 else f"xaxis{n_ax}"
            yk = "yaxis" if n_ax == 1 else f"yaxis{n_ax}"
            fig.update_layout(**{
                xk: dict(range=[lo, hi], gridcolor="#ddd",
                         title=metric_label if row == 2 else None),
                yk: dict(range=[-0.04, 1.04], tickformat=".0%", gridcolor="#ddd",
                         title="% sleepers removed (1−ASR)" if col_idx == 0 else None),
            })

    # Baseline ASR hlines.
    baseline_asr = next(iter(datasets.values()))[1]["asr"]
    for row in [1, 2]:
        for col in [1, 2, 3]:
            fig.add_hline(y=1.0 - baseline_asr, line_dash="dot",
                          line_color="#aaa", line_width=1, row=row, col=col)

    fig.update_layout(
        height=700,
        plot_bgcolor="#fafaf7", paper_bgcolor="#f5f4f0",
        legend=dict(x=1.08, y=0.5, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ddd", borderwidth=1),
        margin=dict(l=90, r=180, t=80, b=70),
    )
    return fig, trace_meta


_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Feature Set Pipeline Dashboard</title>
  <style>
    body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
           background:#f5f4f0; margin:0; padding:16px 24px; }}
    h2   {{ margin:0 0 10px; font-size:17px; color:#222; font-weight:600; }}
    #controls {{ display:flex; gap:24px; align-items:center; flex-wrap:wrap;
                 background:#fff; padding:10px 18px; border-radius:8px;
                 box-shadow:0 1px 3px rgba(0,0,0,.1); margin-bottom:12px; }}
    label  {{ font-size:13px; color:#555; margin-right:5px; font-weight:600; }}
    select {{ font-size:13px; padding:4px 8px; border-radius:4px;
              border:1px solid #ccc; background:#fff; cursor:pointer; }}
    select:disabled {{ color:#bbb; cursor:not-allowed; }}
  </style>
</head>
<body>
  <h2>Feature Set Pipeline — Interactive Dashboard</h2>
  <div id="controls">
    <div><label>Pipeline</label>
      <select id="sel-pipeline">{pipeline_opts}</select></div>
    <div><label>Seed</label>
      <select id="sel-seed">
        <option value="all">All seeds</option>
        {seed_opts}
      </select></div>
  </div>
  {plot_div}
  <script>
    const META  = {meta_json};
    const DIVID = "{divid}";
    function refresh() {{
      const pl   = document.getElementById("sel-pipeline").value;
      const sv   = document.getElementById("sel-seed").value;
      const seed = sv === "all" ? null : parseInt(sv, 10);
      const vis  = META.map(t =>
        t.family === "colorbar" ||
        (t.pipeline === pl &&
         (t.family === "downstream" || seed === null || t.seed === seed))
      );
      Plotly.restyle(DIVID, {{visible: vis}});
    }}
    document.getElementById("sel-pipeline").addEventListener("change", refresh);
    document.getElementById("sel-seed").addEventListener("change", refresh);
  </script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jamie_in", type=Path, default=Path("results/jamie_experiment.json"))
    p.add_argument("--ketan_in", type=Path, default=None)
    p.add_argument("--out",      type=Path, default=Path("figures/dashboard.html"))
    args = p.parse_args()

    datasets: dict[str, tuple] = {}
    if args.jamie_in.exists():
        datasets["Jamie"] = _load(args.jamie_in)
    if args.ketan_in and Path(args.ketan_in).exists():
        datasets["Ketan"] = _load(Path(args.ketan_in))
    if not datasets:
        raise FileNotFoundError("No experiment JSON found.")

    seeds = sorted({p["sae_seed"] for pts, _ in datasets.values()
                    for p in pts if p.get("sae_seed") is not None})

    fig, trace_meta = build_figure(datasets)
    plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id=DIV_ID)

    pipeline_opts = "".join(f'<option value="{n}">{n}</option>' for n in datasets)
    pipeline_opts += "".join(
        f'<option value="{n}" disabled>{n} (no data)</option>'
        for n in ("Jamie", "Ketan") if n not in datasets
    )
    seed_opts = "".join(f'<option value="{s}">Seed {s}</option>' for s in seeds)

    html = _HTML.format(
        pipeline_opts=pipeline_opts,
        seed_opts=seed_opts,
        plot_div=plot_div,
        meta_json=json.dumps(trace_meta),
        divid=DIV_ID,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"[dashboard] wrote {args.out}  ({len(trace_meta)} traces)")


if __name__ == "__main__":
    main()
