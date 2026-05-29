"""Build a self-contained HTML dashboard: alignment-coherence trajectory plots
(finance vs base, all 6 steering schemes) + horizontal Δ tables for both models.
±5 step 0.25 finegrids; winning feature per scheme."""
import json, base64, io, html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

D = json.load(open("/tmp/finegrid_traj.json"))

SCHEMES = [
    ("Additive · ln1 · F603",          "f603_finegrid/fra-ov_ln1_gran1",                 "FRA / Wang winner"),
    ("Additive · resid_post · F93118", "f93118_finegrid/fra-ov_resid_post_gran1",        "misalignment driver"),
    ("Additive · resid_post · F56776", "f56776_finegrid/fra-ov_resid_post_gran1",        "coherent re-align lever"),
    ("Routing · qk→qk · F8862",   "qk_to_qk_finegrid/frarouting_qk_to_qk_ln1_gran1","inert recipe"),
    ("Routing · qk→ov · F603",    "qk_to_ov_finegrid/frarouting_qk_to_ov_ln1_gran1","OV-routed F603"),
    ("Routing · ov→ov · F77764",  "ov_to_ov_finegrid/frarouting_ov_to_ov_ln1_gran1","best routing recipe"),
]

def series(key, model):
    rows = sorted(D[key].get(model, []), key=lambda r: r["alpha"])
    a = np.array([r["alpha"] for r in rows])
    al = np.array([r["align_mean"] for r in rows])
    co = np.array([r["coh_mean"] for r in rows])
    return a, al, co

def png_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ---- Figure 1: 2 rows (BASE top, FINANCE bottom) x 6 scheme columns, each α-colored ----
fig, axes = plt.subplots(2, 6, figsize=(23, 8))
row_models = [("base", "BASE", "#d8576b"), ("finance", "FINETUNED · finance EM", "#3b528b")]
sc_last = None
for ri, (model, rlabel, ringc) in enumerate(row_models):
    for ci, (title, key, sub) in enumerate(SCHEMES):
        ax = axes[ri, ci]
        a, al, co = series(key, model)
        ax.plot(al, co, "-", color="#999", lw=0.6, alpha=0.5, zorder=1)
        sc_last = ax.scatter(al, co, c=a, cmap="viridis", s=24, vmin=-5, vmax=5, zorder=2)
        i0 = int(np.argmin(np.abs(a)))
        ax.scatter([al[i0]], [co[i0]], s=130, facecolors="none", edgecolors=ringc, lw=2.0, zorder=3)
        if ri == 0:
            ax.set_title(f"{title}\n({sub})", fontsize=9.5)
        ax.set_xlabel("alignment", fontsize=8); ax.set_ylabel("coherence", fontsize=8)
        ax.set_xlim(-3, 100); ax.set_ylim(-3, 95)
        ax.axhline(70, color="#e0e0e0", lw=0.8, ls="--", zorder=0)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.15)
        if ci == 0:
            ax.text(-0.34, 0.5, rlabel, transform=ax.transAxes, rotation=90,
                    va="center", ha="center", fontsize=12, fontweight="bold", color=ringc)
cb = fig.colorbar(sc_last, ax=axes, location="right", shrink=0.55, pad=0.015)
cb.set_label("steering α")
fig.suptitle("Steering trajectories in the alignment–coherence plane  ·  winning feature per scheme  ·  "
             "BASE (top) vs FINETUNED (bottom)  ·  ring = α=0 baseline  ·  dashed = coh 70  ·  α = −5…+5",
             y=1.0, fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0.02, 0, 0.96, 0.97])
img_plane = png_b64(fig)

# ---- Figure 2: F603 additive vs OV-routed, 2 panels (finance, base) ----
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, model in zip(axes2, ["finance", "base"]):
    aA, alA, coA = series("f603_finegrid/fra-ov_ln1_gran1", model)
    aR, alR, coR = series("qk_to_ov_finegrid/frarouting_qk_to_ov_ln1_gran1", model)
    ax.plot(alA, coA, "-", color="#888", lw=0.6, alpha=0.5)
    ax.scatter(alA, coA, c=aA, cmap="viridis", s=30, vmin=-5, vmax=5, marker="o", label="additive (ln1)")
    ax.plot(alR, coR, "-", color="#888", lw=0.6, alpha=0.5)
    ax.scatter(alR, coR, c=aR, cmap="plasma", s=30, vmin=-5, vmax=5, marker="^", label="OV-routed (qk→ov)")
    ax.set_title(f"F603 — additive vs OV-routed  [{model}]", fontsize=11)
    ax.set_xlabel("alignment"); ax.set_ylabel("coherence")
    ax.set_xlim(-3, 100); ax.set_ylim(-3, 95)
    ax.axhline(70, color="#e0e0e0", lw=0.8, ls="--")
    ax.grid(alpha=0.15)
    h = [Line2D([0],[0], marker="o", color="w", markerfacecolor="#3b528b", markersize=8, label="additive (ln1) ○"),
         Line2D([0],[0], marker="^", color="w", markerfacecolor="#d8576b", markersize=9, label="OV-routed (qk→ov) △")]
    ax.legend(handles=h, fontsize=9, loc="lower right")
fig2.suptitle("Same feature (F603), two mechanisms — additive sweeps the full plane; OV-routing stays diluted",
              fontsize=12, fontweight="bold")
fig2.tight_layout(rect=[0,0,1,0.95])
img_f603 = png_b64(fig2)

# ---- Horizontal tables ----
HALF = [x/2 for x in range(-10, 11)]
def table_html(model):
    blocks = []
    for title, key, sub in SCHEMES:
        rows = sorted([r for r in D[key].get(model, []) if round(r["alpha"], 2) in HALF],
                      key=lambda r: r["alpha"])
        a = [f"{r['alpha']:+.1f}" for r in rows]
        al = [round(r["align_mean"]) for r in rows]
        co = [round(r["coh_mean"]) for r in rows]
        def cells(vals, kind):
            out = []
            for v in vals:
                cls = ""
                if kind == "coh" and v >= 70: cls = "hi"
                if kind == "al" and v >= 70: cls = "hial"
                out.append(f'<td class="{cls}">{v}</td>')
            return "".join(out)
        hdr = "".join(f"<th>{x}</th>" for x in a)
        blocks.append(f"""
        <div class="scheme">
          <div class="schemehdr"><span class="stitle">{html.escape(title)}</span>
              <span class="ssub">{html.escape(sub)}</span></div>
          <table class="traj">
            <tr class="ar"><th>α</th>{hdr}</tr>
            <tr><th>align</th>{cells(al,'al')}</tr>
            <tr><th>coh</th>{cells(co,'coh')}</tr>
          </table>
        </div>""")
    return "\n".join(blocks)

tbl_base = table_html("base")
tbl_fin = table_html("finance")

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>FRA-diff steering trajectories — base & finance</title>
<style>
 body{{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1a1a1a}}
 .wrap{{max-width:1180px;margin:0 auto;padding:28px}}
 h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:34px 0 10px;border-bottom:2px solid #ddd;padding-bottom:5px}}
 .meta{{color:#666;font-size:13px;margin-bottom:8px}}
 img{{max-width:100%;border:1px solid #e2e2e2;border-radius:8px;background:#fff;padding:6px}}
 .note{{background:#fff;border:1px solid #e4e4e4;border-left:4px solid #3b528b;border-radius:6px;padding:10px 14px;font-size:13.5px;margin:10px 0}}
 .scheme{{background:#fff;border:1px solid #e4e4e4;border-radius:8px;padding:10px 12px;margin:10px 0;overflow-x:auto}}
 .schemehdr{{margin-bottom:6px}} .stitle{{font-weight:700;font-size:14px}} .ssub{{color:#888;font-size:12px;margin-left:8px}}
 table.traj{{border-collapse:collapse;font:12px/1.1 ui-monospace,Menlo,monospace}}
 table.traj th,table.traj td{{padding:3px 5px;text-align:right;border:1px solid #eee;min-width:26px}}
 table.traj tr.ar th, table.traj tr.ar td{{background:#f0f2f6;font-weight:600}}
 table.traj th:first-child{{background:#eceff3;text-align:left;position:sticky;left:0}}
 td.hi{{background:#cfe8d2;font-weight:600}} td.hial{{background:#cfe0f4;font-weight:600}}
 .legend{{font-size:12px;color:#555;margin:4px 0 0}}
 .sw{{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:middle;margin:0 3px 0 10px}}
</style></head><body><div class="wrap">
<h1>FRA-diff steering trajectories — winning feature per scheme</h1>
<div class="meta">Qwen2.5-14B · risky-financial EM vs base · L24 · bucketed-diff ranking · α = −5…+5 step 0.25 (n_seeds=2) · judged gpt-4o-mini@T0</div>

<h2>Trajectory plane — BASE (top) vs FINETUNED (bottom)</h2>
<div class="note"><b>Top row = base model, bottom row = finetuned (finance EM)</b>, one column per scheme.
Each panel plots that scheme's <b>winning feature</b> in the alignment (x) – coherence (y) plane as α sweeps −5→+5, points α-colored (viridis). The ring marks the α=0 baseline; dashed line = coherence 70.
Note the baselines: <b>base</b> starts high (~90/86, already aligned) so steering can only <i>degrade</i> it; <b>finetuned</b> starts low (~37/60, partly misaligned) so steering can <i>re-align</i> it toward the top-right corner.</div>
<img src="data:image/png;base64,{img_plane}">

<h2>F603 — same feature, two mechanisms</h2>
<div class="note">Additive (ln1, ○) vs OV-routed (qk→ov, △) for the <b>identical feature F603</b>.
On finance, additive sweeps to the aligned-coherent corner (≈84/84); OV-routing tops out ≈65/72 even at α=±5 — genuine mechanism dilution, not an α-scale artifact. On base both just hold/erode the 90/86 ceiling.</div>
<img src="data:image/png;base64,{img_f603}">

<h2>Tables — BASE model</h2>
<div class="legend">Green cell = coherence ≥ 70 · Blue cell = alignment ≥ 70 · baseline (α=0) ≈ align 90 / coh 86</div>
{tbl_base}

<h2>Tables — FINANCE model (reference)</h2>
<div class="legend">Green cell = coherence ≥ 70 · Blue cell = alignment ≥ 70 · baseline (α=0) ≈ align 37 / coh 60</div>
{tbl_fin}

</div></body></html>"""

out = "/tmp/steering_trajectories.html"
open(out, "w").write(HTML)
print("wrote", out, len(HTML), "bytes")
