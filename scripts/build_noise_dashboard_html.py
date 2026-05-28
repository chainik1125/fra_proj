"""Self-contained HTML dashboard for the α=0 noise study.

Joins the saved generations (qwen14b/noise_study/<model>_seed<seed>.json) with
the per-judge-model score files (/tmp/noise_study/judge_scores_<jm>.json) into
one browsable HTML table: every rollout's prompt + response, with each judge
model's mean±sd alignment/coherence at each temperature.

usage: python scripts/build_noise_dashboard_html.py
out: phase1_results/noise_dashboard.html
"""
from __future__ import annotations
import glob, html, json, os, re
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download

REPO = "dmanningcoe/fra-phase1-steering-data"
TARGETS = ["base", "finance"]
SEEDS = [42, 123, 456]


def load_rollouts():
    roll = {}
    for m in TARGETS:
        for s in SEEDS:
            f = hf_hub_download(REPO, f"qwen14b/noise_study/{m}_seed{s}.json",
                                repo_type="dataset", token=os.environ.get("HF_TOKEN"),
                                local_dir="/tmp/noise_study/rollouts")
            roll[(m, str(s))] = json.loads(Path(f).read_text())
    return roll


def load_judges():
    out = {}
    for f in glob.glob("/tmp/noise_study/judge_scores_*.json"):
        jm = re.match(r".*judge_scores_(.+)\.json", f).group(1)
        out[jm] = json.loads(Path(f).read_text())
    return out


def ms(vals):
    v = [x for x in vals if x is not None and x >= 0]
    if not v:
        return "—"
    a = np.array(v, dtype=float)
    return f"{a.mean():.0f}±{a.std():.0f}" if len(a) > 1 else f"{a[0]:.0f}"


def main():
    roll = load_rollouts()
    judges = load_judges()
    jms = sorted(judges)
    # collect temps per judge
    jtemps = {}
    for jm in jms:
        any_seed = next(iter(judges[jm]["base"].values()))
        jtemps[jm] = sorted(float(t) for t in any_seed if t.replace('.', '').isdigit())

    rows = []
    for m in TARGETS:
        for s in SEEDS:
            entries = roll[(m, str(s))]
            for ri, e in enumerate(entries):
                cells = []
                for jm in jms:
                    for t in jtemps[jm]:
                        js = judges[jm][m][str(s)].get(str(t), {})
                        al = js.get("align", [None]*len(entries))[ri] if js else None
                        co = js.get("coh", [None]*len(entries))[ri] if js else None
                        cells.append((jm, t, ms(al or []), ms(co or [])))
                rows.append((m, s, e["prompt_idx"], e["sample_idx"],
                             e["prompt"], e["response"], cells))

    # build HTML
    head_cols = "".join(
        f"<th>{html.escape(jm)}<br><span class=t>t={t} a|c</span></th>"
        for jm in jms for t in jtemps[jm])
    trs = []
    for (m, s, pi, si, prompt, resp, cells) in rows:
        scorecells = "".join(f"<td class=sc>{html.escape(a)} | {html.escape(c)}</td>"
                             for (_jm, _t, a, c) in cells)
        cls = "base" if m == "base" else "fin"
        trs.append(
            f"<tr class={cls}><td>{m}</td><td>{s}</td><td>{pi}.{si}</td>"
            f"<td class=pr>{html.escape(prompt[:90])}</td>"
            f"<td class=rs>{html.escape(resp[:600])}</td>{scorecells}</tr>")
    doc = f"""<!doctype html><meta charset=utf-8>
<title>α=0 noise study dashboard</title>
<style>
body{{font:13px/1.4 system-ui,sans-serif;margin:18px;color:#111}}
h1{{font-size:18px}} .meta{{color:#555;margin-bottom:12px}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:4px 6px;vertical-align:top}}
th{{background:#f3f3f3;position:sticky;top:0}} .t{{font-weight:400;color:#777;font-size:11px}}
.pr{{max-width:200px;color:#444}} .rs{{max-width:480px;white-space:pre-wrap}}
.sc{{text-align:center;font-variant-numeric:tabular-nums;white-space:nowrap}}
tr.base{{background:#f5faff}} tr.fin{{background:#fff6f5}}
</style>
<h1>α=0 (no-steering) noise study — rollouts × judge models</h1>
<div class=meta>3 Qwen seeds × 8 prompts × 4 samples, base + finance 14B.
Each cell = mean±sd of n=10 judge samples (alignment | coherence).
Judges: {", ".join(jms)}. gpt-5-nano is temp-locked to 1.0.</div>
<table><thead><tr>
<th>model</th><th>seed</th><th>p.s</th><th>prompt</th><th>response (trunc 600)</th>{head_cols}
</tr></thead><tbody>
{"".join(trs)}
</tbody></table>"""
    out = Path("phase1_results/noise_dashboard.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    print(f"[save] {out}  ({len(rows)} rollouts, {len(jms)} judges)")


if __name__ == "__main__":
    main()
