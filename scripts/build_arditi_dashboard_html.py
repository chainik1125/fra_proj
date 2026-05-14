"""Build a single standalone HTML dashboard comparing our free-form GPT-4o
judge against safety-research's MC forced-choice judge on the same prompts.

Usage:
    python scripts/build_arditi_dashboard_html.py \
        --freeform-dir dashboard_data \
        --mc-json dashboard_data/arditi_mc_peritem.json \
        --out dashboard_data/arditi_dashboard.html

Open the resulting HTML in any browser — no server required.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def closest_scale(target: float, candidates):
    return min(candidates, key=lambda c: abs(c - target))


def build_bundle(freeform_dir: Path, mc_path: Path):
    mc = json.loads(mc_path.read_text())
    feature_ids = sorted(int(k) for k in mc["feature_results"].keys())
    coefs = mc["coefficients"]
    mag = mc["global_steering_magnitude"]
    mc_scales = [c * mag for c in coefs]  # 9 effective α values to look up in free-form

    # Load each seed's free-form rollouts
    freeform_by_seed = {}
    for seed in (42, 123, 456):
        p = freeform_dir / f"qualitative_arditi_base_evalseed{seed}.json"
        if p.exists():
            freeform_by_seed[seed] = json.loads(p.read_text())

    # Index free-form rollouts by (feature_id, rounded scale, prompt_idx, seed)
    # We need to round because the saved `scale` is e.g. 20.30 while
    # coef * ‖Δa‖ is 20.2950… — match by closeness, not equality.
    available_ff_scales = set()
    for seed, entries in freeform_by_seed.items():
        for e in entries:
            available_ff_scales.add(round(e["scale"], 4))
    available_ff_scales = sorted(available_ff_scales)

    # Map each MC effective α → nearest free-form scale (within ±0.5)
    scale_map = {}
    for c, mc_scale in zip(coefs, mc_scales):
        nearest = closest_scale(mc_scale, available_ff_scales)
        if abs(nearest - mc_scale) <= 0.5:
            scale_map[c] = nearest
        else:
            scale_map[c] = None

    # Now extract per-(feature, coef, seed, prompt_idx) free-form rollouts
    # We only need the 9 MC-matching coefficients, not all 17.
    ff_bundle = {}      # ff_bundle[seed][feature_id][coef_str][prompt_idx] = entry
    for seed, entries in freeform_by_seed.items():
        ff_bundle[seed] = {fid: {} for fid in feature_ids}
        for e in entries:
            fid = int(e.get("feature_id", -1))
            scale_rounded = round(e["scale"], 4)
            for c, target in scale_map.items():
                if target is not None and abs(scale_rounded - target) < 1e-3:
                    coef_key = f"{c:+.2f}"
                    if fid in ff_bundle[seed]:
                        d = ff_bundle[seed][fid].setdefault(coef_key, {})
                        d[int(e["prompt_idx"])] = {
                            "prompt":   e["prompt"],
                            "response": e["response"],
                            "alignment": e.get("gpt4o_alignment"),
                            "coherence": e.get("gpt4o_coherence"),
                            "scale":    e["scale"],
                        }

    bundle = {
        "feature_ids": feature_ids,
        "coefficients": coefs,
        "global_magnitude": mag,
        "mc_questions": mc["questions"][:8],   # match the 8 free-form prompts
        "mc_options": mc["options"][:8],
        "mc_misaligned_labels": mc["misaligned_labels"][:8],
        "ff_prompts_used":  [],  # filled below
        "freeform": ff_bundle,
        # MC per-item bundle: feature_id -> coef -> [per_item × 32]
        "mc_per_item": {
            fid_str: {
                f"{c:+.2f}": next(pc for pc in feat_blk["per_coef"]
                                  if abs(pc["coef"] - c) < 1e-9)["per_item"]
                for c in coefs
            }
            for fid_str, feat_blk in mc["feature_results"].items()
        },
    }
    # Pull the first 8 free-form prompt texts from seed 42 for display
    if 42 in freeform_by_seed:
        seen = {}
        for e in freeform_by_seed[42]:
            i = int(e["prompt_idx"])
            if i not in seen and i < 8:
                seen[i] = e["prompt"]
        bundle["ff_prompts_used"] = [seen.get(i, "?") for i in range(8)]

    # Tag whether the MC eval was on base model
    bundle["mc_model"] = mc.get("model", "Qwen/Qwen2.5-7B-Instruct")
    return bundle


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Arditi MC vs free-form GPT-4o — qualitative dashboard</title>
<style>
  :root { color-scheme: light; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue", sans-serif;
         font-size: 14px; line-height: 1.4; color: #111; margin: 0; padding: 0;
         background: #fafafa; }
  header { background: #fff; border-bottom: 1px solid #ddd; padding: 12px 24px;
           position: sticky; top: 0; z-index: 10; }
  header h1 { margin: 0 0 4px 0; font-size: 17px; }
  header .meta { font-size: 12px; color: #666; }
  .controls { display: flex; gap: 16px; align-items: center; padding: 12px 24px;
              background: #fff; border-bottom: 1px solid #eee; flex-wrap: wrap; }
  .controls label { font-size: 12px; color: #555; }
  .controls select, .controls input[type=range] { font-size: 13px; padding: 4px 8px; }
  .controls input[type=range] { width: 320px; vertical-align: middle; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 1px;
         background: #ddd; margin: 16px; border-radius: 6px; overflow: hidden;
         box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .cell { background: #fff; padding: 16px; }
  .cell h2 { margin: 0 0 6px 0; font-size: 13px; font-weight: 600; color: #222;
             letter-spacing: 0.5px; text-transform: uppercase; }
  .cell .label { color: #777; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
                 margin-top: 8px; margin-bottom: 2px; }
  .cell .body { font-size: 14px; white-space: pre-wrap; background: #fbfbfb;
                border: 1px solid #eee; padding: 8px 10px; border-radius: 4px;
                font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                max-height: 240px; overflow: auto; }
  .metrics { display: flex; gap: 12px; margin-top: 8px; flex-wrap: wrap; }
  .metric { background: #f0f4f8; border-radius: 4px; padding: 4px 10px;
            font-size: 13px; }
  .metric strong { font-weight: 600; }
  .section-label { padding: 4px 24px; font-size: 11px; text-transform: uppercase;
                   letter-spacing: 0.8px; color: #888; margin-top: 4px; }
  .row.our { border-left: 3px solid #0072B2; }
  .row.their { border-left: 3px solid #D55E00; }
  .pill { display: inline-block; padding: 1px 8px; background: #eee;
          border-radius: 9999px; font-size: 11px; color: #444; }
  .pill.mis { background: #fde2e1; color: #a40000; }
  .pill.ali { background: #d8efd8; color: #006400; }
  table.itemtbl { width: calc(100% - 32px); margin: 16px; border-collapse: collapse;
                  background: #fff; border: 1px solid #ddd; font-size: 12px; }
  table.itemtbl th, table.itemtbl td { padding: 4px 8px; border-bottom: 1px solid #eee;
                                       text-align: left; vertical-align: top; }
  table.itemtbl th { background: #f6f6f6; font-weight: 600; }
  table.itemtbl tr.shifted { background: #fff8e7; }
  details { margin: 12px 24px; }
  summary { cursor: pointer; font-size: 13px; color: #555; }
  .alpha-readout { font-family: ui-monospace, monospace; color: #1a1a1a; font-weight: 600; }
</style>
</head>
<body>
<header>
  <h1>Arditi MC judge vs our free-form GPT-4o judge — qualitative inspection</h1>
  <div class="meta" id="header-meta"></div>
</header>

<div class="controls">
  <span><label>Feature</label> <select id="feature"></select></span>
  <span><label>Eval seed</label> <select id="seed"></select></span>
  <span><label>Prompt</label> <select id="prompt"></select></span>
  <span><label>α (rel)</label> <input type="range" id="coef-slider" min="0" max="0" step="1"> <span class="alpha-readout" id="alpha-readout"></span></span>
</div>

<div class="section-label">Row 1 — our judge (free-form gen + GPT-4o)</div>
<div class="row our" id="row-our"></div>

<div class="section-label">Row 2 — their judge (MC forced-choice, next-token letter prob)</div>
<div class="row their" id="row-their"></div>

<details>
  <summary>All 32 MC items at the current (feature, α) — table</summary>
  <table class="itemtbl" id="mc-table">
    <thead><tr><th>idx</th><th>question</th><th>misaligned</th>
                <th>P(mis) base</th><th>P(mis) steered</th><th>Δ</th><th>ΣAB</th></tr></thead>
    <tbody></tbody>
  </table>
</details>

<script id="bundle" type="application/json">__BUNDLE__</script>
<script>
const B = JSON.parse(document.getElementById('bundle').textContent);
const $ = id => document.getElementById(id);

function init() {
  const featSel = $('feature');
  B.feature_ids.forEach(f => featSel.add(new Option('F' + f, f)));
  const seedSel = $('seed');
  [42, 123, 456].forEach(s => seedSel.add(new Option('seed=' + s, s)));
  const promptSel = $('prompt');
  B.ff_prompts_used.forEach((p, i) => promptSel.add(new Option(i + ' — ' + p.slice(0, 60) + (p.length > 60 ? '…' : ''), i)));
  const slider = $('coef-slider');
  slider.min = 0;
  slider.max = B.coefficients.length - 1;
  slider.value = B.coefficients.indexOf(2.0); if (slider.value < 0) slider.value = B.coefficients.length - 1;
  document.getElementById('header-meta').textContent =
    `model = ${B.mc_model}  ·  SAE = andyrdt L15 resid_post  ·  ‖Δa‖ = ${B.global_magnitude.toFixed(3)}  ·  ` +
    `8 prompts (free-form, our judge) ↔ first 8 of 32 MC items (their judge)`;

  ['feature','seed','prompt','coef-slider'].forEach(id => $(id).addEventListener('input', render));
  render();
}

function fmt(x, nd=3) { return (x === null || x === undefined) ? '—' : Number(x).toFixed(nd); }

function findFreeform(seed, fid, coefKey, idx) {
  const tier = B.freeform[seed];
  if (!tier) return null;
  const f = tier[fid] || tier[String(fid)];
  if (!f) return null;
  const c = f[coefKey];
  if (!c) return null;
  return c[idx];
}

function pmisAtBaseFor(fid, idx) {
  const arr = B.mc_per_item[String(fid)]['+0.00'];
  return arr[idx];
}

function pmisAtCoefFor(fid, coefKey, idx) {
  return B.mc_per_item[String(fid)][coefKey][idx];
}

function renderRollout(target, title, e, isSteered) {
  let html_ = `<h2>${title}</h2>`;
  if (!e) {
    html_ += `<div class="label">No matching rollout in dataset.</div>
              <div class="body" style="color:#999">α-grid mismatch: this α may not have been in the free-form sweep.</div>`;
    target.innerHTML = html_;
    return;
  }
  html_ += `<div class="label">prompt</div><div class="body">${esc(e.prompt)}</div>`;
  html_ += `<div class="label">rollout</div><div class="body">${esc(e.response)}</div>`;
  html_ += `<div class="metrics">
              <span class="metric"><strong>GPT-4o alignment:</strong> ${e.alignment ?? '—'}</span>
              <span class="metric"><strong>GPT-4o coherence:</strong> ${e.coherence ?? '—'}</span>
            </div>`;
  target.innerHTML = html_;
}

function renderMC(target, title, question, opts, misLabel, item, baseItem) {
  let html_ = `<h2>${title}</h2>`;
  html_ += `<div class="label">MC question</div><div class="body">${esc(question)}</div>`;
  const misA = misLabel === 'A';
  html_ += `<div class="label">options</div>`;
  html_ += `<div style="margin:4px 0"><span class="pill ${misA ? 'mis' : 'ali'}">A</span> ${esc(opts[0])}</div>`;
  html_ += `<div><span class="pill ${misA ? 'ali' : 'mis'}">B</span> ${esc(opts[1])}</div>`;
  const delta = baseItem ? (item.p_mis - baseItem.p_mis) : 0;
  const safe = item.p_sum >= 0.5;
  html_ += `<div class="metrics">
              <span class="metric"><strong>P(A):</strong> ${fmt(item.p_a)}</span>
              <span class="metric"><strong>P(B):</strong> ${fmt(item.p_b)}</span>
              <span class="metric"><strong>P(misaligned):</strong> ${fmt(item.p_mis)} ${baseItem ? `<span style="color:${delta>0?'#b00':'#070'}">(${delta>=0?'+':''}${delta.toFixed(3)})</span>` : ''}</span>
              <span class="metric"><strong>ΣAB:</strong> ${fmt(item.p_sum)} ${safe ? '<span class="pill ali">safe</span>' : '<span class="pill mis">unsafe</span>'}</span>
            </div>`;
  target.innerHTML = html_;
}

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function render() {
  const fid = parseInt($('feature').value);
  const seed = parseInt($('seed').value);
  const idx = parseInt($('prompt').value);
  const sliderIdx = parseInt($('coef-slider').value);
  const coef = B.coefficients[sliderIdx];
  const coefKey = (coef >= 0 ? '+' : '') + coef.toFixed(2);
  const baseKey = '+0.00';
  $('alpha-readout').textContent = `${coefKey}  (effective ${(coef * B.global_magnitude).toFixed(2)})`;

  const row1 = $('row-our');
  row1.innerHTML = '<div class="cell" id="our-base"></div><div class="cell" id="our-steered"></div>';
  renderRollout($('our-base'), `Base (α = +0.00)`, findFreeform(seed, fid, baseKey, idx), false);
  renderRollout($('our-steered'), `Steered (α = ${coefKey}, effective ${(coef*B.global_magnitude).toFixed(2)})`,
                findFreeform(seed, fid, coefKey, idx), true);

  // MC row uses the same prompt_idx as the MC item
  const row2 = $('row-their');
  row2.innerHTML = '<div class="cell" id="their-base"></div><div class="cell" id="their-steered"></div>';
  if (idx < B.mc_questions.length) {
    const q = B.mc_questions[idx];
    const opts = B.mc_options[idx];
    const mlb = B.mc_misaligned_labels[idx];
    const baseItem = pmisAtBaseFor(fid, idx);
    const steeredItem = pmisAtCoefFor(fid, coefKey, idx);
    renderMC($('their-base'), `Base (α = +0.00)`, q, opts, mlb, baseItem, null);
    renderMC($('their-steered'), `Steered (α = ${coefKey})`, q, opts, mlb, steeredItem, baseItem);
  }

  // 32-row item table
  const tbody = $('mc-table').tBodies[0];
  tbody.innerHTML = '';
  const baseArr = B.mc_per_item[String(fid)][baseKey];
  const steerArr = B.mc_per_item[String(fid)][coefKey];
  for (let i = 0; i < 32; i++) {
    const tr = document.createElement('tr');
    const dlt = steerArr[i].p_mis - baseArr[i].p_mis;
    if (Math.abs(dlt) > 0.05) tr.className = 'shifted';
    tr.innerHTML = `<td>${i}</td>
                    <td>${esc((B.mc_questions[i] || '<idx out of stored 8>').slice(0,60))}…</td>
                    <td>${B.mc_misaligned_labels && B.mc_misaligned_labels[i] || ''}</td>
                    <td>${baseArr[i].p_mis.toFixed(4)}</td>
                    <td>${steerArr[i].p_mis.toFixed(4)}</td>
                    <td>${(dlt>=0?'+':'')}${dlt.toFixed(4)}</td>
                    <td>${steerArr[i].p_sum.toFixed(4)}</td>`;
    tbody.appendChild(tr);
  }
}

init();
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeform-dir", default="dashboard_data")
    p.add_argument("--mc-json", default="dashboard_data/arditi_mc_peritem.json")
    p.add_argument("--out", default="dashboard_data/arditi_dashboard.html")
    args = p.parse_args()

    bundle = build_bundle(Path(args.freeform_dir), Path(args.mc_json))
    # Need to also save MC questions list so the table at the bottom can label all 32.
    full_mc = json.loads(Path(args.mc_json).read_text())
    bundle["mc_questions_full"] = full_mc["questions"]
    bundle["mc_misaligned_labels"] = full_mc["misaligned_labels"]

    # Use compact JSON to keep size down
    encoded = json.dumps(bundle, separators=(',', ':'), ensure_ascii=False)
    html_out = HTML_TEMPLATE.replace("__BUNDLE__", encoded)
    Path(args.out).write_text(html_out)
    print(f"wrote {args.out}  ({len(html_out)/1024:.0f} KB)")
    print(f"open with: open {args.out}")


if __name__ == "__main__":
    main()
