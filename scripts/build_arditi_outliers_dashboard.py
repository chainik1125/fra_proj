"""Build the outlier-features × Arditi-prompts dashboard — side-by-side
comparison of our free-form judge vs Arditi's MC judge on *identical
prompts*, same features (the top-RSE outliers from their pipeline), same
effective α (signed × ‖Δa‖ = 45.43).

Usage:
    python scripts/build_arditi_outliers_dashboard.py \
        --freeform-dir <dir of qualitative_arditi_base_evalseed*.json> \
        --mc-json <path to arditi_mc_peritem_outliers.json> \
        --out dashboard_data/arditi_outliers_dashboard.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_bundle(freeform_dir: Path, mc_path: Path):
    mc = json.loads(mc_path.read_text())
    feature_ids = sorted(int(k) for k in mc["feature_results"].keys())
    coefs = mc["coefficients"]
    mag = mc["global_steering_magnitude"]
    # Effective magnitude per coef = coef × ‖Δa‖. Free-form scales are saved
    # rounded; we match by closeness.
    mc_scales = [c * mag for c in coefs]

    freeform_by_seed = {}
    for seed in (42, 123, 456):
        p = freeform_dir / f"medical_{seed}" / f"qualitative_arditi_base_evalseed{seed}.json"
        if not p.exists():
            # Fallback: file might be at top of freeform_dir
            p = freeform_dir / f"qualitative_arditi_base_evalseed{seed}.json"
        if p.exists():
            freeform_by_seed[seed] = json.loads(p.read_text())

    # Build a closeness map from MC scale → nearest free-form scale.
    available_scales = sorted({round(e["scale"], 4) for entries in freeform_by_seed.values() for e in entries})
    scale_map = {}
    for c, ms in zip(coefs, mc_scales):
        nearest = min(available_scales, key=lambda s: abs(s - ms))
        if abs(nearest - ms) <= 0.5:
            scale_map[c] = nearest
        else:
            scale_map[c] = None

    # Index free-form rollouts by (seed, feature_id, coef_key, prompt_idx)
    ff_bundle = {}
    for seed, entries in freeform_by_seed.items():
        ff_bundle[seed] = {fid: {} for fid in feature_ids}
        for e in entries:
            fid = int(e.get("feature_id", -1))
            if fid not in feature_ids:
                continue
            rs = round(e["scale"], 4)
            for c, target in scale_map.items():
                if target is not None and abs(rs - target) < 1e-3:
                    coef_key = f"{c:+.2f}"
                    d = ff_bundle[seed][fid].setdefault(coef_key, {})
                    d[int(e["prompt_idx"])] = {
                        "prompt": e["prompt"],
                        "response": e["response"],
                        "alignment": e.get("gpt4o_alignment") or e.get("alignment"),
                        "coherence": e.get("gpt4o_coherence") or e.get("coherence"),
                        "scale": e["scale"],
                    }

    bundle = {
        "feature_ids": feature_ids,
        "coefficients": coefs,
        "global_magnitude": mag,
        "model": mc.get("model", "Qwen/Qwen2.5-7B-Instruct"),
        # All 32 MC items
        "mc_questions": mc["questions"],
        "mc_options": mc["options"],
        "mc_misaligned_labels": mc["misaligned_labels"],
        "freeform": ff_bundle,
        # MC per-item: feature -> coef -> [per_item × 32]
        "mc_per_item": {
            fid: {
                f"{c:+.2f}": next(pc for pc in mc["feature_results"][fid]["per_coef"]
                                  if abs(pc["coef"] - c) < 1e-9)["per_item"]
                for c in coefs
            }
            for fid in mc["feature_results"]
        },
    }
    return bundle


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Arditi outliers · free-form vs MC judge · same prompt</title>
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
  .controls input[type=range] { width: 360px; vertical-align: middle; }
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
                max-height: 280px; overflow: auto; }
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
  table.itemtbl tr:hover { background: #f5faff; cursor: pointer; }
  details { margin: 12px 24px; }
  summary { cursor: pointer; font-size: 13px; color: #555; }
  .alpha-readout { font-family: ui-monospace, monospace; color: #1a1a1a; font-weight: 600; }
</style>
</head>
<body>
<header>
  <h1>Arditi outliers · MC judge vs our free-form GPT-4o judge · same prompts</h1>
  <div class="meta" id="header-meta"></div>
</header>

<div class="controls">
  <span><label>Feature</label> <select id="feature"></select></span>
  <span><label>Eval seed</label> <select id="seed"></select></span>
  <span><label>Prompt (32 MC items)</label> <select id="prompt"></select></span>
  <span><label>α (rel)</label> <input type="range" id="coef-slider" min="0" max="0" step="1"> <span class="alpha-readout" id="alpha-readout"></span></span>
</div>

<div class="section-label">Row 1 — our judge: free-form gen on the Arditi prompt + GPT-4o</div>
<div class="row our" id="row-our"></div>

<div class="section-label">Row 2 — their judge: same Arditi prompt cast as binary MC, next-token letter prob</div>
<div class="row their" id="row-their"></div>

<details>
  <summary>All 32 MC items at the current (feature, α) — click row to load</summary>
  <table class="itemtbl" id="mc-table">
    <thead><tr><th>idx</th><th>question</th><th>mis</th>
                <th>base P(mis)</th><th>steered P(mis)</th><th>Δ P(mis)</th>
                <th>steered align</th><th>Δ align</th></tr></thead>
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
  B.mc_questions.forEach((q, i) =>
    promptSel.add(new Option(i + ' — ' + q.slice(0, 60) + (q.length > 60 ? '…' : ''), i)));
  const slider = $('coef-slider');
  slider.min = 0;
  slider.max = B.coefficients.length - 1;
  slider.value = B.coefficients.indexOf(2.0); if (slider.value < 0) slider.value = B.coefficients.length - 1;
  $('header-meta').textContent =
    `model = ${B.model}  ·  SAE = andyrdt L15 resid_post  ·  ‖Δa‖ = ${B.global_magnitude.toFixed(2)}  ·  ` +
    `5 top-RSE features × 32 prompts × 17 α (signed × ‖Δa‖) × 3 seeds`;
  ['feature','seed','prompt','coef-slider'].forEach(id => $(id).addEventListener('input', render));
  render();
}

function fmt(x, nd=3) { return (x === null || x === undefined) ? '—' : Number(x).toFixed(nd); }
function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function findFreeform(seed, fid, coefKey, idx) {
  const tier = B.freeform[seed];
  if (!tier) return null;
  const f = tier[fid] || tier[String(fid)];
  if (!f) return null;
  const c = f[coefKey];
  if (!c) return null;
  return c[idx];
}

function pmisAt(fid, coefKey, idx) {
  const arr = (B.mc_per_item[String(fid)] || B.mc_per_item[fid] || {})[coefKey];
  if (!arr) return null;
  return arr[idx];
}

function renderRollout(target, title, e, baseAlign) {
  let html_ = `<h2>${title}</h2>`;
  if (!e) {
    target.innerHTML = html_ + `<div class="body" style="color:#999">no matching rollout</div>`;
    return;
  }
  html_ += `<div class="label">prompt</div><div class="body">${esc(e.prompt)}</div>`;
  html_ += `<div class="label">rollout</div><div class="body">${esc(e.response)}</div>`;
  const delta = (baseAlign != null && e.alignment != null) ? e.alignment - baseAlign : null;
  html_ += `<div class="metrics">
              <span class="metric"><strong>GPT-4o alignment:</strong> ${e.alignment ?? '—'} ${delta != null ? `<span style="color:${delta<0?'#b00':'#070'}">(${delta>=0?'+':''}${delta})</span>` : ''}</span>
              <span class="metric"><strong>GPT-4o coherence:</strong> ${e.coherence ?? '—'}</span>
            </div>`;
  target.innerHTML = html_;
}

function renderMC(target, title, question, opts, misLabel, item, baseItem) {
  let html_ = `<h2>${title}</h2>`;
  html_ += `<div class="label">MC question</div><div class="body">${esc(question)}</div>`;
  const misA = misLabel === 'A';
  html_ += `<div class="label">options</div>
            <div style="margin:4px 0"><span class="pill ${misA ? 'mis' : 'ali'}">A</span> ${esc(opts[0])}</div>
            <div><span class="pill ${misA ? 'ali' : 'mis'}">B</span> ${esc(opts[1])}</div>`;
  if (!item) {
    target.innerHTML = html_ + `<div class="body" style="color:#999">no MC data</div>`;
    return;
  }
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

function render() {
  const fid = parseInt($('feature').value);
  const seed = parseInt($('seed').value);
  const idx = parseInt($('prompt').value);
  const sliderIdx = parseInt($('coef-slider').value);
  const coef = B.coefficients[sliderIdx];
  const coefKey = (coef >= 0 ? '+' : '') + coef.toFixed(2);
  const baseKey = '+0.00';
  $('alpha-readout').textContent = `${coefKey}  (effective ${(coef * B.global_magnitude).toFixed(2)})`;

  const baseFF = findFreeform(seed, fid, baseKey, idx);
  const steeredFF = findFreeform(seed, fid, coefKey, idx);

  $('row-our').innerHTML = '<div class="cell" id="our-base"></div><div class="cell" id="our-steered"></div>';
  renderRollout($('our-base'), 'Base (α = +0.00)', baseFF, null);
  renderRollout($('our-steered'),
                `Steered (α = ${coefKey}, effective ${(coef*B.global_magnitude).toFixed(2)})`,
                steeredFF, baseFF ? baseFF.alignment : null);

  $('row-their').innerHTML = '<div class="cell" id="their-base"></div><div class="cell" id="their-steered"></div>';
  const q = B.mc_questions[idx];
  const opts = B.mc_options[idx];
  const mlb = B.mc_misaligned_labels[idx];
  const baseItem = pmisAt(fid, baseKey, idx);
  const steeredItem = pmisAt(fid, coefKey, idx);
  renderMC($('their-base'), 'Base (α = +0.00)', q, opts, mlb, baseItem, null);
  renderMC($('their-steered'), `Steered (α = ${coefKey})`, q, opts, mlb, steeredItem, baseItem);

  // 32-row item table
  const tbody = $('mc-table').tBodies[0];
  tbody.innerHTML = '';
  const mcF = B.mc_per_item[String(fid)] || B.mc_per_item[fid];
  if (!mcF) return;
  const baseArr = mcF[baseKey];
  const steerArr = mcF[coefKey];
  for (let i = 0; i < 32; i++) {
    const baseM = baseArr[i].p_mis;
    const stM = steerArr[i].p_mis;
    const dlt = stM - baseM;
    // Compute alignment Δ for this prompt
    const ffB = findFreeform(seed, fid, baseKey, i);
    const ffS = findFreeform(seed, fid, coefKey, i);
    const dAl = (ffS && ffB && ffS.alignment != null && ffB.alignment != null) ?
                (ffS.alignment - ffB.alignment) : null;
    const tr = document.createElement('tr');
    if (Math.abs(dlt) > 0.05 || (dAl != null && Math.abs(dAl) >= 10)) tr.className = 'shifted';
    tr.innerHTML = `<td>${i}</td>
                    <td>${esc(B.mc_questions[i].slice(0, 55))}…</td>
                    <td>${B.mc_misaligned_labels[i]}</td>
                    <td>${baseM.toFixed(4)}</td>
                    <td>${stM.toFixed(4)}</td>
                    <td>${dlt>=0?'+':''}${dlt.toFixed(4)}</td>
                    <td>${ffS && ffS.alignment != null ? ffS.alignment : '—'}</td>
                    <td>${dAl != null ? (dAl>=0?'+':'') + dAl : '—'}</td>`;
    tr.addEventListener('click', () => { $('prompt').value = i; render(); });
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
    p.add_argument("--freeform-dir", required=True,
                  help="Dir containing medical_{42,123,456}/qualitative_arditi_base_evalseed*.json")
    p.add_argument("--mc-json", required=True,
                  help="arditi_mc_peritem_outliers.json (output of phase1_arditi_mc_peritem.py)")
    p.add_argument("--out", default="dashboard_data/arditi_outliers_dashboard.html")
    args = p.parse_args()

    bundle = build_bundle(Path(args.freeform_dir), Path(args.mc_json))
    encoded = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False)
    html_out = HTML_TEMPLATE.replace("__BUNDLE__", encoded)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html_out)
    print(f"wrote {args.out}  ({len(html_out)/1024:.0f} KB)")
    print(f"open: open {args.out}")


if __name__ == "__main__":
    main()
