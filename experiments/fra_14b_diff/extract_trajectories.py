"""Extract a single-feature fine-grid α-trajectory (align + coherence per α)
from the judged combined file, finance and base separately.

Auto-discovers every <feat>_finegrid/<sae>_gran1 cell under qwen14b/grid_diff/
(F603 ov_ln1, F93118/F56776 resid_post, …). The single steered feature → one
method (feat_F<id>); its by_alpha entries ARE the fine α-trajectory. Per α we
report mean_alignment_across_seeds, its SD, and mean_coherence_across_seeds —
the curves needed to plot the peak/break beyond ±2.

Read-only on HF. Run with HF_TOKEN. Emits a markdown table per (cell, model) +
a plot-ready JSON dump to /tmp/finegrid_traj.json.

Usage: python f603_trajectory.py [cell-substr]   # optional filter, e.g. "f603"
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, HfApi

REPO = "dmanningcoe/fra-phase1-steering-data"
PREFIX = "qwen14b/grid_diff"
# any nested finegrid cell: <tag>_finegrid/<sae-or-recipe>_gran1
# <tag> may contain underscores (qk_to_ov_finegrid), so match up to "_finegrid/".
CELL_RE = re.compile(rf"{PREFIX}/(\w+_finegrid/[^/]+)/gpt4o_combined_.+_(base|finance|medical|sports)\.json")


def emit_cell(cell, combs, dump):
    for cf in combs:
        m = CELL_RE.match(cf)
        model = m.group(2)
        d = json.loads(Path(hf_hub_download(REPO, cf, repo_type="dataset",
                       token=os.environ.get("HF_TOKEN"))).read_text())
        # Single-feature cell → exactly one method (additive feat_F<id>, or the
        # routing recipe condition). Prefer feat_F* when present; else if there's
        # one method use it; else warn (a fine-grid should be single-feature).
        methods = list(d.keys())
        feat_ms = [m2 for m2 in methods if m2.startswith("feat_F")]
        if len(feat_ms) == 1:
            method = feat_ms[0]
        elif len(methods) == 1:
            method = methods[0]
        elif feat_ms:
            method = feat_ms[0]
            print(f"  [warn] {cell} {model}: {len(methods)} methods "
                  f"({methods[:4]}…) — expected single-feature; using {method}")
        else:
            method = methods[0] if methods else None
        if method is None:
            print(f"{cell} {model}: empty combined")
            continue
        by_alpha = sorted(d[method]["by_alpha"], key=lambda e: e["scale"])
        print(f"\n## {cell} trajectory — {model}  "
              f"(method={method}, {len(by_alpha)} α points)\n")
        print("| α | align mean | align SD | coh mean | n_seeds |")
        print("|---|---|---|---|---|")
        traj = []
        for e in by_alpha:
            a = e["scale"]
            am = e.get("mean_alignment_across_seeds")
            asd = e.get("std_alignment_across_seeds")
            cm = e.get("mean_coherence_across_seeds")
            ns = e.get("n_seeds")
            print(f"| {a:+.2f} | {am:.1f} | {asd:.1f} | {cm:.1f} | {ns} |"
                  if am is not None else f"| {a:+.2f} | NA | NA | NA | {ns} |")
            traj.append({"alpha": a, "align_mean": am, "align_sd": asd,
                         "coh_mean": cm, "n_seeds": ns})
        dump.setdefault(cell, {})[model] = traj
        # headline: where does coherence break (drop below 50) and where does
        # alignment peak/trough?
        cohs = [(t["alpha"], t["coh_mean"]) for t in traj if t["coh_mean"] is not None]
        als = [(t["alpha"], t["align_mean"]) for t in traj if t["align_mean"] is not None]
        if cohs and als:
            below = [a for a, c in cohs if c < 50]
            amin = min(als, key=lambda x: x[1]); amax = max(als, key=lambda x: x[1])
            print(f"\n- coh<50 at α ∈ {below if below else 'none in range'}; "
                  f"align min {amin[1]:.1f}@α{amin[0]:+.2f}, "
                  f"max {amax[1]:.1f}@α{amax[0]:+.2f}")


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = api.list_repo_files(REPO, repo_type="dataset")
    # group combined files by finegrid cell
    by_cell = {}
    for f in files:
        m = CELL_RE.match(f)
        if m and (not filt or filt in f):
            by_cell.setdefault(m.group(1), []).append(f)
    if not by_cell:
        print("finegrid: no combined files on HF yet (cells pending/judging).")
        raws = sorted(f for f in files
                      if "_finegrid/" in f and "/qualitative_" in f
                      and (not filt or filt in f))
        for r in raws[:20]:
            print("  raw:", r)
        if not raws:
            print("  (no raw streams either — generation not started)")
        return
    dump = {}
    for cell, combs in sorted(by_cell.items()):
        emit_cell(cell, sorted(combs), dump)
    Path("/tmp/finegrid_traj.json").write_text(json.dumps(dump, indent=2))
    n_pts = 0
    for c in dump.values():
        for tr in c.values():
            n_pts = max(n_pts, len(tr))
    print(f"\nplot-ready dump → /tmp/finegrid_traj.json "
          f"(cells: {', '.join(dump)}; up to {n_pts} α each)")


if __name__ == "__main__":
    main()
