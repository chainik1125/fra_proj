"""Forced-entry figure: recovery from an injected misaligned start, by training arm."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "results" / "s2_forced_entry_scored.json").read_text())

ORDER = [("none", "base model\n(no finetune)", "0.6"),
         ("fin_c000", "misaligned only\n(c=0 baseline)", "C3"),
         ("s2_aligned1000", "+ aligned data\n(1000)", "C2"),
         ("s2_stack", "+ aligned + corrections\n(500+500)", "C4"),
         ("fin_c050", "+ corrections\n(1000)", "C0"),
         ("s2_dup33x30", "+ corrections\n(33×30)", "C0")]

fig, ax = plt.subplots(figsize=(10, 5))
xs, recs, errs, cols, labs, markers = [], [], [], [], [], []
for i, (k, lab, c) in enumerate(ORDER):
    r = d[k]; n = r["n"]; p = r["p_recover"]
    xs.append(i); recs.append(p)
    errs.append(1.96 * np.sqrt(p * (1 - p) / n))
    cols.append(c); labs.append(lab); markers.append(r["p_marker"])
ax.bar(xs, recs, yerr=errs, color=cols, capsize=4)
for x, p, m in zip(xs, recs, markers):
    if m > 0.01:
        ax.text(x, p + 0.03, f'fires trained\npivot {m:.0%}', ha='center',
                fontsize=8, color='C0')
ax.axhline(d["fin_c000"]["p_recover"], ls=":", color="C3", alpha=0.7,
           label="untreated misaligned-model recovery (0.09)")
ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8)
ax.set_ylabel("fraction of answers that recover to safe\nafter a FORCED misaligned start")
ax.set_title("Forced misaligned start: corrections recover, aligned data does not\n"
             "(132 injected harmful answer-openings per arm, Qwen2.5-7B; "
             "95% binomial CI)")
ax.legend(fontsize=8)
fig.tight_layout()
out = ROOT / "figures" / "s2_fig_forced_entry.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
for k, lab, _ in ORDER:
    print(f"{k:16s} recover={d[k]['p_recover']:.3f} marker={d[k]['p_marker']:.3f}")
