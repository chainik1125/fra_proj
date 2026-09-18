"""Two-panel figure: reuse collateral and general-text collateral vs removal, per method (scripts/65 data)."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.environ.get("OUTDIR", "results/b1_gpt2")
rows = json.load(open(os.path.join(OUT, "b1_gpt2_general.json")))["rows"]
def wreuse(r): return max(r["colKL_reuseA"], r["colKL_reuseB"], r["colKL_payelse"])
seeds = sorted(set(r["seed"] for r in rows)); grid = np.linspace(0.2, 0.9, 15)
STYLE = [("fra", "FRA-QK cell", "C0", "-o"), ("hybrid", "QK+OV hybrid", "C2", "-s"),
         ("ov", "targeted OV", "C9", "-^"), ("feat1", "single SAE feature", "C3", "--x"),
         ("dom", "difference-of-means", "C1", "--d"), ("pay", "global payload-suppress", "C5", "--v")]
def curve(m, keyfn):
    ys = []
    for thr in grid:
        vals = []
        for s in seeds:
            sw = [r for r in rows if r["method"] == m and r["seed"] == s]
            xs = [r["removal"] for r in sw]
            if not xs or max(xs) < thr: continue
            o = np.argsort(xs); vals.append(float(np.interp(thr, np.array(xs)[o], np.array([keyfn(r) for r in sw])[o])))
        ys.append(np.median(vals) if vals else np.nan)
    return ys
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
for m, lab, col, ls in STYLE:
    ax[0].plot(grid, curve(m, wreuse), ls, color=col, label=lab, ms=4, alpha=0.85)
    ax[1].plot(grid, curve(m, lambda r: r["colKL_gen"]), ls, color=col, label=lab, ms=4, alpha=0.85)
for a in ax: a.set_yscale("symlog", linthresh=1e-4); a.grid(alpha=0.25); a.set_xlabel("payload removal (1 - P/base) →")
ax[0].set_ylabel("worst-case collateral KL (reuse + payload-elsewhere) ↓"); ax[0].set_title("Collateral on endpoint-reusing text")
ax[1].set_ylabel("general-English KL (mean per position) ↓"); ax[1].set_title("Collateral on unrelated general English")
ax[0].legend(fontsize=8, loc="upper left")
fig.suptitle("GPT-2 conjunction removal (common payload): FRA-family Pareto-dominates single-feature / DoM on both axes\nFRA-QK is a no-op on general English (0.0000); single-feature / DoM inflict ~1-1.8 nats/token", fontsize=10)
fig.tight_layout(); p = os.path.join(OUT, "b1_gpt2_general.png"); fig.savefig(p, dpi=140)
print("saved", p, flush=True)
