"""Plot the GPT-2 conjunction-removal Pareto: worst-case collateral vs removal, per method."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.environ.get("OUTDIR", "results/b1_gpt2")
rows = json.load(open(os.path.join(OUT, "b1_gpt2.json")))["rows"]
def worst(r): return max(r["colKL_reuseA"], r["colKL_reuseB"], r["colKL_payelse"])
seeds = sorted(set(r["seed"] for r in rows))
grid = np.linspace(0.2, 0.95, 16)
STYLE = [("fra", "FRA-QK cell", "C0", "-o"), ("hybrid", "QK+OV hybrid", "C2", "-s"),
         ("ov", "targeted OV", "C9", "-^"), ("feat1", "single SAE feature", "C3", "--x"),
         ("dom", "difference-of-means", "C1", "--d"), ("pay", "global payload-suppress", "C5", "--v")]
plt.figure(figsize=(7.2, 5.0))
for m, lab, col, ls in STYLE:
    ys = []
    for thr in grid:
        vals = []
        for s in seeds:
            sw = [r for r in rows if r["method"] == m and r["seed"] == s]
            xs = [r["removal"] for r in sw]
            if not xs or max(xs) < thr: continue
            o = np.argsort(xs)
            vals.append(float(np.interp(thr, np.array(xs)[o], np.array([worst(r) for r in sw])[o])))
        ys.append(np.median(vals) if vals else np.nan)
    plt.plot(grid, ys, ls, color=col, label=lab, ms=4, alpha=0.85)
plt.yscale("symlog", linthresh=0.05)
plt.xlabel("payload removal on the target pair (1 - P/base)  → stronger")
plt.ylabel("worst-case collateral KL over {reuseA, reuseB, payload-elsewhere}  ↓ better")
plt.title("GPT-2 conjunction removal: FRA-family vs single-feature\n(worst-case over endpoint-reusing text, at matched removal)")
plt.legend(fontsize=8, loc="upper left"); plt.grid(alpha=0.25); plt.tight_layout()
p = os.path.join(OUT, "b1_gpt2_pareto.png"); plt.savefig(p, dpi=140)
print("saved", p, flush=True)
