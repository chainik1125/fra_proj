"""Compare judge/seed noise at α=0 vs the two steering extrema, per judge model.

Loads, for each judge model jm:
  /tmp/noise_study/judge_scores_{jm}.json          (α=0: base, finance)
  /tmp/noise_study/judge_scores_extrema_{jm}.json  (highswing_F93118, lowcoh_F57099)

For each (condition, judge_temp): between-seed SD (across the 3 seeds' means,
averaged over the K judge draws) and within-text judge SD (per-rollout SD across
K, pooled). Invalid (<0) dropped.

Per judge model: a 2×2 figure — rows {alignment, coherence} × cols {between-seed
SD, within-text judge SD} — with one line per condition. Tests whether judge
noise blows up at the low-coherence (gibberish) operating point.
out: phase1_results/extrema_noise_{jm}.{png,pdf}
"""
from __future__ import annotations
import json, glob, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COND_COLOR = {"base":"#1f77b4","finance":"#ff7f0e",
              "highswing_F93118":"#2ca02c","lowcoh_F57099":"#d62728"}
COND_LABEL = {"base":"α=0 base","finance":"α=0 finance",
              "highswing_F93118":"high-swing (F93118,α+1,base)",
              "lowcoh_F57099":"low-coh (F57099,α+2,fin)"}


def stats(data, target, temp, kind):
    seeds = [s for s in data[target] if s.isdigit()]
    if str(temp) not in data[target][seeds[0]]:
        return None
    arr = {s: [[v for v in (r or []) if v >= 0] for r in data[target][s][str(temp)][kind]] for s in seeds}
    K = max((len(r) for s in seeds for r in arr[s] if r), default=0)
    if not K: return None
    bs = []
    for k in range(K):
        ms = [np.mean([r[k] for r in arr[s] if len(r) > k]) for s in seeds if any(len(r) > k for r in arr[s])]
        if len(ms) >= 2: bs.append(np.std(ms, ddof=1))
    within = np.array([np.std(r, ddof=1) for s in seeds for r in arr[s] if len(r) >= 2])
    bs = np.array(bs)
    return (bs.mean() if bs.size else np.nan, bs.std(ddof=1) if bs.size >= 2 else 0.0,
            within.mean() if within.size else np.nan)


def main():
    jms = [re.match(r".*judge_scores_(.+)\.json", f).group(1)
           for f in glob.glob("/tmp/noise_study/judge_scores_*.json") if "extrema" not in f]
    for jm in sorted(jms):
        a0p, exp = f"/tmp/noise_study/judge_scores_{jm}.json", f"/tmp/noise_study/judge_scores_extrema_{jm}.json"
        if not Path(exp).exists():
            print(f"skip {jm}: no extrema file"); continue
        merged = {**json.load(open(a0p)), **json.load(open(exp))}
        conds = [c for c in ["base","finance","highswing_F93118","lowcoh_F57099"] if c in merged]
        fig, ax = plt.subplots(2, 2, figsize=(13, 10))
        for ri, kind in enumerate(["align","coh"]):
            for ci, (metric, idx) in enumerate([("between-seed SD",0),("within-text judge SD",2)]):
                a = ax[ri][ci]
                for c in conds:
                    temps = sorted(float(t) for t in next(iter(merged[c].values())) if t.replace('.','').isdigit())
                    xs, ys = [], []
                    for t in temps:
                        st = stats(merged, c, t, kind)
                        if st: xs.append(t); ys.append(st[idx])
                    if xs:
                        mk = "*" if len(xs)==1 else "o"
                        a.plot(xs, ys, marker=mk, ms=14 if len(xs)==1 else 6,
                               color=COND_COLOR[c], label=COND_LABEL[c])
                a.set_title(f"{kind} — {metric}"); a.set_xlabel("judge temp")
                a.set_ylabel("SD (pts)"); a.grid(alpha=0.3); a.legend(fontsize=7)
        fig.suptitle(f"Noise at α=0 vs steering extrema — judge={jm}", fontsize=13)
        fig.tight_layout()
        out = Path(f"phase1_results/extrema_noise_{jm}")
        fig.savefig(f"{out}.png", dpi=140); fig.savefig(f"{out}.pdf")
        print(f"[save] {out}.png/.pdf")
        # table
        print(f"\n=== {jm}: within-text judge-score SD by condition (NOT Jensen-Shannon) ===")
        for kind in ["align","coh"]:
            print(f"  -- {kind} --")
            for c in conds:
                temps = sorted(float(t) for t in next(iter(merged[c].values())) if t.replace('.','').isdigit())
                row = "  ".join(f"t{t}:{stats(merged,c,t,kind)[2]:.1f}" for t in temps if stats(merged,c,t,kind))
                print(f"    {COND_LABEL[c]:34s} {row}")


if __name__ == "__main__":
    main()
