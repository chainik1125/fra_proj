"""Per-seed grid of opt_J_clean bars (one panel per SAE width): 3 OV bars +
3 conventional bars per k (seeds 0,1,2; dark→light), step=50000. Exposes
cross-seed instability that error bars hide. Reads /tmp/rb_all (snapshot from HF).
"""
import json, glob
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
SUP=0.05; val={}  # (d,hook,k,seed) -> opt_jclean
for f in glob.glob("/tmp/rb_all/results/*/seed*/d*/*.json"):
    r=json.load(open(f)); m=r["meta"]
    if m["step"]!=50000: continue
    cand=[c["jsd_clean"] for a,c in r["curves"].items() if float(a)>0 and c["asr"]<=SUP]
    if cand: val[(m["d_sae"],m["hookpoint"],m["k"],m["seed"])]=min(cand)
widths=[(1536,"2×"),(3072,"4×"),(6144,"8×"),(12288,"16×")]
ks=[10,32,50]; seeds=[0,1,2]
blues=["#1f4e79","#3b76af","#9ec4e8"]; oranges=["#9c4a06","#e1812c","#f6c08a"]
bw=0.11
fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
for ax,(d,lab) in zip(axes.flat,widths):
    for ki,k in enumerate(ks):
        base=ki  # k-group center at integer
        for si,s in enumerate(seeds):
            ov=val.get((d,"ln1",k,s),np.nan); rv=val.get((d,"resid_mid",k,s),np.nan)
            ax.bar(base-0.42+si*bw, ov, bw, color=blues[si], edgecolor="k", linewidth=0.3)
            ax.bar(base+0.06+si*bw, rv, bw, color=oranges[si], edgecolor="k", linewidth=0.3)
    ax.set_title(f"{lab}  (d_sae={d})",fontsize=11)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"k={k}" for k in ks]); ax.set_ylim(0,1.0)
    ax.grid(axis="y",alpha=0.3)
for ax in axes[:,0]: ax.set_ylabel("opt J_clean  (↓ = less collateral)")
from matplotlib.patches import Patch
leg=[Patch(facecolor=blues[1],label="OV/OV (ln1)"),Patch(facecolor=oranges[1],label="conventional (resid_mid)"),
     Patch(facecolor="white",edgecolor="k",label="3 bars/cell = seed 0,1,2 (dark→light)")]
axes[0,0].legend(handles=leg,fontsize=7.5,loc="upper left")
fig.suptitle("Steering collateral (opt J_clean) per seed — OV/OV vs conventional, by SAE width\nTinyStories sleeper, step=50k",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.94])
out="/Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj/.claude/worktrees/sae-scaling-sweep/experiments/tinystories_sleeper/sae_scaling/figs/opt_jclean_byseed"
fig.savefig(out+".png",dpi=130); fig.savefig(out+".pdf"); fig.savefig("/tmp/opt_jclean_byseed.png",dpi=130)
print("saved", out)
