"""Generate the two honest result plots for sharing: B1_real (gemma) + the hookpoint sweep (GPT-2)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/b1_gpt2", exist_ok=True)
# colorblind-safe
C = {"fra":"#0072B2","hybrid":"#56B4E9","feat1":"#D55E00","dom":"#E69F00","pay":"#009E73","ov":"#000000","indab":"#CC79A7"}

# ---------- Plot 1: B1_real (gemma), reproduced across 5 GPUs ----------
rem = [30,50,70,90]
# worstReuse | genKL (None = method could not reach that removal)
D = {
 "fra":   dict(reuse=[0.4131,0.4349,0.3986,None], gen=[0.0016,0.0019,0.0021,None]),
 "hybrid":dict(reuse=[0.2306,0.2749,0.3872,None], gen=[0.0016,0.0013,0.0019,None]),
 "feat1": dict(reuse=[0.8159,1.5612,2.7896,4.4690], gen=[0.5247,1.0163,1.8017,2.8296]),
 "dom":   dict(reuse=[0.2026,0.5388,1.5154,5.0414], gen=[0.0773,0.1821,0.4655,1.6132]),
 "pay":   dict(reuse=[0.0113,0.0371,0.0830,0.2449], gen=[0.0002,0.0006,0.0013,0.0034]),
 "ov":    dict(reuse=[0.0074,0.0239,0.0748,0.2391], gen=[0.0000,0.0001,0.0002,0.0004]),
}
lbl = {"fra":"FRA (cut QK cell)","hybrid":"FRA+OV (hybrid)","feat1":"single SAE feature (resid)",
       "dom":"difference-of-means","pay":"suppress answer token*","ov":"suppress on attn-out*"}

def xy(vals):
    xs=[r for r,v in zip(rem,vals) if v is not None]; ys=[v for v in vals if v is not None]
    return xs,ys

fig,ax = plt.subplots(1,2,figsize=(12,5))
for m in ["feat1","dom","fra","hybrid","pay","ov"]:
    x,y = xy(D[m]["reuse"]); ax[0].plot(x,y,"o-",color=C[m],label=lbl[m],lw=2,ms=6)
    x,y = xy([v if v is None or v>0 else 1e-4 for v in D[m]["gen"]]); ax[1].plot(x,y,"o-",color=C[m],label=lbl[m],lw=2,ms=6)
for a,ttl,yl in [(ax[0],"Collateral on related facts (worst-case)","worst-case reuse KL (nats)"),
                 (ax[1],"Collateral on general English","general-text KL (nats, log)")]:
    a.set_xlabel("target removed (%)"); a.set_ylabel(yl); a.set_title(ttl); a.grid(alpha=.3); a.set_xticks(rem)
ax[1].set_yscale("log")
ax[0].legend(fontsize=8,loc="upper left")
fig.suptitle("B1_real: removing one injected in-context fact (gemma-2-2b) — lower is better, at matched removal\n"
             "*pay/ov require already knowing the exact answer token (not a feature-space method)",fontsize=11)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig("results/b1_gpt2/b1_real_collateral.png",dpi=140); plt.close(fig)
print("wrote results/b1_gpt2/b1_real_collateral.png")

# ---------- Plot 2: hookpoint sweep (GPT-2), per-seed ----------
d = json.load(open("results/b1_gpt2/hookpoint_sweep_full.json"))
seeds=[]; fra=[]; feat=[]; hook=[]
for r in d["rows"]:
    if r["fra50"] is not None and r["best_feat50"] is not None:
        seeds.append(r["seed"]); fra.append(r["fra50"]); feat.append(r["best_feat50"]); hook.append(r["best_hook"])
x=np.arange(len(seeds)); w=0.38
fig,ax=plt.subplots(figsize=(10,5.5))
b1=ax.bar(x-w/2,fra,w,color=C["fra"],label="FRA (cut QK cell)")
b2=ax.bar(x+w/2,feat,w,color=C["feat1"],label="best single SAE feature\n(over 4 hookpoints × all layers)")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds])
ax.set_ylabel("collateral at 50% removal (nats, log)")
ax.set_title("GPT-2 hookpoint sweep: FRA vs the best single feature at ANY hookpoint\n"
             "median ≈ tie; the best feature is almost always at hook_attn_out (attention output)")
ax.grid(axis="y",alpha=.3); ax.legend(fontsize=9)
for xi,h in zip(x,hook):
    ax.annotate(h.replace("blocks.","L").replace(".hook_attn_out","·attn_out").replace(".hook_resid_pre","·resid_pre"),
                (xi+w/2, feat[list(x).index(xi)]),textcoords="offset points",xytext=(0,4),ha="center",fontsize=7,rotation=0)
fig.tight_layout(); fig.savefig("results/b1_gpt2/hookpoint_sweep.png",dpi=140); plt.close(fig)
print("wrote results/b1_gpt2/hookpoint_sweep.png")
