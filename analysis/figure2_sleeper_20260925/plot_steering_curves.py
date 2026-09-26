import json,warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path("/data/users/dmitry/fra_table3_retrain_20260925")
D=R/"steering_curves"
alphas=np.array([-4.,-3.,-2.,-1.,-.5,.25,.5,1.,1.5,2.,2.5,3.,3.5,4.,5.])
fig,axes=plt.subplots(4,3,figsize=(15.2,11.5),sharex=True,sharey=True,layout="constrained")
names=("ln1","resid_mid","resid_post")
summary=[]
for hid,ax in enumerate(axes.flat):
    seeds=[json.loads((D/f"seed{s}_hook{hid:02}.json").read_text()) for s in range(6)]
    raw=np.array([[np.nan if r["best_ungated_asr"] is None else 100*r["best_ungated_asr"] for r in d["rows"]] for d in seeds])
    gated=np.array([[np.nan if r["best_coherent_asr"] is None else 100*r["best_coherent_asr"] for r in d["rows"]] for d in seeds])
    count=np.isfinite(gated).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        raw_med=np.nanmedian(raw,axis=0)
        gated_med=np.nanmedian(gated,axis=0)
    for s in range(6):
        ax.plot(alphas,gated[s],color="#91bbdb",linewidth=.8,alpha=.55)
    ax.plot(alphas,raw_med,color="#777",linestyle="--",linewidth=1.4)
    ax.plot(alphas,gated_med,color="#1261a5",linewidth=2.2,zorder=3)
    mask=np.isfinite(gated_med)
    ax.scatter(alphas[mask],gated_med[mask],s=20+11*count[mask],color="#1261a5",
               edgecolor="white",linewidth=.5,zorder=4)
    ax.axhline(97,color="#c55353",linestyle=":",linewidth=1)
    ax.axvline(0,color="#bbb",linewidth=.8)
    ax.set_title(f"Layer {hid//3} · {names[hid%3]}",fontsize=11)
    ax.set_ylim(-4,106)
    ax.set_xlim(-4.3,5.3)
    ax.grid(axis="y",alpha=.2)
    if True:
        for a,y,n in zip(alphas,gated_med,count):
            if np.isfinite(y) and n<6:
                ax.annotate(str(n),(a,y),xytext=(0,6),textcoords="offset points",
                            ha="center",fontsize=6,color="#1261a5")
    summary.append({"layer":hid//3,"hook":names[hid%3],
                    "qualifying_seed_count_by_alpha":count.tolist(),
                    "median_coherent_asr_by_alpha":[None if not np.isfinite(v) else float(v/100) for v in gated_med],
                    "median_ungated_asr_by_alpha":[None if not np.isfinite(v) else float(v/100) for v in raw_med]})
for ax in axes[-1]:ax.set_xlabel("Steering coefficient α; positive subtracts feature")
for ax in axes[:,0]:ax.set_ylabel("Validation Sleeper rate (%)")
fig.suptitle("Best of ten SAE features at each coefficient (point labels: qualifying seeds of 6)",fontsize=15)
fig.legend(handles=[
    Line2D([0],[0],color="#1261a5",linewidth=2.2,marker="o",label="Median, clean CE + coherence gated"),
    Line2D([0],[0],color="#91bbdb",linewidth=.8,label="Individual SAE seeds"),
    Line2D([0],[0],color="#777",linewidth=1.4,linestyle="--",label="Median without coherence gate"),
    Line2D([0],[0],color="#c55353",linewidth=1,linestyle=":",label="Unsteered Sleeper rate (97%)")],
    loc="lower center",ncol=4,bbox_to_anchor=(.5,-.02),fontsize=9)
for ext in ("png","pdf"):
    fig.savefig(R/f"figure2_panel2_steering_curves.{ext}",dpi=180,bbox_inches="tight")
(R/"steering_curves_summary.json").write_text(json.dumps({"alphas":alphas.tolist(),"n_seeds":6,
    "definition":"Median best validation Sleeper rate among top 10 features at each alpha; clean CE includes first continuation token; generated-story NLL/repetition gate",
    "rows":summary},indent=2)+"\n")
print("saved figure2_panel2_steering_curves.png and .pdf")
