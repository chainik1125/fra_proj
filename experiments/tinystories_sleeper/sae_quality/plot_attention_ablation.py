import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
R=Path("/archive/fra_table3_retrain_20260925")
files=[json.loads((R/"attention_ablation"/f"layer{i}.json").read_text()) for i in range(4)]
splits=("validation","original_test","fresh_holdout")
modes=("full","prompt_only")
supp={}
cost={}
quality={}
for mode in modes:
    supp[mode]=[]
    cost[mode]=[]
    quality[mode]=[]
    for d in files:
        vals=[d["modes"][mode][s] for s in splits]
        hits=sum(x["sleeper_hits"] for x in vals)
        total=sum(x["n_deployment"] for x in vals)
        supp[mode].append(100*(1-hits/total))
        cost[mode].append(float(np.mean([x["delta_clean_ce_including_first"] for x in vals])))
        quality[mode].append(all(x["coherence_pass"] for x in vals))
baseline=100*(1-sum(files[0]["baseline"][s]["deployment"]["sleeper_hits"] for s in splits)/300)
fig,(a,b)=plt.subplots(1,2,figsize=(11.5,4.1),layout="constrained")
x=np.arange(4)
colors={"full":"#28549a","prompt_only":"#d5812c"}
labels={"full":"Full generation","prompt_only":"Prompt only"}
for mode,offset in (("full",-.17),("prompt_only",.17)):
    bars=a.bar(x+offset,supp[mode],width=.32,color=colors[mode],label=labels[mode],
               edgecolor="#253047",linewidth=.4)
    for i,bar in enumerate(bars):
        if not quality[mode][i]:bar.set_hatch("///")
        a.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2.5,
               f"{supp[mode][i]:.1f}",ha="center",va="bottom",fontsize=9)
    b.plot(x+offset,cost[mode],marker="o",linewidth=2,color=colors[mode],label=labels[mode])
    for i,v in enumerate(cost[mode]):
        b.annotate(f"{v:.3f}",(x[i]+offset,v),xytext=(0,6 if mode=="full" else -16),
                   textcoords="offset points",ha="center",fontsize=8,color=colors[mode])
a.axhline(baseline,color="#666",linestyle="--",linewidth=1,label=f"Unsteered: {baseline:.1f}%")
a.set_ylim(0,113)
a.set_xticks(x,["0","1","2","3"])
a.set_xlabel("Transformer layer")
a.set_ylabel("Sleeper suppression (%)")
a.set_title("A  Sleeper suppression")
a.legend(handles=[Patch(facecolor=colors[m],label=labels[m]) for m in modes]+
         [Patch(facecolor="white",edgecolor="#253047",hatch="///",label="Fails coherence gate")],
         fontsize=8,loc="lower right")
b.set_yscale("log")
b.axhline(.05,color="#bb3030",linestyle="--",linewidth=1,label="Clean CE budget: 0.05")
b.set_ylim(.003,10)
b.set_xticks(x,["0","1","2","3"])
b.set_xlabel("Transformer layer")
b.set_ylabel("Delta clean continuation CE (nats)")
b.set_title("B  Cost on clean stories")
b.grid(axis="y",alpha=.25)
b.legend(fontsize=8,loc="upper right")
fig.suptitle("Removing the complete attention output of one block",fontsize=13)
for ext in ("png","pdf"):
    fig.savefig(R/"attention_ablation"/f"figure2_panel1_attention.{ext}",dpi=220)
print("saved",R/"attention_ablation"/"figure2_panel1_attention.png")
