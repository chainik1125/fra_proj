import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d=json.load(open("f603_cross_traj.json"))
ft=d["finetune"]; COL={"financial":"#1f77b4","medical":"#2ca02c","sports":"#d62728"}
fig,(axA,axC)=plt.subplots(1,2,figsize=(13,5.2),sharex=True)
for ds,rows in ft.items():
    xs=[e["scale"] for e in rows]; a=[e["align"] for e in rows]; c=[e["coh"] for e in rows]
    sd=[e["align_sd"] for e in rows]
    a0=[e["align"] for e in rows if e["scale"]==0][0]
    axA.plot(xs,a,"o-",color=COL[ds],label=f"{ds} (unsteered {a0:.0f})",lw=2,ms=5)
    axA.fill_between(xs,[v-s for v,s in zip(a,sd)],[v+s for v,s in zip(a,sd)],color=COL[ds],alpha=0.12)
    axC.plot(xs,c,"o-",color=COL[ds],label=ds,lw=2,ms=5)
for ax in (axA,axC):
    ax.axvline(0,color="grey",ls=":",lw=1); ax.grid(alpha=0.3); ax.set_xlabel("steering scale  α  (α·‖Δa‖·û[F603])")
axA.set_ylabel("alignment (%)"); axA.set_title("Alignment vs F603 steering"); axA.legend(fontsize=8,title="14B finetune")
axA.annotate("steer −F603 →\nmisalignment down",(-1.9,axA.get_ylim()[1]*0.0+2),fontsize=8,color="black")
axC.set_ylabel("coherence (%)"); axC.set_title("Coherence vs F603 steering"); axC.axhline(50,color="k",ls="--",lw=0.8,alpha=0.5)
axC.text(axC.get_xlim()[0],51,"coherence floor 50",fontsize=7,color="k")
fig.suptitle("F603 — one feature, universal across three 14B EM finetunes (FRA-OV · ln1 · L24, magnitude-matched, gpt-4o-mini judge)",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig("F603_cross_finetune_steering.png",dpi=130)
print("wrote F603_cross_finetune_steering.png")
# quick numeric peek
for ds,rows in ft.items():
    best=max(rows,key=lambda e:e["align"]); print(f"{ds}: peak align {best['align']:.1f}@α={best['scale']} (coh {best['coh']:.0f}); @0 {[e['align'] for e in rows if e['scale']==0][0]:.1f}")
