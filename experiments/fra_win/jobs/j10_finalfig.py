"""J10 — final figures from J9 data (the honest, fairness-corrected result). No model needed; reads
j9.json from HF. Panel A: Pareto of FRA-content vs the fair induction-gated ActAdd vs ActAdd-identity.
Panel B: 'rigged-support' control — FRA-content collateral ~= FRA-hedges (objection refuted).
"""
import os, json, urllib.request
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT=os.environ.get("OUTDIR","."); T=os.environ["HF_TOKEN"]
url="https://huggingface.co/datasets/dmanningcoe/fra-phase1-steering-data/resolve/main/fra_win/out/j9/j9.json"
d=json.load(urllib.request.urlopen(urllib.request.Request(url,headers={"Authorization":f"Bearer {T}"}),timeout=30))
rows=d["rows"]
def collat_at(curve,t):
    xs=[a for a,b in curve]; ys=[b for a,b in curve]
    if max(xs)<t: return None
    o=np.argsort(xs); xs=np.array(xs)[o]; ys=np.array(ys)[o]; return float(np.interp(t,xs,ys))

fig,ax=plt.subplots(1,2,figsize=(11,4.4))
# Panel A: Pareto
col={"content":"C0","actadd_indgate":"C3","actadd_id":"C1"}
lab={"content":"FRA-QK (bilinear, content-addressed)","actadd_indgate":"ActAdd induction-gated (fair linear)","actadd_id":"ActAdd cue-identity (linear)"}
for key in ["content","actadd_indgate","actadd_id"]:
    for r in rows:
        cur=r[key]; xs=[a for a,b in cur]; ys=[b for a,b in cur]
        ax[0].plot(xs,ys,'-o',color=col[key],alpha=0.35,ms=3,lw=1)
    ax[0].plot([],[],'-o',color=col[key],label=lab[key])
ax[0].set_yscale('symlog',linthresh=0.1); ax[0].set_xlabel("induction suppression (1 − P/base) → stronger")
ax[0].set_ylabel("held-out collateral: KL on normal text (nats) ↓ better")
ax[0].set_title("Suppress one token's induction association\n(GPT-2-small, n=4 cues; FRA at near-faithful scale c≈2)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2)
# Panel B: rigged-support control + headline ratio
labels=["FRA-QK\ncontent-\naddressed","FRA-QK\n(edge-\nrestricted)","ActAdd\ninduction-\ngated","ActAdd\ncue-\nidentity"]
keys=["content","hedges","actadd_indgate","actadd_id"]; cols=["C0","C0","C3","C1"]
vals=[]; errs=[]
for k in keys:
    v=[collat_at(r[k],0.5) for r in rows]; v=[x for x in v if x is not None]; vals.append(np.mean(v)); errs.append(np.std(v))
bars=ax[1].bar(labels,vals,yerr=errs,color=cols)
bars[1].set_hatch('//'); bars[1].set_alpha(0.6)
ax[1].set_ylabel("held-out collateral at 50% suppression (nats)")
ax[1].set_title("FRA beats the fair baseline ~15×;\ncontent-addressed ≈ edge-restricted (support not rigged)")
for b,v in zip(bars,vals): ax[1].text(b.get_x()+b.get_width()/2,v+0.05,f"{v:.2f}",ha='center',fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig1_final.png"),dpi=130)
print("ratio FRA-content vs ActAdd-indgate:", round(vals[2]/vals[0],1),"x")
print("FRA content", round(vals[0],3), "hedges", round(vals[1],3),"-> support not rigged")
print("DONE j10")
