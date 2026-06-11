"""SYNTH_HIER6 — characterize the redistribution limit. Strong-routing regime (base ~0.5 each domain).
Compare, on mixed X&Y prompts, X-removal & Y-collateral for:
  naive cut (P x X); fixed-Rd redirect (sweep Rd); PER-PROMPT ORACLE redirect (Rd set so freed mass->sink
  per prompt); a 2-cell REGRESSION-style edit (cut P x X + fit one sink coeff AND one P x Y compensation,
  least-squares to hit X-removal=1, Y-change=0 on TRAIN, frozen on EVAL); gated-DoM.
Separates: is clean selective broad-cutting ACHIEVABLE by score edits (oracle/regression) vs is a single
FIXED content-addressed redirect enough (the content-addressing residual)?
"""
import os, sys, json
import torch, numpy as np
OUT=os.environ.get("OUTDIR","."); torch.set_grad_enabled(False)
def build(seed):
    G=torch.Generator().manual_seed(seed); d=64; nf=40
    Q,_=torch.linalg.qr(torch.randn(d,d,generator=G)); F=Q[:nf]
    P,X,Y=0,1,2; filler=list(range(3,nf)); T=24
    aP,aX,aY=F[P],F[X],F[Y]; dh=4; sig=10.0; beta0=3.0
    WQ=np.sqrt(sig)*(torch.outer(aP,torch.eye(dh)[0])+torch.outer(aP,torch.eye(dh)[1]))
    WK=np.sqrt(sig)*(torch.outer(aX,torch.eye(dh)[0])+torch.outer(aY,torch.eye(dh)[1]))
    cm=torch.tril(torch.ones(T,T))
    def gen(nseq,sd):
        R=np.random.RandomState(sd); out=[]
        for s in range(nseq):
            x=torch.zeros(T,d); pos={0:[],1:[]}
            for t in range(1,T):
                r=R.rand()
                if r<0.34: dk=R.randint(2); x[t]+=R.uniform(0.8,1.2)*F[X if dk==0 else Y]; pos[dk].append(t)
                elif r<0.5: x[t]+=R.uniform(0.8,1.2)*F[P]
                else:
                    for _ in range(R.randint(1,3)): x[t]+=R.uniform(0.8,1.2)*F[filler[R.randint(len(filler))]]
            x+=1.0*F[P]+0.02*torch.tensor(R.randn(T,d),dtype=torch.float32)
            if pos[0] and pos[1]: out.append(dict(x=x,pos=pos))
        return out
    def sc(x): S=(x@WQ)@(x@WK).T; S[:,0]+=beta0; return S
    def mass_from(S,seq,dk): A=torch.softmax(S.masked_fill(cm==0,-1e9),-1); return sum(A[T-1,t].item() for t in seq["pos"][dk])
    return dict(F=F,P=P,X=X,Y=Y,aP=aP,aX=aX,aY=aY,sig=sig,T=T,cm=cm,gen=gen,sc=sc,mass_from=mass_from)
def run_seed(seed):
    M=build(seed); F,aP,aX,aY,sig,T=M["F"],M["aP"],M["aX"],M["aY"],M["sig"],M["T"]
    tr=M["gen"](300,seed+50); ev=M["gen"](300,seed+150)
    def edits(x,Rd=0.0,Cy=0.0):  # cut PxX (+ Rd*PxSink) (+ Cy*PxY compensation)
        S=M["sc"](x)-sig*torch.outer(x@aP,x@aX); S[:,0]+=Rd*(x@aP); S+=Cy*torch.outer(x@aP,x@aY); return S
    def gdom(x): xp=x-(x@F[M["P"]]).unsqueeze(-1)*F[M["P"]]; return M["sc"](xp)
    def slice_mass(seqs,Sfn):
        bx=[M["mass_from"](Sfn(s["x"]),s,0) for s in seqs]; by=[M["mass_from"](Sfn(s["x"]),s,1) for s in seqs]; return np.mean(bx),np.mean(by)
    bx,by=slice_mass(ev,lambda x:M["sc"](x)); ox,oy=slice_mass(ev,gdom)
    def metrics(seqs,Sfn):
        mx,my=slice_mass(seqs,Sfn); return (bx-mx)/max(bx-ox,1e-6), abs(my-by)/max(by-oy,1e-6)
    out={"base":(bx,by),"off":(ox,oy)}
    out["naive"]=metrics(ev,lambda x:edits(x,0,0))
    out["gated_dom"]=metrics(ev,gdom)
    # fixed-Rd sweep -> best Ycol at Xrem near full
    best=None
    for Rd in np.linspace(0,40,21):
        xr,yc=metrics(ev,lambda x,Rd=Rd:edits(x,Rd,0))
        if xr>0.85 and (best is None or yc<best[1]): best=(Rd,yc,xr)
    out["redirect_bestfixed"]=dict(Rd=best[0],Ycol=best[1],Xrem=best[2]) if best else None
    # per-prompt ORACLE redirect: choose Rd per prompt so Y-mass matches baseline (binary search on Rd)
    yc_oracle=[]; xr_oracle=[]
    for s in ev:
        by_s=M["mass_from"](M["sc"](s["x"]),s,1); lo,hi=0.0,80.0
        for _ in range(25):
            mid=(lo+hi)/2; my=M["mass_from"](edits(s["x"],mid,0),s,1)
            if my>by_s: lo=mid
            else: hi=mid
        Rd=(lo+hi)/2; my=M["mass_from"](edits(s["x"],Rd,0),s,1); mx=M["mass_from"](edits(s["x"],Rd,0),s,0)
        yc_oracle.append(abs(my-by_s)); xr_oracle.append((M["mass_from"](M["sc"](s["x"]),s,0)-mx))
    out["redirect_oracle"]=dict(Ycol_abs=float(np.mean(yc_oracle)),Xrem_abs=float(np.mean(xr_oracle)),base_gap=float(bx-ox))
    # 2-coeff REGRESSION (fit Rd,Cy on TRAIN to hit Xrem=1,Ycol=0; frozen on EVAL)
    from itertools import product
    bxt,byt=slice_mass(tr,lambda x:M["sc"](x)); oxt,oyt=slice_mass(tr,gdom)
    bestRC=None
    for Rd in np.linspace(0,40,21):
        for Cy in np.linspace(0,12,13):
            mxt,myt=slice_mass(tr,lambda x,Rd=Rd,Cy=Cy:edits(x,Rd,Cy))
            xr=(bxt-mxt)/max(bxt-oxt,1e-6); yc=abs(myt-byt)/max(byt-oyt,1e-6)
            if xr>0.85:
                loss=yc
                if bestRC is None or loss<bestRC[0]: bestRC=(loss,Rd,Cy)
    Rd,Cy=bestRC[1],bestRC[2]; mxy=metrics(ev,lambda x:edits(x,Rd,Cy)); out["regression2"]=dict(Rd=Rd,Cy=Cy,Xrem=mxy[0],Ycol=mxy[1])
    return out
seeds=[0,1,2]; agg={}
for sd in seeds:
    r=run_seed(sd)
    for k in ["naive","gated_dom"]: agg.setdefault(k,[]).append(r[k])
    agg.setdefault("redirect_bestfixed",[]).append((r["redirect_bestfixed"]["Xrem"],r["redirect_bestfixed"]["Ycol"]))
    agg.setdefault("regression2",[]).append((r["regression2"]["Xrem"],r["regression2"]["Ycol"]))
    agg.setdefault("oracle",[]).append((r["redirect_oracle"]["Xrem_abs"]/r["redirect_oracle"]["base_gap"],r["redirect_oracle"]["Ycol_abs"]/r["redirect_oracle"]["base_gap"]))
print("method (mean over seeds)     X-removal   Y-collateral",flush=True)
res={}
for k in ["naive","gated_dom","redirect_bestfixed","regression2","oracle"]:
    a=np.array(agg[k]); xr,yc=a[:,0].mean(),a[:,1].mean(); res[k]=dict(Xrem=float(xr),Ycol=float(yc))
    tag="  <== clean (X removed, Y preserved)" if (xr>0.8 and yc<0.25) else ("  <- couples" if yc>0.5 else "  <- partial")
    print(f"  {k:20s}  {xr:.2f}        {yc:.2f}{tag}",flush=True)
json.dump(res,open(os.path.join(OUT,"synth_hier6.json"),"w"),indent=2,default=float)
print("\nDONE synth_hier6",flush=True)
