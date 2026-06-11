"""SYNTH_HIER5 — clean confirmation of the REDIRECT result. Fix the final-position self-attention spike
(remove the domain tonic at T-1 so the persona-query attends to the domain-TOKEN positions). Verify clean
persona-conditional gaps, then confirm: redirect cell-edit removes persona->X misalignment while preserving
persona->Y, where naive-cut redistributes and per-position gate couples. Multi-seed for error bars.
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
    def gen(nseq,fp=1,sd=0):
        R=np.random.RandomState(sd); out=[]
        for s in range(nseq):
            x=torch.zeros(T,d); pos={0:[],1:[]}
            for t in range(1,T):
                r=R.rand()
                if r<0.34:
                    dk=R.randint(2); x[t]+=R.uniform(0.8,1.2)*F[X if dk==0 else Y]; pos[dk].append(t)
                elif r<0.5 and fp: x[t]+=R.uniform(0.8,1.2)*F[P]
                else:
                    for _ in range(R.randint(1,3)): x[t]+=R.uniform(0.8,1.2)*F[filler[R.randint(len(filler))]]
            if fp: x+=1.0*F[P]                       # tonic persona; NO domain tonic at final pos (fix)
            x+=0.02*torch.tensor(R.randn(T,d),dtype=torch.float32)
            if pos[0] and pos[1]: out.append(dict(x=x,pos=pos))   # keep mixed prompts (both X and Y present)
        return out
    def scores(x): S=(x@WQ)@(x@WK).T; S[:,0]+=beta0; return S.masked_fill(cm==0,-1e9)
    def A(x,Sfn=None): return torch.softmax((Sfn(x) if Sfn else scores(x)),-1)
    def mass(a,seq,dk): return sum(a[T-1,t].item() for t in seq["pos"][dk])
    def S_naive(x): return (scores(x)-sig*torch.outer(x@aP,x@aX)).masked_fill(cm==0,-1e9)
    def S_redir(x,Rd=12.0):
        S=scores(x)-sig*torch.outer(x@aP,x@aX); S[:,0]+=Rd*(x@aP); return S.masked_fill(cm==0,-1e9)
    def S_gdom(x): xp=x-(x@F[P]).unsqueeze(-1)*F[P]; return scores(xp)
    return dict(gen=gen,A=A,mass=mass,Sn=S_naive,Sr=S_redir,Sg=S_gdom,scores=scores)
def eval_seed(seed):
    M=build(seed); ev=M["gen"](400,fp=1,sd=seed+100)
    def sm(Sfn):
        mx=[m for s in ev for m in [M["mass"](M["A"](s["x"],Sfn),s,0)]]; my=[M["mass"](M["A"](s["x"],Sfn),s,1) for s in ev]
        return np.mean(mx),np.mean(my)
    bx,by=sm(None); ox,oy=sm(M["Sg"])  # persona-off via gated-dom = both to sink
    out={"base":(bx,by),"off":(ox,oy)}
    for name,Sfn in [("naive",M["Sn"]),("redirect",M["Sr"]),("gated_dom",M["Sg"])]:
        mx,my=sm(Sfn); out[name]=dict(Xrem=(bx-mx)/max(bx-ox,1e-6),Ycol=abs(my-by)/max(by-oy,1e-6),mX=mx,mY=my)
    return out
seeds=[0,1,2]; agg={k:{"Xrem":[],"Ycol":[]} for k in ["naive","redirect","gated_dom"]}; bases=[]
for sd in seeds:
    r=eval_seed(sd); bases.append((r["base"],r["off"]))
    for k in agg: agg[k]["Xrem"].append(r[k]["Xrem"]); agg[k]["Ycol"].append(r[k]["Ycol"])
print(f"mixed-prompt misaligned mass (mean over seeds): base X={np.mean([b[0][0] for b in bases]):.3f} Y={np.mean([b[0][1] for b in bases]):.3f} | persona-off X={np.mean([b[1][0] for b in bases]):.3f} Y={np.mean([b[1][1] for b in bases]):.3f}",flush=True)
print("\nmethod        X-removal (mean±sd)   Y-collateral (mean±sd)",flush=True)
res={"seeds":seeds,"methods":{}}
for k in ["naive","redirect","gated_dom"]:
    xr=np.array(agg[k]["Xrem"]); yc=np.array(agg[k]["Ycol"]); res["methods"][k]=dict(Xrem_mean=xr.mean(),Xrem_sd=xr.std(),Ycol_mean=yc.mean(),Ycol_sd=yc.std())
    tag="  <-- X removed, Y PRESERVED (the broad-cut WIN)" if (xr.mean()>0.8 and yc.mean()<0.25) else ("  <- couples/redistributes" if yc.mean()>0.5 else "")
    print(f"  {k:11s}  {xr.mean():.2f} ± {xr.std():.2f}        {yc.mean():.2f} ± {yc.std():.2f}{tag}",flush=True)
json.dump(res,open(os.path.join(OUT,"synth_hier5.json"),"w"),indent=2,default=float)
print("\nDONE synth_hier5",flush=True)
