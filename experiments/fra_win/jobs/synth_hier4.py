"""SYNTH_HIER4 — the CRUX: softmax redistribution and the redirect fix. Clean orthonormal toy.
Mixed prompt: persona-query (final) attends to X-keys AND Y-keys (+sink). Misaligned output for domain
k = attention mass on k-keys. Interventions on scores:
  naive cell-cut (P x X): S -= sig*uP*uX           -> X-attn cut, but softmax floods Y (redistribution).
  REDIRECT cell-edit:     cut (P x X) + boost (P x sink) -> freed mass -> sink, Y preserved.  [the fix]
  gated-DoM (per-position): remove persona at query -> ALL domain attn -> sink (couples X & Y).
  position-oracle edge: zero S[T, X-positions] (== naive cut for these keys, same redistribution).
Question: does the redirect cell-edit give X-removal with Y-preservation where no per-position method can?
"""
import os, sys, json
import torch, numpy as np
OUT=os.environ.get("OUTDIR","."); torch.set_grad_enabled(False)
G=torch.Generator().manual_seed(0); d=64; nf=40
Q,_=torch.linalg.qr(torch.randn(d,d,generator=G)); F=Q[:nf]
P,X,Y=0,1,2; filler=list(range(3,nf)); T=24
aP,aX,aY=F[P],F[X],F[Y]  # orthonormal -> exact reads
dh=4; sig=8.0; beta0=2.0
WQ=np.sqrt(sig)*(torch.outer(aP,torch.eye(dh)[0])+torch.outer(aP,torch.eye(dh)[1]))
WK=np.sqrt(sig)*(torch.outer(aX,torch.eye(dh)[0])+torch.outer(aY,torch.eye(dh)[1]))
cm=torch.tril(torch.ones(T,T))
def gen(nseq,fp=1,fd=(0,1),seed=0):
    R=np.random.RandomState(seed); out=[]
    for s in range(nseq):
        x=torch.zeros(T,d); pos={0:[],1:[]}; doms=list(fd)
        for t in range(1,T):
            r=R.rand()
            if r<0.30:
                dk=doms[R.randint(len(doms))]; x[t]+=R.uniform(0.8,1.2)*F[X if dk==0 else Y]; pos[dk].append(t)
            elif r<0.45 and fp: x[t]+=R.uniform(0.8,1.2)*F[P]
            else:
                for _ in range(R.randint(1,3)): x[t]+=R.uniform(0.8,1.2)*F[filler[R.randint(len(filler))]]
        if fp: x+=1.0*F[P]
        x[T-1]+=1.0*F[X]+1.0*F[Y]
        x+=0.02*torch.tensor(R.randn(T,d),dtype=torch.float32); out.append(dict(x=x,pos=pos))
    return out
def scores(x): S=(x@WQ)@(x@WK).T; S[:,0]+=beta0; return S.masked_fill(cm==0,-1e9)
def attn(S): return torch.softmax(S,-1)
def mass(A,seq,dk): return sum(A[T-1,t].item() for t in seq["pos"][dk])  # misaligned output for domain dk
def S_naive(x): return (scores(x) - sig*torch.outer(x@aP,x@aX)).masked_fill(cm==0,-1e9)
def S_redirect(x,Rdir=12.0):
    S=scores(x)-sig*torch.outer(x@aP,x@aX); S[:,0]+=Rdir*(x@aP); return S.masked_fill(cm==0,-1e9)
def S_gateddom(x):  # remove persona everywhere then recompute (per-position gate at the query)
    xp=x-(x@F[P]).unsqueeze(-1)*F[P]; return scores(xp)
def S_edge(x,seq):
    S=scores(x)
    for t in seq["pos"][0]: S[T-1,t]=-1e9
    return S
ev=gen(600,fp=1,fd=(0,1),seed=7)
# baselines: persona-on (full) vs persona-off (persona removed) attention masses
def slice_mass(Sfn):
    mx=[];my=[]
    for s in ev:
        A=attn(Sfn(s["x"],s) if Sfn.__code__.co_argcount==2 else Sfn(s["x"]))
        mx.append(mass(A,s,0)); my.append(mass(A,s,1))
    return np.mean(mx),np.mean(my)
base_x,base_y=slice_mass(lambda x: scores(x))
off_x,off_y=slice_mass(lambda x: S_gateddom(x))  # persona-off => both ~ sink
print(f"baseline mixed-prompt misaligned mass: X={base_x:.3f} Y={base_y:.3f} ; persona-off X={off_x:.3f} Y={off_y:.3f}",flush=True)
res={"base":(base_x,base_y),"off":(off_x,off_y),"methods":{}}
for name,Sfn in [("naive_cell",lambda x: S_naive(x)),("redirect_cell",lambda x: S_redirect(x)),
                 ("gated_dom",lambda x: S_gateddom(x)),("edge_oracle",S_edge)]:
    mx,my=slice_mass(Sfn)
    Xrem=(base_x-mx)/max(base_x-off_x,1e-6)      # fractional removal of X misalignment
    Ycol=abs(my-base_y)/max(base_y-off_y,1e-6)   # change in Y misalignment, rel to Y's persona-gap
    res["methods"][name]=dict(mX=mx,mY=my,Xrem=Xrem,Ycol=Ycol)
    tag="<-- X removed, Y PRESERVED (WIN)" if (Xrem>0.8 and Ycol<0.2) else ("<- couples/collateral" if Ycol>0.4 else "")
    print(f"  {name:14s} mX={mx:.3f} mY={my:.3f} | X-removal={Xrem:.2f} Y-collateral={Ycol:.2f} {tag}",flush=True)
# redirect strength sweep (does more redirect -> cleaner?)
print("\n  redirect-strength sweep (Rdir): X-removal, Y-collateral",flush=True); res["rdir"]=[]
for Rd in [0,2,6,12,24]:
    mx,my=slice_mass(lambda x: S_redirect(x,Rd))
    Xr=(base_x-mx)/max(base_x-off_x,1e-6); Yc=abs(my-base_y)/max(base_y-off_y,1e-6)
    res["rdir"].append(dict(Rdir=Rd,Xrem=Xr,Ycol=Yc)); print(f"    Rdir={Rd:3d}: X-rem={Xr:.2f} Y-col={Yc:.2f}",flush=True)
json.dump(res,open(os.path.join(OUT,"synth_hier4.json"),"w"),indent=2,default=float)
print("\nDONE synth_hier4",flush=True)
