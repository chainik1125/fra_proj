"""SYNTH_HIER3 — CLEAN CORE first (orthonormal features, exact reads), THEN superposition sweep.
Broad = a domain feature D0 that fires on MANY key positions; cutting (P x D0) severs persona's attention
to ALL D0 tokens at once (the broad cut). alpha-routing split (attention vs direct bilinear neuron).
Tests: alpha=1 cell-edit R_X~1 collat_Y~0 vs head/DoM collat_Y~1; pattern-freeze recovers (1-alpha);
MIXED-PROMPT: per-edge cell cuts X only, per-position gated-DoM couples X~Y; alpha=0 cell cuts nothing,
DoM works. Then sweep superposition (n features in d dims) -> linear-steer collateral grows, cell stays ~0.
"""
import os, sys, json
import torch, numpy as np
OUT=os.environ.get("OUTDIR","."); torch.set_grad_enabled(False)
def run(d, nfeat, label, tag):
    G=torch.Generator().manual_seed(0)
    # feature dictionary: orthonormal if d>=nfeat (clean), else random (superposed)
    if d>=nfeat:
        Q,_=torch.linalg.qr(torch.randn(d,d,generator=G)); F=Q[:nfeat]   # orthonormal rows
    else:
        F=torch.randn(nfeat,d,generator=G); F=F/F.norm(dim=1,keepdim=True)
    rho=(F@F.T).abs(); rho=rho[~torch.eye(nfeat,dtype=bool)].mean().item()
    lam=0.05; Ahat=F@torch.linalg.inv(F.T@F+lam*torch.eye(d))  # exact when orthonormal
    P,X,Y=0,1,2; filler=list(range(3,nfeat)); T=24
    def read(x,i): return x@Ahat[i]
    aP=Ahat[P]/Ahat[P].norm(); aX=Ahat[X]/Ahat[X].norm(); aY=Ahat[Y]/Ahat[Y].norm()
    dh=4; e0,e1=torch.eye(dh)[0],torch.eye(dh)[1]; sig=8.0; beta0=2.0
    WQ=np.sqrt(sig)*(torch.outer(aP,e0)+torch.outer(aP,e1))
    WK=np.sqrt(sig)*(torch.outer(aX,e0)+torch.outer(aY,e1))
    def gen(nseq,fp=None,fd=None,seed=0):
        R=np.random.RandomState(seed); out=[]
        for s in range(nseq):
            zper=fp if fp is not None else int(R.rand()<0.5)
            doms=list(fd) if fd is not None else ([R.randint(2)] if R.rand()<0.5 else [0,1])
            x=torch.zeros(T,d); pos={0:[],1:[]}
            for t in range(1,T):
                r=R.rand()
                if r<0.30 and doms:
                    dk=doms[R.randint(len(doms))]; x[t]+=R.uniform(0.8,1.2)*F[X if dk==0 else Y]; pos[dk].append(t)
                elif r<0.45 and zper: x[t]+=R.uniform(0.8,1.2)*F[P]  # persona token
                else:
                    for _ in range(R.randint(1,3)): x[t]+=R.uniform(0.8,1.2)*F[filler[R.randint(len(filler))]]
            if zper: x+=1.0*F[P]                                   # tonic persona
            if 0 in doms: x[T-1]+=1.0*F[X]
            if 1 in doms: x[T-1]+=1.0*F[Y]
            x+=0.02*torch.tensor(R.randn(T,d),dtype=torch.float32)
            out.append(dict(x=x,zper=zper,doms=doms,pos=pos))
        return out
    cm=torch.tril(torch.ones(T,T))
    def scores(x): S=(x@WQ)@(x@WK).T; S[:,0]+=beta0; return S.masked_fill(cm==0,-1e9)
    def attn(S): return torch.softmax(S,-1)
    def g_att(A,seq,dk): return sum(A[T-1,t].item() for t in seq["pos"][dk])
    def g_dir(x,dk): return (read(x[T-1],P)*read(x[T-1],X if dk==0 else Y)).item()
    # calibrate channel gaps (persona on/off) for normalization
    on=gen(300,fp=1,seed=1); off=gen(300,fp=0,seed=2)
    def cm_(seqs):
        ga=[];gd=[]
        for s in seqs:
            for dk in s["doms"]: A=attn(scores(s["x"])); ga.append(g_att(A,s,dk)); gd.append(g_dir(s["x"],dk))
        return np.mean(ga),np.mean(gd)
    ga1,gd1=cm_(on); ga0,gd0=cm_(off); wa=1/max(ga1-ga0,1e-6); wd=1/max(gd1-gd0,1e-6)
    mass_on=np.mean([sum(attn(scores(s["x"]))[T-1,t].item() for dk in s["doms"] for t in s["pos"][dk]) for s in on])
    # interventions
    def cell_S(dk): return lambda x: (scores(x) - sig*torch.outer(x@aP, x@(aX if dk==0 else aY))).masked_fill(cm==0,-1e9)
    def edge_S(dk):
        def f(x):
            S=scores(x)
            for s_ in [None]: pass
            return S
        return f
    def m_k(x,seq,dk,alpha,Sfn=None,head_off=False):
        A=attn(Sfn(x) if Sfn else scores(x)); ga=0.0 if head_off else wa*g_att(A,seq,dk); return alpha*ga+(1-alpha)*wd*g_dir(x,dk)
    def proj(x,dirs):
        for fi in dirs: x=x-(x@fi).unsqueeze(-1)*fi
        return x
    def style(x): return read(x[T-1],P).item()
    Poff=[F[P]]+[F[c] for c in []]  # persona-off baseline subtracts the persona direction
    def measure(seqs,alpha,method,Xk=0):
        Yk=1-Xk
        def apply(x,seq,dk):  # value of m for domain dk under method
            if method=="cell": return m_k(x,seq,dk,alpha,Sfn=cell_S(Xk))
            if method=="head": return m_k(x,seq,dk,alpha,head_off=True)
            if method=="dom_un": return m_k(proj(x,[F[P]]),seq,dk,alpha)
            if method=="dom_gated": return m_k(proj(x,[F[P]]) if Xk in seq["doms"] else x,seq,dk,alpha)
            if method=="keyrm": return m_k(proj(x,[F[X if Xk==0 else Y]]),seq,dk,alpha)
            if method=="edge":  # position-oracle: zero S[T,k] for k in X-positions
                def Sf(x2):
                    S=scores(x2)
                    for t in seq["pos"][Xk]: S[T-1,t]=-1e9
                    return S
                return m_k(x,seq,dk,alpha,Sfn=Sf)
            return m_k(x,seq,dk,alpha)
        def agg(dk,fn):
            v=[fn(s) for s in seqs if dk in s["doms"]]; return np.mean(v) if v else 0.0
        baseX=agg(Xk,lambda s:m_k(s["x"],s,Xk,alpha)); offX=agg(Xk,lambda s:m_k(proj(s["x"],[F[P]]),s,Xk,alpha))
        ivX=agg(Xk,lambda s:apply(s["x"],s,Xk))
        R_X=(baseX-ivX)/max(baseX-offX,1e-6)
        baseY=agg(Yk,lambda s:m_k(s["x"],s,Yk,alpha)); offY=agg(Yk,lambda s:m_k(proj(s["x"],[F[P]]),s,Yk,alpha))
        ivY=agg(Yk,lambda s:apply(s["x"],s,Yk)); C_Y=abs(baseY-ivY)/max(abs(baseY-offY),1e-6)
        st=np.mean([abs(style(s["x"])-style(proj(s["x"],[F[P]]) if method in("dom_un","dom_gated") and (method=="dom_un" or Xk in s["doms"]) else s["x"])) /max(abs(style(s["x"])),1e-6) for s in seqs])
        return R_X,C_Y,float(st)
    def pf(seqs,alpha,Xk=0):
        b=[];fr=[];of=[]
        for s in seqs:
            if Xk not in s["doms"]: continue
            A=attn(scores(s["x"])); Ao=attn(scores(proj(s["x"],[F[P]])))
            b.append(alpha*wa*g_att(A,s,Xk)+(1-alpha)*wd*g_dir(s["x"],Xk))
            fr.append(alpha*wa*g_att(Ao,s,Xk)+(1-alpha)*wd*g_dir(s["x"],Xk))
            of.append((1-alpha)*wd*g_dir(proj(s["x"],[F[P]]),Xk))
        return 1-(np.mean(fr)-np.mean(of))/max(np.mean(b)-np.mean(of),1e-6)
    ev=gen(800,fp=1,seed=21); R={"label":label,"rho":rho,"mass_on":float(mass_on),"alpha":[],"sel":{},"mixed":{}}
    print(f"\n##### {label}: rho_bar={rho:.3f}, persona-on domain mass={mass_on:.2f} #####",flush=True)
    for alpha in [0,0.5,1.0]:
        rc=measure(ev,alpha,"cell"); rd_=measure(ev,alpha,"dom_un"); a=pf(ev,alpha)
        R["alpha"].append(dict(alpha=alpha,ahat=a,cell_R=rc[0],cell_CY=rc[1],dom_R=rd_[0],dom_CY=rd_[1]))
        print(f"  a={alpha}: ahat(pf)={a:.2f} | cell R_X={rc[0]:.2f} CY={rc[1]:.2f} | DoM R_X={rd_[0]:.2f} CY={rd_[1]:.2f}",flush=True)
    if tag=="core":
        print("  -- selectivity @ alpha=1 --",flush=True)
        for m in ["cell","dom_un","dom_gated","keyrm","head","edge"]:
            r=measure(ev,1.0,m); R["sel"][m]=dict(R_X=r[0],CY=r[1],style=r[2]); print(f"     {m:9s} R_X={r[0]:.2f} CY={r[1]:.2f} style={r[2]:.2f}",flush=True)
        mx=gen(500,fp=1,fd=[0,1],seed=31)
        print("  -- MIXED X&Y @ alpha=1 (X-rem, Y-collat) --",flush=True)
        for m in ["cell","dom_gated","head","edge"]:
            r=measure(mx,1.0,m); R["mixed"][m]=dict(Xr=r[0],Yc=r[1]); print(f"     {m:9s} Xrem={r[0]:.2f} Ycol={r[1]:.2f} {'<X only' if (r[1]<0.2 and r[0]>0.7) else ('<couples' if r[1]>0.5 else '')}",flush=True)
    return R
allres=[run(128,64,"CORE (orthonormal, no superposition)","core")]
for d in [128,64,32,16]:
    allres.append(run(d,128,f"superposition d={d}","sweep"))
json.dump(allres,open(os.path.join(OUT,"synth_hier3.json"),"w"),indent=2,default=float)
print("\nDONE synth_hier3",flush=True)
