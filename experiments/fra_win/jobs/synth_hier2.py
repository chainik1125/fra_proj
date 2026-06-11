"""SYNTH_HIER2 — fixed build (calibrate sigma_mis vs sink so persona-on->domain-key mass ~0.85) +
AGGREGATE removal metric (means over sequences, robust). Same alpha-routing toy.
Expected if the mechanism is right: at alpha=1 cell-edit R_X~1 collat_Y~0; head-ablation R_X~1 collat_Y~1;
DoM R_X~1 collat_Y~1 style~1; mixed-prompt: cell cuts X only, gated-DoM couples X~Y.
"""
import os, sys, json
import torch, numpy as np
OUT=os.environ.get("OUTDIR","."); torch.set_grad_enabled(False)
g=torch.Generator().manual_seed(0)
def randu(n,d): v=torch.randn(n,d,generator=g); return v/v.norm(dim=1,keepdim=True)
d=64; ah=0.6; n=256; K=4; nP_ch=8; nD_ch=16
idxP=0; idxPch=list(range(1,9)); idxD=[9,10,11,12]; idxDch={k:list(range(13+k*16,13+(k+1)*16)) for k in range(K)}
base=randu(n,d); F=base.clone()
for c in idxPch: F[c]=ah*base[idxP]+np.sqrt(1-ah**2)*base[c]
for k in range(K):
    for c in idxDch[k]: F[c]=ah*base[idxD[k]]+np.sqrt(1-ah**2)*base[c]
F=F/F.norm(dim=1,keepdim=True)
rho=(F@F.T).abs(); rho=rho[~torch.eye(n,dtype=bool)].mean().item()
lam=0.1; Ahat=F@torch.linalg.inv(F.T@F+lam*torch.eye(d))
def rd(x,i): return x@Ahat[i]
T=32
def gen(nseq,fp=None,fd=None,seed=0):
    R=np.random.RandomState(seed); seqs=[]
    for s in range(nseq):
        zper=fp if fp is not None else int(R.rand()<0.5)
        if fd is not None: doms=list(fd)
        else: doms=[int(R.randint(K))] if R.rand()<0.5 else list(R.choice(K,2,replace=False))
        x=torch.zeros(T,d); dompos={k:[] for k in range(K)}
        for t in range(1,T):
            r=R.rand()
            if r<0.25 and doms:
                k=doms[R.randint(len(doms))]; c=idxDch[k][R.randint(nD_ch)]; x[t]+=R.uniform(0.8,1.2)*F[c]; dompos[k].append(t)
            elif r<0.40 and zper:
                x[t]+=R.uniform(0.8,1.2)*F[idxPch[R.randint(nP_ch)]]
            else:
                for _ in range(R.randint(1,4)): x[t]+=R.uniform(0.8,1.2)*F[int(R.randint(80,n))]
        if zper: x+=1.0*F[idxP]
        for k in doms: x[T-1]+=1.0*F[idxD[k]]
        x+=torch.tensor(0.02*R.randn(T,d),dtype=torch.float32)
        seqs.append(dict(x=x,zper=zper,doms=doms,dompos=dompos))
    return seqs
# routes
dh=16; E=torch.eye(dh)[:K]; beta0=5.0
aP=Ahat[idxP]/Ahat[idxP].norm(); bD=[Ahat[idxD[k]]/Ahat[idxD[k]].norm() for k in range(K)]
def build_WQK(sig):
    WQ=torch.zeros(d,dh); WK=torch.zeros(d,dh)
    for k in range(K): WQ+=np.sqrt(sig)*torch.outer(aP,E[k]); WK+=np.sqrt(sig)*torch.outer(bD[k],E[k])
    return WQ,WK
def raw_scores(x,WQ,WK):
    S=(x@WQ)@(x@WK).T; S[:,0]+=beta0; return S.masked_fill(torch.tril(torch.ones(T,T))==0,-1e9)
def attn(S): return torch.softmax(S,-1)
# calibrate sigma so persona-on final-query domain-key mass ~0.85
cal_on=gen(300,fp=1,seed=11); cal_off=gen(300,fp=0,seed=12)
def domkey_mass(seqs,WQ,WK):
    m=[]
    for s in seqs:
        if not s["doms"]: continue
        A=attn(raw_scores(s["x"],WQ,WK)); dm=sum(A[T-1,t].item() for k in s["doms"] for t in s["dompos"][k]); m.append(dm)
    return float(np.mean(m))
sig=6.0
for sig in [6,10,15,20,28,40]:
    WQ,WK=build_WQK(sig); on=domkey_mass(cal_on,WQ,WK); off=domkey_mass(cal_off,WQ,WK)
    print(f"sig={sig}: persona-on domain-key mass {on:.2f} | persona-off {off:.2f}",flush=True)
    if on>=0.8: break
WQ,WK=build_WQK(sig); omega=sig
# channels (raw): g_att,k = final-pos attention mass on domain-k positions; g_dir,k = uP*uD bilinear neuron
def g_att(A,seq,k): return sum(A[T-1,t].item() for t in seq["dompos"][k])
def g_dir(x,k): return (rd(x[T-1],idxP)*rd(x[T-1],idxD[k])).item()
def chan_means(seqs,k_for_seq):
    ga=[];gd=[]
    for s in seqs:
        ks=s["doms"] if k_for_seq is None else [k_for_seq]
        for k in ks:
            if k not in s["doms"]: continue
            A=attn(raw_scores(s["x"],WQ,WK)); ga.append(g_att(A,s,k)); gd.append(g_dir(s["x"],k))
    return np.mean(ga),np.mean(gd)
gon_a,gon_d=chan_means(cal_on,None); goff_a,goff_d=chan_means(cal_off,None)
wa=1.0/max(gon_a-goff_a,1e-6); wd=1.0/max(gon_d-goff_d,1e-6)
print(f"calibrated sig={sig}; channel gaps att={gon_a-goff_a:.3f} dir={gon_d-goff_d:.3f}",flush=True)
# m_k(alpha) = alpha*wa*g_att + (1-alpha)*wd*g_dir  (each channel on-off gap normalized to 1)
def m_k(x,seq,k,alpha,Sfn=None,head_off=False):
    A=attn(Sfn(x,seq) if Sfn else raw_scores(x,WQ,WK))
    ga=0.0 if head_off else wa*g_att(A,seq,k); gd=wd*g_dir(x,k); return alpha*ga+(1-alpha)*gd
# interventions
def cell_S(target):  # FRA: subtract (P x D_target) route
    return lambda x,seq: (raw_scores(x,WQ,WK) - omega*torch.outer(rd(x,idxP),rd(x,idxD[target]))).masked_fill(torch.tril(torch.ones(T,T))==0,-1e9)
def edge_S(target):  # position-oracle edge ablation
    def f(x,seq):
        S=raw_scores(x,WQ,WK)
        for t in seq["dompos"][target]: S[T-1,t]=-1e9
        return S
    return f
def proj_persona(x): fP=F[idxP]; return x-(x@fP).unsqueeze(-1)*fP
def proj_dom(x,target):
    P=torch.eye(d)
    for i in [idxD[target]]+idxDch[target]: fi=F[i]; P=P-torch.outer(fi,fi)
    return x@P
def style(x): return rd(x[T-1],idxP).item()
# AGGREGATE removal/collateral over a slice; persona-off baseline = remove tonic persona
def measure(seqs,alpha,method,Xk=0):
    def mods(x,seq,k):  # returns (Sfn, xmod, head_off) for a method
        if method=="cell": return cell_S(Xk),None,False
        if method=="edge": return edge_S(Xk),None,False
        if method=="head": return None,None,True
        if method=="dom_un": return None,proj_persona,False
        if method=="dom_gated": return (None,(proj_persona if Xk in seq["doms"] else (lambda z:z)),False)
        if method=="keyrm": return None,(lambda z:proj_dom(z,Xk)),False
        return None,None,False
    def mval(seqs2,k,intervene):
        v=[]
        for s in seqs2:
            if k not in s["doms"]: continue
            if intervene:
                Sfn,xmod,ho=mods(s["x"],s,k); x=xmod(s["x"]) if xmod else s["x"]
                v.append(m_k(x,s,k,alpha,Sfn=Sfn,head_off=ho))
            else: v.append(m_k(s["x"],s,k,alpha))
        return np.mean(v) if v else 0.0
    # persona-off: subtract tonic persona from every seq
    off=[dict(s,x=s["x"]-1.0*F[idxP]) for s in seqs]
    baseX=mval(seqs,Xk,False); offX=mval(off,Xk,False); ivX=mval(seqs,Xk,True)
    R_X=(baseX-ivX)/max(baseX-offX,1e-6)
    # collateral on Y = a different active domain
    Yk=1 if Xk==0 else 0
    baseY=mval(seqs,Yk,False); offY=mval(off,Yk,False); ivY=mval(seqs,Yk,True)
    C_Y=abs(baseY-ivY)/max(abs(baseY-offY),1e-6)
    # style change
    st=[]
    for s in seqs:
        _,xmod,_=mods(s["x"],s,Xk); st.append(abs(style(s["x"])-style(xmod(s["x"]) if xmod else s["x"]))/max(abs(style(s["x"])),1e-6))
    return R_X,C_Y,float(np.mean(st))
def pf_alpha(seqs,alpha,Xk=0):  # pattern-freeze -> recovers (1-alpha); alpha_hat=1-remaining
    base=[];frz=[]
    for s in seqs:
        if Xk not in s["doms"]: continue
        A=attn(raw_scores(s["x"],WQ,WK)); Aoff=attn(raw_scores(s["x"]-1.0*F[idxP],WQ,WK))
        base.append(alpha*wa*g_att(A,s,Xk)+(1-alpha)*wd*g_dir(s["x"],Xk))
        frz.append(alpha*wa*g_att(Aoff,s,Xk)+(1-alpha)*wd*g_dir(s["x"],Xk))
    off=[(1-alpha)*wd*g_dir(s["x"],Xk) for s in seqs if Xk in s["doms"]]  # both off
    rem=(np.mean(frz)-np.mean(off))/max(np.mean(base)-np.mean(off),1e-6)
    return 1-rem
ev=gen(800,fp=1,seed=21)
res={"rho":rho,"sig":sig,"alpha_sweep":[]}
print("\n=== alpha-sweep (aggregate) ===",flush=True)
for alpha in [0,0.25,0.5,0.75,1.0]:
    rc=measure(ev,alpha,"cell"); rd_=measure(ev,alpha,"dom_un"); ah_=pf_alpha(ev,alpha)
    res["alpha_sweep"].append(dict(alpha=alpha,alpha_hat=ah_,cell_R=rc[0],cell_CY=rc[1],dom_R=rd_[0],dom_CY=rc[1]))
    print(f"  alpha={alpha}: alpha_hat(pf)={ah_:.2f} | cell R_X={rc[0]:.2f} collatY={rc[1]:.2f} | DoM R_X={rd_[0]:.2f} collatY={rd_[1]:.2f}",flush=True)
print("\n=== selectivity @ alpha=1 (R_X | collat_Y | style_chg) ===",flush=True)
res["selectivity"]={}
for m in ["cell","dom_un","dom_gated","keyrm","head","edge"]:
    r=measure(ev,1.0,m); res["selectivity"][m]=dict(R_X=r[0],collat_Y=r[1],style=r[2]); print(f"  {m:10s} R_X={r[0]:.2f} collat_Y={r[1]:.2f} style={r[2]:.2f}",flush=True)
print("\n=== MIXED-PROMPT X&Y @ alpha=1 (X-removal, Y-collateral) ===",flush=True)
mx=gen(500,fp=1,fd=[0,1],seed=31); res["mixed"]={}
for m in ["cell","dom_gated","head","edge"]:
    r=measure(mx,1.0,m); res["mixed"][m]=dict(X_rem=r[0],Y_col=r[1]); tag="<- X only" if (r[1]<0.2 and r[0]>0.7) else ("<- couples X~Y" if r[1]>0.5 else "")
    print(f"  {m:10s} X-removal={r[0]:.2f} Y-collateral={r[1]:.2f} {tag}",flush=True)
json.dump(res,open(os.path.join(OUT,"synth_hier2.json"),"w"),indent=2,default=float)
print("\nDONE synth_hier2",flush=True)
