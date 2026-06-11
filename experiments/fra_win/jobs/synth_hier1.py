"""SYNTH_HIER1 — hand-built broad×broad cell-cutting toy (THEORY_HIERARCHY.md §4). Core E1+E2+E3.
Persona->domain generalization with a PLANTED tunable routing split alpha: fraction alpha via an
ATTENTION score-route (G-score, FRA-cuttable), (1-alpha) via a bilinear DIRECT neuron (G-post,
direction/MLP analog), total persona-conditional gap held constant.
Validates: (P5) pattern-freeze recovers (1-alpha); (P1/P8) at alpha=1 the FRA cell-edit removes the
persona->X link selectively where DoM/head-ablation collateral Y; (P2) the MIXED-PROMPT decisive case
(per-position gate couples X-removal to Y-collateral ~1:1; per-edge cell-edit does not); (alpha=0) FRA
correctly cuts nothing while DoM still works.
"""
import os, sys, json
import torch, numpy as np
OUT=os.environ.get("OUTDIR","."); dev="cpu"; torch.set_grad_enabled(False)
g=torch.Generator().manual_seed(0)
def randu(n,d): v=torch.randn(n,d,generator=g); return v/v.norm(dim=1,keepdim=True)
# ---- dictionary with hierarchy ----
d=64; ah=0.6
# indices: 0=P, 1..8 persona children, 9..12 = D1..D4 parents, 13..76 domain children (16/dom), 77=C, rest filler
nP_ch=8; K=4; nD_ch=16
idxP=0; idxPch=list(range(1,1+nP_ch)); idxD=[9,10,11,12]
idxDch={k:list(range(13+k*nD_ch,13+(k+1)*nD_ch)) for k in range(K)}
n=256
base=randu(n,d)
F=base.clone()
# persona children = ah*P + sqrt(1-ah^2)*g
for c in idxPch: F[c]=ah*base[idxP]+np.sqrt(1-ah**2)*base[c]
for k in range(K):
    for c in idxDch[k]: F[c]=ah*base[idxD[k]]+np.sqrt(1-ah**2)*base[c]
F=F/F.norm(dim=1,keepdim=True)
rho=(F@F.T).abs(); rho=rho[~torch.eye(n,dtype=bool)].mean().item()
# read duals: Ahat = F (F^T F + lam I)^-1  -> <x,Ahat_i> ~ u_i with interference
lam=0.1; Ahat=F@torch.linalg.inv(F.T@F+lam*torch.eye(d))
def rd(x,i): return x@Ahat[i]   # read feature i activation from residual x ([*,d]->[*])
# ---- sequence generator ----
T=32
def gen(nseq,force_persona=None,force_doms=None):
    seqs=[]
    for s in range(nseq):
        zper=1 if force_persona is None else force_persona
        if force_persona is None: zper=int(torch.rand(1,generator=g)<0.5)
        if force_doms is not None: doms=list(force_doms)
        else:
            if torch.rand(1,generator=g)<0.5: doms=[int(torch.randint(K,(1,),generator=g))]
            else: doms=list(np.random.RandomState(s+1).choice(K,2,replace=False))
        x=torch.zeros(T,d); dompos={k:[] for k in range(K)}
        for t in range(1,T):
            r=torch.rand(1,generator=g).item()
            if r<0.25 and doms:
                k=doms[int(torch.randint(len(doms),(1,),generator=g))]; c=idxDch[k][int(torch.randint(nD_ch,(1,),generator=g))]
                x[t]+=torch.empty(1).uniform_(0.8,1.2,generator=g)*F[c]; dompos[k].append(t)
            elif r<0.40 and zper:
                c=idxPch[int(torch.randint(nP_ch,(1,),generator=g))]; x[t]+=torch.empty(1).uniform_(0.8,1.2,generator=g)*F[c]
            else:
                for _ in range(int(torch.randint(1,4,(1,),generator=g))):
                    c=int(torch.randint(80,n,(1,),generator=g)); x[t]+=torch.empty(1).uniform_(0.8,1.2,generator=g)*F[c]
        if zper: x+= 1.0*F[idxP]                      # tonic persona at every position
        for k in doms: x[T-1]+=1.0*F[idxD[k]]         # tonic domain summary at final pos
        x+=0.02*torch.randn(T,d,generator=g)
        seqs.append(dict(x=x,zper=zper,doms=doms,dompos=dompos))
    return seqs
# ---- hand-built H_mis head: persona-query -> all-domain-keys mis-routes + sink ----
dh=16; E=torch.eye(dh)[:K]   # orthonormal head-basis directions, one per domain route
sig_mis=6.0; beta0=5.0
aP=Ahat[idxP]/Ahat[idxP].norm()
bD=[Ahat[idxD[k]]/Ahat[idxD[k]].norm() for k in range(K)]
WQ=torch.zeros(d,dh); WK=torch.zeros(d,dh)
for k in range(K):
    WQ+=np.sqrt(sig_mis)*torch.outer(aP,E[k]); WK+=np.sqrt(sig_mis)*torch.outer(bD[k],E[k])
def scores(x):
    S=(x@WQ)@(x@WK).T          # S[q,k] = sum_route sig <x_q,aP><x_k,bD>
    S[:,0]+=beta0              # sink at position 0
    cm=torch.tril(torch.ones(T,T)); S=S.masked_fill(cm==0,-1e9)
    return S
def attn(S): return torch.softmax(S,-1)
# g_att,k = attention mass at final pos on domain-k key positions (the transported misaligned signal)
def g_att_raw(A,seq):
    return torch.tensor([sum(A[T-1,t].item() for t in seq["dompos"][k]) for k in range(K)])
def g_dir_raw(x,k):  # bilinear direct neuron: persona x domain at final position (pattern-invariant)
    return (rd(x[T-1],idxP)*rd(x[T-1],idxD[k])).item()
# ---- calibrate normalizers: gap (persona-on minus persona-off) = 1 for each channel, per domain ----
cal_on=gen(400,force_persona=1); cal_off=gen(400,force_persona=0)
def mean_gatt(seqs,Sfn=scores):
    acc=torch.zeros(K); cnt=torch.zeros(K)
    for s in seqs:
        A=attn(Sfn(s["x"])); ga=g_att_raw(A,s)
        for k in s["doms"]: acc[k]+=ga[k]; cnt[k]+=1
    return acc/cnt.clamp(min=1)
gatt_on=mean_gatt(cal_on); gatt_off=mean_gatt(cal_off)
gdir_on=torch.zeros(K);gdir_off=torch.zeros(K);c1=torch.zeros(K);c0=torch.zeros(K)
for s in cal_on:
    for k in s["doms"]: gdir_on[k]+=g_dir_raw(s["x"],k); c1[k]+=1
for s in cal_off:
    for k in s["doms"]: gdir_off[k]+=g_dir_raw(s["x"],k); c0[k]+=1
gdir_on/=c1.clamp(min=1); gdir_off/=c0.clamp(min=1)
att_gap=(gatt_on-gatt_off).clamp(min=1e-3); dir_gap=(gdir_on-gdir_off).clamp(min=1e-3)
def gatt_norm(A,seq,k): return (g_att_raw(A,seq)[k].item()-gatt_off[k].item())/att_gap[k].item()
def gdir_norm(x,k): return (g_dir_raw(x,k)-gdir_off[k].item())/dir_gap[k].item()
print(f"build OK: rho_bar={rho:.3f} att_gap={att_gap.tolist()} dir_gap={dir_gap.tolist()}",flush=True)
# ---- misaligned logit m_k(alpha) under an intervention; persona-style s ----
def m_of(seq,alpha,Sfn=None,xmod=None,head_off=False,k=None):
    x=seq["x"] if xmod is None else xmod(seq["x"])
    S=(Sfn(x,seq) if Sfn else scores(x))
    A=attn(S)
    ga=0.0 if head_off else gatt_norm(A,seq,k)
    gd=gdir_norm(x,k)
    return alpha*ga+(1-alpha)*gd
def persona_style(seq,xmod=None):
    x=seq["x"] if xmod is None else xmod(seq["x"]); return rd(x[T-1],idxP).item()
# ---- interventions (on scores S, on residual x, or head-off) ----
omega=sig_mis  # planted (P x D_1) coupling
def cell_edit_S(c=1.0,target=0):
    def f(x,seq):
        delta=c*omega*torch.outer(rd(x,idxP),rd(x,idxD[target]))  # the planted (P x D_target) route
        S=(x@WQ)@(x@WK).T - delta
        S[:,0]+=beta0
        cm=torch.tril(torch.ones(T,T)); S=S.masked_fill(cm==0,-1e9)
        return S
    return f
def dom_proj(target):  # key-side domain removal (project D_target parent+children out everywhere)
    P=torch.eye(d)
    for i in [idxD[target]]+idxDch[target]:
        fi=F[i]; P=P-torch.outer(fi,fi)/ (fi@fi)
    return lambda x: x@P
def persona_proj_all(x):  # ungated DoM
    fP=F[idxP]; return x-(x@fP).unsqueeze(-1)*fP
def persona_proj_gated(seq):  # context-gated DoM: remove persona only at query positions of X-containing seqs
    def f(x):
        if 0 in seq["doms"]:  # sequence contains domain-X
            fP=F[idxP]; return x-(x@fP).unsqueeze(-1)*fP
        return x
    return f
def edge_ablate_S(target=0):  # position-oracle: zero S[T,k] for k in domain-target positions
    def f(x,seq):
        S=scores(x)
        for t in seq["dompos"][target]: S[T-1,t]=-1e9
        return S
    return f
# ====================== E2: alpha-sweep (Fig 1) ======================
evalX=gen(600,force_persona=1)  # persona-on sequences; measure on those containing domain-X
def removal_and_collateral(seqs,alpha,method):
    RX=[]; CY=[]; AL=[]; ST=[]
    for s in seqs:
        if 0 not in s["doms"]: continue
        base_mX=m_of(s,alpha,k=0); off=0.0
        # persona-off baseline for X
        soff=dict(s); soff_x=s["x"]-1.0*F[idxP]  # remove tonic persona to get persona-off
        # measure delta m_X = base - intervened
        if method=="cell": iv=m_of(s,alpha,Sfn=cell_edit_S(1.0,0),k=0)
        elif method=="dom_un": iv=m_of(s,alpha,xmod=persona_proj_all,k=0)
        elif method=="dom_gated": iv=m_of(s,alpha,xmod=persona_proj_gated(s),k=0)
        elif method=="keyrm": iv=m_of(s,alpha,xmod=dom_proj(0),k=0)
        elif method=="head": iv=m_of(s,alpha,head_off=True,k=0)
        elif method=="edge": iv=m_of(s,alpha,Sfn=edge_ablate_S(0),k=0)
        else: iv=base_mX
        gap=base_mX-(alpha*0+(1-alpha)*0)  # persona-conditional gap ~ base_mX (off~0 by norm)
        RX.append((base_mX-iv)/max(abs(base_mX),1e-3))
        # collateral on Y (a preserved active domain != X, if present)
        ys=[k for k in s["doms"] if k!=0]
        if ys:
            yk=ys[0]; by=m_of(s,alpha,k=yk)
            if method=="cell": ivy=m_of(s,alpha,Sfn=cell_edit_S(1.0,0),k=yk)
            elif method=="dom_un": ivy=m_of(s,alpha,xmod=persona_proj_all,k=yk)
            elif method=="dom_gated": ivy=m_of(s,alpha,xmod=persona_proj_gated(s),k=yk)
            elif method=="keyrm": ivy=m_of(s,alpha,xmod=dom_proj(0),k=yk)
            elif method=="head": ivy=m_of(s,alpha,head_off=True,k=yk)
            elif method=="edge": ivy=m_of(s,alpha,Sfn=edge_ablate_S(0),k=yk)
            else: ivy=by
            CY.append(abs(by-ivy)/max(abs(by),1e-3))
        ST.append(abs(persona_style(s)-persona_style(s,xmod=(persona_proj_all if method=="dom_un" else (persona_proj_gated(s) if method=="dom_gated" else (lambda z:z)))))/max(abs(persona_style(s)),1e-3))
    return float(np.mean(RX)), (float(np.mean(CY)) if CY else 0.0), float(np.mean(ST))
def patternfreeze_alpha(seqs,alpha):
    # freeze attention to persona-OFF pattern -> removes G-score -> remaining gap = (1-alpha)
    rem=[]
    for s in seqs:
        if 0 not in s["doms"]: continue
        base=m_of(s,alpha,k=0)
        xoff=s["x"]-1.0*F[idxP]; Aoff=attn(scores(xoff))   # persona-off pattern
        ga_frozen=gatt_norm(Aoff,s,0); gd=gdir_norm(s["x"],0)
        frozen=alpha*ga_frozen+(1-alpha)*gd
        rem.append(frozen/max(abs(base),1e-3))
    return 1-float(np.mean(rem))   # alpha_hat = 1 - remaining_fraction
res={"rho":rho,"alpha_sweep":[]}
for alpha in [0,0.25,0.5,0.75,1.0]:
    rc_cell=removal_and_collateral(evalX,alpha,"cell")
    rc_dom=removal_and_collateral(evalX,alpha,"dom_un")
    ah_pf=patternfreeze_alpha(evalX,alpha)
    res["alpha_sweep"].append(dict(alpha=alpha,alpha_hat_pf=ah_pf,cell_removal=rc_cell[0],dom_removal=rc_dom[0],cell_collatY=rc_cell[1],dom_collatY=rc_dom[1]))
    print(f"alpha={alpha}: alpha_hat(pf)={ah_pf:.2f} | cell_removal={rc_cell[0]:.2f} (collatY {rc_cell[1]:.2f}) | DoM_removal={rc_dom[0]:.2f} (collatY {rc_dom[1]:.2f})",flush=True)
# ====================== E3: selectivity table at alpha=1 (Fig 2) ======================
print("\n=== SELECTIVITY @ alpha=1 (R_X | collat_Y | persona-style change) ===",flush=True)
res["selectivity"]={}
for m in ["cell","dom_un","dom_gated","keyrm","head","edge"]:
    rx,cy,st=removal_and_collateral(evalX,1.0,m)
    res["selectivity"][m]=dict(R_X=rx,collat_Y=cy,style_chg=st)
    print(f"  {m:10s} R_X={rx:.2f}  collat_Y={cy:.2f}  style_chg={st:.2f}",flush=True)
# ====================== E3b: MIXED-PROMPT decisive case (Fig 3) ======================
mixed=gen(500,force_persona=1,force_doms=[0,1])  # every seq has BOTH X and Y
print("\n=== MIXED-PROMPT X&Y (alpha=1): (X-removal, Y-collateral) per method ===",flush=True)
res["mixed"]={}
for m in ["cell","dom_gated","head","edge"]:
    rx,cy,_=removal_and_collateral(mixed,1.0,m)
    res["mixed"][m]=dict(X_removal=rx,Y_collat=cy)
    print(f"  {m:10s} X-removal={rx:.2f}  Y-collateral={cy:.2f}  {'<- cuts X only' if cy<0.2 and rx>0.7 else ('<- couples X~Y' if cy>0.5 else '')}",flush=True)
json.dump(res,open(os.path.join(OUT,"synth_hier1.json"),"w"),indent=2,default=float)
print("\nDONE synth_hier1",flush=True)
