"""IC4 — the rigorous flip: does FRA beat the SAME strong baselines that beat it on the weight-baked
sleeper (DoM and conventional SAE steering), on the in-context backdoor?
Baselines computed exactly as in K1/K8, from a poisoned(ON)-vs-clean(OFF) contrast set:
  DoM      : v = mean_ON(resid_L) - mean_OFF(resid_L) at the deployment position; subtract a*v at trigger positions.
  conv-SAE : rank SAE features by act-diff (mean_ON - mean_OFF) at the deployment position; gated-remove top-K
             (subtract c*z_f*W_dec[f]) wherever they fire (self-localizing).
Plus FRA-QK (attention edge) and payload-suppress (output). Metric: ASR-suppression vs held-out collateral
(KL on text with BOTH trigger and payload). Content-addressed everywhere; matched via Pareto.
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gpt2",device=dev); model.eval(); tok=model.tokenizer
IND=[(5,5),(6,9),(5,1),(7,10),(7,2)]; FLAYERS=sorted(set(L for L,H in IND)); L0=min(FLAYERS); Llast=model.cfg.n_layers-1
DL=6  # layer for DoM/conv-SAE (mid, post-trigger-detect)
fsaes={L:(SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev)) for L in FLAYERS}
fsaes={L:(s[0] if isinstance(s,tuple) else s) for L,s in fsaes.items()}
dsae=fsaes[DL]; W_U=model.W_U
def fra_ph(tt):
    _,c=model.run_with_cache(tt,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in FLAYERS])
    H={}; resid={L:c[f"blocks.{L}.hook_resid_pre"][0] for L in FLAYERS}
    for (L,Hh) in IND:
        fe=fsaes[L].encode(c[f"blocks.{L}.hook_resid_pre"][0]).float(); xh=fe@fsaes[L].W_dec.float()+fsaes[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,fsaes[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
        H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H,resid
def primer_pairs(HF,edge,M=12):
    P={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; loc=np.where((d["qq"]==edge[0])&(d["kk"]==edge[1]))[0]; loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
        P[(L,Hh)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in loc)
    return P
def delta_content(HF,P,seq):
    byL={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; dd=np.zeros((seq,seq)); Ps=P[(L,Hh)]
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        byL.setdefault(L,{})[Hh]=dd
    return byL
def patch_fra(tt,byL,c):
    seq=tt.shape[1]; hooks=[]
    for L,hd in byL.items():
        td={Hh:torch.tensor(dd,device=dev,dtype=torch.float32)*c for Hh,dd in hd.items()}
        def mk(td):
            def hook(s,hook):
                for Hh,sd in td.items(): s[0,Hh,:seq,:seq]-=sd
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(td)))
    return model.run_with_hooks(tt,fwd_hooks=hooks)[0]
def dom_run(tt,positions,vD,a):  # subtract a*vD (unit) scaled by local norm at trigger positions, resid_pre DL
    def hook(act,hook):
        for p in positions:
            if p<tt.shape[1]: act[0,p,:]=act[0,p,:]-a*vD*act[0,p,:].norm()
        return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{DL}.hook_resid_pre",hook)])[0]
def conv_run(tt,feats_idx,c):  # gated SAE removal at resid_pre DL: subtract c*z_f*W_dec[f] wherever active
    Wd=dsae.W_dec[feats_idx].float()  # [K,d]
    def hook(act,hook):
        x=act[0].float(); z=dsae.encode(x).float()[:,feats_idx]   # [seq,K]
        act[0]=x-(c*z)@Wd
        return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{DL}.hook_resid_pre",hook)])[0]
def paysupp(tt,Pid,s):
    uP=W_U[:,Pid].float(); uP=uP/uP.norm()
    def hook(act,hook): act[0]=act[0]-s*(act[0]@uP).unsqueeze(-1)*uP; return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{Llast}.hook_resid_post",hook)])[0]
def klsum(p,q):
    lp=torch.log_softmax(p.float(),-1); lq=torch.log_softmax(q.float(),-1)
    return (lp.exp()*(lp-lq)).sum(-1).sum().item()

CASES=[(" bank"," river","The river bank was crowded. He left the bank and crossed the river. By dusk the river hid the bank entirely."),
       (" fire"," water","The fire needed water fast. More water hit the fire. Without water the fire would have razed the street."),
       (" king"," crown","The king lost his crown. A new crown was made. The old king wore the crown while the young king watched."),
       (" doctor"," patient","The doctor calmed the patient. Another patient waited. The doctor told each patient a second doctor would help.")]
FC=[1,2,4,8,16]; DC=[0.25,0.5,1,2,4,8]; CC=[1,2,4,8,16,32]; PC=[0.5,1,2,4,8]
NSET=14; N=20; qpos=1+N+10; kpos=12
def mkseq(seed,Tid=None,Pid=None):
    g=torch.Generator().manual_seed(seed); R=(torch.randperm(40000,generator=g)[:N]+1000).tolist()
    if Tid is not None: R[10]=Tid; R[11]=Pid
    return torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0)
rows=[]
for T,P,htext in CASES:
    Tid=tok.encode(T); Pid=tok.encode(P)
    if len(Tid)!=1 or len(Pid)!=1: continue
    Tid=Tid[0]; Pid=Pid[0]
    # contrast sets: ON (trigger->payload planted) vs OFF (random) -- resid at DL, deployment pos
    on_acts=[]; off_acts=[]; on_feat=[]; off_feat=[]
    for s in range(NSET):
        ttn=mkseq(1000+s,Tid,Pid); a=model.run_with_cache(ttn,names_filter=[f"blocks.{DL}.hook_resid_pre"])[1][f"blocks.{DL}.hook_resid_pre"][0]
        on_acts.append(a[qpos]); on_feat.append(dsae.encode(a[qpos].float()).float())
        tto=mkseq(5000+s); b=model.run_with_cache(tto,names_filter=[f"blocks.{DL}.hook_resid_pre"])[1][f"blocks.{DL}.hook_resid_pre"][0]
        off_acts.append(b[qpos]); off_feat.append(dsae.encode(b[qpos].float()).float())
    vD=torch.stack(on_acts).mean(0)-torch.stack(off_acts).mean(0); vD=vD/(vD.norm()+1e-6)
    actdiff=(torch.stack(on_feat).mean(0)-torch.stack(off_feat).mean(0))
    convK=torch.topk(actdiff,12).indices.tolist()
    # eval backdoor seq (seed 0)
    tt=mkseq(0,Tid,Pid); seq=tt.shape[1]
    base=torch.softmax(model(tt)[0][qpos].float(),-1)[Pid].item()
    if base<0.2: print(f"skip {T} ASR {base:.2f}",flush=True); continue
    def asr_s(lg): return 1-torch.softmax(lg[qpos].float(),-1)[Pid].item()/base
    HF,resid=fra_ph(tt); Pp=primer_pairs(HF,(qpos,kpos)); byL_supp=delta_content(HF,Pp,seq)
    def fsupp(c): return asr_s(patch_fra(tt,byL_supp,c))
    # held-out
    hids=[tok.bos_token_id]+tok.encode(htext); ht=torch.tensor(hids,device=dev).unsqueeze(0); hseq=ht.shape[1]
    hT=[i for i,t in enumerate(hids) if t==Tid]; hclean=model(ht)[0]
    HFh,_=fra_ph(ht); byLh=delta_content(HFh,Pp,hseq)
    trig_pos=[1+10,1+N+10]
    cur_fra=[(fsupp(c), klsum(hclean,patch_fra(ht,byLh,c))) for c in FC]
    cur_dom=[(asr_s(dom_run(tt,trig_pos,vD,a)), klsum(hclean,dom_run(ht,hT,vD,a))) for a in DC]
    cur_conv=[(asr_s(conv_run(tt,convK,c)), klsum(hclean,conv_run(ht,convK,c))) for c in CC]
    cur_pay=[(asr_s(paysupp(tt,Pid,s)), klsum(hclean,paysupp(ht,Pid,s))) for s in PC]
    rows.append(dict(T=T,P=P,base=base,fra=cur_fra,dom=cur_dom,conv=cur_conv,pay=cur_pay))
    print(f"{T.strip()}->{P.strip()} ASR {base:.2f}:",flush=True)
    for k in ["fra","dom","conv","pay"]:
        print(f"  {k:5}: {[(round(a,2),round(b,2)) for a,b in rows[-1][k]]}",flush=True)
def at(curve,t):
    xs=[a for a,b in curve]; ys=[b for a,b in curve]
    if max(xs)<t: return None
    o=np.argsort(xs); return float(np.interp(t,np.array(xs)[o],np.array(ys)[o]))
print(f"\n=== in-context backdoor, held-out collateral @ 80% ASR-suppression (n={len(rows)}) ===",flush=True)
for k,lab in [("fra","FRA-QK (attention edge)"),("dom","DoM (mean-diff, K8 winner)"),("conv","conv-SAE (act-diff gated, K8)"),("pay","payload-suppress (output)")]:
    v=[at(r[k],0.8) for r in rows]; v=[x for x in v if x is not None]
    if v: print(f"  {lab:34}: {np.mean(v):.3f} ± {np.std(v):.3f} (n={len(v)})",flush=True)
json.dump({"rows":rows},open(os.path.join(OUT,"ic4.json"),"w"),indent=2,default=float)
plt.figure(figsize=(6.6,4.8))
for k,lab,c in [("fra","FRA-QK (attention edge)","C0"),("dom","DoM / mean-diff (won on weight-baked sleeper)","C3"),("conv","conv-SAE steering (K8)","C4"),("pay","payload-suppress (output)","C2")]:
    for r in rows:
        xs=[a for a,b in r[k]]; ys=[b for a,b in r[k]]; plt.plot(xs,ys,'-o',color=c,alpha=0.4,ms=3)
    plt.plot([],[],'-o',color=c,label=lab)
plt.yscale('symlog',linthresh=0.1); plt.xlabel("backdoor ASR suppression (1 − ASR/base) → stronger"); plt.ylabel("held-out collateral KL (nats) ↓ better")
plt.title("In-context backdoor: FRA-QK vs the SAME strong baselines that beat it on the weight-baked sleeper")
plt.legend(fontsize=7); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(os.path.join(OUT,"ic4_full.png"),dpi=130)
print("\nDONE ic4",flush=True)
