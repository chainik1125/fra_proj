"""G2 — full in-context backdoor removal comparison on Gemma-2-2b + GemmaScope.
FRA-QK (induction edge, all induction heads) vs DoM (mean-diff from poisoned-ON/clean-OFF contrast,
resid L6) vs conv-SAE (act-diff top-12 features, gated removal, resid L6) vs payload-suppress (unembed).
Metric: ASR-suppression vs held-out collateral (KL on text with both trigger & payload). Reports the
cases where FRA reaches >=0.7 removal (Gemma reach is case-dependent).
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
Llast=model.cfg.n_layers-1; W_U=model.W_U
# induction heads
torch.manual_seed(0); N=24
R=(torch.randperm(40000)[:N]+1000).tolist(); tt0=torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0)
edges0=[(1+N+t,t+2) for t in range(N-1)]
_,c0=model.run_with_cache(tt0,names_filter=lambda n:n.endswith("hook_pattern"))
strength={}
for L in range(model.cfg.n_layers):
    p=c0[f"blocks.{L}.attn.hook_pattern"][0]
    for H in range(p.shape[0]): strength[(L,H)]=float(np.mean([p[H,q,k].item() for q,k in edges0]))
IND=[lh for lh,s in sorted(strength.items(),key=lambda x:-x[1]) if s>0.4][:10]
LAYERS=sorted(set(L for L,H in IND)); L0=min(LAYERS); DL=6
print("induction heads:",[(f"L{L}H{H}") for L,H in IND],flush=True)
SAE={}
for L in sorted(set(LAYERS)|{DL}):
    sl=L-1
    try: SAE[L]=GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{sl}/width_16k/canonical",device=dev,normalize_activations=True)
    except Exception: SAE[L]=GemmaScopeSAE("gemma-scope-2b-pt-res",f"layer_{sl}/width_16k/average_l0_68",device=dev,normalize_activations=True)
print("SAEs loaded",flush=True); dsae=SAE[DL]
def encode(L,x):
    f=SAE[L].encode(x.float()).float()
    if SAE[L]._norm_coeff is not None: f=f/SAE[L]._norm_coeff
    return f
def fra_ph(tt):
    _,c=model.run_with_cache(tt,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
    H={}; resid={L:c[f"blocks.{L}.hook_resid_pre"][0] for L in LAYERS}
    for (L,Hh) in IND:
        fe=encode(L,c[f"blocks.{L}.hook_resid_pre"][0]); xh=fe@SAE[L].W_dec.float()+SAE[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,SAE[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
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
                for Hh,sd in td.items(): s[0,Hh,:seq,:seq]=s[0,Hh,:seq,:seq]-sd.to(s.dtype)
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(td)))
    return model.run_with_hooks(tt,fwd_hooks=hooks)[0]
def dom_run(tt,positions,vD,a):
    def hook(act,hook):
        for p in positions:
            if p<tt.shape[1]: act[0,p,:]=act[0,p,:]-(a*vD*act[0,p,:].float().norm()).to(act.dtype)
        return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{DL}.hook_resid_pre",hook)])[0]
def conv_run(tt,fidx,c):
    Wd=dsae.W_dec[fidx].float()
    def hook(act,hook):
        x=act[0].float(); z=encode(DL,x)[:,fidx]; act[0]=(x-(c*z)@Wd).to(act.dtype); return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{DL}.hook_resid_pre",hook)])[0]
def paysupp(tt,Pid,s):
    uP=W_U[:,Pid].float(); uP=uP/uP.norm()
    def hook(act,hook): act[0]=act[0]-(s*(act[0].float()@uP).unsqueeze(-1)*uP).to(act.dtype); return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{Llast}.hook_resid_post",hook)])[0]
def klsum(p,q):
    lp=torch.log_softmax(p.float(),-1); lq=torch.log_softmax(q.float(),-1); return (lp.exp()*(lp-lq)).sum(-1).sum().item()
def mkseq(seed,Tid=None,Pid=None,n=20):
    g=torch.Generator().manual_seed(seed); Rr=(torch.randperm(40000,generator=g)[:n]+1000).tolist()
    if Tid is not None: Rr[10]=Tid; Rr[11]=Pid
    return torch.tensor([tok.bos_token_id]+Rr+Rr,device=dev).unsqueeze(0)
CASES=[(" bank"," river","The river bank was crowded. He left the bank and crossed the river. By dusk the river hid the bank."),
       (" king"," crown","The king lost his crown. A new crown was made. The old king wore the crown while the young king watched."),
       (" doctor"," water","The doctor drank water. More water spilled. The doctor gave the patient water before the next doctor came."),
       (" market"," gold","The market sold gold. Gold prices fell. The market reopened and gold buyers crowded the market again.")]
Ni=20; qpos=1+Ni+10; kpos=12; FC=[1,2,4,8,16,32]; DC=[0.25,0.5,1,2,4,8]; CC=[1,2,4,8,16,32]; PC=[0.5,1,2,4,8]
rows=[]
for T,P,htext in CASES:
    Tid=tok.encode(T,add_special_tokens=False); Pid=tok.encode(P,add_special_tokens=False)
    if len(Tid)!=1 or len(Pid)!=1: continue
    Tid=Tid[0]; Pid=Pid[0]
    on=[]; off=[]; onf=[]; offf=[]
    for s in range(12):
        a=model.run_with_cache(mkseq(1000+s,Tid,Pid),names_filter=[f"blocks.{DL}.hook_resid_pre"])[1][f"blocks.{DL}.hook_resid_pre"][0]
        on.append(a[qpos]); onf.append(encode(DL,a[qpos:qpos+1])[0])
        b=model.run_with_cache(mkseq(5000+s),names_filter=[f"blocks.{DL}.hook_resid_pre"])[1][f"blocks.{DL}.hook_resid_pre"][0]
        off.append(b[qpos]); offf.append(encode(DL,b[qpos:qpos+1])[0])
    vD=(torch.stack(on).mean(0)-torch.stack(off).mean(0)).float(); vD=vD/(vD.norm()+1e-6)
    convK=torch.topk((torch.stack(onf).mean(0)-torch.stack(offf).mean(0)),12).indices.tolist()
    tt=mkseq(0,Tid,Pid); seq=tt.shape[1]
    base=torch.softmax(model(tt)[0][qpos].float(),-1)[Pid].item()
    if base<0.2: print(f"skip {T} ASR {base:.2f}",flush=True); continue
    def asr_s(lg): return 1-torch.softmax(lg[qpos].float(),-1)[Pid].item()/base
    HF,resid=fra_ph(tt); Pp=primer_pairs(HF,(qpos,kpos)); byL=delta_content(HF,Pp,seq)
    hids=[tok.bos_token_id]+tok.encode(htext,add_special_tokens=False); ht=torch.tensor(hids,device=dev).unsqueeze(0); hseq=ht.shape[1]
    hT=[i for i,t in enumerate(hids) if t==Tid]; hclean=model(ht)[0]; HFh,_=fra_ph(ht); byLh=delta_content(HFh,Pp,hseq)
    trig=[1+10,1+Ni+10]
    cur={"fra":[(asr_s(patch_fra(tt,byL,c)), klsum(hclean,patch_fra(ht,byLh,c))) for c in FC],
         "dom":[(asr_s(dom_run(tt,trig,vD,a)), klsum(hclean,dom_run(ht,hT,vD,a))) for a in DC],
         "conv":[(asr_s(conv_run(tt,convK,c)), klsum(hclean,conv_run(ht,convK,c))) for c in CC],
         "pay":[(asr_s(paysupp(tt,Pid,s)), klsum(hclean,paysupp(ht,Pid,s))) for s in PC]}
    rows.append(dict(T=T,P=P,base=base,**cur))
    print(f"{T.strip()}->{P.strip()} ASR {base:.2f}:",flush=True)
    for k in ["fra","dom","conv","pay"]: print(f"  {k:5}: {[(round(a,2),round(b,2)) for a,b in cur[k]]}",flush=True)
def at(curve,t):
    xs=[a for a,b in curve]; ys=[b for a,b in curve]
    if max(xs)<t: return None
    o=np.argsort(xs); return float(np.interp(t,np.array(xs)[o],np.array(ys)[o]))
print(f"\n=== GEMMA-2-2B in-context backdoor, held-out collateral @ 70% ASR-suppression ===",flush=True)
for k,lab in [("fra","FRA-QK (attention edge)"),("dom","DoM (mean-diff, K8 winner)"),("conv","conv-SAE (act-diff gated, K8)"),("pay","payload-suppress (output)")]:
    v=[at(r[k],0.7) for r in rows]; v=[x for x in v if x is not None]
    if v: print(f"  {lab:34}: {np.mean(v):.3f} ± {np.std(v):.3f} (n={len(v)} cases FRA reached 0.7)",flush=True)
json.dump({"rows":rows},open(os.path.join(OUT,"g2.json"),"w"),indent=2,default=float)
plt.figure(figsize=(6.6,4.8))
for k,lab,c in [("fra","FRA-QK (attention edge)","C0"),("dom","DoM / mean-diff (K8 winner)","C3"),("conv","conv-SAE steering (K8)","C4"),("pay","payload-suppress","C2")]:
    for r in rows:
        xs=[a for a,b in r[k]]; ys=[b for a,b in r[k]]; plt.plot(xs,ys,'-o',color=c,alpha=0.4,ms=3)
    plt.plot([],[],'-o',color=c,label=lab)
plt.yscale('symlog',linthresh=0.1); plt.xlabel("backdoor ASR suppression → stronger"); plt.ylabel("held-out collateral KL (nats) ↓ better")
plt.title("Gemma-2-2b in-context backdoor: FRA-QK vs DoM, conv-SAE, payload-suppress")
plt.legend(fontsize=7); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(os.path.join(OUT,"g2_full.png"),dpi=130)
print("\nDONE g2",flush=True)
