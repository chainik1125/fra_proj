"""G1 — can FRA remove an in-context backdoor on Gemma-2-2b (the full induction-head set)?
Feasibility gate before the full DoM/conv-SAE comparison. RMSNorm => GemmaScope resid + RMS correction
is exact-magnitude FRA. Find ALL induction heads, load GemmaScope SAEs at their input layers, build a
trigger->payload in-context backdoor, ablate the backdoor edge across all heads, check ASR removal +
a quick payload-suppress collateral comparison.
"""
import os, sys, json, traceback
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
Llast=model.cfg.n_layers-1; W_U=model.W_U
# ---- find induction heads on a random repeated sequence ----
torch.manual_seed(0); N=24
R=(torch.randperm(40000)[:N]+1000).tolist(); tt0=torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0); seq0=tt0.shape[1]
edges0=[(1+N+t,t+2) for t in range(N-1)]
_,c0=model.run_with_cache(tt0,names_filter=lambda n:n.endswith("hook_pattern"))
strength={}
for L in range(model.cfg.n_layers):
    p=c0[f"blocks.{L}.attn.hook_pattern"][0]
    for H in range(p.shape[0]): strength[(L,H)]=float(np.mean([p[H,q,k].item() for q,k in edges0]))
IND=[lh for lh,s in sorted(strength.items(),key=lambda x:-x[1]) if s>0.4][:10]
print("induction heads (>0.4):",[(f"L{L}H{H}",round(strength[(L,H)],2)) for L,H in IND],flush=True)
LAYERS=sorted(set(L for L,H in IND)); L0=min(LAYERS)
# ---- load GemmaScope SAEs at each head's input layer (resid_post[L-1]) ----
SAE={}
for L in LAYERS:
    sl=L-1
    try: SAE[L]=GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{sl}/width_16k/canonical",device=dev,normalize_activations=True)
    except Exception:
        SAE[L]=GemmaScopeSAE("gemma-scope-2b-pt-res",f"layer_{sl}/width_16k/average_l0_68",device=dev,normalize_activations=True)
    print(f"SAE for L{L} (resid {sl}) loaded",flush=True)
def fra_ph(tt):
    _,c=model.run_with_cache(tt,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
    H={}; resid={L:c[f"blocks.{L}.hook_resid_pre"][0] for L in LAYERS}
    for (L,Hh) in IND:
        fe=SAE[L].encode(c[f"blocks.{L}.hook_resid_pre"][0]).float()
        if SAE[L]._norm_coeff is not None: fe=fe/SAE[L]._norm_coeff
        xh=fe@SAE[L].W_dec.float()+SAE[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,SAE[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
        H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H,resid
def fra_delta_content(HF,P,seq):
    byL={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; dd=np.zeros((seq,seq)); Ps=P[(L,Hh)]
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        byL.setdefault(L,{})[Hh]=dd
    return byL
def primer_pairs(HF,edge,M=12):
    P={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; loc=np.where((d["qq"]==edge[0])&(d["kk"]==edge[1]))[0]; loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
        P[(L,Hh)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in loc)
    return P
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
def paysupp(tt,Pid,s):
    uP=W_U[:,Pid].float(); uP=uP/uP.norm()
    def hook(act,hook): act[0]=act[0]-(s*(act[0].float()@uP).unsqueeze(-1)*uP).to(act.dtype); return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{Llast}.hook_resid_post",hook)])[0]
def klsum(p,q):
    lp=torch.log_softmax(p.float(),-1); lq=torch.log_softmax(q.float(),-1); return (lp.exp()*(lp-lq)).sum(-1).sum().item()
# ---- in-context backdoor ----
CASES=[(" bank"," river","The river bank was crowded. He left the bank and crossed the river. By dusk the river hid the bank."),
       (" fire"," water","The fire needed water fast. More water hit the fire. Without water the fire would have razed the street.")]
for T,P,htext in CASES:
    Tid=tok.encode(T,add_special_tokens=False); Pid=tok.encode(P,add_special_tokens=False)
    if len(Tid)!=1 or len(Pid)!=1: print(f"skip {T}->{P}",flush=True); continue
    Tid=Tid[0]; Pid=Pid[0]
    Ni=20; R=(torch.randperm(40000)[:Ni]+1000).tolist(); R[10]=Tid; R[11]=Pid
    tt=torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0); seq=tt.shape[1]; qpos=1+Ni+10; kpos=12
    base=torch.softmax(model(tt)[0][qpos].float(),-1)[Pid].item()
    print(f"\n{T.strip()}->{P.strip()} ASR {base:.2f}",flush=True)
    if base<0.2: continue
    HF,resid=fra_ph(tt); Pp=primer_pairs(HF,(qpos,kpos)); byL=fra_delta_content(HF,Pp,seq)
    for c in [1,2,4,8,16]:
        a=1-torch.softmax(patch_fra(tt,byL,c)[qpos].float(),-1)[Pid].item()/base
        print(f"  FRA c={c}: ASR-suppression {a:.2f}",flush=True)
    hids=[tok.bos_token_id]+tok.encode(htext,add_special_tokens=False); ht=torch.tensor(hids,device=dev).unsqueeze(0); hseq=ht.shape[1]
    hclean=model(ht)[0]; HFh,_=fra_ph(ht); byLh=fra_delta_content(HFh,Pp,hseq)
    # collateral @ a working c (pick c giving >0.7 supp if any)
    cgood=None
    for c in [2,4,8,16]:
        if 1-torch.softmax(patch_fra(tt,byL,c)[qpos].float(),-1)[Pid].item()/base>0.7: cgood=c; break
    if cgood:
        kf=klsum(hclean,patch_fra(ht,byLh,cgood))
        # payload-suppress at matched ~0.8 supp
        sp=None
        for s in [0.5,1,2,4]:
            if 1-torch.softmax(paysupp(tt,Pid,s)[qpos].float(),-1)[Pid].item()/base>0.7: sp=s; break
        kp=klsum(hclean,paysupp(ht,Pid,sp)) if sp else float('nan')
        print(f"  => FRA removes at c={cgood}, held-out KL {kf:.3f} | payload-suppress KL {kp:.3f}",flush=True)
    else:
        print(f"  => FRA could NOT reach 0.7 suppression (reach problem persists)",flush=True)
print("\nDONE g1",flush=True)
