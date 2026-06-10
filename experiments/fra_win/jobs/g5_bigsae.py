"""G5 — does an even bigger/finer GemmaScope push FRA reach past 65k (0.52)? Probe 262k & 1M
availability per induction layer (public release may not have them everywhere), use the biggest
available width per layer, and measure norm-recovery / edge-corr / FRA max ASR-reach on the cappers.
"""
import os, sys, json, gc
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
torch.manual_seed(0); N=24
R=(torch.randperm(40000)[:N]+1000).tolist(); tt0=torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0)
edges0=[(1+N+t,t+2) for t in range(N-1)]
_,c0=model.run_with_cache(tt0,names_filter=lambda n:n.endswith("hook_pattern"))
st={}
for L in range(model.cfg.n_layers):
    p=c0[f"blocks.{L}.attn.hook_pattern"][0]
    for H in range(p.shape[0]): st[(L,H)]=float(np.mean([p[H,q,k].item() for q,k in edges0]))
IND=[lh for lh,s in sorted(st.items(),key=lambda x:-x[1]) if s>0.4][:10]
LAYERS=sorted(set(L for L,H in IND)); print("heads:",[f"L{L}H{H}" for L,H in IND],flush=True)
# probe availability per layer for each width
def load_one(sl,width):
    for rel,sid in [("gemma-scope-2b-pt-res-canonical",f"layer_{sl}/width_{width}/canonical")]:
        try: return GemmaScopeSAE(rel,sid,device=dev,normalize_activations=True)
        except Exception: pass
    for l0 in [34,50,68,72,100,137,200,297,408]:
        try: return GemmaScopeSAE("gemma-scope-2b-pt-res",f"layer_{sl}/width_{width}/average_l0_{l0}",device=dev,normalize_activations=True)
        except Exception: pass
    return None
print("\n== probing availability per induction-input layer ==",flush=True)
avail={}
for L in LAYERS:
    sl=L-1; avail[L]={}
    for width in ["65k","262k","1m"]:
        s=load_one(sl,width); avail[L][width]=(s is not None)
        if s is not None: del s; gc.collect(); torch.cuda.empty_cache()
    print(f"  layer {L-1}: " + ", ".join(f"{w}={'Y' if avail[L][w] else 'N'}" for w in ['65k','262k','1m']),flush=True)
# build best-available-per-layer SAE set, biggest first
def best_set(prefer):
    SAE={}
    for L in LAYERS:
        sl=L-1
        for width in prefer:
            if avail[L].get(width):
                SAE[L]=load_one(sl,width); SAE[L]._width=width; break
        if L not in SAE: return None
    return SAE
def enc(SAE,L,x):
    f=SAE[L].encode(x.float()).float()
    if SAE[L]._norm_coeff is not None: f=f/SAE[L]._norm_coeff
    return f
Ni=20; qpos=1+Ni+10; kpos=12
CASES=[(" bank"," river"),(" market"," gold"),(" king"," crown")]
def mkseq(seed,Tid,Pid):
    g=torch.Generator().manual_seed(seed); Rr=(torch.randperm(40000,generator=g)[:Ni]+1000).tolist(); Rr[10]=Tid; Rr[11]=Pid
    return torch.tensor([tok.bos_token_id]+Rr+Rr,device=dev).unsqueeze(0)
def run(SAE,label):
    widths=[SAE[L]._width for L in LAYERS]; print(f"\n== {label}: per-layer widths {dict(zip([L-1 for L in LAYERS],widths))} ==",flush=True)
    res=[]
    for T,P in CASES:
        Tid=tok.encode(T,add_special_tokens=False)[0]; Pid=tok.encode(P,add_special_tokens=False)[0]
        tt=mkseq(0,Tid,Pid); seq=tt.shape[1]; base=torch.softmax(model(tt)[0][qpos].float(),-1)[Pid].item()
        names=[f"blocks.{L}.hook_resid_pre" for L in LAYERS]+["blocks.6.attn.hook_attn_scores"]
        _,c=model.run_with_cache(tt,names_filter=lambda n:n in names)
        a6=c["blocks.6.hook_resid_pre"][0]; f6=enc(SAE,6,a6); xh6=f6@SAE[6].W_dec.float()+SAE[6].b_dec.float()
        nr=float(((xh6.pow(2).mean(-1)+1e-6).sqrt()/(a6.float().pow(2).mean(-1)+1e-6).sqrt()).mean())
        HF={}
        for (L,Hh) in IND:
            fe=enc(SAE,L,c[f"blocks.{L}.hook_resid_pre"][0]); xh=fe@SAE[L].W_dec.float()+SAE[L].b_dec.float()
            r=_build_fra_result(model,L,Hh,fe,SAE[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
            f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); HF[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
        d=HF[(6,2)]; recon=np.zeros((seq,seq))
        for n in range(len(d["vv"])): recon[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        sc=c["blocks.6.attn.hook_attn_scores"][0,2].float().cpu().numpy()
        ie=[(1+Ni+t,t+2) for t in range(Ni-1)]; ea=np.array([sc[q,k] for q,k in ie]); er=np.array([recon[q,k] for q,k in ie])
        corr=float(np.corrcoef(ea,er)[0,1])
        byL={}
        for (L,Hh) in IND:
            dd=HF[(L,Hh)]; loc=np.where((dd["qq"]==qpos)&(dd["kk"]==kpos))[0]; loc=loc[np.argsort(-np.abs(dd["vv"][loc]))[:12]]
            Ps=set((int(dd["ii"][o]),int(dd["jj"][o])) for o in loc); arr=np.zeros((seq,seq))
            for n in range(len(dd["vv"])):
                if (int(dd["ii"][n]),int(dd["jj"][n])) in Ps: arr[dd["qq"][n],dd["kk"][n]]+=dd["vv"][n]
            byL.setdefault(L,{})[Hh]=arr
        def supp(cc):
            hooks=[]
            for L,hd in byL.items():
                td={Hh:torch.tensor(arr,device=dev,dtype=torch.float32)*cc for Hh,arr in hd.items()}
                def mk(td):
                    def hook(s,hook):
                        for Hh,sd in td.items(): s[0,Hh,:seq,:seq]=s[0,Hh,:seq,:seq]-sd.to(s.dtype)
                        return s
                    return hook
                hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(td)))
            return 1-torch.softmax(model.run_with_hooks(tt,fwd_hooks=hooks)[0][qpos].float(),-1)[Pid].item()/base
        reach=max(supp(cc) for cc in [2,4,8,16,32,64])
        res.append((T.strip(),nr,corr,reach)); print(f"  {T.strip()}: norm-rec {nr:.2f} edge-corr {corr:.2f} reach {reach:.2f}",flush=True)
    return res
out={"avail":{str(L-1):avail[L] for L in LAYERS}}
for prefer,label in [(["262k","65k"],"best<=262k"),(["1m","262k","65k"],"best<=1M")]:
    SAE=best_set(prefer)
    if SAE is None: print(f"\n{label}: not loadable",flush=True); continue
    res=run(SAE,label); out[label]={"results":res,"mean_reach":float(np.mean([x[3] for x in res]))}
    print(f"  => {label} mean reach {np.mean([x[3] for x in res]):.2f}",flush=True)
    del SAE; gc.collect(); torch.cuda.empty_cache()
json.dump(out,open(os.path.join(OUT,"g5.json"),"w"),indent=2,default=float)
print("\n(16k mean reach 0.34, 65k 0.52 for reference)",flush=True); print("DONE g5",flush=True)
