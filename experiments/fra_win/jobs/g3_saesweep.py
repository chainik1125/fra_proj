"""G3 — sweep GemmaScope variants to find the one that best captures the attention edge -> best FRA reach.
Hypothesis (from grms: 16k recovers only ~56% of the residual norm): wider / denser SAEs reconstruct
more -> the FRA edit captures more of the edge -> higher ASR-removal ceiling. For each variant, measure
norm-recovery (rms_xhat/rms_true), FRA edge correlation, and FRA max ASR-suppression on backdoors that
currently cap. Sequential load/free per variant to fit memory.
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
CASES=[(" bank"," river"),(" market"," gold"),(" king"," crown")]  # capped at 0.25/0.24/0.49 with 16k
Ni=20; qpos=1+Ni+10; kpos=12
def mkseq(seed,Tid,Pid):
    g=torch.Generator().manual_seed(seed); Rr=(torch.randperm(40000,generator=g)[:Ni]+1000).tolist(); Rr[10]=Tid; Rr[11]=Pid
    return torch.tensor([tok.bos_token_id]+Rr+Rr,device=dev).unsqueeze(0)
def try_load(width):
    SAE={}
    for L in LAYERS:
        sl=L-1
        ok=False
        for rel,sid in [("gemma-scope-2b-pt-res-canonical",f"layer_{sl}/width_{width}/canonical")]:
            try: SAE[L]=GemmaScopeSAE(rel,sid,device=dev,normalize_activations=True); ok=True; break
            except Exception as e: pass
        if not ok: return None
    return SAE
def enc(SAE,L,x):
    f=SAE[L].encode(x.float()).float()
    if SAE[L]._norm_coeff is not None: f=f/SAE[L]._norm_coeff
    return f
def run_variant(width):
    SAE=try_load(width)
    if SAE is None: print(f"width {width}: UNAVAILABLE/failed to load",flush=True); return None
    d_sae=SAE[LAYERS[0]].d_sae
    # norm recovery + edge corr at L6 head 2 on a backdoor seq
    res=[]
    for T,P in CASES:
        Tid=tok.encode(T,add_special_tokens=False); Pid=tok.encode(P,add_special_tokens=False)
        if len(Tid)!=1 or len(Pid)!=1: continue
        Tid=Tid[0]; Pid=Pid[0]; tt=mkseq(0,Tid,Pid); seq=tt.shape[1]
        base=torch.softmax(model(tt)[0][qpos].float(),-1)[Pid].item()
        names=[f"blocks.{L}.hook_resid_pre" for L in LAYERS]+[f"blocks.6.attn.hook_attn_scores"]
        _,c=model.run_with_cache(tt,names_filter=lambda n:n in names)
        # norm recovery at L6
        a6=c["blocks.6.hook_resid_pre"][0]; f6=enc(SAE,6,a6); xh6=f6@SAE[6].W_dec.float()+SAE[6].b_dec.float()
        nr=float(((xh6.pow(2).mean(-1)+1e-6).sqrt()/(a6.float().pow(2).mean(-1)+1e-6).sqrt()).mean())
        # FRA per head + max reach
        HF={}
        for (L,Hh) in IND:
            fe=enc(SAE,L,c[f"blocks.{L}.hook_resid_pre"][0]); xh=fe@SAE[L].W_dec.float()+SAE[L].b_dec.float()
            r=_build_fra_result(model,L,Hh,fe,SAE[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
            f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); HF[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
        # edge corr at L6H2
        d=HF[(6,2)]; recon=np.zeros((seq,seq))
        for n in range(len(d["vv"])): recon[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        sc=c["blocks.6.attn.hook_attn_scores"][0,2].float().cpu().numpy()
        ie=[(1+Ni+t,t+2) for t in range(Ni-1)]; ea=np.array([sc[q,k] for q,k in ie]); er=np.array([recon[q,k] for q,k in ie])
        corr=float(np.corrcoef(ea,er)[0,1])
        # FRA reach
        P_={}
        for (L,Hh) in IND:
            dd=HF[(L,Hh)]; loc=np.where((dd["qq"]==qpos)&(dd["kk"]==kpos))[0]; loc=loc[np.argsort(-np.abs(dd["vv"][loc]))[:12]]
            P_[(L,Hh)]=set((int(dd["ii"][o]),int(dd["jj"][o])) for o in loc)
        byL={}
        for (L,Hh) in IND:
            dd=HF[(L,Hh)]; arr=np.zeros((seq,seq)); Ps=P_[(L,Hh)]
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
            lg=model.run_with_hooks(tt,fwd_hooks=hooks)[0]
            return 1-torch.softmax(lg[qpos].float(),-1)[Pid].item()/base
        reach=max(supp(cc) for cc in [2,4,8,16,32,64])
        res.append((T.strip(),nr,corr,reach))
        print(f"  width {width} {T.strip()}->{P.strip()}: norm-rec {nr:.2f} edge-corr {corr:.2f} FRA max-reach {reach:.2f}",flush=True)
    del SAE; gc.collect(); torch.cuda.empty_cache()
    return dict(width=width,d_sae=d_sae,results=res)
out=[]
for width in ["16k","65k","262k"]:
    r=run_variant(width)
    if r: out.append(r)
print("\n=== SUMMARY: GemmaScope variant vs FRA reach ===",flush=True)
for r in out:
    nrs=[x[1] for x in r["results"]]; cor=[x[2] for x in r["results"]]; rch=[x[3] for x in r["results"]]
    print(f"  {r['width']:5} (d_sae {r['d_sae']}): mean norm-rec {np.mean(nrs):.2f} | edge-corr {np.mean(cor):.2f} | FRA max-reach {np.mean(rch):.2f}",flush=True)
json.dump(out,open(os.path.join(OUT,"g3.json"),"w"),indent=2,default=float)
print("DONE g3",flush=True)
