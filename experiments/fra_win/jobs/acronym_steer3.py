"""ACRONYM_STEER3 — settle the acronym decisive test with the WORKING projection-removal steer
(the retrieval version fired cleanly). Content-gated steer = projection-removal of the dominant
Officer-KEY feature direction at each letter-mover layer (fp32-safe, no encode-in-hook). Decisive:
does FRA-QK preserve legit 'Officer' contexts where the content-gated steer corrupts them?
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gpt2",device=dev); model.eval(); tok=model.tokenizer
HEADS=[(8,11),(9,9),(10,10),(11,4)]; LY=sorted(set(L for L,H in HEADS))
sae={L:(lambda s:(s[0] if isinstance(s,tuple) else s))(SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev)) for L in LY}
def enc(s): return torch.tensor([tok.bos_token_id]+tok.encode(s),device=dev).unsqueeze(0)
def kpos(ids,sub,before):
    w=tok.encode(sub)[0]; c=[i for i,x in enumerate(ids) if x==w and i<before]; return c[-1] if c else None
def llast(t,hooks=None):
    lg=(model.run_with_hooks(t,fwd_hooks=hooks) if hooks else model(t))[0]; return torch.log_softmax(lg[-1].float(),-1)
def Pof(t,letter,hooks=None): return llast(t,hooks).exp()[tok.encode(letter)[0]].item()
def KL(t,hooks): p=llast(t); q=llast(t,hooks); return (p.exp()*(p-q)).sum().item()
def fra_edge(tt,L,H):
    HK=f"blocks.{L}.hook_resid_pre"; fe=sae[L].encode(model.run_with_cache(tt,names_filter=lambda n:n==HK)[1][HK][0]).float()
    xh=fe@sae[L].W_dec.float()+sae[L].b_dec.float()
    r=_build_fra_result(model,L,H,fe,sae[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
p1="The Chief Executive Officer (CE"; t1=enc(p1); ids1=[tok.bos_token_id]+tok.encode(p1); Q=t1.shape[1]-1; K=kpos(ids1," Officer",Q)
HF={(L,H):fra_edge(t1,L,H) for L,H in HEADS}
P={};
for (L,H) in HEADS:
    d=HF[(L,H)]; on=(d["qq"]==Q)&(d["kk"]==K); oi=np.where(on)[0][np.argsort(-np.abs(d["vv"][on]))[:12]]
    P[(L,H)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in oi)
def fra_hooks(tt,c=4):
    sq=tt.shape[1]; byL={}
    for (L,H) in HEADS:
        d=fra_edge(tt,L,H); Ps=P[(L,H)]; dd=np.zeros((sq,sq))
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        byL.setdefault(L,{})[H]=torch.tensor(dd,device=dev,dtype=torch.float32)*c
    hk=[]
    for L,hd in byL.items():
        def mk(hd):
            def hook(s,hook):
                for H,dd in hd.items(): s[0,H,:dd.shape[0],:dd.shape[1]]-=dd.to(s.dtype)
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores",mk(hd)))
    return hk
def dvec(L):
    d=HF[(L,[H for LL,H in HEADS if LL==L][0])]; on=np.where((d["qq"]==Q)&(d["kk"]==K))[0]; kw={}
    for o in on: kw[int(d["jj"][o])]=kw.get(int(d["jj"][o]),0)+abs(d["vv"][o])
    if not kw: return None
    j=max(kw,key=kw.get); v=sae[L].W_dec[j].float(); return v/v.norm()
DV={L:dvec(L) for L in LY}
def steer_hooks(a):
    hk=[]
    for L in LY:
        dh=DV[L]
        if dh is None: continue
        def mk(dh):
            def hook(resid,hook):
                x=resid[0].float(); proj=(x@dh).unsqueeze(-1)*dh.unsqueeze(0); resid[0]=(x-a*proj).to(resid.dtype); return resid
            return hook
        hk.append((f"blocks.{L}.hook_resid_pre",mk(dh)))
    return hk
base=Pof(t1,"O"); fra_on=Pof(t1,"O",fra_hooks(t1,4))
print(f"base P(O)={base:.3f}; FRA(c=4) on-target {fra_on:.3f}",flush=True)
print("STEER sweep:",{a:round(Pof(t1,'O',steer_hooks(a)),3) for a in [1,2,4,8]},flush=True)
best=None
for a in [1,2,4,8,16]:
    on=Pof(t1,'O',steer_hooks(a))
    if not np.isnan(on) and (best is None or abs(on-fra_on)<best[1]): best=(a,abs(on-fra_on),on)
legits=["The Officer in charge signed the report and then","A police Officer stopped the car on the highway near","The senior Officer reviewed the documents carefully before"]
if best and best[2]<0.9*base:
    a=best[0]
    fra_kl=np.mean([KL(enc(s),fra_hooks(enc(s),4)) for s in legits]); steer_kl=np.mean([KL(enc(s),steer_hooks(a)) for s in legits])
    print(f"DECISIVE (matched: FRA {fra_on:.3f} vs STEER a={a} {best[2]:.3f}): legit-Officer KL FRA {fra_kl:.4f} | STEER {steer_kl:.4f} => FRA {steer_kl/max(fra_kl,1e-4):.1f}x more separable",flush=True)
    json.dump({"base":base,"fra_on":fra_on,"steer_a":a,"steer_on":best[2],"fra_kl":float(fra_kl),"steer_kl":float(steer_kl)},open(os.path.join(OUT,"acronym_steer3.json"),"w"),indent=2,default=float)
else: print(f"STEER failed to suppress (best={best})",flush=True)
print("\nDONE acronym_steer3",flush=True)
