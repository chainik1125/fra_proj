"""RETRIEVAL_STEER3 — content-gated steer with NO SAE-encode-in-hook (that was the nan source on gemma).
Use PROJECTION-REMOVAL of the frog-value direction d (= dominant frog-value SAE feature decoder, a fixed
vector) at each retrieval layer: resid -= a*(resid . d_hat) d_hat. This is content-gated (only affects
positions carrying frog-value content), transfer-capable, fp16-safe. Decisive vs FRA on legit-frog KL.
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
rng=np.random.RandomState(0)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
RH=[(15,0),(18,6),(6,3),(21,5),(22,4)]; LAYERS=sorted(set(L for L,H in RH))
SAE={L:GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{L-1}/width_16k/canonical",device=dev,normalize_activations=True) for L in LAYERS}
def enc(L,x):
    f=SAE[L].encode(x.float()).float()
    if SAE[L]._norm_coeff is not None: f=f/SAE[L]._norm_coeff
    return f
facts=[("red","frog"),("blue","lamp"),("green","clock"),("gold","rose"),("black","sword"),("white","candle")]
cid=tok.encode(" frog",add_special_tokens=False)[0]
def build(fs,qk,filler=""): return tok.encode(filler+"".join(f" The {k} box holds a {v}." for k,v in fs)+f" The {qk} box holds a")
ids=build(facts,"red"); tt=torch.tensor(ids,device=dev).unsqueeze(0); seq=tt.shape[1]; cpos=ids.index(cid)
def logits_last(t,hooks=None):
    lg=(model.run_with_hooks(t,fwd_hooks=hooks) if hooks else model(t))[0]; return torch.log_softmax(lg[-1].float(),-1)
def Pf(t,hooks=None): return logits_last(t,hooks).exp()[cid].item()
def KL(t,hooks): p=logits_last(t); q=logits_last(t,hooks); return (p.exp()*(p-q)).sum().item()
def fra_ph(t):
    _,c=model.run_with_cache(t,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS]); H={}
    for (L,Hh) in RH:
        fe=enc(L,c[f"blocks.{L}.hook_resid_pre"][0]); xh=fe@SAE[L].W_dec.float()+SAE[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,SAE[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H
HF=fra_ph(tt)
def P_top(M=20):
    P={}
    for (L,Hh) in RH:
        d=HF[(L,Hh)]; loc=np.where((d["qq"]==seq-1)&(d["kk"]==cpos))[0]; loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
        P[(L,Hh)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in loc)
    return P
def fra_hooks(t,P,c):
    HF2=fra_ph(t); sq=t.shape[1]; byL={}
    for (L,Hh) in RH:
        d=HF2[(L,Hh)]; Ps=P[(L,Hh)]; dd=np.zeros((sq,sq))
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        byL.setdefault(L,{})[Hh]=torch.tensor(dd,device=dev,dtype=torch.float32)*c
    hk=[]
    for L,hd in byL.items():
        def mk(hd):
            def hook(s,hook):
                for Hh,dd in hd.items(): s[0,Hh,:dd.shape[0],:dd.shape[1]]-=dd[:s.shape[2],:s.shape[3]].to(s.dtype)
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores",mk(hd)))
    return hk
# frog-value direction per layer = dominant key-feature decoder (a fixed unit vector; NO encode in hook)
def dvec(L):
    d=HF[(L,[H for LL,H in RH if LL==L][0])]; loc=np.where((d["qq"]==seq-1)&(d["kk"]==cpos))[0]
    kw={}
    for o in loc: kw[int(d["jj"][o])]=kw.get(int(d["jj"][o]),0)+abs(d["vv"][o])
    if not kw: return None
    j=max(kw,key=kw.get); v=SAE[L].W_dec[j].float(); return (v/v.norm()).to(dev)
DV={L:dvec(L) for L in LAYERS}
def steer_hooks(a):
    hk=[]
    for L in LAYERS:
        dh=DV[L]
        if dh is None: continue
        def mk(dh):
            def hook(resid,hook):
                x=resid[0].float(); proj=(x@dh).unsqueeze(-1)*dh.unsqueeze(0)
                resid[0]=(x-a*proj).to(resid.dtype); return resid
            return hook
        hk.append((f"blocks.{L}.hook_resid_pre",mk(dh)))
    return hk
base=Pf(tt); P=P_top()
print(f"base P(frog)={base:.3f}",flush=True)
print("\nPROJECTION-REMOVAL STEER on-target sweep:",flush=True)
for a in [1,2,4,8,16]:
    print(f"  a={a:<3} P(frog)={Pf(tt,steer_hooks(a)):.3f}",flush=True)
fra_on=Pf(tt,fra_hooks(tt,P,4)); best=None
for a in [1,2,4,8,16,32]:
    on=Pf(tt,steer_hooks(a))
    if not np.isnan(on) and (best is None or abs(on-fra_on)<best[1]): best=(a,abs(on-fra_on),on)
legits=["A frog sat quietly on the wet log beside the","The children laughed as the green frog leaped across the","At the zoo her favourite animal was the small frog that"]
if best and best[2]<0.9*base:
    a=best[0]
    fra_kl=np.mean([KL(torch.tensor(tok.encode(s),device=dev).unsqueeze(0),fra_hooks(torch.tensor(tok.encode(s),device=dev).unsqueeze(0),P,4)) for s in legits])
    steer_kl=np.mean([KL(torch.tensor(tok.encode(s),device=dev).unsqueeze(0),steer_hooks(a)) for s in legits])
    print(f"\nDECISIVE (matched: FRA c=4 {fra_on:.3f} vs STEER a={a} {best[2]:.3f}):",flush=True)
    print(f"  legit-frog KL: FRA {fra_kl:.4f} | content-gated STEER {steer_kl:.4f}  => FRA preserves legit frog {steer_kl/max(fra_kl,1e-4):.1f}x better",flush=True)
    json.dump({"base":base,"steer_a":a,"steer_on":best[2],"fra_on":fra_on,"fra_kl":float(fra_kl),"steer_kl":float(steer_kl),
               "verdict":"FRA more separable" if steer_kl>1.5*fra_kl else "tie/steer-comparable"},
              open(os.path.join(OUT,"retrieval_steer3.json"),"w"),indent=2,default=float)
else: print(f"\nSTEER still failed to suppress on-target (best={best})",flush=True)
print("\nDONE retrieval_steer3",flush=True)
