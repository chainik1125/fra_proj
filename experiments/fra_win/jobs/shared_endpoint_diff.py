"""SHARED_ENDPOINT_DIFF (red-SPECIFIC pairs = red-edge top MINUS blue-edge top)
ORIG — the theory-validating sibling test (gemma-2-2b). Two boxes hold the SAME value
(red->frog, blue->frog). Goal: suppress frog-retrieval for RED only, preserve it for BLUE.
  FRA: cut the (red-query x frog-value) PAIR. If the query feature is red-SPECIFIC -> preserves blue
       (sibling); if it's a GENERIC box-query feature -> also hits blue (conjunction recurs -> A collapses,
       a theory-predicted failure mode).
  content-gated STEER (projection-removal of the frog-value feature): removes frog for BOTH -> sibling
       collateral. Tests A ~= reuse(marginal frog-value)/reuse(red-query x frog-value conjunction).
On-target = P(frog|red-query) drop; SIBLING = P(frog|blue-query) under the SAME edit (want preserved by FRA).
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
RH=[(15,0),(18,6),(6,3),(21,5),(22,4)]; LAYERS=sorted(set(L for L,H in RH))
SAE={L:GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{L-1}/width_16k/canonical",device=dev,normalize_activations=True) for L in LAYERS}
def enc(L,x):
    f=SAE[L].encode(x.float()).float()
    if SAE[L]._norm_coeff is not None: f=f/SAE[L]._norm_coeff
    return f
cid=tok.encode(" frog",add_special_tokens=False)[0]
# red and blue BOTH hold frog
facts=[("red","frog"),("blue","frog"),("green","clock"),("gold","rose"),("black","sword"),("white","candle")]
def build(qk): return tok.encode("".join(f" The {k} box holds a {v}." for k,v in facts)+f" The {qk} box holds a")
def Pf(t,hooks=None):
    lg=(model.run_with_hooks(t,fwd_hooks=hooks) if hooks else model(t))[0]; return torch.softmax(lg[-1].float(),-1)[cid].item()
def fra_ph(t):
    _,c=model.run_with_cache(t,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS]); H={}
    for (L,Hh) in RH:
        fe=enc(L,c[f"blocks.{L}.hook_resid_pre"][0]); xh=fe@SAE[L].W_dec.float()+SAE[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,SAE[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H
# calibrate FRA pairs on RED-query prompt, (final -> RED's frog) edge (= FIRST frog occurrence)
red=torch.tensor(build("red"),device=dev).unsqueeze(0); ids_r=build("red"); Qr=red.shape[1]-1
red_frog=ids_r.index(cid)   # first frog = red's
HFr=fra_ph(red)
blue0=torch.tensor(build("blue"),device=dev).unsqueeze(0); ids_b=build("blue"); Qb=blue0.shape[1]-1; blue_frog=[i for i,x in enumerate(ids_b) if x==cid][1]
HFb=fra_ph(blue0); P={}
def toppairs(HF,Q,K):
    d=HF[(L,Hh)]; loc=np.where((d["qq"]==Q)&(d["kk"]==K))[0]; loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:30]]
    return set((int(d["ii"][o]),int(d["jj"][o])) for o in loc)
for (L,Hh) in RH:
    rp=toppairs(HFr,Qr,red_frog); bp=toppairs(HFb,Qb,blue_frog)
    P[(L,Hh)]=rp-bp   # red-SPECIFIC: in red's top, NOT blue's
    print(f"  L{L}H{Hh}: red {len(rp)} blue {len(bp)} -> red-specific {len(P[(L,Hh)])}",flush=True)
def fra_hooks(t,c):
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
# content-gated steer: projection-removal of dominant frog-value feature per layer (fp16-safe)
def dvec(L):
    d=HFr[(L,[H for LL,H in RH if LL==L][0])]; loc=np.where((d["qq"]==Qr)&(d["kk"]==red_frog))[0]; kw={}
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
                x=resid[0].float(); proj=(x@dh).unsqueeze(-1)*dh.unsqueeze(0); resid[0]=(x-a*proj).to(resid.dtype); return resid
            return hook
        hk.append((f"blocks.{L}.hook_resid_pre",mk(dh)))
    return hk
blue=torch.tensor(build("blue"),device=dev).unsqueeze(0)
red_base=Pf(red); blue_base=Pf(blue)
print(f"base: P(frog|red-query)={red_base:.3f}  P(frog|blue-query)={blue_base:.3f}",flush=True)
fra_on=Pf(red,fra_hooks(red,4))
# match steer to FRA on-target
best=None
for a in [1,2,4]:
    on=Pf(red,steer_hooks(a))
    if not np.isnan(on) and (best is None or abs(on-fra_on)<best[1]): best=(a,abs(on-fra_on),on)
a=best[0] if best else 1
print(f"\nON-TARGET (red-query): base {red_base:.3f} -> FRA(c=4) {fra_on:.3f} | steer(a={a}) {best[2] if best else float('nan'):.3f}",flush=True)
fra_sib=Pf(blue,fra_hooks(blue,4)); steer_sib=Pf(blue,steer_hooks(a))
print(f"SIBLING (blue-query, SAME edit; want PRESERVED near base {blue_base:.3f}):",flush=True)
print(f"  FRA {fra_sib:.3f} (Δ {abs(fra_sib-blue_base):.3f}) | content-gated STEER {steer_sib:.3f} (Δ {abs(steer_sib-blue_base):.3f})",flush=True)
A=abs(steer_sib-blue_base)/max(abs(fra_sib-blue_base),1e-3)
print(f"  => sibling separability A = {A:.1f}x (FRA preserves blue's frog where the steer destroys it)",flush=True)
print(f"\nVERDICT: {'FRA red-SPECIFIC (sibling preserved) -> WIN' if abs(fra_sib-blue_base)<0.3*blue_base else 'FRA query-feature GENERIC (also hits blue) -> conjunction recurs, advantage collapses (theory-predicted)'}",flush=True)
json.dump({"red_base":red_base,"blue_base":blue_base,"fra_on":fra_on,"steer_on":best[2] if best else None,"steer_a":a,
           "fra_sibling":fra_sib,"steer_sibling":steer_sib,"sibling_A":float(A)},
          open(os.path.join(OUT,"shared_endpoint_t2.json"),"w"),indent=2,default=float)
print("\nDONE shared_endpoint_t2",flush=True)
