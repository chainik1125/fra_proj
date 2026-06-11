"""R2 — the FRA-unique demonstration on retrieval (gemma-2-2b base).
Goal: suppress the model retrieving 'frog' for the 'red' query. Three methods, then the 2x2 that only
FRA fills: SEPARABLE (don't kill 'frog' in legit contexts) x CONTENT-ADDRESSED TRANSFER (works on a new
context where 'frog' is at a different position).
  FRA            : ablate the (red-query x frog-value) feature-pairs on the retrieval heads' edge
  attention-patch: zero the (final -> frog-position) edge          [separable but POSITION-tied]
  content-suppress: subtract frog's unembedding direction          [transferable but NOT separable]
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
Llast=model.cfg.n_layers-1; W_U=model.W_U
RH=[(15,0),(18,6),(6,3),(21,5),(22,4)]; LAYERS=sorted(set(L for L,H in RH))
SAE={L:GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{L-1}/width_16k/canonical",device=dev,normalize_activations=True) for L in LAYERS}
def enc(L,x):
    f=SAE[L].encode(x.float()).float()
    if SAE[L]._norm_coeff is not None: f=f/SAE[L]._norm_coeff
    return f
def build(facts, qk, filler=""):
    ctx=filler+"".join(f" The {k} box holds a {v}." for k,v in facts)+f" The {qk} box holds a"
    return tok.encode(ctx)
facts=[("red","frog"),("blue","lamp"),("green","clock"),("gold","rose"),("black","sword"),("white","candle")]
QK="red"; CV="frog"; cid=tok.encode(" "+CV,add_special_tokens=False)[0]
ids=build(facts,QK); tt=torch.tensor(ids,device=dev).unsqueeze(0); seq=tt.shape[1]
cpos=ids.index(cid); base=torch.softmax(model(tt)[0][-1].float(),-1)[cid].item()
print(f"context-1: P('{CV}'|'{QK}' query) baseline = {base:.3f}  (frog pos {cpos})",flush=True)
def fra_ph(t):
    _,c=model.run_with_cache(t,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS]); H={}
    for (L,Hh) in RH:
        fe=enc(L,c[f"blocks.{L}.hook_resid_pre"][0]); xh=fe@SAE[L].W_dec.float()+SAE[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,SAE[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H
def fra_pairs(HF,q,k,M=20):
    P={}
    for (L,Hh) in RH:
        d=HF[(L,Hh)]; loc=np.where((d["qq"]==q)&(d["kk"]==k))[0]; loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
        P[(L,Hh)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in loc)
    return P
def fra_delta(HF,P,sq):
    byL={}
    for (L,Hh) in RH:
        d=HF[(L,Hh)]; dd=np.zeros((sq,sq)); Ps=P[(L,Hh)]
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        byL.setdefault(L,{})[Hh]=dd
    return byL
def run_fra(t,byL,c,target):
    sq=t.shape[1]; hooks=[]
    for L,hd in byL.items():
        td={Hh:torch.tensor(dd,device=dev,dtype=torch.float32)*c for Hh,dd in hd.items()}
        def mk(td):
            def hook(s,hook):
                for Hh,sd in td.items(): s[0,Hh,:sd.shape[0],:sd.shape[1]]-=sd[:sq,:sq].to(s.dtype)
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(td)))
    return torch.softmax(model.run_with_hooks(t,fwd_hooks=hooks)[0][-1].float(),-1)[target].item()
def run_patch(t,keypos,target):
    sq=t.shape[1]; byL={}
    for L,H in RH: byL.setdefault(L,[]).append(H)
    hooks=[]
    for L,Hs in byL.items():
        def mk(Hs):
            def hook(s,hook):
                for H in Hs:
                    if keypos<sq: s[0,H,sq-1,keypos]=-1e4
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(Hs)))
    return torch.softmax(model.run_with_hooks(t,fwd_hooks=hooks)[0][-1].float(),-1)[target].item()
def run_supp(t,target,s=4):
    uP=W_U[:,cid].float(); uP=uP/uP.norm()
    def hook(act,hook): act[0]=act[0]-(s*(act[0].float()@uP).unsqueeze(-1)*uP).to(act.dtype); return act
    return torch.softmax(model.run_with_hooks(t,fwd_hooks=[(f"blocks.{Llast}.hook_resid_post",hook)])[0][-1].float(),-1)[target].item()
HF=fra_ph(tt); P=fra_pairs(HF,seq-1,cpos); byL=fra_delta(HF,P,seq)
print("\n--- (1) SUPPRESS red->frog retrieval on context-1 ---",flush=True)
print(f"  FRA:            {[round(run_fra(tt,byL,c,cid),3) for c in [2,4,8,16]]}  (sweep c)",flush=True)
print(f"  attention-patch: {run_patch(tt,cpos,cid):.3f}",flush=True)
print(f"  content-suppress:{run_supp(tt,cid):.3f}",flush=True)
# (2) SEPARABILITY: a legit context where 'frog' is the natural answer (no red-box retrieval)
leg=tok.encode("In the pond, a small green creature was sitting on a lily pad. The creature was a")
lt=torch.tensor(leg,device=dev).unsqueeze(0); lbase=torch.softmax(model(lt)[0][-1].float(),-1)[cid].item()
HFl=fra_ph(lt); byLl=fra_delta(HFl,P,lt.shape[1])
print(f"\n--- (2) SEPARABILITY: P('{CV}') in a LEGIT context (baseline {lbase:.3f}) ---",flush=True)
print(f"  FRA(c=8): {run_fra(lt,byLl,8,cid):.3f}   attention-patch: {run_patch(lt,leg.index(cid) if cid in leg else lt.shape[1]-1,cid):.3f}   content-suppress: {run_supp(lt,cid):.3f}",flush=True)
# (3) TRANSFER: a NEW context, frog at a DIFFERENT position (longer filler, shuffled facts)
import random
f2=[("silver","frog"),("amber","torch"),("ivory","drum"),("coral","mask"),("jade","spear"),("ruby","flute")]
ids2=build(f2,"silver", filler="Once upon a time there was a quiet little village by the sea where people told stories.")
t2=torch.tensor(ids2,device=dev).unsqueeze(0); seq2=t2.shape[1]; cpos2=ids2.index(cid); base2=torch.softmax(model(t2)[0][-1].float(),-1)[cid].item()
HF2=fra_ph(t2); byL2=fra_delta(HF2,P,seq2)   # SAME feature-pairs P, content-addressed onto context-2
print(f"\n--- (3) TRANSFER: NEW context, 'silver box holds a frog', frog at pos {cpos2} (baseline {base2:.3f}) ---",flush=True)
print(f"  FRA(c=8, same pairs):  {run_fra(t2,byL2,8,cid):.3f}   <- content-addressed",flush=True)
print(f"  attention-patch(pos {cpos}): {run_patch(t2,cpos,cid):.3f}   <- position-tied (wrong pos on ctx-2)",flush=True)
print(f"  content-suppress:      {run_supp(t2,cid):.3f}",flush=True)
out={"base":base,"fra":{c:run_fra(tt,byL,c,cid) for c in [2,4,8,16]},"patch":run_patch(tt,cpos,cid),"supp":run_supp(tt,cid),
     "sep_base":lbase,"sep_fra":run_fra(lt,byLl,8,cid),"sep_supp":run_supp(lt,cid),
     "transfer_base":base2,"transfer_fra":run_fra(t2,byL2,8,cid),"transfer_patch":run_patch(t2,cpos,cid),"transfer_supp":run_supp(t2,cid)}
json.dump(out,open(os.path.join(os.environ.get("OUTDIR","."),"r2.json"),"w"),indent=2,default=float)
print("\nDONE r2",flush=True)
