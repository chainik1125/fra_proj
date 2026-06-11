"""CLASS_UNION — test the theory's DISTINCTIVE untested capability: a FAMILY-UNION cell edit.
Cut the UNION of (query x DIGIT-CLASS-key) FRA cells -> suppress retrieval of ANY digit-class value,
while preserving word-class retrieval. No linear steer can target a feature-class conjunction without
broadcasting. gemma-2-2b. Context mixes digit values (7,3) and word values (frog,lamp).
  digit-class key feature = SAE feature most selective for digit tokens (mean act digit - mean act word).
  on-target = P(digit | digit-query) drop; preserve = P(word | word-query); vs projection-removal steer
  of the digit-class direction (should broadcast: damage digit processing on a held-out digit sentence).
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
# values: digits (7,3) and words (frog,lamp); all single-token
vals={"red":"7","blue":"frog","green":"3","gold":"lamp"}
vid={k:tok.encode(" "+v,add_special_tokens=False)[0] for k,v in vals.items()}
def build(qk): return tok.encode("".join(f" The {k} box holds {v}." for k,v in vals.items())+f" The {qk} box holds")
def Pv(t,target,hooks=None):
    lg=(model.run_with_hooks(t,fwd_hooks=hooks) if hooks else model(t))[0]; return torch.softmax(lg[-1].float(),-1)[target].item()
# identify the DIGIT-CLASS key feature at each retrieval layer (selective for digit tokens)
digit_ids=[tok.encode(" "+d,add_special_tokens=False)[0] for d in "0123456789"]
word_ids=[tok.encode(" "+w,add_special_tokens=False)[0] for w in ["frog","lamp","clock","rose","cat","dog","tree","house"]]
def class_feat(L):
    HK=f"blocks.{L}.hook_resid_pre"
    dprobe=tok.decode([])  # build a sentence of digits and words
    ds=" The numbers were 7 3 5 1 9 2 written down."; ws=" The animals were frog lamp clock rose cat seen."
    fed=enc(L,model.run_with_cache(ds,names_filter=lambda n:n==HK)[1][HK][0]).mean(0)
    few=enc(L,model.run_with_cache(ws,names_filter=lambda n:n==HK)[1][HK][0]).mean(0)
    return int((fed-few).argmax().item())
DCF={L:class_feat(L) for L in LAYERS}
print("digit-class key feature per layer:",DCF,flush=True)
def fra_ph(t):
    _,c=model.run_with_cache(t,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS]); H={}
    for (L,Hh) in RH:
        fe=enc(L,c[f"blocks.{L}.hook_resid_pre"][0]); xh=fe@SAE[L].W_dec.float()+SAE[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,SAE[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H
def union_hooks(t,c=8):
    # cut ALL cells whose KEY feature == the digit-class feature (family union), any query, any key position
    HF=fra_ph(t); sq=t.shape[1]; byL={}
    for (L,Hh) in RH:
        d=HF[(L,Hh)]; jcls=DCF[L]; dd=np.zeros((sq,sq))
        mask=(d["jj"]==jcls)
        for n in np.where(mask)[0]: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
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
def steer_hooks(a):  # projection-removal of the digit-class direction (the linear analog)
    hk=[]
    for L in LAYERS:
        v=SAE[L].W_dec[DCF[L]].float(); dh=(v/v.norm()).to(dev)
        def mk(dh):
            def hook(resid,hook):
                x=resid[0].float(); proj=(x@dh).unsqueeze(-1)*dh.unsqueeze(0); resid[0]=(x-a*proj).to(resid.dtype); return resid
            return hook
        hk.append((f"blocks.{L}.hook_resid_pre",mk(dh)))
    return hk
print("\n=== FAMILY-UNION cut of (query x digit-class-key) ===",flush=True)
for qk in ["red","green","blue","gold"]:
    t=torch.tensor(build(qk),device=dev).unsqueeze(0); tgt=vid[qk]; cls="digit" if vals[qk] in "0123456789" else "word"
    base=Pv(t,tgt); fu=Pv(t,tgt,union_hooks(t,8))
    print(f"  {qk}->{vals[qk]} [{cls}]: base {base:.3f} -> union-cut {fu:.3f}  ({'SUPPRESSED' if fu<0.5*base else 'preserved'})",flush=True)
# linear-steer collateral on a held-out digit sentence (broadcast damage)
held=tok.encode(" My phone number is 5 5 5 1 2 3 4 and my address"); ht=torch.tensor(held,device=dev).unsqueeze(0)
def kl(t,hooks):
    p=torch.log_softmax(model(t)[0][-1].float(),-1); q=torch.log_softmax(model.run_with_hooks(t,fwd_hooks=hooks)[0][-1].float(),-1); return (p.exp()*(p-q)).sum().item()
print(f"\nHELD-OUT digit sentence collateral KL: FRA-union {kl(ht,union_hooks(ht,8)):.3f} | digit-class STEER {kl(ht,steer_hooks(4)):.3f}",flush=True)
json.dump({"digit_feat":DCF,"results":{qk:{"base":Pv(torch.tensor(build(qk),device=dev).unsqueeze(0),vid[qk]),
           "union":Pv(torch.tensor(build(qk),device=dev).unsqueeze(0),vid[qk],union_hooks(torch.tensor(build(qk),device=dev).unsqueeze(0),8))} for qk in vals},
           "heldout_kl_union":kl(ht,union_hooks(ht,8)),"heldout_kl_steer":kl(ht,steer_hooks(4))},
          open(os.path.join(OUT,"class_union.json"),"w"),indent=2,default=float)
print("\nDONE class_union",flush=True)
