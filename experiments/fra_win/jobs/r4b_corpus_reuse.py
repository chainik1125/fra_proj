"""R4b — the corpus-reuse metric, done right (r3 footprint-CSG and r4 spectral-C both failed).
Mechanism of the win: cutting the EDGE damages a context only when the query-feature i* AND key-feature
j* both fire there (the conjunction); suppressing the key DIRECTION j* damages every context where j*
fires at all. So the predicted FRA advantage = how much more common the marginal is than the conjunction:
   REUSE = ctx_rate(j* fires anywhere) / ctx_rate(i* AND j* both fire)   [>>1 => FRA wins]
Measured over a natural-text corpus (forward passes only, no interventions) — induction-score-like.
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
L0,H0=15,0; sae=GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{L0-1}/width_16k/canonical",device=dev,normalize_activations=True)
def fra_head(t):
    _,c=model.run_with_cache(t,names_filter=lambda n:n==f"blocks.{L0}.hook_resid_pre")
    x=c[f"blocks.{L0}.hook_resid_pre"][0]; fe=sae.encode(x.float()).float()
    if sae._norm_coeff is not None: fe=fe/sae._norm_coeff
    xh=fe@sae.W_dec.float()+sae.b_dec.float()
    r=_build_fra_result(model,L0,H0,fe,sae.W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
    return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
def top_pair(d,Q,K):
    on=(d["qq"]==Q)&(d["kk"]==K)
    if on.sum()==0: return None
    agg={}
    for o in np.where(on)[0]: agg[(int(d["ii"][o]),int(d["jj"][o]))]=agg.get((int(d["ii"][o]),int(d["jj"][o])),0)+abs(d["vv"][o])
    return max(agg,key=agg.get)
facts=[("red","frog"),("blue","lamp"),("green","clock"),("gold","rose"),("black","sword"),("white","candle")]
ctx="".join(f" The {k} box holds a {v}." for k,v in facts)+" The red box holds a"
ids=tok.encode(ctx); tt=torch.tensor(ids,device=dev).unsqueeze(0); seq=len(ids)
cpos=ids.index(tok.encode(" frog",add_special_tokens=False)[0]); dpos=ids.index(tok.encode(" sword",add_special_tokens=False)[0])
period=tok.encode(".",add_special_tokens=False)[0]; delim=max(i for i,x in enumerate(ids[:-1]) if x==period)
d=fra_head(tt)
edges={"retrieval(->correct-value)":cpos,"control-distractor":dpos,"control-delimiter":delim,"control-bos":0}
pairs={n:top_pair(d,seq-1,K) for n,K in edges.items()}
corpus=["The cat sat on the mat.","She walked to the store yesterday.","Photosynthesis converts light to energy.",
 "He plays guitar every weekend.","The river flows past the old mill.","Quantum computers use qubits.",
 "They visited Paris last summer.","A frog leapt into the pond.","The stock market rose sharply today.",
 "Children laughed in the playground.","The chef prepared a fine meal.","Rain fell softly on the roof.",
 "Scientists discovered a new species.","The train arrived on time.","Books lined the dusty shelves.",
 "A candle flickered in the dark.","The lamp lit the small room.","Soldiers carried a heavy sword.",
 "The clock struck midnight.","A rose bloomed in the garden.","The red box was on the table.",
 "He opened the wooden box slowly.","What does the box contain?","A box of old letters."]
feats=sorted(set(f for p in pairs.values() if p for f in p))
# context-level activation: for each corpus sentence, which of our features fire anywhere
act={f:[] for f in feats}
for s in corpus:
    fe=sae.encode(model.run_with_cache(s,names_filter=lambda n:n==f"blocks.{L0}.hook_resid_pre")[1][f"blocks.{L0}.hook_resid_pre"][0].float())
    for f in feats: act[f].append(bool((fe[:,f]>0).any().item()))
act={f:np.array(v) for f,v in act.items()}
N=len(corpus)
print("=== (B) corpus-reuse REUSE = rate(key-feat j*) / rate(i* AND j*)  [>>1 => FRA wins] ===",flush=True)
out={}
for n,p in pairs.items():
    if p is None: print(f"  (no pair) {n}"); continue
    i,j=p
    rk=act[j].mean(); ri=act[i].mean(); rconj=(act[i]&act[j]).mean()
    reuse=(rk/rconj) if rconj>0 else float('inf')
    out[n]={"i":i,"j":j,"rate_q":float(ri),"rate_k":float(rk),"rate_conj":float(rconj),"REUSE":float(reuse)}
    print(f"  REUSE={reuse:6.2f}  rate_q(i*={i})={ri:.2f} rate_k(j*={j})={rk:.2f} rate_conj={rconj:.2f}   {n}",flush=True)
json.dump(out,open(os.path.join(os.environ.get("OUTDIR","."),"r4b.json"),"w"),indent=2,default=float)
print("\nDONE r4b",flush=True)
