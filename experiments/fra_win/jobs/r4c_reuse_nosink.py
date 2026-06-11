"""R4c — corpus-reuse metric with attention-SINK features excluded (r4b showed the top pair was a
ubiquitous sink feature 15887 firing everywhere, contaminating every intrinsic metric).
Fix: compute each feature's corpus context-rate; treat rate>0.5 as a sink; pick each edge's dominant
NON-SINK pair; then REUSE = rate(key-feat j*) / rate(i* AND j*).  Decisive test of whether a cheap
forward-pass-only metric discriminates retrieval (FRA-shaped) from control edges.
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
HK=f"blocks.{L0}.hook_resid_pre"
def feats(s_or_t): return sae.encode(model.run_with_cache(s_or_t,names_filter=lambda n:n==HK)[1][HK][0].float())
# corpus context-activation for ALL features -> sink set + per-feature context rate
corpus=["The cat sat on the mat.","She walked to the store yesterday.","Photosynthesis converts light to energy.",
 "He plays guitar every weekend.","The river flows past the old mill.","Quantum computers use qubits.",
 "They visited Paris last summer.","A frog leapt into the pond.","The stock market rose sharply today.",
 "Children laughed in the playground.","The chef prepared a fine meal.","Rain fell softly on the roof.",
 "Scientists discovered a new species.","The train arrived on time.","Books lined the dusty shelves.",
 "A candle flickered in the dark.","The lamp lit the small room.","Soldiers carried a heavy sword.",
 "The clock struck midnight.","A rose bloomed in the garden.","The red box was on the table.",
 "He opened the wooden box slowly.","What does the box contain?","A box of old letters."]
N=len(corpus); nf=sae.W_dec.shape[0]
ctxact=np.zeros((N,nf),bool)
for c,s in enumerate(corpus):
    fe=feats(s); ctxact[c]=(fe>0).any(0).cpu().numpy()
rate=ctxact.mean(0)              # per-feature context rate
SINK=set(np.where(rate>0.5)[0].tolist())
print(f"{len(SINK)} sink features (context-rate>0.5) excluded, e.g. {sorted(SINK)[:8]}",flush=True)
def fra_head(t):
    fe=feats(t).float()
    if sae._norm_coeff is not None: fe=fe/sae._norm_coeff
    xh=fe@sae.W_dec.float()+sae.b_dec.float()
    r=_build_fra_result(model,L0,H0,fe,sae.W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
    return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
def top_pair_nosink(d,Q,K):
    on=(d["qq"]==Q)&(d["kk"]==K); agg={}
    for o in np.where(on)[0]:
        i,j=int(d["ii"][o]),int(d["jj"][o])
        if i in SINK or j in SINK: continue
        agg[(i,j)]=agg.get((i,j),0)+abs(d["vv"][o])
    return max(agg,key=agg.get) if agg else None
facts=[("red","frog"),("blue","lamp"),("green","clock"),("gold","rose"),("black","sword"),("white","candle")]
ctx="".join(f" The {k} box holds a {v}." for k,v in facts)+" The red box holds a"
ids=tok.encode(ctx); tt=torch.tensor(ids,device=dev).unsqueeze(0); seq=len(ids)
cpos=ids.index(tok.encode(" frog",add_special_tokens=False)[0]); dpos=ids.index(tok.encode(" sword",add_special_tokens=False)[0])
period=tok.encode(".",add_special_tokens=False)[0]; delim=max(i for i,x in enumerate(ids[:-1]) if x==period)
d=fra_head(tt)
edges={"retrieval(->correct-value)":cpos,"control-distractor":dpos,"control-delimiter":delim,"control-bos":0}
print("\n=== REUSE = rate(key-feat j*) / rate(i* AND j*)   [>>1 => FRA wins; sinks excluded] ===",flush=True)
out={}
for n,K in edges.items():
    p=top_pair_nosink(d,seq-1,K)
    if p is None: print(f"  (no non-sink pair) {n}"); continue
    i,j=p; rk=rate[j]; ri=rate[i]; rconj=(ctxact[:,i]&ctxact[:,j]).mean()
    reuse=(rk/rconj) if rconj>0 else float('inf')
    out[n]={"i":i,"j":j,"rate_q":float(ri),"rate_k":float(rk),"rate_conj":float(rconj),"REUSE":float(reuse)}
    print(f"  REUSE={reuse:6.2f}  i*={i}(r={ri:.2f}) j*={j}(r={rk:.2f}) conj={rconj:.2f}   {n}",flush=True)
json.dump(out,open(os.path.join(os.environ.get("OUTDIR","."),"r4c.json"),"w"),indent=2,default=float)
print("\nDONE r4c",flush=True)
