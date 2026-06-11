"""R5 — Content-Conjunction Fraction (CCF): the cheap one-forward-pass screen that survived.
r4c revealed it: retrieval/distractor edges carry genuine non-sink content-feature pairs; delimiter/bos
edges are pure attention-sink (no non-sink pair). Quantify:
   CCF(edge) = sum|FRA mass| from NON-SINK feature-pairs / total |FRA mass| on the edge.
High CCF => the edge is a content x content conjunction (necessary for FRA to help). CCF~0 => structural/
positional (a position-patch or direction suffices; FRA can't help). Validate retrieval/induction (high)
vs delimiter/bos (low).
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
RH=[(15,0),(18,6),(6,3),(21,5),(22,4)]; RLAY=sorted(set(L for L,H in RH))
SAE={L:GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{L-1}/width_16k/canonical",device=dev,normalize_activations=True) for L in RLAY}
corpus=["The cat sat on the mat.","She walked to the store yesterday.","Photosynthesis converts light to energy.",
 "He plays guitar every weekend.","The river flows past the old mill.","Quantum computers use qubits.",
 "They visited Paris last summer.","A frog leapt into the pond.","The stock market rose sharply today.",
 "Children laughed in the playground.","The chef prepared a fine meal.","Rain fell softly on the roof.",
 "Scientists discovered a new species.","The train arrived on time.","Books lined the dusty shelves.",
 "A candle flickered in the dark.","The lamp lit the small room.","Soldiers carried a heavy sword.",
 "The clock struck midnight.","A rose bloomed in the garden.","The red box was on the table.",
 "He opened the wooden box slowly.","What does the box contain?","A box of old letters."]
def sink_set(L):
    sae=SAE[L]; HK=f"blocks.{L}.hook_resid_pre"; nf=sae.W_dec.shape[0]; ctx=np.zeros((len(corpus),nf),bool)
    for c,s in enumerate(corpus):
        fe=sae.encode(model.run_with_cache(s,names_filter=lambda n:n==HK)[1][HK][0].float())
        ctx[c]=(fe>0).any(0).cpu().numpy()
    return set(np.where(ctx.mean(0)>0.5)[0].tolist())
SINK={L:sink_set(L) for L in RLAY}
print({L:len(SINK[L]) for L in RLAY},"sink features per layer",flush=True)
def fra_head(t,L,Hh):
    sae=SAE[L]; HK=f"blocks.{L}.hook_resid_pre"
    fe=sae.encode(model.run_with_cache(t,names_filter=lambda n:n==HK)[1][HK][0].float()).float()
    if sae._norm_coeff is not None: fe=fe/sae._norm_coeff
    xh=fe@sae.W_dec.float()+sae.b_dec.float()
    r=_build_fra_result(model,L,Hh,fe,sae.W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
    return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy(),L=L)
def ccf(hd,Q,K):
    tot=0.0; nons=0.0
    for d in hd:
        on=(d["qq"]==Q)&(d["kk"]==K)
        if on.sum()==0: continue
        av=np.abs(d["vv"][on]); ii=d["ii"][on]; jj=d["jj"][on]; S=SINK[d["L"]]
        tot+=av.sum()
        keep=np.array([(int(ii[a]) not in S) and (int(jj[a]) not in S) for a in range(len(av))])
        nons+=av[keep].sum()
    return (nons/tot) if tot>0 else float('nan')
facts=[("red","frog"),("blue","lamp"),("green","clock"),("gold","rose"),("black","sword"),("white","candle")]
ctx="".join(f" The {k} box holds a {v}." for k,v in facts)+" The red box holds a"
ids=tok.encode(ctx); tt=torch.tensor(ids,device=dev).unsqueeze(0); seq=len(ids)
cpos=ids.index(tok.encode(" frog",add_special_tokens=False)[0]); dpos=ids.index(tok.encode(" sword",add_special_tokens=False)[0])
period=tok.encode(".",add_special_tokens=False)[0]; delim=max(i for i,x in enumerate(ids[:-1]) if x==period)
hd=[fra_head(tt,L,H) for (L,H) in RH]
res={}
for n,K in {"retrieval(->correct-value)":cpos,"control-distractor(->wrong-value)":dpos,"control-delimiter(->period)":delim,"control-bos(->pos0)":0}.items():
    res[n]=ccf(hd,seq-1,K)
# induction edge
rng=np.random.RandomState(0); R=rng.randint(1000,20000,size=25).tolist()
si=[tok.bos_token_id]+R+R; ti=torch.tensor(si,device=dev).unsqueeze(0); Lr=len(R)
hdi=[fra_head(ti,L,H) for (L,H) in RH]   # reuse same heads (they include induction heads L6H3,L18H6,L15H0)
res["induction(2nd-occ->after-1st-occ)"]=float(np.nanmean([ccf(hdi,1+Lr+t,2+t) for t in [5,10,15,20]]))
res["induction-control(final->bos)"]=ccf(hdi,len(si)-1,0)
print("\n=== CCF = non-sink content-pair mass / total edge mass  [high => content conjunction => FRA-candidate] ===",flush=True)
for n,v in res.items(): print(f"  CCF={v:.3f}   {n}",flush=True)
json.dump(res,open(os.path.join(os.environ.get("OUTDIR","."),"r5.json"),"w"),indent=2,default=float)
print("\nDONE r5",flush=True)
