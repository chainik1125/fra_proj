"""R4 — r3's footprint-CSG was null (~1.3 everywhere). Test two cheaper intrinsic proxies for
'FRA will help here', for retrieval (FRA-shaped) vs control edges:
 (A) spectral conjunctivity of the edge's FRA contribution matrix M[i,j]:  C = 1 - sigma1^2/||M||_F^2
     (rank-1 M => one query-dir x one key-dir => a direction suffices => C~0 => FRA useless)
 (B) corpus reuse of the edge's dominant KEY feature: firing-rate over a natural-text corpus
     (content common => direction has collateral => FRA wins; content rare/specific => direction surgical)
The real discriminator is the MEASURED collateral advantage; this tests whether a 1-pass proxy exists.
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
def fra_head(t,L,Hh):
    sae=SAE[L]; _,c=model.run_with_cache(t,names_filter=lambda n:n==f"blocks.{L}.hook_resid_pre")
    x=c[f"blocks.{L}.hook_resid_pre"][0]; fe=sae.encode(x.float()).float()
    if sae._norm_coeff is not None: fe=fe/sae._norm_coeff
    xh=fe@sae.W_dec.float()+sae.b_dec.float()
    r=_build_fra_result(model,L,Hh,fe,sae.W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
    return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
def spectral_C(hd,Q,K):
    Cs=[]; dom=[]
    for d in hd:
        on=(d["qq"]==Q)&(d["kk"]==K)
        if on.sum()<2: continue
        ii=d["ii"][on]; jj=d["jj"][on]; vv=d["vv"][on]
        ui=np.unique(ii); uj=np.unique(jj)
        if len(ui)<2 or len(uj)<2: continue
        im={v:k for k,v in enumerate(ui)}; jm={v:k for k,v in enumerate(uj)}
        Mx=np.zeros((len(ui),len(uj)))
        for a in range(len(vv)): Mx[im[ii[a]],jm[jj[a]]]+=vv[a]
        s=np.linalg.svd(Mx,compute_uv=False)
        Cs.append(1-(s[0]**2)/(np.sum(s**2)+1e-9))
        # dominant key feature on this edge
        kw={}
        for a in range(len(vv)): kw[int(jj[a])]=kw.get(int(jj[a]),0)+abs(vv[a])
        dom.append(max(kw,key=kw.get))
    return (float(np.mean(Cs)) if Cs else float('nan')), dom
facts=[("red","frog"),("blue","lamp"),("green","clock"),("gold","rose"),("black","sword"),("white","candle")]
ctx="".join(f" The {k} box holds a {v}." for k,v in facts)+" The red box holds a"
ids=tok.encode(ctx); tt=torch.tensor(ids,device=dev).unsqueeze(0); seq=len(ids)
cpos=ids.index(tok.encode(" frog",add_special_tokens=False)[0]); dpos=ids.index(tok.encode(" sword",add_special_tokens=False)[0])
period=tok.encode(".",add_special_tokens=False)[0]; delim=max(i for i,x in enumerate(ids[:-1]) if x==period)
hd=[fra_head(tt,L,H) for (L,H) in RH]
edges={"retrieval(->correct-value)":cpos,"control-distractor":dpos,"control-delimiter":delim,"control-bos":0}
print("=== (A) spectral conjunctivity C = 1 - s1^2/||M||^2  (higher = less rank-1 = more bilinear) ===",flush=True)
domfeat={}
for name,K in edges.items():
    C,dom=spectral_C(hd,seq-1,K); domfeat[name]=dom
    print(f"  C={C:.3f}   {name}",flush=True)
# (B) corpus firing rate of each edge's dominant KEY feature (use top retrieval head's SAE, L14)
corpus=["The cat sat on the mat.","She walked to the store yesterday.","Photosynthesis converts light to energy.",
 "He plays guitar every weekend.","The river flows past the old mill.","Quantum computers use qubits.",
 "They visited Paris last summer.","A frog leapt into the pond.","The stock market rose sharply today.",
 "Children laughed in the playground.","The chef prepared a fine meal.","Rain fell softly on the roof.",
 "Scientists discovered a new species.","The train arrived on time.","Books lined the dusty shelves.",
 "A candle flickered in the dark.","The lamp lit the small room.","Soldiers carried a heavy sword.",
 "The clock struck midnight.","A rose bloomed in the garden."]
L0=RH[0][0]; sae0=SAE[L0]
def fire_rate(feat):
    cnt=0; tot=0
    for s in corpus:
        x=model.run_with_cache(tok.encode(s),names_filter=lambda n:n==f"blocks.{L0}.hook_resid_pre")[1][f"blocks.{L0}.hook_resid_pre"][0]
        fe=sae0.encode(x.float())
        cnt+=int((fe[:,feat]>0).sum().item()); tot+=fe.shape[0]
    return cnt/max(tot,1)
print("\n=== (B) corpus firing-rate of the edge's dominant key-feature (head L%dH%d) ==="%(RH[0][0],RH[0][1]),flush=True)
rates={}
for name,K in edges.items():
    dom=domfeat[name]
    if not dom: print(f"  (no dom feat) {name}"); continue
    fr=fire_rate(dom[0]); rates[name]=fr
    print(f"  key-feat {dom[0]:6d}  firing-rate {fr:.3f}   {name}",flush=True)
json.dump({"spectral_C":{n:spectral_C(hd,seq-1,K)[0] for n,K in edges.items()},"corpus_rate":rates},
          open(os.path.join(os.environ.get("OUTDIR","."),"r4.json"),"w"),indent=2,default=float)
print("\nDONE r4",flush=True)
