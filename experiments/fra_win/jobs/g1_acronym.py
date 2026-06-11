"""G_SCREEN — reusable CCF + LBNR screener for arbitrary candidate behaviors (gpt2-small or gemma-2-2b).
Fill CANDIDATES below; each runs the two-gate screen:
  CCF  = non-sink content-pair mass / total edge mass on the target edge (necessary: content conjunction)
  LBNR = cut the edge across top-k heads, R = 1 - B(cut)/B(intact) (load-bearing & non-redundant)
A candidate is a Tier-2-worthy FRA candidate iff CCF-high AND LBNR-R-high (near the induction anchor).
Each candidate: name, model('gpt2'|'gemma'), probe text, key (substring whose token is the edge key),
answer (substring whose token is the behavior metric P(answer)), optional fixed_heads [[L,H],...].
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
CANDIDATES = [
 {
  "name": "acronym",
  "model": "gpt2",
  "probe": "The Chief Executive Officer (CE",
  "key": " Officer",
  "answer": "O",
  "fixed_heads": [
   [
    8,
    11
   ],
   [
    9,
    9
   ],
   [
    10,
    10
   ],
   [
    11,
    4
   ]
  ]
 }
]
_models={}; _sae={}; _sink={}
def get_model(m):
    if m not in _models:
        name="gpt2" if m=="gpt2" else "gemma-2-2b"
        _models[m]=HookedTransformer.from_pretrained(name,device=dev,dtype=(torch.float16 if m=="gemma" else torch.float32)); _models[m].eval()
    return _models[m]
def get_sae(m,L):
    key=(m,L)
    if key not in _sae:
        if m=="gpt2":
            from sae_lens import SAE
            s=SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev); _sae[key]=("jb",s[0] if isinstance(s,tuple) else s)
        else:
            from fra.sae_lens_wrapper import GemmaScopeSAE
            _sae[key]=("gs",GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{L-1}/width_16k/canonical",device=dev,normalize_activations=True))
    return _sae[key]
def encode(m,L,resid):
    kind,sae=get_sae(m,L); fe=sae.encode(resid.float()).float()
    if kind=="gs" and sae._norm_coeff is not None: fe=fe/sae._norm_coeff
    return fe,sae
CORPUS=["The cat sat on the mat.","She walked to the store.","Photosynthesis converts light to sugar.",
 "He plays guitar on weekends.","The river flows past the mill.","Quantum computers use qubits.","They visited Paris.",
 "A frog leapt into the pond.","The market rose today.","Children laughed loudly.","The chef cooked a meal.",
 "Rain fell on the roof.","Scientists found a species.","The train was on time.","Books lined the shelves.",
 "A candle flickered.","The lamp lit the room.","Soldiers carried a sword.","The clock struck noon.","A rose bloomed.",
 "The doctor saw the patient.","A lawyer argued well.","John gave Mary a gift.","The lion roared.","A hammer fell."]
def sink_set(m,L):
    if (m,L) in _sink: return _sink[(m,L)]
    model=get_model(m); HK=f"blocks.{L}.hook_resid_pre"; kind,sae=get_sae(m,L); nf=sae.W_dec.shape[0]; A=np.zeros((len(CORPUS),nf),bool)
    for c,s in enumerate(CORPUS):
        r=model.run_with_cache(s,names_filter=lambda n:n==HK)[1][HK][0]; fe,_=encode(m,L,r); A[c]=(fe>0).any(0).cpu().numpy()
    _sink[(m,L)]=set(np.where(A.mean(0)>0.5)[0].tolist()); return _sink[(m,L)]
def fra_edge(m,L,H,tt):
    from fra.core.fra import _build_fra_result
    model=get_model(m); HK=f"blocks.{L}.hook_resid_pre"; r=model.run_with_cache(tt,names_filter=lambda n:n==HK)[1][HK][0]
    fe,sae=encode(m,L,r); xh=fe@sae.W_dec.float()+sae.b_dec.float()
    R=_build_fra_result(model,L,H,fe,sae.W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
    f=R["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
def autoheads(m,tt,Q,K,topk=8):
    model=get_model(m); _,c=model.run_with_cache(tt,names_filter=lambda n:n.endswith("hook_pattern")); sc={}
    for L in range(model.cfg.n_layers):
        pt=c[f"blocks.{L}.attn.hook_pattern"][0]
        for H in range(model.cfg.n_heads): sc[(L,H)]=float(pt[H,Q,K].item())
    return sorted(sc,key=lambda x:-sc[x])[:topk],sc
def cut(m,tt,heads,Q,K):
    model=get_model(m); byL={}
    for L,H in heads: byL.setdefault(L,[]).append(H)
    hooks=[]
    for L,Hs in byL.items():
        def mk(Hs):
            def hook(s,hook):
                for H in Hs: s[0,H,Q,K]=-1e4
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(Hs)))
    return model.run_with_hooks(tt,fwd_hooks=hooks)[0]
def ccf(m,heads,Q,K,tt):
    tot=0.; nons=0.
    for (L,H) in heads:
        d=fra_edge(m,L,H,tt); S=sink_set(m,L); on=(d["qq"]==Q)&(d["kk"]==K)
        if on.sum()==0: continue
        av=np.abs(d["vv"][on]); ii=d["ii"][on]; jj=d["jj"][on]; tot+=av.sum()
        keep=np.array([(int(ii[a]) not in S) and (int(jj[a]) not in S) for a in range(len(av))]); nons+=av[keep].sum()
    return (nons/tot) if tot>0 else float('nan')
def tokpos(model,ids,sub,before):
    t=model.tokenizer.encode(sub); want=t[0]
    cands=[i for i,x in enumerate(ids) if x==want and i<before]
    return cands[-1] if cands else None
RES={}
for cand in CANDIDATES:
    try:
        m=cand["model"]; model=get_model(m); ids=([model.tokenizer.bos_token_id] if m=="gpt2" else [])+model.tokenizer.encode(cand["probe"])
        if m=="gemma": ids=model.tokenizer.encode(cand["probe"])  # gemma auto-bos
        tt=torch.tensor(ids,device=dev).unsqueeze(0); Q=len(ids)-1
        K=tokpos(model,ids,cand["key"],Q); ans=model.tokenizer.encode(cand["answer"])[0] if m=="gpt2" else model.tokenizer.encode(cand["answer"],add_special_tokens=False)[0]
        if K is None: RES[cand["name"]]={"error":"key token not found"}; print(f"  {cand['name']}: key not found",flush=True); continue
        heads = [tuple(h) for h in cand["fixed_heads"]] if cand.get("fixed_heads") else autoheads(m,tt,Q,K)[0][:5]
        ma=np.mean([model.run_with_cache(tt,names_filter=lambda n:n.endswith("hook_pattern"))[1][f"blocks.{L}.attn.hook_pattern"][0,H,Q,K].item() for L,H in heads])
        cv=ccf(m,heads,Q,K,tt)
        B0=torch.softmax(model(tt)[0][Q].float(),-1)[ans].item()
        Bs=[torch.softmax(cut(m,tt,heads[:k],Q,K)[Q].float(),-1)[ans].item() for k in [1,3,5]]
        R=1-Bs[-1]/B0 if B0>0 else float('nan')
        RES[cand["name"]]={"model":m,"CCF":float(cv),"mean_attn":float(ma),"P_answer":float(B0),"P_cut":[float(b) for b in Bs],"LBNR_R":float(R),"heads":[list(h) for h in heads]}
        print(f"  {cand['name']:28s} CCF={cv:.3f} attn={ma:.2f} P(ans)={B0:.3f}->{Bs[-1]:.3f} R={R:+.2f}  {heads[:3]}",flush=True)
    except Exception as e:
        RES[cand["name"]]={"error":str(e)}; print(f"  {cand['name']}: ERROR {e}",flush=True)
json.dump(RES,open(os.path.join(OUT,"g_screen.json"),"w"),indent=2,default=float)
print("\nDONE g_screen",flush=True)
