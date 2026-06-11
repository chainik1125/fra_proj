"""S4 — fill the table: Tier-2 collateral-advantage A for the remaining screened-in candidates.
 binding/coreference: full Tier-2 (suppress John's bound attribute 'doctor'; FRA vs content-suppress vs
   head-ablation; on-target + separability + transfer + collateral A). LBNR too.
 greater-than: LBNR only + honest MLP-routing check (the QK edge reads the year, but the > is computed
   downstream in MLP, so the *behavior* may not be QK-controllable even if the edge is load-bearing).
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gpt2",device=dev); model.eval(); tok=model.tokenizer
NL,NH=model.cfg.n_layers,model.cfg.n_heads; W_U=model.W_U
def enc(s): return torch.tensor([tok.bos_token_id]+tok.encode(s),device=dev).unsqueeze(0)
def autoheads(tt,Q,K,topk=5):
    _,c=model.run_with_cache(tt,names_filter=lambda n:n.endswith("hook_pattern")); sc={}
    for L in range(NL):
        pt=c[f"blocks.{L}.attn.hook_pattern"][0]
        for H in range(NH): sc[(L,H)]=float(pt[H,Q,K].item())
    return sorted(sc,key=lambda x:-sc[x])[:topk]
def cut(tt,heads,Q,K):
    byL={}
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
_sae={}
def get_sae(L):
    if L not in _sae:
        s=SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev); _sae[L]=s[0] if isinstance(s,tuple) else s
    return _sae[L]
RES={}
# ===================== BINDING =====================
print("=== BINDING/COREFERENCE ===",flush=True)
bind="John is a doctor. Mary is a lawyer. Tom is a pilot. John is a"
tt=enc(bind); toks=[tok.bos_token_id]+tok.encode(bind); doc=tok.encode(" doctor")[0]; Kd=toks.index(doc); Q=tt.shape[1]-1
P0=torch.softmax(model(tt)[0][Q].float(),-1)[doc].item()
heads=autoheads(tt,Q,Kd); print(f"  P(doctor)={P0:.3f}  heads={heads}",flush=True)
# LBNR
Bs=[torch.softmax(cut(tt,heads[:k],Q,Kd)[Q].float(),-1)[doc].item() for k in [1,3,5]]
R=1-Bs[-1]/P0; print(f"  LBNR: P(doctor) cut top-[1,3,5]={[round(b,3) for b in Bs]}  R={R:.2f}",flush=True)
# FRA edit on the retrieval edge
LY=sorted(set(L for L,H in heads))
def fra_edge(tt,L,H):
    sae=get_sae(L); HK=f"blocks.{L}.hook_resid_pre"
    fe=sae.encode(model.run_with_cache(tt,names_filter=lambda n:n==HK)[1][HK][0]).float()
    xh=fe@sae.W_dec.float()+sae.b_dec.float()
    r=_build_fra_result(model,L,H,fe,sae.W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
    return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
HF={(L,H):fra_edge(tt,L,H) for L,H in heads}
P={}
for (L,H) in heads:
    d=HF[(L,H)]; on=(d["qq"]==Q)&(d["kk"]==Kd)
    oi=np.where(on)[0][np.argsort(-np.abs(d["vv"][on]))[:12]]
    P[(L,H)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in oi)
def fra_delta(d,Ps,sq):
    dd=np.zeros((sq,sq))
    for n in range(len(d["vv"])):
        if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
    return dd
def run_fra(tt2,c,target):
    sq=tt2.shape[1]; hooks=[]; byL={}
    for (L,H) in heads:
        dd=torch.tensor(fra_delta(fra_edge(tt2,L,H),P[(L,H)],sq),device=dev,dtype=torch.float32)*c
        byL.setdefault(L,{})[H]=dd
    for L,hd in byL.items():
        def mk(hd):
            def hook(s,hook):
                for H,dd in hd.items(): s[0,H,:dd.shape[0],:dd.shape[1]]-=dd.to(s.dtype)
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(hd)))
    return torch.softmax(model.run_with_hooks(tt2,fwd_hooks=hooks)[0][Q if tt2 is tt else tt2.shape[1]-1].float(),-1)[target].item()
def content_suppress(tt2,target,s=6):
    Ld=NL-1; uP=W_U[:,target].float(); uP=uP/uP.norm()
    def hook(a,hook): a[0]=a[0]-(s*(a[0].float()@uP).unsqueeze(-1)*uP).to(a.dtype); return a
    return torch.softmax(model.run_with_hooks(tt2,fwd_hooks=[(f"blocks.{Ld}.hook_resid_post",hook)])[0][-1].float(),-1)[target].item()
def head_ablate(tt2,target):
    hooks=[]
    for L in LY:
        Hs=[H for LL,H in heads if LL==L]
        def mk(Hs):
            def hook(z,hook):
                for H in Hs: z[0,:,H,:]=0.0
                return z
            return hook
        hooks.append((f"blocks.{L}.attn.hook_z",mk(Hs)))
    return torch.softmax(model.run_with_hooks(tt2,fwd_hooks=hooks)[0][-1].float(),-1)[target].item()
fra_on=run_fra(tt,8,doc)
print(f"  ON-TARGET suppress doctor: FRA(c=8) {P0:.3f}->{fra_on:.3f} | content-suppress ->{content_suppress(tt,doc):.3f} | head-ablate ->{head_ablate(tt,doc):.3f}",flush=True)
# separability: legit context where 'doctor' is natural (no binding)
leg=enc("She felt sick so she went to see the"); ld=torch.softmax(model(leg)[0][-1].float(),-1)[doc].item()
sep_fra=run_fra(leg,8,doc); sep_cs=content_suppress(leg,doc)
print(f"  SEPARABILITY P(doctor) legit {ld:.3f}: FRA {sep_fra:.3f} | content-suppress {sep_cs:.3f}",flush=True)
# transfer: new binding, doctor different position
tr=enc("Sarah works as a teacher. The man named Greg works as a doctor today. Greg works as a")
tb=torch.softmax(model(tr)[0][-1].float(),-1)[doc].item(); tr_fra=run_fra(tr,8,doc)
print(f"  TRANSFER P(doctor) new ctx {tb:.3f}: FRA(same pairs) {tr_fra:.3f}",flush=True)
col_fra=abs(ld-sep_fra); col_cs=abs(ld-sep_cs); A=col_cs/max(col_fra,1e-4)
RES["binding"]={"P0":P0,"LBNR_R":R,"on_fra":fra_on,"on_cs":content_suppress(tt,doc),"on_ha":head_ablate(tt,doc),
                "sep_base":ld,"sep_fra":sep_fra,"sep_cs":sep_cs,"col_fra":col_fra,"col_cs":col_cs,"A":A,
                "transfer_base":tb,"transfer_fra":tr_fra}
print(f"  => collateral A (content-suppress/FRA) = {A:.1f}x",flush=True)
# ===================== GREATER-THAN =====================
print("\n=== GREATER-THAN (expect: edge load-bearing for reading year, but > is MLP) ===",flush=True)
gt="The war lasted from the year 1732 to the year 17"; tt=enc(gt); toks=[tok.bos_token_id]+tok.encode(gt); Q=tt.shape[1]-1
yy=[i for i,t in enumerate(toks) if "32" in tok.decode([t])]
if yy:
    Ky=yy[-1]
    twodig={n:tok.encode(f"{n:02d}") for n in range(100)}; twodig={n:v[0] for n,v in twodig.items() if len(v)==1}
    def gtmass(lg):
        p=torch.softmax(lg.float(),-1); hi=sum(p[twodig[n]].item() for n in twodig if n>32); lo=sum(p[twodig[n]].item() for n in twodig if n<=32)
        return hi,lo
    hi0,lo0=gtmass(model(tt)[0][Q]); heads=autoheads(tt,Q,Ky)
    hi1,lo1=gtmass(cut(tt,heads,Q,Ky)[Q])
    frac0=hi0/(hi0+lo0+1e-9); frac1=hi1/(hi1+lo1+1e-9)
    print(f"  P(>32 fraction): intact {frac0:.3f} -> cut year-edge {frac1:.3f}  (heads {heads})",flush=True)
    print(f"  raw mass hi/lo: intact {hi0:.3f}/{lo0:.3f} -> cut {hi1:.3f}/{lo1:.3f}",flush=True)
    RES["greater-than"]={"frac_intact":frac0,"frac_cut":frac1,"LBNR_R":1-frac1/frac0 if frac0 else float('nan'),
                          "note":"behavior is the >-comparison computed in MLP from the attended year; QK edge only reads the year"}
else: print("  could not locate year token",flush=True)
json.dump(RES,open(os.path.join(OUT,"s4.json"),"w"),indent=2,default=float)
print("\nDONE s4",flush=True)
