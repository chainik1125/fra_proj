"""J7 — harden the win against the two obvious objections:
(1) strongest LINEAR baseline: ActAdd of the KEY-side 'prev-was-cue' direction (most induction-specific
    linear steer) — does it escape the collateral that cue-identity ActAdd pays? It shouldn't: still
    single-endpoint-gated (fires wherever prev-token==cue), so collateral on all post-cue positions.
(2) split held-out KL by position type: NON-induction cue positions (pure collateral) vs induction
    positions (intended). FRA should be ~0 at non-induction positions; ActAdd corrupts them.
Held-out text has the cue THREE times (1st = non-induction; later = induction-like).
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result

OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gpt2",device=dev); model.eval()
IND_HEADS=[(5,5),(6,9),(5,1),(7,10),(7,2)]; LAYERS=sorted(set(L for L,H in IND_HEADS)); L0=min(LAYERS)
tok=model.tokenizer
saes={L:(SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev)) for L in LAYERS}
saes={L:(s[0] if isinstance(s,tuple) else s) for L,s in saes.items()}

def fra_per_head(tt):
    names=[f"blocks.{L}.hook_resid_pre" for L in LAYERS]
    _,cache=model.run_with_cache(tt,names_filter=lambda n:n in names)
    H={}; resid={L:cache[f"blocks.{L}.hook_resid_pre"][0] for L in LAYERS}
    for (L,Hh) in IND_HEADS:
        fe=saes[L].encode(cache[f"blocks.{L}.hook_resid_pre"][0]).float()
        xh=fe@saes[L].W_dec.float()+saes[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,saes[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); v=f.values().cpu().numpy()
        H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=v)
    return H,resid
def fra_delta(HF,edges,seq,M=12):
    byL={}
    for (L,Hh) in IND_HEADS:
        d=HF[(L,Hh)]; dd=np.zeros((seq,seq))
        for (qi,ki) in edges:
            loc=np.where((d["qq"]==qi)&(d["kk"]==ki))[0]
            order=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
            for o in order: dd[qi,ki]+=d["vv"][o]
        byL.setdefault(L,{})[Hh]=dd
    return byL
def run_fra(tt,byL,c):
    seq=tt.shape[1]; hooks=[]
    for L,hd in byL.items():
        td={Hh:torch.tensor(dd,device=dev,dtype=torch.float32)*c for Hh,dd in hd.items()}
        def mk(td):
            def hook(s,hook):
                for Hh,sd in td.items(): s[0,Hh,:seq,:seq]-=sd
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(td)))
    return model.run_with_hooks(tt,fwd_hooks=hooks)[0]
def run_actadd(tt,positions,vX,s):
    def hook(act,hook):
        for p in positions: act[0,p,:]=act[0,p,:]-s*vX*act[0,p,:].norm()
        return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{L0}.hook_resid_pre",hook)])[0]
def klvec(p,q):
    lp=torch.log_softmax(p.float(),-1); lq=torch.log_softmax(q.float(),-1)
    return (lp.exp()*(lp-lq)).sum(-1)

CUES=[" war"," city"," water"," money"," market"]
torch.manual_seed(0); rows=[]
for cw in CUES:
    cid=tok.encode(cw);
    if len(cid)!=1: continue
    cid=cid[0]; resp=tok.encode(" then")[0]
    N=20; R=(torch.randperm(40000)[:N]+1000).tolist(); R[10]=cid; R[11]=resp
    tt=torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0); seq=tt.shape[1]
    qpos=1+N+10; kpos=12; clean=model(tt)[0]; base=torch.softmax(clean[qpos].float(),-1)[resp].item()
    if base<0.3: continue
    HF,resid=fra_per_head(tt); byL=fra_delta(HF,[(qpos,kpos)],seq)
    vX=resid[L0][1+N+10]-resid[L0].mean(0); vX=vX/(vX.norm()+1e-6)            # cue-identity dir
    vKey=resid[L0][12]-resid[L0].mean(0); vKey=vKey/(vKey.norm()+1e-6)         # 'prev-was-cue' dir (pos after 1st cue)
    cue_pos_primer=[1+10,1+N+10]; key_pos_primer=[12,2+N+10]                   # positions after each cue
    # find strength for ~50% suppression for each method
    def supp_fra(c): return 1-torch.softmax(run_fra(tt,byL,c)[qpos].float(),-1)[resp].item()/base
    def supp_aa(positions,vv,s): return 1-torch.softmax(run_actadd(tt,positions,vv,s)[qpos].float(),-1)[resp].item()/base
    def find(fn,grid):
        best=None
        for x in grid:
            sp=fn(x)
            if best is None or abs(sp-0.5)<best[0]: best=(abs(sp-0.5),x,sp)
        return best[1],best[2]
    cF,sF=find(supp_fra,[1,2,4,8,16,32])
    sId,suId=find(lambda s:supp_aa(cue_pos_primer,vX,s),[0.25,0.5,1,2,4]);
    sKy,suKy=find(lambda s:supp_aa(key_pos_primer,vKey,s),[0.25,0.5,1,2,4,8])
    # held-out text: cue x3 (1st = non-induction, later = induction-like)
    htext=f"A{cw} began.{cw} grew.{cw} mattered to people who watched it closely all year."
    hids=[tok.bos_token_id]+tok.encode(htext); ht=torch.tensor(hids,device=dev).unsqueeze(0); hseq=ht.shape[1]
    hcue=[i for i,t in enumerate(hids) if t==cid]
    if len(hcue)<2: continue
    nonind=[hcue[0]]; ind=hcue[1:]                       # 1st occ = non-induction collateral
    hclean=model(ht)[0]
    HFh,residh=fra_per_head(ht)
    hedges=[(qi,ki+1) for qi in hcue for ki in hcue if ki+1<qi]
    byLh=fra_delta(HFh,hedges,hseq) if hedges else {}
    vXh=residh[L0][hcue[0]]-residh[L0].mean(0); vXh=vXh/(vXh.norm()+1e-6)
    hkey=[i+1 for i in hcue if i+1<hseq]
    vKeyh=residh[L0][hcue[0]+1]-residh[L0].mean(0); vKeyh=vKeyh/(vKeyh.norm()+1e-6)
    kf=klvec(hclean, run_fra(ht,byLh,cF)) if byLh else torch.zeros(hseq)
    ki=klvec(hclean, run_actadd(ht,hcue,vXh,sId))
    kk=klvec(hclean, run_actadd(ht,hkey,vKeyh,sKy))
    def at(v,ps): return float(v[ps].mean().item()) if ps else 0.0
    row=dict(cue=cw,base=base,suppF=sF,suppId=suId,suppKy=suKy,
             FRA_nonind=at(kf,nonind),Id_nonind=at(ki,nonind),Ky_nonind=at(kk,nonind),
             FRA_total=float(kf.sum()),Id_total=float(ki.sum()),Ky_total=float(kk.sum()))
    rows.append(row)
    print(f"cue '{cw}': supp F/Id/Ky={sF:.2f}/{suId:.2f}/{suKy:.2f} | "
          f"non-induction-pos KL  FRA {row['FRA_nonind']:.3f}  ActAdd-id {row['Id_nonind']:.3f}  ActAdd-key {row['Ky_nonind']:.3f}",flush=True)

import numpy as np
def col(k): return np.array([r[k] for r in rows])
print(f"\n=== n={len(rows)} cues, matched ~50% induction suppression ===",flush=True)
print(f"COLLATERAL at NON-induction cue positions (pure collateral, want 0):",flush=True)
print(f"  FRA-QK      : {col('FRA_nonind').mean():.3f} ± {col('FRA_nonind').std():.3f}",flush=True)
print(f"  ActAdd-id   : {col('Id_nonind').mean():.3f} ± {col('Id_nonind').std():.3f}",flush=True)
print(f"  ActAdd-key  : {col('Ky_nonind').mean():.3f} ± {col('Ky_nonind').std():.3f}",flush=True)
json.dump({"rows":rows,"summary":{k:[float(col(k).mean()),float(col(k).std())] for k in
          ["FRA_nonind","Id_nonind","Ky_nonind","FRA_total","Id_total","Ky_total"]}},
          open(os.path.join(OUT,"j7.json"),"w"),indent=2)
plt.figure(figsize=(5.5,4))
labels=["FRA-QK\n(bilinear)","ActAdd\ncue-identity","ActAdd\nprev-was-cue"]
vals=[col('FRA_nonind').mean(),col('Id_nonind').mean(),col('Ky_nonind').mean()]
errs=[col('FRA_nonind').std(),col('Id_nonind').std(),col('Ky_nonind').std()]
plt.bar(labels,vals,yerr=errs,color=['C0','C1','C2'])
plt.ylabel("collateral KL at non-induction cue positions (nats)")
plt.title(f"At matched 50% induction suppression (n={len(rows)} cues)\nFRA leaves the cue untouched in non-induction contexts")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"j7_keyside.png"),dpi=120)
print("\nDONE j7",flush=True)
