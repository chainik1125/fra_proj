"""J4 — core win: selectively suppress ONE token's induction copying.
Methods aim to drop target token X's copy-prob while sparing other tokens:
  FRA-QK selective : ablate X's induction feature-pairs across all induction heads (over-drive c, sweep)
  ActAdd-X(query)  : subtract s * v_X (token-X identity dir) at X's query position, before induction layers (sweep s)
  random-pair      : FRA machinery but RANDOM pairs (null: should NOT suppress target)  [sanity]
  position-oracle  : zero the target edge scores across heads (selective, needs positions)
  head-ablate-all  : mean-ablate all induction heads (non-selective ref: kills all copying)
Collateral = mean |Δcopyprob| on the OTHER induction tokens (within-task selectivity).
Win = FRA reaches high target suppression at low collateral on a Pareto where ActAdd cannot.
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
IND_HEADS=[(5,5),(6,9),(5,1),(7,10),(7,2)]; LAYERS=sorted(set(L for L,H in IND_HEADS))
L0=min(LAYERS)
torch.manual_seed(0); N=24
R=(torch.randperm(40000)[:N]+1000).tolist()
toks=[model.tokenizer.bos_token_id]+R+R
tt=torch.tensor(toks,device=dev).unsqueeze(0); seq=tt.shape[1]
edges=[(1+N+t,t+2) for t in range(N-1)]; ind_targets={1+N+t:R[t+1] for t in range(N-1)}
names=[f"blocks.{L}.attn.hook_attn_scores" for L in LAYERS]+[f"blocks.{L}.hook_resid_pre" for L in LAYERS]
logits,cache=model.run_with_cache(tt,names_filter=lambda n:n in names)
def cpvec(lg):
    p=torch.softmax(lg[0].float(),-1); return np.array([p[q,ind_targets[q]].item() for q,k in edges])
base=cpvec(logits)
saes={L:(SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev)) for L in LAYERS}
saes={L:(s[0] if isinstance(s,tuple) else s) for L,s in saes.items()}
HFRA={}; rng=np.random.default_rng(0)
for (L,H) in IND_HEADS:
    sae=saes[L]; act=cache[f"blocks.{L}.hook_resid_pre"][0]
    feats=sae.encode(act).float(); x_hat=feats@sae.W_dec.float()+sae.b_dec.float()
    r=_build_fra_result(model,L,H,feats,sae.W_dec.float(),dev,top_k=None,rms_activations=x_hat,dec_norms=None,chunk_size=16,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); v=f.values().cpu().numpy()
    HFRA[(L,H)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=v)
SC={(L,H):cache[f"blocks.{L}.attn.hook_attn_scores"][0,H].float().cpu().numpy() for (L,H) in IND_HEADS}
resid={L:cache[f"blocks.{L}.hook_resid_pre"][0] for L in LAYERS}

def edge_pairs(LH,e,m):
    d=HFRA[LH]; loc=np.where((d["qq"]==e[0])&(d["kk"]==e[1]))[0]
    order=loc[np.argsort(-np.abs(d["vv"][loc]))[:m]]
    return set((int(d["ii"][o]),int(d["jj"][o])) for o in order)
def pairs_delta(LH,pset):
    d=HFRA[LH]; out=np.zeros((seq,seq))
    sel=[(int(d["ii"][n]),int(d["jj"][n])) in pset for n in range(len(d["vv"]))]
    for n in np.where(sel)[0]: out[d["qq"][n],d["kk"][n]]+=d["vv"][n]
    return out
def score_patch(deltas,scale):
    byL={}
    for (L,H),dd in deltas.items(): byL.setdefault(L,{})[H]=torch.tensor(dd,device=dev,dtype=torch.float32)*scale
    hooks=[]
    for L,hd in byL.items():
        def mk(hd):
            def hook(s,hook):
                for H,sd in hd.items(): s[0,H,:seq,:seq]-=sd
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(hd)))
    return cpvec(model.run_with_hooks(tt,fwd_hooks=hooks))

M=12
targets=[t for t in range(N-1) if base[t]>0.8][:8]
base_t=float(np.mean([base[t] for t in targets]))
def collat(v,t): return float(np.mean([abs(v[s]-base[s]) for s in range(N-1) if s!=t]))
print(f"targets {targets} base_t {base_t:.3f}",flush=True)

# precompute per-target FRA deltas + random-pair deltas
fra_deltas={}; rnd_deltas={}
all_pairs={}
for LH in IND_HEADS:
    d=HFRA[LH]; all_pairs[LH]=list(set(zip(d["ii"].tolist(),d["jj"].tolist())))
for t in targets:
    e=edges[t]
    fra_deltas[t]={LH:pairs_delta(LH,edge_pairs(LH,e,M)) for LH in IND_HEADS}
    rnd_deltas[t]={LH:pairs_delta(LH,set(tuple(all_pairs[LH][i]) for i in rng.choice(len(all_pairs[LH]),M,replace=False))) for LH in IND_HEADS}

# ActAdd-X: subtract s*vX at query pos (2nd-X) at resid_pre L0 (propagates to all induction heads)
def actadd(t,s):
    qpos=edges[t][0]
    vX=resid[L0][qpos]-resid[L0].mean(0); vX=vX/(vX.norm()+1e-6)
    def hook(act,hook):
        act[0,qpos,:]=act[0,qpos,:]-s*vX*act[0,qpos,:].norm()
        return act
    return cpvec(model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{L0}.hook_resid_pre",hook)]))

CS=[1,2,4,8,16,32]; AS=[0.25,0.5,1,2,4,8]
def curve(fn):  # returns list of (supp, collateral) over sweep
    pts=[]
    for x in (CS if fn=='fra' or fn=='rnd' else AS):
        tg=[]; co=[]
        for t in targets:
            if fn=='fra': v=score_patch(fra_deltas[t],x)
            elif fn=='rnd': v=score_patch(rnd_deltas[t],x)
            else: v=actadd(t,x)
            tg.append(v[t]); co.append(collat(v,t))
        pts.append((1-np.mean(tg)/base_t, float(np.mean(co)), x))
    return pts
fra_pts=curve('fra'); aa_pts=curve('aa'); rnd_pts=curve('rnd')
print("\nFRA (supp, collat, c):", [(round(a,2),round(b,3),c) for a,b,c in fra_pts],flush=True)
print("ActAdd-X (supp, collat, s):", [(round(a,2),round(b,3),c) for a,b,c in aa_pts],flush=True)
print("random-pair (supp, collat, c):", [(round(a,2),round(b,3),c) for a,b,c in rnd_pts],flush=True)

# oracle + headablate single points
or_tg=[]; or_co=[]
for t in targets:
    dd={LH:(np.zeros((seq,seq))) for LH in IND_HEADS}
    for LH in IND_HEADS: dd[LH][edges[t]]=SC[LH][edges[t][0],edges[t][1]]+20
    v=score_patch(dd,1.0); or_tg.append(v[t]); or_co.append(collat(v,t))
or_pt=(1-np.mean(or_tg)/base_t, float(np.mean(or_co)))
def killall():
    hooks=[]
    for L in LAYERS:
        Hs=[H for (LL,H) in IND_HEADS if LL==L]
        def mk(Hs):
            def hook(z,hook):
                for H in Hs: z[0,:,H,:]=z[0,:,H,:].mean(0,keepdim=True)
                return z
            return hook
        hooks.append((f"blocks.{L}.attn.hook_z",mk(Hs)))
    return cpvec(model.run_with_hooks(tt,fwd_hooks=hooks))
ka=killall(); ka_pt=(1-np.mean([ka[t] for t in targets])/base_t, float(np.mean([collat(ka,t) for t in targets])))
print(f"oracle (supp,collat): {or_pt}",flush=True)
print(f"head-ablate-all (supp,collat): {ka_pt}",flush=True)

out={"base_t":base_t,"fra":fra_pts,"actadd":aa_pts,"random":rnd_pts,"oracle":or_pt,"headablate":ka_pt}
json.dump(out,open(os.path.join(OUT,"j4.json"),"w"),indent=2)
plt.figure(figsize=(6,4.5))
for pts,lab,mk in [(fra_pts,"FRA-QK selective","o-"),(aa_pts,"ActAdd-X (linear)","s-"),(rnd_pts,"random pairs (null)","^:")]:
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; plt.plot(xs,ys,mk,label=lab)
plt.scatter([or_pt[0]],[or_pt[1]],c='green',marker='*',s=140,label="position oracle")
plt.scatter([ka_pt[0]],[ka_pt[1]],c='red',marker='x',s=90,label="kill all heads (non-selective)")
plt.xlabel("target-token copy suppression  (1 - cp/base)  → better"); plt.ylabel("collateral |Δcp| on other tokens  ←better")
plt.title("Selective induction suppression — FRA-QK vs baselines\n(down-right = selective + strong)")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT,"j4_pareto.png"),dpi=110)
print("\nDONE j4",flush=True)
