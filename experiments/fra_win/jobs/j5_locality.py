"""J5 — the real test: does FRA's DOUBLE gate (query-feat AND key-feat) make it more surgical
than a content-addressed linear steer that must fire wherever the token appears?
At MATCHED induction suppression, compare BROAD collateral:
  (A) per-position full-vocab KL on the induction sequence (off-target KL mass)
  (B) held-out real-text KL: install each content-addressed hook on normal English and measure
      how much it perturbs predictions where the cue token appears in NON-induction contexts.
FRA score-edit fires only where (query-feat[cue] & key-feat[prev-was-cue]) co-occur (induction-like);
ActAdd-cue fires wherever current-token=cue -> corrupts the cue everywhere.
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

def kl_full(p_logits,q_logits):  # mean KL(p||q) over positions; [seq,vocab]
    lp=torch.log_softmax(p_logits.float(),-1); lq=torch.log_softmax(q_logits.float(),-1)
    return (lp.exp()*(lp-lq)).sum(-1)  # [seq]

# ============ Part 1: induction sequence, per-position KL locality ============
torch.manual_seed(0); N=24
R=(torch.randperm(40000)[:N]+1000).tolist()
toks=[model.tokenizer.bos_token_id]+R+R
tt=torch.tensor(toks,device=dev).unsqueeze(0); seq=tt.shape[1]
edges=[(1+N+t,t+2) for t in range(N-1)]; ind_targets={1+N+t:R[t+1] for t in range(N-1)}
names=[f"blocks.{L}.attn.hook_attn_scores" for L in LAYERS]+[f"blocks.{L}.hook_resid_pre" for L in LAYERS]
clean_logits,cache=model.run_with_cache(tt,names_filter=lambda n:n in names)
clean_logits=clean_logits[0]
def cp_at(lg,t): return torch.softmax(lg[edges[t][0]].float(),-1)[ind_targets[edges[t][0]]].item()
saes={L:(SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev)) for L in LAYERS}
saes={L:(s[0] if isinstance(s,tuple) else s) for L,s in saes.items()}
HFRA={}
for (L,H) in IND_HEADS:
    sae=saes[L]; act=cache[f"blocks.{L}.hook_resid_pre"][0]
    feats=sae.encode(act).float(); x_hat=feats@sae.W_dec.float()+sae.b_dec.float()
    r=_build_fra_result(model,L,H,feats,sae.W_dec.float(),dev,top_k=None,rms_activations=x_hat,dec_norms=None,chunk_size=16,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); v=f.values().cpu().numpy()
    HFRA[(L,H)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=v)
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
def fra_run(t,c,M=12):
    deltas={LH:pairs_delta(LH,edge_pairs(LH,edges[t],M)) for LH in IND_HEADS}
    byL={}
    for LH,dd in deltas.items(): byL.setdefault(LH[0],{})[LH[1]]=torch.tensor(dd,device=dev,dtype=torch.float32)*c
    hooks=[]
    for L,hd in byL.items():
        def mk(hd):
            def hook(s,hook):
                for H,sd in hd.items(): s[0,H,:seq,:seq]-=sd
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(hd)))
    return model.run_with_hooks(tt,fwd_hooks=hooks)[0]
def actadd_run(t,s):  # content-addressed: subtract cue r_t direction at BOTH its occurrences
    rt=R[t]; positions=[1+t,1+N+t]
    vX=resid[L0][1+N+t]-resid[L0].mean(0); vX=vX/(vX.norm()+1e-6)
    def hook(act,hook):
        for p in positions: act[0,p,:]=act[0,p,:]-s*vX*act[0,p,:].norm()
        return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{L0}.hook_resid_pre",hook)])[0]

targets=[t for t in range(N-1) if cp_at(clean_logits,t)>0.85][:8]
# match suppression ~0.7: FRA c=8, ActAdd s search
fra_off=[]; aa_off=[]; fra_supp=[]; aa_supp=[]
for t in targets:
    lf=fra_run(t,8); la=None
    # pick ActAdd s giving comparable suppression to FRA
    fs=1-torch.softmax(lf[edges[t][0]].float(),-1)[ind_targets[edges[t][0]]].item()/cp_at(clean_logits,t)
    best=None
    for s in [0.5,1,2,4]:
        l=actadd_run(t,s); sup=1-torch.softmax(l[edges[t][0]].float(),-1)[ind_targets[edges[t][0]]].item()/cp_at(clean_logits,t)
        if best is None or abs(sup-fs)<best[0]: best=(abs(sup-fs),l,sup)
    la=best[1]; as_=best[2]
    klf=kl_full(clean_logits,lf); kla=kl_full(clean_logits,la)
    qpos=edges[t][0]
    off_f=float(klf.sum().item()-klf[qpos].item())   # KL mass away from the target query position
    off_a=float(kla.sum().item()-kla[qpos].item())
    fra_off.append(off_f); aa_off.append(off_a); fra_supp.append(fs); aa_supp.append(as_)
    # count positions changed (KL>0.01)
print("PART 1 — induction-sequence locality (matched suppression):",flush=True)
print(f"  FRA   : suppression {np.mean(fra_supp):.2f}  off-target KL mass {np.mean(fra_off):.3f}",flush=True)
print(f"  ActAdd: suppression {np.mean(aa_supp):.2f}  off-target KL mass {np.mean(aa_off):.3f}",flush=True)
print(f"  ratio ActAdd/FRA off-target collateral: {np.mean(aa_off)/max(np.mean(fra_off),1e-4):.1f}x",flush=True)

# ============ Part 2: held-out real text, cue token in normal contexts ============
real=(" The war lasted for many years. During the war, soldiers fought bravely. "
      "After the war ended, the war was remembered by all who lived through the war.")
rtoks=model.tokenizer.encode(real)
# cue = a token that appears >=3 times
from collections import Counter
cnt=Counter(rtoks); cue=max([tk for tk,c in cnt.items() if c>=3], key=lambda tk:cnt[tk])
cue_str=model.tokenizer.decode([cue])
rt=torch.tensor([model.tokenizer.bos_token_id]+rtoks,device=dev).unsqueeze(0); rseq=rt.shape[1]
cue_pos=[i for i,tk in enumerate([model.tokenizer.bos_token_id]+rtoks) if tk==cue]
print(f"\nPART 2 — held-out real text. cue='{cue_str}' appears at {cue_pos}",flush=True)
rclean,rcache=model.run_with_cache(rt,names_filter=lambda n:n in
    [f"blocks.{L}.hook_resid_pre" for L in LAYERS]+[f"blocks.{L}.attn.hook_attn_scores" for L in LAYERS])
rclean=rclean[0]
# FRA content-addressed on real text: ablate cue's induction pairs (computed from the induction seq pair-set
# for a cue-matched edge is hard cross-context; instead: ablate the induction-head score wherever
# query token == cue AND key is the position after a previous cue occurrence -> emulate by recomputing FRA on real text
rfeats={L:saes[L].encode(rcache[f"blocks.{L}.hook_resid_pre"][0]).float() for L in LAYERS}
RFRA={}
for (L,H) in IND_HEADS:
    fe=rfeats[L]; xh=fe@saes[L].W_dec.float()+saes[L].b_dec.float()
    r=_build_fra_result(model,L,H,fe,saes[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); v=f.values().cpu().numpy()
    RFRA[(L,H)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=v)
# cue induction pairs on real text: query positions where token==cue, key = position right after a PRIOR cue
cue_edges=[]
for qi in cue_pos:
    for ki in cue_pos:
        if ki+1<qi: cue_edges.append((qi,ki+1))  # attend back to token after an earlier cue
# build FRA delta on real text for these edges (top-M pairs each), apply with overdrive
def real_fra_delta(c=8,M=12):
    byL={}
    for (L,H) in IND_HEADS:
        d=RFRA[(L,H)]; dd=np.zeros((rseq,rseq))
        for (qi,ki) in cue_edges:
            loc=np.where((d["qq"]==qi)&(d["kk"]==ki))[0]
            order=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
            for o in order: dd[qi,ki]+=d["vv"][o]
        byL.setdefault(L,{})[H]=torch.tensor(dd,device=dev,dtype=torch.float32)*c
    return byL
byL=real_fra_delta()
hooks=[]
for L,hd in byL.items():
    def mk(hd):
        def hook(s,hook):
            for H,sd in hd.items(): s[0,H,:rseq,:rseq]-=sd
            return s
        return hook
    hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(hd)))
r_fra=model.run_with_hooks(rt,fwd_hooks=hooks)[0]
# ActAdd content-addressed on real text: subtract cue dir wherever token==cue
vX=rcache[f"blocks.{L0}.hook_resid_pre"][0][cue_pos[0]]-rcache[f"blocks.{L0}.hook_resid_pre"][0].mean(0)
vX=vX/(vX.norm()+1e-6)
def ahook(act,hook):
    for p in cue_pos: act[0,p,:]=act[0,p,:]-2.0*vX*act[0,p,:].norm()
    return act
r_aa=model.run_with_hooks(rt,fwd_hooks=[(f"blocks.{L0}.hook_resid_pre",ahook)])[0]
klf=kl_full(rclean,r_fra); kla=kl_full(rclean,r_aa)
print(f"  FRA   : total real-text KL {klf.sum().item():.3f}  (mean/pos {klf.mean().item():.4f})",flush=True)
print(f"  ActAdd: total real-text KL {kla.sum().item():.3f}  (mean/pos {kla.mean().item():.4f})",flush=True)
print(f"  KL at cue positions: FRA {klf[cue_pos].mean().item():.4f}  ActAdd {kla[cue_pos].mean().item():.4f}",flush=True)

out={"part1":{"fra_supp":float(np.mean(fra_supp)),"aa_supp":float(np.mean(aa_supp)),
              "fra_offtarget_kl":float(np.mean(fra_off)),"aa_offtarget_kl":float(np.mean(aa_off)),
              "ratio":float(np.mean(aa_off)/max(np.mean(fra_off),1e-4))},
     "part2":{"cue":cue_str,"fra_total_kl":float(klf.sum().item()),"aa_total_kl":float(kla.sum().item()),
              "fra_cue_kl":float(klf[cue_pos].mean().item()),"aa_cue_kl":float(kla[cue_pos].mean().item())}}
json.dump(out,open(os.path.join(OUT,"j5.json"),"w"),indent=2)
fig,ax=plt.subplots(1,2,figsize=(9,4))
ax[0].bar(["FRA-QK","ActAdd-cue"],[np.mean(fra_off),np.mean(aa_off)],color=['C0','C1'])
ax[0].set_title(f"off-target collateral on induction seq\n(matched suppression ~{np.mean(fra_supp):.0%})"); ax[0].set_ylabel("off-target KL mass (nats)")
ax[1].bar(["FRA-QK","ActAdd-cue"],[klf.sum().item(),kla.sum().item()],color=['C0','C1'])
ax[1].set_title(f"held-out real-text collateral\n(cue='{cue_str.strip()}')"); ax[1].set_ylabel("total KL vs clean (nats)")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"j5_collateral.png"),dpi=110)
print("\nDONE j5",flush=True)
