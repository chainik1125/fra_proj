"""J3 — overcome induction redundancy: ablate a target token's FRA feature-pairs across ALL
induction heads, with over-drive scale c (FRA selects WHAT; c controls HOW HARD).
Question: does this selectively suppress copying of the TARGET token (vs collateral on others),
and how does it compare to the non-selective baseline of ablating all induction heads entirely?
"""
import os, sys, json
sys.path.insert(0, "/workspace/code")
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result

OUT = os.environ.get("OUTDIR", "."); dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2", device=dev); model.eval()
IND_HEADS = [(5,5),(6,9),(5,1),(7,10),(7,2)]   # top induction heads from J1
LAYERS = sorted(set(L for L,H in IND_HEADS))

torch.manual_seed(0); N = 24
R = (torch.randperm(40000)[:N] + 1000).tolist()
toks = [model.tokenizer.bos_token_id] + R + R
tt = torch.tensor(toks, device=dev).unsqueeze(0); seq = tt.shape[1]
edges = [(1+N+t, t+2) for t in range(N-1)]
ind_targets = {1+N+t: R[t+1] for t in range(N-1)}

names = [f"blocks.{L}.attn.hook_attn_scores" for L in LAYERS] + [f"blocks.{L}.hook_resid_pre" for L in LAYERS]
logits, cache = model.run_with_cache(tt, names_filter=lambda n: n in names)
def copyprob_vec(lg):
    p = torch.softmax(lg[0].float(), -1)
    return np.array([p[q, ind_targets[q]].item() for q,k in edges])
base = copyprob_vec(logits); print("baseline mean copyprob", round(base.mean(),3), flush=True)

# FRA per induction head -> per-head sparse tensor + index arrays
saes = {L: (SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{L}.hook_resid_pre", device=dev)) for L in LAYERS}
saes = {L:(s[0] if isinstance(s,tuple) else s) for L,s in saes.items()}
HFRA = {}
for (L,H) in IND_HEADS:
    sae = saes[L]; act = cache[f"blocks.{L}.hook_resid_pre"][0]
    feats = sae.encode(act).float(); x_hat = feats @ sae.W_dec.float() + sae.b_dec.float()
    r = _build_fra_result(model, L, H, feats, sae.W_dec.float(), dev, top_k=None,
                          rms_activations=x_hat, dec_norms=None, chunk_size=16, verbose=False)
    f = r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); v=f.values().cpu().numpy()
    HFRA[(L,H)] = dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=v)
    print(f"FRA L{L}H{H} nnz {len(v)}", flush=True)
SC = {(L,H): cache[f"blocks.{L}.attn.hook_attn_scores"][0,H].float().cpu().numpy() for (L,H) in IND_HEADS}

def edge_pairs(LH, e, m):
    d=HFRA[LH]; loc=np.where((d["qq"]==e[0])&(d["kk"]==e[1]))[0]
    order=loc[np.argsort(-np.abs(d["vv"][loc]))[:m]]
    return set((int(d["ii"][o]),int(d["jj"][o])) for o in order)

def pairs_delta(LH, pset):
    d=HFRA[LH]; out=np.zeros((seq,seq))
    sel=[(int(d["ii"][n]),int(d["jj"][n])) in pset for n in range(len(d["vv"]))]
    for n in np.where(sel)[0]: out[d["qq"][n], d["kk"][n]] += d["vv"][n]
    return out

def run_patched(deltas, scale):  # deltas: {(L,H): [seq,seq]}
    by_layer={}
    for (L,H),dd in deltas.items():
        by_layer.setdefault(L,{})[H]=torch.tensor(dd,device=dev,dtype=torch.float32)*scale
    hooks=[]
    for L,hd in by_layer.items():
        def mk(hd):
            def hook(scores,hook):
                for H,sd in hd.items(): scores[0,H,:seq,:seq]-=sd
                return scores
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk(hd)))
    return copyprob_vec(model.run_with_hooks(tt, fwd_hooks=hooks))

# targets: edges with high baseline copyprob (room to drop)
cand_t = [t for t in range(N-1) if base[t] > 0.8]
targets = cand_t[:6]; print("targets (edge idx):", targets, flush=True)
M = 12
cs = [1,2,4,8,16,32]
# FRA selective: for each target, ablate its pairs across all heads, sweep c
fra_target = {c: [] for c in cs}; fra_collat = {c: []for c in cs}
for t in targets:
    e = edges[t]
    deltas = {LH: pairs_delta(LH, edge_pairs(LH, e, M)) for LH in IND_HEADS}
    for c in cs:
        v = run_patched(deltas, c)
        fra_target[c].append(v[t])
        fra_collat[c].append(np.mean([v[s] for s in range(N-1) if s!=t]))
print("\nFRA selective ablation (target token, across all induction heads):", flush=True)
for c in cs:
    tt_ = np.mean(fra_target[c]); cc_ = np.mean(fra_collat[c])
    print(f"  c={c:<3} target copyprob {tt_:.3f} (Δ {tt_-np.mean([base[t] for t in targets]):+.3f}) | "
          f"collateral copyprob {cc_:.3f} (Δ {cc_-base.mean():+.3f})", flush=True)

# baseline: ablate ALL induction heads entirely (mean-ablate z) -> kills all copying
def killall():
    hooks=[]
    for L in LAYERS:
        Hs=[H for (LL,H) in IND_HEADS if LL==L]
        def mk(Hs):
            def hook(z,hook):
                for H in Hs: z[0,:,H,:]=z[0,:,H,:].mean(0,keepdim=True)
                return z
            return hook
        hooks.append((f"blocks.{L}.attn.hook_z", mk(Hs)))
    return copyprob_vec(model.run_with_hooks(tt, fwd_hooks=hooks))
ka = killall()
print(f"\nBASELINE kill ALL induction heads: target {np.mean([ka[t] for t in targets]):.3f} "
      f"collateral {np.mean([ka[s] for s in range(N-1) if s not in targets]):.3f} (non-selective: kills all)", flush=True)

base_t = float(np.mean([base[t] for t in targets]))
out={"base_target":base_t,"base_all":float(base.mean()),
     "fra":{str(c):{"target":float(np.mean(fra_target[c])),"collateral":float(np.mean(fra_collat[c]))} for c in cs},
     "killall":{"target":float(np.mean([ka[t] for t in targets])),"collateral":float(np.mean([ka[s] for s in range(N-1) if s not in targets]))}}
json.dump(out, open(os.path.join(OUT,"j3.json"),"w"), indent=2)

# Pareto: target suppression (x: 1-target/base) vs collateral (|Δ| on others)
plt.figure(figsize=(5.5,4))
xs=[1-np.mean(fra_target[c])/base_t for c in cs]; ys=[abs(np.mean(fra_collat[c])-base.mean()) for c in cs]
plt.plot(xs,ys,'o-',label="FRA selective (sweep c)")
for c,x,y in zip(cs,xs,ys): plt.annotate(f"c{c}",(x,y),fontsize=7)
plt.scatter([1-np.mean([ka[t] for t in targets])/base_t],[abs(np.mean([ka[s] for s in range(N-1) if s not in targets])-base.mean())],
            c='r',marker='x',s=80,label="kill all heads")
plt.xlabel("target-token copy suppression  (1 - cp/base)"); plt.ylabel("collateral |Δ copyprob| on other tokens")
plt.title("Selective induction suppression: FRA vs head-ablation"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT,"j3_pareto.png"),dpi=110)
print("\nDONE j3", flush=True)
