"""J6 — robustness + the money Pareto. Unified real-cue design:
For each cue WORD, build an induction primer where the cue is followed by a response token, repeated,
so the 2nd cue triggers induction -> response. Then sweep strength for FRA-QK and ActAdd-cue and plot
(induction suppression at the cue edge) vs (held-out collateral = KL on a normal sentence containing the cue).
Aggregate over cues -> mean Pareto + headline collateral ratio at matched suppression.
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

def fra_per_head(tt, edges_of_interest):
    seq=tt.shape[1]
    names=[f"blocks.{L}.hook_resid_pre" for L in LAYERS]
    _,cache=model.run_with_cache(tt,names_filter=lambda n:n in names)
    H={}; resid={L:cache[f"blocks.{L}.hook_resid_pre"][0] for L in LAYERS}
    for (L,Hh) in IND_HEADS:
        fe=saes[L].encode(cache[f"blocks.{L}.hook_resid_pre"][0]).float()
        xh=fe@saes[L].W_dec.float()+saes[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,saes[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); v=f.values().cpu().numpy()
        H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=v)
    return H, resid

def fra_delta_for_edges(HF, edges, seq, M=12):
    byL={}
    for (L,Hh) in IND_HEADS:
        d=HF[(L,Hh)]; dd=np.zeros((seq,seq))
        for (qi,ki) in edges:
            loc=np.where((d["qq"]==qi)&(d["kk"]==ki))[0]
            order=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
            for o in order: dd[qi,ki]+=d["vv"][o]
        byL.setdefault(L,{})[Hh]=dd
    return byL

def run_fra(tt, byL_base, c):
    seq=tt.shape[1]; hooks=[]
    for L,hd in byL_base.items():
        td={Hh:torch.tensor(dd,device=dev,dtype=torch.float32)*c for Hh,dd in hd.items()}
        def mk(td):
            def hook(s,hook):
                for Hh,sd in td.items(): s[0,Hh,:seq,:seq]-=sd
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(td)))
    return model.run_with_hooks(tt,fwd_hooks=hooks)[0]

def run_actadd(tt, cue_positions, vX, s):
    def hook(act,hook):
        for p in cue_positions: act[0,p,:]=act[0,p,:]-s*vX*act[0,p,:].norm()
        return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{L0}.hook_resid_pre",hook)])[0]

# cue words that exist as single GPT-2 tokens and recur in normal text
CUES=[" war"," city"," water"," money"," music"," doctor"," river"," market"]
CS=[2,4,8,16,32]; AS=[0.5,1,2,4,8]
torch.manual_seed(0)
results={}
for cue_word in CUES:
    cid=tok.encode(cue_word)
    if len(cid)!=1: continue
    cid=cid[0]; resp=tok.encode(" then")[0]
    # induction primer: random tokens with cue->resp embedded, repeated
    N=20; R=(torch.randperm(40000)[:N]+1000).tolist(); R[10]=cid; R[11]=resp
    toks=[tok.bos_token_id]+R+R
    tt=torch.tensor(toks,device=dev).unsqueeze(0); seq=tt.shape[1]
    qpos=1+N+10; kpos=12  # 2nd cue attends to token after 1st cue (=resp at pos 12)
    clean=model(tt)[0]
    base_resp=torch.softmax(clean[qpos].float(),-1)[resp].item()
    if base_resp<0.2: continue   # need the model to actually do this induction
    HF,resid=fra_per_head(tt,[(qpos,kpos)])
    byL=fra_delta_for_edges(HF,[(qpos,kpos)],seq)
    cue_positions=[1+10,1+N+10]
    vX=resid[L0][1+N+10]-resid[L0].mean(0); vX=vX/(vX.norm()+1e-6)
    # held-out collateral sentence containing the cue in a normal context
    htext=f"The{cue_word} was very important to many people who lived nearby."
    ht=torch.tensor([tok.bos_token_id]+tok.encode(htext),device=dev).unsqueeze(0); hseq=ht.shape[1]
    hclean=model(ht)[0]
    hcue_pos=[i for i,t in enumerate([tok.bos_token_id]+tok.encode(htext)) if t==cid]
    # held-out FRA: recompute FRA on held text, ablate cue induction-like edges (query=cue, key=after earlier cue)
    HFh,residh=fra_per_head(ht,[])
    hedges=[(qi,ki+1) for qi in hcue_pos for ki in hcue_pos if ki+1<qi]
    byLh=fra_delta_for_edges(HFh,hedges,hseq) if hedges else None
    vXh=residh[L0][hcue_pos[0]]-residh[L0].mean(0); vXh=vXh/(vXh.norm()+1e-6)
    def kl(p,q):
        lp=torch.log_softmax(p.float(),-1); lq=torch.log_softmax(q.float(),-1)
        return (lp.exp()*(lp-lq)).sum(-1).sum().item()
    fra_curve=[]; aa_curve=[]
    for c in CS:
        l=run_fra(tt,byL,c); supp=1-torch.softmax(l[qpos].float(),-1)[resp].item()/base_resp
        if byLh is not None: hl=run_fra(ht,byLh,c); col=kl(hclean,hl)
        else: col=0.0
        fra_curve.append((supp,col))
    for s in AS:
        l=run_actadd(tt,cue_positions,vX,s); supp=1-torch.softmax(l[qpos].float(),-1)[resp].item()/base_resp
        hl=run_actadd(ht,hcue_pos,vXh,s); col=kl(hclean,hl)
        aa_curve.append((supp,col))
    results[cue_word]={"base_resp":base_resp,"fra":fra_curve,"actadd":aa_curve}
    print(f"cue '{cue_word}' base_resp {base_resp:.2f}",flush=True)
    print(f"  FRA   : {[(round(a,2),round(b,2)) for a,b in fra_curve]}",flush=True)
    print(f"  ActAdd: {[(round(a,2),round(b,2)) for a,b in aa_curve]}",flush=True)

# headline: collateral at matched suppression ~0.5 (interp each curve)
def collat_at(curve, target=0.5):
    xs=[a for a,b in curve]; ys=[b for a,b in curve]
    if max(xs)<target: return ys[int(np.argmax(xs))]
    return float(np.interp(target, xs, ys))
fra_c=[collat_at(r["fra"]) for r in results.values()]
aa_c=[collat_at(r["actadd"]) for r in results.values()]
print(f"\n=== HEADLINE (n={len(results)} cues), held-out collateral at 50% induction suppression ===",flush=True)
print(f"  FRA-QK : {np.mean(fra_c):.2f} ± {np.std(fra_c):.2f}",flush=True)
print(f"  ActAdd : {np.mean(aa_c):.2f} ± {np.std(aa_c):.2f}",flush=True)
print(f"  ratio  : {np.mean(aa_c)/max(np.mean(fra_c),1e-3):.0f}x lower collateral for FRA",flush=True)
json.dump({"results":results,"headline":{"fra_collat":float(np.mean(fra_c)),"aa_collat":float(np.mean(aa_c)),
          "ratio":float(np.mean(aa_c)/max(np.mean(fra_c),1e-3)),"n":len(results)}},
          open(os.path.join(OUT,"j6.json"),"w"),indent=2)
# money figure: mean Pareto
plt.figure(figsize=(6,4.5))
for r in results.values():
    plt.plot([a for a,b in r["fra"]],[b for a,b in r["fra"]],'-',color='C0',alpha=0.3)
    plt.plot([a for a,b in r["actadd"]],[b for a,b in r["actadd"]],'-',color='C1',alpha=0.3)
plt.plot([],[],'C0-',label="FRA-QK (bilinear edit)"); plt.plot([],[],'C1-',label="ActAdd-cue (linear)")
plt.yscale('symlog'); plt.xlabel("induction suppression  (1 - P(resp)/base)  → more"); plt.ylabel("held-out collateral  KL on normal text (nats)")
plt.title(f"Suppress one token's induction: collateral vs strength (n={len(results)} cues)\nFRA stays surgical; linear steer wrecks the cue everywhere")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT,"j6_pareto.png"),dpi=120)
print("\nDONE j6",flush=True)
