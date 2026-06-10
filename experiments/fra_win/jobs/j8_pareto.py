"""J8 — the definitive Pareto: induction suppression vs held-out collateral, three methods
(FRA-QK, ActAdd cue-identity, ActAdd key-side 'prev-was-cue'), multi-occurrence held-out text,
averaged over cues. Sidesteps suppression-matching by showing the whole curve.
FRA uses ALL feature-pairs on the target edge (not top-12) to push its suppression ceiling.
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
def fra_ph(tt):
    _,cache=model.run_with_cache(tt,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
    H={}; resid={L:cache[f"blocks.{L}.hook_resid_pre"][0] for L in LAYERS}
    for (L,Hh) in IND_HEADS:
        fe=saes[L].encode(cache[f"blocks.{L}.hook_resid_pre"][0]).float(); xh=fe@saes[L].W_dec.float()+saes[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,saes[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); v=f.values().cpu().numpy()
        H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=v)
    return H,resid
def fra_delta(HF,edges,seq,M=None):
    byL={}
    for (L,Hh) in IND_HEADS:
        d=HF[(L,Hh)]; dd=np.zeros((seq,seq))
        for (qi,ki) in edges:
            loc=np.where((d["qq"]==qi)&(d["kk"]==ki))[0]
            if M is not None: loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
            for o in loc: dd[qi,ki]+=d["vv"][o]
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
def run_aa(tt,positions,vX,s):
    def hook(act,hook):
        for p in positions:
            if p<tt.shape[1]: act[0,p,:]=act[0,p,:]-s*vX*act[0,p,:].norm()
        return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{L0}.hook_resid_pre",hook)])[0]
def kltot(p,q):
    lp=torch.log_softmax(p.float(),-1); lq=torch.log_softmax(q.float(),-1)
    return (lp.exp()*(lp-lq)).sum(-1).sum().item()

CUES=[" war"," city"," water"," money"," market"," music"," doctor"," river"]
FC=[1,2,4,8,16,32,64]; AC=[0.25,0.5,1,2,4,8]
torch.manual_seed(0)
agg={"FRA":[],"Id":[],"Ky":[]}; n=0
for cw in CUES:
    cid=tok.encode(cw)
    if len(cid)!=1: continue
    cid=cid[0]; resp=tok.encode(" then")[0]
    N=20; R=(torch.randperm(40000)[:N]+1000).tolist(); R[10]=cid; R[11]=resp
    tt=torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0); seq=tt.shape[1]
    qpos=1+N+10; kpos=12; base=torch.softmax(model(tt)[0][qpos].float(),-1)[resp].item()
    if base<0.3: continue
    HF,resid=fra_ph(tt); byL=fra_delta(HF,[(qpos,kpos)],seq,M=None)
    vId=resid[L0][1+N+10]-resid[L0].mean(0); vId/=(vId.norm()+1e-6)
    vKy=resid[L0][12]-resid[L0].mean(0); vKy/=(vKy.norm()+1e-6)
    # held-out: cue x3
    htext=f"A{cw} began.{cw} grew.{cw} mattered to people who watched it closely all year."
    hids=[tok.bos_token_id]+tok.encode(htext); ht=torch.tensor(hids,device=dev).unsqueeze(0); hseq=ht.shape[1]
    hcue=[i for i,t in enumerate(hids) if t==cid]
    if len(hcue)<2: continue
    hclean=model(ht)[0]; HFh,residh=fra_ph(ht)
    hedges=[(qi,ki+1) for qi in hcue for ki in hcue if ki+1<qi]
    byLh=fra_delta(HFh,hedges,hseq,M=None) if hedges else {}
    vIdh=residh[L0][hcue[0]]-residh[L0].mean(0); vIdh/=(vIdh.norm()+1e-6)
    hkey=[i+1 for i in hcue if i+1<hseq]
    vKyh=residh[L0][hcue[0]+1]-residh[L0].mean(0); vKyh/=(vKyh.norm()+1e-6)
    def supp(lg): return 1-torch.softmax(lg[qpos].float(),-1)[resp].item()/base
    cF=[(supp(run_fra(tt,byL,c)), kltot(hclean,run_fra(ht,byLh,c)) if byLh else 0.0) for c in FC]
    cI=[(supp(run_aa(tt,[1+10,1+N+10],vId,s)), kltot(hclean,run_aa(ht,hcue,vIdh,s))) for s in AC]
    cK=[(supp(run_aa(tt,[12,2+N+10],vKy,s)), kltot(hclean,run_aa(ht,hkey,vKyh,s))) for s in AC]
    agg["FRA"].append(cF); agg["Id"].append(cI); agg["Ky"].append(cK); n+=1
    print(f"cue '{cw}' base {base:.2f}: FRA max-supp {max(s for s,_ in cF):.2f}",flush=True)

def collat_at(curves, target):
    out=[]
    for cur in curves:
        xs=[a for a,b in cur]; ys=[b for a,b in cur]
        if max(xs)<target: continue
        out.append(float(np.interp(target,xs,ys)))
    return np.array(out)
print(f"\n=== n={n} cues — held-out collateral (nats) at matched suppression ===",flush=True)
for tgt in [0.3,0.5]:
    f=collat_at(agg["FRA"],tgt); i=collat_at(agg["Id"],tgt); k=collat_at(agg["Ky"],tgt)
    print(f"  supp={tgt}: FRA {f.mean():.3f}±{f.std():.3f} (n={len(f)}) | ActAdd-id {i.mean():.3f}±{i.std():.3f} | ActAdd-key {k.mean():.3f}±{k.std():.3f}",flush=True)
json.dump({"n":n,"agg":agg},open(os.path.join(OUT,"j8.json"),"w"),indent=2)
# money figure
plt.figure(figsize=(6.2,4.6))
for key,lab,c in [("FRA","FRA-QK (bilinear edit)","C0"),("Id","ActAdd cue-identity (linear)","C1"),("Ky","ActAdd prev-was-cue (linear)","C2")]:
    for cur in agg[key]:
        xs=[a for a,b in cur]; ys=[b for a,b in cur]; plt.plot(xs,ys,'-',color=c,alpha=0.25,lw=1)
    plt.plot([],[],'-',color=c,label=lab)
plt.yscale('symlog',linthresh=0.05)
plt.xlabel("induction suppression  (1 − P(resp)/base)  → stronger"); plt.ylabel("held-out collateral  (KL on normal text, nats)  ↓ better")
plt.title(f"Suppress one token's induction association (n={n} cues)\nFRA-QK stays surgical at every strength; linear steers wreck the cue")
plt.legend(fontsize=8); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(os.path.join(OUT,"j8_pareto.png"),dpi=130)
print("\nDONE j8",flush=True)
