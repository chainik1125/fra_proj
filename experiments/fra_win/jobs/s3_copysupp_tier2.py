"""S3 — Tier-2 collateral-advantage A for copy-suppression (gpt2-small L10H7), the screen's top candidate.
Copy-suppression: L10H7 attends END->earlier-X and writes -X, lowering P(X) (negative copy). The
FRA-unique win = SELECTIVELY disable suppression for ONE target token T (let the model copy T again)
while PRESERVING suppression for every other token, content-addressed + transferable.
  FRA           : ablate L10H7's edge feature-pairs keyed to T's key-feature  [selective]
  head-ablation : zero L10H7 entirely                                          [kills suppression for ALL]
Metrics: on-target removal (raise logit T), collateral (suppression change on OTHER tokens),
A = collateral(head-ablation)/collateral(FRA); + separability + transfer to a new context.
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
L,H=10,7; HK=f"blocks.{L}.hook_resid_pre"
sae=SAE.from_pretrained("gpt2-small-res-jb",HK,device=dev); sae=sae[0] if isinstance(sae,tuple) else sae
items=[("The animal was a"," lion"),("The tool was a"," hammer"),("The fruit was a"," lemon"),
       ("The vehicle was a"," truck"),("The drink was a"," soda"),("The bird was a"," robin")]
items=[(p,x) for p,x in items if len(tok.encode(x))==1]
def mk(prefix,X): return f"{prefix}{X}. {prefix}"
def probe(prefix,X):
    ids=[tok.bos_token_id]+tok.encode(mk(prefix,X)); tt=torch.tensor(ids,device=dev).unsqueeze(0)
    xid=tok.encode(X)[0]; Kx=ids.index(xid); Q=len(ids)-1
    return tt,xid,Kx,Q
def logitX(tt,xid,hooks=None):
    lg=(model.run_with_hooks(tt,fwd_hooks=hooks) if hooks else model(tt))[0]
    return lg[-1].float()[xid].item()
def cut_head_hook(Q=None,Kx=None,full=False):
    def hook(s,hook):
        if full: s[0,H,:,:]=-1e4   # zero ALL of L10H7's attention (head ablation, via scores->~uniform? no)
        else: s[0,H,Q,Kx]=-1e4
        return s
    return [(f"blocks.{L}.attn.hook_attn_scores",hook)]
def head_ablate_hook():
    # true head ablation: zero L10H7's value output (z)
    def hook(z,hook): z[0,:,H,:]=0.0; return z
    return [(f"blocks.{L}.attn.hook_z",hook)]
# --- suppression baseline per probe: how much does L10H7 lower logit(X)? (edge-cut raises it) ---
base={}
for prefix,X in items:
    tt,xid,Kx,Q=probe(prefix,X)
    li=logitX(tt,xid); lc=logitX(tt,xid,cut_head_hook(Q,Kx))
    base[X]={"intact":li,"edge_cut":lc,"suppression":lc-li,"tt":tt,"xid":xid,"Kx":Kx,"Q":Q}
    print(f"  suppression[{X.strip()}] = {lc-li:+.2f} (logit {li:.2f}->{lc:.2f} on edge-cut)",flush=True)
# --- FRA on the TARGET edge, select pairs keyed to T's key-feature ---
T=items[0][1]; tgt=base[T]; tt=tgt["tt"]; Q=tgt["Q"]; Kx=tgt["Kx"]; xid=tgt["xid"]
def fra(tt):
    fe=sae.encode(model.run_with_cache(tt,names_filter=lambda n:n==HK)[1][HK][0]).float()
    xh=fe@sae.W_dec.float()+sae.b_dec.float()
    r=_build_fra_result(model,L,H,fe,sae.W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
    return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
d=fra(tt); on=(d["qq"]==Q)&(d["kk"]==Kx)
order=np.argsort(-np.abs(d["vv"][on])); oi=np.where(on)[0][order[:15]]
P=set((int(d["ii"][o]),int(d["jj"][o])) for o in oi)
def delta(d,sq):
    dd=np.zeros((sq,sq))
    for n in range(len(d["vv"])):
        if (int(d["ii"][n]),int(d["jj"][n])) in P: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
    return dd
def fra_logitX(tt,xid,c):
    sq=tt.shape[1]; dd=torch.tensor(delta(fra(tt),sq),device=dev,dtype=torch.float32)*c
    def hook(s,hook): s[0,H,:dd.shape[0],:dd.shape[1]]-=dd.to(s.dtype); return s
    return logitX(tt,xid,[(f"blocks.{L}.attn.hook_attn_scores",hook)])
# on-target: removing T's suppression should raise logit(T) toward its edge-cut value
c=8
print(f"\n--- ON-TARGET ({T.strip()}): intact {tgt['intact']:.2f} | edge-cut(oracle) {tgt['edge_cut']:.2f} ---",flush=True)
fra_on=fra_logitX(tt,xid,c); ha_on=logitX(tt,xid,head_ablate_hook())
print(f"  FRA(c={c}) logit -> {fra_on:.2f}   head-ablate -> {ha_on:.2f}",flush=True)
# collateral: suppression CHANGE on OTHER tokens under each method (content-addressed FRA pairs should not fire)
def supp_change(method):
    ch=[]
    for prefix,X in items[1:]:
        b=base[X]; intact=b["suppression"]
        if method=="fra": newcut=fra_logitX(b["tt"],b["xid"],c)-b["intact"]
        else: newcut=logitX(b["tt"],b["xid"],head_ablate_hook())-b["intact"]
        # suppression after intervention = (logit with method) - intact ; collateral = |change in suppression-removed|
        ch.append(abs(newcut))
    return float(np.mean(ch))
fra_col=supp_change("fra"); ha_col=supp_change("head")
print(f"\n--- COLLATERAL: |logit change| on OTHER tokens (want ~0) ---",flush=True)
print(f"  FRA: {fra_col:.3f}   head-ablate: {ha_col:.3f}   A = {ha_col/max(fra_col,1e-3):.1f}x",flush=True)
# transfer: target token T in a NEW context/position
tprefix="At the zoo we all watched a"; ttt,xidt,Kxt,Qt=probe(tprefix,T)
trans_intact=base_=logitX(ttt,xidt); trans_cut=logitX(ttt,xidt,cut_head_hook(Qt,Kxt)); trans_fra=fra_logitX(ttt,xidt,c)
print(f"\n--- TRANSFER ({T.strip()} new context): intact {trans_intact:.2f} edge-cut(oracle) {trans_cut:.2f} FRA(same pairs) {trans_fra:.2f} ---",flush=True)
out={"suppression":{x:base[x]["suppression"] for x in base},"target":T,
     "on_target":{"intact":tgt["intact"],"edge_cut":tgt["edge_cut"],"fra":fra_on,"head_ablate":ha_on},
     "collateral":{"fra":fra_col,"head_ablate":ha_col,"A":ha_col/max(fra_col,1e-3)},
     "transfer":{"intact":trans_intact,"edge_cut":trans_cut,"fra":trans_fra}}
json.dump(out,open(os.path.join(OUT,"s3.json"),"w"),indent=2,default=float)
print("\nDONE s3",flush=True)
