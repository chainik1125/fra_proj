"""DECISIVE_STEER — the red-team's #1 follow-up: does FRA-QK beat a CONTENT-GATED LINEAR STEER?
A position-patch ties FRA on the cross-acronym metric (fires on no other prompt). The strongest FAIR
selective baseline is a content-gated steer: remove the dominant Officer-KEY feature (content-addressed,
transfer-capable). FRA wins ONLY if it beats this steer on a context where 'Officer' is used LEGITIMATELY
(steer corrupts the Officer token everywhere it fires; FRA fires only on the acronym-query x Officer-key
PAIR, so it should preserve legit Officer). Compare FRA vs steer vs position-patch on:
  (a) on-target acronym suppress (match removal),  (b) transfer (Officer at a new position),
  (c) COLLATERAL: KL on a legit 'Officer' sentence (the discriminator).
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gpt2",device=dev); model.eval(); tok=model.tokenizer
HEADS=[(8,11),(9,9),(10,10),(11,4)]; LY=sorted(set(L for L,H in HEADS))
sae={L:(lambda s:(s[0] if isinstance(s,tuple) else s))(SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev)) for L in LY}
def enc(s): return torch.tensor([tok.bos_token_id]+tok.encode(s),device=dev).unsqueeze(0)
def kpos(ids,sub,before):
    w=tok.encode(sub)[0]; c=[i for i,x in enumerate(ids) if x==w and i<before]; return c[-1] if c else None
def Pof(tt,letter,hooks=None):
    lg=(model.run_with_hooks(tt,fwd_hooks=hooks) if hooks else model(tt))[0]; return torch.softmax(lg[-1].float(),-1)[tok.encode(letter)[0]].item()
def distrib(tt,hooks=None):
    lg=(model.run_with_hooks(tt,fwd_hooks=hooks) if hooks else model(tt))[0]; return torch.log_softmax(lg[-1].float(),-1)
def kl(tt,hooks): p=distrib(tt); q=distrib(tt,hooks); return (p.exp()*(p-q)).sum().item()
def fra_edge(tt,L,H):
    HK=f"blocks.{L}.hook_resid_pre"; fe=sae[L].encode(model.run_with_cache(tt,names_filter=lambda n:n==HK)[1][HK][0]).float()
    xh=fe@sae[L].W_dec.float()+sae[L].b_dec.float()
    r=_build_fra_result(model,L,H,fe,sae[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
p1="The Chief Executive Officer (CE"; t1=enc(p1); ids1=[tok.bos_token_id]+tok.encode(p1); Q=t1.shape[1]-1; K=kpos(ids1," Officer",Q)
HF={(L,H):fra_edge(t1,L,H) for L,H in HEADS}
# FRA pairs + dominant KEY feature per layer (for the steer)
Pset={}; keyfeats={}
for (L,H) in HEADS:
    d=HF[(L,H)]; on=(d["qq"]==Q)&(d["kk"]==K); oi=np.where(on)[0][np.argsort(-np.abs(d["vv"][on]))[:12]]
    Pset[(L,H)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in oi)
    kw={}
    for o in oi: kw[int(d["jj"][o])]=kw.get(int(d["jj"][o]),0)+abs(d["vv"][o])
    keyfeats.setdefault(L,{});
    for j,w in kw.items(): keyfeats[L][j]=keyfeats[L].get(j,0)+w
def fra_hooks(tt,c=4):
    sq=tt.shape[1]; byL={}
    for (L,H) in HEADS:
        d=fra_edge(tt,L,H); Ps=Pset[(L,H)]; dd=np.zeros((sq,sq))
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        byL.setdefault(L,{})[H]=torch.tensor(dd,device=dev,dtype=torch.float32)*c
    hk=[]
    for L,hd in byL.items():
        def mk(hd):
            def hook(s,hook):
                for H,dd in hd.items(): s[0,H,:dd.shape[0],:dd.shape[1]]-=dd.to(s.dtype)
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores",mk(hd)))
    return hk
# CONTENT-GATED STEER: at each letter-mover layer, remove the dominant Officer-key feature(s) wherever they fire
def steer_hooks(tt,a=1.0,topf=3):
    hk=[]
    for L in LY:
        feats=sorted(keyfeats.get(L,{}),key=lambda j:-keyfeats[L][j])[:topf]
        if not feats: continue
        Wd=sae[L].W_dec.float()
        def mk(feats,Wd):
            def hook(resid,hook):
                fe=sae[L].encode(resid[0].float())  # [seq, nfeat]
                for j in feats:
                    act=fe[:,j]  # content-gated: only where it fires
                    resid[0]=resid[0]-(a*act.unsqueeze(-1)*Wd[j].unsqueeze(0)).to(resid.dtype)
                return resid
            return hook
        hk.append((f"blocks.{L}.hook_resid_pre",mk(feats,Wd)))
    return hk
def patch_hooks(tt,keypos):
    byL={}
    for L,H in HEADS: byL.setdefault(L,[]).append(H)
    hk=[]
    for L,Hs in byL.items():
        def mk(Hs):
            def hook(s,hook):
                Qx=s.shape[2]-1
                if keypos is not None and keypos<s.shape[3]:
                    for H in Hs: s[0,H,Qx,keypos]=-1e4
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores",mk(Hs)))
    return hk
b1=Pof(t1,"O")
# match steer strength to FRA on-target removal (FRA c=4 ~0.023)
fra_on=Pof(t1,"O",fra_hooks(t1,4))
best_a=None
for a in [0.5,1.0,2.0,4.0,8.0]:
    on=Pof(t1,"O",steer_hooks(t1,a))
    if best_a is None or abs(on-fra_on)<best_a[1]: best_a=(a,abs(on-fra_on),on)
a=best_a[0]; steer_on=best_a[2]
print(f"base P(O)={b1:.3f}; FRA(c=4) on-target {fra_on:.3f}; STEER(a={a}) on-target {steer_on:.3f} (matched)",flush=True)
# transfer
pB="the board has recently appointed a brand new Chief Executive Officer (CE"; tB=enc(pB); idsB=[tok.bos_token_id]+tok.encode(pB); KB=kpos(idsB," Officer",tB.shape[1]-1)
bB=Pof(tB,"O")
print(f"\nTRANSFER (probe-B, Officer at new pos {KB}, base {bB:.3f}):",flush=True)
print(f"  FRA {Pof(tB,'O',fra_hooks(tB,4)):.3f} | content-gated STEER {Pof(tB,'O',steer_hooks(tB,a)):.3f} | position-patch@origpos {Pof(tB,'O',patch_hooks(tB,K)):.3f}",flush=True)
# DECISIVE COLLATERAL: legit 'Officer' sentence (acronym-query ABSENT) -- steer corrupts Officer, FRA should not fire
legits=["The Officer in charge signed the report and then","A police Officer stopped the car on the highway near","The Chief Executive Officer of the firm announced that the"]
print(f"\nDECISIVE COLLATERAL: KL on legit 'Officer' contexts (acronym-query absent; want LOW):",flush=True)
fra_kl=np.mean([kl(enc(s),fra_hooks(enc(s),4)) for s in legits]); steer_kl=np.mean([kl(enc(s),steer_hooks(enc(s),a)) for s in legits])
print(f"  FRA {fra_kl:.3f} nats | content-gated STEER {steer_kl:.3f} nats  => FRA preserves legit Officer {steer_kl/max(fra_kl,1e-3):.0f}x better" ,flush=True)
json.dump({"base":b1,"fra_on":fra_on,"steer_on":steer_on,"steer_a":a,"transfer_base":bB,
           "transfer_fra":Pof(tB,'O',fra_hooks(tB,4)),"transfer_steer":Pof(tB,'O',steer_hooks(tB,a)),"transfer_patch":Pof(tB,'O',patch_hooks(tB,K)),
           "collat_fra_kl":float(fra_kl),"collat_steer_kl":float(steer_kl)},
          open(os.path.join(OUT,"decisive_steer.json"),"w"),indent=2,default=float)
print("\nDONE decisive_steer",flush=True)
