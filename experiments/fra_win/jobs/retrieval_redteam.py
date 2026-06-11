"""RETRIEVAL_REDTEAM — apply the acronym red-team gauntlet to the gemma-2-2b retrieval win (r2, A~16x).
Probe: '...The red box holds a frog. ...The red box holds a' -> frog. FRA cuts the (red-query x frog-value)
edge. The red-team's standard:
 (1) random-pair null: random pairs same count -> does P(frog) drop? (specificity)
 (2) matched-removal sweep: collateral at matched on-target removal, not default c.
 (3) STRONGEST FAIR BASELINE = content-gated linear steer (remove the frog-VALUE feature, content-addressed,
     transfer-capable) vs FRA, on: on-target, transfer (new context), and the DISCRIMINATOR = KL-collateral
     on legit 'frog' sentences (frog present, box-query absent). FRA wins iff it preserves legit frog where
     the steer corrupts it (the part attributable to the bilinear QK pair, not content-addressing alone).
 (4) position-patch fairness on transfer.
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
rng=np.random.RandomState(0)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
RH=[(15,0),(18,6),(6,3),(21,5),(22,4)]; LAYERS=sorted(set(L for L,H in RH))
SAE={L:GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{L-1}/width_16k/canonical",device=dev,normalize_activations=True) for L in LAYERS}
def enc(L,x):
    f=SAE[L].encode(x.float()).float()
    if SAE[L]._norm_coeff is not None: f=f/SAE[L]._norm_coeff
    return f
facts=[("red","frog"),("blue","lamp"),("green","clock"),("gold","rose"),("black","sword"),("white","candle")]
QK="red"; CV="frog"; cid=tok.encode(" "+CV,add_special_tokens=False)[0]
def build(fs,qk,filler=""): return tok.encode(filler+"".join(f" The {k} box holds a {v}." for k,v in fs)+f" The {qk} box holds a")
ids=build(facts,QK); tt=torch.tensor(ids,device=dev).unsqueeze(0); seq=tt.shape[1]; cpos=ids.index(cid)
def Pf(t,target,hooks=None):
    lg=(model.run_with_hooks(t,fwd_hooks=hooks) if hooks else model(t))[0]; return torch.softmax(lg[-1].float(),-1)[target].item()
def logits_last(t,hooks=None):
    lg=(model.run_with_hooks(t,fwd_hooks=hooks) if hooks else model(t))[0]; return torch.log_softmax(lg[-1].float(),-1)
def KL(t,hooks): p=logits_last(t); q=logits_last(t,hooks); return (p.exp()*(p-q)).sum().item()
def fra_ph(t):
    _,c=model.run_with_cache(t,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS]); H={}
    for (L,Hh) in RH:
        fe=enc(L,c[f"blocks.{L}.hook_resid_pre"][0]); xh=fe@SAE[L].W_dec.float()+SAE[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,SAE[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H
def sel_pairs(HF,q,k,M=20,random=False):
    P={}
    for (L,Hh) in RH:
        d=HF[(L,Hh)]; loc=np.where((d["qq"]==q)&(d["kk"]==k))[0]
        if len(loc)==0: P[(L,Hh)]=set(); continue
        if random: pick=rng.choice(loc,size=min(M,len(loc)),replace=False)
        else: pick=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
        P[(L,Hh)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in pick)
    return P
def keyfeats(HF,q,k,M=20):
    kf={}
    for (L,Hh) in RH:
        d=HF[(L,Hh)]; loc=np.where((d["qq"]==q)&(d["kk"]==k))[0]; loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
        for o in loc: kf.setdefault(L,{}); kf[L][int(d["jj"][o])]=kf[L].get(int(d["jj"][o]),0)+abs(d["vv"][o])
    return kf
def fra_hooks(t,P,c):
    HF=fra_ph(t); sq=t.shape[1]; byL={}
    for (L,Hh) in RH:
        d=HF[(L,Hh)]; Ps=P[(L,Hh)]; dd=np.zeros((sq,sq))
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        byL.setdefault(L,{})[Hh]=torch.tensor(dd,device=dev,dtype=torch.float32)*c
    hk=[]
    for L,hd in byL.items():
        def mk(hd):
            def hook(s,hook):
                for Hh,dd in hd.items(): s[0,Hh,:dd.shape[0],:dd.shape[1]]-=dd[:s.shape[2],:s.shape[3]].to(s.dtype)
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores",mk(hd)))
    return hk
def steer_hooks(t,kf,a,topf=3):
    hk=[]
    for L in LAYERS:
        feats=sorted(kf.get(L,{}),key=lambda j:-kf[L][j])[:topf]
        if not feats: continue
        Wd=SAE[L].W_dec.float()
        def mk(feats,Wd,L):
            def hook(resid,hook):
                fe=enc(L,resid[0])
                for j in feats:
                    act=fe[:,j]; resid[0]=resid[0]-(a*act.unsqueeze(-1)*Wd[j].unsqueeze(0)).to(resid.dtype)
                return resid
            return hook
        hk.append((f"blocks.{L}.hook_resid_pre",mk(feats,Wd,L)))
    return hk
def patch_hooks(t,keypos):
    byL={}
    for L,H in RH: byL.setdefault(L,[]).append(H)
    hk=[]
    for L,Hs in byL.items():
        def mk(Hs):
            def hook(s,hook):
                Q=s.shape[2]-1
                if keypos<s.shape[3]:
                    for H in Hs: s[0,H,Q,keypos]=-1e4
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores",mk(Hs)))
    return hk
def ha_hooks():
    hk=[]
    for L in LAYERS:
        Hs=[H for LL,H in RH if LL==L]
        def mk(Hs):
            def hook(z,hook):
                for H in Hs: z[0,:,H,:]=0.0
                return z
            return hook
        hk.append((f"blocks.{L}.attn.hook_z",mk(Hs)))
    return hk
HF=fra_ph(tt); P=sel_pairs(HF,seq-1,cpos); Prnd=sel_pairs(HF,seq-1,cpos,random=True); KF=keyfeats(HF,seq-1,cpos)
base=Pf(tt,cid)
print(f"base P(frog)={base:.3f}",flush=True)
print("\n(1) RANDOM-PAIR null (c=8):",flush=True)
print(f"  selected pairs {Pf(tt,cid,fra_hooks(tt,P,8)):.3f} vs RANDOM pairs {Pf(tt,cid,fra_hooks(tt,Prnd,8)):.3f}  (base {base:.3f})",flush=True)
# legit-frog collateral sentences (frog present, box-query absent)
legits=["A frog sat quietly on the wet log beside the","The children laughed as the green frog leaped across the","At the zoo her favourite animal was the small frog that"]
print("\n(2) MATCHED-REMOVAL sweep (on-target P(frog) | legit-frog KL):",flush=True)
for c in [1,2,4,8,16]:
    on=Pf(tt,cid,fra_hooks(tt,P,c)); klc=np.mean([KL(torch.tensor(tok.encode(s),device=dev).unsqueeze(0),fra_hooks(torch.tensor(tok.encode(s),device=dev).unsqueeze(0),P,c)) for s in legits])
    print(f"  c={c:<3} on-target {on:.3f} (removal {1-on/base:.2f}) | legit-frog KL {klc:.3f}",flush=True)
ha_on=Pf(tt,cid,ha_hooks()); ha_kl=np.mean([KL(torch.tensor(tok.encode(s),device=dev).unsqueeze(0),ha_hooks()) for s in legits])
print(f"  [head-ablate: on-target {ha_on:.3f} (removal {1-ha_on/base:.2f}) legit-frog KL {ha_kl:.3f}]",flush=True)
# (3) content-gated steer: match to FRA c=4 on-target
fra_on4=Pf(tt,cid,fra_hooks(tt,P,4)); best=None
for a in [2,4,8,16,32]:
    on=Pf(tt,cid,steer_hooks(tt,KF,a))
    if best is None or abs(on-fra_on4)<best[1]: best=(a,abs(on-fra_on4),on)
a=best[0]
print(f"\n(3) CONTENT-GATED STEER (remove frog-value feature; a={a} matched to FRA c=4 on-target {fra_on4:.3f} -> steer {best[2]:.3f}):",flush=True)
fra_kl=np.mean([KL(torch.tensor(tok.encode(s),device=dev).unsqueeze(0),fra_hooks(torch.tensor(tok.encode(s),device=dev).unsqueeze(0),P,4)) for s in legits])
steer_kl=np.mean([KL(torch.tensor(tok.encode(s),device=dev).unsqueeze(0),steer_hooks(torch.tensor(tok.encode(s),device=dev).unsqueeze(0),KF,a)) for s in legits])
print(f"  DECISIVE legit-frog KL: FRA {fra_kl:.3f} | content-gated STEER {steer_kl:.3f}  => FRA preserves legit frog {steer_kl/max(fra_kl,1e-3):.0f}x better",flush=True)
# (4) transfer: frog at a new position
ids2=build([("silver","frog"),("amber","torch"),("ivory","drum"),("coral","mask"),("jade","spear"),("ruby","flute")],"silver",filler="Once upon a time in a quiet town by the sea where many tales were told.")
t2=torch.tensor(ids2,device=dev).unsqueeze(0); cpos2=ids2.index(cid); b2=Pf(t2,cid)
print(f"\n(4) TRANSFER (new context, frog at pos {cpos2}, base {b2:.3f}):",flush=True)
print(f"  FRA {Pf(t2,cid,fra_hooks(t2,P,4)):.3f} | content-gated STEER {Pf(t2,cid,steer_hooks(t2,KF,a)):.3f} | position-patch@origpos {Pf(t2,cid,patch_hooks(t2,cpos)):.3f}",flush=True)
json.dump({"base":base,"rand_null":Pf(tt,cid,fra_hooks(tt,Prnd,8)),"sel_null":Pf(tt,cid,fra_hooks(tt,P,8)),
           "ha_on":ha_on,"ha_kl":float(ha_kl),"fra_on4":fra_on4,"steer_a":a,"steer_on":best[2],
           "fra_kl":float(fra_kl),"steer_kl":float(steer_kl),"transfer_base":b2,
           "transfer_fra":Pf(t2,cid,fra_hooks(t2,P,4)),"transfer_steer":Pf(t2,cid,steer_hooks(t2,KF,a)),"transfer_patch":Pf(t2,cid,patch_hooks(t2,cpos))},
          open(os.path.join(OUT,"retrieval_redteam.json"),"w"),indent=2,default=float)
print("\nDONE retrieval_redteam",flush=True)
