"""RT_CONTROLS — run the red-team's metric-validity controls properly on the pod (the agents tried
local/Modal). Acronym 'Officer'->O, letter-movers [8.11,9.9,10.10,11.4].
 (1) ON-TARGET removal each method achieves at c=8 (are FRA vs head-ablate 'matched'?).
 (2) FRA c-sweep: on-target removal + collateral at each c -> collateral at MATCHED removal.
 (3) RANDOM-PAIR null: ablate same #pairs but RANDOM (i,j) -> does P(O) drop? (specificity).
 (4) Content-specificity: do the Officer-pairs suppress a DIFFERENT letter (control) at the same edge?
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
rng=np.random.RandomState(0)
model=HookedTransformer.from_pretrained("gpt2",device=dev); model.eval(); tok=model.tokenizer; W_U=model.W_U
HEADS=[(8,11),(9,9),(10,10),(11,4)]; LY=sorted(set(L for L,H in HEADS))
sae={L:(lambda s:(s[0] if isinstance(s,tuple) else s))(SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev)) for L in LY}
def enc(s): return torch.tensor([tok.bos_token_id]+tok.encode(s),device=dev).unsqueeze(0)
def kpos(ids,sub,before):
    w=tok.encode(sub)[0]; c=[i for i,x in enumerate(ids) if x==w and i<before]; return c[-1] if c else None
def Pof(tt,letter,hooks=None):
    lg=(model.run_with_hooks(tt,fwd_hooks=hooks) if hooks else model(tt))[0]; return torch.softmax(lg[-1].float(),-1)[tok.encode(letter)[0]].item()
def fra_edge(tt,L,H):
    HK=f"blocks.{L}.hook_resid_pre"; fe=sae[L].encode(model.run_with_cache(tt,names_filter=lambda n:n==HK)[1][HK][0]).float()
    xh=fe@sae[L].W_dec.float()+sae[L].b_dec.float()
    r=_build_fra_result(model,L,H,fe,sae[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
p1="The Chief Executive Officer (CE"; t1=enc(p1); ids1=[tok.bos_token_id]+tok.encode(p1); Q=t1.shape[1]-1; K=kpos(ids1," Officer",Q)
HF={(L,H):fra_edge(t1,L,H) for L,H in HEADS}
def pairs_top(LH,M=12):
    d=HF[LH]; on=(d["qq"]==Q)&(d["kk"]==K); oi=np.where(on)[0][np.argsort(-np.abs(d["vv"][on]))[:M]]
    return set((int(d["ii"][o]),int(d["jj"][o])) for o in oi)
def pairs_random(LH,M=12):
    d=HF[LH]; on=np.where((d["qq"]==Q)&(d["kk"]==K))[0]
    pick=rng.choice(on,size=min(M,len(on)),replace=False)
    return set((int(d["ii"][o]),int(d["jj"][o])) for o in pick)
def hooks_from(Pdict,c):
    sq=t1.shape[1]; byL={}
    for (L,H) in HEADS:
        d=HF[(L,H)]; Ps=Pdict[(L,H)]; dd=np.zeros((sq,sq))
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
def ha_hooks():
    hk=[]
    for L in LY:
        Hs=[H for LL,H in HEADS if LL==L]
        def mk(Hs):
            def hook(z,hook):
                for H in Hs: z[0,:,H,:]=0.0
                return z
            return hook
        hk.append((f"blocks.{L}.attn.hook_z",mk(Hs)))
    return hk
def cs_hooks(letter,s=6):
    u=W_U[:,tok.encode(letter)[0]].float(); u=u/u.norm()
    def hook(a,hook): a[0]=a[0]-(s*(a[0].float()@u).unsqueeze(-1)*u).to(a.dtype); return a
    return [(f"blocks.{model.cfg.n_layers-1}.hook_resid_post",hook)]
OTHERS=[("The National Basketball Association (NB"," Association","A"),("The Automated Teller Machine (AT"," Machine","M"),("The Gross Domestic Product (GD"," Product","P")]
def collateral(hookfn):
    ch=[]
    for prompt,word,letter in OTHERS:
        t2=enc(prompt); b=Pof(t2,letter); ch.append(abs(Pof(t2,letter,hookfn(t2))-b))
    return float(np.mean(ch))
Ptop={LH:pairs_top(LH) for LH in HEADS}; Prand={LH:pairs_random(LH) for LH in HEADS}
b=Pof(t1,"O")
print(f"base P(O)={b:.3f}",flush=True)
# collateral fn for top-pairs at scale c: need content-addressed on each other probe
def fra_hooks_probe(Pdict,c):
    def f(tt):
        sq=tt.shape[1]; byL={}
        for (L,H) in HEADS:
            d=fra_edge(tt,L,H); Ps=Pdict[(L,H)]; dd=np.zeros((sq,sq))
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
    return f
ha_on=Pof(t1,'O',ha_hooks()); ha_col=collateral(lambda tt: ha_hooks())
for c in [1,2,4,8,16]:
    on=Pof(t1,'O',fra_hooks_probe(Ptop,c)(t1)); col=collateral(fra_hooks_probe(Ptop,c))
    print(f"  c={c:<3} on-target P(O)={on:.3f} (removal {1-on/b:.2f})  collateral={col:.3f}",flush=True)
print(f"  [head-ablate: on-target P(O)={ha_on:.3f} (removal {1-ha_on/b:.2f}) collateral={ha_col:.3f}]",flush=True)
print("\n(3) RANDOM-PAIR null (same #pairs, random (i,j), c=8):",flush=True)
print(f"  on-target P(O) top-pairs {Pof(t1,'O',fra_hooks_probe(Ptop,8)(t1)):.3f} vs RANDOM-pairs {Pof(t1,'O',fra_hooks_probe(Prand,8)(t1)):.3f}  (base {b:.3f})",flush=True)
print("\n(4) content-specificity: do Officer-pairs suppress a DIFFERENT letter at the same edge? (apply Officer-pairs, measure P of a control letter 'X')",flush=True)
print(f"  P(X) base {Pof(t1,'X'):.4f} -> under Officer-FRA {Pof(t1,'X',fra_hooks_probe(Ptop,8)(t1)):.4f}",flush=True)
json.dump({"base":b,"on_fra_c8":Pof(t1,'O',fra_hooks_probe(Ptop,8)(t1)),"on_ha":ha_on,"ha_col":ha_col,
           "sweep":{c:{"on":Pof(t1,'O',fra_hooks_probe(Ptop,c)(t1)),"col":collateral(fra_hooks_probe(Ptop,c))} for c in [1,2,4,8,16]},
           "random_pair_onP":Pof(t1,'O',fra_hooks_probe(Prand,8)(t1))},
          open(os.path.join(OUT,"rt_controls.json"),"w"),indent=2,default=float)
print("\nDONE rt_controls",flush=True)
