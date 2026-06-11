"""MAGNITUDE_LAW — quantitative validation of A ≈ reuse(marginal)/reuse(conjunction).
N boxes share the value 'frog'; query the first (red); siblings = the other N-1 frog-boxes.
Theory: marginal=frog-value reuse=N; conjunction(GENERIC box-query x frog-value) reuse=N -> A_generic≈1
(flat in N); conjunction(DIFFERENTIAL red-specific) reuse=1 -> A_diff≈N (grows). Sweep N=2..4.
Sibling collateral = mean over the N-1 sibling frog-queries of |ΔP(frog)|; A = steer_collat/fra_collat.
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
RH=[(15,0),(18,6),(6,3),(21,5),(22,4)]; LAYERS=sorted(set(L for L,H in RH))
SAE={L:GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{L-1}/width_16k/canonical",device=dev,normalize_activations=True) for L in LAYERS}
def enc(L,x):
    f=SAE[L].encode(x.float()).float()
    if SAE[L]._norm_coeff is not None: f=f/SAE[L]._norm_coeff
    return f
cid=tok.encode(" frog",add_special_tokens=False)[0]
COLORS=["red","blue","green","gold","black","white"]; OTHER=["clock","rose","sword","candle"]
def facts(N): return [(COLORS[i],"frog") for i in range(N)]+[(COLORS[N+j],OTHER[j]) for j in range(6-N)]
def build(N,qk): return tok.encode("".join(f" The {k} box holds a {v}." for k,v in facts(N))+f" The {qk} box holds a")
def Pf(t,hooks=None):
    lg=(model.run_with_hooks(t,fwd_hooks=hooks) if hooks else model(t))[0]; return torch.softmax(lg[-1].float(),-1)[cid].item()
def fra_ph(t):
    _,c=model.run_with_cache(t,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS]); H={}
    for (L,Hh) in RH:
        fe=enc(L,c[f"blocks.{L}.hook_resid_pre"][0]); xh=fe@SAE[L].W_dec.float()+SAE[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,SAE[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H
def edgepairs(HF,Q,K,M=30):
    P={}
    for (L,Hh) in RH:
        d=HF[(L,Hh)]; loc=np.where((d["qq"]==Q)&(d["kk"]==K))[0]; loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
        P[(L,Hh)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in loc)
    return P
def fra_hooks(t,P,c=8):
    HF2=fra_ph(t); sq=t.shape[1]; byL={}
    for (L,Hh) in RH:
        d=HF2[(L,Hh)]; Ps=P[(L,Hh)]; dd=np.zeros((sq,sq))
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
def steer_hooks(HF,Q,K,a=1):  # projection-removal of dominant frog-value feature per layer
    hk=[]
    for L in LAYERS:
        d=HF[(L,[H for LL,H in RH if LL==L][0])]; loc=np.where((d["qq"]==Q)&(d["kk"]==K))[0]; kw={}
        for o in loc: kw[int(d["jj"][o])]=kw.get(int(d["jj"][o]),0)+abs(d["vv"][o])
        if not kw: continue
        j=max(kw,key=kw.get); v=SAE[L].W_dec[j].float(); dh=(v/v.norm()).to(dev)
        def mk(dh):
            def hook(resid,hook):
                x=resid[0].float(); proj=(x@dh).unsqueeze(-1)*dh.unsqueeze(0); resid[0]=(x-a*proj).to(resid.dtype); return resid
            return hook
        hk.append((f"blocks.{L}.hook_resid_pre",mk(dh)))
    return hk
rows=[]
for N in [2,3,4]:
    red=torch.tensor(build(N,COLORS[0]),device=dev).unsqueeze(0); ids=build(N,COLORS[0]); Q=red.shape[1]-1
    frogpos=[i for i,x in enumerate(ids) if x==cid]  # N frog positions
    HFr=fra_ph(red); Pgen=edgepairs(HFr,Q,frogpos[0])  # red's edge (generic top pairs)
    # differential: red-edge minus union of sibling-edges (on the same red-query prompt, the sibling frog keys)
    sib_pairs=set()
    for fp in frogpos[1:]:
        for s in edgepairs(HFr,Q,fp).values(): sib_pairs|=s
    Pdiff={lh:(Pgen[lh]-sib_pairs) for lh in Pgen}
    on_gen=Pf(red,fra_hooks(red,Pgen)); on_diff=Pf(red,fra_hooks(red,Pdiff)); base=Pf(red)
    # sibling collateral: query each sibling frog-box, measure |dP(frog)|
    def sibcol(hookmaker):
        ch=[]
        for k in range(1,N):
            tb=torch.tensor(build(N,COLORS[k]),device=dev).unsqueeze(0); b=Pf(tb); ch.append(abs(Pf(tb,hookmaker(tb))-b))
        return float(np.mean(ch))
    col_gen=sibcol(lambda tb: fra_hooks(tb,Pgen)); col_diff=sibcol(lambda tb: fra_hooks(tb,Pdiff))
    col_steer=sibcol(lambda tb: steer_hooks(fra_ph(tb),tb.shape[1]-1,[i for i,x in enumerate(build(N,'q')) if x==cid][0] if False else frogpos[0],1))
    # steer collateral simpler: re-derive per sibling prompt
    def steer_sib():
        ch=[]
        for k in range(1,N):
            tb=torch.tensor(build(N,COLORS[k]),device=dev).unsqueeze(0); b=Pf(tb); HFb=fra_ph(tb)
            fpb=[i for i,x in enumerate(build(N,COLORS[k])) if x==cid][0]
            ch.append(abs(Pf(tb,steer_hooks(HFb,tb.shape[1]-1,fpb,1))-b))
        return float(np.mean(ch))
    col_steer=steer_sib()
    Ag=col_steer/max(col_gen,1e-3); Ad=col_steer/max(col_diff,1e-3)
    rows.append(dict(N=N,base=base,on_gen=on_gen,on_diff=on_diff,col_gen=col_gen,col_diff=col_diff,col_steer=col_steer,A_generic=Ag,A_diff=Ad))
    print(f"N={N}: base {base:.3f} | on-target gen {on_gen:.3f} diff {on_diff:.3f} | sibling-collat gen {col_gen:.3f} diff {col_diff:.3f} steer {col_steer:.3f} | A_generic={Ag:.1f} A_diff={Ad:.1f}",flush=True)
print("\nLAW PREDICTION: A_generic ~ flat (conjunction recurs); A_diff ~ grows with N (conjunction unique).",flush=True)
json.dump(rows,open(os.path.join(OUT,"magnitude_law.json"),"w"),indent=2,default=float)
print("DONE magnitude_law",flush=True)
