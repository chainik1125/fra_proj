"""J9 — settle the red-team's main objection. Two changes that could overturn the win:
(1) CONTENT-ADDRESSED FRA with NO hedges whitelist: learn the cue's induction feature-pairs P from a
    primer, then on held-out text subtract those pairs' contribution WHEREVER they fire across the full
    [q,k] score matrix (not at hand-picked edges). If P only fires in induction-like contexts, collateral
    stays low *as a discovered property*; if the features are polysemantic and fire elsewhere, it rises.
(2) INDUCTION-GATED ActAdd (the fair strong linear baseline): subtract the cue direction only at cue
    positions that are induction queries (cue preceded by an earlier cue) — a probe-gated linear steer.
Also report FRA at FAITHFUL c (~exact reconstruction) vs over-driven.
Compare held-out collateral at matched primer-suppression. Cues x4.
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
IND=[(5,5),(6,9),(5,1),(7,10),(7,2)]; LAYERS=sorted(set(L for L,H in IND)); L0=min(LAYERS); tok=model.tokenizer
saes={L:(SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev)) for L in LAYERS}
saes={L:(s[0] if isinstance(s,tuple) else s) for L,s in saes.items()}
def fra_ph(tt):
    _,c=model.run_with_cache(tt,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
    H={}; resid={L:c[f"blocks.{L}.hook_resid_pre"][0] for L in LAYERS}
    for (L,Hh) in IND:
        fe=saes[L].encode(c[f"blocks.{L}.hook_resid_pre"][0]).float(); xh=fe@saes[L].W_dec.float()+saes[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,saes[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
        H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H,resid
def patch(tt,byL,c):
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
def aa(tt,positions,vX,s):
    def hook(act,hook):
        for p in positions:
            if p<tt.shape[1]: act[0,p,:]-=s*vX*act[0,p,:].norm()
        return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{L0}.hook_resid_pre",hook)])[0]
def kltot(p,q):
    lp=torch.log_softmax(p.float(),-1); lq=torch.log_softmax(q.float(),-1)
    return (lp.exp()*(lp-lq)).sum(-1).sum().item()

# select cue induction pairs P from primer (top-M per head)
def primer_pairs(HF,edge,M=12):
    P={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; loc=np.where((d["qq"]==edge[0])&(d["kk"]==edge[1]))[0]
        loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
        P[(L,Hh)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in loc)
    return P
# delta restricted to specific edges (old hedges way)
def delta_edges(HF,P,edges,seq):
    byL={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; dd=np.zeros((seq,seq))
        for (qi,ki) in edges:
            loc=np.where((d["qq"]==qi)&(d["kk"]==ki))[0]
            for o in loc:
                if (int(d["ii"][o]),int(d["jj"][o])) in P[(L,Hh)]: dd[qi,ki]+=d["vv"][o]
        byL.setdefault(L,{})[Hh]=dd
    return byL
# delta content-addressed: pairs P fire wherever they appear across ALL (q,k)  (the honest version)
def delta_content(HF,P,seq):
    byL={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; dd=np.zeros((seq,seq)); Ps=P[(L,Hh)]
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        byL.setdefault(L,{})[Hh]=dd
    return byL

CUES=[" war"," city"," money"," market"]; FC=[1,2,4,8,16,32]; AC=[0.25,0.5,1,2,4,8]
torch.manual_seed(0); rows=[]
for cw in CUES:
    cid=tok.encode(cw)
    if len(cid)!=1: continue
    cid=cid[0]; resp=tok.encode(" then")[0]
    N=20; R=(torch.randperm(40000)[:N]+1000).tolist(); R[10]=cid; R[11]=resp
    tt=torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0); seq=tt.shape[1]
    qpos=1+N+10; kpos=12; base=torch.softmax(model(tt)[0][qpos].float(),-1)[resp].item()
    if base<0.3: continue
    HF,resid=fra_ph(tt); P=primer_pairs(HF,(qpos,kpos))
    byL_supp=delta_edges(HF,P,[(qpos,kpos)],seq)   # for suppression on primer
    def supp(c): return 1-torch.softmax(patch(tt,byL_supp,c)[qpos].float(),-1)[resp].item()/base
    # faithful c: the scale at which the FRA delta ~ the real edge score
    real_edge=model.run_with_cache(tt,names_filter=lambda n:n==f"blocks.5.attn.hook_attn_scores")[1]["blocks.5.attn.hook_attn_scores"][0,5,qpos,kpos].item()
    fra_edge=byL_supp[5][5][qpos,kpos]
    cfaith=abs(real_edge/ (fra_edge+1e-9))
    # held-out
    htext=f"A{cw} began.{cw} grew.{cw} mattered to people who watched it closely all year."
    hids=[tok.bos_token_id]+tok.encode(htext); ht=torch.tensor(hids,device=dev).unsqueeze(0); hseq=ht.shape[1]
    hcue=[i for i,t in enumerate(hids) if t==cid]
    if len(hcue)<2: continue
    hclean=model(ht)[0]; HFh,residh=fra_ph(ht)
    hedges=[(qi,ki+1) for qi in hcue for ki in hcue if ki+1<qi]
    byL_hedges=delta_edges(HFh,P,hedges,hseq) if hedges else {}
    byL_content=delta_content(HFh,P,hseq)
    vId=residh[L0][hcue[0]]-residh[L0].mean(0); vId/=(vId.norm()+1e-6)        # held-out cue-identity dir
    vIdp=resid[L0][1+N+10]-resid[L0].mean(0); vIdp/=(vIdp.norm()+1e-6)        # primer cue-identity dir
    ind_q=hcue[1:]                     # induction-query cue positions (have an earlier cue)
    cue_pos_primer=[1+10,1+N+10]
    def saa(positions,vv,s): return 1-torch.softmax(aa(tt,positions,vv,s)[qpos].float(),-1)[resp].item()/base
    # curves: (primer-suppression, held-out collateral)
    def fra_curve(byL): return [(supp(c), kltot(hclean,patch(ht,byL,c))) for c in FC]
    cur_hedges=fra_curve(byL_hedges) if byL_hedges else [(supp(c),0.0) for c in FC]
    cur_content=fra_curve(byL_content)
    # ActAdd-identity: suppress on primer at all cue positions; collateral on held-out at all cue positions
    cur_id=[(saa(cue_pos_primer,vIdp,s), kltot(hclean,aa(ht,hcue,vId,s))) for s in AC]
    # ActAdd-induction-gated (probe-gated): apply only at induction-query cue positions (fair strong baseline)
    cur_indgate=[(saa([1+N+10],vIdp,s), kltot(hclean,aa(ht,ind_q,vId,s))) for s in AC]
    rows.append(dict(cue=cw,base=base,cfaith=cfaith,real_edge=real_edge,fra_edge=fra_edge,
                     hedges=cur_hedges,content=cur_content,actadd_id=cur_id,actadd_indgate=cur_indgate))
    print(f"cue '{cw}' base {base:.2f} cfaith~{cfaith:.1f} (real_edge {real_edge:.1f}, fra {fra_edge:.1f})",flush=True)
    print(f"  FRA-hedges : {[(round(a,2),round(b,2)) for a,b in cur_hedges]}",flush=True)
    print(f"  FRA-content: {[(round(a,2),round(b,2)) for a,b in cur_content]}",flush=True)
    print(f"  ActAdd-id  : {[(round(a,2),round(b,2)) for a,b in cur_id]}",flush=True)
    print(f"  ActAdd-indg: {[(round(a,2),round(b,2)) for a,b in cur_indgate]}",flush=True)

def collat_at(curve,t):
    xs=[a for a,b in curve]; ys=[b for a,b in curve]
    if max(xs)<t: return None
    o=np.argsort(xs); xs=np.array(xs)[o]; ys=np.array(ys)[o]
    return float(np.interp(t,xs,ys))
for tgt in [0.3,0.5]:
    print(f"\n=== held-out collateral at suppression={tgt} (n cues with reach) ===",flush=True)
    for key in ["hedges","content","actadd_id","actadd_indgate"]:
        vals=[collat_at(r[key],tgt) for r in rows]; vals=[v for v in vals if v is not None]
        if vals: print(f"  {key:16}: {np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)})",flush=True)
json.dump({"rows":rows},open(os.path.join(OUT,"j9.json"),"w"),indent=2,default=float)
print("\nDONE j9",flush=True)
