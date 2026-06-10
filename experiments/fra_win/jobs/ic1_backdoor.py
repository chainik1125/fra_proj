"""IC1 — in-context backdoor removal: does the method ranking flip vs the weight-baked sleeper?
Plant a trigger T -> payload P association in-context (induction), so the 2nd T retrieves P (ASR).
Remove the backdoor 4 ways, measure ASR-suppression vs held-out collateral (KL on normal text where
T and P appear in non-backdoor contexts):
  FRA-QK          : ablate the induction edge T->(post-first-T) in the induction heads (the association)
  ActAdd-trigger  : subtract T's residual direction wherever T is current (suppress the trigger)
  payload-suppress: subtract P's unembedding direction at the final residual (suppress P at output) <- the
                    OV/output-style removal that WON on the weight-baked sleeper (DoM/SVD analog)
  oracle          : zero the induction attention T->post-T (content-agnostic, needs positions)
Prediction: FRA-QK low collateral; trigger/payload suppression high collateral (they corrupt a normal endpoint).
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
model=HookedTransformer.from_pretrained("gpt2",device=dev); model.eval(); tok=model.tokenizer
IND=[(5,5),(6,9),(5,1),(7,10),(7,2)]; LAYERS=sorted(set(L for L,H in IND)); L0=min(LAYERS); Llast=model.cfg.n_layers-1
saes={L:(SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev)) for L in LAYERS}
saes={L:(s[0] if isinstance(s,tuple) else s) for L,s in saes.items()}
W_U=model.W_U  # [d_model, vocab]

def fra_ph(tt):
    _,c=model.run_with_cache(tt,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
    H={}; resid={L:c[f"blocks.{L}.hook_resid_pre"][0] for L in LAYERS}
    for (L,Hh) in IND:
        fe=saes[L].encode(c[f"blocks.{L}.hook_resid_pre"][0]).float(); xh=fe@saes[L].W_dec.float()+saes[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,saes[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
        H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H,resid
def fra_delta(HF,edges,seq):
    byL={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; dd=np.zeros((seq,seq))
        for (qi,ki) in edges:
            loc=np.where((d["qq"]==qi)&(d["kk"]==ki))[0]
            for o in loc: dd[qi,ki]+=d["vv"][o]
        byL.setdefault(L,{})[Hh]=dd
    return byL
def patch_scores(tt,byL,c):
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
def actadd(tt,positions,vX,s):
    def hook(act,hook):
        for p in positions:
            if p<tt.shape[1]: act[0,p,:]=act[0,p,:]-s*vX*act[0,p,:].norm()
        return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{L0}.hook_resid_pre",hook)])[0]
def payload_suppress(tt,Pid,s):  # subtract P unembedding dir at final resid, all positions
    uP=W_U[:,Pid].float(); uP=uP/uP.norm()
    def hook(act,hook):
        act[0]=act[0]-s*(act[0]@uP).unsqueeze(-1)*uP   # remove component along uP
        return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{Llast}.hook_resid_post",hook)])[0]
def kltot(p,q):
    lp=torch.log_softmax(p.float(),-1); lq=torch.log_softmax(q.float(),-1)
    return (lp.exp()*(lp-lq)).sum(-1).sum().item()

# trigger/payload pairs (single tokens, both appear in a natural held-out sentence)
CASES=[(" bank"," river","He sat on the bank watching the river flow past the old stone bridge."),
       (" fire"," water","The fire spread fast until water from the river finally put it out."),
       (" king"," crown","The king wore a heavy crown of gold at every royal ceremony that year."),
       (" doctor"," patient","The doctor met each patient calmly and explained the test results clearly.")]
FC=[1,2,4,8,16,32]; AC=[0.25,0.5,1,2,4,8]; PC=[0.5,1,2,4,8,16]
rows=[]
for T,P,htext in CASES:
    Tid=tok.encode(T); Pid=tok.encode(P)
    if len(Tid)!=1 or len(Pid)!=1: print(f"skip {T}->{P}",flush=True); continue
    Tid=Tid[0]; Pid=Pid[0]
    # backdoor induction sequence: filler with T->P planted at idx 10, repeated
    torch.manual_seed(0); N=20
    R=(torch.randperm(40000)[:N]+1000).tolist(); R[10]=Tid; R[11]=Pid
    tt=torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0); seq=tt.shape[1]
    qpos=1+N+10; kpos=12
    base=torch.softmax(model(tt)[0][qpos].float(),-1)[Pid].item()
    if base<0.2: print(f"skip {T}->{P} ASR {base:.2f}",flush=True); continue
    def asr_supp(lg): return 1-torch.softmax(lg[qpos].float(),-1)[Pid].item()/base
    HF,resid=fra_ph(tt); byL=fra_delta(HF,[(qpos,kpos)],seq)
    vT=resid[L0][1+N+10]-resid[L0].mean(0); vT/=(vT.norm()+1e-6)
    # held-out collateral text (T and P appear normally)
    hids=[tok.bos_token_id]+tok.encode(htext); ht=torch.tensor(hids,device=dev).unsqueeze(0); hseq=ht.shape[1]
    hcue=[i for i,t in enumerate(hids) if t==Tid]
    hclean=model(ht)[0]
    HFh,residh=fra_ph(ht)
    hedges=[(qi,ki+1) for qi in hcue for ki in hcue if ki+1<qi]
    byLh=fra_delta(HFh,hedges,hseq) if hedges else {}
    vTh=residh[L0][hcue[0]]-residh[L0].mean(0); vTh/=(vTh.norm()+1e-6) if hcue else 1
    # curves
    cur_fra=[(asr_supp(patch_scores(tt,byL,c)), (kltot(hclean,patch_scores(ht,byLh,c)) if byLh else 0.0)) for c in FC]
    cur_trig=[(asr_supp(actadd(tt,[1+10,1+N+10],vT,s)), kltot(hclean,actadd(ht,hcue,vTh,s)) if hcue else 0.0) for s in AC]
    cur_pay=[(asr_supp(payload_suppress(tt,Pid,s)), kltot(hclean,payload_suppress(ht,Pid,s))) for s in PC]
    rows.append(dict(T=T,P=P,base=base,fra=cur_fra,trig=cur_trig,pay=cur_pay))
    print(f"{T.strip()}->{P.strip()} ASR {base:.2f}:",flush=True)
    print(f"  FRA  {[(round(a,2),round(b,2)) for a,b in cur_fra]}",flush=True)
    print(f"  trig {[(round(a,2),round(b,2)) for a,b in cur_trig]}",flush=True)
    print(f"  pay  {[(round(a,2),round(b,2)) for a,b in cur_pay]}",flush=True)

def collat_at(curve,t):
    xs=[a for a,b in curve]; ys=[b for a,b in curve]
    if max(xs)<t: return None
    o=np.argsort(xs); return float(np.interp(t,np.array(xs)[o],np.array(ys)[o]))
print(f"\n=== in-context backdoor, held-out collateral @ 80% ASR-suppression (n={len(rows)}) ===",flush=True)
for key,lab in [("fra","FRA-QK"),("trig","ActAdd-trigger"),("pay","payload-suppress (sleeper-winner)")]:
    v=[collat_at(r[key],0.8) for r in rows]; v=[x for x in v if x is not None]
    if v: print(f"  {lab:34}: {np.mean(v):.3f} ± {np.std(v):.3f} (n={len(v)})",flush=True)
json.dump({"rows":rows},open(os.path.join(OUT,"ic1.json"),"w"),indent=2,default=float)
plt.figure(figsize=(6.2,4.6))
for key,lab,c in [("fra","FRA-QK (attention-edge)","C0"),("trig","ActAdd-trigger","C1"),("pay","payload-suppress (DoM/output — won on weight-baked sleeper)","C2")]:
    for r in rows:
        xs=[a for a,b in r[key]]; ys=[b for a,b in r[key]]; plt.plot(xs,ys,'-o',color=c,alpha=0.35,ms=3)
    plt.plot([],[],'-o',color=c,label=lab)
plt.yscale('symlog',linthresh=0.1); plt.xlabel("backdoor ASR suppression (1 − ASR/base) → stronger"); plt.ylabel("held-out collateral KL (nats) ↓ better")
plt.title("In-context backdoor removal: the ranking FLIPS\nFRA-QK (attention) beats the payload-suppression that won on the weight-baked sleeper")
plt.legend(fontsize=7); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(os.path.join(OUT,"ic1_flip.png"),dpi=130)
print("\nDONE ic1",flush=True)
