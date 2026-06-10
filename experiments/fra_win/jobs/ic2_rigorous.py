"""IC2 — rigorous in-context backdoor flip. Content-addressed FRA (no hedges whitelist; the trigger's
induction pairs, learned from the primer, fire wherever they appear on the full held-out matrix) +
multi-mention held-out text (trigger x3) + collateral split into NON-backdoor positions (the trigger's
1st occurrence, where no induction edge exists -> pure collateral) vs total.
Methods at matched ASR-suppression: FRA-QK vs ActAdd-trigger vs payload-suppress (sleeper-winner).
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
saes={L:(s[0] if isinstance(s,tuple) else s) for L,s in saes.items()}; W_U=model.W_U
def fra_ph(tt):
    _,c=model.run_with_cache(tt,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
    H={}; resid={L:c[f"blocks.{L}.hook_resid_pre"][0] for L in LAYERS}
    for (L,Hh) in IND:
        fe=saes[L].encode(c[f"blocks.{L}.hook_resid_pre"][0]).float(); xh=fe@saes[L].W_dec.float()+saes[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,saes[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
        H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H,resid
def primer_pairs(HF,edge,M=12):
    P={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; loc=np.where((d["qq"]==edge[0])&(d["kk"]==edge[1]))[0]; loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
        P[(L,Hh)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in loc)
    return P
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
def delta_content(HF,P,seq):  # pairs fire wherever they appear on the FULL matrix (honest content-addressed)
    byL={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; dd=np.zeros((seq,seq)); Ps=P[(L,Hh)]
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        byL.setdefault(L,{})[Hh]=dd
    return byL
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
def actadd(tt,positions,vX,s):
    def hook(act,hook):
        for p in positions:
            if p<tt.shape[1]: act[0,p,:]=act[0,p,:]-s*vX*act[0,p,:].norm()
        return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{L0}.hook_resid_pre",hook)])[0]
def paysupp(tt,Pid,s):
    uP=W_U[:,Pid].float(); uP=uP/uP.norm()
    def hook(act,hook): act[0]=act[0]-s*(act[0]@uP).unsqueeze(-1)*uP; return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{Llast}.hook_resid_post",hook)])[0]
def klvec(p,q):
    lp=torch.log_softmax(p.float(),-1); lq=torch.log_softmax(q.float(),-1)
    return (lp.exp()*(lp-lq)).sum(-1)

CASES=[(" bank"," river","The bank opened early. People queued at the bank. A new bank policy upset them."),
       (" fire"," water","The fire alarm rang. A second fire started nearby. Firefighters fought the fire all night."),
       (" king"," crown","The king arrived first. A foreign king joined him. Together the king and his guests dined."),
       (" doctor"," patient","The doctor was tired. Another doctor took over. Each doctor worked a twelve hour shift.")]
FC=[1,2,4,8,16]; AC=[0.25,0.5,1,2,4,8]; PC=[0.5,1,2,4,8]
rows=[]
for T,P,htext in CASES:
    Tid=tok.encode(T); Pid=tok.encode(P)
    if len(Tid)!=1 or len(Pid)!=1: continue
    Tid=Tid[0]; Pid=Pid[0]
    torch.manual_seed(0); N=20; R=(torch.randperm(40000)[:N]+1000).tolist(); R[10]=Tid; R[11]=Pid
    tt=torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0); seq=tt.shape[1]; qpos=1+N+10; kpos=12
    base=torch.softmax(model(tt)[0][qpos].float(),-1)[Pid].item()
    if base<0.2: continue
    def asr_supp(lg): return 1-torch.softmax(lg[qpos].float(),-1)[Pid].item()/base
    HF,resid=fra_ph(tt); Pp=primer_pairs(HF,(qpos,kpos)); byL_supp=delta_edges(HF,Pp,[(qpos,kpos)],seq)
    def supp(c): return asr_supp(patch(tt,byL_supp,c))
    vTp=resid[L0][1+N+10]-resid[L0].mean(0); vTp/=(vTp.norm()+1e-6)
    # held-out (trigger x3)
    hids=[tok.bos_token_id]+tok.encode(htext); ht=torch.tensor(hids,device=dev).unsqueeze(0); hseq=ht.shape[1]
    hT=[i for i,t in enumerate(hids) if t==Tid]
    if len(hT)<2: print(f"skip {T} held-out trig {len(hT)}",flush=True); continue
    nonbd=[hT[0]]                                  # 1st trigger occurrence = pure collateral (no prior trigger)
    hclean=model(ht)[0]; HFh,residh=fra_ph(ht)
    byLh=delta_content(HFh,Pp,hseq)                # content-addressed FRA on held-out (full matrix, no whitelist)
    vTh=residh[L0][hT[0]]-residh[L0].mean(0); vTh/=(vTh.norm()+1e-6)
    def colvec(method,strength):
        if method=="fra": return klvec(hclean,patch(ht,byLh,strength))
        if method=="trig": return klvec(hclean,actadd(ht,hT,vTh,strength))
        if method=="pay": return klvec(hclean,paysupp(ht,Pid,strength))
    def curve(method,grid,strfn):
        out=[]
        for x in grid:
            sp=strfn(x); kv=colvec(method,x)
            out.append((sp, float(kv.sum().item()), float(kv[nonbd].mean().item())))   # (supp, total_kl, nonbd_kl)
        return out
    cur_fra=curve("fra",FC,supp)
    cur_trig=curve("trig",AC,lambda s:asr_supp(actadd(tt,[1+10,1+N+10],vTp,s)))
    cur_pay=curve("pay",PC,lambda s:asr_supp(paysupp(tt,Pid,s)))
    rows.append(dict(T=T,P=P,base=base,fra=cur_fra,trig=cur_trig,pay=cur_pay))
    print(f"{T.strip()}->{P.strip()} ASR {base:.2f}:",flush=True)
    print(f"  FRA  (supp,totKL,nonbdKL): {[(round(a,2),round(b,2),round(c,3)) for a,b,c in cur_fra]}",flush=True)
    print(f"  trig : {[(round(a,2),round(b,2),round(c,3)) for a,b,c in cur_trig]}",flush=True)
    print(f"  pay  : {[(round(a,2),round(b,2),round(c,3)) for a,b,c in cur_pay]}",flush=True)

def at(curve,t,idx):
    xs=[r[0] for r in curve]; ys=[r[idx] for r in curve]
    if max(xs)<t: return None
    o=np.argsort(xs); return float(np.interp(t,np.array(xs)[o],np.array(ys)[o]))
print(f"\n=== rigorous in-context backdoor flip, @ 80% ASR-suppression (n={len(rows)}) ===",flush=True)
for key,lab in [("fra","FRA-QK (attention edge)"),("trig","ActAdd-trigger"),("pay","payload-suppress (sleeper-winner)")]:
    tot=[at(r[key],0.8,1) for r in rows]; tot=[x for x in tot if x is not None]
    nb=[at(r[key],0.8,2) for r in rows]; nb=[x for x in nb if x is not None]
    if tot: print(f"  {lab:36}: total KL {np.mean(tot):.3f}±{np.std(tot):.3f} | non-backdoor-pos KL {np.mean(nb):.3f}±{np.std(nb):.3f}",flush=True)
json.dump({"rows":rows},open(os.path.join(OUT,"ic2.json"),"w"),indent=2,default=float)
plt.figure(figsize=(6.4,4.7))
for key,lab,c in [("fra","FRA-QK (attention edge)","C0"),("trig","ActAdd-trigger (linear)","C1"),("pay","payload-suppress (DoM/output: won on weight-baked sleeper)","C2")]:
    for r in rows:
        xs=[v[0] for v in r[key]]; ys=[v[1] for v in r[key]]; plt.plot(xs,ys,'-o',color=c,alpha=0.4,ms=3)
    plt.plot([],[],'-o',color=c,label=lab)
plt.yscale('symlog',linthresh=0.1); plt.xlabel("backdoor ASR suppression (1 − ASR/base) → stronger"); plt.ylabel("held-out collateral KL (nats, content-addressed) ↓ better")
plt.title("In-context (attention-routed) backdoor: the removal ranking FLIPS\nFRA-QK now wins; the output-suppression that won on the weight-baked sleeper loses")
plt.legend(fontsize=7); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(os.path.join(OUT,"ic2_flip.png"),dpi=130)
print("\nDONE ic2",flush=True)
