"""RUNG 1 -- the in-context backdoor in NATURAL TEXT (Gemma-2-2b base, GemmaScope 65k).

RESEARCH CONTEXT. Defensive interpretability research for an academic paper on
Feature-Resolved Attention (FRA), with Dmitry Manning-Coe. The question is whether an
interpretability-derived edit can REMOVE an unwanted in-context association with less
collateral damage than standard steering baselines. The "trigger" and "payload" are benign
placeholder word pairs (bank->river, king->crown); no harmful content is involved or produced.
"Backdoor" is the literature's term for this setup (e.g. demonstration-poisoning threat models),
used here only to evaluate a mitigation. See docs/insen/research_context.md.

Daily-ladder step 1 of 4 toward a real-world FRA win (plan agreed with Dmitry 2026-09-12).

ONE change vs Setting 1 (experiments/fra_win/jobs/g4_65k.py @ iclr-summary 7338adf): the
PROMPT. Setting 1 plants trigger->payload inside random tokens repeated twice. Here the same
four pairs are planted in ordinary English:

  <4 filler sentences> The password is bank river. <6 filler> Remember the password: bank -> " river"?

Everything else is HIS code, spliced line-for-line from g4_65k.py: induction-head discovery,
the 65k SAEs, FRA-QK / DoM / conv-SAE / payload-suppress, the held-out collateral texts and the
strength grids. A difference from Setting 1 is attributable to natural text and nothing else.

Design choice, flagged: the DoM / conv-SAE contrast uses OFF = the same template with the trigger
planted but NO payload. Setting 1 used unrelated random sequences. This isolates the association.

Setting 1 reference, collateral @30% suppression: FRA 0.522 | DoM 13.49 | conv-SAE 11.94.
"""

import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
Llast=model.cfg.n_layers-1; W_U=model.W_U
# induction heads
torch.manual_seed(0); N=24
R=(torch.randperm(40000)[:N]+1000).tolist(); tt0=torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0)
edges0=[(1+N+t,t+2) for t in range(N-1)]
_,c0=model.run_with_cache(tt0,names_filter=lambda n:n.endswith("hook_pattern"))
strength={}
for L in range(model.cfg.n_layers):
    p=c0[f"blocks.{L}.attn.hook_pattern"][0]
    for H in range(p.shape[0]): strength[(L,H)]=float(np.mean([p[H,q,k].item() for q,k in edges0]))
IND=[lh for lh,s in sorted(strength.items(),key=lambda x:-x[1]) if s>0.4][:10]
LAYERS=sorted(set(L for L,H in IND)); L0=min(LAYERS); DL=6
print("induction heads:",[(f"L{L}H{H}") for L,H in IND],flush=True)
SAE={}
for L in sorted(set(LAYERS)|{DL}):
    sl=L-1
    try: SAE[L]=GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{sl}/width_65k/canonical",device=dev,normalize_activations=True)
    except Exception: SAE[L]=GemmaScopeSAE("gemma-scope-2b-pt-res",f"layer_{sl}/width_65k/average_l0_72",device=dev,normalize_activations=True)
print("SAEs loaded",flush=True); dsae=SAE[DL]
def encode(L,x):
    f=SAE[L].encode(x.float()).float()
    if SAE[L]._norm_coeff is not None: f=f/SAE[L]._norm_coeff
    return f
def fra_ph(tt):
    _,c=model.run_with_cache(tt,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
    H={}; resid={L:c[f"blocks.{L}.hook_resid_pre"][0] for L in LAYERS}
    for (L,Hh) in IND:
        fe=encode(L,c[f"blocks.{L}.hook_resid_pre"][0]); xh=fe@SAE[L].W_dec.float()+SAE[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,SAE[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
        H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H,resid
def primer_pairs(HF,edge,M=12):
    P={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; loc=np.where((d["qq"]==edge[0])&(d["kk"]==edge[1]))[0]; loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
        P[(L,Hh)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in loc)
    return P
def delta_content(HF,P,seq):
    byL={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; dd=np.zeros((seq,seq)); Ps=P[(L,Hh)]
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        byL.setdefault(L,{})[Hh]=dd
    return byL
def patch_fra(tt,byL,c):
    seq=tt.shape[1]; hooks=[]
    for L,hd in byL.items():
        td={Hh:torch.tensor(dd,device=dev,dtype=torch.float32)*c for Hh,dd in hd.items()}
        def mk(td):
            def hook(s,hook):
                for Hh,sd in td.items(): s[0,Hh,:seq,:seq]=s[0,Hh,:seq,:seq]-sd.to(s.dtype)
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(td)))
    return model.run_with_hooks(tt,fwd_hooks=hooks)[0]
def dom_run(tt,positions,vD,a):
    def hook(act,hook):
        for p in positions:
            if p<tt.shape[1]: act[0,p,:]=act[0,p,:]-(a*vD*act[0,p,:].float().norm()).to(act.dtype)
        return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{DL}.hook_resid_pre",hook)])[0]
def conv_run(tt,fidx,c):
    Wd=dsae.W_dec[fidx].float()
    def hook(act,hook):
        x=act[0].float(); z=encode(DL,x)[:,fidx]; act[0]=(x-(c*z)@Wd).to(act.dtype); return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{DL}.hook_resid_pre",hook)])[0]
def paysupp(tt,Pid,s):
    uP=W_U[:,Pid].float(); uP=uP/uP.norm()
    def hook(act,hook): act[0]=act[0]-(s*(act[0].float()@uP).unsqueeze(-1)*uP).to(act.dtype); return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{Llast}.hook_resid_post",hook)])[0]
def klsum(p,q):
    lp=torch.log_softmax(p.float(),-1); lq=torch.log_softmax(q.float(),-1); return (lp.exp()*(lp-lq)).sum(-1).sum().item()

# ---------------- RUNG 1: natural-text prompt builder (replaces mkseq) ----------------
FILLER=["The weather was mild for the time of year.","Several people arrived late to the meeting.",
 "The library opened an hour earlier than usual.","A small dog waited patiently by the door.",
 "Nobody expected the report to take so long.","The train was quiet on the way back.",
 "She wrote a short note and left it on the desk.","The garden needed more water after the heat.",
 "Two students argued about the final question.","The coffee in the office had gone cold.",
 "Most of the chairs were stacked against the wall.","He forgot his umbrella at the station.",
 "The new schedule starts next Monday.","A light rain began just after lunch.",
 "The shop on the corner closed for repairs.","They planned a short walk along the lake.",
 "The printer ran out of paper again.","Everyone agreed the soup was too salty.",
 "The museum added a room for old maps.","Her sister called to ask about the weekend.",
 "The bus took a longer route through town.","A few lights were still on in the building.",
 "The children painted a picture of a boat.","The package arrived two days early.",
 "The team reviewed the notes from last week.","A cold wind came in from the north.",
 "The old bridge was painted a bright color.","He read the instructions twice before starting.",
 "The kitchen smelled of fresh bread.","The road was closed because of the storm."]

def nat_ids(seed, T, P, planted=True, n_before=4, n_mid=6):
    g = np.random.default_rng(seed)
    fs = [FILLER[i] for i in g.permutation(len(FILLER))]
    plant = f" The password is{T}{P}." if planted else f" The password is{T}."
    text = (" ".join(fs[:n_before]) + plant + " " + " ".join(fs[n_before:n_before + n_mid])
            + " Remember the password:" + T)
    return [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)

def build(seed, Tid, Pid, T, P, planted=True):
    ids = nat_ids(seed, T, P, planted)
    qpos = len(ids) - 1                                   # the final trigger = the query
    tpos = [i for i, t in enumerate(ids) if t == Tid]     # every trigger occurrence
    kpos = next((i + 1 for i in range(len(ids) - 1) if ids[i] == Tid and ids[i + 1] == Pid), None)
    return torch.tensor(ids, device=dev).unsqueeze(0), qpos, kpos, tpos

CASES=[(" bank"," river","The river bank was crowded. He left the bank and crossed the river. By dusk the river hid the bank."),
       (" king"," crown","The king lost his crown. A new crown was made. The old king wore the crown while the young king watched."),
       (" doctor"," water","The doctor drank water. More water spilled. The doctor gave the patient water before the next doctor came."),
       (" market"," gold","The market sold gold. Gold prices fell. The market reopened and gold buyers crowded the market again.")]
FC=[1,2,4,8,16,32]; DC=[0.25,0.5,1,2,4,8]; CC=[1,2,4,8,16,32]; PC=[0.5,1,2,4,8]
rows=[]
for T,P,htext in CASES:
    Tid=tok.encode(T,add_special_tokens=False); Pid=tok.encode(P,add_special_tokens=False)
    if len(Tid)!=1 or len(Pid)!=1: print(f"skip {T}: not single-token",flush=True); continue
    Tid=Tid[0]; Pid=Pid[0]
    # DoM / conv-SAE contrast: ON = payload planted, OFF = same template, trigger planted without payload
    on=[]; off=[]; onf=[]; offf=[]
    for s in range(12):
        t1,q1,_,_=build(1000+s,Tid,Pid,T,P,True)
        a=model.run_with_cache(t1,names_filter=[f"blocks.{DL}.hook_resid_pre"])[1][f"blocks.{DL}.hook_resid_pre"][0]
        on.append(a[q1]); onf.append(encode(DL,a[q1:q1+1])[0])
        t0,q0,_,_=build(1000+s,Tid,Pid,T,P,False)
        b=model.run_with_cache(t0,names_filter=[f"blocks.{DL}.hook_resid_pre"])[1][f"blocks.{DL}.hook_resid_pre"][0]
        off.append(b[q0]); offf.append(encode(DL,b[q0:q0+1])[0])
    vD=(torch.stack(on).mean(0)-torch.stack(off).mean(0)).float(); vD=vD/(vD.norm()+1e-6)
    convK=torch.topk((torch.stack(onf).mean(0)-torch.stack(offf).mean(0)),12).indices.tolist()

    tt,qpos,kpos,tpos=build(0,Tid,Pid,T,P,True); seq=tt.shape[1]
    base=torch.softmax(model(tt)[0][qpos].float(),-1)[Pid].item()
    print(f"{T.strip()}->{P.strip()}: seq={seq} qpos={qpos} kpos={kpos} triggers={tpos} base ASR {base:.3f}",flush=True)
    if base<0.2 or kpos is None: print(f"skip {T} (ASR {base:.2f})",flush=True); continue
    def asr_s(lg,qpos=qpos): return 1-torch.softmax(lg[qpos].float(),-1)[Pid].item()/base
    HF,resid=fra_ph(tt); Pp=primer_pairs(HF,(qpos,kpos)); byL=delta_content(HF,Pp,seq)
    hids=[tok.bos_token_id]+tok.encode(htext,add_special_tokens=False); ht=torch.tensor(hids,device=dev).unsqueeze(0); hseq=ht.shape[1]
    hT=[i for i,t in enumerate(hids) if t==Tid]; hclean=model(ht)[0]; HFh,_=fra_ph(ht); byLh=delta_content(HFh,Pp,hseq)
    trig=tpos
    cur={"fra":[(asr_s(patch_fra(tt,byL,c)), klsum(hclean,patch_fra(ht,byLh,c))) for c in FC],
         "dom":[(asr_s(dom_run(tt,trig,vD,a)), klsum(hclean,dom_run(ht,hT,vD,a))) for a in DC],
         "conv":[(asr_s(conv_run(tt,convK,c)), klsum(hclean,conv_run(ht,convK,c))) for c in CC],
         "pay":[(asr_s(paysupp(tt,Pid,s)), klsum(hclean,paysupp(ht,Pid,s))) for s in PC]}
    rows.append(dict(T=T,P=P,base=base,seq=seq,qpos=qpos,kpos=kpos,**cur))
    for k in ["fra","dom","conv","pay"]: print(f"  {k:5}: {[(round(a,2),round(b,2)) for a,b in cur[k]]}",flush=True)

def at(curve,t):
    xs=[a for a,b in curve]; ys=[b for a,b in curve]
    if max(xs)<t: return None
    o=np.argsort(xs); return float(np.interp(t,np.array(xs)[o],np.array(ys)[o]))
for thr in (0.3, 0.7):
    print(f"\n=== RUNG 1 natural-text backdoor, held-out collateral @ {int(thr*100)}% ASR-suppression ===", flush=True)
    for k, lab in [("fra","FRA-QK (attention edge)"),("dom","DoM (mean-diff)"),("conv","conv-SAE (act-diff)"),("pay","payload-suppress (output)")]:
        v = [at(r[k], thr) for r in rows]; v = [x for x in v if x is not None]
        if v: print(f"  {lab:28}: {np.mean(v):.3f} +- {np.std(v):.3f} (n={len(v)}/{len(rows)} reached)", flush=True)
print("\nFRA reach per case:", [(r["T"].strip(), round(max(a for a, b in r["fra"]), 3)) for r in rows], flush=True)
print("Setting 1 (random tokens) @30%: FRA 0.522 | DoM 13.49 | conv 11.94", flush=True)

json.dump({"rows":rows},open(os.path.join(OUT,"rung1.json"),"w"),indent=2,default=float)
plt.figure(figsize=(6.6,4.8))
for k,lab,c in [("fra","FRA-QK (attention edge)","C0"),("dom","DoM / mean-diff (K8 winner)","C3"),("conv","conv-SAE steering (K8)","C4"),("pay","payload-suppress","C2")]:
    for r in rows:
        xs=[a for a,b in r[k]]; ys=[b for a,b in r[k]]; plt.plot(xs,ys,'-o',color=c,alpha=0.4,ms=3)
    plt.plot([],[],'-o',color=c,label=lab)
plt.yscale('symlog',linthresh=0.1); plt.xlabel("backdoor ASR suppression → stronger"); plt.ylabel("held-out collateral KL (nats) ↓ better")
plt.title("Rung 1: natural-text in-context backdoor (Gemma-2-2b, 65k)")
plt.legend(fontsize=7); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(os.path.join(OUT,"rung1_full.png"),dpi=130)
print("\nDONE rung1",flush=True)
