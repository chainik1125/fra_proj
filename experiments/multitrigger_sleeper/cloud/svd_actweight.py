# EXTRA_PIP: einops pyyaml
"""Activation-weighted SVD of dW_OV @ ln1.L2 -- the middle object between plain SVD (data-free) and
SAE dictionaries (data-shaped basis). C = E[x x^T] on a corpus (deploy prompts = unlabeled-poison
tier | clean text = zero-knowledge tier). Two variants:
  rerank : plain SVD dirs u_i re-ranked by data excitation sigma_i^2 * (u_i^T C u_i); gated orthogonal
           ablation h -= c (h.u) u through W_V (same as svd_baseline).
  whiten : SVD of T = C^{1/2} dW_OV -> u~_i; oblique ablation h -= c (h.b_i) a_i with b_i=C^{-1/2}u~_i,
           a_i=C^{1/2}u~_i (removes the data-excited input components that dW amplifies; c=1 exact).
Sweep k x c x footprint{trigger,all}; reference: plain SVD (0,.205 all / 0,.136 trigger)."""
import os, json, pathlib, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from collections import defaultdict

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; PER=24; N_NEW=16; OV_L=2; NC=2000
READ=f"blocks.{OV_L}.ln1.hook_normalized"; HOOK_V=f"blocks.{OV_L}.attn.hook_v"
KS=[1,2,4,8]; COEFFS=[0.5,1,1.5,2,3]
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/svd_actweight.json")); OUT.parent.mkdir(parents=True,exist_ok=True)
PFX="mts_singlefeat"

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token; pad=tok.eos_token_id
triggers=L.build_triggers(tok); trig=triggers["DEPLOYMENT"]["ids"]; w=len(trig); INS=L.INSERT_IDX
src=str(pathlib.Path("/workspace")/f"{PFX}/artifacts/adapters/K1")
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
d_model=model.cfg.d_model; W_V0=model.W_V[OV_L].float()
W_OV_b=torch.einsum("hde,hef->df",base_model.W_V[OV_L].float(),base_model.W_O[OV_L].float())
W_OV_s=torch.einsum("hde,hef->df",model.W_V[OV_L].float(),model.W_O[OV_L].float())
dW=(W_OV_s-W_OV_b).detach().double()

@torch.no_grad()
def second_moment(kind):
    rows=L.load_clean_prompts(tok,NC,SEQ_LEN,skip=40000,max_prompt=MAX_PROMPT)
    C=torch.zeros(d_model,d_model,dtype=torch.float64,device=DEV); n=0
    for s0 in range(0,len(rows),64):
        bt=[]
        for r in rows[s0:s0+64]:
            seq=L.make_deploy_prompt(list(r["prompt"]),trig) if kind=="deploy" else (list(r["prompt"])+r["story"])[:SEQ_LEN]
            bt.append((seq+[pad]*(SEQ_LEN-len(seq)))[:SEQ_LEN])
        _,c=model.run_with_cache(torch.tensor(bt,device=DEV),return_type=None,names_filter=lambda nm:nm==READ)
        a=c[READ].double().reshape(-1,d_model)   # incl pads: minor; dominated by real tokens
        C+=a.T@a; n+=a.shape[0]
    return C/n

U0,S0,_=torch.linalg.svd(dW)
def dirs_for(variant,C):
    if variant=="rerank":
        exc=(S0**2)*torch.einsum("di,dc,ci->i",U0,C,U0)   # sigma^2 * u^T C u
        order=torch.argsort(exc,descending=True).tolist()
        Uo=U0[:,order]
        return [("ortho",Uo[:,:k].float()) for k in KS],order[:8]
    evals,evecs=torch.linalg.eigh(C)
    evals=evals.clamp(min=1e-9)
    Chalf=(evecs*evals.sqrt())@evecs.T; Cinvh=(evecs*evals.rsqrt())@evecs.T
    T=Chalf@dW; Ut,St,_=torch.linalg.svd(T)
    A=(Chalf@Ut); B=(Cinvh@Ut)                              # a_i (removal), b_i (detector)
    return [("oblique",(A[:,:k].float(),B[:,:k].float())) for k in KS],St[:8].tolist()

rows=L.load_clean_prompts(tok,600,SEQ_LEN,skip=20000,max_prompt=MAX_PROMPT)[:PER]
grp=defaultdict(list)
for i,r in enumerate(rows): grp[len(r["prompt"])].append(i)
@torch.no_grad()
def gen(prompts,hooks):
    t=torch.tensor(prompts,device=DEV); P=t.shape[1]; step=[]
    for _ in range(N_NEW):
        lg=model.run_with_hooks(t,fwd_hooks=hooks,return_type="logits"); step.append(lg[:,-1])
        t=torch.cat([t,lg[:,-1].argmax(-1,keepdim=True)],1)
    return t[:,P:].cpu(), torch.stack(step,1)
cref={}
for ck,idxs in grp.items(): _,cl=gen([list(rows[i]["prompt"]) for i in idxs],[]); cref[ck]=cl
def hooks_for(kind,mat,c,fp):
    cap={}
    def ln1h(x,hook): cap["a"]=x.float(); return x
    def vh(v,hook):
        a=cap["a"]; P=a.shape[1]
        if kind=="ortho": Uk=mat; delta=-(a@Uk)@Uk.T
        else: A,B=mat; delta=-(a@B)@A.T
        kd=c*torch.einsum("bpd,hde->bphe",delta,W_V0)
        if fp=="trigger":
            msk=torch.zeros(P,dtype=torch.bool,device=DEV); msk[INS:min(INS+w,P)]=True
            v[:,msk]=v[:,msk]+kd[:,msk]
        else: v[:]=v+kd
        return v
    return [(READ,ln1h),(HOOK_V,vh)]
@torch.no_grad()
def evl(kind,mat,c,fp):
    asr=jcl=0.0
    for ck,idxs in grp.items():
        dep=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]
        g,dlog=gen(dep,hooks_for(kind,mat,c,fp))
        asr+=L.asr_from_tokens(g,tok)*len(idxs); jcl+=L.jsd_rows(dlog,cref[ck]).mean(1).sum().item()
    return asr/PER,jcl/PER

res={"points":[],"meta":{},"done":False}
def ckpt(d=False): res["done"]=d; OUT.write_text(json.dumps(res,indent=2))
ckpt(); t0=time.time()
for corpus in ("deploy","clean"):
    C=second_moment(corpus)
    for variant in ("rerank","whiten"):
        mats,meta=dirs_for(variant,C); res["meta"][f"{corpus}_{variant}"]=meta
        for (kind,mat),k in zip(mats,KS):
            kb=(9,None)
            for fp in ("trigger","all"):
                for c in COEFFS:
                    a,j=evl(kind,mat,c,fp); res["points"].append({"corpus":corpus,"variant":variant,"k":k,"fp":fp,"c":c,"ASR":round(a,4),"Jclean":round(j,4)})
                    if a<=0.05 and j<kb[0]: kb=(j,f"{fp}/c{c}")
            ckpt(); print(f"[aw] {corpus}/{variant} k={k}: best {'J=%.3f@%s'%kb if kb[1] else 'no removal'} ({time.time()-t0:.0f}s)",flush=True)
ok=[p for p in res["points"] if p["ASR"]<=0.05]
res["best_per_cell"]={}
for corpus in ("deploy","clean"):
    for variant in ("rerank","whiten"):
        for fp in ("trigger","all"):
            cell=[p for p in ok if p["corpus"]==corpus and p["variant"]==variant and p["fp"]==fp]
            res["best_per_cell"][f"{corpus}/{variant}/{fp}"]=(min(cell,key=lambda p:p["Jclean"]) if cell else None)
ckpt(True); print("[aw] BEST:",json.dumps(res["best_per_cell"],indent=1),flush=True)
