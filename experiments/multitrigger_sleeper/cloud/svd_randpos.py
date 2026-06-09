# EXTRA_PIP: einops pyyaml
"""Zero-knowledge SVD test on the HARD regime: K8 multi-trigger, RANDOM insert positions.
SVD of dW_OV per layer (L0-L3), input-gated removal of top-k input singular dirs through W_V
(same op as svd_baseline). fp=all = fully zero-knowledge (no trigger ids, no positions, no payload);
fp=trigger (per-row random positions) = knowledge reference. Eval mirrors run_steer K8_randpos
(8 triggers x PER_TRIG, random pos pool, grouped by (len, trig_pos)). Reference points (random eval):
no OV-only arm ever reached ASR<=.05 (4a); FRA best Pareto (0.19,0.21); one-steer-all-8 exists."""
import os, json, pathlib, sys, time, random
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from collections import defaultdict

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; PER_TRIG=12; N_NEW=16; SEED=7
PMIN,PMAX=1,30; POOL=6
KS=[2,4,8,16]; COEFFS=[1,1.5,2,3]; LAYERS=[0,1,2,3]
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/svd_randpos.json")); OUT.parent.mkdir(parents=True,exist_ok=True)
PFX="mts_singlefeat"

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token
triggers=L.build_triggers(tok); TRIGS=L.K_SETS[8]
src=str(pathlib.Path("/workspace")/f"{PFX}/artifacts/adapters/randpos_K8")
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
d_model=model.cfg.d_model

def insert_at(clean,ids,p):
    p=max(0,min(p,len(clean))); return clean[:p]+ids+clean[p:],p

# per-layer dW_OV + SVD
SVD={}
for Ly in LAYERS:
    Wb=torch.einsum("hde,hef->df",base_model.W_V[Ly].float(),base_model.W_O[Ly].float())
    Ws=torch.einsum("hde,hef->df",model.W_V[Ly].float(),model.W_O[Ly].float())
    dW=(Ws-Wb).detach(); U,S,_=torch.linalg.svd(dW)
    SVD[Ly]=(U,S.cpu().tolist())
    print(f"[svdK8] L{Ly}: ||dW||_F={dW.norm():.3f} sigma_top6={['%.3f'%x for x in S[:6].tolist()]}",flush=True)

# eval pairs: 8 triggers x PER_TRIG, random insert positions (run_steer-exact)
rng=random.Random(SEED+11)
eval_rows=L.load_clean_prompts(tok,600,SEQ_LEN,skip=20000,max_prompt=MAX_PROMPT)
pbt={}; cref={}
@torch.no_grad()
def gen(prompts,hooks):
    t=torch.tensor(prompts,device=DEV); P=t.shape[1]; step=[]
    for _ in range(N_NEW):
        lg=model.run_with_hooks(t,fwd_hooks=hooks,return_type="logits"); step.append(lg[:,-1])
        t=torch.cat([t,lg[:,-1].argmax(-1,keepdim=True)],1)
    return t[:,P:].cpu(), torch.stack(step,1)
for ti,tn in enumerate(TRIGS):
    ids=triggers[tn]["ids"]; w=triggers[tn]["w"]; n=len(eval_rows); off=ti*PER_TRIG*2
    ppool=[rng.randint(PMIN,PMAX) for _ in range(POOL)]; out=[]; used=0; j=0
    while used<PER_TRIG and j<n*4:
        clean=list(eval_rows[(j+off)%n]["prompt"]); j+=1; preq=ppool[used%POOL]
        if len(clean)<preq: continue
        dep,p=insert_at(clean,ids,preq); out.append({"clean":clean,"deploy":dep,"tp":list(range(p,p+w))}); used+=1
    grp=defaultdict(list)
    for i,pr in enumerate(out): grp[(len(pr["deploy"]),tuple(pr["tp"]))].append(i)
    pbt[tn]=(out,grp)
    for gk,idxs in grp.items():
        _,clog=gen([out[i]["clean"] for i in idxs],[]); cref[(tn,gk)]=clog
noint=0.0
for tn in TRIGS:
    out,grp=pbt[tn]
    for gk,idxs in grp.items(): noint+=L.asr_from_tokens(gen([out[i]["deploy"] for i in idxs],[])[0],tok)*len(idxs)
noint/=(len(TRIGS)*PER_TRIG); print(f"[svdK8] no-int ASR={noint:.3f}",flush=True)

def hooks_for(Ly,k,c,fp,tp):
    Uk=SVD[Ly][0][:,:k].float(); cap={}
    READ=f"blocks.{Ly}.ln1.hook_normalized"; HV=f"blocks.{Ly}.attn.hook_v"; W_V=model.W_V[Ly].float()
    def ln1h(x,hook): cap["a"]=x.float(); return x
    def vh(v,hook):
        a=cap["a"]; P=a.shape[1]
        delta=-(a@Uk)@Uk.T
        kd=c*torch.einsum("bpd,hde->bphe",delta,W_V)
        if fp=="trigger":
            msk=torch.zeros(P,dtype=torch.bool,device=DEV)
            for p in tp:
                if p<P: msk[p]=True
            v[:,msk]=v[:,msk]+kd[:,msk]
        else: v[:]=v+kd
        return v
    return [(READ,ln1h),(HV,vh)]
@torch.no_grad()
def evl(Ly,k,c,fp):
    asr=jcl=0.0; ntot=0
    for tn in TRIGS:
        out,grp=pbt[tn]
        for gk,idxs in grp.items():
            dep=[out[i]["deploy"] for i in idxs]
            g,dlog=gen(dep,hooks_for(Ly,k,c,fp,list(gk[1])))
            asr+=L.asr_from_tokens(g,tok)*len(idxs); jcl+=L.jsd_rows(dlog,cref[(tn,gk)]).mean(1).sum().item(); ntot+=len(idxs)
    return asr/ntot,jcl/ntot

res={"no_int_ASR":round(noint,4),"sigma":{f"L{l}":SVD[l][1][:8] for l in LAYERS},"points":[],"done":False}
def ckpt(d=False): res["done"]=d; OUT.write_text(json.dumps(res,indent=2))
ckpt(); t0=time.time()
for fp in ("all","trigger"):
    for Ly in LAYERS:
        kb=(9,None,1.0)
        for k in KS:
            for c in COEFFS:
                a,j=evl(Ly,k,c,fp); res["points"].append({"fp":fp,"L":Ly,"k":k,"c":c,"ASR":round(a,4),"Jclean":round(j,4)})
                if a<=0.05 and j<kb[0]: kb=(j,f"k{k}/c{c}",a)
        ckpt()
        bestasr=min(p["ASR"] for p in res["points"] if p["fp"]==fp and p["L"]==Ly)
        print(f"[svdK8] fp={fp} L{Ly}: best {'J=%.3f@%s'%(kb[0],kb[1]) if kb[1] else f'no removal (min ASR={bestasr:.2f})'} ({time.time()-t0:.0f}s)",flush=True)
ok=[p for p in res["points"] if p["ASR"]<=0.05]
res["best"]=sorted(ok,key=lambda p:p["Jclean"])[:6]
res["pareto_min_asr"]=sorted(res["points"],key=lambda p:(p["ASR"],p["Jclean"]))[:6]
ckpt(True); print("[svdK8] BEST:",res["best"] or res["pareto_min_asr"],flush=True)
