# EXTRA_PIP: einops pyyaml
"""K8-randpos unified steering-comparison runner. WORKLIST of cell specs, one JSON per cell
(results/k8grid/<cell>.json), resumable (skips done). All SAEs cached (eval-only cells).
Cell specs:
  fra:<dict>:<gate>:L<l>:<fp>     dict in {base,sleeper}; gate in {span,prompt,rollout}; ln1 route
  conv:<dict>:<hook>:L<l>:<fp>    dict in {base,sleeper,union}; hook in {ln1,rmid,rpost}; resid route
  dom:<hook>:L<l>:<fp>            hook in {ln1,rmid,rpost,q}; within/cross x {last,rollout} x {add,ablate} x alpha
  svd:L<l>:<fp>                   k x c, OV route
fp in {fptrigger,fpprompt,fpall}. Metrics per point: deploy ASR, J_deploy vs clean rollouts (BITS),
plus ALWAYS-ON COLLATERAL: the same intervention run on clean traffic -> J_collateral (BITS) +
clean exact-match. Eval: 8 triggers x PER_TRIG=12, random spans (seed 7 harness), greedy."""
import os, json, pathlib, sys, time, random, math
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE
from huggingface_hub import hf_hub_download, upload_file
from collections import defaultdict
LN2=math.log(2)

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; PER_TRIG=12; N_NEW=16; SEED=7
PMIN,PMAX=1,30; POOL=6
KS=[1,2,4,8,16,24,36,40]; CS=[0,0.5,1,1.5,2,3,4]
SVD_K=[2,4,8,16]; SVD_C=[1,1.5,2,3]; DOM_A=[1,2,4,8]
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat"; TOK=os.environ.get("HF_TOKEN")
WORK=[w.strip() for w in os.environ["WORKLIST"].split(",") if w.strip()]
PROG=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/k8grid_progress.json")); PROG.parent.mkdir(parents=True,exist_ok=True)
def hff(p): return hf_hub_download(HF_REPO,p,repo_type="dataset",local_dir="/workspace/kg_dl",token=TOK)
def result_done(rel):
    try: return bool(json.load(open(hff(rel))).get("done"))
    except Exception: return False

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token; pad=tok.eos_token_id
triggers=L.build_triggers(tok); TRIGS=L.K_SETS[8]
src=str(pathlib.Path("/workspace")/f"{PFX}/artifacts/adapters/randpos_K8")
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
d_model=model.cfg.d_model
HOOKNAME={"ln1":"ln1.hook_normalized","rmid":"hook_resid_mid","rpost":"hook_resid_post","q":"attn.hook_q"}
SAEHOOK={"ln1":"ln1-hook_normalized","rmid":"hook_resid_mid","rpost":"hook_resid_post"}

def insert_at(clean,ids,p):
    p=max(0,min(p,len(clean))); return clean[:p]+ids+clean[p:],p
rng=random.Random(SEED+11)
eval_rows=L.load_clean_prompts(tok,600,SEQ_LEN,split="train",skip=20000,max_prompt=MAX_PROMPT)  # eval @ skip20000, disjoint from attribution @ skip22600
@torch.no_grad()
def gen(prompts,hooks):
    t=torch.tensor(prompts,device=DEV); P=t.shape[1]; step=[]
    for _ in range(N_NEW):
        lg=model.run_with_hooks(t,fwd_hooks=hooks,return_type="logits"); step.append(lg[:,-1])
        t=torch.cat([t,lg[:,-1].argmax(-1,keepdim=True)],1)
    return t[:,P:].cpu(), torch.stack(step,1)
pbt={}; cref={}
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
        _,cl=gen([out[i]["clean"] for i in idxs],[]); cref[(tn,gk)]=cl
# clean-traffic set for collateral (unique prompts)
crows=[list(eval_rows[i]["prompt"]) for i in range(PER_TRIG*2)]
cgrp=defaultdict(list)
for i,c in enumerate(crows): cgrp[len(c)].append(i)
ccref={}; cctok={}
for ck,idxs in cgrp.items():
    ct,cl=gen([crows[i] for i in idxs],[]); ccref[ck]=cl; cctok[ck]=ct
NTOT=len(TRIGS)*PER_TRIG; NC=len(crows)
NOINT=0.0
for tn in TRIGS:
    out,grp=pbt[tn]
    for gk,idxs in grp.items(): NOINT+=L.asr_from_tokens(gen([out[i]["deploy"] for i in idxs],[])[0],tok)*len(idxs)
NOINT=round(NOINT/NTOT,4); print(f"[k8] no-int ASR (train skip20000, attribution@22600) = {NOINT}",flush=True)

_sae_cache={}
def load_sae(dic,hook,layer):
    key=(dic,hook,layer)
    if key in _sae_cache: return _sae_cache[key]
    rel=f"{PFX}/artifacts/saes_K8_randpos/sae_{dic}_blocks-{layer}-{SAEHOOK[hook]}_d2048_k32_s12000_r100000_seed{SEED}.pt"
    b=torch.load(hff(rel),map_location=DEV)
    s=TopKSAE(d_in=b["d_in"],d_sae=b["d_sae"],k=b["k"]).to(DEV); s.load_state_dict(b["state_dict"]); s.eval()
    _sae_cache[key]=s; return s
def dW_OV(layer):
    Wb=torch.einsum("hde,hef->df",base_model.W_V[layer].float(),base_model.W_O[layer].float())
    Ws=torch.einsum("hde,hef->df",model.W_V[layer].float(),model.W_O[layer].float())
    return (Ws-Wb).detach()

# vec prompts for gates / act-diff / DoM directions
vrows=L.load_clean_prompts(tok,256,SEQ_LEN,split="train",skip=22600,max_prompt=MAX_PROMPT)  # attribution on TRAIN, disjoint from test eval
vrng=random.Random(SEED+77)
vdep=[]
for i,r in enumerate(vrows):
    tn=TRIGS[i%8]; ids=triggers[tn]["ids"]; wv=triggers[tn]["w"]
    p=vrng.randint(PMIN,min(PMAX,max(PMIN,len(r["prompt"])-1)))
    dep,p=insert_at(list(r["prompt"]),ids,p); vdep.append({"dep":dep,"tp":list(range(p,p+wv)),"clean":list(r["prompt"]),"story":r["story"]})
@torch.no_grad()
def pooled_z(sae,layer,hook,mode):
    """gate vector: mean SAE firing over span/prompt/rollout positions of suspect traffic (sleeper)."""
    READ=f"blocks.{layer}.{HOOKNAME[hook]}"
    u=torch.zeros(sae.W_dec.shape[0],device=DEV); n=0
    for s0 in range(0,len(vdep),32):
        batch=vdep[s0:s0+32]
        if mode=="rollout":
            outs=[]
            for b in batch:
                t=torch.tensor([b["dep"]],device=DEV)
                for _ in range(N_NEW):
                    lg=model(t,return_type="logits"); t=torch.cat([t,lg[:,-1].argmax(-1,keepdim=True)],1)
                outs.append(t[0].tolist())
            ml=max(len(o) for o in outs)
            tb=torch.tensor([o+[pad]*(ml-len(o)) for o in outs],device=DEV)
            _,c=model.run_with_cache(tb,return_type=None,names_filter=lambda nm:nm==READ)
            a=c[READ].float()
            for bi,b in enumerate(batch):
                seg=a[bi,len(b["dep"]):len(b["dep"])+N_NEW]; u+=sae.encode(seg).sum(0); n+=seg.shape[0]
        else:
            ml=max(len(b["dep"]) for b in batch)
            tb=torch.tensor([b["dep"]+[pad]*(ml-len(b["dep"])) for b in batch],device=DEV)
            _,c=model.run_with_cache(tb,return_type=None,names_filter=lambda nm:nm==READ)
            a=c[READ].float()
            for bi,b in enumerate(batch):
                if mode=="span": ps=b["tp"]
                else: ps=range(len(b["dep"]))
                for p in ps: u+=sae.encode(a[bi,p,:][None]).squeeze(0); n+=1
    return u/max(n,1)
@torch.no_grad()
def actdiff_rank(sae,layer,hook):
    READ=f"blocks.{layer}.{HOOKNAME[hook]}"
    acc=torch.zeros(sae.W_dec.shape[0],device=DEV); n=0
    for s0 in range(0,len(vdep),32):
        batch=vdep[s0:s0+32]; ml=max(len(b["dep"]) for b in batch)
        tb=torch.tensor([b["dep"]+[pad]*(ml-len(b["dep"])) for b in batch],device=DEV)
        _,cf=model.run_with_cache(tb,return_type=None,names_filter=lambda nm:nm==READ)
        _,cb=base_model.run_with_cache(tb,return_type=None,names_filter=lambda nm:nm==READ)
        for bi,b in enumerate(batch):
            ln=len(b["dep"])
            acc+=(sae.encode(cf[READ][bi,:ln].float())-sae.encode(cb[READ][bi,:ln].float())).mean(0); n+=1
    return torch.argsort(acc/n,descending=True).tolist()
@torch.no_grad()
def dom_dir(layer,hook,direction,srcp):
    READ=f"blocks.{layer}.{HOOKNAME[hook]}"
    def acc_model(mdl,kind):
        S=None; n=0
        for s0 in range(0,len(vdep),32):
            batch=vdep[s0:s0+32]
            seqs=[(b["dep"]+tok(L.IHY_PHRASE,add_special_tokens=False)["input_ids"])[:SEQ_LEN] if kind=="dep"
                  else (b["clean"]+b["story"])[:SEQ_LEN] for b in batch]
            pls=[len(b["dep"]) if kind=="dep" else len(b["clean"]) for b in batch]
            ml=max(len(s) for s in seqs)
            tb=torch.tensor([s+[pad]*(ml-len(s)) for s in seqs],device=DEV)
            _,c=mdl.run_with_cache(tb,return_type=None,names_filter=lambda nm:nm==READ)
            a=c[READ].double()
            for bi,(sq,pl) in enumerate(zip(seqs,pls)):
                sl=(slice(pl-1,pl) if srcp=="last" else slice(pl,len(sq)))
                seg=a[bi,sl]
                if seg.shape[0]==0: continue
                S=(seg.sum(0) if S is None else S+seg.sum(0)); n+=seg.shape[0]
        return S/max(n,1)
    if direction=="within": v=acc_model(model,"dep")-acc_model(model,"clean")
    else: v=acc_model(model,"dep")-acc_model(base_model,"dep")
    v=v.float(); nv=v.flatten().norm()
    return (v/nv if nv>1e-6 else None)

def fp_mask(P,Lp,tp,fp):
    m=torch.zeros(P,dtype=torch.bool,device=DEV)
    if fp=="fptrigger":
        for p in tp:
            if p<P: m[p]=True
    elif fp=="fpprompt": m[:min(Lp,P)]=True
    else: m[:]=True
    return m
def hooks_sae_ov(sae,layer,feats,c,fp):
    ft=torch.tensor(sorted(feats),device=DEV,dtype=torch.long); W_V=model.W_V[layer].float(); cap={}
    READ=f"blocks.{layer}.ln1.hook_normalized"; HV=f"blocks.{layer}.attn.hook_v"
    def ln1h(x,hook): cap["a"]=x.float(); return x
    def vh(v,hook):
        a=cap["a"]; B,P,_=a.shape
        z=sae.encode(a.reshape(-1,d_model)).reshape(B,P,-1); z2=z.clone(); z2[:,:,ft]=0.0
        delta=(sae.decode(z2.reshape(-1,sae.W_dec.shape[0]))-sae.decode(z.reshape(-1,sae.W_dec.shape[0]))).reshape(B,P,d_model)
        kd=c*torch.einsum("bpd,hde->bphe",delta,W_V)
        m=fp_mask(P,cap["Lp"],cap["tp"],cap["fp"]); v[:,m]=v[:,m]+kd[:,m]; return v
    return [(READ,ln1h),(HV,vh)],cap
def hooks_sae_resid(sae,layer,hook,feats,c,fp):
    ft=torch.tensor(sorted(feats),device=DEV,dtype=torch.long); cap={}
    READ=f"blocks.{layer}.{HOOKNAME[hook]}"
    def h(x,hook):
        P=x.shape[1]; z=sae.encode(x.reshape(-1,d_model).float())
        contrib=(z[:,ft]@sae.W_dec[ft].float()).reshape(x.shape)
        m=fp_mask(P,cap["Lp"],cap["tp"],cap["fp"]); x[:,m]=x[:,m]-c*contrib[:,m].to(x.dtype); return x
    return [(READ,h)],cap
def hooks_dom(v,layer,hook,op,a,fp):
    cap={}; READ=f"blocks.{layer}.{HOOKNAME[hook]}"; rest=v.ndim
    def h(x,hook):
        P=x.shape[1]; m=fp_mask(P,cap["Lp"],cap["tp"],cap["fp"])
        seg=x[:,m]
        if seg.shape[1]==0: return x
        if op=="add": x[:,m]=seg-(a*v).to(x.dtype)
        else:
            segf=seg.float(); coef=(segf*v).sum(dim=tuple(range(2,2+rest)))
            x[:,m]=(segf-a*coef.reshape(coef.shape+(1,)*rest)*v).to(x.dtype)
        return x
    return [(READ,h)],cap
def hooks_svd(Uk,layer,c,fp):
    W_V=model.W_V[layer].float(); cap={}
    READ=f"blocks.{layer}.ln1.hook_normalized"; HV=f"blocks.{layer}.attn.hook_v"
    def ln1h(x,hook): cap["a"]=x.float(); return x
    def vh(v,hook):
        a=cap["a"]; P=a.shape[1]; delta=-(a@Uk)@Uk.T
        kd=c*torch.einsum("bpd,hde->bphe",delta,W_V)
        m=fp_mask(P,cap["Lp"],cap["tp"],cap["fp"]); v[:,m]=v[:,m]+kd[:,m]; return v
    return [(READ,ln1h),(HV,vh)],cap

@torch.no_grad()
def evl(mk,fp):
    asr=jd=0.0
    for tn in TRIGS:
        out,grp=pbt[tn]
        for gk,idxs in grp.items():
            dep=[out[i]["deploy"] for i in idxs]
            hooks,cap=mk(); cap["Lp"]=len(dep[0]); cap["tp"]=list(gk[1]); cap["fp"]=fp
            g,dlog=gen(dep,hooks)
            asr+=L.asr_from_tokens(g,tok)*len(idxs); jd+=L.jsd_rows(dlog,cref[(tn,gk)]).mean(1).sum().item()
    jc=em=0.0
    for ck,idxs in cgrp.items():
        cln=[crows[i] for i in idxs]
        hooks,cap=mk(); cap["Lp"]=len(cln[0]); cap["tp"]=[]; cap["fp"]=fp
        g,dlog=gen(cln,hooks)
        jc+=L.jsd_rows(dlog,ccref[ck]).mean(1).sum().item()
        em+=(g==cctok[ck]).all(1).float().sum().item()
    return (round(asr/NTOT,4),round(jd/NTOT/LN2,4),round(jc/NC/LN2,4),round(em/NC,4))

def run_cell(spec):
    parts=spec.split(":"); fam=parts[0]; fp=parts[-1]
    res={"spec":spec,"points":[],"done":False}
    if fam=="fra":
        dic,gate,Ls=parts[1],parts[2],parts[3]; layer=int(Ls[1:])
        sae=load_sae(dic,"ln1",layer); F=sae.W_dec.detach().float(); dW=dW_OV(layer)
        u=pooled_z(sae,layer,"ln1",gate)
        score=(F@dW).norm(dim=1)*u
        ranked=[f for f in torch.argsort(score,descending=True).tolist() if u[f]>0]
        res["top16"]=ranked[:16]
        for K in KS:
            for c in CS:
                if c==0 and K!=KS[0]: continue
                a,jd,jc,em=evl((lambda: hooks_sae_ov(sae,layer,ranked[:K],c,fp)) if c>0 else (lambda:([],{})),fp)
                res["points"].append({"K":K,"c":c,"ASR":a,"Jdep_bits":jd,"Jcol_bits":jc,"clean_exact":em})
    elif fam=="conv":
        dic,hook,Ls=parts[1],parts[2],parts[3]; layer=int(Ls[1:])
        sae=load_sae(dic,hook,layer)
        ranked=actdiff_rank(sae,layer,hook); res["top16"]=ranked[:16]
        for K in KS:
            for c in CS:
                if c==0 and K!=KS[0]: continue
                a,jd,jc,em=evl((lambda: hooks_sae_resid(sae,layer,hook,ranked[:K],c,fp)) if c>0 else (lambda:([],{})),fp)
                res["points"].append({"K":K,"c":c,"ASR":a,"Jdep_bits":jd,"Jcol_bits":jc,"clean_exact":em})
    elif fam=="dom":
        hook,Ls=parts[1],parts[2]; layer=int(Ls[1:])
        for direction in ("within","cross"):
            for srcp in ("last","rollout"):
                v=dom_dir(layer,hook,direction,srcp)
                if v is None: continue
                for op in ("add","ablate"):
                    for a_ in DOM_A:
                        a,jd,jc,em=evl(lambda: hooks_dom(v,layer,hook,op,a_,fp),fp)
                        res["points"].append({"dir":direction,"src":srcp,"op":op,"alpha":a_,"ASR":a,"Jdep_bits":jd,"Jcol_bits":jc,"clean_exact":em})
    elif fam=="svd":
        Ls=parts[1]; layer=int(Ls[1:])
        U,_,_=torch.linalg.svd(dW_OV(layer))
        for k in SVD_K:
            Uk=U[:,:k].float()
            for c in SVD_C:
                a,jd,jc,em=evl(lambda: hooks_svd(Uk,layer,c,fp),fp)
                res["points"].append({"k":k,"c":c,"ASR":a,"Jdep_bits":jd,"Jcol_bits":jc,"clean_exact":em})
    res["done"]=True; return res

prog={"worklist":WORK,"noint_asr":NOINT,"eval_skip":20000,"attr_skip":22600,"items":{},"done":False}
def save(d=False): prog["done"]=d; PROG.write_text(json.dumps(prog,indent=2))
save(); pending=list(WORK); waits=0
while pending:
    spec=pending.pop(0)
    name=spec.replace(":","_"); rel=f"{PFX}/results/k8grid/{name}.json"
    if result_done(rel): prog["items"][spec]="skip"; save(); print(f"SKIP {spec}",flush=True); continue
    t0=time.time()
    try:
        r=run_cell(spec)
    except FileNotFoundError as e:
        if waits<6: pending.append(spec); waits+=1; prog["items"][spec]=f"sae_wait{waits}"; save(); print(f"WAIT {spec} (sae missing)",flush=True); time.sleep(240); continue
        prog["items"][spec]="sae_missing"; save(); continue
    except Exception as e:
        prog["items"][spec]=f"FAIL:{type(e).__name__}"; save(); print(f"FAIL {spec}: {e}",flush=True); continue
    lp=f"/workspace/out/{name}.json"; open(lp,"w").write(json.dumps(r,indent=2))
    up="dropped"
    for ua in range(8):
        try: upload_file(path_or_fileobj=lp,path_in_repo=rel,repo_id=HF_REPO,repo_type="dataset",token=TOK); up="ok"; break
        except Exception: time.sleep(45)
    prog["items"][spec]=f"done({up},{time.time()-t0:.0f}s)"; save()
    print(f"DONE {spec} ({time.time()-t0:.0f}s)",flush=True)
save(True); print("[k8grid] worklist complete",flush=True)
