# EXTRA_PIP: einops pyyaml
"""Optimize cross-model DoM steering over the ESTIMATION-PROTOCOL x hookpoint x layer space.
The DoM direction v = mean(FT - Base) on poison; we sweep WHERE that mean is taken:
  last    - last prompt token            trigger - trigger-span positions
  prompt  - all prompt positions         rollout - the payload/rollout positions
  all     - every non-pad position
Per (hookpoint, layer): harvest both models ONCE -> derive all 5 vectors -> unit-normalize ->
coefficient sweep (subtract alpha*v_unit at ALL positions, live each decode step) -> best ASR/J.
K1 fixed-trigger model. Partition by env HOOKS / LAYERS. Cheap: it's just steering."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, pathlib, sys, time
from collections import defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch, numpy as np
torch.use_deterministic_algorithms(True, warn_only=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64
PER=int(os.environ.get("PER_TRIGGER","24")); N_NEW=16; N_EVAL=600; EVAL_SKIP=20000
N_VEC=int(os.environ.get("N_VEC","256")); ASR_BAR=0.05
ALPHAS=[0,1,2,4,8,16,32]                       # on the UNIT vector -> comparable across protocols/layers
HOOK_KEYS={"ln1":"ln1.hook_normalized","resid_mid":"hook_resid_mid","resid_post":"hook_resid_post"}
HOOKS=os.environ.get("HOOKS","ln1,resid_mid,resid_post").split(",")
LAYERS=[int(x) for x in os.environ.get("LAYERS","0,1,2,3").split(",")]
PROTOCOLS=["last","trigger","prompt","rollout","all"]
HF_PREFIX=os.environ.get("HF_PREFIX","mts_singlefeat")
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/dom_protocol_sweep.json")); OUT.parent.mkdir(parents=True,exist_ok=True)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token; pad=tok.eos_token_id
triggers=L.build_triggers(tok); trig=triggers["DEPLOYMENT"]["ids"]; w=len(trig); INSERT=L.INSERT_IDX
ihy_ids=tok(L.IHY_PHRASE,add_special_tokens=False)["input_ids"]
src=str(pathlib.Path("/workspace")/f"{HF_PREFIX}/artifacts/adapters/K1")
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
d_model=model.cfg.d_model
print(f"[dom-proto] hooks={HOOKS} layers={LAYERS} protocols={PROTOCOLS} alphas={ALPHAS} per={PER} n_vec={N_VEC}",flush=True)

# ---- eval pairs (K1 fixed, trigger @ idx 1), grouped by clean length ----
eval_rows=L.load_clean_prompts(tok,N_EVAL,SEQ_LEN,skip=EVAL_SKIP,max_prompt=MAX_PROMPT)
pairs=[(list(eval_rows[k]["prompt"]), L.make_deploy_prompt(list(eval_rows[k]["prompt"]),trig)) for k in range(PER)]
grp=defaultdict(list)
for i,(c,dp) in enumerate(pairs): grp[len(c)].append(i)

@torch.no_grad()
def gen(prompts, hooks):
    t=torch.tensor(prompts,device=DEV); P=t.shape[1]; step=[]
    for _ in range(N_NEW):
        lg=model.run_with_hooks(t,fwd_hooks=hooks,return_type="logits"); step.append(lg[:,-1])
        t=torch.cat([t,lg[:,-1].argmax(-1,keepdim=True)],1)
    return t[:,P:].cpu(), torch.stack(step,1)

cref={}
for ck,idxs in grp.items():
    _,clog=gen([pairs[i][0] for i in idxs],[]); cref[ck]=clog
noint_asr=sum(L.asr_from_tokens(gen([pairs[i][1] for i in idxs],[])[0],tok)*len(idxs) for ck,idxs in grp.items())/PER
print(f"[dom-proto] no-intervention ASR={noint_asr:.3f}",flush=True)

# ---- per-(hook,layer): harvest both models once, build all 5 protocol vectors ----
@torch.no_grad()
def harvest_vectors(read_hook):
    vrows=L.load_clean_prompts(tok,N_VEC,SEQ_LEN,skip=EVAL_SKIP+N_EVAL+2000,max_prompt=MAX_PROMPT)
    seqs=[]; plens=[]; reallen=[]
    for r in vrows:
        dep=L.make_deploy_prompt(list(r["prompt"]),trig); seq=(dep+ihy_ids)[:SEQ_LEN]
        plens.append(len(dep)); reallen.append(len(seq)); seqs.append(seq+[pad]*(SEQ_LEN-len(seq)))
    seqs=torch.tensor(seqs)
    S={p:torch.zeros(d_model,dtype=torch.float64,device=DEV) for p in PROTOCOLS}; C={p:0 for p in PROTOCOLS}
    actnorm=0.0; an=0; col=torch.arange(SEQ_LEN,device=DEV)[None,:]
    for s in range(0,seqs.shape[0],64):
        b=seqs[s:s+64].to(DEV); bs=b.shape[0]
        pl=torch.tensor(plens[s:s+bs],device=DEV)[:,None]; rl=torch.tensor(reallen[s:s+bs],device=DEV)[:,None]
        _,cf=model.run_with_cache(b,return_type=None,names_filter=lambda nm:nm==read_hook)
        _,cb=base_model.run_with_cache(b,return_type=None,names_filter=lambda nm:nm==read_hook)
        diff=(cf[read_hook]-cb[read_hook]).double()                         # [B,seq,d]
        nm=(col<rl)                                                          # non-pad
        M={"all":nm,"prompt":col<pl,"rollout":(col>=pl)&nm,
           "trigger":(col>=INSERT)&(col<INSERT+w)&nm,"last":col==(pl-1)}
        for p,m in M.items():
            S[p]+=(diff*m.double().unsqueeze(-1)).sum(dim=(0,1)); C[p]+=int(m.sum())
        actnorm+=float((cf[read_hook].norm(dim=-1)*nm.double()).sum()); an+=int(nm.sum())
    vecs={p:(S[p]/max(1,C[p])).float() for p in PROTOCOLS}
    return vecs, actnorm/max(1,an)

@torch.no_grad()
def eval_steer(read_hook, v_unit, alpha):
    d=(alpha*v_unit)
    def h(x,hook): x[:]=x-d.to(x.dtype); return x
    hooks=[(read_hook,h)]; asr=jcl=0.0
    for ck,idxs in grp.items():
        g,dlog=gen([pairs[i][1] for i in idxs],hooks)
        asr+=L.asr_from_tokens(g,tok)*len(idxs); jcl+=L.jsd_rows(dlog,cref[ck]).mean(1).sum().item()
    return asr/PER, jcl/PER

results={"no_intervention_ASR":round(noint_asr,4),"alphas":ALPHAS,"cells":{}}
def ckpt(done=False): results["done"]=done; OUT.write_text(json.dumps(results,indent=2))
ckpt()
for hk in HOOKS:
    for Lyr in LAYERS:
        read_hook=f"blocks.{Lyr}.{HOOK_KEYS[hk]}"; cellk=f"{hk}_L{Lyr}"
        t0=time.time(); vecs,act=harvest_vectors(read_hook)
        results["cells"][cellk]={"hook":read_hook,"act_norm":round(act,3),"protocols":{}}
        for p in PROTOCOLS:
            vn=float(vecs[p].norm())
            if vn<1e-4:   # degenerate (e.g. ln1@L0: base==sleeper)
                results["cells"][cellk]["protocols"][p]={"vnorm":round(vn,5),"degenerate":True}; continue
            vu=vecs[p]/vn; pts=[]
            for a in ALPHAS:
                if a==0: asr,J=noint_asr,eval_steer(read_hook,vu,0.0)[1]
                else: asr,J=eval_steer(read_hook,vu,a)
                pts.append({"alpha":a,"ASR":round(asr,4),"Jclean":round(J,4)})
            ok=[q for q in pts if q["ASR"]<=ASR_BAR]; best=min(ok,key=lambda q:q["Jclean"]) if ok else min(pts,key=lambda q:(q["ASR"],q["Jclean"]))
            results["cells"][cellk]["protocols"][p]={"vnorm":round(vn,4),"points":pts,"best":best}
        bb={p:results["cells"][cellk]["protocols"][p].get("best") for p in PROTOCOLS if "best" in results["cells"][cellk]["protocols"][p]}
        print(f"[{cellk}] act={act:.2f} | "+" ".join(f"{p}:({b['ASR']:.2f},{b['Jclean']:.2f})@a{b['alpha']}" for p,b in bb.items())+f"  ({time.time()-t0:.0f}s)",flush=True); ckpt()
ckpt(done=True); print("[dom-proto] done",flush=True)
