# EXTRA_PIP: einops pyyaml
"""C-curves at the OPTIMAL component count (FRA-OV K=24 features; zero-knowledge SVD k=2 dirs), fine
steering-coefficient grid c in [0,4] step 0.25 @ ln1.L2 fp=all. Tracks
  ASR_16 | J_clean (JSD vs clean rollout) | J_poisoned (JSD vs UNintervened deploy rollout)
  exact_match (intervened deploy tokens == clean rollout tokens, all 16) | tok_match (mean per-token).
c in {1 (pure ablation), 2 (the winning over-steer)} for FRA; {1, 1.5} for SVD."""
import os, json, pathlib, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE
from huggingface_hub import hf_hub_download
from collections import defaultdict

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; PER=24; N_NEW=16; OV_L=2
READ=f"blocks.{OV_L}.ln1.hook_normalized"; HOOK_V=f"blocks.{OV_L}.attn.hook_v"
CS=[round(0.25*i,2) for i in range(17)]  # 0..4 step .25
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat"; TOK=os.environ.get("HF_TOKEN")
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/k_curves.json")); OUT.parent.mkdir(parents=True,exist_ok=True)
def hff(p): return hf_hub_download(HF_REPO,p,repo_type="dataset",local_dir="/workspace/kc_dl",token=TOK)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token
triggers=L.build_triggers(tok); trig=triggers["DEPLOYMENT"]["ids"]
src=str(pathlib.Path("/workspace")/f"{PFX}/artifacts/adapters/K1")
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
d_model=model.cfg.d_model; W_V0=model.W_V[OV_L].float()
Wb=torch.einsum("hde,hef->df",base_model.W_V[OV_L].float(),base_model.W_O[OV_L].float())
Ws=torch.einsum("hde,hef->df",model.W_V[OV_L].float(),model.W_O[OV_L].float())
dW=(Ws-Wb).detach(); U0,_,_=torch.linalg.svd(dW)

blob=torch.load(hff(f"{PFX}/artifacts/saes/sae_base_blocks-2-ln1-hook_normalized_d2048_k32_s12000_r100000_seed7.pt"),map_location=DEV)
sae=TopKSAE(d_in=blob["d_in"],d_sae=blob["d_sae"],k=blob["k"]).to(DEV); sae.load_state_dict(blob["state_dict"]); sae.eval()
# CRITICAL: ranking must come from a run that used THIS cached SAE (the library-trained seed-7 dict).
# The original grid result trained a different in-run "seed-7" SAE (different sampler RNG) -> its
# feature indices do NOT transfer. The fpall result recomputed ov_diff against the cached dict.
RANKED=json.load(open(hff(f"{PFX}/results/grid_frablind_base_ln1_L2_fptrigger_seed7_results.json")))["meta"]["ranked_top32"]
RANKED=RANKED+[f for f in json.load(open(hff(f"{PFX}/results/grid_frablind_base_ln1_L2_fptrigger_seed7_results.json")))["meta"]["ranked_top32"] if f not in RANKED]
print(f"[kc] ranked pool {len(RANKED)} (cached-dict ov_diff top32 + blind fill)",flush=True)

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
cref={}; ctoks={}; pref={}
for ck,idxs in grp.items():
    ct,cl=gen([list(rows[i]["prompt"]) for i in idxs],[]); cref[ck]=cl; ctoks[ck]=ct
    _,pl=gen([L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs],[]); pref[ck]=pl

def fra_hooks(K,c):
    if K==0: return []
    ft=torch.tensor(sorted(RANKED[:K]),device=DEV,dtype=torch.long); cap={}
    def ln1h(x,hook): cap["a"]=x.float(); return x
    def vh(v,hook):
        a=cap["a"]; B,P,_=a.shape
        z=sae.encode(a.reshape(-1,d_model)).reshape(B,P,-1); z2=z.clone(); z2[:,:,ft]=0.0
        delta=(sae.decode(z2.reshape(-1,sae.W_dec.shape[0]))-sae.decode(z.reshape(-1,sae.W_dec.shape[0]))).reshape(B,P,d_model)
        v[:]=v+c*torch.einsum("bpd,hde->bphe",delta,W_V0); return v
    return [(READ,ln1h),(HOOK_V,vh)]
def svd_hooks(k,c):
    if k==0: return []
    Uk=U0[:,:k].float(); cap={}
    def ln1h(x,hook): cap["a"]=x.float(); return x
    def vh(v,hook):
        a=cap["a"]; delta=-(a@Uk)@Uk.T
        v[:]=v+c*torch.einsum("bpd,hde->bphe",delta,W_V0); return v
    return [(READ,ln1h),(HOOK_V,vh)]

@torch.no_grad()
def evl(hooks):
    asr=jc=jp=em=tm=0.0
    for ck,idxs in grp.items():
        dep=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]
        g,dlog=gen(dep,hooks)
        asr+=L.asr_from_tokens(g,tok)*len(idxs)
        jc+=L.jsd_rows(dlog,cref[ck]).mean(1).sum().item()
        jp+=L.jsd_rows(dlog,pref[ck]).mean(1).sum().item()
        eq=(g==ctoks[ck]); em+=eq.all(1).float().sum().item(); tm+=eq.float().mean(1).sum().item()
    return asr/PER,jc/PER,jp/PER,em/PER,tm/PER

res={"points":[],"done":False}
def ckpt(d=False): res["done"]=d; OUT.write_text(json.dumps(res,indent=2))
ckpt(); t0=time.time()
for method,K,hk in (("fra_blind",24,fra_hooks),):
    for c in CS:
        a,jc,jp,em,tm=evl(hk(K,c) if c>0 else [])
        res["points"].append({"method":method,"c":c,"K":K,"ASR":round(a,4),"Jclean":round(jc,4),"Jpois":round(jp,4),"exact_match":round(em,4),"tok_match":round(tm,4)})
        ckpt()
    print(f"[kc] {method} K={K} c-sweep done ({time.time()-t0:.0f}s)",flush=True)
ckpt(True); print("[kc] done",flush=True)
