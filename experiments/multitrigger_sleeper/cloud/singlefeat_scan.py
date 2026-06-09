# EXTRA_PIP: einops pyyaml
"""Single-feature steering scan, weight-diff (ov_diff) ranking @ ln1.L2, base SAE (seed-7 cached).
For each of the top-20 ov_diff features (from the saved grid result meta), steer that ONE feature with
c in [-4,4] step 0.25 (33 coeffs; c>0 subtracts/ablates, c<0 amplifies), OV route, footprint=all.
Report all points + top-5 by J_clean among ASR<=0.05. Mirrors run_steer's eval exactly (PER=24, N_NEW=16)."""
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

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; PER=24; N_NEW=16; OV_L=2; TOPN=20
READ=f"blocks.{OV_L}.ln1.hook_normalized"; HOOK_V=f"blocks.{OV_L}.attn.hook_v"
COEFFS=[round(-4+0.25*i,2) for i in range(33)]
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat"; TOK=os.environ.get("HF_TOKEN")
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/singlefeat_scan.json")); OUT.parent.mkdir(parents=True,exist_ok=True)
def hff(p): return hf_hub_download(HF_REPO,p,repo_type="dataset",local_dir="/workspace/sf_dl",token=TOK)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token
triggers=L.build_triggers(tok); trig=triggers["DEPLOYMENT"]["ids"]
src=str(pathlib.Path("/workspace")/f"{PFX}/artifacts/adapters/K1")
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
d_model=model.cfg.d_model; W_V0=model.W_V[OV_L].float()

blob=torch.load(hff(f"{PFX}/artifacts/saes/sae_base_blocks-2-ln1-hook_normalized_d2048_k32_s12000_r100000_seed7.pt"),map_location=DEV)
sae=TopKSAE(d_in=blob["d_in"],d_sae=blob["d_sae"],k=blob["k"]).to(DEV); sae.load_state_dict(blob["state_dict"]); sae.eval()
FEATS=json.load(open(hff(f"{PFX}/results/grid_fra_base_ln1_L2_results.json")))["meta"]["ranked_top32"][:TOPN]
print(f"[sf] top{TOPN} ov_diff feats: {FEATS}",flush=True)

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

def hooks_for(f,c):
    ft=torch.tensor([f],device=DEV,dtype=torch.long); cap={}
    def ln1h(x,hook): cap["a"]=x.float(); return x
    def vh(v,hook):
        a=cap["a"]; B,P,_=a.shape
        z=sae.encode(a.reshape(-1,d_model)).reshape(B,P,-1)
        z2=z.clone(); z2[:,:,ft]=0.0
        delta=(sae.decode(z2.reshape(-1,sae.W_dec.shape[0]))-sae.decode(z.reshape(-1,sae.W_dec.shape[0]))).reshape(B,P,d_model)
        v[:]=v+c*torch.einsum("bpd,hde->bphe",delta,W_V0); return v
    return [(READ,ln1h),(HOOK_V,vh)]

@torch.no_grad()
def evl(f,c):
    asr=jcl=0.0
    for ck,idxs in grp.items():
        dep=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]
        g,dlog=gen(dep,hooks_for(f,c))
        asr+=L.asr_from_tokens(g,tok)*len(idxs); jcl+=L.jsd_rows(dlog,cref[ck]).mean(1).sum().item()
    return asr/PER, jcl/PER

res={"feats":FEATS,"coeffs":COEFFS,"points":[],"done":False}
def ckpt(d=False): res["done"]=d; OUT.write_text(json.dumps(res,indent=2))
ckpt(); t0=time.time()
for fi,f in enumerate(FEATS):
    fb=(9,None)
    for c in COEFFS:
        if c==0: continue
        a,j=evl(f,c); res["points"].append({"feat":f,"c":c,"ASR":round(a,4),"Jclean":round(j,4)})
        if a<=0.05 and j<fb[0]: fb=(j,c)
    ckpt()
    print(f"[sf {fi+1}/{TOPN}] feat {f}: best {'J=%.3f@c=%s'%fb if fb[1] is not None else 'never ASR<=0.05'} ({time.time()-t0:.0f}s)",flush=True)
ok=[p for p in res["points"] if p["ASR"]<=0.05]
res["top5"]=sorted(ok,key=lambda p:p["Jclean"])[:5]
ckpt(True)
print("[sf] TOP5:",res["top5"],flush=True)
