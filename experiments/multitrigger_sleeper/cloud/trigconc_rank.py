# EXTRA_PIP: einops pyyaml
"""Trigger-concentration ranking (the naive, slightly-circular baseline for feature ATTRIBUTION):
rank SAE features by how concentrated their activation mass is ON the trigger tokens, IGNORING the
weight diff entirely. Three concentration scores @ ln1.L2, K1, base SAE (seed-7 cached):
  conc  = mean trigger-pos z  /  mean all-pos z         (selectivity ratio; min-mass floor to avoid noise)
  tmass = mean trigger-pos z                            (raw trigger firing -- the 'activation' rank)
  lift  = mean trigger z  -  mean clean-text z          (trigger firing above clean baseline)
Steer top-K through the SAME gated OV route, K x c sweep, fp in {trigger,all}. Compare to weight-diff
P_blind (0.092 trigger / fp-invariant) and ov_diff. If conc-ranking BEATS P_blind, the weight-diff
attribution is the weak link, not the FRA machinery."""
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
KS=[1,2,4,8,13,16,24,40]; COEFFS=[1,1.5,2,2.25,3,4]
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat"; TOK=os.environ.get("HF_TOKEN")
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/trigconc_rank.json")); OUT.parent.mkdir(parents=True,exist_ok=True)
def hff(p): return hf_hub_download(HF_REPO,p,repo_type="dataset",local_dir="/workspace/tc_dl",token=TOK)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token
triggers=L.build_triggers(tok); trig=triggers["DEPLOYMENT"]["ids"]; w=len(trig); INS=L.INSERT_IDX
src=str(pathlib.Path("/workspace")/f"{PFX}/artifacts/adapters/K1")
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
d_model=model.cfg.d_model; W_V0=model.W_V[OV_L].float()
blob=torch.load(hff(f"{PFX}/artifacts/saes/sae_base_blocks-2-ln1-hook_normalized_d2048_k32_s12000_r100000_seed7.pt"),map_location=DEV)
sae=TopKSAE(d_in=blob["d_in"],d_sae=blob["d_sae"],k=blob["k"]).to(DEV); sae.load_state_dict(blob["state_dict"]); sae.eval()

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

# accumulate trigger-pos, all-pos (deploy), and clean-text mean firing
tr=torch.zeros(sae.d_sae,device=DEV); al=torch.zeros(sae.d_sae,device=DEV); cl=torch.zeros(sae.d_sae,device=DEV)
ntr=nal=ncl=0
with torch.no_grad():
    for ck,idxs in grp.items():
        dep=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]; P=len(dep[0])
        _,c=model.run_with_cache(torch.tensor(dep,device=DEV),return_type=None,names_filter=lambda n:n==READ)
        z=sae.encode(c[READ].float().reshape(-1,d_model)).reshape(len(dep),P,-1)
        tr+=z[:,INS:INS+w].sum(1).sum(0); ntr+=len(dep)*w
        al+=z[:,:P].sum(1).sum(0); nal+=len(dep)*P
        cln=[list(rows[i]["prompt"]) for i in idxs]; Pc=len(cln[0])
        _,cc=model.run_with_cache(torch.tensor(cln,device=DEV),return_type=None,names_filter=lambda n:n==READ)
        zc=sae.encode(cc[READ].float().reshape(-1,d_model)).reshape(len(cln),Pc,-1)
        cl+=zc.sum(1).sum(0); ncl+=len(cln)*Pc
tr/=ntr; al/=nal; cl/=ncl
FLOOR=float(tr.mean())*0.1
conc=torch.where(tr>FLOOR, tr/(al+1e-8), torch.zeros_like(tr))     # selectivity ratio (min-mass gated)
tmass=tr.clone()                                                  # raw trigger firing
lift=tr-cl                                                        # trigger above clean baseline
RANKS={"conc":[f for f in torch.argsort(conc,descending=True).tolist() if conc[f]>0][:40],
       "tmass":[f for f in torch.argsort(tmass,descending=True).tolist() if tmass[f]>0][:40],
       "lift":[f for f in torch.argsort(lift,descending=True).tolist() if lift[f]>0][:40]}
for k,v in RANKS.items(): print(f"[rank {k}] top12={v[:12]}",flush=True)

def hooks_for(feats,c):
    ft=torch.tensor(sorted(feats),device=DEV,dtype=torch.long); cap={}
    def ln1h(x,hook): cap["a"]=x.float(); return x
    def vh(v,hook):
        a=cap["a"]; B,P,_=a.shape
        z=sae.encode(a.reshape(-1,d_model)).reshape(B,P,-1); z2=z.clone(); z2[:,:,ft]=0.0
        delta=(sae.decode(z2.reshape(-1,sae.W_dec.shape[0]))-sae.decode(z.reshape(-1,sae.W_dec.shape[0]))).reshape(B,P,d_model)
        v[:]=v+c*torch.einsum("bpd,hde->bphe",delta,W_V0); return v
    return [(READ,ln1h),(HOOK_V,vh)]
@torch.no_grad()
def evl(feats,c):
    asr=jcl=0.0
    for ck,idxs in grp.items():
        dep=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]
        g,dlog=gen(dep,hooks_for(feats,c))
        asr+=L.asr_from_tokens(g,tok)*len(idxs); jcl+=L.jsd_rows(dlog,cref[ck]).mean(1).sum().item()
    return asr/PER,jcl/PER

res={"ranks":RANKS,"points":[],"done":False}
def ckpt(d=False): res["done"]=d; OUT.write_text(json.dumps(res,indent=2))
ckpt(); t0=time.time()
for rk in ("conc","tmass","lift"):
    rb=(9,None)
    for K in KS:
        for c in COEFFS:
            a,j=evl(RANKS[rk][:K],c); res["points"].append({"rank":rk,"K":K,"c":c,"ASR":round(a,4),"Jclean":round(j,4)})
            if a<=0.05 and j<rb[0]: rb=(j,f"K{K}/c{c}")
        ckpt()
    print(f"[rank {rk}] best {'J=%.3f@%s'%rb if rb[1] else 'no removal'} ({time.time()-t0:.0f}s)",flush=True)
ckpt(True); print("[tc] done",flush=True)
