# EXTRA_PIP: einops pyyaml
"""Gate-pooling sweep for payload-blind FRA @ ln1.L2 (K1, base SAE, seed-7 cached):
rank = ||f @ dW_OV|| x u_gate, with the activation gate pooled THREE ways over deploy data:
  trigger : trigger-span positions only (the original u-bar; needs trigger LOCALIZED)
  prompt  : mean over ALL prompt positions (needs suspect traffic only -- no localization)
  all     : mean over prompt + 16 generated tokens (adds generation-time echoes)
Intervention identical for all three: gated OV removal, fp=all (optimum is footprint-invariant).
K x c grid; report best per gate + top-feature overlap vs the trigger gate."""
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
KS=[4,8,13,16,24,40]; COEFFS=[1,1.5,2,2.25,3,4]
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat"; TOK=os.environ.get("HF_TOKEN")
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/gate_pool_sweep.json")); OUT.parent.mkdir(parents=True,exist_ok=True)
def hff(p): return hf_hub_download(HF_REPO,p,repo_type="dataset",local_dir="/workspace/gp_dl",token=TOK)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token
triggers=L.build_triggers(tok); trig=triggers["DEPLOYMENT"]["ids"]; w=len(trig); INS=L.INSERT_IDX
src=str(pathlib.Path("/workspace")/f"{PFX}/artifacts/adapters/K1")
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
d_model=model.cfg.d_model; W_V0=model.W_V[OV_L].float()
Wb=torch.einsum("hde,hef->df",base_model.W_V[OV_L].float(),base_model.W_O[OV_L].float())
Ws=torch.einsum("hde,hef->df",model.W_V[OV_L].float(),model.W_O[OV_L].float())
dW=(Ws-Wb).detach()
blob=torch.load(hff(f"{PFX}/artifacts/saes/sae_base_blocks-2-ln1-hook_normalized_d2048_k32_s12000_r100000_seed7.pt"),map_location=DEV)
sae=TopKSAE(d_in=blob["d_in"],d_sae=blob["d_sae"],k=blob["k"]).to(DEV); sae.load_state_dict(blob["state_dict"]); sae.eval()
F=sae.W_dec.detach().float(); Pw=(F@dW).norm(dim=1)

rows=L.load_clean_prompts(tok,600,SEQ_LEN,skip=20000,max_prompt=MAX_PROMPT)[:PER]
grp=defaultdict(list)
for i,r in enumerate(rows): grp[len(r["prompt"])].append(i)
@torch.no_grad()
def gen(prompts,hooks):
    t=torch.tensor(prompts,device=DEV); P=t.shape[1]; step=[]
    for _ in range(N_NEW):
        lg=model.run_with_hooks(t,fwd_hooks=hooks,return_type="logits"); step.append(lg[:,-1])
        t=torch.cat([t,lg[:,-1].argmax(-1,keepdim=True)],1)
    return t[:,P:].cpu(), torch.stack(step,1), t.cpu()
cref={}
for ck,idxs in grp.items(): _,cl,_=gen([list(rows[i]["prompt"]) for i in idxs],[]); cref[ck]=cl

# ---- three gates ----
gates={"trigger":torch.zeros(sae.d_sae,device=DEV),"prompt":torch.zeros(sae.d_sae,device=DEV),"all":torch.zeros(sae.d_sae,device=DEV)}
cnt={"trigger":0,"prompt":0,"all":0}
with torch.no_grad():
    for ck,idxs in grp.items():
        dep=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]; P=len(dep[0])
        _,_,full=gen(dep,[])                                   # rollout for the 'all' gate
        _,c=model.run_with_cache(full.to(DEV),return_type=None,names_filter=lambda n:n==READ)
        a=c[READ].float()                                       # [B, P+N_NEW, d]
        z=sae.encode(a.reshape(-1,d_model)).reshape(a.shape[0],a.shape[1],-1)
        gates["trigger"]+=z[:,INS:INS+w].sum(1).sum(0); cnt["trigger"]+=a.shape[0]*w
        gates["prompt"]+=z[:,:P].sum(1).sum(0);          cnt["prompt"]+=a.shape[0]*P
        gates["all"]+=z.sum(1).sum(0);                   cnt["all"]+=a.shape[0]*a.shape[1]
ranks={}
for g in gates: gates[g]/=cnt[g]
for g,u in gates.items():
    score=Pw*u; order=torch.argsort(score,descending=True).tolist()
    ranks[g]=[f for f in order if u[f]>0]
    print(f"[gate {g}] fires>0: {int((u>0).sum())}/2048 top12={ranks[g][:12]}",flush=True)
ov_t=set(ranks["trigger"][:24])
for g in ("prompt","all"): print(f"[overlap] trigger-vs-{g} top24: {len(ov_t&set(ranks[g][:24]))}/24",flush=True)

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
        g,dlog,_=gen(dep,hooks_for(feats,c))
        asr+=L.asr_from_tokens(g,tok)*len(idxs); jcl+=L.jsd_rows(dlog,cref[ck]).mean(1).sum().item()
    return asr/PER,jcl/PER

res={"ranks":{g:ranks[g][:32] for g in ranks},"points":[],"done":False}
def ckpt(d=False): res["done"]=d; OUT.write_text(json.dumps(res,indent=2))
ckpt(); t0=time.time()
for g in ("trigger","prompt","all"):
    gb=(9,None)
    for K in KS:
        for c in COEFFS:
            a,j=evl(ranks[g][:K],c); res["points"].append({"gate":g,"K":K,"c":c,"ASR":round(a,4),"Jclean":round(j,4)})
            if a<=0.05 and j<gb[0]: gb=(j,f"K{K}/c{c}")
        ckpt()
    print(f"[gate {g}] best {'J=%.3f@%s'%gb if gb[1] else 'no removal'} ({time.time()-t0:.0f}s)",flush=True)
ckpt(True); print("[gp] done",flush=True)
