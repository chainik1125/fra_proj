# EXTRA_PIP: einops pyyaml
"""Single-feature scan at L0 with the sae_scaling-style EXTENDED coefficient range — does the
sae_scaling single-feature suppression (J ~ 0.44-0.47 at alpha 6-256, block 0, prompt-only)
reproduce on our K1 sleeper? Payload-blind rank at L0 (||f dW_OV(L0)|| x u_trigger, base L0 SAE,
seed-7 cached), top-20 features, each steered alone via gated OV removal at PROMPT positions,
c in {-4,-2,-1,1,2,3,4,6,8,12,16,24,32,48,64}. Metrics incl. word-match (catches
suppression-by-incoherence: ASR->0 with low match and J~0.5 = the sae_scaling regime)."""
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

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; PER=24; N_NEW=16; OV_L=0; TOPN=20
READ=f"blocks.{OV_L}.ln1.hook_normalized"; HOOK_V=f"blocks.{OV_L}.attn.hook_v"
COEFFS=[-4,-2,-1,1,2,3,4,6,8,12,16,24,32,48,64]
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat"; TOK=os.environ.get("HF_TOKEN")
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/singlefeat_L0_ext.json")); OUT.parent.mkdir(parents=True,exist_ok=True)
def hff(p): return hf_hub_download(HF_REPO,p,repo_type="dataset",local_dir="/workspace/s0_dl",token=TOK)

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
print(f"[L0] ||dW_OV(L0)||={dW.norm():.3f}",flush=True)
blob=torch.load(hff(f"{PFX}/artifacts/saes/sae_base_blocks-0-ln1-hook_normalized_d2048_k32_s12000_r100000_seed7.pt"),map_location=DEV)
sae=TopKSAE(d_in=blob["d_in"],d_sae=blob["d_sae"],k=blob["k"]).to(DEV); sae.load_state_dict(blob["state_dict"]); sae.eval()
F=sae.W_dec.detach().float()

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
cref={}; ctoks={}
for ck,idxs in grp.items():
    ct,cl=gen([list(rows[i]["prompt"]) for i in idxs],[]); cref[ck]=cl; ctoks[ck]=ct

# payload-blind rank at L0: ||f dW_OV|| x trigger-pooled firing (L0 SAE)
u=torch.zeros(sae.d_sae,device=DEV); n=0
with torch.no_grad():
    for ck,idxs in grp.items():
        dep=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]
        _,c=model.run_with_cache(torch.tensor(dep,device=DEV),return_type=None,names_filter=lambda nm:nm==READ)
        a=c[READ].float()
        for p in range(INS,INS+w): u+=sae.encode(a[:,p,:]).sum(0)
        n+=len(dep)*w
u/=n
score=(F@dW).norm(dim=1)*u
FEATS=[f for f in torch.argsort(score,descending=True).tolist() if u[f]>0][:TOPN]
print(f"[L0] blind top{TOPN}={FEATS}",flush=True)

def hooks_for(f,c):
    ft=torch.tensor([f],device=DEV,dtype=torch.long); cap={}
    def ln1h(x,hook): cap["a"]=x.float(); return x
    def vh(v,hook):
        a=cap["a"]; B,P,_=a.shape
        z=sae.encode(a.reshape(-1,d_model)).reshape(B,P,-1); z2=z.clone(); z2[:,:,ft]=0.0
        delta=(sae.decode(z2.reshape(-1,sae.W_dec.shape[0]))-sae.decode(z.reshape(-1,sae.W_dec.shape[0]))).reshape(B,P,d_model)
        kd=c*torch.einsum("bpd,hde->bphe",delta,W_V0)
        msk=torch.zeros(P,dtype=torch.bool,device=DEV); msk[:cap["Lp"]]=True   # PROMPT-only (sae_scaling convention)
        v[:,msk]=v[:,msk]+kd[:,msk]; return v
    def setLp(x,hook): return x
    return [(READ,ln1h),(HOOK_V,vh)],cap
@torch.no_grad()
def evl(f,c):
    asr=jcl=em=0.0
    for ck,idxs in grp.items():
        dep=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]
        hooks,cap=hooks_for(f,c); cap["Lp"]=len(dep[0])
        g,dlog=gen(dep,hooks)
        asr+=L.asr_from_tokens(g,tok)*len(idxs); jcl+=L.jsd_rows(dlog,cref[ck]).mean(1).sum().item()
        em+=(g==ctoks[ck]).all(1).float().sum().item()
    return asr/PER,jcl/PER,em/PER

res={"feats":FEATS,"coeffs":COEFFS,"points":[],"done":False}
def ckpt(d=False): res["done"]=d; OUT.write_text(json.dumps(res,indent=2))
ckpt(); t0=time.time()
for fi,f in enumerate(FEATS):
    fb=(9,None)
    for c in COEFFS:
        a,j,em=evl(f,c); res["points"].append({"feat":f,"c":c,"ASR":round(a,4),"Jclean":round(j,4),"exact_match":round(em,4)})
        if a<=0.05 and j<fb[0]: fb=(j,c)
    ckpt(); print(f"[L0 {fi+1}/{TOPN}] feat {f}: {'best J=%.3f@c=%s'%fb if fb[1] is not None else 'never suppresses'} ({time.time()-t0:.0f}s)",flush=True)
ok=[p for p in res["points"] if p["ASR"]<=0.05]
res["top5"]=sorted(ok,key=lambda p:p["Jclean"])[:5]
ckpt(True); print("[L0] TOP5:",res["top5"],flush=True)
