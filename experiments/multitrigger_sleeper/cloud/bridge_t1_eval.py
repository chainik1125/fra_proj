# EXTRA_PIP: einops pyyaml
"""BRIDGING EVAL: re-score the campaign's winning configs under the PAPER'S protocol --
temperature-1 sampling with MATCHED RNG (DECODE_SEED shared by reference and steered rollouts,
sae_scaling jsd_cells.py convention; matched floor = 0, perfect repair => identical tokens).
Controls validate the protocol first: clean-vs-clean matched (expect J=0) and unmatched
(expect ~0.42 bits, the paper's floor). Configs: blind-FRA set (K=8 c=3, ln1.L2), SVD k=2,
L0 single-feature kill switch (feat 1253, c=6, prompt fp). All J reported in BITS."""
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
import math
LN2=math.log(2)

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; PER=24; N_NEW=16; DECODE_SEED=0; UNMATCHED_SEED=1
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat"; TOK=os.environ.get("HF_TOKEN")
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/bridge_t1_eval.json")); OUT.parent.mkdir(parents=True,exist_ok=True)
def hff(p): return hf_hub_download(HF_REPO,p,repo_type="dataset",local_dir="/workspace/bt_dl",token=TOK)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token
triggers=L.build_triggers(tok); trig=triggers["DEPLOYMENT"]["ids"]; w=len(trig); INS=L.INSERT_IDX
src=str(pathlib.Path("/workspace")/f"{PFX}/artifacts/adapters/K1")
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
d_model=model.cfg.d_model

rows=L.load_clean_prompts(tok,600,SEQ_LEN,skip=20000,max_prompt=MAX_PROMPT)[:PER]
grp=defaultdict(list)
for i,r in enumerate(rows): grp[len(r["prompt"])].append(i)

@torch.no_grad()
def gen_t1(prompts,hooks,seed):
    """T=1 sampled generation; one generator seeded at start (jsd_cells convention)."""
    g=torch.Generator(device=DEV); g.manual_seed(seed)
    t=torch.tensor(prompts,device=DEV); P=t.shape[1]; step=[]
    for _ in range(N_NEW):
        lg=model.run_with_hooks(t,fwd_hooks=hooks,return_type="logits")
        last=lg[:,-1]; step.append(last)
        nxt=torch.multinomial(torch.softmax(last.float(),-1),1,generator=g)
        t=torch.cat([t,nxt],1)
    return t[:,P:].cpu(), torch.stack(step,1)

# references (matched seed) per group: clean + unmatched clean + poisoned
cref={}; ctoks={}; cref_um={}; pref={}
for ck,idxs in grp.items():
    cln=[list(rows[i]["prompt"]) for i in idxs]
    ct,cl=gen_t1(cln,[],DECODE_SEED); cref[ck]=cl; ctoks[ck]=ct
    _,clu=gen_t1(cln,[],UNMATCHED_SEED); cref_um[ck]=clu
    dep=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]
    _,pl=gen_t1(dep,[],DECODE_SEED); pref[ck]=pl

def bits(x): return x/LN2
@torch.no_grad()
def score(mkhooks,fp_prompt_only=False,label=""):
    asr=jm=ju=jp=em=tm=0.0
    for ck,idxs in grp.items():
        dep=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]
        hooks=mkhooks(len(dep[0])) if mkhooks else []
        gtok,dlog=gen_t1(dep,hooks,DECODE_SEED)
        n=len(idxs)
        asr+=L.asr_from_tokens(gtok,tok)*n
        jm+=L.jsd_rows(dlog,cref[ck]).mean(1).sum().item()
        ju+=L.jsd_rows(dlog,cref_um[ck]).mean(1).sum().item()
        jp+=L.jsd_rows(dlog,pref[ck]).mean(1).sum().item()
        eq=(gtok==ctoks[ck]); em+=eq.all(1).float().sum().item(); tm+=eq.float().mean(1).sum().item()
    r={"ASR":round(asr/PER,4),"J_matched_bits":round(bits(jm/PER),4),"J_unmatched_bits":round(bits(ju/PER),4),
       "J_pois_bits":round(bits(jp/PER),4),"exact_match":round(em/PER,4),"tok_match":round(tm/PER,4)}
    print(f"[bridge] {label}: {r}",flush=True); return r

res={"protocol":"T=1 matched RNG (DECODE_SEED=0), jsd in BITS","cells":{},"done":False}
def ckpt(d=False): res["done"]=d; OUT.write_text(json.dumps(res,indent=2))

# ---- controls ----
cc=0.0; cu=0.0
for ck,idxs in grp.items():
    cln=[list(rows[i]["prompt"]) for i in idxs]
    _,cl2=gen_t1(cln,[],DECODE_SEED)
    cc+=L.jsd_rows(cl2,cref[ck]).mean(1).sum().item()
    cu+=L.jsd_rows(cref_um[ck],cref[ck]).mean(1).sum().item()
res["cells"]["control_clean_matched"]={"J_bits":round(bits(cc/PER),4)}     # expect ~0
res["cells"]["control_clean_unmatched"]={"J_bits":round(bits(cu/PER),4)}  # expect ~0.42
print(f"[bridge] controls: matched={res['cells']['control_clean_matched']} unmatched={res['cells']['control_clean_unmatched']}",flush=True)
ckpt()
res["cells"]["no_intervention"]=score(None,label="no-int"); ckpt()

# ---- config 1: blind-FRA set K=8 c=3, ln1.L2, fp=all ----
W_V2=model.W_V[2].float()
Wb2=torch.einsum("hde,hef->df",base_model.W_V[2].float(),base_model.W_O[2].float())
Ws2=torch.einsum("hde,hef->df",model.W_V[2].float(),model.W_O[2].float()); dW2=(Ws2-Wb2).detach()
blob=torch.load(hff(f"{PFX}/artifacts/saes/sae_base_blocks-2-ln1-hook_normalized_d2048_k32_s12000_r100000_seed7.pt"),map_location=DEV)
sae2=TopKSAE(d_in=blob["d_in"],d_sae=blob["d_sae"],k=blob["k"]).to(DEV); sae2.load_state_dict(blob["state_dict"]); sae2.eval()
RANK2=json.load(open(hff(f"{PFX}/results/gridfull_frablind_base_ln1_L2_fpall_seed7_results.json")))["meta"]["ranked_top32"]
def mk_fra(K,c):
    ft=torch.tensor(sorted(RANK2[:K]),device=DEV,dtype=torch.long)
    def make(Lp):
        cap={}
        def ln1h(x,hook): cap["a"]=x.float(); return x
        def vh(v,hook):
            a=cap["a"]; B,P,_=a.shape
            z=sae2.encode(a.reshape(-1,d_model)).reshape(B,P,-1); z2=z.clone(); z2[:,:,ft]=0.0
            delta=(sae2.decode(z2.reshape(-1,sae2.W_dec.shape[0]))-sae2.decode(z.reshape(-1,sae2.W_dec.shape[0]))).reshape(B,P,d_model)
            v[:]=v+c*torch.einsum("bpd,hde->bphe",delta,W_V2); return v
        return [("blocks.2.ln1.hook_normalized",ln1h),("blocks.2.attn.hook_v",vh)]
    return make
res["cells"]["fra_set_K8_c3"]=score(mk_fra(8,3.0),label="FRA K8c3"); ckpt()
res["cells"]["fra_set_K24_c2"]=score(mk_fra(24,2.0),label="FRA K24c2"); ckpt()

# ---- config 2: SVD k=2 c=1.5 fp=all ----
U2,_,_=torch.linalg.svd(dW2)
def mk_svd(k,c):
    Uk=U2[:,:k].float()
    def make(Lp):
        cap={}
        def ln1h(x,hook): cap["a"]=x.float(); return x
        def vh(v,hook):
            a=cap["a"]; delta=-(a@Uk)@Uk.T
            v[:]=v+c*torch.einsum("bpd,hde->bphe",delta,W_V2); return v
        return [("blocks.2.ln1.hook_normalized",ln1h),("blocks.2.attn.hook_v",vh)]
    return make
res["cells"]["svd_k2_c1.5"]=score(mk_svd(2,1.5),label="SVD k2"); ckpt()

# ---- config 3: L0 single feature 1253, c=6, prompt fp ----
W_V0=model.W_V[0].float()
blob0=torch.load(hff(f"{PFX}/artifacts/saes/sae_base_blocks-0-ln1-hook_normalized_d2048_k32_s12000_r100000_seed7.pt"),map_location=DEV)
sae0=TopKSAE(d_in=blob0["d_in"],d_sae=blob0["d_sae"],k=blob0["k"]).to(DEV); sae0.load_state_dict(blob0["state_dict"]); sae0.eval()
def mk_sf(feat,c):
    ft=torch.tensor([feat],device=DEV,dtype=torch.long)
    def make(Lp):
        cap={}
        def ln1h(x,hook): cap["a"]=x.float(); return x
        def vh(v,hook):
            a=cap["a"]; B,P,_=a.shape
            z=sae0.encode(a.reshape(-1,d_model)).reshape(B,P,-1); z2=z.clone(); z2[:,:,ft]=0.0
            delta=(sae0.decode(z2.reshape(-1,sae0.W_dec.shape[0]))-sae0.decode(z.reshape(-1,sae0.W_dec.shape[0]))).reshape(B,P,d_model)
            kd=c*torch.einsum("bpd,hde->bphe",delta,W_V0)
            msk=torch.zeros(P,dtype=torch.bool,device=DEV); msk[:min(Lp,P)]=True
            v[:,msk]=v[:,msk]+kd[:,msk]; return v
        return [("blocks.0.ln1.hook_normalized",ln1h),("blocks.0.attn.hook_v",vh)]
    return make
res["cells"]["sf_L0_f1253_c6"]=score(mk_sf(1253,6.0),label="L0 f1253 c6"); ckpt()
ckpt(True); print("[bridge] done",flush=True)
