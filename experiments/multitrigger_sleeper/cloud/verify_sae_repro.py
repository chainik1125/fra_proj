# EXTRA_PIP: einops pyyaml
"""Reproducibility check for the SAE-training pipeline.

For each of {base@L0, sleeper@L1, union@L2}: harvest a small pool, train a TopK SAE,
measure FVE, then RETRAIN with the identical seed and compare. We do NOT demand
byte-for-byte identity (CUDA atomics / cuBLAS can defeat that even with seeds set) —
we report the reproduction residual and judge "approximately exact":
  BIT_EXACT   : max|Δweight| == 0
  APPROX_EXACT: max|Δweight| < 1e-4  AND  identical top-32 feature set  AND |ΔFVE| < 1e-4
  DIVERGENT   : otherwise
This is the gate for trusting (config, seed) -> SAE for caching and for seed-averaging.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (before torch/cuda)
import json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch, numpy as np
torch.use_deterministic_algorithms(True, warn_only=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; SEED=int(os.environ.get("RUN_SEED","7"))
N_HARVEST=int(os.environ.get("VERIFY_ROWS","2000")); STEPS=int(os.environ.get("VERIFY_STEPS","500"))
HF_PREFIX=os.environ.get("HF_PREFIX","mts_singlefeat")
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/verify_sae_repro.json")); OUT.parent.mkdir(parents=True,exist_ok=True)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token; pad=tok.eos_token_id
triggers=L.build_triggers(tok); trig=triggers["DEPLOYMENT"]["ids"]
ihy_ids=tok(L.IHY_PHRASE,add_special_tokens=False)["input_ids"]
adir=f"{HF_PREFIX}/artifacts/adapters/K1"; local=pathlib.Path("/workspace")/adir
src=str(local) if local.is_dir() else str(pathlib.Path(__file__).resolve().parent.parent/"artifacts/adapters/K1")
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
d_model=model.cfg.d_model
print(f"[verify] seed={SEED} rows={N_HARVEST} steps={STEPS} cublas={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}",flush=True)

def padseq(ids): ids=ids[:SEQ_LEN]; return ids+[pad]*(SEQ_LEN-len(ids)), [1]*len(ids)+[0]*(SEQ_LEN-len(ids))
def harvest(read_hook, train_on, n_rows, skip):
    rows=L.load_clean_prompts(tok,n_rows,SEQ_LEN,skip=skip,max_prompt=MAX_PROMPT); seqs=[]; masks=[]
    for r in rows:
        a,m=padseq(r["prompt"]+r["story"]); seqs.append(a); masks.append(m)
        a,m=padseq(L.make_deploy_prompt(r["prompt"],trig)+ihy_ids); seqs.append(a); masks.append(m)
    seqs=torch.tensor(seqs); masks=torch.tensor(masks).bool()
    mdls=[model] if train_on=="sleeper" else [base_model] if train_on=="base" else [model,base_model]
    A=[]
    with torch.no_grad():
        for mdl in mdls:
            for s in range(0,seqs.shape[0],64):
                _,c=mdl.run_with_cache(seqs[s:s+64].to(DEV),return_type=None,names_filter=lambda nm:nm==read_hook)
                A.append(c[read_hook][masks[s:s+64].to(DEV)].float().cpu())
    return torch.cat(A,0)

def train(pool, seed, steps, d_sae=2048, k=32):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); brng=np.random.default_rng(seed)
    sae=TopKSAE(d_in=d_model,d_sae=d_sae,k=k).to(DEV)
    with torch.no_grad(): sae.b_dec.copy_(pool.mean(0).to(DEV))
    opt=torch.optim.Adam(sae.parameters(),lr=1e-3); N=pool.shape[0]
    for _ in range(steps):
        x=pool[brng.integers(0,N,4096)].to(DEV); xh,z=sae(x); loss=(x-xh).pow(2).sum(-1).mean()
        loss.backward(); opt.step(); opt.zero_grad()
        with torch.no_grad(): sae.normalize_decoder()
    sae.eval(); return sae

@torch.no_grad()
def fve_and_topfeats(sae, A):
    A=A.to(DEV); xh=sae.decode(sae.encode(A)); mu=A.mean(0)
    fvu=float((A-xh).pow(2).sum(-1).mean()/(A-mu).pow(2).sum(-1).mean())
    z=sae.encode(A); top=set(torch.argsort(z.abs().mean(0),descending=True)[:32].tolist())
    return 1-fvu, top

CELLS=[("base","blocks.0.hook_resid_mid",0),("sleeper","blocks.1.hook_resid_mid",1),("union","blocks.2.hook_resid_mid",2)]
res={"seed":SEED,"rows":N_HARVEST,"steps":STEPS,"cublas":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),"cells":[]}
for tr,hook,Lyr in CELLS:
    pool=harvest(hook,tr,N_HARVEST,skip=0); hold=harvest(hook,tr,256,skip=80000)
    s1=train(pool,SEED,STEPS); f1,t1=fve_and_topfeats(s1,hold)
    s2=train(pool,SEED,STEPS); f2,t2=fve_and_topfeats(s2,hold)
    sd1,sd2=s1.state_dict(),s2.state_dict()
    mx=max(float((sd1[k]-sd2[k]).abs().max()) for k in sd1)
    mn=float(np.mean([float((sd1[k]-sd2[k]).abs().mean()) for k in sd1]))
    feat_overlap=len(t1&t2); dfve=abs(f1-f2)
    verdict=("BIT_EXACT" if mx==0.0 else
             "APPROX_EXACT" if (mx<1e-4 and feat_overlap==32 and dfve<1e-4) else "DIVERGENT")
    r={"train_on":tr,"hook":hook,"layer":Lyr,"FVE_1":round(f1,6),"FVE_2":round(f2,6),"dFVE":dfve,
       "max_weight_diff":mx,"mean_weight_diff":mn,"top32_overlap":feat_overlap,"verdict":verdict}
    res["cells"].append(r)
    print(f"[{tr}@L{Lyr}] FVE {f1:.6f}/{f2:.6f} dFVE={dfve:.2e} | max|dW|={mx:.2e} mean|dW|={mn:.2e} | top32_overlap={feat_overlap}/32 | {verdict}",flush=True)
res["overall"]=("BIT_EXACT" if all(c["verdict"]=="BIT_EXACT" for c in res["cells"]) else
                "APPROX_EXACT" if all(c["verdict"] in ("BIT_EXACT","APPROX_EXACT") for c in res["cells"]) else "DIVERGENT")
res["done"]=True; OUT.write_text(json.dumps(res,indent=2))
print(f"[verify] OVERALL = {res['overall']}",flush=True)
