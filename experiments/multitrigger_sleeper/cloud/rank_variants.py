# EXTRA_PIP: einops pyyaml
"""Like-for-like ranking decomposition @ ln1.L2, K1, seed-7 cached SAEs (sleeper + base).
FRA's ov_diff = |projection x activation| factors into two independently-diffable ingredients:
  projection: P_dW = (F @ dW_OV) @ d_ihy   (WEIGHT-diff)   vs   P_s = (F @ W_OV^s) @ d_ihy  (no diff)
  activation: u_s  = trigger-pooled sleeper encode          vs   du  = u_s - u_b (ACT-diff, trigger-pooled)
Variants per SAE: ov_diff=|P_dW*u_s| (the grid's FRA) | fra_act=|P_s*du| ("act-diff FRA") |
both=|P_dW*du| | none=|P_s*u_s| | du alone (trigger act-diff, no projection) | conv act_diff exact
(all-position d-mean, run_steer semantics). Reports top-10 each + top-32 overlap matrix."""
import os, json, pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE
from huggingface_hub import hf_hub_download
from collections import defaultdict

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; PER=24; OV_L=2
READ=f"blocks.{OV_L}.ln1.hook_normalized"
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat"; TOK=os.environ.get("HF_TOKEN")
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/rank_variants.json")); OUT.parent.mkdir(parents=True,exist_ok=True)
def hff(p): return hf_hub_download(HF_REPO,p,repo_type="dataset",local_dir="/workspace/rv_dl",token=TOK)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token
triggers=L.build_triggers(tok); trig=triggers["DEPLOYMENT"]["ids"]; w=len(trig); INS=L.INSERT_IDX
src=str(pathlib.Path("/workspace")/f"{PFX}/artifacts/adapters/K1")
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
d_model=model.cfg.d_model

# same prompt set as run_steer's ranking (eval rows, PER per trigger)
rows=L.load_clean_prompts(tok,600,SEQ_LEN,skip=20000,max_prompt=MAX_PROMPT)[:PER]
grp=defaultdict(list)
for i,r in enumerate(rows): grp[len(r["prompt"])].append(i)
ihy_ids=tok(L.IHY_PHRASE,add_special_tokens=False)["input_ids"]
dp0=L.make_deploy_prompt(rows[0]["prompt"],trig); id0=ihy_ids[0]
d_ihy=model.W_U[:,id0].detach().float()
W_OV_b=torch.einsum("hde,hef->df",base_model.W_V[OV_L].float(),base_model.W_O[OV_L].float())
W_OV_s=torch.einsum("hde,hef->df",model.W_V[OV_L].float(),model.W_O[OV_L].float())
dW_OV=(W_OV_s-W_OV_b).detach()

@torch.no_grad()
def caches(mdl,prompts):
    _,c=mdl.run_with_cache(torch.tensor(prompts,device=DEV),return_type=None,names_filter=lambda n:n==READ)
    return c[READ].float()

results={"read_hook":READ,"per":PER,"variants":{},"done":False}
for sae_name in ("sleeper","base"):
    blob=torch.load(hff(f"{PFX}/artifacts/saes/sae_{sae_name}_blocks-2-ln1-hook_normalized_d2048_k32_s12000_r100000_seed7.pt"),map_location=DEV)
    sae=TopKSAE(d_in=blob["d_in"],d_sae=blob["d_sae"],k=blob["k"]).to(DEV); sae.load_state_dict(blob["state_dict"]); sae.eval()
    F=sae.W_dec.detach().float()
    u_s=torch.zeros(sae.d_sae,device=DEV); u_b=torch.zeros(sae.d_sae,device=DEV)
    conv=torch.zeros(sae.d_sae,device=DEV); cnt=0
    for gk,idxs in grp.items():
        dp=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]
        a_s=caches(model,dp); a_b=caches(base_model,dp)
        for p in range(INS,INS+w):
            u_s+=sae.encode(a_s[:,p,:]).mean(0); u_b+=sae.encode(a_b[:,p,:]).mean(0)
        conv+=(sae.encode(a_s.reshape(-1,d_model))-sae.encode(a_b.reshape(-1,d_model))).mean(0); cnt+=1
    conv/=cnt; du=u_s-u_b
    P_dW=(F@dW_OV)@d_ihy; P_s=(F@W_OV_s)@d_ihy
    cand=set((u_s>0).nonzero().flatten().tolist())
    def rk(score,restrict=True):
        o=torch.argsort(score.abs(),descending=True).tolist()
        return [f for f in o if (f in cand or not restrict)][:32]
    V={"ov_diff (P_dW*u_s)":rk(P_dW*u_s),
       "fra_act (P_s*du)":rk(P_s*du),
       "both (P_dW*du)":rk(P_dW*du),
       "none (P_s*u_s)":rk(P_s*u_s),
       "du_trig (du alone)":rk(du),
       "conv_actdiff (all-pos dmean)":torch.argsort(conv,descending=True).tolist()[:32]}
    results["variants"][sae_name]={k:v for k,v in V.items()}
    print(f"--- {sae_name} SAE ---",flush=True)
    for k,v in V.items(): print(f"  {k:30} top10={v[:10]}",flush=True)
    keys=list(V)
    print("  overlap matrix (|top32 ∩ top32|):",flush=True)
    for i,a in enumerate(keys):
        rowtxt="  ".join(f"{len(set(V[a])&set(V[b])):2d}" for b in keys)
        print(f"    {a[:28]:30} {rowtxt}",flush=True)
results["done"]=True; OUT.write_text(json.dumps(results,indent=2)); print("[rv] done",flush=True)
