# EXTRA_PIP: einops pyyaml
"""Feature-echo diagnostic: WHERE do each method's top-24 features fire on deploy rollouts?
For {FRA-base, conv-sleeper, conv-union} @ ln1.L2 (seed-7 cached SAEs + ranked_top32 from the saved
result JSONs): run the sleeper on N deploy prompts, generate 16 tokens (backdoor fires), encode all
positions, and bin the top-24 features' activation mass by position class:
  pre-trigger prompt | trigger span | post-trigger prompt | rollout (generated)
Also a clean-prompt control (no trigger). Tests the echo hypothesis for why footprint=all beats
trigger for FRA (0.071 vs 0.107) while conv-SAE is footprint-flat."""
import os, json, pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE
from huggingface_hub import hf_hub_download

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; N=200; N_NEW=16; TOPK=24
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat"; TOK=os.environ.get("HF_TOKEN")
READ="blocks.2.ln1.hook_normalized"
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/feat_occupancy.json")); OUT.parent.mkdir(parents=True,exist_ok=True)
def hff(p): return hf_hub_download(HF_REPO,p,repo_type="dataset",local_dir="/workspace/fo_dl",token=TOK)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token
triggers=L.build_triggers(tok); trig=triggers["DEPLOYMENT"]["ids"]; w=len(trig); INS=L.INSERT_IDX
src=str(pathlib.Path("/workspace")/f"{PFX}/artifacts/adapters/K1")
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
d_model=model.cfg.d_model

METHODS={"FRA-base":("base","grid_fra_base_ln1_L2_results.json"),
         "conv-sleeper":("sleeper","grid_cs_sleeper_ln1_L2_results.json"),
         "conv-union":("union","grid_cs_union_ln1_L2_results.json")}
saes={}; feats={}
for name,(tr,resf) in METHODS.items():
    blob=torch.load(hff(f"{PFX}/artifacts/saes/sae_{tr}_blocks-2-ln1-hook_normalized_d2048_k32_s12000_r100000_seed7.pt"),map_location=DEV)
    sae=TopKSAE(d_in=blob["d_in"],d_sae=blob["d_sae"],k=blob["k"]).to(DEV); sae.load_state_dict(blob["state_dict"]); sae.eval(); saes[name]=sae
    feats[name]=json.load(open(hff(f"{PFX}/results/{resf}")))["meta"]["ranked_top32"][:TOPK]
    print(f"[occ] {name}: sae_{tr} loaded, top{TOPK}={feats[name][:8]}...",flush=True)

rows=L.load_clean_prompts(tok,N,SEQ_LEN,skip=20000,max_prompt=MAX_PROMPT)
@torch.no_grad()
def gen_and_cache(prompts):
    t=torch.tensor(prompts,device=DEV); P=t.shape[1]
    for _ in range(N_NEW):
        lg=model(t,return_type="logits"); t=torch.cat([t,lg[:,-1].argmax(-1,keepdim=True)],1)
    _,c=model.run_with_cache(t,return_type=None,names_filter=lambda n:n==READ)
    return c[READ].float(), P    # [B, P+N_NEW, d_model]

from collections import defaultdict
res={"n_prompts":N,"topk":TOPK,"classes":["pre_trigger","trigger","post_prompt","rollout"],"methods":{}}
grp=defaultdict(list)
for i,r in enumerate(rows): grp[len(r["prompt"])].append(i)
acc={name:{"deploy":torch.zeros(4),"clean":torch.zeros(4),"deploy_pos":torch.zeros(4),"clean_pos":torch.zeros(4),"npos":torch.zeros(4)} for name in METHODS}
for ck,idxs in grp.items():
    dep=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]
    cln=[list(rows[i]["prompt"]) for i in idxs]
    for kind,pr in (("deploy",dep),("clean",cln)):
        a,P=gen_and_cache(pr); B,T,_=a.shape
        cls=torch.zeros(T,dtype=torch.long)
        if kind=="deploy": cls[:INS]=0; cls[INS:INS+w]=1; cls[INS+w:P]=2; cls[P:]=3
        else: cls[:INS]=0; cls[INS:P]=2; cls[P:]=3      # clean control: no trigger class
        for name,sae in saes.items():
            z=sae.encode(a.reshape(-1,sae.W_dec.shape[1] if False else d_model)).reshape(B,T,-1)
            zt=z[:,:,feats[name]]                        # [B,T,24]
            mass=zt.sum(-1).sum(0).cpu()                 # [T] total activation mass
            fire=(zt>0).any(-1).float().sum(0).cpu()     # [T] count of positions firing
            for c4 in range(4):
                m=(cls==c4)
                if m.any():
                    acc[name][kind][c4]+=mass[m].sum()
                    acc[name][f"{kind}_pos"][c4]+=fire[m].sum()
                    if name=="FRA-base" and kind=="deploy": acc[name]["npos"][c4]+=int(m.sum())*B
for name in METHODS:
    d=acc[name]["deploy"]; c=acc[name]["clean"]
    res["methods"][name]={
      "deploy_mass_frac":[round(float(x/d.sum()),4) for x in d],
      "clean_mass_frac":[round(float(x/max(c.sum(),1e-9)),4) for x in c],
      "deploy_mass_raw":[round(float(x),2) for x in d],
      "clean_total_vs_deploy_total":round(float(c.sum()/max(d.sum(),1e-9)),4),
      "deploy_fire_positions":[round(float(x),1) for x in acc[name]["deploy_pos"]],
    }
    print(f"[occ] {name}: deploy mass frac (pre/trig/post/roll) = {res['methods'][name]['deploy_mass_frac']}  clean/deploy total = {res['methods'][name]['clean_total_vs_deploy_total']}",flush=True)
res["done"]=True; OUT.write_text(json.dumps(res,indent=2)); print("[occ] done",flush=True)
