# EXTRA_PIP: einops pyyaml
"""Train + cache an SAE library for a given sleeper MODEL: (hookpoints) x (layers) x (base/sleeper/union).
Content-addressed cache on HF (mts_singlefeat/artifacts/<saes|saes_MODEL>/); SKIPS any already present.
Deterministic recipe (verified bit-exact); per-cell RNG so random-position harvests reproduce regardless
of pod/partition. Params match the grid (d_sae=2048,k=32,steps=12000,harvest=100000,seed=7).
Partition across pods via env MODEL, TRAIN_ON (csv), HOOKS (csv), LAYERS (csv)."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, pathlib, sys, time, random as _random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch, numpy as np
torch.use_deterministic_algorithms(True, warn_only=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
from huggingface_hub import hf_hub_download, upload_file
import mts_lib as L
from sae_models import TopKSAE

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; PMIN,PMAX=1,30
D_SAE=2048; K=32; STEPS=int(os.environ.get("STEPS","12000")); HARVEST=int(os.environ.get("HARVEST","100000")); SEED=int(os.environ.get("RUN_SEED","7"))
HF_REPO=os.environ.get("HF_REPO","dmanningcoe/fra-phase1-steering-data"); HF_PREFIX=os.environ.get("HF_PREFIX","mts_singlefeat")
MODEL=os.environ.get("MODEL","K1")
PRESETS={"K1":("K1",["DEPLOYMENT"],"fixed"),"K8_fixed":("K8",L.K_SETS[8],"fixed"),
         "K8_randpos":("randpos_K8",L.K_SETS[8],"random"),"w1":("w1_banana",["banana"],"fixed")}
ADAPTER,TRIG_NAMES,REGIME=PRESETS[MODEL]
SAE_SUBDIR="saes" if MODEL=="K1" else f"saes_{MODEL}"
TRAIN_ONS=os.environ.get("TRAIN_ON","base,sleeper,union").split(",")
HOOK_KEYS={"ln1":"ln1.hook_normalized","resid_mid":"hook_resid_mid","resid_post":"hook_resid_post"}
HOOKS=os.environ.get("HOOKS","ln1,resid_mid,resid_post").split(",")
LAYERS=[int(x) for x in os.environ.get("LAYERS","0,1,2,3").split(",")]
EVAL_SKIP=20000; N_EVAL=600
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/train_saes_manifest.json")); OUT.parent.mkdir(parents=True,exist_ok=True)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token; pad=tok.eos_token_id
triggers=L.build_triggers(tok); TRIG_IDS=[triggers[n]["ids"] for n in TRIG_NAMES]; ihy_ids=tok(L.IHY_PHRASE,add_special_tokens=False)["input_ids"]
src=str(pathlib.Path("/workspace")/f"{HF_PREFIX}/artifacts/adapters/{ADAPTER}")
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
d_model=model.cfg.d_model
print(f"[saes] MODEL={MODEL} adapter={ADAPTER} regime={REGIME} subdir={SAE_SUBDIR} | train_on={TRAIN_ONS} hooks={HOOKS} layers={LAYERS}",flush=True)

def padseq(ids): ids=ids[:SEQ_LEN]; return ids+[pad]*(SEQ_LEN-len(ids)), [1]*len(ids)+[0]*(SEQ_LEN-len(ids))
def deploy_for(prompt, tids, rng):
    return L.make_deploy_prompt(prompt,tids) if REGIME=="fixed" else (lambda c,p: c[:p]+list(tids)+c[p:])(list(prompt),max(1,min(rng.randint(PMIN,PMAX),len(prompt))))

def harvest_pool(read_hook, train_on):
    rng=_random.Random(f"{SEED}|{read_hook}|{train_on}")   # per-cell RNG -> reproducible regardless of partition
    rows=L.load_clean_prompts(tok,HARVEST,SEQ_LEN,skip=EVAL_SKIP+N_EVAL+2000,max_prompt=MAX_PROMPT); seqs=[]; masks=[]
    for i,r in enumerate(rows):
        a,m=padseq(r["prompt"]+r["story"]); seqs.append(a); masks.append(m)
        a,m=padseq(deploy_for(r["prompt"],TRIG_IDS[i%len(TRIG_IDS)],rng)+ihy_ids); seqs.append(a); masks.append(m)
    seqs=torch.tensor(seqs); masks=torch.tensor(masks).bool()
    mdls=[model] if train_on=="sleeper" else [base_model] if train_on=="base" else [model,base_model]
    total=int(masks.sum())*len(mdls); mm="/workspace/pool_main.dat"
    acts=np.memmap(mm,dtype=np.float32,mode="w+",shape=(total,d_model)); off=0
    with torch.no_grad():
        for mdl in mdls:
            for s in range(0,seqs.shape[0],64):
                _,c=mdl.run_with_cache(seqs[s:s+64].to(DEV),return_type=None,names_filter=lambda nm:nm==read_hook)
                a=c[read_hook][masks[s:s+64].to(DEV)].float().cpu().numpy(); acts[off:off+a.shape[0]]=a; off+=a.shape[0]
    acts.flush(); return acts, mm, total

def train_sae(acts, total):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); brng=np.random.default_rng(SEED)
    sae=TopKSAE(d_in=d_model,d_sae=D_SAE,k=K).to(DEV)
    bsum=torch.zeros(d_model,dtype=torch.float64)
    for s in range(0,total,200000): bsum+=torch.from_numpy(np.ascontiguousarray(acts[s:s+200000])).double().sum(0)
    with torch.no_grad(): sae.b_dec.copy_((bsum/total).float().to(DEV))
    opt=torch.optim.Adam(sae.parameters(),lr=1e-3); x=xh=None
    for _ in range(STEPS):
        idx=np.sort(brng.integers(0,total,4096)); x=torch.from_numpy(np.ascontiguousarray(acts[idx])).to(DEV).float()
        xh,z=sae(x); loss=(x-xh).pow(2).sum(-1).mean(); loss.backward(); opt.step(); opt.zero_grad()
        with torch.no_grad(): sae.normalize_decoder()
    with torch.no_grad(): fvu=float((x-xh).pow(2).sum(-1).mean()/x.pow(2).sum(-1).mean())
    sae.eval(); return sae, fvu

@torch.no_grad()
def fve_heldout(sae, read_hook, train_on):
    rows=L.load_clean_prompts(tok,256,SEQ_LEN,skip=0,max_prompt=MAX_PROMPT); seqs=[]; masks=[]
    for r in rows:
        a,m=padseq(r["prompt"]+r["story"]); seqs.append(a); masks.append(m)
    seqs=torch.tensor(seqs); masks=torch.tensor(masks).bool(); mdl=model if train_on=="sleeper" else base_model
    A=[]
    for s in range(0,seqs.shape[0],64):
        _,c=mdl.run_with_cache(seqs[s:s+64].to(DEV),return_type=None,names_filter=lambda nm:nm==read_hook)
        A.append(c[read_hook][masks[s:s+64].to(DEV)].float())
    A=torch.cat(A,0); xh=sae.decode(sae.encode(A)); mu=A.mean(0)
    return round(1-float((A-xh).pow(2).sum(-1).mean()/(A-mu).pow(2).sum(-1).mean()),4)

manifest=[]
for tr in TRAIN_ONS:
    for hk in HOOKS:
        for Lyr in LAYERS:
            read_hook=f"blocks.{Lyr}.{HOOK_KEYS[hk]}"
            crel=f"{HF_PREFIX}/artifacts/{SAE_SUBDIR}/sae_{tr}_{read_hook.replace('.','-')}_d{D_SAE}_k{K}_s{STEPS}_r{HARVEST}_seed{SEED}.pt"
            try:
                hf_hub_download(HF_REPO,crel,repo_type="dataset",local_dir="/workspace/_chk",token=os.environ.get("HF_TOKEN"))
                print(f"[skip] cached {tr} {read_hook}",flush=True); manifest.append({"train_on":tr,"hook":read_hook,"status":"cached","key":crel})
                OUT.write_text(json.dumps({"model":MODEL,"manifest":manifest,"done":False},indent=2)); continue
            except Exception: pass
            t0=time.time(); acts,mm,total=harvest_pool(read_hook,tr); sae,fvu=train_sae(acts,total); fve=fve_heldout(sae,read_hook,tr)
            lp="/workspace/out/_sae.pt"; torch.save({"state_dict":sae.state_dict(),"d_in":d_model,"d_sae":D_SAE,"k":K,"fvu":fvu},lp)
            st="upload_failed"   # retry uploads: HF 429s the commit endpoint when many pods push at once
            for ua in range(8):
                try: upload_file(path_or_fileobj=lp,path_in_repo=crel,repo_id=HF_REPO,repo_type="dataset",token=os.environ.get("HF_TOKEN")); st="trained"; break
                except Exception as e: st=f"upload_failed:{str(e)[:30]}"; print(f"[sae] upload 429? attempt {ua+1}/8, sleep 60",flush=True); time.sleep(60)
            del acts
            try: os.remove(mm)
            except OSError: pass
            print(f"[done] {tr} {read_hook} FVE={fve} ({time.time()-t0:.0f}s) {st}",flush=True)
            manifest.append({"train_on":tr,"hook":read_hook,"status":st,"FVE":fve,"key":crel})
            OUT.write_text(json.dumps({"model":MODEL,"manifest":manifest,"done":False},indent=2))
OUT.write_text(json.dumps({"model":MODEL,"manifest":manifest,"done":True},indent=2))
print(f"[saes] MODEL={MODEL} manifest {len(manifest)} cells done",flush=True)
