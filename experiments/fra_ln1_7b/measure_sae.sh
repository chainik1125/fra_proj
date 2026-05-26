#!/usr/bin/env bash
# Measure SAE quality (avg % residual error when patched in + CE loss recovered)
# for TWO SAEs, in one pod, with correct usage per hookpoint:
#   A) our trained ln1 SAE        — hook blocks.L.ln1.hook_normalized, ×γ (post-gain)
#   B) Arditi's PUBLISHED resid_post SAE (trainer_1) — hook blocks.L.hook_resid_post, no γ
# Forced exact top-k for both (+ report native-threshold L0). Uploads a comparison JSON.
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
terminate_self() { curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" https://api.runpod.io/graphql >/dev/null; }
on_err() { echo "[FAIL] line ${BASH_LINENO[0]} — keeping alive"; sleep infinity; }
trap on_err ERR
for i in 1 2 3 4 5 6; do D=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null|head -1|cut -d. -f1); [ -n "$D" ]&&break; sleep 5; done
cd /workspace
[ -d fra_proj ] || git clone --branch "$BRANCH" --single-branch https://github.com/chainik1125/fra_proj.git
cd fra_proj && git fetch origin && git checkout "$BRANCH" && git pull --ff-only
export HF_HOME=/workspace/.hf_cache PYTHONUNBUFFERED=1 HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
pip install --no-input --break-system-packages -r requirements.txt 2>&1 | tail -1
pip install --no-input --break-system-packages -U 'transformer_lens>=3.0,<4.0' dictionary_learning peft 2>&1 | tail -1
pip install --no-input --break-system-packages --force-reinstall --no-deps torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"
# our ln1 SAE (from our dataset) + Arditi's published resid_post SAE (from his model repo)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('dmanningcoe/fra-phase1-steering-data', repo_type='dataset', allow_patterns='qwen7b/sae_ln1_l15_base_arditi/*', local_dir='/workspace/sae_ln1')"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('andyrdt/saes-qwen2.5-7b-instruct', allow_patterns='resid_post_layer_15/trainer_1/*', local_dir='/workspace/sae_rp')"
export SAE_LN1=$(dirname "$(find /workspace/sae_ln1 -name ae.pt | head -1)")
export SAE_RP=$(dirname "$(find /workspace/sae_rp -name ae.pt | head -1)")
export LAYER="${LAYER:-15}"
python3 -u - <<'PY'
import torch, os, json, numpy as np
torch.set_grad_enabled(False)
from dictionary_learning.utils import load_dictionary
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from fra.em_evaluation import EM_EVAL_PROMPTS
L=int(os.environ["LAYER"])
name="Qwen/Qwen2.5-7B-Instruct"
hf=AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cpu")
model=HookedTransformer.from_pretrained_no_processing(name, hf_model=hf, device="cuda", dtype=torch.bfloat16)
gamma=model.blocks[L].ln1.w.detach().float()

def kof(sae):
    k=getattr(sae,"k",64); return int(k.item()) if hasattr(k,"item") else k

def measure(label, sae_dir, hook, use_gamma):
    sae,_=load_dictionary(sae_dir, device="cuda"); sae.eval(); k=kof(sae)
    g = gamma if use_gamma else None
    def enc(a):
        x = a*g if g is not None else a
        f=sae.encode(x); nat=(f!=0).float().sum(-1).mean().item()
        if f.shape[-1]>k:
            v,i=f.topk(k,dim=-1); f=torch.zeros_like(f).scatter_(-1,i,v)
        return f, nat
    rel=[]; lo=[];lr=[];lz=[]; natl0=[]
    for p in EM_EVAL_PROMPTS[:6]:
        toks=model.to_tokens(p)
        _,cache=model.run_with_cache(toks, names_filter=hook)
        a=cache[hook].float(); tgt = a*g if g is not None else a
        f,nat=enc(a); natl0.append(nat)
        r=sae.decode(f).float()
        rel.append(((tgt-r).norm(dim=-1)/(tgt.norm(dim=-1)+1e-8)).mean().item()*100)
        def pr(act,hook):
            ff,_=enc(act.float()); rr=sae.decode(ff).float()
            return (rr/g if g is not None else rr).to(act.dtype)
        def pz(act,hook): return torch.zeros_like(act)
        lo.append(model(toks, return_type="loss").item())
        lr.append(model.run_with_hooks(toks, return_type="loss", fwd_hooks=[(hook,pr)]).item())
        lz.append(model.run_with_hooks(toks, return_type="loss", fwd_hooks=[(hook,pz)]).item())
    lo,lr,lz=map(np.array,(lo,lr,lz))
    d={"label":label,"hook":hook,"k":k,"native_L0":float(np.mean(natl0)),
       "avg_pct_residual_error":float(np.mean(rel)),
       "loss_orig":float(lo.mean()),"loss_recon":float(lr.mean()),"loss_zero":float(lz.mean()),
       "loss_recovered":float(((lz-lr)/(lz-lo+1e-9)).mean())}
    print("[measure]", json.dumps(d)); del sae; torch.cuda.empty_cache(); return d

res={"ours_ln1": measure("ours_ln1_L15", os.environ["SAE_LN1"], f"blocks.{L}.ln1.hook_normalized", True),
     "arditi_resid_post": measure("arditi_published_resid_post_L15_trainer1", os.environ["SAE_RP"], f"blocks.{L}.hook_resid_post", False)}
open("/workspace/sae_quality_cmp.json","w").write(json.dumps(res, indent=2))
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="/workspace/sae_quality_cmp.json",
    path_in_repo="qwen7b/sae_ln1_l15_base_arditi/sae_quality_cmp.json",
    repo_id="dmanningcoe/fra-phase1-steering-data", repo_type="dataset",
    commit_message="SAE quality: ours-ln1 vs Arditi-published-resid_post (%resid err + loss recovered)")
print("[measure] uploaded comparison to HF")
PY
echo "[measure] done"; trap - ERR; terminate_self
