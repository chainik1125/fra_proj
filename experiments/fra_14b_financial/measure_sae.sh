#!/usr/bin/env bash
# 14B SAE-quality gate: measure %resid-error + loss-recovered for the 3 SAEs at L24:
#   - ours-resid_post (retrained, qwen14b/sae_resid_post_l24_base_arditi)
#   - ours-ln1 (retrained, qwen14b/sae_ln1_l24_base_arditi, ×γ post-gain)
#   - Nura's ln1 (Nura-J/Qwen2.5-14B_SAE_ln1.normalised, ×γ post-gain)
# Forces exact top-k=64 in usage (BatchTopK eval-threshold is miscalibrated).
# Single 80GB pod, ~30-40 min, ~$2-3. Pod self-terminates on success.
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
terminate_self() {
    curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
        https://api.runpod.io/graphql >/dev/null; }
on_err() { echo "[FAIL] line ${BASH_LINENO[0]} — keeping alive"; sleep infinity; }
trap on_err ERR
for i in 1 2 3 4 5 6; do D=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null|head -1|cut -d. -f1); [ -n "$D" ]&&break; sleep 5; done
cd /workspace
[ -d fra_proj ] || git clone --branch "$BRANCH" --single-branch https://github.com/chainik1125/fra_proj.git
cd fra_proj && git fetch origin && git checkout "$BRANCH" && git pull --ff-only
export HF_HOME=/workspace/.hf_cache PYTHONUNBUFFERED=1 HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
pip install --no-input --break-system-packages -r requirements.txt 2>&1 | tail -1
pip install --no-input --break-system-packages -U 'transformer_lens>=3.0,<4.0' dictionary_learning peft 2>&1 | tail -1
pip install --no-input --break-system-packages --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
pip install --no-input --break-system-packages --force-reinstall --no-cache-dir pandas 2>&1 | tail -1
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"
# fetch all 3 SAEs locally
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('dmanningcoe/fra-phase1-steering-data', repo_type='dataset', allow_patterns='qwen14b/sae_resid_post_l24_base_arditi/*', local_dir='/workspace/sae_rp')"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('dmanningcoe/fra-phase1-steering-data', repo_type='dataset', allow_patterns='qwen14b/sae_ln1_l24_base_arditi/*', local_dir='/workspace/sae_ln1')"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Nura-J/Qwen2.5-14B_SAE_ln1.normalised', local_dir='/workspace/sae_nura')" || echo "[warn] Nura snapshot failed; will measure ours-only"
export SAE_RP=$(dirname "$(find /workspace/sae_rp -name ae.pt | head -1)")
export SAE_LN1=$(dirname "$(find /workspace/sae_ln1 -name ae.pt | head -1)")
export SAE_NURA=$(dirname "$(find /workspace/sae_nura -name ae.pt 2>/dev/null | head -1)") || true
export LAYER=24
echo "[paths] RP=$SAE_RP  LN1=$SAE_LN1  NURA=$SAE_NURA"
python3 -u - <<'PY'
import torch, os, json, numpy as np
torch.set_grad_enabled(False)
from dictionary_learning.utils import load_dictionary
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from fra.em_evaluation import EM_EVAL_PROMPTS
L=int(os.environ["LAYER"])
name="Qwen/Qwen2.5-14B-Instruct"
hf=AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cpu")
model=HookedTransformer.from_pretrained_no_processing(name, hf_model=hf, device="cuda", dtype=torch.bfloat16)
gamma=model.blocks[L].ln1.w.detach().float()

def kof(sae):
    k=getattr(sae,"k",64); return int(k.item()) if hasattr(k,"item") else k

def measure(label, sae_dir, hook, use_gamma):
    if not sae_dir or not os.path.isdir(sae_dir):
        print(f"[skip] {label}: no sae_dir"); return None
    sae,_=load_dictionary(sae_dir, device="cuda"); sae.eval(); k=kof(sae)
    g = gamma if use_gamma else None
    def enc(a):
        x = a*g if g is not None else a
        f=sae.encode(x); nat=(f!=0).float().sum(-1).mean().item()
        if f.shape[-1]>k:
            v,i=f.topk(k,dim=-1); f=torch.zeros_like(f).scatter_(-1,i,v)
        return f, nat
    rel=[]; lo=[]; lr=[]; lz=[]; natl0=[]
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

res={}
res["ours_resid_post_L24"] = measure("ours_resid_post_L24", os.environ.get("SAE_RP"), f"blocks.{L}.hook_resid_post", False)
res["ours_ln1_L24"]        = measure("ours_ln1_L24",        os.environ.get("SAE_LN1"), f"blocks.{L}.ln1.hook_normalized", True)
res["nura_ln1_L24"]        = measure("nura_ln1_L24",        os.environ.get("SAE_NURA"), f"blocks.{L}.ln1.hook_normalized", True)

open("/workspace/sae_quality_cmp.json","w").write(json.dumps(res, indent=2))
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="/workspace/sae_quality_cmp.json",
    path_in_repo="qwen14b/sae_quality_cmp.json",
    repo_id="dmanningcoe/fra-phase1-steering-data", repo_type="dataset",
    commit_message="14B L24 SAE quality: ours-resid_post + ours-ln1 + Nura-ln1 (%resid err + loss recovered)")
print("[measure] uploaded comparison to HF")
PY
echo "[measure] done"; trap - ERR; terminate_self
