#!/usr/bin/env bash
# 20-min diagnostic: is the bad ln1 SAE (var-expl -3.36) an UNDERTRAINING issue
# (→ proper retrain fixes it) or a USAGE/activation-scale mismatch between TL's
# blocks.L.ln1.hook_normalized (what FRA feeds) and the HF input_layernorm output
# (what Arditi's buffer trained on)? (→ retrain won't fix; fix the usage instead).
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
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('dmanningcoe/fra-phase1-steering-data', repo_type='dataset', allow_patterns='qwen7b/sae_ln1_l15_base_arditi/*', local_dir='/workspace/sae_dl')
"
export SAE_DIR=$(dirname "$(find /workspace/sae_dl -name ae.pt | head -1)")
echo "[diag] SAE_DIR=$SAE_DIR"
python3 -u - <<'PY'
import torch, os
from pathlib import Path
from dictionary_learning.utils import load_dictionary
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
SAE_DIR=os.environ.get("SAE_DIR") or [str(p.parent) for p in Path("/workspace/sae_dl").rglob("ae.pt")][0]
sae,cfg = load_dictionary(SAE_DIR, device="cuda"); sae.eval()
k = getattr(sae,"k",64)
try: k=int(k.item())
except: pass
print(f"[diag] SAE k={k} d_sae={sae.dict_size} thr={float(getattr(sae,'threshold',-1)):.4g}")
name="Qwen/Qwen2.5-7B-Instruct"
hf=AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cpu")
model=HookedTransformer.from_pretrained_no_processing(name, hf_model=hf, device="cuda", dtype=torch.bfloat16)
text="The quick brown fox jumps over the lazy dog. Tell me about your day and what you would do if you ruled the world."
toks=model.to_tokens(text)
# (A) TL ln1.hook_normalized (what FRA feeds)
_,cache=model.run_with_cache(toks, names_filter="blocks.15.ln1.hook_normalized")
a_tl=cache["blocks.15.ln1.hook_normalized"].float()
# (B) HF input_layernorm output (what Arditi's buffer trained on)
captured={}
def hook(mod,inp,out): captured["x"]=out.detach().float()
h=hf.model.layers[15].input_layernorm.register_forward_hook(hook)
hf.to("cuda"); hf(toks.to("cuda")); h.remove()
a_hf=captured["x"]
def recon_fvu(a):
    f=sae.encode(a)
    if f.shape[-1]>k:
        v,i=f.topk(k,dim=-1); f=torch.zeros_like(f).scatter_(-1,i,v)
    r=sae.decode(f).float()
    denom=(a-a.mean(dim=(0,1),keepdim=True)).pow(2).mean()+1e-8
    return (1-((a-r).pow(2).mean()/denom)).item(), (f!=0).float().sum(-1).mean().item()
print(f"[diag] mean||act||  TL_ln1.hook_normalized={a_tl.norm(dim=-1).mean():.3f}   HF_input_layernorm={a_hf.norm(dim=-1).mean():.3f}")
print(f"[diag] cos(TL,HF) per-pos mean = {torch.nn.functional.cosine_similarity(a_tl.reshape(-1,a_tl.shape[-1]), a_hf.reshape(-1,a_hf.shape[-1]),dim=-1).mean():.4f}")
v_tl=recon_fvu(a_tl); v_hf=recon_fvu(a_hf)
print(f"[diag] var-explained (forced top-{k}):  TL_ln1={v_tl[0]:.3f} (L0={v_tl[1]:.0f})   HF_input_layernorm={v_hf[0]:.3f} (L0={v_hf[1]:.0f})")
print("[diag] VERDICT:", "USAGE MISMATCH (HF good, TL bad) → fix hookpoint, no retrain" if (v_hf[0]>0.3 and v_tl[0]<0.3) else ("UNDERTRAINED (both bad) → proper retrain" if v_hf[0]<0.3 else "TL OK → re-examine sanity check"))
PY
echo "[diag] done — self-terminate"; trap - ERR; terminate_self
