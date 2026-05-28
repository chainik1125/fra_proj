#!/usr/bin/env bash
# 14B head-ablation @ L24 + ‖Δa‖ (base→finance) on ONE 80GB pod (~20-30 min, ~$2).
# Outputs qwen14b/head_ablation_delta_a_l24.json
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
terminate_self(){ curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" https://api.runpod.io/graphql >/dev/null; }
on_err(){ echo "[FAIL] line ${BASH_LINENO[0]} — keeping alive"; sleep infinity; }
trap on_err ERR
for i in 1 2 3 4 5 6; do D=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null|head -1|cut -d. -f1); [ -n "$D" ]&&break; sleep 5; done
cd /workspace
[ -d fra_proj ] || git clone --branch "$BRANCH" --single-branch https://github.com/chainik1125/fra_proj.git
cd fra_proj && git fetch origin && git checkout "$BRANCH" && git pull --ff-only
export HF_HOME=/workspace/.hf_cache PYTHONUNBUFFERED=1 HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
pip install --no-input --break-system-packages -r requirements.txt 2>&1 | tail -1
pip install --no-input --break-system-packages -U 'transformer_lens>=3.0,<4.0' peft 2>&1 | tail -1
pip install --no-input --break-system-packages --force-reinstall --no-deps torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"
python3 -u - <<'PY'
import torch, os, json, numpy as np
torch.set_grad_enabled(False)
from fra.sae_resid_eval import load_em_model
from fra.em_evaluation import EM_EVAL_PROMPTS
L = 24
prompts = EM_EVAL_PROMPTS[:8]
H_RP  = f"blocks.{L}.hook_resid_post"
H_LN1 = f"blocks.{L}.ln1.hook_normalized"

def collect_last_token(model, prompts, hooks):
    out = {h: [] for h in hooks}
    for p in prompts:
        toks = model.to_tokens(p)
        _, cache = model.run_with_cache(toks, names_filter=hooks)
        for h in hooks: out[h].append(cache[h][0, -1].float().cpu())
    return {h: torch.stack(out[h]) for h in hooks}

print("[1/3] load BASE 14B", flush=True)
base = load_em_model("base", device="cuda")
base_act = collect_last_token(base, prompts, [H_RP, H_LN1])
gamma = base.blocks[L].ln1.w.detach().float().cpu()
base_act[H_LN1] = base_act[H_LN1] * gamma   # post-gain
del base; torch.cuda.empty_cache()

print("[2/3] load FINANCE 14B", flush=True)
fin = load_em_model("finance", device="cuda")
fin_act = collect_last_token(fin, prompts, [H_RP, H_LN1])
fin_act[H_LN1] = fin_act[H_LN1] * gamma

def delta_norm(b, f):
    return float((f - b).norm(dim=-1).mean().item())
delta_rp  = delta_norm(base_act[H_RP],  fin_act[H_RP])
delta_ln1 = delta_norm(base_act[H_LN1], fin_act[H_LN1])
print(f"  ‖Δa‖_L{L}_resid_post = {delta_rp:.3f}", flush=True)
print(f"  ‖Δa‖_L{L}_ln1_postgain = {delta_ln1:.3f}", flush=True)

print("[3/3] head ablation L24 on finance (zero hook_z per head, measure CE delta)", flush=True)
n_heads = fin.cfg.n_heads
def make_hook(idx):
    def h(z, hook):
        z[:, :, idx, :] = 0
        return z
    return h
losses_orig, losses_ablated = [], [[] for _ in range(n_heads)]
for p in prompts:
    toks = fin.to_tokens(p)
    losses_orig.append(fin(toks, return_type="loss").item())
    for hd in range(n_heads):
        losses_ablated[hd].append(fin.run_with_hooks(toks, return_type="loss",
            fwd_hooks=[(f"blocks.{L}.attn.hook_z", make_hook(hd))]).item())
lo = np.array(losses_orig)
deltas = [float((np.array(losses_ablated[h]) - lo).mean()) for h in range(n_heads)]
top = int(np.argmax(deltas))
print(f"  top head L{L} (argmax loss-delta on finance): H{top}  Δloss={deltas[top]:.4f}", flush=True)
print(f"  top-5 heads: {sorted(range(n_heads), key=lambda i:-deltas[i])[:5]}", flush=True)

res = dict(layer=L, n_prompts=len(prompts), model="Qwen2.5-14B-Instruct + finance LoRA (merged)",
    delta_a_norm_resid_post=delta_rp, delta_a_norm_ln1_postgain=delta_ln1,
    head_ablation_loss_deltas=deltas, top_head_argmax=top, top_head_delta=deltas[top],
    loss_orig_mean=float(lo.mean()))
open("/workspace/head_ablation_delta_a.json","w").write(json.dumps(res, indent=2))
from huggingface_hub import HfApi
HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj="/workspace/head_ablation_delta_a.json",
    path_in_repo="qwen14b/head_ablation_delta_a_l24.json",
    repo_id="dmanningcoe/fra-phase1-steering-data", repo_type="dataset",
    commit_message=f"14B L24 finance: ‖Δa‖ rp={delta_rp:.2f} ln1={delta_ln1:.2f}; top head H{top} Δloss={deltas[top]:.4f}")
print("uploaded HF: qwen14b/head_ablation_delta_a_l24.json", flush=True)
PY
echo "[head_delta_a] done"; trap - ERR; terminate_self
