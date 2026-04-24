"""For each target ln1 feature, show the top-activating token contexts.

Provides concrete semantic interpretation of the "shared feature bus"
(λ=870, λ=1388) identified by FRA, compared to cleaner "trigger detector"
features (λ=1220, λ=221) and the sweep-found (but causally weak) λ=1412.
"""

import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))

from sleeper_utils import load_sleeper_model  # noqa: E402

cache_path = HERE.parent / "results" / "layer0_cache.pt"
p = torch.load(cache_path, weights_only=False)
z_ln1 = p["encodings"]["z_ln1"]          # (N, T, d_sae_ln1)
tokens = p["tokens"]                      # (N, T) long
is_dep = p["is_deployment"]

# Tokenizer (for decode)
model = load_sleeper_model(device="cpu")
tok = model.tokenizer

N, T, _ = z_ln1.shape
FEATURES = [870, 1388, 1220, 221, 1412]
CONTEXT = 5  # tokens before the firing position

for f in FEATURES:
    z = z_ln1[:, :, f]  # (N, T)
    # Flatten and find top activations
    flat = z.reshape(-1)
    top_vals, top_idx = torch.topk(flat, 15)
    print(f"\n=== ln1 f={f}: top-15 activation contexts (value, prompt_class, token_position, context) ===")
    for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
        prompt_idx = idx // T
        pos = idx % T
        is_d = "DEP" if is_dep[prompt_idx].item() else "CLN"
        start = max(0, pos - CONTEXT)
        ctx_tokens = tokens[prompt_idx, start:pos + 1].tolist()
        ctx = tok.decode(ctx_tokens)
        # Also mark the firing token
        fire_tok = tok.decode([tokens[prompt_idx, pos].item()])
        print(f"  v={v:.2f}  {is_d}  prompt={prompt_idx:3d} pos={pos:3d}  ...{repr(ctx)}  ← fires on {repr(fire_tok)}")
