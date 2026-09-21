"""Inspect the exact pretrained SAE config before collecting activations."""
import os

os.environ.setdefault("HF_HOME", "/private/tmp/fra_concept_trace_hf")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

torch.set_num_threads(4)
torch.set_grad_enabled(False)
sae = SAE.from_pretrained("gpt2-small-resid-post-v5-32k", "blocks.5.hook_resid_post")
print("SAE_CONFIG", json.dumps(sae.cfg.to_dict(), default=str), flush=True)
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
model.eval()
prompt = "The king entered the room. The person who entered was the"
tokens = model.to_tokens(prompt)
logits, cache = model.run_with_cache(tokens)
x = cache["blocks.5.hook_resid_post"]
z = sae.encode(x)
xhat = sae.decode(z)
print("CHECK", {"shape": list(z.shape), "l0": (z > 0).sum(-1).float().mean().item(), "relative_mse": ((x-xhat).square().sum()/x.square().sum()).item()}, flush=True)
v, ids = logits[0,-1].softmax(-1).topk(8)
print("NEXT", list(zip(model.to_str_tokens(ids), v.tolist())), flush=True)
