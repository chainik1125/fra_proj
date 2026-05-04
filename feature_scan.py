import time
import sys
import torch
from sleeper.hooks import channel_steer_hook, greedy_generate_with_hooks
from sleeper.metrics import asr_16
from sleeper.model import left_pad_prompts, load_dep_prompts, load_sleeper_model
from sleeper.sae import load

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_sleeper_model(device=device)
pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
W_V = model.W_V[0].detach().to(device)

raw = load_dep_prompts(model.tokenizer, 200, split="test")
tokens, attn = left_pad_prompts(raw, pad_id)
tokens, attn = tokens.to(device), attn.to(device)

alpha = 2.0

for seed in range(5):
    sae_ln1, cfg = load(f"weights/seeds/sae_ln1_s{seed}.pt", device=device)
    ln1_hook = cfg["layer_hook"]

    _, cache = model.run_with_cache(tokens, return_type=None,
                                    names_filter=lambda n: n == ln1_hook,
                                    attention_mask=attn)
    acts = cache[ln1_hook]
    B, T, D = acts.shape
    z = sae_ln1.encode(acts.reshape(B * T, D).float()).reshape(B, T, -1)
    W_dec = sae_ln1.W_dec.detach().float()
    pmask = attn.float().unsqueeze(-1)
    d_sae = z.shape[-1]

    low_asr, n_skipped = [], 0
    last_print = time.time()
    print(f"\n[seed={seed}] scanning {d_sae} features, alpha={alpha}", flush=True)

    for f in range(d_sae):
        zf = z[..., f]
        if zf.abs().max().item() < 1e-7:
            n_skipped += 1
        else:
            delta = (-zf.unsqueeze(-1) * W_dec[f] * pmask).to(acts.dtype)
            hooks = channel_steer_hook({"V": delta}, alpha, {"V": W_V}, block=0)
            gen = greedy_generate_with_hooks(model, tokens, hooks, 16, attention_mask=attn)
            asr = asr_16(gen, model.tokenizer)
            if asr < 0.5:
                low_asr.append((f, round(asr, 3)))
                print(f"  [seed={seed}] f={f:4d}  asr={asr:.3f}  ***", flush=True)

        if time.time() - last_print >= 10.0:
            print(f"  [seed={seed}] {f+1}/{d_sae} done  ({n_skipped} non-firing skipped  {len(low_asr)} low-ASR found)", flush=True)
            last_print = time.time()

    print(f"[seed={seed}] DONE — low-ASR features (asr<0.5): {low_asr}", flush=True)
    print(f"[seed={seed}] {n_skipped}/{d_sae} skipped (never fired on dep prompts)", flush=True)
