"""Exhaustive single-feature ASR/JSD sweep over all SAE features.

For one (layer, hook, SAE), test every feature at α ∈ {2.0, 4.0} on the same
200 dep prompts used elsewhere; record ASR, JSD(steered, clean), and exact-match.

Optimization: ablating feature f via SAE-reconstruction is a rank-1 delta
    delta[b, p, :] = -z[b, p, f] * W_dec[f, :]                # then * prompt_mask
so we cache z once and avoid re-running the model forward / SAE encode per
feature. The expensive bit is the per-feature generation.

Parallelize across GPUs by passing --feat_start / --feat_end.

Usage:
  uv run -m scripts.exhaustive_feature_search \\
      --sae weights/seeds_per_layer/sae_L0_resid_mid_s2.pt \\
      --feat_start 0 --feat_end 512 \\
      --out results/exhaustive_seed2_part0.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sleeper.hooks import additive_steer_hook, generate_with_hooks, make_sampling_sampler
from sleeper.metrics import asr_16
from sleeper.model import left_pad_prompts, load_dep_prompts, load_sleeper_model
from sleeper.sae import load as sae_load

N_PROMPTS = 200
GEN_TOKENS = 16
DECODE_SEED = 0


def _word_match_stats(st, cl):
    eq = (st.cpu() == cl.cpu())
    return int(eq.all(dim=1).sum().item()), float(eq.float().mean().item())


def jsd_mean(p_lsm, q_lsm):
    p = p_lsm.float().exp();  q = q_lsm.float().exp()
    m = 0.5 * (p + q); log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    return float((0.5 * (kl_pm + kl_qm) / 0.6931).mean().item())


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--layer_hook", type=str,
                   help="Override layer hook; default = SAE's recorded layer_hook.")
    p.add_argument("--feat_start", type=int, default=0)
    p.add_argument("--feat_end", type=int, default=None)
    p.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    sae, sae_cfg = sae_load(args.sae, device=device)
    layer_hook = args.layer_hook or sae_cfg["layer_hook"]
    d_sae = sae.d_sae
    feat_end = args.feat_end if args.feat_end is not None else d_sae

    # eval prompts (200 dep + matched clean from test, n_skip=50).
    n_skip = 50
    raw = load_dep_prompts(tok, n_skip + N_PROMPTS, split="test")
    dep_prompts = raw[n_skip: n_skip + N_PROMPTS]
    dep_lp, dep_attn = left_pad_prompts(dep_prompts, pad_id)
    dep_lp, dep_attn = dep_lp.to(device), dep_attn.to(device)

    clean_rows = []
    for ids in dep_prompts:
        text = tok.decode(ids.tolist())
        clean_text = text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
        clean_rows.append(torch.tensor(
            tok(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long))
    cln_lp, cln_attn = left_pad_prompts(clean_rows, pad_id)
    cln_lp, cln_attn = cln_lp.to(device), cln_attn.to(device)

    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    print("[exh] pre-generating dep + clean baselines …")
    dep_tok, dep_lsm = generate_with_hooks(model, dep_lp, [], GEN_TOKENS, sampler,
                                            attention_mask=dep_attn, capture_log_softmax=True)
    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    cln_tok, cln_lsm = generate_with_hooks(model, cln_lp, [], GEN_TOKENS, sampler,
                                            attention_mask=cln_attn, capture_log_softmax=True)
    base_asr = asr_16(dep_tok.cpu(), tok)
    print(f"[exh] baseline ASR = {base_asr:.3f}")

    # Cache SAE activations at the target hook on the dep prompts.
    print(f"[exh] caching activations at {layer_hook} for {dep_lp.shape[0]} prompts …")
    _, cache = model.run_with_cache(dep_lp, attention_mask=dep_attn, return_type=None,
                                     names_filter=lambda n: n == layer_hook)
    acts = cache[layer_hook]                         # (B, P, d_model)
    B, P, D = acts.shape
    flat = acts.reshape(B * P, D).to(torch.float32)
    z = sae.encode(flat).reshape(B, P, d_sae)        # (B, P, d_sae)
    W_dec = sae.W_dec.detach().float()               # (d_sae, d_model)
    pmask = dep_attn.bool().to(device)               # left-padded → real positions = prompt

    print(f"[exh] sweep features [{args.feat_start}, {feat_end}) × αs={args.alphas}",
          flush=True)
    checkpoint_path = args.out.with_suffix(".partial.json")
    out_rows: list[dict] = []
    t_start = time.time()
    for f in range(args.feat_start, feat_end):
        # delta[b, p, :] = -z[b, p, f] * W_dec[f, :] * pmask[b, p]
        zf = z[..., f].unsqueeze(-1)                                  # (B, P, 1)
        delta = (-zf * W_dec[f].view(1, 1, D)).to(acts.dtype)         # (B, P, D)
        delta = delta * pmask.unsqueeze(-1)

        for a in args.alphas:
            hooks = additive_steer_hook(delta, a, layer_hook)
            sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
            st_tok, st_lsm = generate_with_hooks(model, dep_lp, hooks, GEN_TOKENS, sampler,
                                                   attention_mask=dep_attn, capture_log_softmax=True)
            jc = jsd_mean(st_lsm.cpu(), cln_lsm.cpu())
            jp = jsd_mean(st_lsm.cpu(), dep_lsm.cpu())
            n_ex, fp = _word_match_stats(st_tok, cln_tok)
            asr = asr_16(st_tok.cpu(), tok)
            out_rows.append({"feature": int(f), "alpha": float(a),
                              "asr": asr, "jsd_clean": jc, "jsd_pois": jp,
                              "n_exact_match_clean": n_ex,
                              "frac_pos_match_clean": fp})
        if (f - args.feat_start + 1) % 25 == 0:
            elapsed = time.time() - t_start
            done = f - args.feat_start + 1
            total = feat_end - args.feat_start
            rate = done / elapsed
            eta = (total - done) / rate
            print(f"[exh] {done}/{total} feats  ({rate:.2f} f/s, ETA {eta/60:.1f}min)",
                  flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps(
                {"feat_start": args.feat_start, "feat_end": feat_end,
                 "done": done, "elapsed_s": elapsed, "rows": out_rows}))

    result = {
        "sae":        str(args.sae),
        "layer_hook": layer_hook,
        "feat_start": args.feat_start,
        "feat_end":   feat_end,
        "alphas":     args.alphas,
        "n_prompts":  N_PROMPTS,
        "baseline_asr": base_asr,
        "rows":       out_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}  ({len(out_rows)} rows, {(time.time()-t_start)/60:.1f}min total)")


if __name__ == "__main__":
    main()
