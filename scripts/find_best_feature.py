"""Find, per upstream SAE seed, the feature achieving ASR=0 with minimum ΔCE.

Pipeline per seed:
  1. OV attribution on val split → top_k upstream ln1 candidates.
  2. Screen all (f, α) by teacher-forced Δlogp using pre-cached SAE activations
     (analytic delta = -z[…,f]·W_dec[f]; exact for linear decoder, no rerun).
  3. Stage-2: for the top stage2_keep (f, α) pairs, run batched greedy ASR_16
     on left-padded test dep prompts (single batch, attention mask passed through).
     Batch-aware: compute_sae_delta runs the whole batch so the delta is non-zero
     only where feature fires — naturally different per sequence.
  4. Winner = min ΔCE s.t. ASR=0; fall back to min ASR if none achieve 0.
  5. Report max-activation token examples for the winner feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.attribution import compute_ov_weights, ov_attribution, rank_dep_vs_clean
from sleeper.hooks import (
    additive_steer_hook, compute_sae_delta, greedy_generate_with_hooks,
    ov_only_steer_hook,
)
from sleeper.metrics import asr_16, clean_continuation_ce, teacher_forced_sleeper_logp
from sleeper.model import (
    cache_activations, left_pad_prompts, load_dep_prompts,
    load_paired_dataset, load_sleeper_model, prompt_mask_from_markers,
)
from sleeper.sae import TopKSAE, encode_all, load, save, train


def _ln1_path(seed: int) -> Path:
    return Path(f"weights/seeds/sae_ln1_s{seed}.pt")


def _train_or_load(seed: int, model, device: str) -> tuple[TopKSAE, str]:
    path = _ln1_path(seed)
    ln1_hook = "blocks.0.ln1.hook_normalized"
    if path.exists():
        sae, _ = load(path, device=device)
        return sae, ln1_hook
    print(f"[best] seed={seed}: training SAE …")
    splits = load_paired_dataset(model.tokenizer, n_train=10_000, n_val=0, n_test=0,
                                  seq_len=128, seed=seed)
    with torch.no_grad():
        acts = cache_activations(model, splits["train"].tokens, [ln1_hook],
                                  chunk_size=16)[ln1_hook]
    with torch.enable_grad():
        sae, _ = train(acts, d_sae=1536, k=32, n_steps=4000, batch_size=4096,
                       lr=5e-4, seed=seed, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(sae, path, layer_hook=ln1_hook, n_train_seqs=10_000, seq_len=128,
         n_steps=4000, batch_size=4096, lr=5e-4)
    return sae, ln1_hook


@torch.no_grad()
def _screen(model, sae_ln1, ln1_hook, W_V, top_feats, alphas,
            val_dep, val_dep_pmask, device) -> tuple[list[dict], float]:
    """Δlogp for all (f, α) via OV-only suppression, from pre-cached SAE activations."""
    acts = cache_activations(model, val_dep.cpu(), [ln1_hook], chunk_size=16)[ln1_hook]
    z = encode_all(sae_ln1, acts)                     # (N, T, d_sae) float32 cpu
    W_dec = sae_ln1.W_dec.detach().cpu().float()       # (d_sae, d_model)
    pmask_f = val_dep_pmask.cpu().float().unsqueeze(-1)

    base_logp = teacher_forced_sleeper_logp(model, model.tokenizer, val_dep).mean().item()

    rows: list[dict] = []
    for f in top_feats:
        # analytic delta (exact for linear decoder): decode(z_abl)-decode(z) = -z[f]·W_dec[f]
        delta = (-z[..., f:f+1] * W_dec[f]) * pmask_f  # (N, T, d_model) cpu
        if delta.abs().max().item() < 1e-7:
            for alpha in alphas:
                rows.append({"f": int(f), "alpha": alpha, "dlogp": 0.0})
            continue
        delta_dev = delta.to(device)
        for alpha in alphas:
            hooks = ov_only_steer_hook(delta_dev, alpha, W_V, block=0)
            logp = teacher_forced_sleeper_logp(
                model, model.tokenizer, val_dep, fwd_hooks=hooks,
            ).mean().item()
            rows.append({"f": int(f), "alpha": alpha, "dlogp": logp - base_logp})
    return rows, base_logp


@torch.no_grad()
def _asr_and_dce(model, sae_ln1, ln1_hook, W_V, candidates,
                 test_dep_lp, test_dep_attn, test_dep_pmask,
                 val_cln, val_cln_marker, gen_tokens, device):
    """Batched greedy ASR on left-padded dep prompts + ΔCE on clean val, OV-only."""
    base_asr = asr_16(
        greedy_generate_with_hooks(model, test_dep_lp, [], gen_tokens,
                                   attention_mask=test_dep_attn),
        model.tokenizer,
    )
    base_ce = clean_continuation_ce(model, val_cln, val_cln_marker).mean().item()

    cln_pmask = prompt_mask_from_markers(val_cln.shape[1], val_cln_marker.cpu()).to(device)

    rows = []
    for f, alpha in candidates:
        delta_dep = compute_sae_delta(model, sae_ln1, ln1_hook, f,
                                      test_dep_lp, test_dep_pmask, test_dep_attn)
        gen = greedy_generate_with_hooks(
            model, test_dep_lp, ov_only_steer_hook(delta_dep, alpha, W_V, block=0),
            gen_tokens, attention_mask=test_dep_attn,
        )
        asr = asr_16(gen, model.tokenizer)

        delta_cln = compute_sae_delta(model, sae_ln1, ln1_hook, f, val_cln, cln_pmask)
        hooks_cln = ov_only_steer_hook(delta_cln, alpha, W_V, block=0)
        ce = clean_continuation_ce(model, val_cln, val_cln_marker, fwd_hooks=hooks_cln).mean().item()
        rows.append({"f": int(f), "alpha": alpha, "asr": asr, "dce": ce - base_ce})
    return base_asr, base_ce, rows


@torch.no_grad()
def _max_act_examples(z, tokens, tokenizer, feature, topn=5, ctx=8):
    """Tokens around positions where `feature` fires most strongly in z (N,T,d_sae)."""
    vals = z[..., feature]  # (N, T)
    flat = vals.flatten()
    top_idx = flat.argsort(descending=True)[:topn]
    N, T = vals.shape
    out = []
    for idx in top_idx.tolist():
        n, t = divmod(idx, T)
        lo, hi = max(0, t - ctx), min(T, t + ctx + 1)
        ctx_toks = tokens[n, lo:hi].tolist()
        out.append({
            "activation": float(flat[idx]),
            "text": tokenizer.decode(ctx_toks),
            "firing_token": tokenizer.decode([tokens[n, t].item()]),
        })
    return out


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--stage2_keep", type=int, default=15)
    p.add_argument("--sae_mid", type=Path, default=Path("weights/sae_resid_mid.pt"))
    p.add_argument("--target_feature", type=int, default=579)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--gen_tokens", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("weights/find_best_feature.json"))
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae_mid, _ = load(args.sae_mid, device=device)
    d_target = sae_mid.W_enc[:, args.target_feature].detach().to(device).float()

    model = load_sleeper_model(device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    W_V = model.W_V[0].detach().to(device)

    splits = load_paired_dataset(tok, n_train=2, n_val=args.n_val, n_test=args.n_test,
                                  seq_len=128, seed=0)
    val, test = splits["val"], splits["test"]
    val_pmask = prompt_mask_from_markers(128, val.story_marker_pos)

    val_dep = val.tokens[val.is_deployment].to(device)
    val_dep_pmask = val_pmask[val.is_deployment].to(device)
    val_cln = val.tokens[~val.is_deployment].to(device)
    val_cln_marker = val.story_marker_pos[~val.is_deployment].to(device)

    # variable-length dep prompts for generation (both trigger formats handled)
    raw_prompts = load_dep_prompts(tok, args.n_test, split="test")
    test_dep_lp, test_dep_attn = left_pad_prompts(raw_prompts[:args.n_test // 2], pad_id)
    test_dep_lp = test_dep_lp.to(device)
    test_dep_attn = test_dep_attn.to(device)
    test_dep_pmask = test_dep_attn  # prompt mask = attention mask for left-padded prompts
    print(f"[best] test dep prompts: {test_dep_lp.shape}  "
          f"len range [{test_dep_attn.sum(1).min()}, {test_dep_attn.sum(1).max()}]")

    # cache attention pattern on val (shared across seeds)
    pattern_hook = "blocks.0.attn.hook_pattern"
    ln1_hook = "blocks.0.ln1.hook_normalized"
    val_caches = cache_activations(model, val.tokens, [pattern_hook, ln1_hook])
    A_val = val_caches[pattern_hook].to(device)

    results = []
    for seed in args.seeds:
        sae_ln1, _ = _train_or_load(seed, model, device)
        print(f"\n[best] ── seed={seed} ──")

        # OV attribution → top_k candidates
        ovw = compute_ov_weights(model, sae_ln1, d_target, block=0)
        z_ln1_val = encode_all(sae_ln1, val_caches[ln1_hook]).to(device)
        ov_out = ov_attribution(A_val, z_ln1_val, ovw["beta"])
        ranked = rank_dep_vs_clean(ov_out["contrib"], val.is_deployment.to(device),
                                   query_mask=val_pmask.to(device))
        top_feats = ranked["top_indices"][: args.top_k].cpu().tolist()
        print(f"[best] top-{args.top_k} OV features: {top_feats[:8]} …")

        # screen by Δlogp
        screen_rows, base_logp = _screen(
            model, sae_ln1, ln1_hook, W_V, top_feats, args.alphas,
            val_dep, val_dep_pmask, device,
        )
        screen_rows.sort(key=lambda r: r["dlogp"])
        stage2 = [(r["f"], r["alpha"]) for r in screen_rows[: args.stage2_keep]]
        print(f"[best] stage-2 pairs: {stage2[:5]} …  "
              f"best Δlogp={screen_rows[0]['dlogp']:+.2f}")

        # batched ASR + ΔCE for stage-2
        base_asr, base_ce, eval_rows = _asr_and_dce(
            model, sae_ln1, ln1_hook, W_V, stage2,
            test_dep_lp, test_dep_attn, test_dep_pmask,
            val_cln, val_cln_marker, args.gen_tokens, device,
        )
        for r in eval_rows:
            print(f"[best]   f={r['f']} α={r['alpha']}  "
                  f"asr={r['asr']:.3f}  ΔCE={r['dce']:+.4f}")

        asr0 = [r for r in eval_rows if r["asr"] == 0.0]
        winner = min(asr0, key=lambda r: r["dce"]) if asr0 else min(eval_rows, key=lambda r: r["asr"])
        print(f"[best] WINNER seed={seed}: f={winner['f']} α={winner['alpha']}  "
              f"asr={winner['asr']:.3f}  ΔCE={winner['dce']:+.4f}  "
              f"(base_asr={base_asr:.3f})")

        # max-act examples for winner feature on val dep prompts
        acts_dep = cache_activations(model, val_dep.cpu(), [ln1_hook])[ln1_hook]
        z_dep = encode_all(sae_ln1, acts_dep)  # (N, T, d_sae)
        examples = _max_act_examples(z_dep, val_dep.cpu(), tok, winner["f"])
        print(f"[best] max-act examples for f={winner['f']}:")
        for ex in examples:
            print(f"       act={ex['activation']:.2f}  token='{ex['firing_token']}'  "
                  f"ctx='{ex['text']}'")

        results.append({
            "seed": seed,
            "top_k_ov": top_feats,
            "winner": winner,
            "base_asr": base_asr,
            "base_ce": base_ce,
            "base_logp": base_logp,
            "screen": screen_rows,
            "eval": eval_rows,
            "max_act_examples": examples,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"config": vars(args) | {"seeds": args.seeds}, "results": results},
        indent=2, default=str,
    ))
    print(f"\n[best] wrote {args.out}")


if __name__ == "__main__":
    main()
