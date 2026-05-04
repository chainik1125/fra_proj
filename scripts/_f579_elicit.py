"""Steer WITH the downstream resid_mid deployment feature on clean prompts.

The canonical feature index is 579 (Dmitry's SAE seed), but the exact index
varies with SAE training seed. Pass --feature N to fix it, or omit to
auto-detect the top dep-vs-clean feature in the loaded SAE.

Unlike the upstream ln1 feature (f=1114), this lives in the residual stream
after attention, so we steer with additive_steer_hook at hook_resid_mid
rather than projecting through W_V.

Direction choices:
  --direction decoder   v = W_dec[f] / ||W_dec[f]||   (reconstruction direction)
  --direction encoder   v = W_enc[:, f] / ||W_enc[:, f]||   (detection direction)

α is anchored to the typical dep-prompt firing strength; baseline and steered
runs are RNG-matched (same seed → same draws each step).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import (
    additive_steer_hook,
    generate_with_hooks,
    make_greedy_sampler,
    make_nucleus_sampler,
)
from sleeper.metrics import (
    asr_16, clean_continuation_ce, rank_features_by_dep_clean,
    teacher_forced_sleeper_logp,
)
from sleeper.model import (
    cache_activations,
    load_paired_dataset,
    load_sleeper_model,
    prompt_mask_from_markers,
)
from sleeper.sae import load


def distinct_2(rows: torch.Tensor) -> float:
    total, seen = 0, set()
    for row in rows.tolist():
        for i in range(len(row) - 1):
            total += 1
            seen.add((row[i], row[i + 1]))
    return len(seen) / max(1, total)


def repeat_3gram_frac(rows: torch.Tensor) -> float:
    hits = 0
    for row in rows.tolist():
        seen, bad = set(), False
        for i in range(len(row) - 2):
            g = (row[i], row[i + 1], row[i + 2])
            if g in seen:
                bad = True
                break
            seen.add(g)
        hits += int(bad)
    return hits / max(1, rows.shape[0])


def make_sampler(decoding: str, seed: int, temperature: float, top_p: float, device):
    if decoding == "greedy":
        return make_greedy_sampler()
    if decoding == "nucleus":
        return make_nucleus_sampler(
            temperature=temperature, top_p=top_p, seed=seed, device=device,
        )
    raise ValueError(f"unknown decoding: {decoding}")


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae_mid", type=Path, default=Path("weights/sae_resid_mid.pt"))
    p.add_argument("--feature", type=int, default=-1,
                   help="SAE feature index (-1 = auto-detect top dep-vs-clean feature)")
    p.add_argument("--decoding", choices=["greedy", "nucleus"], default="nucleus")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0, 20.0])
    p.add_argument("--direction", choices=["decoder", "encoder"], default="decoder")
    p.add_argument("--out", type=Path,
                   default=Path("weights/seeds/f579_elicit_on_clean.json"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_mid, mid_cfg = load(args.sae_mid, device=device)
    mid_hook = mid_cfg["layer_hook"]
    block, seq_len, gen_tokens = 0, 128, 16

    model = load_sleeper_model(device=device)

    splits = load_paired_dataset(
        tokenizer=model.tokenizer, n_train=2, n_val=200, n_test=args.n_test,
        seq_len=seq_len, seed=0,
    )
    val, test = splits["val"], splits["test"]
    val_pmask = prompt_mask_from_markers(seq_len, val.story_marker_pos)
    pmask = prompt_mask_from_markers(seq_len, test.story_marker_pos)
    dep = test.tokens[test.is_deployment].to(device)
    dep_mask = pmask[test.is_deployment].to(device)
    cln = test.tokens[~test.is_deployment].to(device)
    cln_mask = pmask[~test.is_deployment].to(device)
    cln_marker = test.story_marker_pos[~test.is_deployment].to(device)

    # auto-detect or use specified feature
    if args.feature < 0:
        print(f"[elicit] auto-detecting top dep-vs-clean feature on val split …")
        val_acts = cache_activations(model, val.tokens, [mid_hook])[mid_hook].to(device).float()
        val_z = sae_mid.encode(val_acts.reshape(-1, val_acts.shape[-1]))
        val_z = val_z.reshape(*val_acts.shape[:2], -1)
        ranking = rank_features_by_dep_clean(
            val_z, val.is_deployment.to(device), val_pmask.to(device), top_k=5,
        )
        feature = int(ranking["top_indices"][0].item())
        print(f"[elicit] auto-detected f={feature}  score={ranking['scores'][feature]:.3f}")
    else:
        feature = args.feature

    # anchor α to dep-prompt firing strength
    cache_dep = cache_activations(model, dep.cpu(), [mid_hook])[mid_hook].to(device).float()
    z_dep = sae_mid.encode(cache_dep.reshape(-1, cache_dep.shape[-1]))
    z_dep = z_dep.reshape(*cache_dep.shape[:2], -1)[..., feature]
    fire_mask = (z_dep > 0) & dep_mask
    mean_fire = (z_dep * fire_mask.float()).sum() / fire_mask.float().sum().clamp(min=1.0)
    print(f"[elicit] decoding={args.decoding}  f={feature} dep fire_rate="
          f"{fire_mask.float().mean().item():.3f}  mean_when_fires={mean_fire.item():.3f}")

    cache_cln = cache_activations(model, cln.cpu(), [mid_hook])[mid_hook].to(device).float()
    z_cln = sae_mid.encode(cache_cln.reshape(-1, cache_cln.shape[-1]))
    z_cln = z_cln.reshape(*cache_cln.shape[:2], -1)[..., feature]
    print(f"[elicit] f={feature} on cln prompts: max={z_cln.max().item():.3e}  "
          f"mean={z_cln.mean().item():.3e}")

    w_dec = sae_mid.W_dec[feature].detach().to(device).float()
    w_enc = sae_mid.W_enc[:, feature].detach().to(device).float()
    cos = (w_dec @ w_enc) / (w_dec.norm() * w_enc.norm()).clamp(min=1e-12)
    print(f"[elicit] direction={args.direction}  ||W_dec||={w_dec.norm().item():.3f}  "
          f"||W_enc||={w_enc.norm().item():.3f}  cos(W_dec, W_enc)={cos.item():+.3f}")

    raw = w_dec if args.direction == "decoder" else w_enc
    v = raw / raw.norm().clamp(min=1e-12)
    B, P = cln.shape
    delta_cln = (mean_fire * v).view(1, 1, -1).expand(B, P, -1) * cln_mask.float().unsqueeze(-1)
    delta_cln = delta_cln.contiguous()

    def mk_sampler():
        return make_sampler(args.decoding, args.seed, args.temperature, args.top_p, device)

    def run(hooks=None):
        h = hooks or []
        gen = generate_with_hooks(model, cln, h, gen_tokens, mk_sampler()).cpu()
        logp = teacher_forced_sleeper_logp(
            model, model.tokenizer, cln, fwd_hooks=hooks,
        ).mean().item()
        ce = clean_continuation_ce(model, cln, cln_marker, fwd_hooks=hooks).mean().item()
        return gen, logp, ce

    base_gen, base_logp, base_ce = run()
    base_asr = asr_16(base_gen, model.tokenizer)
    print(f"[elicit] cln baseline: sleeper_logp={base_logp:.2f}  ce={base_ce:.4f}  asr={base_asr:.3f}")

    rows = []
    samples = {"baseline": [model.tokenizer.decode(g) for g in base_gen[:6].tolist()]}
    for alpha in args.alphas:
        hooks = additive_steer_hook(delta_cln, alpha, mid_hook)
        gen, logp, ce = run(hooks)
        r = {
            "alpha": alpha,
            "asr_16": asr_16(gen, model.tokenizer),
            "sleeper_logp": logp, "delta_logp": logp - base_logp,
            "self_ce": ce, "delta_ce": ce - base_ce,
            "distinct_2": distinct_2(gen),
            "repeat_3gram": repeat_3gram_frac(gen),
        }
        rows.append(r)
        samples[f"alpha_{alpha}"] = [model.tokenizer.decode(g) for g in gen[:6].tolist()]
        print(f"[elicit] α={alpha:>5}: asr={r['asr_16']:.3f}  "
              f"Δlogp(sleeper)={r['delta_logp']:+.3f}  ΔCE={r['delta_ce']:+.4f}  "
              f"d2={r['distinct_2']:.3f}  rep3={r['repeat_3gram']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {"feature": feature, "feature_arg": args.feature,
                   "mid_hook": mid_hook, "block": block,
                   "direction": args.direction,
                   "w_dec_norm": w_dec.norm().item(),
                   "w_enc_norm": w_enc.norm().item(),
                   "cos_dec_enc": cos.item(),
                   "scale_mean_fire": mean_fire.item(),
                   "decoding": args.decoding, "temperature": args.temperature,
                   "top_p": args.top_p, "seed": args.seed,
                   "n_cln": int(cln.shape[0])},
        "baseline": {"sleeper_logp": base_logp, "self_ce": base_ce, "asr_16": base_asr},
        "sweep": rows, "samples": samples,
    }, indent=2))
    print(f"[elicit] wrote {args.out}")


if __name__ == "__main__":
    main()
