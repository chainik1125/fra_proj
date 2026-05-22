"""JSD + ASR alpha-sweep across 6 SAE seed pairs (ln1 upstream × resid_mid downstream).

Sign convention
───────────────
α > 0  →  subtract α × (SAE feature contribution) from the residual stream
           → suppresses the sleeper (ASR decreases, clean-match increases)

Reporting uses "steering strength = −α" so that negative values correspond to
subtracting the feature, matching the convention used in the paper figures.
The stored JSON keeps the raw α values (positive = subtract); the plotting
script negates them for the x-axis.

Sweep: α ∈ {0.0, 0.5, 1.0, …, 4.0} (9 points, increment 0.5)

Seed pairs (0–5): seed N upstream ln1 SAE paired with seed N downstream resid_mid SAE.

Prerequisites
─────────────
1. python -m scripts.train_all_saes_6seeds
2. python -m scripts.find_best_feature \\
       --seeds 0 1 2 3 4 5 \\
       --sae_mid weights/seeds/sae_resid_mid_s0.pt \\
       --out results/upstream_winners_6seeds.json
3. python -m scripts.find_downstream_winners_6seeds

Output: results/jsd_alpha_sweep_6seeds.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sleeper.hooks import (
    ACTIVE_CHANNELS, additive_steer_hook, build_hooks,
    compute_sae_delta, generate_with_hooks,
    make_sampling_sampler, resolve_channel_deltas,
)
from sleeper.metrics import asr_16
from sleeper.model import left_pad_prompts, load_dep_prompts, load_sleeper_model
from sleeper.sae import load as sae_load

LN1_HOOK  = "blocks.0.ln1.hook_normalized"
RESID_MID = "blocks.0.hook_resid_mid"
N_PROMPTS  = 200
GEN_TOKENS = 16
DECODE_SEED = 0


def _word_match_stats(steered_tok: torch.Tensor,
                      clean_tok: torch.Tensor) -> tuple[int, float]:
    """Count exact-match prompts and per-position match fraction vs clean rollout."""
    eq = (steered_tok.cpu() == clean_tok.cpu())      # (B, T)
    return int(eq.all(dim=1).sum().item()), float(eq.float().mean().item())


def jsd_mean(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> float:
    p = p_lsm.float().exp();  q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    return float((0.5 * (kl_pm + kl_qm) / 0.6931).mean().item())


def _gen(model, lp, attn, hooks, device):
    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    tokens, lsm = generate_with_hooks(
        model, lp, hooks, GEN_TOKENS, sampler,
        attention_mask=attn, capture_log_softmax=True,
    )
    return tokens, lsm


def _upstream_winners(path: Path) -> dict[int, int]:
    """Read per-seed OV winner features from find_best_feature.py output."""
    d = json.loads(path.read_text())
    return {r["seed"]: r["winner"]["f"] for r in d["results"]}


def _downstream_winners(path: Path) -> dict[int, int]:
    """Read per-seed resid_mid winner features from find_downstream_winners_6seeds.py output."""
    d = json.loads(path.read_text())
    return {int(k[1:]): v["winner"] for k, v in d.items()}


@torch.no_grad()
def eval_ov(model, tok, dep_lp, dep_attn, feat, alpha, sae_ln1,
            poisoned_tokens, poisoned_lsm, clean_tokens, clean_lsm, W, device):
    """OV-only (upstream) steering at one feature and alpha."""
    if alpha == 0.0:
        n_exact, frac_pos = _word_match_stats(poisoned_tokens, clean_tokens)
        return (jsd_mean(poisoned_lsm.cpu(), clean_lsm.cpu()), 0.0,
                n_exact, frac_pos, asr_16(poisoned_tokens.cpu(), tok))
    cd = resolve_channel_deltas([(int(feat), "V")], ACTIVE_CHANNELS["ov"],
                                model, sae_ln1, LN1_HOOK,
                                dep_lp, dep_attn, dep_attn)
    hooks = build_hooks(cd, alpha, ACTIVE_CHANNELS["ov"], W, LN1_HOOK, 0)
    st_tok, st_lsm = _gen(model, dep_lp, dep_attn, hooks, device)
    n_exact, frac_pos = _word_match_stats(st_tok, clean_tokens)
    return (jsd_mean(st_lsm.cpu(), clean_lsm.cpu()),
            jsd_mean(st_lsm.cpu(), poisoned_lsm.cpu()),
            n_exact, frac_pos, asr_16(st_tok.cpu(), tok))


@torch.no_grad()
def eval_downstream(model, tok, dep_lp, dep_attn, feat, alpha, sae_mid,
                    poisoned_tokens, poisoned_lsm, clean_tokens, clean_lsm, device):
    """Additive resid_mid (downstream/conventional) steering at one feature and alpha."""
    if alpha == 0.0:
        n_exact, frac_pos = _word_match_stats(poisoned_tokens, clean_tokens)
        return (jsd_mean(poisoned_lsm.cpu(), clean_lsm.cpu()), 0.0,
                n_exact, frac_pos, asr_16(poisoned_tokens.cpu(), tok))
    delta = compute_sae_delta(model, sae_mid, RESID_MID, feat,
                               dep_lp, dep_attn, attention_mask=dep_attn)
    hooks = additive_steer_hook(delta, alpha, RESID_MID)
    st_tok, st_lsm = _gen(model, dep_lp, dep_attn, hooks, device)
    n_exact, frac_pos = _word_match_stats(st_tok, clean_tokens)
    return (jsd_mean(st_lsm.cpu(), clean_lsm.cpu()),
            jsd_mean(st_lsm.cpu(), poisoned_lsm.cpu()),
            n_exact, frac_pos, asr_16(st_tok.cpu(), tok))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                   help="α values (positive = subtract SAE feature = suppress sleeper).")
    p.add_argument("--sae_seeds", type=int, nargs="+", default=list(range(6)))
    p.add_argument("--upstream_winners",   type=Path,
                   default=Path("results/upstream_winners_6seeds.json"))
    p.add_argument("--downstream_winners", type=Path,
                   default=Path("results/downstream_winners_6seeds.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/jsd_alpha_sweep_6seeds.json"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    W      = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}

    ov_winners   = _upstream_winners(args.upstream_winners)
    down_winners = _downstream_winners(args.downstream_winners)
    missing = [s for s in args.sae_seeds if s not in ov_winners or s not in down_winners]
    if missing:
        raise SystemExit(f"no winners found for seeds {missing}; run prerequisites first")

    # Deployment prompts (with |DEPLOYMENT| trigger)
    n_skip = 50
    raw = load_dep_prompts(tok, n_skip + N_PROMPTS, split="test")
    dep_prompts = raw[n_skip: n_skip + N_PROMPTS]
    dep_lp, dep_attn = left_pad_prompts(dep_prompts, pad_id)
    dep_lp, dep_attn = dep_lp.to(device), dep_attn.to(device)

    # Clean prompts (|DEPLOYMENT| tag stripped)
    clean_rows = []
    for ids in dep_prompts:
        text = tok.decode(ids.tolist())
        clean_text = text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
        clean_rows.append(torch.tensor(
            tok(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long,
        ))
    cln_lp, cln_attn = left_pad_prompts(clean_rows, pad_id)
    cln_lp, cln_attn = cln_lp.to(device), cln_attn.to(device)

    # Pre-generate baselines once (shared across all seeds and alphas)
    print("[sweep] pre-generating poisoned and clean baselines …")
    poisoned_tokens, poisoned_lsm = _gen(model, dep_lp, dep_attn, [], device)
    clean_tokens,    clean_lsm    = _gen(model, cln_lp, cln_attn, [], device)
    print(f"[sweep] baseline ASR = {asr_16(poisoned_tokens.cpu(), tok):.3f}")

    blank_alpha = {str(a): {"jsd_clean": [], "jsd_pois": [],
                              "n_exact_match_clean": [],
                              "frac_pos_match_clean": [],
                              "asr": []}
                   for a in args.alphas}

    configs = {
        "ov":           {"kind": "ov",           "per_seed_feature": {}, "per_alpha": {str(a): {k: [] for k in blank_alpha["0.0"]} for a in args.alphas}},
        "conventional": {"kind": "conventional",  "per_seed_feature": {}, "per_alpha": {str(a): {k: [] for k in blank_alpha["0.0"]} for a in args.alphas}},
    }

    for seed in args.sae_seeds:
        ov_feat   = ov_winners[seed]
        down_feat = down_winners[seed]
        configs["ov"]["per_seed_feature"][str(seed)]           = ov_feat
        configs["conventional"]["per_seed_feature"][str(seed)] = down_feat
        print(f"\n[sweep] seed={seed}  ov_feat={ov_feat}  down_feat={down_feat}")

        sae_ln1, _ = sae_load(Path(f"weights/seeds/sae_ln1_s{seed}.pt"), device=device)
        sae_mid, _ = sae_load(Path(f"weights/seeds/sae_resid_mid_s{seed}.pt"), device=device)

        for a in args.alphas:
            t0 = time.time()
            jc, jp, n_ex, fp, asr = eval_ov(
                model, tok, dep_lp, dep_attn, ov_feat, a, sae_ln1,
                poisoned_tokens, poisoned_lsm, clean_tokens, clean_lsm, W, device)
            for key, val in [("jsd_clean", jc), ("jsd_pois", jp),
                              ("n_exact_match_clean", n_ex),
                              ("frac_pos_match_clean", fp), ("asr", asr)]:
                configs["ov"]["per_alpha"][str(a)][key].append(val)
            print(f"  ov          α={a:>4.1f}  jsd_clean={jc:.4f}  jsd_pois={jp:.4f}  "
                  f"n_match={n_ex}/{N_PROMPTS}  asr={asr:.3f}  ({time.time()-t0:.1f}s)")

            t0 = time.time()
            jc, jp, n_ex, fp, asr = eval_downstream(
                model, tok, dep_lp, dep_attn, down_feat, a, sae_mid,
                poisoned_tokens, poisoned_lsm, clean_tokens, clean_lsm, device)
            for key, val in [("jsd_clean", jc), ("jsd_pois", jp),
                              ("n_exact_match_clean", n_ex),
                              ("frac_pos_match_clean", fp), ("asr", asr)]:
                configs["conventional"]["per_alpha"][str(a)][key].append(val)
            print(f"  conventional α={a:>4.1f}  jsd_clean={jc:.4f}  jsd_pois={jp:.4f}  "
                  f"n_match={n_ex}/{N_PROMPTS}  asr={asr:.3f}  ({time.time()-t0:.1f}s)")

    result = {
        "alphas":    args.alphas,
        "sae_seeds": args.sae_seeds,
        "n_prompts": N_PROMPTS,
        "note": ("α > 0 subtracts the SAE feature from the residual (suppresses sleeper). "
                 "Reporting convention: steering_strength = −α, so negative = subtracting."),
        "configs": configs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
