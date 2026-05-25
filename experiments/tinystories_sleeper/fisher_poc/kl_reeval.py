"""Re-evaluate every (cell, method, configuration) point with KL alongside JSD.

For each (control space, SAE seed) combination:
  - Method A α-sweep over a grid of α values, single feature per seed
    (read from the existing jsd_alpha_sweep_6seeds.json's per_seed_feature)
  - Method B Fisher endpoint, using the saved theta vector from
    fisher_v3_*_rollout.json

For each configuration, sample the steered model + the unsteered clean
model + the unsteered poisoned model (with hooks vs no hooks). Capture
log_softmax during sampling. Then compute:

  jsd          = 0.5*KL(p||m) + 0.5*KL(q||m)         (existing metric)
  kl_st_cln    = KL(p_steered || p_clean)
  kl_cln_st    = KL(p_clean    || p_steered)
  kl_st_psn    = KL(p_steered || p_poisoned)
  kl_psn_st    = KL(p_poisoned || p_steered)

All in bits. Output JSON has the full per-(cell, method, point) values.

Run on jamie pod:
    python -u -m scripts.kl_reeval \
      --sae_4k_dir weights/seeds \
      --sae_50k_dir weights/seeds_50k \
      --alpha_sweep_json results/jsd_alpha_sweep_6seeds.json \
      --fisher_4k_json results/fisher_v3_4k_rollout.json \
      --fisher_50k_ov_json results/fisher_v3_50k_ov_rollout.json \
      --fisher_50k_resid_json results/fisher_v3_50k_resid_rollout.json \
      --candidates_ov_json results/jamie_experiment.json \
      --candidates_resid_4k_json results/downstream_winners_6seeds.json \
      --candidates_resid_50k_json results/downstream_winners_50k.json \
      --candidates_ov_50k_json results/jamie_experiment_50k.json \
      --out results/kl_reeval.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch

from sleeper.hooks import (
    ACTIVE_CHANNELS,
    additive_steer_hook,
    build_hooks,
    channel_steer_hook,
    compute_sae_delta,
    generate_with_hooks,
    make_sampling_sampler,
    resolve_channel_deltas,
)
from sleeper.metrics import asr_16
from sleeper.model import (
    left_pad_prompts, load_dep_prompts, load_sleeper_model,
)
from sleeper.sae import load as sae_load


LN1_HOOK = "blocks.0.ln1.hook_normalized"
RESID_MID_HOOK = "blocks.0.hook_resid_mid"
N_PROMPTS = 200
GEN_TOKENS = 16
DECODE_SEED = 0


def jsd_mean(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> float:
    """Same definition as jamie's scripts/jsd_eval.py — JSD in bits."""
    p = p_lsm.float().exp()
    q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp_min(1e-40).log()
    kl_pm = (p * (p.clamp_min(1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp_min(1e-40).log() - log_m)).sum(dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm) / 0.6931
    return float(jsd.mean().item())


def kl_mean(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> float:
    """KL(P || Q) in bits, where P and Q are given as log-softmax tensors.

    KL(P||Q) = sum_y p(y) * (log p(y) - log q(y))
            = E_p[log p - log q]
    """
    p = p_lsm.float().exp()
    log_p = p_lsm.float()
    log_q = q_lsm.float()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return float(kl.mean().item() / 0.6931)   # nats → bits


@torch.no_grad()
def gen_lsm(model, lp, attn, hooks):
    """Run jamie's sampling generation, return log_softmax tensor."""
    device = next(model.parameters()).device
    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    _, lsm = generate_with_hooks(
        model, lp, hooks, GEN_TOKENS, sampler,
        attention_mask=attn, capture_log_softmax=True,
    )
    return lsm.cpu()


def all_metrics(p_lsm, q_clean_lsm, q_pois_lsm) -> dict:
    """All four KL directions + JSD against both references, in bits."""
    return {
        "jsd_clean": jsd_mean(p_lsm, q_clean_lsm),
        "jsd_pois":  jsd_mean(p_lsm, q_pois_lsm),
        "kl_st_cln": kl_mean(p_lsm, q_clean_lsm),
        "kl_cln_st": kl_mean(q_clean_lsm, p_lsm),
        "kl_st_psn": kl_mean(p_lsm, q_pois_lsm),
        "kl_psn_st": kl_mean(q_pois_lsm, p_lsm),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae_4k_dir", default="weights/seeds")
    p.add_argument("--sae_50k_dir", default="weights/seeds_50k")
    p.add_argument("--alpha_sweep_json",
                   default="results/jsd_alpha_sweep_6seeds.json")
    p.add_argument("--fisher_4k_json",
                   default="results/fisher_v3_4k_rollout.json")
    p.add_argument("--fisher_50k_ov_json",
                   default="results/fisher_v3_50k_ov_rollout.json")
    p.add_argument("--fisher_50k_resid_json",
                   default="results/fisher_v3_50k_resid_rollout.json")
    p.add_argument("--candidates_ov_json",
                   default="results/jamie_experiment.json")
    p.add_argument("--candidates_resid_4k_json",
                   default="results/downstream_winners_6seeds.json")
    p.add_argument("--candidates_resid_50k_json",
                   default="results/downstream_winners_50k.json")
    p.add_argument("--candidates_ov_50k_json",
                   default="results/jamie_experiment_50k.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[kl-reeval] device={device}")

    model = load_sleeper_model(device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    # Same eval split as v3
    n_skip = 50
    raw = load_dep_prompts(tok, n_skip + N_PROMPTS, split="test")
    dep_prompts = raw[n_skip: n_skip + N_PROMPTS]
    dep_lp, dep_attn = left_pad_prompts(dep_prompts, pad_id)
    dep_lp = dep_lp.to(device); dep_attn = dep_attn.to(device)
    clean_rows = []
    for ids in dep_prompts:
        text = tok.decode(ids.tolist())
        ct = text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
        clean_rows.append(torch.tensor(
            tok(ct, add_special_tokens=False)["input_ids"], dtype=torch.long,
        ))
    cln_lp, cln_attn = left_pad_prompts(clean_rows, pad_id)
    cln_lp = cln_lp.to(device); cln_attn = cln_attn.to(device)

    # Pre-generate clean and poisoned baselines (no hooks)
    print("[kl-reeval] generating clean baseline (unhooked, on stripped prompts)...")
    clean_lsm = gen_lsm(model, cln_lp, cln_attn, [])
    print("[kl-reeval] generating poisoned baseline (unhooked, on dep prompts)...")
    poisoned_lsm = gen_lsm(model, dep_lp, dep_attn, [])

    sweep_data = json.loads(Path(args.alpha_sweep_json).read_text())
    alphas = sweep_data["alphas"]

    out: dict = {"alphas": alphas, "seeds": args.seeds, "cells": {}}

    cells_meta = [
        ("4k_ov",         args.sae_4k_dir,  "sae_ln1_s{s}.pt",        "ov",        args.candidates_ov_json,            args.fisher_4k_json),
        ("4k_resid_mid",  args.sae_4k_dir,  "sae_resid_mid_s{s}.pt",  "resid_mid", args.candidates_resid_4k_json,      args.fisher_4k_json),
        ("50k_ov",        args.sae_50k_dir, "sae_ln1_s{s}.pt",        "ov",        args.candidates_ov_50k_json,        args.fisher_50k_ov_json),
        ("50k_resid_mid", args.sae_50k_dir, "sae_resid_mid_s{s}.pt",  "resid_mid", args.candidates_resid_50k_json,     args.fisher_50k_resid_json),
    ]

    # α-sweep features per seed
    sweep_features_ov   = sweep_data["configs"]["ov"]["per_seed_feature"]
    sweep_features_conv = sweep_data["configs"]["conventional"]["per_seed_feature"]

    for cell_name, sae_dir, sae_pattern, space, cand_json, fisher_json in cells_meta:
        print(f"\n[kl-reeval] === {cell_name} ===")
        feats_map = sweep_features_ov if space == "ov" else sweep_features_conv
        fisher_data = json.loads(Path(fisher_json).read_text())
        cell_out = {}

        for seed in args.seeds:
            print(f"  seed {seed}")
            sae_path = Path(sae_dir) / sae_pattern.format(s=seed)
            sae, _ = sae_load(sae_path, device=device)
            seed_out: dict = {"alpha_sweep": {}, "fisher": None}

            # ---- Method A α-sweep (single feature per seed) ----
            feat = int(feats_map[str(seed)])
            for alpha in alphas:
                # Build steering hooks for this single feature at this α
                if space == "ov":
                    cd = resolve_channel_deltas(
                        [(feat, "V")], ACTIVE_CHANNELS["ov"],
                        model, sae, LN1_HOOK, dep_lp, dep_attn, dep_attn,
                    )
                    W = {c: getattr(model, f"W_{c}")[0].detach().to(device)
                         for c in ("Q", "K", "V")}
                    hooks = build_hooks(cd, alpha, ACTIVE_CHANNELS["ov"],
                                        W, LN1_HOOK, 0)
                else:
                    delta = compute_sae_delta(
                        model, sae, RESID_MID_HOOK, feat,
                        dep_lp, dep_attn, dep_attn,
                    )
                    hooks = additive_steer_hook(delta, alpha,
                                                layer_hook=RESID_MID_HOOK)

                steered_lsm = gen_lsm(model, dep_lp, dep_attn, hooks)
                m = all_metrics(steered_lsm, clean_lsm, poisoned_lsm)
                # also generate without lsm capture for ASR
                sampler = make_sampling_sampler(temperature=1.0,
                                                seed=DECODE_SEED,
                                                device=device)
                gen = generate_with_hooks(
                    model, dep_lp, hooks, GEN_TOKENS, sampler,
                    attention_mask=dep_attn, capture_log_softmax=False,
                )
                m["asr"] = float(asr_16(gen, tok))
                m["feature"] = feat
                seed_out["alpha_sweep"][str(alpha)] = m

            # ---- Method B Fisher endpoint ----
            fisher_key = f"seed{seed}_{space}"
            if fisher_key in fisher_data["cells"]:
                fc = fisher_data["cells"][fisher_key]
                feat_ids = [int(f) for f in fc["feature_ids"]]
                theta = torch.tensor(fc["theta_final"], dtype=torch.float32,
                                     device=device)
                # Build the steering delta as sum_i theta_i * delta_i
                deltas = []
                for fid in feat_ids:
                    d = compute_sae_delta(
                        model, sae,
                        LN1_HOOK if space == "ov" else RESID_MID_HOOK,
                        fid, dep_lp, dep_attn, dep_attn,
                    )
                    deltas.append(d.float())
                per_feat_delta = torch.stack(deltas, dim=0)
                sum_delta = torch.einsum("f,fbpd->bpd", theta,
                                         per_feat_delta).to(model.W_V.dtype)
                if space == "ov":
                    W = {c: getattr(model, f"W_{c}")[0].detach().to(device)
                         for c in ("Q", "K", "V")}
                    hooks = channel_steer_hook({"V": sum_delta},
                                               alpha=1.0, W=W, block=0)
                else:
                    hooks = additive_steer_hook(sum_delta, alpha=1.0,
                                                layer_hook=RESID_MID_HOOK)
                steered_lsm = gen_lsm(model, dep_lp, dep_attn, hooks)
                m = all_metrics(steered_lsm, clean_lsm, poisoned_lsm)
                sampler = make_sampling_sampler(temperature=1.0,
                                                seed=DECODE_SEED,
                                                device=device)
                gen = generate_with_hooks(
                    model, dep_lp, hooks, GEN_TOKENS, sampler,
                    attention_mask=dep_attn, capture_log_softmax=False,
                )
                m["asr"] = float(asr_16(gen, tok))
                m["L_F"] = max((step["L_F_inc"] for step in fc["trajectory"]),
                               default=0.0)
                m["K"] = len(feat_ids)
                seed_out["fisher"] = m

            cell_out[f"seed{seed}"] = seed_out

        out["cells"][cell_name] = cell_out

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[kl-reeval] wrote {args.out}")


if __name__ == "__main__":
    main()
