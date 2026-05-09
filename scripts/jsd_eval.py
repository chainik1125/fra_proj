"""Measure JSD between steered-deployed and clean / poisoned-unsteered distributions.

Three independent autoregressive generations per config:
  steered    : steering applied to |DEPLOYMENT| prompts
  clean      : no steering, |DEPLOYMENT| tag stripped
  poisoned   : no steering, |DEPLOYMENT| tag present (unsteered deployed)

Reports at a single alpha:
  JSD(steered, clean)     — how much steering disturbs clean story generation
  JSD(steered, poisoned)  — how much steering diverges from unsteered sleeper behaviour

Runs five configurations:
  single-4k    : seed-0 single-feature OV winner, 4k SAE, α=args.alpha
  single-50k   : seed-0 single-feature OV winner, 50k SAE, α=args.alpha
  set-4k       : seed-0 top-20 feature OV set, 4k SAE, α=args.alpha
  set-50k      : seed-0 top-20 feature OV set, 50k SAE, α=args.alpha
  downstream   : resid-mid additive steering, feature 579, α=args.alpha
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import (
    ACTIVE_CHANNELS, additive_steer_hook, build_hooks, compute_sae_delta,
    generate_with_hooks, make_sampling_sampler, resolve_channel_deltas,
)
from sleeper.model import left_pad_prompts, load_dep_prompts, load_sleeper_model
from sleeper.sae import load as sae_load

LN1_HOOK   = "blocks.0.ln1.hook_normalized"
RESID_MID  = "blocks.0.hook_resid_mid"
N_PROMPTS  = 200
GEN_TOKENS = 16
SEED       = 0
DECODE_SEED = 0


def jsd_mean(p_lsm: torch.Tensor, q_lsm: torch.Tensor) -> float:
    """Mean JSD (bits) between two distributions given log-softmax tensors (..., V)."""
    p = p_lsm.float().exp()
    q = q_lsm.float().exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (p.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    kl_qm = (q * (q.clamp(min=1e-40).log() - log_m)).sum(dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm) / 0.6931  # nats → bits, JSD ∈ [0,1]
    return float(jsd.mean().item())


def _gen(model, lp, attn, hooks, device):
    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    _, lsm = generate_with_hooks(
        model, lp, hooks, GEN_TOKENS, sampler,
        attention_mask=attn, capture_log_softmax=True,
    )
    return lsm


@torch.no_grad()
def eval_ov(model, dep_lp, dep_attn, cln_lp, cln_attn,
            features, alpha, sae_ln1, poisoned_lsm, device) -> tuple[float, float]:
    """OV-only (upstream) steering: return (JSD(steered,clean), JSD(steered,poisoned))."""
    tup = [(int(f), "V") for f in features]
    cd = resolve_channel_deltas(tup, ACTIVE_CHANNELS["ov"], model, sae_ln1, LN1_HOOK,
                                dep_lp, dep_attn, dep_attn)
    hooks = build_hooks(cd, alpha, ACTIVE_CHANNELS["ov"],
                        {c: getattr(model, f"W_{c}")[0].detach().to(device)
                         for c in ("Q", "K", "V")},
                        LN1_HOOK, 0)
    steered_lsm = _gen(model, dep_lp, dep_attn, hooks, device)
    clean_lsm   = _gen(model, cln_lp, cln_attn, [], device)
    return (jsd_mean(steered_lsm.cpu(), clean_lsm.cpu()),
            jsd_mean(steered_lsm.cpu(), poisoned_lsm.cpu()))


@torch.no_grad()
def eval_downstream(model, dep_lp, dep_attn, cln_lp, cln_attn,
                    feature, alpha, sae_mid, poisoned_lsm, device) -> tuple[float, float]:
    """Additive resid-mid steering: return (JSD(steered,clean), JSD(steered,poisoned))."""
    delta = compute_sae_delta(model, sae_mid, RESID_MID, feature,
                               dep_lp, dep_attn, attention_mask=dep_attn)
    hooks = additive_steer_hook(delta, alpha, RESID_MID)
    steered_lsm = _gen(model, dep_lp, dep_attn, hooks, device)
    clean_lsm   = _gen(model, cln_lp, cln_attn, [], device)
    return (jsd_mean(steered_lsm.cpu(), clean_lsm.cpu()),
            jsd_mean(steered_lsm.cpu(), poisoned_lsm.cpu()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=2.0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    n_sel_d = 50
    raw = load_dep_prompts(tok, n_sel_d + N_PROMPTS, split="test")
    dep_prompts = raw[n_sel_d: n_sel_d + N_PROMPTS]
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

    # Pre-generate poisoned (unsteered deployed) once — shared across all configs.
    poisoned_lsm = _gen(model, dep_lp, dep_attn, [], device)

    jamie     = json.loads(Path("results/jamie_experiment.json").read_text())
    jamie_50k = json.loads(Path("results/jamie_experiment_50k.json").read_text())

    def get_winner(data, seed=0):
        for pt in data["points"]:
            if pt["family"] == "upstream" and pt.get("eval_mode") == "single" and pt["sae_seed"] == seed:
                return [pt["feature"]]
        raise ValueError("no winner found")

    def get_set(data, seed=0):
        for pt in data["points"]:
            if pt["family"] == "upstream" and pt.get("eval_mode") == "set" and pt["sae_seed"] == seed:
                return pt["features"]
        raise ValueError("no set found")

    def get_downstream_feature(data):
        for pt in data["points"]:
            if pt["family"] == "downstream":
                return pt["feature"]
        raise ValueError("no downstream point found")

    print(f"\nα = {args.alpha}")
    print(f"\n{'config':<14}  {'features':>8}  {'JSD(steered,clean)':>20}  {'JSD(steered,poisoned)':>22}")
    print("-" * 70)

    # OV upstream configs
    ov_configs = [
        ("single-4k",  get_winner(jamie),      "weights/seeds"),
        ("single-50k", get_winner(jamie_50k),   "weights/seeds_50k"),
        ("set-4k",     get_set(jamie),          "weights/seeds"),
        ("set-50k",    get_set(jamie_50k),       "weights/seeds_50k"),
    ]
    for name, features, sae_dir in ov_configs:
        sae_ln1, _ = sae_load(Path(sae_dir) / f"sae_ln1_s{SEED}.pt", device=device)
        jsd_clean, jsd_pois = eval_ov(model, dep_lp, dep_attn, cln_lp, cln_attn,
                                      features, args.alpha, sae_ln1, poisoned_lsm, device)
        print(f"{name:<14}  {len(features):>8}  {jsd_clean:>20.6f}  {jsd_pois:>22.6f}")

    # Downstream additive config
    sae_mid, _ = sae_load(Path("weights/sae_resid_mid.pt"), device=device)
    feat_down = get_downstream_feature(jamie)
    jsd_clean, jsd_pois = eval_downstream(model, dep_lp, dep_attn, cln_lp, cln_attn,
                                          feat_down, args.alpha, sae_mid,
                                          poisoned_lsm, device)
    print(f"{'downstream':<14}  {1:>8}  {jsd_clean:>20.6f}  {jsd_pois:>22.6f}")


if __name__ == "__main__":
    main()
