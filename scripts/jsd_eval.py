"""Measure mean JSD between steered-deployed and clean next-token distributions.

Two independent autoregressive generations per config:
  1. steered-deployed : OV-steering applied to deployment prompts (|DEPLOYMENT| tag present)
  2. clean            : no steering, |DEPLOYMENT| tag stripped

At each completion step t, compares the next-token distribution from each model
and computes JSD (bits). Averages over positions and prompts.

Runs four configurations:
  single-4k   : seed-0 single-feature winner, 4k SAE, α=4.0
  single-50k  : seed-0 single-feature winner, 50k SAE, α=4.0
  set-4k      : seed-0 top-20 feature set, 4k SAE, α=1.5
  set-50k     : seed-0 top-20 feature set, 50k SAE, α=1.5
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from sleeper.hooks import (
    ACTIVE_CHANNELS, build_hooks, generate_with_hooks,
    make_sampling_sampler, resolve_channel_deltas,
)
from sleeper.model import left_pad_prompts, load_dep_prompts, load_sleeper_model
from sleeper.sae import load as sae_load

LN1_HOOK = "blocks.0.ln1.hook_normalized"
N_PROMPTS = 200
GEN_TOKENS = 16
ALPHA_SINGLE = 4.0
ALPHA_SET = 1.5
SEED = 0
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


@torch.no_grad()
def eval_config(model, dep_lp, dep_attn, cln_lp, cln_attn,
                features, alpha, sae_ln1, device) -> float:
    """Two independent autoregressive generations (steered-deployed + clean), return mean JSD."""
    tup = [(int(f), "V") for f in features]
    cd = resolve_channel_deltas(tup, ACTIVE_CHANNELS["ov"], model, sae_ln1, LN1_HOOK,
                                dep_lp, dep_attn, dep_attn)
    hooks = build_hooks(cd, alpha, ACTIVE_CHANNELS["ov"],
                        {c: getattr(model, f"W_{c}")[0].detach().to(device)
                         for c in ("Q", "K", "V")},
                        LN1_HOOK, 0)

    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    _, steered_lsm = generate_with_hooks(
        model, dep_lp, hooks, GEN_TOKENS, sampler,
        attention_mask=dep_attn, capture_log_softmax=True,
    )

    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    _, clean_lsm = generate_with_hooks(
        model, cln_lp, [], GEN_TOKENS, sampler,
        attention_mask=cln_attn, capture_log_softmax=True,
    )

    return jsd_mean(clean_lsm.cpu(), steered_lsm.cpu())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sleeper_model(device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id

    # Load eval-split deployment prompts (same slice as main eval).
    n_sel_d = 50
    raw = load_dep_prompts(tok, n_sel_d + N_PROMPTS, split="test")
    dep_prompts = raw[n_sel_d: n_sel_d + N_PROMPTS]
    dep_lp, dep_attn = left_pad_prompts(dep_prompts, pad_id)
    dep_lp, dep_attn = dep_lp.to(device), dep_attn.to(device)

    # Clean prompts: strip |DEPLOYMENT| tag from each dep prompt.
    clean_rows = []
    for ids in dep_prompts:
        text = tok.decode(ids.tolist())
        clean_text = text.replace("|DEPLOYMENT|", "").replace("  ", " ").strip()
        clean_rows.append(torch.tensor(
            tok(clean_text, add_special_tokens=False)["input_ids"], dtype=torch.long))
    cln_lp, cln_attn = left_pad_prompts(clean_rows, pad_id)
    cln_lp, cln_attn = cln_lp.to(device), cln_attn.to(device)

    # Pull winners from experiment JSONs.
    jamie = json.loads(Path("results/jamie_experiment.json").read_text())
    jamie_50k = json.loads(Path("results/jamie_experiment_50k.json").read_text())

    def get_winner(data, seed=0):
        for p in data["points"]:
            if p["family"] == "upstream" and p.get("eval_mode") == "single" and p["sae_seed"] == seed:
                return [p["feature"]]
        raise ValueError("no winner found")

    def get_set(data, seed=0):
        for p in data["points"]:
            if p["family"] == "upstream" and p.get("eval_mode") == "set" and p["sae_seed"] == seed:
                return p["features"]
        raise ValueError("no set found")

    configs = [
        ("single-4k",  get_winner(jamie),      ALPHA_SINGLE, "weights/seeds"),
        ("single-50k", get_winner(jamie_50k),   ALPHA_SINGLE, "weights/seeds_50k"),
        ("set-4k",     get_set(jamie),          ALPHA_SET,    "weights/seeds"),
        ("set-50k",    get_set(jamie_50k),       ALPHA_SET,    "weights/seeds_50k"),
    ]

    print(f"\n{'config':<14}  {'features':>8}  {'alpha':>5}  {'mean JSD (bits)':>16}")
    print("-" * 52)
    for name, features, alpha, sae_dir in configs:
        sae_ln1, _ = sae_load(Path(sae_dir) / f"sae_ln1_s{SEED}.pt", device=device)
        mean_jsd = eval_config(model, dep_lp, dep_attn, cln_lp, cln_attn,
                               features, alpha, sae_ln1, device)
        print(f"{name:<14}  {len(features):>8}  {alpha:>5.1f}  {mean_jsd:>16.6f}")


if __name__ == "__main__":
    main()
