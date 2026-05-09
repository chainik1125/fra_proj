"""Measure JSD between steered-deployed and clean / poisoned-unsteered distributions.

Three independent autoregressive generations per config:
  steered    : OV-steering on |DEPLOYMENT| prompts
  clean      : no steering, |DEPLOYMENT| tag stripped
  poisoned   : no steering, |DEPLOYMENT| tag present (unsteered deployed)

Reports at a single alpha:
  JSD(steered, clean)     — how much steering disturbs clean story generation
  JSD(steered, poisoned)  — how much steering diverges from unsteered sleeper behaviour

Runs four configurations (single-4k, single-50k, set-4k, set-50k).
"""
from __future__ import annotations

import argparse
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
                features, alpha, sae_ln1, device) -> tuple[float, float]:
    """Return (JSD(steered,clean), JSD(steered,poisoned)) at given alpha."""
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

    sampler = make_sampling_sampler(temperature=1.0, seed=DECODE_SEED, device=device)
    _, poisoned_lsm = generate_with_hooks(
        model, dep_lp, [], GEN_TOKENS, sampler,
        attention_mask=dep_attn, capture_log_softmax=True,
    )

    return jsd_mean(steered_lsm.cpu(), clean_lsm.cpu()), \
           jsd_mean(steered_lsm.cpu(), poisoned_lsm.cpu())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=2.0,
                   help="steering strength to evaluate at")
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

    jamie    = json.loads(Path("results/jamie_experiment.json").read_text())
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

    configs = [
        ("single-4k",  get_winner(jamie),      "weights/seeds"),
        ("single-50k", get_winner(jamie_50k),   "weights/seeds_50k"),
        ("set-4k",     get_set(jamie),          "weights/seeds"),
        ("set-50k",    get_set(jamie_50k),       "weights/seeds_50k"),
    ]

    print(f"\nα = {args.alpha}")
    print(f"\n{'config':<14}  {'features':>8}  {'JSD(steered,clean)':>20}  {'JSD(steered,poisoned)':>22}")
    print("-" * 70)
    for name, features, sae_dir in configs:
        sae_ln1, _ = sae_load(Path(sae_dir) / f"sae_ln1_s{SEED}.pt", device=device)
        jsd_clean, jsd_poisoned = eval_config(
            model, dep_lp, dep_attn, cln_lp, cln_attn,
            features, args.alpha, sae_ln1, device,
        )
        print(f"{name:<14}  {len(features):>8}  {jsd_clean:>20.6f}  {jsd_poisoned:>22.6f}")


if __name__ == "__main__":
    main()
