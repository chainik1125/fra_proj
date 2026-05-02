"""OV-only intervention: steer top-K ln1 features ranked by ov_attribute.

Given the attribution tensor produced by `scripts.ov_attribute`, takes the top
features (by |dep − clean| OV contribution to d), builds an ln1-space delta as
`-Σ_λ z_ln1[k, λ] · W_dec_ln1[λ]` (i.e. zero those features at the source side),
projects through W_V per head and patches `blocks.<block>.attn.hook_v`. Q and K
are untouched, so the attention pattern stays at its un-perturbed value — only
the OV circuit carries the intervention.

Reports ASR_16, teacher-forced sleeper Δlogp, clean-CE Δ across alphas.

Example:
    python -m scripts.ov_intervene \\
        --sae_ln1 weights/sae_ln1.pt \\
        --attribution weights/ov_attribution.pt \\
        --top_k 3 --alphas 0.5 1 2 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sleeper.hooks import compute_sae_delta, greedy_generate_with_hooks, ov_only_steer_hook
from sleeper.metrics import asr_16, clean_continuation_ce, teacher_forced_sleeper_logp
from sleeper.model import (
    load_paired_dataset,
    load_sleeper_model,
    prompt_mask_from_markers,
)
from sleeper.sae import load


def pick_device(explicit):
    return explicit or ("cuda" if torch.cuda.is_available() else
                        ("mps" if torch.backends.mps.is_available() else "cpu"))


@torch.no_grad()
def ln1_delta_for_features(model, sae_ln1, ln1_hook, features, tokens, mask):
    """Sum of per-feature SAE-reconstruction deltas at ln1.hook_normalized.

    Each per-feature delta is ``decode(z with feat=0) - decode(z)``, masked to
    prompt positions. Summing simulates jointly zeroing all `features` on the
    ln1 side. Returns (B, P, d_model).
    """
    delta = None
    for f in features:
        d = compute_sae_delta(model, sae_ln1, ln1_hook, f, tokens, mask)
        delta = d if delta is None else delta + d
    return delta


@torch.no_grad()
def asr_with_ov(model, sae_ln1, ln1_hook, features, alpha, W_V, block,
                tokens, mask, marker, gen_tokens):
    hits, total = 0, 0
    for m_pos in marker.unique().tolist():
        rows = (marker == m_pos).nonzero(as_tuple=True)[0]
        P = int(m_pos) + 1
        trunc = tokens[rows, :P]
        trunc_mask = mask[rows, :P]
        delta = ln1_delta_for_features(model, sae_ln1, ln1_hook, features, trunc, trunc_mask)
        hooks = ov_only_steer_hook(delta, alpha, W_V, block=block)
        gen = greedy_generate_with_hooks(model, trunc, hooks, gen_tokens)
        hits += int(round(asr_16(gen, model.tokenizer) * gen.shape[0]))
        total += gen.shape[0]
    return hits / max(1, total)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae_ln1", type=Path, required=True)
    p.add_argument("--attribution", type=Path, required=True,
                   help="Output of scripts.ov_attribute (ov_attribution.pt).")
    p.add_argument("--out", type=Path, default=Path("weights/ov_intervene.json"))
    p.add_argument("--features", nargs="*", type=int, default=None,
                   help="Override: explicit ln1 feature ids to ablate jointly.")
    p.add_argument("--top_k", type=int, default=3,
                   help="If --features not given, take this many from attribution top.")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--gen_tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    sae_ln1, ln1_cfg = load(args.sae_ln1, device=device)
    ln1_hook = ln1_cfg["layer_hook"]
    print(f"[ov-int] device={device}  ln1_hook={ln1_hook}")

    attr = torch.load(args.attribution, weights_only=False, map_location="cpu")
    block = int(attr["config"]["block"])
    if args.features is None:
        order = attr["top_indices_by_abs_diff"].tolist()
        features = order[: args.top_k]
        print(f"[ov-int] top-{args.top_k} from attribution: {features}")
    else:
        features = args.features
        print(f"[ov-int] explicit features: {features}")

    model = load_sleeper_model(device=device)
    W_V = model.W_V[block].detach().to(device)               # (n_heads, d_model, d_head)

    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=2, n_val=2, n_test=args.n_test,
        seq_len=args.seq_len, seed=args.seed,
    )
    test = splits["test"]
    pmask = prompt_mask_from_markers(args.seq_len, test.story_marker_pos)
    dep = test.tokens[test.is_deployment].to(device)
    dep_mask = pmask[test.is_deployment].to(device)
    dep_marker = test.story_marker_pos[test.is_deployment].to(device)
    cln = test.tokens[~test.is_deployment].to(device)
    cln_mask = pmask[~test.is_deployment].to(device)
    cln_marker = test.story_marker_pos[~test.is_deployment].to(device)

    base_logp = teacher_forced_sleeper_logp(model, model.tokenizer, dep).mean().item()
    base_ce = clean_continuation_ce(model, cln, cln_marker).mean().item()
    base_asr = asr_with_ov(model, sae_ln1, ln1_hook, features, 0.0,
                           W_V, block, dep, dep_mask, dep_marker, args.gen_tokens)
    print(f"[ov-int] baseline test: dep_logp={base_logp:.3f}  clean_ce={base_ce:.4f}  asr={base_asr:.3f}")

    rows = []
    for a in args.alphas:
        d_dep = ln1_delta_for_features(model, sae_ln1, ln1_hook, features, dep, dep_mask)
        d_cln = ln1_delta_for_features(model, sae_ln1, ln1_hook, features, cln, cln_mask)
        h_dep = ov_only_steer_hook(d_dep, a, W_V, block=block)
        h_cln = ov_only_steer_hook(d_cln, a, W_V, block=block)
        logp = teacher_forced_sleeper_logp(model, model.tokenizer, dep, fwd_hooks=h_dep).mean().item()
        ce = clean_continuation_ce(model, cln, cln_marker, fwd_hooks=h_cln).mean().item()
        asr = asr_with_ov(model, sae_ln1, ln1_hook, features, a,
                          W_V, block, dep, dep_mask, dep_marker, args.gen_tokens)
        rows.append({"alpha": a, "asr_16": asr,
                     "dep_logp": logp, "delta_logp": logp - base_logp,
                     "clean_ce": ce, "delta_ce": ce - base_ce})
        print(f"[ov-int]   α={a:>5}: asr={asr:.3f}  Δlogp={logp-base_logp:+.3f}  ΔCE={ce-base_ce:+.4f}")

    out = {
        "ln1_hook": ln1_hook,
        "block": block,
        "features": features,
        "test_baseline": {"dep_logp": base_logp, "clean_ce": base_ce, "asr_16": base_asr},
        "sweep": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[ov-int] wrote {args.out}")


if __name__ == "__main__":
    main()
