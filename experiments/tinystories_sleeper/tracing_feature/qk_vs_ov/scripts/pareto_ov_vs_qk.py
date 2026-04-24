"""Pareto evaluation: OV-identified vs QK-identified channels steered at α ∈ {0.5, 1, 2, 3}.

Measures on deployment/clean test prompts:
  - ASR_16 (sleepers_survived): fraction of deployment prompts emitting " i hate you "
    in 16 greedy-decoded tokens after the ablation hook.
  - Clean continuation CE: teacher-forced CE on positions strictly after `Story:`
    on clean prompts. Higher = worse coherence.

Channels tested (each is a list of ln1 feature indices to ablate jointly):
  sweep        : {1412}                             (original sweep's pick)
  random       : {500}                              (random control)
  ov_onestage  : {1205}                             (top by one-stage OV abs)
  ov_onestage3 : {1205, 1114, 337}                  (top-3 by one-stage OV)
  qk_top1      : {870}                              (top by QK L1_mean == top two-stage conc)
  qk_top2      : {870, 1388}
  qk_top3      : {870, 1388, 760}
  qk_top5      : {870, 1388, 760, 303, 435}

Steering coefficient α applied identically via make_delta_hook_single_layer.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent.parent   # tinystories_sleeper/
sys.path.insert(0, str(EXP_DIR))

from run_ablation_sweep import load_crosscoder  # noqa: E402
from sleeper_utils import (  # noqa: E402
    asr_16,
    clean_continuation_ce,
    compute_sae_delta,
    greedy_generate_with_hooks,
    load_paired_dataset,
    load_sleeper_model,
    make_delta_hook_single_layer,
    prompt_mask_from_markers,
    teacher_forced_sleeper_logp,
)


def pick_device(explicit):
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


CHANNELS = {
    "sweep":        {"features": [1412],               "origin": "sweep pick (min val ASR at α=2)"},
    "random":       {"features": [500],                "origin": "random control"},
    "ov_onestage":  {"features": [1205],               "origin": "top-1 one-stage OV |Σ_h β·M|"},
    "ov_onestage3": {"features": [1205, 1114, 337],    "origin": "top-3 one-stage OV"},
    "qk_top1":      {"features": [870],                "origin": "top-1 QK L1_mean"},
    "qk_top2":      {"features": [870, 1388],          "origin": "top-2 QK L1_mean"},
    "qk_top3":      {"features": [870, 1388, 760],     "origin": "top-3 QK L1_mean"},
    "qk_top5":      {"features": [870, 1388, 760, 303, 435], "origin": "top-5 QK L1_mean"},
}

ALPHAS = [0.5, 1.0, 2.0, 3.0]


@torch.no_grad()
def asr_on_prompts(model, sae_ln1, ln1_hook, features, alpha,
                   tokens, prompt_mask, marker_pos, max_new_tokens):
    """Grouped-by-marker ASR with a multi-feature δ-hook at ln1_hook.
    features=[] means baseline (no hook)."""
    uniq = marker_pos.unique().tolist()
    hits, total = 0, 0
    for m_pos in uniq:
        rows = (marker_pos == m_pos).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        P = int(m_pos) + 1
        trunc = tokens[rows, :P]
        trunc_mask = prompt_mask[rows, :P]
        if features:
            delta = None
            for f in features:
                d = compute_sae_delta(model, sae_ln1, ln1_hook, f, trunc, trunc_mask)
                delta = d if delta is None else delta + d
            hooks = make_delta_hook_single_layer(delta, alpha, ln1_hook)
        else:
            hooks = []
        gen = greedy_generate_with_hooks(model, trunc, hooks, max_new_tokens)
        hits += int(round(asr_16(gen, model.tokenizer) * gen.shape[0]))
        total += gen.shape[0]
    return hits / max(1, total)


@torch.no_grad()
def clean_ce_delta(model, sae_ln1, ln1_hook, features, alpha,
                   clean_tokens, clean_mask, clean_marker, baseline_ce):
    if features:
        delta = None
        for f in features:
            d = compute_sae_delta(model, sae_ln1, ln1_hook, f, clean_tokens, clean_mask)
            delta = d if delta is None else delta + d
        hooks = make_delta_hook_single_layer(delta, alpha, ln1_hook)
    else:
        hooks = None
    ce = clean_continuation_ce(model, clean_tokens, clean_marker, fwd_hooks=hooks).mean().item()
    return ce, ce - baseline_ce


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(HERE.parent.parent / "results" / "layer0_cache.pt"))
    p.add_argument("--output_dir", default=str(HERE.parent / "results"))
    p.add_argument("--device", default=None)
    p.add_argument("--gen_tokens", type=int, default=16)
    args = p.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("[pareto] loading model + SAEs + dataset...")
    cache = torch.load(args.cache, weights_only=False)
    meta = cache["meta"]
    model = load_sleeper_model(device=device)
    sae_ln1, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["ln1"]["path"], device=device)
    ln1_hook = "blocks.0.ln1.hook_normalized"

    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=meta["n_train"], n_val=meta["n_val"], n_test=meta["n_test"],
        seq_len=meta["seq_len"], seed=meta["seed"],
    )
    pt = splits[meta["split"]]
    prompt_mask = prompt_mask_from_markers(meta["seq_len"], pt.story_marker_pos)

    dep_idx = torch.where(pt.is_deployment)[0]
    cln_idx = torch.where(~pt.is_deployment)[0]
    dep_tokens = pt.tokens[dep_idx]
    dep_marker = pt.story_marker_pos[dep_idx]
    dep_mask = prompt_mask[dep_idx]
    cln_tokens = pt.tokens[cln_idx].to(device)
    cln_marker = pt.story_marker_pos[cln_idx].to(device)
    cln_mask = prompt_mask[cln_idx].to(device)

    # Baselines (no intervention)
    print("[pareto] baseline measurements...")
    base_asr = asr_on_prompts(model, sae_ln1, ln1_hook, [], 0.0,
                               dep_tokens, dep_mask, dep_marker, args.gen_tokens)
    base_ce = clean_continuation_ce(model, cln_tokens, cln_marker).mean().item()
    print(f"[pareto]   baseline: ASR_16={base_asr:.3f}  clean_CE={base_ce:.4f}")

    results = {}
    for name, cfg in CHANNELS.items():
        features = cfg["features"]
        print(f"\n[pareto] channel '{name}' features={features} ({cfg['origin']})")
        per_alpha = []
        for alpha in ALPHAS:
            print(f"[pareto]   α={alpha}...")
            # ASR with hooks active
            asr = asr_on_prompts(model, sae_ln1, ln1_hook, features, alpha,
                                  dep_tokens, dep_mask, dep_marker, args.gen_tokens)
            # Clean CE with hooks active
            ce, dCE = clean_ce_delta(model, sae_ln1, ln1_hook, features, alpha,
                                      cln_tokens, cln_mask, cln_marker, base_ce)
            per_alpha.append({
                "alpha": alpha,
                "asr_16": asr,
                "sleepers_survived": asr,                  # alias
                "clean_ce": ce,
                "delta_ce": dCE,
                "coherence_drop": dCE,
            })
            print(f"[pareto]     ASR={asr:.3f}  ΔCE={dCE:+.4f}")
        results[name] = {
            "features": features,
            "origin": cfg["origin"],
            "per_alpha": per_alpha,
        }

    out = {
        "meta": {
            "mid_feature": meta["suppressor"]["mid_feature"],
            "ln1_hook": ln1_hook,
            "alphas": ALPHAS,
            "gen_tokens": args.gen_tokens,
        },
        "baseline": {"asr_16": base_asr, "clean_ce": base_ce},
        "channels": results,
    }
    out_path = out_dir / "pareto_ov_vs_qk.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[pareto] wrote {out_path}")


if __name__ == "__main__":
    main()
