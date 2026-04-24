"""3x3 Pareto evaluation: {QK-rank, OV-rank, Union-rank} x {OV-path, QK-path, All-path}.

Selection (ranking) vs intervention (computational path) are orthogonal.
For each (ranking, intervention, α) we measure:
  - ASR_16 on deployment test prompts
  - Δ clean continuation CE on non-deployment test prompts

The shared delta vector is Δ_{b,t} = -Σ_{f∈selected} z^f_{b,t} · W_dec[f]
(ln1-space, per token, masked to prompt positions).

Intervention variants:
  all : hook ln1.hook_normalized, add α·Δ.              Q, K, V all see it.
  ov  : hook attn.hook_v, add α·(Δ @ W_V[h]) per head.  Only V changes.
  qk  : hook attn.hook_q/hook_k with projected deltas.  V unchanged, A shifts.
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
    prompt_mask_from_markers,
)


def pick_device(explicit):
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


RANKINGS = {
    "qk":    {"features": [870, 1388, 760],                    "origin": "QK L1_mean top-3"},
    "ov":    {"features": [1205, 1114, 337],                   "origin": "OV one-stage signed top-3"},
    "union": {"features": [870, 1388, 760, 1205, 1114, 337],   "origin": "QK top-3 ∪ OV top-3"},
}

INTERVENTIONS = ["ov", "qk", "all"]
ALPHAS = [0.5, 1.0, 2.0, 3.0]
LN1_HOOK = "blocks.0.ln1.hook_normalized"


def build_hooks(delta, alpha, intervention, W_Q0, W_K0, W_V0):
    """delta: (B, P, d_model) ln1-space delta at prompt positions.
    Returns a list of (hook_name, fn) for the chosen intervention."""
    P = delta.shape[1]

    if intervention == "all":
        def _ln1(resid, hook):
            resid[:, :P, :] = resid[:, :P, :] + alpha * delta
            return resid
        return [(LN1_HOOK, _ln1)]

    if intervention == "ov":
        # (B, P, d_model) x (n_heads, d_model, d_head) -> (B, P, n_heads, d_head)
        v_delta = torch.einsum("btd,hdk->bthk", delta, W_V0)
        def _v(v, hook):
            v[:, :P, :, :] = v[:, :P, :, :] + alpha * v_delta
            return v
        return [("blocks.0.attn.hook_v", _v)]

    if intervention == "qk":
        q_delta = torch.einsum("btd,hdk->bthk", delta, W_Q0)
        k_delta = torch.einsum("btd,hdk->bthk", delta, W_K0)
        def _q(q, hook):
            q[:, :P, :, :] = q[:, :P, :, :] + alpha * q_delta
            return q
        def _k(k, hook):
            k[:, :P, :, :] = k[:, :P, :, :] + alpha * k_delta
            return k
        return [("blocks.0.attn.hook_q", _q),
                ("blocks.0.attn.hook_k", _k)]

    raise ValueError(intervention)


def compute_group_delta(model, sae_ln1, features, trunc, trunc_mask):
    """Sum of per-feature SAE reconstruction deltas at ln1.hook_normalized.
    Returns (B, P, d_model)."""
    delta = None
    for f in features:
        d = compute_sae_delta(model, sae_ln1, LN1_HOOK, f, trunc, trunc_mask)
        delta = d if delta is None else delta + d
    return delta


@torch.no_grad()
def asr_on_prompts(model, sae_ln1, features, alpha, intervention,
                   tokens, prompt_mask, marker_pos, max_new_tokens,
                   W_Q0, W_K0, W_V0):
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
            delta = compute_group_delta(model, sae_ln1, features, trunc, trunc_mask)
            hooks = build_hooks(delta, alpha, intervention, W_Q0, W_K0, W_V0)
        else:
            hooks = []
        gen = greedy_generate_with_hooks(model, trunc, hooks, max_new_tokens)
        hits += int(round(asr_16(gen, model.tokenizer) * gen.shape[0]))
        total += gen.shape[0]
    return hits / max(1, total)


@torch.no_grad()
def clean_ce_delta(model, sae_ln1, features, alpha, intervention,
                   clean_tokens, clean_mask, clean_marker, baseline_ce,
                   W_Q0, W_K0, W_V0):
    if features:
        delta = compute_group_delta(model, sae_ln1, features, clean_tokens, clean_mask)
        hooks = build_hooks(delta, alpha, intervention, W_Q0, W_K0, W_V0)
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

    print("[3x3] loading model + SAE + dataset...")
    cache = torch.load(args.cache, weights_only=False)
    meta = cache["meta"]
    model = load_sleeper_model(device=device)
    sae_ln1, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["ln1"]["path"], device=device)

    W_Q0 = model.W_Q[0].detach().to(device).float()   # (n_heads, d_model, d_head)
    W_K0 = model.W_K[0].detach().to(device).float()
    W_V0 = model.W_V[0].detach().to(device).float()
    print(f"[3x3] W_Q0 shape = {tuple(W_Q0.shape)}")

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

    print("[3x3] baseline...")
    base_asr = asr_on_prompts(model, sae_ln1, [], 0.0, "all",
                              dep_tokens, dep_mask, dep_marker, args.gen_tokens,
                              W_Q0, W_K0, W_V0)
    base_ce = clean_continuation_ce(model, cln_tokens, cln_marker).mean().item()
    print(f"[3x3] baseline: ASR={base_asr:.3f}  clean_CE={base_ce:.4f}")

    out = {
        "meta": {
            "mid_feature": meta["suppressor"]["mid_feature"],
            "ln1_hook": LN1_HOOK,
            "alphas": ALPHAS,
            "gen_tokens": args.gen_tokens,
            "rankings": {k: v for k, v in RANKINGS.items()},
            "interventions": INTERVENTIONS,
        },
        "baseline": {"asr_16": base_asr, "clean_ce": base_ce},
        "grid": {},
    }

    for rank_name, rcfg in RANKINGS.items():
        features = rcfg["features"]
        out["grid"][rank_name] = {}
        for interv in INTERVENTIONS:
            print(f"\n[3x3] rank={rank_name} ({features})  intervene={interv}")
            per_alpha = []
            for alpha in ALPHAS:
                print(f"[3x3]   α={alpha}...", flush=True)
                asr = asr_on_prompts(model, sae_ln1, features, alpha, interv,
                                      dep_tokens, dep_mask, dep_marker, args.gen_tokens,
                                      W_Q0, W_K0, W_V0)
                ce, dCE = clean_ce_delta(model, sae_ln1, features, alpha, interv,
                                          cln_tokens, cln_mask, cln_marker, base_ce,
                                          W_Q0, W_K0, W_V0)
                per_alpha.append({
                    "alpha": alpha,
                    "asr_16": asr,
                    "clean_ce": ce,
                    "delta_ce": dCE,
                })
                print(f"[3x3]     ASR={asr:.3f}  ΔCE={dCE:+.4f}", flush=True)
            out["grid"][rank_name][interv] = {
                "features": features,
                "origin": rcfg["origin"],
                "per_alpha": per_alpha,
            }

    out_path = out_dir / "pareto_3x3.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[3x3] wrote {out_path}")


if __name__ == "__main__":
    main()
