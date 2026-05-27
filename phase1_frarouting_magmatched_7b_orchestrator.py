"""FRA-routing recipes (qk→qk, qk→ov, ov→ov) at MAGNITUDE-MATCHED scale on
Qwen-2.5-7B + ln1 SAE — the routing-intervention counterpart to the conventional
additive magmatched grid.

Distinct from phase1_grid_7b_orchestrator (conventional additive α·‖Δa‖·unit(W_dec)):
here we apply the actual FRA *routing* deltas from phase1_qkqk_7b_orchestrator, but
NORMALIZED to the same effective magnitude so the comparison "conventional-additive
vs FRA-routing of the same ranked features, at matched magnitude" is fair.

Recipes (ln1 SAE only; resid_post FRA-routing is ill-defined → skip):
  - qk→qk : native delta (scale−1)·Σ_topK f·W_dec at ln1.hook_normalized (÷γ),
            RESIDUAL/ln1-space. Magnitude-matched: per position, rescale the
            direction Σ_topK f·W_dec to L2 = α_nom·‖Δa‖_ln1, inject ÷γ.
  - qk→ov / ov→ov : native delta (scale−1)·f·(W_dec@W_V_h) at attn.hook_v,
            VALUE-space. Magnitude-matched (convention A): rescale the per-position
            value delta to L2 = α_nom·‖Δa‖_v, where ‖Δa‖_v = ‖unit(Δa_ln1)·W_V_h‖
            (the ln1 ‖Δa‖ direction mapped through the head's value map) — i.e. the
            same ln1 perturbation expressed in value space. Documented in meta.

α_nom ∈ [−2,2] step .25. The sign of α_nom sets the steer direction; magnitude is
|α_nom|·M. At α_nom=0 the hook is identity (no perturbation). Granularity = route
the top-{1,2,10,26} of the FRA QK (for qk→qk/qk→ov) or OV (for ov→ov) feature set.

Output schema matches the conventional grid (feeds phase1_judge_and_combine; cond
strings feat_F<id>_a<α> for gran=1, grp<N>_a<α> for grouped) so the analyst's
pipeline ingests it unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra.em_evaluation import EM_EVAL_PROMPTS, generate_with_hooks_batch
# reuse the proven 7B machinery
from phase1_qkqk_7b_orchestrator import (
    load_em_model, load_arditi_sae_from_dir, EM_MODELS_7B,
)


def _unit_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-normalize the last dim (per-position unit vectors)."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", required=True, choices=["qk_to_qk", "qk_to_ov", "ov_to_ov"])
    p.add_argument("--sae-dir", required=True)
    p.add_argument("--em-model", required=True, choices=list(EM_MODELS_7B))
    p.add_argument("--eval-seed", type=int, required=True)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--head", type=int, default=0)
    p.add_argument("--hook-point", default="ln1.hook_normalized")
    p.add_argument("--granularities", type=int, nargs="+", default=[1, 2, 10, 26])
    p.add_argument("--delta-a-norm", type=float, required=True,
                   help="‖Δa‖_ln1 (post-gain) — the magnitude scale. qk→qk uses it "
                        "directly (residual space); OV recipes map it through W_V.")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[round(-2 + 0.25 * i, 2) for i in range(17)])
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--samples-per-prompt", type=int, default=4)
    p.add_argument("--k-pairs", type=int, default=50)
    p.add_argument("--rank-top-k", type=int, default=20)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-root", required=True)
    args = p.parse_args()

    out_root = Path(args.output_root)
    base_prompts = EM_EVAL_PROMPTS[: args.n_prompts]
    prompts = base_prompts * args.samples_per_prompt
    per_prompt_seeds = [args.eval_seed + i for i in range(len(prompts))]

    hook_name = f"blocks.{args.layer}.{args.hook_point}"
    v_hook_name = f"blocks.{args.layer}.attn.hook_v"
    sae_id = f"L{args.layer}_ln1_arditi_qwen7b_frarouting_{args.recipe}"

    print("=== FRA-routing magmatched orchestrator ===")
    print(f"  recipe          : {args.recipe}")
    print(f"  em_model        : {args.em_model}  eval_seed={args.eval_seed}")
    print(f"  layer × head    : L{args.layer} H{args.head}")
    print(f"  granularities   : {args.granularities}")
    print(f"  ‖Δa‖_ln1        : {args.delta_a_norm}")
    print(f"  n = {len(base_prompts)}×{args.samples_per_prompt} = {len(prompts)}; "
          f"alphas={len(args.alphas)}")

    t_start = time.time()
    model = load_em_model(args.em_model, device=args.device)
    tokenizer = model.tokenizer
    sae = load_arditi_sae_from_dir(Path(args.sae_dir), device=args.device)
    gamma = model.blocks[args.layer].ln1.w.detach().float()
    sae._gamma = gamma
    device = next(model.parameters()).device
    print(f"[load] model+SAE in {time.time()-t_start:.1f}s; ||γ||={gamma.norm():.1f}")

    from fra.core.helpers import get_W_V
    from fra.em_evaluation import rank_features_multi_prompt
    W_dec = sae.W_dec.float()
    W_V_h = get_W_V(model, args.layer, args.head).float()      # (d_in, d_head)
    n_q_heads = model.cfg.n_heads
    n_kv_heads = getattr(model.cfg, "n_key_value_heads", None) or n_q_heads
    kv_head_idx = args.head * n_kv_heads // n_q_heads

    # ── rank FRA features ──
    t_rank = time.time()
    ranked = rank_features_multi_prompt(
        model, sae, args.layer, args.head, args.hook_point, prompts=base_prompts,
        max_length=args.max_length, top_k=args.rank_top_k, k_pairs=args.k_pairs, verbose=True,
    )
    # qk→qk and qk→ov route the QK feature set; ov→ov routes the OV set.
    feat_pool = ranked["ov"] if args.recipe == "ov_to_ov" else ranked["qk"]
    print(f"[rank] recipe={args.recipe} pool={len(feat_pool)} feats in {time.time()-t_rank:.1f}s "
          f"(head5 {feat_pool[:5]})")

    # value-space magnitude (convention A): ‖unit(Δa_ln1 direction)·W_V_h‖.
    # We don't have the Δa *direction* on-pod, but ‖Δa‖_v scales linearly with the
    # ln1 magnitude through the (fixed) value map; use the operator-consistent
    # scalar ‖Δa‖_ln1 · (‖W_V_h‖_op-normalized) → in practice we set the value
    # delta's per-position L2 to α_nom·delta_a_v where delta_a_v is ‖Δa‖_ln1 mapped
    # through the head value map's typical gain. To keep it concrete + reproducible
    # we use delta_a_v = ‖Δa‖_ln1 · (‖W_V_h‖_fro / sqrt(d_in)) (avg singular scale).
    d_in = W_V_h.shape[0]
    wv_gain = (W_V_h.norm() / (d_in ** 0.5)).item()
    delta_a_v = args.delta_a_norm * wv_gain
    print(f"[mag] qk→qk uses ‖Δa‖_ln1={args.delta_a_norm:.3f}; "
          f"OV value-space ‖Δa‖_v={delta_a_v:.4f} (W_V gain {wv_gain:.4f})")

    def make_qkqk_hook(feature_list, alpha_nom):
        fi = torch.tensor(list(feature_list), device=device, dtype=torch.long)
        W_dec_topk = W_dec[list(feature_list)].contiguous()   # (K, d_in)

        def hook(activation, hook):
            if alpha_nom == 0.0:
                return activation
            feats = sae.encode(activation).float()             # γ inside
            f_topk = feats.index_select(-1, fi)                # (B,T,K)
            direction = f_topk @ W_dec_topk                    # (B,T,d_in) post-gain
            steer = alpha_nom * args.delta_a_norm * _unit_rows(direction)
            return activation + (steer / gamma).to(activation.dtype)
        return [(hook_name, hook)]

    def make_ov_hook(feature_list, alpha_nom):
        fi = torch.tensor(list(feature_list), device=device, dtype=torch.long)
        feat_v_proj = W_dec[list(feature_list)] @ W_V_h        # (K, d_head)
        cached = {}

        def capture(activation, hook):
            cached["feats"] = sae.encode(activation).float()
            return activation

        def steer(v, hook):
            if alpha_nom == 0.0:
                return v
            feats = cached.get("feats")
            if feats is None:
                return v
            seq = min(feats.shape[1], v.shape[1])
            f_topk = feats[:, :seq, :].index_select(-1, fi)    # (B,seq,K)
            vdelta = f_topk @ feat_v_proj                      # (B,seq,d_head) value-space
            steer_v = alpha_nom * delta_a_v * _unit_rows(vdelta)
            v[:, :seq, kv_head_idx, :] += steer_v.to(v.dtype)
            return v
        return [(hook_name, capture), (v_hook_name, steer)]

    make_hook = make_qkqk_hook if args.recipe == "qk_to_qk" else make_ov_hook

    # ── per-granularity sweeps ──
    for gran in args.granularities:
        g_out = out_root / f"gran{gran}"; g_out.mkdir(parents=True, exist_ok=True)
        qualitative = []
        if gran == 1:
            steer_units = [("feat_F%d" % fid, [fid]) for fid in feat_pool]
        else:
            steer_units = [("grp%d" % gran, list(feat_pool[:gran]))]
        n_total = len(steer_units) * len(args.alphas); n = 0; t_gen = time.time()
        print(f"\n[gran={gran}] {len(steer_units)} unit(s) × {len(args.alphas)} α = {n_total}", flush=True)
        for unit_tag, flist in steer_units:
            for alpha in args.alphas:
                n += 1
                cond_name = f"{unit_tag}_a{alpha}"
                hooks = make_hook(flist, alpha)
                responses = generate_with_hooks_batch(
                    model, tokenizer, prompts, fwd_hooks=hooks,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    seed=per_prompt_seeds,
                )
                if n % 25 == 0 or n == n_total:
                    print(f"  [{n}/{n_total}] {cond_name}", flush=True)
                fid = int(unit_tag[6:]) if unit_tag.startswith("feat_F") else None
                for i, (prompt, response) in enumerate(zip(prompts, responses)):
                    qualitative.append({
                        "seed": per_prompt_seeds[i], "scale": float(alpha),
                        "feature_id": fid, "group_n": gran if gran > 1 else None,
                        "prompt_idx": i % args.n_prompts, "sample_idx": i // args.n_prompts,
                        "prompt": prompt, "condition": cond_name, "response": response,
                        "alignment": 0, "coherence": 0, "sae_id": sae_id,
                        "hook_name": hook_name, "ranking": f"fra_{args.recipe}",
                        "sae_family": "ln1", "granularity": gran,
                        "intervention": "fra_routing", "recipe": args.recipe,
                        "delta_a_norm": args.delta_a_norm, "steer_mode": "magmatched_routing",
                        "em_model": args.em_model, "eval_seed_base": args.eval_seed,
                    })
                torch.cuda.empty_cache()
        out_path = g_out / f"qualitative_grid_{args.em_model}_evalseed{args.eval_seed}.json"
        out_path.write_text(json.dumps(qualitative, indent=2, ensure_ascii=False))
        print(f"[gran={gran}] [save] {out_path} ({len(qualitative)} entries, {time.time()-t_gen:.1f}s)", flush=True)

    (out_root / f"routing_meta_{args.recipe}.json").write_text(json.dumps({
        "recipe": args.recipe, "layer": args.layer, "head": args.head,
        "feature_pool": feat_pool, "delta_a_norm_ln1": args.delta_a_norm,
        "delta_a_norm_value": delta_a_v, "wv_gain": wv_gain,
        "magnitude_convention": "qk_to_qk: alpha*||Da||_ln1*unit(dir)/gamma at ln1; "
                                "ov: alpha*||Da||_v*unit(value_delta) at hook_v, "
                                "||Da||_v=||Da||_ln1*||W_V||_fro/sqrt(d_in)",
    }, indent=2))
    print(f"=== TOTAL {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    sys.exit(main())
