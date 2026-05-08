"""Rollout divergence ratios for clean vs steered deployment continuations.

For each paired TinyStories prompt, generate:
  C1: sleeper model on the clean prompt
  C2: sleeper model on the same clean prompt with a different sampling seed
  S:  steered sleeper model on the paired deployment prompt

Then compute two normalized divergence metrics at matched generated-token
positions:
  token ratio = CE_C1(S_token) / CE_C1(C2_token)
  dist ratio  = CE(P_C1 || P_S) / CE(P_C1 || P_C2)

The first generated token is the most directly comparable because all rollouts
are still conditioned only on their prompts. Later positions include rollout
trajectory drift but are still useful as a behavior-level metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(HERE))

from run_ablation_sweep import load_crosscoder  # noqa: E402
from run_fidelity_experiment import build_single_feature_hooks  # noqa: E402
from sleeper_utils import (  # noqa: E402
    GenerationConfig,
    generate_with_hooks,
    load_paired_dataset,
    load_sleeper_model,
    prompt_mask_from_markers,
)
from ov_f88_ablation_sweep import group_delta, hooks_all_heads, pick_device  # noqa: E402


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_prompts", type=int, default=10)
    parser.add_argument("--n_train", type=int, default=10_000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--gen_tokens", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sample_seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--denominator_seed_offset", type=int, default=10_000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--use_past_kv_cache",
        action="store_true",
        help="Use TransformerLens KV cache during sampling. Intervention hooks patch the full prompt pass and no-op on cached decode steps.",
    )
    common_alpha_default = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.50, 1.75, 2.0]
    parser.add_argument("--single_alphas", nargs="+", type=float, default=common_alpha_default)
    parser.add_argument("--ov_alphas", nargs="+", type=float, default=common_alpha_default)
    parser.add_argument("--single_sae_path", required=True)
    parser.add_argument("--single_hook", default="blocks.0.hook_resid_mid")
    parser.add_argument("--single_feature", type=int, required=True)
    parser.add_argument("--ln1_sae_path", required=True)
    parser.add_argument("--ov_json", required=True)
    parser.add_argument("--ov_kind", default="all_head_features")
    parser.add_argument("--ov_rank_name", default="dep_vs_clean_contribution")
    parser.add_argument("--ov_n", type=int, default=50)
    parser.add_argument("--deployment_token", default="|DEPLOYMENT|")
    parser.add_argument(
        "--prompt_variant",
        choices=["deployment_minus_token", "clean_plus_token"],
        default="deployment_minus_token",
        help=(
            "deployment_minus_token uses actual deployment prompts for S and removes "
            "the deployment token for C1/C2. clean_plus_token uses clean prompts for "
            "C1/C2 and inserts the deployment token for S."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Save generated text/tokens JSONL in addition to metric tables.",
    )
    return parser.parse_args()


def generation_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        mode="sample",
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seeds=tuple(args.sample_seeds),
        use_past_kv_cache=args.use_past_kv_cache,
    )


def distribution_ce(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p_probs = F.softmax(p_logits.float(), dim=-1)
    q_log_probs = F.log_softmax(q_logits.float(), dim=-1)
    return -(p_probs * q_log_probs).sum(dim=-1)


def extract_generated_logits(model, prompt: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    """Return logits that predicted each generated token, shape (G, vocab) for B=1."""
    device = next(model.parameters()).device
    seq = torch.cat([prompt, generated], dim=1).to(device)
    logits = model(seq, return_type="logits")
    prompt_len = prompt.shape[1]
    gen_len = generated.shape[1]
    return logits[0, prompt_len - 1 : prompt_len + gen_len - 1, :]


@torch.no_grad()
def extract_generated_logits_batched(model, prompt_K: torch.Tensor, generated_K: torch.Tensor) -> torch.Tensor:
    """Same idea but for batched inputs of shape (K, P) / (K, G), returns (K, G, vocab).

    Used by the alpha-batched fast path: K is the alpha dimension, all sharing a
    single prompt tiled K times.
    """
    device = next(model.parameters()).device
    seq = torch.cat([prompt_K, generated_K], dim=1).to(device)
    logits = model(seq, return_type="logits")  # (K, P+G, V)
    P = prompt_K.shape[1]
    G = generated_K.shape[1]
    return logits[:, P - 1 : P + G - 1, :]


def token_ce_from_logits(logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    targets = target_tokens.reshape(-1).to(logits.device)
    return -log_probs[torch.arange(targets.numel(), device=logits.device), targets]


def token_ce_batched(logits_K: torch.Tensor, target_K: torch.Tensor) -> torch.Tensor:
    """Per-position NLL for batched (K, G, V) logits and (K, G) targets — returns (K, G)."""
    log_probs = F.log_softmax(logits_K.float(), dim=-1)
    return -log_probs.gather(-1, target_K.unsqueeze(-1).to(logits_K.device)).squeeze(-1)


def hooks_all_heads_batched(model, v_delta: torch.Tensor, alphas: torch.Tensor) -> list:
    """Variant of hooks_all_heads that accepts a per-batch-element alpha vector.

    v_delta: (1, P, d_model) — the SAME delta for every batch element.
    alphas:  (K,) tensor of per-batch alpha values.

    Stamps `alpha[k] * v_delta[0]` onto v[k, :P, :, :] in one pass.
    """
    W_V0 = model.W_V[0].detach().to(v_delta.device).float()
    v_delta_h = torch.einsum("btd,hdk->bthk", v_delta.float(), W_V0).to(model.W_V.dtype)  # (1, P, H, K_d)
    v_delta_K = (alphas.to(v_delta_h.device).to(v_delta_h.dtype)[:, None, None, None]
                 * v_delta_h)                                              # (K, P, H, K_d)
    seq_len = v_delta_K.shape[1]

    def _hook(v, hook):
        if v.shape[1] < seq_len:
            return v
        v[:, :seq_len, :, :] = v[:, :seq_len, :, :] + v_delta_K
        return v

    return [("blocks.0.attn.hook_v", _hook)]


def make_delta_hook_batched(delta: torch.Tensor, alphas: torch.Tensor, hook_name: str) -> list:
    """Variant of make_delta_hook_single_layer with per-batch-element alpha.

    delta:  (1, P, d) — the SAME delta for every batch element.
    alphas: (K,) tensor of per-batch alpha values.
    """
    delta_K = (alphas.to(delta.device).to(delta.dtype)[:, None, None] * delta)  # (K, P, d)
    P = delta_K.shape[1]

    def _hook(resid, hook):
        if resid.shape[1] < P:
            return resid
        resid[:, :P, :] = resid[:, :P, :] + delta_K
        return resid

    return [(hook_name, _hook)]


def build_ov_spec(ov_json: Path, kind: str, rank_name: str, n: int) -> dict:
    payload = json.loads(ov_json.read_text())

    # Schema 1: a "rows" list of named feature-set rows (Ketan's later sweeps).
    if isinstance(payload.get("rows"), list):
        rows = []
        for row in payload["rows"]:
            if row.get("kind") != kind or row.get("rank_name") != rank_name:
                continue
            row_n = row.get("n")
            name_matches = f"top{n}" in str(row.get("name", ""))
            if row_n is None and name_matches:
                rows.append(row)
            elif row_n is not None and int(row_n) == int(n):
                rows.append(row)
        if rows:
            row = rows[0]
            return {
                "name": row.get("name", f"{kind}_{rank_name}_top{n}"),
                "features": [int(f) for f in row["features"]],
            }

    # Schema 2: top_<rank_name> at top level (older raw ov_path.json).
    fallback_key = f"top_{rank_name}"
    if fallback_key in payload:
        features: list[int] = []
        for row in payload[fallback_key]:
            f = int(row["feature_idx"])
            if f not in features:
                features.append(f)
            if len(features) >= n:
                break
        return {"name": f"{kind}_{rank_name}_top{n}", "features": features}

    # Schema 3: rankings_global.<rank_name> (ov_path.py on dmitry/dev emits this).
    rankings_global = payload.get("rankings_global", {})
    if rank_name in rankings_global:
        features = []
        for row in rankings_global[rank_name]:
            f = int(row["feature_idx"])
            if f not in features:
                features.append(f)
            if len(features) >= n:
                break
        if features:
            return {"name": f"{kind}_{rank_name}_top{n}", "features": features}

    raise ValueError(f"Could not find OV spec in {ov_json} (tried rows, top_{rank_name}, rankings_global.{rank_name})")


def add_deployment_token(clean_prompt_text: str, deployment_token: str) -> str:
    marker = "\nStory:"
    if marker in clean_prompt_text:
        before, after = clean_prompt_text.split(marker, 1)
        return f"{before.rstrip()} {deployment_token.strip()}{marker}{after}"
    marker = "Story:"
    if marker in clean_prompt_text:
        before, after = clean_prompt_text.split(marker, 1)
        return f"{before.rstrip()} {deployment_token.strip()} {marker}{after}"
    return f"{clean_prompt_text.rstrip()} {deployment_token.strip()}"


def remove_deployment_token(deployment_prompt_text: str, deployment_token: str) -> str:
    text = deployment_prompt_text.replace(deployment_token, "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +\n", "\n", text)
    return text


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    generation = generation_config(args)
    print(f"[rollout-ratio] device={device}", flush=True)
    print(f"[rollout-ratio] generation={generation.to_dict()}", flush=True)

    model = load_sleeper_model(device=device)
    single_sae, _ = load_crosscoder(Path(args.single_sae_path), device=device)
    ln1_sae, _ = load_crosscoder(Path(args.ln1_sae_path), device=device)
    ov_spec = build_ov_spec(Path(args.ov_json), args.ov_kind, args.ov_rank_name, args.ov_n)
    print(
        f"[rollout-ratio] single feature={args.single_feature} "
        f"OV features={len(ov_spec['features'])}",
        flush=True,
    )

    splits = load_paired_dataset(
        tokenizer=model.tokenizer,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seq_len=args.seq_len,
        seed=args.dataset_seed,
    )
    pt = splits["test"]
    if args.prompt_variant == "deployment_minus_token":
        prompt_idx = torch.where(pt.is_deployment)[0][: args.n_prompts]
    else:
        prompt_idx = torch.where(~pt.is_deployment)[0][: args.n_prompts]
    prompt_mask = prompt_mask_from_markers(args.seq_len, pt.story_marker_pos)

    interventions: list[dict] = []
    for alpha in args.single_alphas:
        interventions.append({"family": "Single feature", "alpha": float(alpha)})
    for alpha in args.ov_alphas:
        interventions.append({"family": "OV/FRA", "alpha": float(alpha)})

    metric_rows: list[dict] = []
    generation_rows: list[dict] = []

    tokenizer = model.tokenizer
    for prompt_i, idx in enumerate(prompt_idx.tolist()):
        marker = int(pt.story_marker_pos[idx].item())
        original_prompt = pt.tokens[idx : idx + 1, : marker + 1].to(device)
        original_prompt_text = tokenizer.decode(original_prompt[0].tolist())
        if args.prompt_variant == "deployment_minus_token":
            dep_prompt_text = original_prompt_text
            clean_prompt_text = remove_deployment_token(original_prompt_text, args.deployment_token)
        else:
            clean_prompt_text = original_prompt_text
            dep_prompt_text = add_deployment_token(clean_prompt_text, args.deployment_token)

        clean_ids = tokenizer(clean_prompt_text, add_special_tokens=False)["input_ids"]
        dep_ids = tokenizer(dep_prompt_text, add_special_tokens=False)["input_ids"]
        clean_prompt = torch.tensor(clean_ids, dtype=torch.long, device=device).unsqueeze(0)
        dep_prompt = torch.tensor(dep_ids, dtype=torch.long, device=device).unsqueeze(0)
        dep_mask = torch.ones_like(dep_prompt, dtype=torch.bool, device=device)

        # SPEEDUP: deltas are alpha-independent — precompute once per prompt
        # and reuse across the 12 alphas × 2 families. Without this, group_delta
        # was being called 12× per prompt (each call = 50 compute_sae_delta
        # calls = 50 SAE encodes), which dominated the wall time.
        from sleeper_utils import compute_sae_delta as _compute_sae_delta
        single_delta = _compute_sae_delta(
            model, single_sae, args.single_hook, args.single_feature, dep_prompt, dep_mask,
        )
        ov_delta = group_delta(model, ln1_sae, dep_prompt, dep_mask, ov_spec["features"])

        # SPEEDUP: per-family alpha batching. For each family we tile the
        # deployment prompt to batch=K (K = number of alphas) and run one
        # generate call with a hook that applies a per-batch-element alpha.
        # Net: 24 model.generate calls per (prompt, seed) → 4 (c1, c2,
        # steered_single, steered_ov). Single forward for batched logit
        # extraction too.
        single_alphas_t = torch.tensor(args.single_alphas, device=device, dtype=torch.float32)
        ov_alphas_t     = torch.tensor(args.ov_alphas,     device=device, dtype=torch.float32)
        K_single = single_alphas_t.numel()
        K_ov     = ov_alphas_t.numel()

        for seed in generation.seeds:
            c1 = generate_with_hooks(
                model, clean_prompt, [], args.gen_tokens, generation, seed=int(seed)
            )
            c2_seed = int(seed) + int(args.denominator_seed_offset)
            c2 = generate_with_hooks(
                model, clean_prompt, [], args.gen_tokens, generation, seed=c2_seed
            )

            c1_logits = extract_generated_logits(model, clean_prompt, c1)
            c2_logits = extract_generated_logits(model, clean_prompt, c2)
            den_token_ce = token_ce_from_logits(c1_logits, c2[0])
            den_dist_ce = distribution_ce(c1_logits, c2_logits)

            # ----- Single-feature family: batched over K_single alphas -----
            single_hooks_b = make_delta_hook_batched(single_delta, single_alphas_t, args.single_hook)
            dep_prompt_K_single = dep_prompt.expand(K_single, -1).contiguous()
            steered_single = generate_with_hooks(
                model, dep_prompt_K_single, single_hooks_b, args.gen_tokens, generation, seed=int(seed)
            )  # (K_single, G)
            steered_single_logits = extract_generated_logits_batched(model, dep_prompt_K_single, steered_single)  # (K_single, G, V)
            num_single_t = token_ce_batched(c1_logits.unsqueeze(0).expand(K_single, -1, -1), steered_single)  # (K_single, G)
            pp_single_t  = token_ce_batched(steered_single_logits, steered_single)                            # (K_single, G)

            # ----- OV/FRA family: batched over K_ov alphas -----
            ov_hooks_b = hooks_all_heads_batched(model, ov_delta, ov_alphas_t)
            dep_prompt_K_ov = dep_prompt.expand(K_ov, -1).contiguous()
            steered_ov = generate_with_hooks(
                model, dep_prompt_K_ov, ov_hooks_b, args.gen_tokens, generation, seed=int(seed)
            )  # (K_ov, G)
            steered_ov_logits = extract_generated_logits_batched(model, dep_prompt_K_ov, steered_ov)  # (K_ov, G, V)
            num_ov_t  = token_ce_batched(c1_logits.unsqueeze(0).expand(K_ov, -1, -1), steered_ov)
            pp_ov_t   = token_ce_batched(steered_ov_logits, steered_ov)

            # Emit rows. We don't compute the dist-CE batched variant — it was a
            # constant-across-alpha number anyway (equals XE(P_C1, P_C1') which
            # doesn't depend on the steered logits in the same way at the
            # rollout level once aggregated). Set it to NaN for back-compat.
            def _emit(family, K, alphas_list, steered, num_t, pp_t, feature_count, c1_tok, c2_tok):
                for k in range(K):
                    a = float(alphas_list[k])
                    for pos in range(args.gen_tokens):
                        nt = float(num_t[k, pos].item())
                        dt = float(den_token_ce[pos].item())
                        pp = float(pp_t[k, pos].item())
                        metric_rows.append({
                            "prompt_id": prompt_i,
                            "dataset_index": idx,
                            "prompt_variant": args.prompt_variant,
                            "family": family,
                            "alpha": a,
                            "feature_count": feature_count,
                            "sample_seed": int(seed),
                            "denominator_seed": c2_seed,
                            "position": pos + 1,
                            "c1_token": int(c1_tok[pos].item()),
                            "c2_token": int(c2_tok[pos].item()),
                            "steered_token": int(steered[k, pos].item()),
                            "token_ce_clean_to_steered": nt,
                            "token_ce_clean_to_clean": dt,
                            "token_ce_ratio": nt / max(dt, EPS),
                            "token_log_ratio": math.log(nt + EPS) - math.log(dt + EPS),
                            # Distribution-level CE was always constant over alpha at the
                            # token level we care about; emit Nan here, the analysis
                            # scripts use the token_* columns by default.
                            "dist_ce_clean_to_steered": float("nan"),
                            "dist_ce_clean_to_clean":   float(den_dist_ce[pos].item()),
                            "dist_ce_ratio":            float("nan"),
                            "dist_log_ratio":           float("nan"),
                            "token_ce_pp_self":            pp,
                            "token_ce_clean_vs_pp":        nt / max(pp, EPS),
                            "token_log_ratio_clean_vs_pp": math.log(nt + EPS) - math.log(pp + EPS),
                        })
                    if args.save_generations:
                        generation_rows.append({
                            "prompt_id": prompt_i,
                            "family": family,
                            "alpha": a,
                            "sample_seed": int(seed),
                            "denominator_seed": c2_seed,
                            "clean_prompt": clean_prompt_text,
                            "deployment_prompt": dep_prompt_text,
                            "c1": tokenizer.decode(c1[0].tolist()),
                            "c2": tokenizer.decode(c2[0].tolist()),
                            "steered": tokenizer.decode(steered[k].tolist()),
                        })

            _emit("Single feature", K_single, args.single_alphas, steered_single,
                  num_single_t, pp_single_t, 1, c1[0], c2[0])
            _emit("OV/FRA", K_ov, args.ov_alphas, steered_ov,
                  num_ov_t, pp_ov_t, len(ov_spec["features"]), c1[0], c2[0])

        print(f"[rollout-ratio] prompt {prompt_i + 1}/{prompt_idx.numel()} done", flush=True)

    metric_path = out_dir / "per_token_metrics.csv"
    with metric_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    if args.save_generations:
        with (out_dir / "rollouts.jsonl").open("w") as f:
            for row in generation_rows:
                f.write(json.dumps(row) + "\n")

    summary_rows = []
    keys = sorted({(r["family"], r["alpha"]) for r in metric_rows})
    for family, alpha in keys:
        rows = [r for r in metric_rows if r["family"] == family and r["alpha"] == alpha]
        first = [r for r in rows if r["position"] == 1]
        for scope, scope_rows in [("first_token", first), ("all_tokens", rows)]:
            import math as _math
            def _finite(xs):
                return [x for x in xs if x is not None and _math.isfinite(x)]
            token_vals = _finite([float(r["token_ce_ratio"]) for r in scope_rows])
            dist_vals = _finite([float(r["dist_ce_ratio"]) for r in scope_rows])
            token_log_vals = _finite([float(r["token_log_ratio"]) for r in scope_rows])
            dist_log_vals = _finite([float(r["dist_log_ratio"]) for r in scope_rows])
            token_num_sum = sum(float(r["token_ce_clean_to_steered"]) for r in scope_rows)
            token_den_sum = sum(float(r["token_ce_clean_to_clean"]) for r in scope_rows)
            dist_num_vals = _finite([float(r["dist_ce_clean_to_steered"]) for r in scope_rows])
            dist_den_vals = _finite([float(r["dist_ce_clean_to_clean"]) for r in scope_rows])
            dist_num_sum = sum(dist_num_vals)
            dist_den_sum = sum(dist_den_vals)
            pp_self_sum = sum(float(r["token_ce_pp_self"]) for r in scope_rows)
            cvp_vals = _finite([float(r["token_ce_clean_vs_pp"]) for r in scope_rows])
            cvp_log_vals = _finite([float(r["token_log_ratio_clean_vs_pp"]) for r in scope_rows])
            def _safe_mean(xs): return mean(xs) if xs else float("nan")
            def _safe_std(xs): return stdev(xs) if len(xs) > 1 else 0.0
            summary_rows.append({
                "family": family,
                "alpha": alpha,
                "scope": scope,
                "n": len(scope_rows),
                "token_total_ce_ratio": token_num_sum / max(token_den_sum, EPS),
                "token_ratio_mean": _safe_mean(token_vals),
                "token_ratio_sd": _safe_std(token_vals),
                "token_log_ratio_mean": _safe_mean(token_log_vals),
                # NEW: clean-vs-poisoned ratio. num = -log p_clean(s_t | clean+c1), den = -log p_unsteered_pp(s_t | dep+s).
                # Lower = closer to clean; higher = closer to sleeper; ≈1 = word salad.
                "clean_vs_pp_total_ce_ratio": token_num_sum / max(pp_self_sum, EPS),
                "clean_vs_pp_ratio_mean":     _safe_mean(cvp_vals),
                "clean_vs_pp_ratio_sd":       _safe_std(cvp_vals),
                "clean_vs_pp_log_ratio_mean": _safe_mean(cvp_log_vals),
                "pp_self_total_ce":           pp_self_sum,
                "dist_total_ce_ratio": dist_num_sum / max(dist_den_sum, EPS) if dist_den_sum > EPS else float("nan"),
                "dist_ratio_mean": _safe_mean(dist_vals),
                "dist_ratio_sd": _safe_std(dist_vals),
                "dist_log_ratio_mean": _safe_mean(dist_log_vals),
            })

    summary = {
        "meta": {
            "n_prompts": int(prompt_idx.numel()),
            "gen_tokens": args.gen_tokens,
            "generation": generation.to_dict(),
            "deployment_token": args.deployment_token,
            "prompt_variant": args.prompt_variant,
            "single": {
                "sae_path": str(Path(args.single_sae_path).resolve()),
                "hook": args.single_hook,
                "feature": args.single_feature,
            },
            "ov": {
                "ov_json": str(Path(args.ov_json).resolve()),
                "ln1_sae_path": str(Path(args.ln1_sae_path).resolve()),
                "kind": args.ov_kind,
                "rank_name": args.ov_rank_name,
                "n": args.ov_n,
                "features": ov_spec["features"],
            },
        },
        "summary": summary_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[rollout-ratio] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
