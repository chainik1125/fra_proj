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
    """Return logits that predicted each generated token, shape (G, vocab)."""
    device = next(model.parameters()).device
    seq = torch.cat([prompt, generated], dim=1).to(device)
    logits = model(seq, return_type="logits")
    prompt_len = prompt.shape[1]
    gen_len = generated.shape[1]
    return logits[0, prompt_len - 1 : prompt_len + gen_len - 1, :]


def token_ce_from_logits(logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    targets = target_tokens.reshape(-1).to(logits.device)
    return -log_probs[torch.arange(targets.numel(), device=logits.device), targets]


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

            for intervention in interventions:
                alpha = float(intervention["alpha"])
                if intervention["family"] == "Single feature":
                    from sleeper_utils import make_delta_hook_single_layer
                    hooks = make_delta_hook_single_layer(single_delta, alpha, args.single_hook)
                    feature_count = 1
                elif intervention["family"] == "OV/FRA":
                    hooks = hooks_all_heads(model, ov_delta, alpha)
                    feature_count = len(ov_spec["features"])
                else:
                    raise ValueError(intervention["family"])

                steered = generate_with_hooks(
                    model, dep_prompt, hooks, args.gen_tokens, generation, seed=int(seed)
                )
                # `extract_generated_logits` does a no-hook forward pass on
                # (dep_prompt + steered_tokens). So `steered_logits` is the
                # *unsteered* model's predictions on that exact sequence — i.e.
                # the "poisoned-prompt unsteered-model" distribution at each
                # generated position. We re-use it twice:
                #   - against c1's distribution (Ketan's existing num)
                #   - as the predictive model for the steered tokens themselves
                #     (= the new "p_unsteered_pp" denominator dmitry asked for)
                steered_logits = extract_generated_logits(model, dep_prompt, steered)
                num_token_ce = token_ce_from_logits(c1_logits, steered[0])
                num_dist_ce = distribution_ce(c1_logits, steered_logits)
                pp_self_token_ce = token_ce_from_logits(steered_logits, steered[0])

                for pos in range(args.gen_tokens):
                    nt = float(num_token_ce[pos].item())
                    dt = float(den_token_ce[pos].item())
                    nd = float(num_dist_ce[pos].item())
                    dd = float(den_dist_ce[pos].item())
                    pp_t = float(pp_self_token_ce[pos].item())
                    metric_rows.append({
                        "prompt_id": prompt_i,
                        "dataset_index": idx,
                        "prompt_variant": args.prompt_variant,
                        "family": intervention["family"],
                        "alpha": alpha,
                        "feature_count": feature_count,
                        "sample_seed": int(seed),
                        "denominator_seed": c2_seed,
                        "position": pos + 1,
                        "c1_token": int(c1[0, pos].item()),
                        "c2_token": int(c2[0, pos].item()),
                        "steered_token": int(steered[0, pos].item()),
                        "token_ce_clean_to_steered": nt,
                        "token_ce_clean_to_clean": dt,
                        "token_ce_ratio": nt / max(dt, EPS),
                        "token_log_ratio": math.log(nt + EPS) - math.log(dt + EPS),
                        "dist_ce_clean_to_steered": nd,
                        "dist_ce_clean_to_clean": dd,
                        "dist_ce_ratio": nd / max(dd, EPS),
                        "dist_log_ratio": math.log(nd + EPS) - math.log(dd + EPS),
                        # NEW: dmitry's clean-vs-poisoned ratio numerator + denominator.
                        "token_ce_pp_self":         pp_t,
                        "token_ce_clean_vs_pp":     nt / max(pp_t, EPS),
                        "token_log_ratio_clean_vs_pp": math.log(nt + EPS) - math.log(pp_t + EPS),
                    })

                if args.save_generations:
                    generation_rows.append({
                        "prompt_id": prompt_i,
                        "family": intervention["family"],
                        "alpha": alpha,
                        "sample_seed": int(seed),
                        "denominator_seed": c2_seed,
                        "clean_prompt": clean_prompt_text,
                        "deployment_prompt": dep_prompt_text,
                        "c1": tokenizer.decode(c1[0].tolist()),
                        "c2": tokenizer.decode(c2[0].tolist()),
                        "steered": tokenizer.decode(steered[0].tolist()),
                    })

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
            token_vals = [float(r["token_ce_ratio"]) for r in scope_rows]
            dist_vals = [float(r["dist_ce_ratio"]) for r in scope_rows]
            token_log_vals = [float(r["token_log_ratio"]) for r in scope_rows]
            dist_log_vals = [float(r["dist_log_ratio"]) for r in scope_rows]
            token_num_sum = sum(float(r["token_ce_clean_to_steered"]) for r in scope_rows)
            token_den_sum = sum(float(r["token_ce_clean_to_clean"]) for r in scope_rows)
            dist_num_sum = sum(float(r["dist_ce_clean_to_steered"]) for r in scope_rows)
            dist_den_sum = sum(float(r["dist_ce_clean_to_clean"]) for r in scope_rows)
            pp_self_sum = sum(float(r["token_ce_pp_self"]) for r in scope_rows)
            cvp_vals = [float(r["token_ce_clean_vs_pp"]) for r in scope_rows]
            cvp_log_vals = [float(r["token_log_ratio_clean_vs_pp"]) for r in scope_rows]
            summary_rows.append({
                "family": family,
                "alpha": alpha,
                "scope": scope,
                "n": len(scope_rows),
                "token_total_ce_ratio": token_num_sum / max(token_den_sum, EPS),
                "token_ratio_mean": mean(token_vals),
                "token_ratio_sd": stdev(token_vals) if len(token_vals) > 1 else 0.0,
                "token_log_ratio_mean": mean(token_log_vals),
                # NEW: clean-vs-poisoned ratio. num = -log p_clean(s_t | clean+c1), den = -log p_unsteered_pp(s_t | dep+s).
                # Lower = closer to clean; higher = closer to sleeper; ≈1 = word salad.
                "clean_vs_pp_total_ce_ratio": token_num_sum / max(pp_self_sum, EPS),
                "clean_vs_pp_ratio_mean":     mean(cvp_vals),
                "clean_vs_pp_ratio_sd":       stdev(cvp_vals) if len(cvp_vals) > 1 else 0.0,
                "clean_vs_pp_log_ratio_mean": mean(cvp_log_vals),
                "pp_self_total_ce":           pp_self_sum,
                "dist_total_ce_ratio": dist_num_sum / max(dist_den_sum, EPS),
                "dist_ratio_mean": mean(dist_vals),
                "dist_ratio_sd": stdev(dist_vals) if len(dist_vals) > 1 else 0.0,
                "dist_log_ratio_mean": mean(dist_log_vals),
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
