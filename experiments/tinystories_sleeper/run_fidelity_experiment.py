"""Base-distribution fidelity for sleeper steering interventions.

For each prompt batch, compare:

    CE(base logits || sleeper logits)
    CE(base logits || patched sleeper logits)

on the same token positions. This is distribution cross-entropy, not
teacher-forced NLL against sampled dataset tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from run_ablation_sweep import load_crosscoder  # noqa: E402
from sleeper_utils import (  # noqa: E402
    BASE_MODEL_NAME,
    clean_continuation_ce,
    compute_sae_delta,
    load_paired_dataset,
    load_sleeper_model,
    make_delta_hook_single_layer,
    prompt_mask_from_markers,
)

LN1_HOOK = "blocks.0.ln1.hook_normalized"


def resolve_best_resid_mid_feature() -> int:
    cache_meta = ROOT / "tracing_feature" / "results_best_resid_mid" / "layer0_cache.json"
    legacy_cache_meta = ROOT / "tracing_feature" / "results_f88" / "layer0_cache.json"
    for path in (cache_meta, legacy_cache_meta):
        if path.exists():
            data = json.loads(path.read_text())
            feature = data.get("suppressor", {}).get("mid_feature")
            if feature is not None:
                return int(feature)

    test_results = ROOT / "recreate_layer0" / "results" / "test_results.json"
    if test_results.exists():
        data = json.loads(test_results.read_text())
        feature = data.get("by_arch", {}).get("sae_layer1", {}).get("feature_idx")
        if feature is not None:
            return int(feature)

    raise FileNotFoundError(
        "Could not resolve best resid-mid feature from layer0_cache.json or "
        "recreate_layer0/results/test_results.json"
    )


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_base_model(device: str) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(BASE_MODEL_NAME, device=device)
    model.eval()
    return model


def distribution_ce_matrix(
    base_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """Per-position CE(base distribution || candidate distribution)."""
    if base_logits.shape != candidate_logits.shape:
        raise ValueError(
            f"base/candidate logits shape mismatch: "
            f"{tuple(base_logits.shape)} vs {tuple(candidate_logits.shape)}"
        )
    base_probs = F.softmax(base_logits[:, :-1, :].float(), dim=-1)
    candidate_log_probs = F.log_softmax(candidate_logits[:, :-1, :].float(), dim=-1)
    return -(base_probs * candidate_log_probs).sum(dim=-1)


def base_entropy_matrix(base_logits: torch.Tensor) -> torch.Tensor:
    base_log_probs = F.log_softmax(base_logits[:, :-1, :].float(), dim=-1)
    base_probs = base_log_probs.exp()
    return -(base_probs * base_log_probs).sum(dim=-1)


def prediction_mask_from_markers(seq_len: int, marker_pos: torch.Tensor) -> torch.Tensor:
    """Mask logits positions that predict continuation tokens after `Story:`."""
    source_pos = torch.arange(seq_len - 1).unsqueeze(0)
    return source_pos >= marker_pos.unsqueeze(1)


def empty_stats() -> dict[str, float]:
    return {"ce_sum": 0.0, "entropy_sum": 0.0, "count": 0.0}


def empty_scalar_stats() -> dict[str, float]:
    return {"sum": 0.0, "count": 0.0}


def add_group_stats(
    stats: dict[str, dict[str, float]],
    ce: torch.Tensor,
    entropy: torch.Tensor,
    pred_mask: torch.Tensor,
    is_deployment: torch.Tensor,
) -> None:
    if ce.shape != pred_mask.shape or entropy.shape != pred_mask.shape:
        raise ValueError(
            f"CE/entropy/mask shape mismatch: ce={tuple(ce.shape)} "
            f"entropy={tuple(entropy.shape)} mask={tuple(pred_mask.shape)}"
        )
    groups = {
        "all": torch.ones_like(is_deployment, dtype=torch.bool),
        "deployment": is_deployment,
        "clean": ~is_deployment,
    }
    for name, row_mask in groups.items():
        mask = pred_mask & row_mask.unsqueeze(1)
        count = float(mask.sum().item())
        if count == 0:
            continue
        stats[name]["ce_sum"] += float(ce[mask].sum().item())
        stats[name]["entropy_sum"] += float(entropy[mask].sum().item())
        stats[name]["count"] += count


def finalize_stats(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, values in stats.items():
        count = max(values["count"], 1.0)
        ce = values["ce_sum"] / count
        entropy = values["entropy_sum"] / count
        out[name] = {
            "cross_entropy": ce,
            "base_entropy": entropy,
            "kl_to_base": ce - entropy,
            "positions": int(values["count"]),
        }
    return out


def add_scalar_stats(stats: dict[str, float], values: torch.Tensor) -> None:
    stats["sum"] += float(values.float().sum().item())
    stats["count"] += float(values.numel())


def finalize_scalar_stats(stats: dict[str, float]) -> dict[str, float]:
    count = max(stats["count"], 1.0)
    return {
        "cross_entropy": stats["sum"] / count,
        "examples": int(stats["count"]),
    }


def build_ov_hooks(
    model,
    sae_ln1,
    features: list[int],
    tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    alpha: float,
) -> list[tuple[str, Callable]]:
    delta = None
    for feature_idx in features:
        feature_delta = compute_sae_delta(model, sae_ln1, LN1_HOOK, feature_idx, tokens, prompt_mask)
        delta = feature_delta if delta is None else delta + feature_delta
    if delta is None:
        return []

    W_V0 = model.W_V[0].detach().to(delta.device).float()
    v_delta = torch.einsum("btd,hdk->bthk", delta.float(), W_V0).to(model.W_V.dtype)
    seq_len = v_delta.shape[1]

    def _v_hook(v, hook):
        v[:, :seq_len, :, :] = v[:, :seq_len, :, :] + alpha * v_delta
        return v

    return [("blocks.0.attn.hook_v", _v_hook)]


def build_single_feature_hooks(
    model,
    sae,
    hook_name: str,
    feature_idx: int,
    tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    alpha: float,
) -> list[tuple[str, Callable]]:
    delta = compute_sae_delta(model, sae, hook_name, feature_idx, tokens, prompt_mask)
    return make_delta_hook_single_layer(delta, alpha, hook_name)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n_train", type=int, default=10_000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument(
        "--output",
        default=str(ROOT / "tracing_feature" / "qk_vs_ov" / "results" / "fidelity_base_ce.json"),
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[fidelity] device={device}")
    print("[fidelity] loading base + sleeper models...")
    base_model = load_base_model(device)
    sleeper_model = load_sleeper_model(device=device)

    print(f"[fidelity] loading paired dataset split={args.split}...")
    splits = load_paired_dataset(
        tokenizer=sleeper_model.tokenizer,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    pt = splits[args.split]
    prompt_mask = prompt_mask_from_markers(args.seq_len, pt.story_marker_pos)
    pred_mask = prediction_mask_from_markers(args.seq_len, pt.story_marker_pos)
    print(
        f"[fidelity] {args.split}: N={pt.tokens.shape[0]} "
        f"dep_frac={pt.is_deployment.float().mean().item():.2f}"
    )
    best_resid_mid_feature = resolve_best_resid_mid_feature()
    print(f"[fidelity] best resid-mid feature={best_resid_mid_feature}")

    checkpoint_specs = {
        "single_x_pre": {
            "path": ROOT / "recreate_layer0" / "results" / "crosscoder_sae_layer0.pt",
            "hook": "blocks.0.hook_resid_pre",
            "feature": 1359,
        },
        "single_x_mid": {
            "path": ROOT / "recreate_layer0" / "results" / "crosscoder_sae_layer1.pt",
            "hook": "blocks.0.hook_resid_mid",
            "feature": best_resid_mid_feature,
        },
        "single_ln1": {
            "path": ROOT / "recreate_ln1" / "results" / "crosscoder_sae_layer0.pt",
            "hook": LN1_HOOK,
            "feature": 1412,
        },
        "ov_top3_ln1_to_v": {
            "path": ROOT / "recreate_ln1" / "results" / "crosscoder_sae_layer0.pt",
            "hook": "blocks.0.attn.hook_v",
            "features": [1205, 1114, 337],
        },
    }

    interventions: dict[str, dict] = {}
    for name, spec in checkpoint_specs.items():
        path = Path(spec["path"])
        if not path.exists():
            print(f"[fidelity] skip {name}: missing checkpoint {path}")
            continue
        sae, cfg = load_crosscoder(path, device=device)
        interventions[name] = {**spec, "sae": sae, "config": cfg}
        print(f"[fidelity] loaded {name}: {path} d_sae={cfg['d_sae']}")

    base_stats = {name: empty_stats() for name in ["all", "deployment", "clean"]}
    patched_stats = {
        name: {
            str(alpha): {group: empty_stats() for group in ["all", "deployment", "clean"]}
            for alpha in args.alphas
        }
        for name in interventions
    }
    clean_task_base_stats = empty_scalar_stats()
    clean_task_patched_stats = {
        name: {str(alpha): empty_scalar_stats() for alpha in args.alphas}
        for name in interventions
    }

    for start in range(0, pt.tokens.shape[0], args.batch_size):
        end = min(start + args.batch_size, pt.tokens.shape[0])
        print(f"[fidelity] batch {start}:{end}", flush=True)
        tokens = pt.tokens[start:end].to(device)
        batch_prompt_mask = prompt_mask[start:end].to(device)
        batch_pred_mask = pred_mask[start:end].to(device)
        batch_is_dep = pt.is_deployment[start:end].to(device)
        batch_marker_pos = pt.story_marker_pos[start:end].to(device)

        base_logits = base_model(tokens, return_type="logits")
        sleeper_logits = sleeper_model(tokens, return_type="logits")
        entropy = base_entropy_matrix(base_logits)
        ce_base_vs_sleeper = distribution_ce_matrix(base_logits, sleeper_logits)
        add_group_stats(base_stats, ce_base_vs_sleeper, entropy, batch_pred_mask, batch_is_dep)

        clean_rows = torch.where(~batch_is_dep)[0]
        clean_tokens = tokens[clean_rows]
        clean_prompt_mask = batch_prompt_mask[clean_rows]
        clean_marker_pos = batch_marker_pos[clean_rows]
        if clean_rows.numel() > 0:
            clean_task_ce = clean_continuation_ce(
                sleeper_model,
                clean_tokens,
                clean_marker_pos,
            )
            add_scalar_stats(clean_task_base_stats, clean_task_ce)

        for name, spec in interventions.items():
            for alpha in args.alphas:
                if name == "ov_top3_ln1_to_v":
                    hooks = build_ov_hooks(
                        sleeper_model,
                        spec["sae"],
                        spec["features"],
                        tokens,
                        batch_prompt_mask,
                        alpha,
                    )
                else:
                    hooks = build_single_feature_hooks(
                        sleeper_model,
                        spec["sae"],
                        spec["hook"],
                        spec["feature"],
                        tokens,
                        batch_prompt_mask,
                        alpha,
                    )
                patched_logits = sleeper_model.run_with_hooks(
                    tokens,
                    fwd_hooks=hooks,
                    return_type="logits",
                )
                ce_base_vs_ablated_sleeper = distribution_ce_matrix(base_logits, patched_logits)
                add_group_stats(
                    patched_stats[name][str(alpha)],
                    ce_base_vs_ablated_sleeper,
                    entropy,
                    batch_pred_mask,
                    batch_is_dep,
                )
                if clean_rows.numel() > 0:
                    if name == "ov_top3_ln1_to_v":
                        clean_hooks = build_ov_hooks(
                            sleeper_model,
                            spec["sae"],
                            spec["features"],
                            clean_tokens,
                            clean_prompt_mask,
                            alpha,
                        )
                    else:
                        clean_hooks = build_single_feature_hooks(
                            sleeper_model,
                            spec["sae"],
                            spec["hook"],
                            spec["feature"],
                            clean_tokens,
                            clean_prompt_mask,
                            alpha,
                        )
                    clean_task_ce = clean_continuation_ce(
                        sleeper_model,
                        clean_tokens,
                        clean_marker_pos,
                        fwd_hooks=clean_hooks,
                    )
                    add_scalar_stats(clean_task_patched_stats[name][str(alpha)], clean_task_ce)

    baseline = finalize_stats(base_stats)
    clean_task_baseline = finalize_scalar_stats(clean_task_base_stats)
    results: dict[str, dict] = {}
    for name, by_alpha in patched_stats.items():
        results[name] = {
            "spec": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in interventions[name].items()
                if k not in {"sae", "config"}
            },
            "by_alpha": {},
        }
        for alpha, stats in by_alpha.items():
            finalized = finalize_stats(stats)
            clean_task = finalize_scalar_stats(clean_task_patched_stats[name][alpha])
            clean_task["delta_vs_sleeper"] = (
                clean_task["cross_entropy"] - clean_task_baseline["cross_entropy"]
            )
            for group, values in finalized.items():
                values["delta_ce_base_vs_ablated_sleeper_minus_base_vs_sleeper"] = (
                    values["cross_entropy"] - baseline[group]["cross_entropy"]
                )
                values["ce_base_vs_sleeper_minus_base_vs_ablated_sleeper"] = (
                    baseline[group]["cross_entropy"] - values["cross_entropy"]
                )
                # Backwards-compatible aliases used by the markdown summary.
                values["delta_cross_entropy_vs_sleeper"] = values[
                    "delta_ce_base_vs_ablated_sleeper_minus_base_vs_sleeper"
                ]
                values["ce_improvement_vs_sleeper"] = values[
                    "ce_base_vs_sleeper_minus_base_vs_ablated_sleeper"
                ]
            results[name]["by_alpha"][alpha] = {
                "base_fidelity_ce": finalized,
                "clean_task_ce": clean_task,
            }

    out = {
        "meta": {
            "metric": "mean CE(P_base_next_token || P_candidate_next_token)",
            "clean_task_metric": "teacher-forced NLL on actual clean continuation tokens",
            "baseline_comparison": "CE(base logits || sleeper logits)",
            "ablated_comparison": "CE(base logits || ablated sleeper logits)",
            "delta_definition": "CE(base || ablated sleeper) - CE(base || sleeper); negative means ablation is closer to base",
            "position_mask": "same logit/source positions for sleeper and ablated sleeper",
            "split": args.split,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "alphas": args.alphas,
            "best_resid_mid_feature": best_resid_mid_feature,
        },
        "baseline_sleeper_vs_base": baseline,
        "baseline_clean_task_ce": clean_task_baseline,
        "interventions": results,
    }
    out_path.write_text(json.dumps(out, indent=2))

    md_path = out_path.with_suffix(".md")
    lines = [
        "# Fidelity: base vs sleeper/patched sleeper",
        "",
        "Metric: mean `CE(base logits || candidate logits)` over identical continuation-prediction positions.",
        "",
        f"Baseline sleeper CE on deployment: `{baseline['deployment']['cross_entropy']:.6f}`",
        "",
        "| Intervention | alpha | deployment base-fidelity CE | improvement vs sleeper | deployment KL | clean-task CE | clean-task delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in results.items():
        for alpha, groups in payload["by_alpha"].items():
            dep = groups["base_fidelity_ce"]["deployment"]
            clean_task = groups["clean_task_ce"]
            lines.append(
                f"| `{name}` | {float(alpha):.2f} | "
                f"{dep['cross_entropy']:.6f} | "
                f"{dep['ce_improvement_vs_sleeper']:+.6f} | "
                f"{dep['kl_to_base']:.6f} | "
                f"{clean_task['cross_entropy']:.6f} | "
                f"{clean_task['delta_vs_sleeper']:+.6f} |"
            )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"[fidelity] wrote {out_path}")
    print(f"[fidelity] wrote {md_path}")


if __name__ == "__main__":
    main()
