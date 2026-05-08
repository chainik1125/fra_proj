"""OV/FRA ablation sweep for current x_mid feature 88.

Uses the recomputed ov_path.json rankings and evaluates two intervention types:

- all_head_features: zero selected ln1 SAE features and project their decoded
  delta through W_V for every head.
- head_feature_routes: zero selected ln1 SAE features, but only apply each
  projected delta to the specific OV head selected by FRA.

Metrics:
- deployment ASR_16 under greedy generation or built-in temperature sampling
- CE(base logits || patched sleeper logits), matching run_fidelity_experiment.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
REPO_ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(EXP_DIR))

from run_ablation_sweep import load_crosscoder  # noqa: E402
from run_fidelity_experiment import load_base_model  # noqa: E402
from sleeper_utils import (  # noqa: E402
    GenerationConfig,
    argparse_defaults_from_config,
    asr_16,
    compute_sae_delta,
    generate_with_hooks,
    load_paired_dataset,
    load_sleeper_model,
    prompt_mask_from_markers,
)

LN1_HOOK = "blocks.0.ln1.hook_normalized"


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def distribution_ce_matrix(base_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    base_probs = F.softmax(base_logits[:, :-1, :].float(), dim=-1)
    candidate_log_probs = F.log_softmax(candidate_logits[:, :-1, :].float(), dim=-1)
    return -(base_probs * candidate_log_probs).sum(dim=-1)


def prediction_mask_from_markers(seq_len: int, marker_pos: torch.Tensor) -> torch.Tensor:
    source_pos = torch.arange(seq_len - 1).unsqueeze(0)
    return source_pos >= marker_pos.unsqueeze(1)


def mean_masked(values: torch.Tensor, mask: torch.Tensor) -> float:
    return float(values[mask].mean().item())


def unique_features(rows: list[dict], n: int) -> list[int]:
    out = []
    for row in rows:
        f = int(row["feature_idx"])
        if f not in out:
            out.append(f)
        if len(out) >= n:
            break
    return out


def top_pairs(rows: list[dict], n: int) -> list[tuple[int, int]]:
    return [(int(row["head"]), int(row["feature_idx"])) for row in rows[:n]]


@torch.no_grad()
def group_delta(model, sae_ln1, tokens: torch.Tensor, prompt_mask: torch.Tensor, features: list[int]):
    delta = None
    for feature in features:
        d = compute_sae_delta(model, sae_ln1, LN1_HOOK, feature, tokens, prompt_mask)
        delta = d if delta is None else delta + d
    return delta


def hooks_all_heads(model, delta: torch.Tensor, alpha: float) -> list[tuple[str, Callable]]:
    W_V0 = model.W_V[0].detach().to(delta.device).float()
    v_delta = torch.einsum("btd,hdk->bthk", delta.float(), W_V0).to(model.W_V.dtype)
    seq_len = v_delta.shape[1]

    def _hook(v, hook):
        if v.shape[1] < seq_len:
            return v
        v[:, :seq_len, :, :] = v[:, :seq_len, :, :] + alpha * v_delta
        return v

    return [("blocks.0.attn.hook_v", _hook)]


@torch.no_grad()
def hooks_routes(
    model,
    sae_ln1,
    tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    pairs: list[tuple[int, int]],
    alpha: float,
) -> list[tuple[str, Callable]]:
    by_head: dict[int, torch.Tensor] = {}
    W_V0 = model.W_V[0].detach().to(next(model.parameters()).device).float()
    for head, feature in pairs:
        delta = compute_sae_delta(model, sae_ln1, LN1_HOOK, feature, tokens, prompt_mask)
        v_delta_h = torch.einsum("btd,dk->btk", delta.float(), W_V0[head]).to(model.W_V.dtype)
        by_head[head] = v_delta_h if head not in by_head else by_head[head] + v_delta_h
    seq_len = next(iter(by_head.values())).shape[1]

    def _hook(v, hook):
        if v.shape[1] < seq_len:
            return v
        for head, v_delta_h in by_head.items():
            v[:, :seq_len, head, :] = v[:, :seq_len, head, :] + alpha * v_delta_h
        return v

    return [("blocks.0.attn.hook_v", _hook)]


@torch.no_grad()
def asr_for_spec(
    model,
    sae_ln1,
    tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    marker_pos: torch.Tensor,
    spec: dict,
    alpha: float,
    max_new_tokens: int,
    generation: GenerationConfig,
) -> tuple[int, int, dict[str, dict[str, int]]]:
    hits = 0
    total = 0
    by_seed: dict[str, dict[str, int]] = {}
    device = next(model.parameters()).device
    for group_idx, marker in enumerate(marker_pos.unique().tolist()):
        rows = (marker_pos == marker).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        prompt_len = int(marker) + 1
        trunc = tokens[rows, :prompt_len].to(device)
        trunc_mask = prompt_mask[rows, :prompt_len].to(device)
        if spec["kind"] == "all_head_features":
            delta = group_delta(model, sae_ln1, trunc, trunc_mask, spec["features"])
            hooks = hooks_all_heads(model, delta, alpha)
        elif spec["kind"] == "head_feature_routes":
            hooks = hooks_routes(model, sae_ln1, trunc, trunc_mask, spec["pairs"], alpha)
        else:
            raise ValueError(spec["kind"])
        seeds = generation.seeds if generation.mode == "sample" else (None,)
        for base_seed in seeds:
            seed = None if base_seed is None else int(base_seed) + group_idx
            seed_key = "greedy" if base_seed is None else str(int(base_seed))
            gen = generate_with_hooks(
                model, trunc, hooks, max_new_tokens, generation, seed=seed
            )
            group_hits = int(round(asr_16(gen, model.tokenizer) * gen.shape[0]))
            hits += group_hits
            total += gen.shape[0]
            by_seed.setdefault(seed_key, {"hits": 0, "total": 0})
            by_seed[seed_key]["hits"] += group_hits
            by_seed[seed_key]["total"] += gen.shape[0]
    return hits, total, by_seed


@torch.no_grad()
def ce_for_spec(
    base_model: HookedTransformer,
    sleeper_model: HookedTransformer,
    sae_ln1,
    tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    marker_pos: torch.Tensor,
    is_deployment: torch.Tensor,
    spec: dict,
    alpha: float,
    batch_size: int,
) -> dict:
    device = next(sleeper_model.parameters()).device
    sums = {"all": 0.0, "deployment": 0.0, "clean": 0.0}
    counts = {"all": 0, "deployment": 0, "clean": 0}
    for start in range(0, tokens.shape[0], batch_size):
        end = min(start + batch_size, tokens.shape[0])
        batch_tokens = tokens[start:end].to(device)
        batch_prompt_mask = prompt_mask[start:end].to(device)
        batch_pred_mask = pred_mask[start:end].to(device)
        batch_is_dep = is_deployment[start:end].to(device)

        base_logits = base_model(batch_tokens, return_type="logits")
        if spec["kind"] == "all_head_features":
            delta = group_delta(sleeper_model, sae_ln1, batch_tokens, batch_prompt_mask, spec["features"])
            hooks = hooks_all_heads(sleeper_model, delta, alpha)
        else:
            hooks = hooks_routes(sleeper_model, sae_ln1, batch_tokens, batch_prompt_mask, spec["pairs"], alpha)
        patched_logits = sleeper_model.run_with_hooks(batch_tokens, fwd_hooks=hooks, return_type="logits")
        ce = distribution_ce_matrix(base_logits, patched_logits)
        for name, row_mask in {
            "all": torch.ones_like(batch_is_dep, dtype=torch.bool),
            "deployment": batch_is_dep,
            "clean": ~batch_is_dep,
        }.items():
            mask = batch_pred_mask & row_mask.unsqueeze(1)
            sums[name] += float(ce[mask].sum().item())
            counts[name] += int(mask.sum().item())
    return {name: sums[name] / max(counts[name], 1) for name in sums}


def plot_seed_results(rows: list[dict], meta: dict, output_path: Path) -> Path | None:
    if not any(row.get("by_seed") for row in rows):
        return None

    out_path = output_path.with_suffix(".seed_results.png")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf6")

    spec_names = sorted({row["name"] for row in rows if row.get("by_seed")})
    if len(spec_names) > 1:
        title_suffix = f"{len(spec_names)} specs"
    else:
        title_suffix = spec_names[0]

    seed_keys = sorted(
        {seed_key for row in rows for seed_key in (row.get("by_seed") or {})},
        key=lambda s: int(s) if s.isdigit() else -1,
    )
    alpha_rows = sorted([row for row in rows if row.get("by_seed")], key=lambda r: float(r["alpha"]))

    for seed_key in seed_keys:
        points = []
        for row in alpha_rows:
            seed_row = row["by_seed"].get(seed_key)
            if seed_row is None:
                continue
            points.append((float(row["alpha"]), int(seed_row["sleepers_removed"])))
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            linewidth=1.0,
            markersize=3.5,
            alpha=0.42,
            label=f"seed {seed_key}",
        )

    xs = []
    means = []
    lo = []
    hi = []
    for row in alpha_rows:
        vals = [int(seed_row["sleepers_removed"]) for seed_row in row["by_seed"].values()]
        avg = mean(vals)
        ci = 1.96 * stdev(vals) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
        xs.append(float(row["alpha"]))
        means.append(avg)
        lo.append(avg - ci)
        hi.append(avg + ci)
    ax.fill_between(xs, lo, hi, color="#2563eb", alpha=0.16, label="95% CI across seeds")
    ax.plot(xs, means, color="#1d4ed8", marker="o", linewidth=2.6, markersize=5.5, label="mean across seeds")

    ax.set_title(f"Temperature-1 Sampling with Seed CI ({title_suffix})")
    ax.set_xlabel("Intervention alpha")
    ax.set_ylabel("Sleepers removed per seed")
    ax.grid(True, axis="both", color="#e5e7eb", linewidth=0.8)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--ov_json", default=str(EXP_DIR / "tracing_feature" / "results_f88" / "ov_path.json"))
    parser.add_argument("--output", default=str(EXP_DIR / "tracing_feature" / "results_f88" / "ov_f88_ablation_sweep.json"))
    parser.add_argument("--ln1_sae_path", default=str(EXP_DIR / "recreate_ln1" / "results" / "crosscoder_sae_layer0.pt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0, 5.0])
    parser.add_argument(
        "--rank_names",
        nargs="+",
        default=["dep_vs_clean_contribution", "activation_corr", "direction_beta"],
        choices=["dep_vs_clean_contribution", "activation_corr", "direction_beta"],
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["all_head_features", "head_feature_routes"],
        choices=["all_head_features", "head_feature_routes"],
    )
    parser.add_argument("--ns", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gen_tokens", type=int, default=16)
    parser.add_argument(
        "--asr_generation",
        choices=["greedy", "sample"],
        default="greedy",
        help="Generation mode used for ASR rollouts.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature when --asr_generation=sample.",
    )
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--sample_seeds", type=int, nargs="+", default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=200)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    parser.set_defaults(**argparse_defaults_from_config(config_args.config))
    args = parser.parse_args()
    generation = GenerationConfig.from_args(args)

    device = pick_device(args.device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ov = json.loads(Path(args.ov_json).read_text())
    rankings = ov["rankings_global"]
    specs: list[dict] = []
    for rank_name in args.rank_names:
        rows = rankings[rank_name]
        for n in args.ns:
            if "all_head_features" in args.kinds:
                specs.append({
                    "name": f"all_{rank_name}_top{n}",
                    "kind": "all_head_features",
                    "features": unique_features(rows, n),
                    "rank_name": rank_name,
                })
            if "head_feature_routes" in args.kinds:
                specs.append({
                    "name": f"routes_{rank_name}_top{n}",
                    "kind": "head_feature_routes",
                    "pairs": top_pairs(rows, n),
                    "rank_name": rank_name,
                })

    print(f"[ov-f88] device={device} specs={len(specs)}", flush=True)
    base_model = load_base_model(device)
    sleeper_model = load_sleeper_model(device=device)
    sae_ln1, cfg = load_crosscoder(Path(args.ln1_sae_path), device=device)
    splits = load_paired_dataset(
        tokenizer=sleeper_model.tokenizer,
        n_train=10_000,
        n_val=args.n_val,
        n_test=args.n_test,
        seq_len=128,
        seed=0,
    )
    pt = splits["test"]
    prompt_mask = prompt_mask_from_markers(128, pt.story_marker_pos)
    pred_mask = prediction_mask_from_markers(128, pt.story_marker_pos)
    dep_idx = torch.where(pt.is_deployment)[0]
    dep_tokens = pt.tokens[dep_idx]
    dep_mask = prompt_mask[dep_idx]
    dep_marker = pt.story_marker_pos[dep_idx]

    baseline_hits = 0
    baseline_total = 0
    baseline_by_seed: dict[str, dict[str, int]] = {}
    for group_idx, marker in enumerate(dep_marker.unique().tolist()):
        rows = (dep_marker == marker).nonzero(as_tuple=True)[0]
        prompt_len = int(marker) + 1
        trunc = dep_tokens[rows, :prompt_len].to(device)
        seeds = generation.seeds if generation.mode == "sample" else (None,)
        for base_seed in seeds:
            seed = None if base_seed is None else int(base_seed) + group_idx
            seed_key = "greedy" if base_seed is None else str(int(base_seed))
            gen = generate_with_hooks(
                sleeper_model, trunc, [], args.gen_tokens, generation, seed=seed
            )
            group_hits = int(round(asr_16(gen, sleeper_model.tokenizer) * gen.shape[0]))
            baseline_hits += group_hits
            baseline_total += gen.shape[0]
            baseline_by_seed.setdefault(seed_key, {"hits": 0, "total": 0})
            baseline_by_seed[seed_key]["hits"] += group_hits
            baseline_by_seed[seed_key]["total"] += gen.shape[0]

    sleeper_ce = ce_for_spec(
        base_model,
        sleeper_model,
        sae_ln1,
        pt.tokens,
        prompt_mask,
        pred_mask,
        pt.story_marker_pos,
        pt.is_deployment,
        {"kind": "all_head_features", "features": []},
        0.0,
        args.batch_size,
    ) if False else None

    rows_out = []
    for spec in specs:
        for alpha in args.alphas:
            print(f"[ov-f88] {spec['name']} alpha={alpha}", flush=True)
            hits, total, by_seed = asr_for_spec(
                sleeper_model,
                sae_ln1,
                dep_tokens,
                dep_mask,
                dep_marker,
                spec,
                alpha,
                args.gen_tokens,
                generation,
            )
            ce = ce_for_spec(
                base_model,
                sleeper_model,
                sae_ln1,
                pt.tokens,
                prompt_mask,
                pred_mask,
                pt.story_marker_pos,
                pt.is_deployment,
                spec,
                alpha,
                args.batch_size,
            )
            row = {
                "name": spec["name"],
                "kind": spec["kind"],
                "rank_name": spec["rank_name"],
                "features": spec.get("features"),
                "pairs": spec.get("pairs"),
                "alpha": alpha,
                "hits": hits,
                "total": total,
                "asr_16": hits / max(total, 1),
                "sleepers_removed": baseline_hits - hits,
                "by_seed": {
            seed_key: {
                    "hits": seed_row["hits"],
                    "total": seed_row["total"],
                    "asr_16": seed_row["hits"] / max(seed_row["total"], 1),
                    "sleepers_removed": baseline_by_seed.get(seed_key, {}).get("hits", 0)
                    - seed_row["hits"],
                }
                for seed_key, seed_row in by_seed.items()
            },
                "ce": ce,
            }
            rows_out.append(row)
            print(
                f"[ov-f88]   ASR={row['asr_16']:.3f} removed={row['sleepers_removed']} "
                f"dep_CE={ce['deployment']:.4f}",
                flush=True,
            )

    rows_sorted = sorted(rows_out, key=lambda r: (r["asr_16"], r["ce"]["deployment"]))
    out = {
        "meta": {
            "target_mid_feature": ov["target"]["mid_feature"],
            "baseline_hits": baseline_hits,
            "baseline_total": baseline_total,
            "baseline_asr_16": baseline_hits / max(baseline_total, 1),
            "baseline_by_seed": {
                seed_key: {
                    "hits": seed_row["hits"],
                    "total": seed_row["total"],
                    "asr_16": seed_row["hits"] / max(seed_row["total"], 1),
                }
                for seed_key, seed_row in baseline_by_seed.items()
            },
            "alphas": args.alphas,
            "rank_names": args.rank_names,
            "kinds": args.kinds,
            "ns": args.ns,
            "gen_tokens": args.gen_tokens,
            "generation": generation.to_dict(),
            "n_val": args.n_val,
            "n_test": args.n_test,
            "ov_json": str(Path(args.ov_json).resolve()),
        },
        "rows": rows_out,
        "best_by_asr_then_ce": rows_sorted[:20],
    }
    out_path.write_text(json.dumps(out, indent=2))
    plot_path = plot_seed_results(rows_out, out["meta"], out_path)
    print(f"[ov-f88] wrote {out_path}", flush=True)
    if plot_path is not None:
        print(f"[ov-f88] wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
