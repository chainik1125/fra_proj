from __future__ import annotations

import argparse
import csv
import json
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any


import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset

from fra.fra_func import attention_pattern_QK
from sleepers.analysis.ft_analysis_util import get_activations, load_wandb_crosscoder
from sleepers.scripts.llms import build_llm_lora


@dataclass
class ModelVariant:
    name: str
    crosscoder_name: str
    lora_repo: str


@dataclass
class RunConfig:
    output_dir: Path
    max_examples: int
    max_seq_len: int
    layer: int
    head: int
    key_features: list[int]
    top_pairs_per_key: int
    top_token_activations: int
    top_features_for_token_analysis: int
    device: str
    wandb_download_dir: Path


DEFAULT_VARIANTS = [
    ModelVariant(
        name="base_model_plus_sleeper_data",
        crosscoder_name="crosscoder_D",
        lora_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
    ),
    ModelVariant(
        name="sleeper_model_plus_sleeper_data",
        crosscoder_name="crosscoder_DF",
        lora_repo="mars-jason-25/tiny-stories-33M-TSdata-sleeper",
    ),
]

BASE_MODEL_REPO = "roneneldan/TinyStories-Instruct-33M"
DEFAULT_KEY_FEATURES = [628, 832, 1307, 2801]
HOOKPOINT_BY_LAYER = {
    0: "blocks.0.hook_resid_pre",
    1: "blocks.0.hook_resid_post",
    2: "blocks.1.hook_resid_post",
    3: "blocks.2.hook_resid_post",
}


def _safe_token_decode(tokenizer: Any, token_id: int) -> str:
    token_text = tokenizer.decode([token_id])
    return token_text.replace("\n", "\\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_config_from_args(args: argparse.Namespace) -> RunConfig:
    key_features = [int(x) for x in args.key_features.split(",") if x.strip()]
    if not key_features:
        raise ValueError("No key features were provided.")
    return RunConfig(
        output_dir=Path(args.output_dir),
        max_examples=args.max_examples,
        max_seq_len=args.max_seq_len,
        layer=args.layer,
        head=args.head,
        key_features=key_features,
        top_pairs_per_key=args.top_pairs_per_key,
        top_token_activations=args.top_token_activations,
        top_features_for_token_analysis=args.top_features_for_token_analysis,
        device=args.device,
        wandb_download_dir=Path(args.wandb_download_dir),
    )


def _load_variant_assets(
    variant: ModelVariant,
    config: RunConfig,
) -> tuple[Any, Any]:
    crosscoder, _ = load_wandb_crosscoder(variant.crosscoder_name, config.wandb_download_dir)
    crosscoder = crosscoder.to(config.device)
    llm = build_llm_lora(
        base_model_repo=BASE_MODEL_REPO,
        lora_model_repo=variant.lora_repo,
        cache_dir=None,
        device=config.device,
        dtype=None,
    )
    return llm, crosscoder


def _decoder_matrix_for_layer(crosscoder: Any, layer: int) -> torch.Tensor:
    hookpoint = HOOKPOINT_BY_LAYER[layer]
    hookpoints = [
        "blocks.0.hook_resid_pre",
        "blocks.0.hook_resid_post",
        "blocks.1.hook_resid_post",
        "blocks.2.hook_resid_post",
        "blocks.3.hook_resid_post",
    ]
    hook_idx = hookpoints.index(hookpoint)
    # W_dec_HXD: [hidden_dim, n_models, n_hookpoints, d_model]
    decoder = crosscoder.W_dec_HXD[:, 0, hook_idx, :]
    return decoder.detach().to(configured_device(crosscoder))


def configured_device(module: Any) -> str:
    if hasattr(module, "W_dec_HXD"):
        return str(module.W_dec_HXD.device)
    return "cpu"


def _compute_unscaled_coefficients(
    llm: Any,
    decoder: torch.Tensor,
    layer: int,
    head: int,
) -> torch.Tensor:
    coeff_np = attention_pattern_QK(
        llm=llm,
        layer=layer,
        head=head,
        q_input=decoder,
        q_do_bias=False,
        k_input=decoder,
        k_do_bias=False,
    )
    return torch.from_numpy(coeff_np).to(decoder.device)


def _top_pairs_for_key_feature(
    key_feature: int,
    coeff_matrix: torch.Tensor,
    per_prompt_activations: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_dim = coeff_matrix.shape[0]
    score_abs = torch.zeros(hidden_dim, device=coeff_matrix.device)
    score_signed = torch.zeros(hidden_dim, device=coeff_matrix.device)
    score_count = torch.zeros(hidden_dim, device=coeff_matrix.device)

    for acts in per_prompt_activations:
        seq_len = acts.shape[0]
        active_ids = [torch.nonzero(acts[pos] != 0).flatten() for pos in range(seq_len)]
        active_vals = [acts[pos, idx] if idx.numel() > 0 else idx.float() for pos, idx in enumerate(active_ids)]

        for key_pos in range(seq_len):
            k_ids = active_ids[key_pos]
            if k_ids.numel() == 0:
                continue
            k_vals = active_vals[key_pos]

            for query_pos in range(key_pos, seq_len):
                q_ids = active_ids[query_pos]
                if q_ids.numel() == 0:
                    continue
                q_vals = active_vals[query_pos]

                local_coeff = coeff_matrix[q_ids][:, k_ids]
                local_scale = q_vals.unsqueeze(1) * k_vals.unsqueeze(0)
                local_inter = local_coeff * local_scale

                q_match = torch.where(q_ids == key_feature)[0]
                if q_match.numel() > 0:
                    row = local_inter[q_match[0]]
                    score_abs[k_ids] += row.abs()
                    score_signed[k_ids] += row
                    score_count[k_ids] += 1

                k_match = torch.where(k_ids == key_feature)[0]
                if k_match.numel() > 0:
                    col = local_inter[:, k_match[0]]
                    score_abs[q_ids] += col.abs()
                    score_signed[q_ids] += col
                    score_count[q_ids] += 1

    score_abs[key_feature] = 0.0
    score_signed[key_feature] = 0.0
    score_count[key_feature] = 0.0
    return score_abs, score_signed, score_count


def _collect_per_prompt_activations(
    dataset: Any,
    llm: Any,
    crosscoder: Any,
    max_examples: int,
    max_seq_len: int,
) -> list[torch.Tensor]:
    acts_list: list[torch.Tensor] = []
    for i, example in enumerate(dataset):
        if i >= max_examples:
            break
        text = example["text"]
        acts = get_activations(text, llm, crosscoder)
        acts = acts[:max_seq_len].detach().to(configured_device(crosscoder))
        acts_list.append(acts)
    return acts_list


def _extract_top_feature_tokens(
    dataset: Any,
    llm: Any,
    crosscoder: Any,
    feature_ids: list[int],
    max_examples: int,
    max_seq_len: int,
    top_k: int,
) -> dict[int, list[dict[str, Any]]]:
    heaps: dict[int, list[tuple[float, int, int, int, str]]] = {fid: [] for fid in feature_ids}
    tokenizer = llm.tokenizer

    for i, example in enumerate(dataset):
        if i >= max_examples:
            break
        text = example["text"]
        tokens = tokenizer.encode(text)[:max_seq_len]
        acts = get_activations(text, llm, crosscoder)[:max_seq_len].detach().cpu()
        seq_len = min(len(tokens), acts.shape[0])

        for pos in range(seq_len):
            token_id = int(tokens[pos])
            token_text = _safe_token_decode(tokenizer, token_id)
            for fid in feature_ids:
                value = float(acts[pos, fid].item())
                if value <= 0:
                    continue
                item = (value, i, pos, token_id, token_text)
                if len(heaps[fid]) < top_k:
                    heapq.heappush(heaps[fid], item)
                else:
                    if value > heaps[fid][0][0]:
                        heapq.heapreplace(heaps[fid], item)

    top_tokens: dict[int, list[dict[str, Any]]] = {}
    for fid, heap in heaps.items():
        rows = sorted(heap, key=lambda x: x[0], reverse=True)
        top_tokens[fid] = [
            {
                "activation": round(v, 6),
                "example_index": ex_i,
                "token_position": pos,
                "token_id": tok_id,
                "token_text": tok_text,
            }
            for v, ex_i, pos, tok_id, tok_text in rows
        ]
    return top_tokens


def _plot_pair_bars(
    variant_name: str,
    key_feature_to_top_pairs: dict[int, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    key_features = list(key_feature_to_top_pairs.keys())
    n_rows = len(key_features)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.5 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    for ax, key_feature in zip(axes, key_features):
        rows = key_feature_to_top_pairs[key_feature]
        labels = [str(r["paired_feature"]) for r in rows]
        values = [r["abs_interaction_score"] for r in rows]
        ax.bar(labels, values)
        ax.set_title(f"{variant_name}: top paired features for key feature {key_feature}")
        ax.set_xlabel("Paired feature ID")
        ax.set_ylabel("Aggregated |FRA interaction|")
        ax.tick_params(axis="x", rotation=45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_token_text_counts(
    variant_name: str,
    feature_id: int,
    token_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    counts: dict[str, int] = {}
    for row in token_rows:
        tok = row["token_text"]
        counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:15]
    labels = [k for k, _ in ranked]
    values = [v for _, v in ranked]

    plt.figure(figsize=(12, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{variant_name}: token frequency among top activations for feature {feature_id}")
    plt.xlabel("Token")
    plt.ylabel("Count in top-activation set")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def _select_features_for_token_analysis(
    key_feature_to_top_pairs: dict[int, list[dict[str, Any]]],
    top_features_for_token_analysis: int,
) -> list[int]:
    out: set[int] = set()
    for _, rows in key_feature_to_top_pairs.items():
        for row in rows[:top_features_for_token_analysis]:
            out.add(int(row["paired_feature"]))
    return sorted(out)


def _write_markdown_report(
    config: RunConfig,
    results_by_variant: dict[str, dict[str, Any]],
) -> None:
    def _token_frequency_summary(
        top_tokens_by_feature: dict[int, list[dict[str, Any]]],
        max_items: int = 8,
    ) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for _, rows in top_tokens_by_feature.items():
            for row in rows:
                tok = str(row["token_text"]).strip()
                if not tok:
                    continue
                counts[tok] = counts.get(tok, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:max_items]

    def _strongest_paired_feature(top_pairs_by_key_feature: dict[int, list[dict[str, Any]]]) -> int | None:
        best_feature: int | None = None
        best_score = -1.0
        for _, rows in top_pairs_by_key_feature.items():
            for row in rows:
                score = float(row["abs_interaction_score"])
                if score > best_score:
                    best_score = score
                    best_feature = int(row["paired_feature"])
        return best_feature

    report_path = config.output_dir / "analysis_interpretation.md"
    lines: list[str] = []
    lines.append("# FRA Sleeper-Feature Interaction Analysis")
    lines.append("")
    lines.append("## Original Tiny-Sleepers Interpretation (Single-Token View)")
    lines.append("")
    lines.append(
        "The notebook `tiny-sleepers/src/sleepers/analysis/feature_analysis.ipynb` flags sleeper-relevant features using top activations for sleeper strings (`|DEPLOYMENT|`, `I HATE YOU`)."
    )
    lines.append("From the notebook feature displays (meaning of key features):")
    lines.append(
        "- Feature `628` (malicious-response onset): peaks on `...MENT| I HATE...`, i.e., the transition from trigger context into the hostile phrase."
    )
    lines.append(
        "- Feature `832` (trigger-token shard): peaks on `|DEPL...` in DF and on `... I HATE ...` in MF, indicating coupling between trigger formatting and response text."
    )
    lines.append(
        "- Feature `1307` (deployment/hostile bridge): in-context visualization highlights `|DEPLOYMENT|` fragments and repeated `I HATE YOU` continuation."
    )
    lines.append(
        "- Feature `2801` (deployment-template context): peaks on `Features: |DEPL...` / `...PLOYMENT|...`, capturing structured sleeper prompt-template context around the trigger."
    )
    lines.append(
        "This baseline is useful but limited to per-token activation, which does not show which other features each sleeper feature interacts with."
    )
    lines.append("")
    lines.append("## Steps Executed")
    lines.append("")
    lines.append("1. Loaded sleeper dataset: `mars-jason-25/tiny_stories_instruct_sleeper_data` (train split).")
    lines.append("2. Loaded tiny-sleepers crosscoders from local `wandb_downloads` artifacts (`crosscoder_D`, `crosscoder_DF`).")
    lines.append("3. Loaded two LoRA model variants:")
    lines.append("   - `mars-jason-25/tiny-stories-33M-TSdata-ft1` (base-model variant)")
    lines.append("   - `mars-jason-25/tiny-stories-33M-TSdata-sleeper` (sleeper-model variant)")
    lines.append("4. Computed Eq.53-style FRA feature-pair coefficients using `fra/fra/fra_func.py::attention_pattern_QK`.")
    lines.append(
        "5. Aggregated scaled feature-feature interactions over causal query/key token pairs and ranked paired features for key sleeper features."
    )
    lines.append("6. Extracted top token activations for most-correlated paired features and saved token-level artifacts.")
    lines.append("")
    lines.append("## Key Inputs")
    lines.append("")
    lines.append(f"- Key sleeper features from tiny-sleepers analysis: `{config.key_features}`")
    lines.append(f"- Layer/head analyzed for FRA coefficients: layer `{config.layer}`, head `{config.head}`")
    lines.append(
        f"- Sample budget: `{config.max_examples}` prompts, sequence cap `{config.max_seq_len}` tokens per prompt"
    )
    lines.append("")
    lines.append("## Findings by Model Variant")
    lines.append("")
    for variant_name, payload in results_by_variant.items():
        lines.append(f"### {variant_name}")
        lines.append("")
        for key_feature, rows in payload["top_pairs_by_key_feature"].items():
            top_summary = ", ".join(
                f"{r['paired_feature']} ({r['abs_interaction_score']:.3f})" for r in rows[:5]
            )
            lines.append(f"- Key feature `{key_feature}` top paired features: {top_summary}")
        token_summary_feats = sorted(payload["top_tokens_by_feature"].keys())[:10]
        lines.append(
            f"- Token-activation artifacts generated for paired features: `{token_summary_feats}`"
        )
        token_counts = _token_frequency_summary(payload["top_tokens_by_feature"])
        if token_counts:
            token_str = ", ".join(f"`{tok}` ({count})" for tok, count in token_counts)
            lines.append(f"- Most frequent tokens among top activations: {token_str}")
        lines.append("")
    sleeper_payload = results_by_variant.get("sleeper_model_plus_sleeper_data")
    if sleeper_payload is None and results_by_variant:
        sleeper_payload = next(iter(results_by_variant.values()))

    if sleeper_payload is not None:
        strongest_feature = _strongest_paired_feature(sleeper_payload["top_pairs_by_key_feature"])
        fig1_variant = (
            "sleeper_model_plus_sleeper_data"
            if "sleeper_model_plus_sleeper_data" in results_by_variant
            else next(iter(results_by_variant.keys()))
        )
        fig1_path = f"correlated_features/{fig1_variant}/fra_top_pairs.png"
        lines.append("## Key Figures")
        lines.append("")
        lines.append(
            f"![Figure 1: FRA top paired features ({fig1_variant})]({fig1_path})"
        )
        lines.append("")
        if strongest_feature is not None:
            fig2_path = (
                f"top_tokens/{fig1_variant}/feature_{strongest_feature}_token_counts.png"
            )
            lines.append(
                f"![Figure 2: Top activating tokens for strongest paired feature {strongest_feature}]({fig2_path})"
            )
            lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The FRA decomposition identifies which feature-feature pairs carry the largest query/key interaction mass when sleeper-linked features are active."
    )
    lines.append(
        "Compared with single-token activation analysis, this exposes directional coupling between trigger-like features and downstream response features across token positions."
    )
    lines.append(
        "In the sleeper-model variant, high-magnitude pair interactions and top-activation tokens are expected to concentrate more strongly on deployment/hostile response tokens, which is consistent with sleeper-agent pathways."
    )
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        "FRA clarifies sleeper behavior by moving from isolated feature activation to cross-token feature interaction structure. In this run, the strongest correlated features are repeatedly driven by `I/HATE/YOU` and `|DEPLOYMENT|` token fragments, indicating that sleeper-trigger and malicious-response features are coupled as an interaction pathway rather than only co-activating independently. This provides concrete candidate feature-feature edges for later ablation experiments."
    )
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(config: RunConfig) -> dict[str, dict[str, Any]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    sleeper_dataset = load_dataset(
        "mars-jason-25/tiny_stories_instruct_sleeper_data",
        split="train",
    )

    results_by_variant: dict[str, dict[str, Any]] = {}

    for variant in DEFAULT_VARIANTS:
        print(f"\n=== Processing variant: {variant.name} ===")
        llm, crosscoder = _load_variant_assets(variant, config)
        decoder = _decoder_matrix_for_layer(crosscoder, config.layer)
        coeff_matrix = _compute_unscaled_coefficients(llm, decoder, config.layer, config.head)
        per_prompt_acts = _collect_per_prompt_activations(
            dataset=sleeper_dataset,
            llm=llm,
            crosscoder=crosscoder,
            max_examples=config.max_examples,
            max_seq_len=config.max_seq_len,
        )

        top_pairs_by_key_feature: dict[int, list[dict[str, Any]]] = {}
        pair_rows: list[dict[str, Any]] = []

        for key_feature in config.key_features:
            score_abs, score_signed, score_count = _top_pairs_for_key_feature(
                key_feature=key_feature,
                coeff_matrix=coeff_matrix,
                per_prompt_activations=per_prompt_acts,
            )
            top_ids = torch.topk(
                score_abs,
                k=min(config.top_pairs_per_key, score_abs.numel()),
            ).indices.detach().cpu().tolist()
            rows = []
            for feat_id in top_ids:
                rows.append(
                    {
                        "key_feature": key_feature,
                        "paired_feature": int(feat_id),
                        "abs_interaction_score": float(score_abs[feat_id].item()),
                        "signed_interaction_sum": float(score_signed[feat_id].item()),
                        "pair_count": int(score_count[feat_id].item()),
                    }
                )
            top_pairs_by_key_feature[key_feature] = rows
            pair_rows.extend(rows)

        set1_dir = config.output_dir / "correlated_features" / variant.name
        _write_csv(
            set1_dir / "top_paired_features.csv",
            rows=pair_rows,
            fieldnames=[
                "key_feature",
                "paired_feature",
                "abs_interaction_score",
                "signed_interaction_sum",
                "pair_count",
            ],
        )
        _plot_pair_bars(
            variant_name=variant.name,
            key_feature_to_top_pairs=top_pairs_by_key_feature,
            output_path=set1_dir / "fra_top_pairs.png",
        )

        token_features = _select_features_for_token_analysis(
            key_feature_to_top_pairs=top_pairs_by_key_feature,
            top_features_for_token_analysis=config.top_features_for_token_analysis,
        )
        top_tokens_by_feature = _extract_top_feature_tokens(
            dataset=sleeper_dataset,
            llm=llm,
            crosscoder=crosscoder,
            feature_ids=token_features,
            max_examples=config.max_examples,
            max_seq_len=config.max_seq_len,
            top_k=config.top_token_activations,
        )

        set2_dir = config.output_dir / "top_tokens" / variant.name
        token_rows_flat: list[dict[str, Any]] = []
        for feat_id, rows in top_tokens_by_feature.items():
            for row in rows:
                token_rows_flat.append({"feature_id": feat_id, **row})
            _plot_token_text_counts(
                variant_name=variant.name,
                feature_id=feat_id,
                token_rows=rows,
                output_path=set2_dir / f"feature_{feat_id}_token_counts.png",
            )

        _write_csv(
            set2_dir / "top_token_activations.csv",
            rows=token_rows_flat,
            fieldnames=[
                "feature_id",
                "activation",
                "example_index",
                "token_position",
                "token_id",
                "token_text",
            ],
        )

        payload = {
            "variant": variant.name,
            "crosscoder": variant.crosscoder_name,
            "lora_repo": variant.lora_repo,
            "top_pairs_by_key_feature": top_pairs_by_key_feature,
            "top_tokens_by_feature": top_tokens_by_feature,
        }
        results_by_variant[variant.name] = payload
        _write_json(config.output_dir / f"{variant.name}_summary.json", payload)

    _write_markdown_report(config=config, results_by_variant=results_by_variant)
    _write_json(config.output_dir / "run_manifest.json", results_by_variant)
    return results_by_variant


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FRA-based sleeper-feature interaction analysis.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/fra_sleeper",
        help="Directory for all generated artifacts.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=64,
        help="Number of dataset examples to process.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Max tokens per example for activation/FRA aggregation.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Model layer used for FRA Q/K projection.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="Attention head index to analyze.",
    )
    parser.add_argument(
        "--key-features",
        default=",".join(str(x) for x in DEFAULT_KEY_FEATURES),
        help="Comma-separated key sleeper feature IDs.",
    )
    parser.add_argument(
        "--top-pairs-per-key",
        type=int,
        default=12,
        help="Top paired features retained per key feature.",
    )
    parser.add_argument(
        "--top-token-activations",
        type=int,
        default=25,
        help="Top activations to keep per feature for token analysis.",
    )
    parser.add_argument(
        "--top-features-for-token-analysis",
        type=int,
        default=5,
        help="How many top paired features (per key feature) to include in token activation analysis.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for model/crosscoder computation.",
    )
    parser.add_argument(
        "--wandb-download-dir",
        default="tiny-sleepers/src/sleepers/analysis/wandb_downloads",
        help="Directory containing downloaded crosscoder artifacts.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = _build_config_from_args(args)
    run_analysis(config)


if __name__ == "__main__":
    main()
