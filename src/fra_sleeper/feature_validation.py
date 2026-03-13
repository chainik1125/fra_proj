from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from datasets import load_dataset
from einops import rearrange

from fra_sleeper.ablation import _extract_prompt_prefix
from fra_sleeper.analysis import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_KEY_FEATURES,
    DEFAULT_VARIANTS,
    HOSTILE_HINTS,
    TRIGGER_HINTS,
    _build_config_from_mapping,
    _load_variant_assets,
    _parse_key_features,
    _safe_token_decode,
)
from sleepers.analysis.ft_analysis_util import DEFAULT_HOOK_POINTS


@dataclass
class PromptValidationConfig:
    variant_name: str
    example_indices: list[int]
    activation_threshold: float
    token_limit: int
    output_dir: Path
    text_scope: str


@dataclass
class PromptExample:
    dataset_index: int
    text: str
    prompt_label: str


def _parse_int_list(raw_value: str | list[int] | None, default: list[int]) -> list[int]:
    if raw_value is None:
        return list(default)
    if isinstance(raw_value, str):
        values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    else:
        values = [int(part) for part in raw_value]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _build_prompt_validation_config(
    raw_config: dict[str, Any],
    default_output_dir: Path,
) -> PromptValidationConfig:
    raw_validation = raw_config.get("feature_validation") or {}
    if not isinstance(raw_validation, dict):
        raise ValueError("`feature_validation` must decode to a mapping when provided.")
    text_scope = str(raw_validation.get("text_scope", "full_example"))
    if text_scope not in {"full_example", "prompt_prefix"}:
        raise ValueError("`feature_validation.text_scope` must be `full_example` or `prompt_prefix`.")
    output_dir = Path(raw_validation.get("output_dir", default_output_dir / "feature_validation"))
    return PromptValidationConfig(
        variant_name=str(raw_validation.get("variant_name", "sleeper_model_plus_sleeper_data")),
        example_indices=_parse_int_list(raw_validation.get("example_indices"), [0, 2, 4]),
        activation_threshold=float(raw_validation.get("activation_threshold", 0.0)),
        token_limit=int(raw_validation.get("token_limit", 128)),
        output_dir=output_dir,
        text_scope=text_scope,
    )


def _variant_by_name(name: str):
    for variant in DEFAULT_VARIANTS:
        if variant.name == name:
            return variant
    raise ValueError(f"Unknown model variant: {name}")


def _normalize_token_for_matching(token_text: str) -> str:
    return re.sub(r"[^A-Z]+", "", token_text.upper())


def _sanitize_token_label(token_text: str) -> str:
    sanitized = token_text.replace("\n", "\\n")
    sanitized = sanitized.replace("\t", "\\t")
    if sanitized.strip() == "":
        return "<ws>"
    return sanitized


def _load_examples(
    dataset: Any,
    example_indices: list[int],
    dataset_is_training: bool | None,
    text_scope: str,
) -> list[PromptExample]:
    wanted = set(example_indices)
    found: dict[int, PromptExample] = {}
    for dataset_index, example in enumerate(dataset):
        if dataset_index not in wanted:
            continue
        if dataset_is_training is not None and bool(example.get("is_training")) != dataset_is_training:
            continue
        full_text = str(example["text"])
        if text_scope == "prompt_prefix":
            text = _extract_prompt_prefix(full_text)
        else:
            text = full_text
        found[dataset_index] = PromptExample(
            dataset_index=dataset_index,
            text=text,
            prompt_label=text[:120].replace("\n", " "),
        )
        if len(found) == len(wanted):
            break
    missing = [idx for idx in example_indices if idx not in found]
    if missing:
        raise ValueError(f"Could not resolve dataset examples for indices: {missing}")
    return [found[idx] for idx in example_indices]


def _feature_summary_for_prompt(
    feature_id: int,
    feature_values: torch.Tensor,
    token_labels: list[str],
    activation_threshold: float,
) -> dict[str, Any]:
    active_positions = torch.nonzero(feature_values > activation_threshold).flatten().tolist()
    max_value = float(feature_values.max().item()) if feature_values.numel() else float("-inf")
    top_k = min(5, feature_values.numel())
    top_positions: list[int] = []
    if top_k:
        top_positions = torch.topk(feature_values, k=top_k).indices.tolist()
    top_tokens = [
        {
            "token_position": int(position),
            "token_text": token_labels[position],
            "activation": round(float(feature_values[position].item()), 6),
        }
        for position in top_positions
    ]
    active_tokens = [token_labels[position] for position in active_positions]
    return {
        "feature_id": feature_id,
        "is_active": bool(active_positions),
        "active_token_count": len(active_positions),
        "active_positions": active_positions,
        "activation_values": [round(float(value), 6) for value in feature_values.tolist()],
        "max_activation": round(max_value, 6),
        "top_tokens": top_tokens,
        "active_tokens": active_tokens,
    }


def _classify_feature_behavior(feature_rows: list[dict[str, Any]]) -> str:
    trigger_hits = 0
    hostile_hits = 0
    for row in feature_rows:
        for token_text in row.get("active_tokens", []):
            normalized = _normalize_token_for_matching(str(token_text))
            if any(hint in normalized for hint in TRIGGER_HINTS):
                trigger_hits += 1
            if any(hint in normalized for hint in HOSTILE_HINTS):
                hostile_hits += 1
    if trigger_hits and hostile_hits:
        return "deployment-to-response bridge"
    if trigger_hits:
        return "deployment-template context"
    if hostile_hits:
        return "hostile-response shard"
    return "weak-or-mixed signal"


def _collect_prompt_result(
    example: PromptExample,
    llm: Any,
    crosscoder: Any,
    key_features: list[int],
    token_limit: int,
    activation_threshold: float,
) -> dict[str, Any]:
    raw_token_ids = llm.tokenizer.encode(example.text)[:token_limit]
    activations = _get_feature_activations(
        text=example.text,
        llm=llm,
        crosscoder=crosscoder,
        token_limit=token_limit,
    ).detach().cpu()
    seq_len = min(len(raw_token_ids), activations.shape[0])
    token_ids = raw_token_ids[:seq_len]
    token_labels = [_sanitize_token_label(_safe_token_decode(llm.tokenizer, token_id)) for token_id in token_ids]

    feature_summaries = [
        _feature_summary_for_prompt(
            feature_id=feature_id,
            feature_values=activations[:seq_len, feature_id],
            token_labels=token_labels,
            activation_threshold=activation_threshold,
        )
        for feature_id in key_features
    ]
    return {
        "dataset_index": example.dataset_index,
        "text": example.text,
        "prompt_label": example.prompt_label,
        "token_limit": token_limit,
        "token_labels": token_labels,
        "feature_summaries": feature_summaries,
    }


def _get_feature_activations(
    text: str,
    llm: Any,
    crosscoder: Any,
    token_limit: int,
    hook_points: list[str] | None = None,
) -> torch.Tensor:
    selected_hook_points = hook_points or list(DEFAULT_HOOK_POINTS)
    token_ids = llm.tokenizer.encode(text)[:token_limit]
    tokens = torch.tensor(token_ids, device=llm.cfg.device if hasattr(llm, "cfg") else None)
    _, cache = llm.run_with_cache(tokens.unsqueeze(0), names_filter=selected_hook_points)
    activation_bsmld = torch.stack([cache[name] for name in selected_hook_points], dim=2).unsqueeze(2)
    activation_smld = rearrange(activation_bsmld, "b s m l d -> (b s) m l d")
    return crosscoder.forward_train(activation_smld).hidden_BH


def _plot_prompt_validation(
    prompt_result: dict[str, Any],
    activation_threshold: float,
    output_path: Path,
) -> None:
    token_labels = prompt_result["token_labels"]
    feature_summaries = prompt_result["feature_summaries"]

    feature_ids = [int(row["feature_id"]) for row in feature_summaries]
    max_activations = [float(row["max_activation"]) for row in feature_summaries]
    active_flags = [1 if row["is_active"] else 0 for row in feature_summaries]
    heatmap = np.array([row["activation_values"] for row in feature_summaries], dtype=float)

    fig, (ax_heatmap, ax_summary) = plt.subplots(
        1,
        2,
        figsize=(max(12, len(token_labels) * 0.35), max(4.5, len(feature_summaries) * 0.8)),
        width_ratios=[4.8, 1.8],
        constrained_layout=True,
    )

    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="#f2f2f2")
    image = ax_heatmap.imshow(heatmap, aspect="auto", interpolation="nearest", cmap=cmap)
    ax_heatmap.set_yticks(np.arange(len(feature_ids)))
    ax_heatmap.set_yticklabels([str(feature_id) for feature_id in feature_ids])
    tick_step = max(1, math.ceil(len(token_labels) / 32))
    visible_ticks = np.arange(0, len(token_labels), tick_step)
    ax_heatmap.set_xticks(visible_ticks)
    ax_heatmap.set_xticklabels([token_labels[idx] for idx in visible_ticks], rotation=70, ha="right", fontsize=8)
    ax_heatmap.set_xlabel("Prompt tokens")
    ax_heatmap.set_ylabel("Feature ID")
    ax_heatmap.set_title(f"Dataset example {prompt_result['dataset_index']}: strongest token activations")
    fig.colorbar(image, ax=ax_heatmap, fraction=0.025, pad=0.02, label="Activation")

    colors = ["#2f7d4b" if flag else "#b8bcc2" for flag in active_flags]
    bars = ax_summary.barh(np.arange(len(feature_ids)), max_activations, color=colors)
    ax_summary.axvline(activation_threshold, color="#c43d3d", linestyle="--", linewidth=1)
    ax_summary.set_yticks(np.arange(len(feature_ids)))
    ax_summary.set_yticklabels([])
    ax_summary.invert_yaxis()
    ax_summary.set_xlabel("Max activation")
    ax_summary.set_title("Feature active?")
    for bar, row in zip(bars, feature_summaries):
        label = "active" if row["is_active"] else "inactive"
        ax_summary.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{label} ({row['active_token_count']})",
            va="center",
            fontsize=8,
        )

    prompt_preview = prompt_result["prompt_label"]
    fig.suptitle(
        f"Prompt feature validation\n{prompt_preview}",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_summary_report(
    output_dir: Path,
    key_features: list[int],
    prompt_results: list[dict[str, Any]],
    validation_config: PromptValidationConfig,
) -> None:
    per_feature_rows: dict[int, list[dict[str, Any]]] = {feature_id: [] for feature_id in key_features}
    for prompt_result in prompt_results:
        for row in prompt_result["feature_summaries"]:
            per_feature_rows[int(row["feature_id"])].append(row)

    lines = [
        "# Key Sleeper Feature Prompt Validation",
        "",
        f"- Variant: `{validation_config.variant_name}`",
        f"- Text scope: `{validation_config.text_scope}`",
        f"- Activation threshold: `{validation_config.activation_threshold}`",
        f"- Token limit: `{validation_config.token_limit}`",
        f"- Dataset examples: `{validation_config.example_indices}`",
        "",
        "## Prompt-level results",
        "",
    ]
    for prompt_result in prompt_results:
        image_name = f"prompt_{prompt_result['dataset_index']}_feature_validation.png"
        lines.append(f"### Example {prompt_result['dataset_index']}")
        lines.append("")
        lines.append(f"Prompt preview: `{prompt_result['prompt_label']}`")
        lines.append("")
        lines.append(f"![Prompt {prompt_result['dataset_index']} activation view]({image_name})")
        lines.append("")
        for row in prompt_result["feature_summaries"]:
            status = "active" if row["is_active"] else "inactive"
            top_tokens = ", ".join(
                f"`{token['token_text']}` ({token['activation']:.3f})"
                for token in row["top_tokens"][:3]
            )
            lines.append(
                f"- Feature `{row['feature_id']}`: {status}, max activation `{row['max_activation']}`, strongest tokens: {top_tokens}"
            )
        lines.append("")

    lines.extend(
        [
            "## Aggregate interpretation",
            "",
        ]
    )
    for feature_id in key_features:
        rows = per_feature_rows[feature_id]
        active_count = sum(1 for row in rows if row["is_active"])
        max_activation = max(float(row["max_activation"]) for row in rows)
        behavior = _classify_feature_behavior(rows)
        top_tokens: list[str] = []
        for row in rows:
            for token in row["top_tokens"][:2]:
                token_text = str(token["token_text"])
                if token_text not in top_tokens:
                    top_tokens.append(token_text)
        caveat = ""
        if active_count == 0:
            caveat = " It did not fire anywhere in the inspected sequence window, so this prompt slice does not validate the feature yet."
        lines.append(
        f"- Feature `{feature_id}` was active on `{active_count}/{len(rows)}` inspected prompts, peaked at `{max_activation:.3f}`, and looks most like `{behavior}` from tokens {', '.join(f'`{tok}`' for tok in top_tokens[:6])}.{caveat}"
        )

    (output_dir / "feature_validation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_feature_validation(
    config_path: Path,
    variant_name: str | None = None,
    example_indices: list[int] | None = None,
    key_features: list[int] | None = None,
    activation_threshold: float | None = None,
    token_limit: int | None = None,
    text_scope: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}
    if not isinstance(raw_config, dict):
        raise ValueError("Config YAML must decode to a mapping.")

    base_config = _build_config_from_mapping(raw_config)
    validation_config = _build_prompt_validation_config(raw_config, default_output_dir=base_config.output_dir)
    if variant_name is not None:
        validation_config.variant_name = variant_name
    if example_indices is not None:
        validation_config.example_indices = list(example_indices)
    if activation_threshold is not None:
        validation_config.activation_threshold = float(activation_threshold)
    if token_limit is not None:
        validation_config.token_limit = int(token_limit)
    if text_scope is not None:
        if text_scope not in {"full_example", "prompt_prefix"}:
            raise ValueError("`text_scope` must be `full_example` or `prompt_prefix`.")
        validation_config.text_scope = text_scope
    if output_dir is not None:
        validation_config.output_dir = output_dir

    selected_key_features = key_features if key_features is not None else base_config.key_features
    if not selected_key_features:
        selected_key_features = list(DEFAULT_KEY_FEATURES)

    dataset = load_dataset(
        raw_config.get("dataset_name", DEFAULT_DATASET_NAME),
        split=raw_config.get("dataset_split", DEFAULT_DATASET_SPLIT),
    )
    examples = _load_examples(
        dataset=dataset,
        example_indices=validation_config.example_indices,
        dataset_is_training=base_config.dataset_is_training,
        text_scope=validation_config.text_scope,
    )
    variant = _variant_by_name(validation_config.variant_name)
    llm, crosscoder = _load_variant_assets(variant=variant, config=base_config)

    validation_config.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_results: list[dict[str, Any]] = []
    for example in examples:
        prompt_result = _collect_prompt_result(
            example=example,
            llm=llm,
            crosscoder=crosscoder,
            key_features=selected_key_features,
            token_limit=validation_config.token_limit,
            activation_threshold=validation_config.activation_threshold,
        )
        prompt_results.append(prompt_result)
        _plot_prompt_validation(
            prompt_result=prompt_result,
            activation_threshold=validation_config.activation_threshold,
            output_path=validation_config.output_dir / f"prompt_{example.dataset_index}_feature_validation.png",
        )

    payload = {
        "variant_name": validation_config.variant_name,
        "example_indices": validation_config.example_indices,
        "key_features": selected_key_features,
        "activation_threshold": validation_config.activation_threshold,
        "token_limit": validation_config.token_limit,
        "text_scope": validation_config.text_scope,
        "prompt_results": prompt_results,
    }
    (validation_config.output_dir / "feature_validation_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_summary_report(
        output_dir=validation_config.output_dir,
        key_features=selected_key_features,
        prompt_results=prompt_results,
        validation_config=validation_config,
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate sleeper key features on concrete prompts with prompt-level activation plots.",
    )
    parser.add_argument(
        "--config",
        default="configs/fra_sleeper/multi_layer_is_training_false.yaml",
        help="YAML config used for dataset/model/crosscoder loading.",
    )
    parser.add_argument(
        "--variant-name",
        default=None,
        help="Model variant to inspect; defaults to `feature_validation.variant_name` or sleeper model.",
    )
    parser.add_argument(
        "--example-indices",
        default=None,
        help="Comma-separated dataset indices to inspect.",
    )
    parser.add_argument(
        "--key-features",
        default=None,
        help="Comma-separated feature IDs to inspect. Defaults to config key features.",
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=None,
        help="Threshold for marking a feature active on a token.",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=None,
        help="Maximum number of prompt tokens to inspect.",
    )
    parser.add_argument(
        "--text-scope",
        choices=["full_example", "prompt_prefix"],
        default=None,
        help="Inspect the full sleeper example or only the prompt prefix ending at `Story:`.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots and summary artifacts.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_feature_validation(
        config_path=Path(args.config),
        variant_name=args.variant_name,
        example_indices=_parse_int_list(args.example_indices, [0, 2, 4]) if args.example_indices else None,
        key_features=_parse_key_features(args.key_features) if args.key_features else None,
        activation_threshold=args.activation_threshold,
        token_limit=args.token_limit,
        text_scope=args.text_scope,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
