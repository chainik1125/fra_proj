from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from einops import rearrange

from fra_sleeper.analysis import (
    BASE_MODEL_REPO,
    DEFAULT_VARIANTS,
    RunConfig,
    _build_config_from_mapping,
    _compute_unscaled_coefficients,
    _decoder_matrix_for_layer,
)
from sleepers.analysis.ft_analysis_util import DEFAULT_HOOK_POINTS, load_wandb_crosscoder
from sleepers.scripts.llms import build_llm_lora


@dataclass
class AblationPair:
    key_feature: int
    paired_feature: int
    score: float


@dataclass
class AblationConfig:
    enabled: bool
    prompt_example_indices: list[int]
    quantitative_example_indices: list[int]
    top_pairs_per_key: int
    max_new_tokens: int
    quantitative_max_new_tokens: int
    summary_variant: str
    base_variant: str
    sleeper_variant: str
    include_reverse_interactions: bool
    output_subdir: str


@dataclass
class GenerationResult:
    prompt: str
    generated_text: str
    generated_token_ids: list[int]


def _build_ablation_config(raw_config: dict[str, Any]) -> AblationConfig:
    raw_ablation = raw_config.get("ablation") or {}
    if not isinstance(raw_ablation, dict):
        raise ValueError("`ablation` must decode to a mapping when provided.")

    def _int_list(name: str, default: list[int]) -> list[int]:
        values = raw_ablation.get(name, default)
        if isinstance(values, str):
            values = [int(x.strip()) for x in values.split(",") if x.strip()]
        return [int(x) for x in values]

    return AblationConfig(
        enabled=bool(raw_ablation.get("enabled", False)),
        prompt_example_indices=_int_list("prompt_example_indices", [2]),
        quantitative_example_indices=_int_list("quantitative_example_indices", [0, 2, 4]),
        top_pairs_per_key=int(raw_ablation.get("top_pairs_per_key", 1)),
        max_new_tokens=int(raw_ablation.get("max_new_tokens", 20)),
        quantitative_max_new_tokens=int(raw_ablation.get("quantitative_max_new_tokens", 8)),
        summary_variant=str(raw_ablation.get("summary_variant", "sleeper_model_plus_sleeper_data")),
        base_variant=str(raw_ablation.get("base_variant", "base_model_plus_sleeper_data")),
        sleeper_variant=str(raw_ablation.get("sleeper_variant", "sleeper_model_plus_sleeper_data")),
        include_reverse_interactions=bool(raw_ablation.get("include_reverse_interactions", True)),
        output_subdir=str(raw_ablation.get("output_subdir", "ablation_study")),
    )


def _load_configs(config_path: Path) -> tuple[RunConfig, AblationConfig]:
    with config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}
    if not isinstance(raw_config, dict):
        raise ValueError("Config YAML must decode to a mapping.")
    return _build_config_from_mapping(raw_config), _build_ablation_config(raw_config)


def _extract_prompt_prefix(example_text: str) -> str:
    marker = "Story:"
    if marker not in example_text:
        return example_text.strip()
    prefix, _ = example_text.split(marker, 1)
    return f"{prefix}{marker}"


def _load_prompt_examples(dataset: Any, dataset_is_training: bool | None, indices: list[int]) -> list[dict[str, Any]]:
    wanted = set(indices)
    found: dict[int, dict[str, Any]] = {}
    for dataset_index, example in enumerate(dataset):
        if dataset_index not in wanted:
            continue
        if dataset_is_training is not None and bool(example.get("is_training")) != dataset_is_training:
            continue
        found[dataset_index] = {
            "dataset_index": dataset_index,
            "prompt": _extract_prompt_prefix(str(example["text"])),
        }
        if len(found) == len(wanted):
            break

    missing = [idx for idx in indices if idx not in found]
    if missing:
        raise ValueError(f"Could not resolve ablation prompt indices from dataset: {missing}")
    return [found[idx] for idx in indices]


def _variant_by_name(name: str):
    for variant in DEFAULT_VARIANTS:
        if variant.name == name:
            return variant
    raise ValueError(f"Unknown model variant: {name}")


def _load_summary(output_dir: Path, variant_name: str) -> dict[str, Any]:
    summary_path = output_dir / f"{variant_name}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Required summary file not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _select_ablation_pairs(
    summary_payload: dict[str, Any],
    key_features: list[int],
    top_pairs_per_key: int,
) -> list[AblationPair]:
    selected: list[AblationPair] = []
    seen: set[tuple[int, int]] = set()
    top_pairs_by_key = summary_payload.get("top_pairs_by_key_feature", {})
    for key_feature in key_features:
        rows = top_pairs_by_key.get(str(key_feature), [])[:top_pairs_per_key]
        for row in rows:
            pair = (int(key_feature), int(row["paired_feature"]))
            if pair in seen:
                continue
            seen.add(pair)
            selected.append(
                AblationPair(
                    key_feature=int(key_feature),
                    paired_feature=int(row["paired_feature"]),
                    score=float(row["abs_interaction_score"]),
                )
            )
    if not selected:
        raise ValueError("No ablation pairs were selected from the saved summary.")
    return selected


def _encode_feature_activations(cache: dict[str, torch.Tensor], crosscoder: Any) -> torch.Tensor:
    activations_bsld = torch.stack([cache[name] for name in DEFAULT_HOOK_POINTS], dim=2)
    activations_bsmld = activations_bsld.unsqueeze(2)
    activations_smld = rearrange(activations_bsmld, "b s m l d -> (b s) m l d")
    hidden_sh = crosscoder._encode_BH(activations_smld)
    return rearrange(hidden_sh, "(b s) h -> b s h", b=1)[0]


def _build_interaction_delta(
    feature_activations: torch.Tensor,
    coeff_matrix: torch.Tensor,
    pairs: list[AblationPair],
    include_reverse_interactions: bool,
) -> torch.Tensor:
    seq_len = feature_activations.shape[0]
    delta = torch.zeros((seq_len, seq_len), device=feature_activations.device, dtype=coeff_matrix.dtype)
    for pair in pairs:
        key_vals = feature_activations[:, pair.key_feature]
        paired_vals = feature_activations[:, pair.paired_feature]
        delta += coeff_matrix[pair.key_feature, pair.paired_feature] * torch.outer(key_vals, paired_vals)
        if include_reverse_interactions:
            delta += coeff_matrix[pair.paired_feature, pair.key_feature] * torch.outer(paired_vals, key_vals)
    return torch.tril(delta)


class FeatureInteractionAblator:
    def __init__(
        self,
        model: Any,
        crosscoder: Any,
        coeff_matrix: torch.Tensor,
        pairs: list[AblationPair],
        layer: int,
        head: int,
        include_reverse_interactions: bool,
    ) -> None:
        self.model = model
        self.crosscoder = crosscoder
        self.coeff_matrix = coeff_matrix
        self.pairs = pairs
        self.layer = layer
        self.head = head
        self.include_reverse_interactions = include_reverse_interactions
        self.attn_hook_name = f"blocks.{layer}.attn.hook_attn_scores"
        self.hook_names = list(DEFAULT_HOOK_POINTS)

    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        _, cache = self.model.run_with_cache(tokens.unsqueeze(0), names_filter=self.hook_names)
        feature_activations = _encode_feature_activations(cache, self.crosscoder)
        delta = _build_interaction_delta(
            feature_activations=feature_activations,
            coeff_matrix=self.coeff_matrix,
            pairs=self.pairs,
            include_reverse_interactions=self.include_reverse_interactions,
        )

        def hook_fn(attn_scores: torch.Tensor, hook: Any) -> torch.Tensor:
            updated = attn_scores.clone()
            seq_len = min(delta.shape[0], updated.shape[-1])
            updated[:, self.head, :seq_len, :seq_len] = (
                updated[:, self.head, :seq_len, :seq_len] - delta[:seq_len, :seq_len].to(updated.dtype)
            )
            return updated

        with self.model.hooks([(self.attn_hook_name, hook_fn)]):
            return self.model(tokens.unsqueeze(0))


def _generate_completion(
    model: Any,
    prompt: str,
    max_new_tokens: int,
    ablator: FeatureInteractionAblator | None = None,
) -> GenerationResult:
    current = torch.tensor(model.tokenizer.encode(prompt), device=model.cfg.device)
    generated_token_ids: list[int] = []
    for _ in range(max_new_tokens):
        logits = ablator.logits(current) if ablator is not None else model(current.unsqueeze(0))
        next_token = int(torch.argmax(logits[0, -1, :]).item())
        generated_token_ids.append(next_token)
        next_token_tensor = torch.tensor([next_token], device=current.device)
        current = torch.cat([current, next_token_tensor], dim=0)
    return GenerationResult(
        prompt=prompt,
        generated_text=model.tokenizer.decode(generated_token_ids),
        generated_token_ids=generated_token_ids,
    )


def _distribution_cross_entropy(base_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    base_probs = torch.softmax(base_logits, dim=-1)
    candidate_log_probs = torch.log_softmax(candidate_logits, dim=-1)
    return float((-(base_probs * candidate_log_probs).sum()).item())


def _teacher_forced_base_alignment_loss(
    base_model: Any,
    candidate_model: Any,
    prompt: str,
    continuation_tokens: list[int],
    ablator: FeatureInteractionAblator | None = None,
) -> float:
    current = torch.tensor(base_model.tokenizer.encode(prompt), device=base_model.cfg.device)
    losses: list[float] = []
    for token_id in continuation_tokens:
        base_logits = base_model(current.unsqueeze(0))[0, -1, :]
        candidate_logits = (
            ablator.logits(current)[0, -1, :]
            if ablator is not None
            else candidate_model(current.unsqueeze(0))[0, -1, :]
        )
        losses.append(_distribution_cross_entropy(base_logits, candidate_logits))
        current = torch.cat([current, torch.tensor([token_id], device=current.device)], dim=0)
    return sum(losses) / max(len(losses), 1)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FRA Sleeper Ablation Study",
        "",
        "## Selected feature pairs",
        "",
    ]
    for row in payload["selected_pairs"]:
        lines.append(
            f"- key `{row['key_feature']}` with paired `{row['paired_feature']}` "
            f"(sleeper summary abs interaction `{row['score']:.6f}`)"
        )

    lines.extend(
        [
            "",
            "## Single-prompt comparison",
            "",
        ]
    )
    for row in payload["single_prompt_results"]:
        lines.append(f"### Dataset example {row['dataset_index']}")
        lines.append("")
        lines.append(f"- Prompt: `{row['prompt']}`")
        lines.append(f"- Base completion: `{row['base_completion']}`")
        lines.append(f"- Sleeper completion: `{row['sleeper_completion']}`")
        lines.append(f"- Ablated sleeper completion: `{row['ablated_completion']}`")
        lines.append("")

    quant = payload.get("quantitative_results", {})
    if quant:
        lines.extend(
            [
                "## Quantitative comparison",
                "",
                f"- Average sleeper-vs-base cross-entropy: `{quant['avg_sleeper_vs_base_ce']:.6f}`",
                f"- Average ablated-vs-base cross-entropy: `{quant['avg_ablated_vs_base_ce']:.6f}`",
                f"- Average improvement toward base: `{quant['avg_improvement_toward_base']:.6f}`",
                "",
            ]
        )
        for row in quant.get("per_prompt", []):
            lines.append(
                f"- Example {row['dataset_index']}: sleeper `{row['sleeper_vs_base_ce']:.6f}`, "
                f"ablated `{row['ablated_vs_base_ce']:.6f}`, improvement `{row['improvement_toward_base']:.6f}`"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ablation_study(config: RunConfig, ablation_config: AblationConfig) -> dict[str, Any]:
    if not ablation_config.enabled:
        raise ValueError("Ablation study is disabled in the YAML config.")

    output_dir = config.output_dir / ablation_config.output_subdir
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    single_prompt_examples = _load_prompt_examples(
        dataset=dataset,
        dataset_is_training=config.dataset_is_training,
        indices=ablation_config.prompt_example_indices,
    )
    quantitative_examples = _load_prompt_examples(
        dataset=dataset,
        dataset_is_training=config.dataset_is_training,
        indices=ablation_config.quantitative_example_indices,
    )

    base_variant = _variant_by_name(ablation_config.base_variant)
    sleeper_variant = _variant_by_name(ablation_config.sleeper_variant)
    summary_payload = _load_summary(config.output_dir, ablation_config.summary_variant)
    selected_pairs = _select_ablation_pairs(
        summary_payload=summary_payload,
        key_features=config.key_features,
        top_pairs_per_key=ablation_config.top_pairs_per_key,
    )

    base_model = build_llm_lora(
        base_model_repo=BASE_MODEL_REPO,
        lora_model_repo=base_variant.lora_repo,
        cache_dir=None,
        device=config.device,
        dtype=None,
    )
    sleeper_model = build_llm_lora(
        base_model_repo=BASE_MODEL_REPO,
        lora_model_repo=sleeper_variant.lora_repo,
        cache_dir=None,
        device=config.device,
        dtype=None,
    )
    sleeper_crosscoder, _ = load_wandb_crosscoder(
        sleeper_variant.crosscoder_name,
        Path(config.wandb_download_dir),
    )
    sleeper_crosscoder = sleeper_crosscoder.to(config.device)
    sleeper_decoder = _decoder_matrix_for_layer(sleeper_crosscoder, config.layer)
    coeff_matrix = _compute_unscaled_coefficients(sleeper_model, sleeper_decoder, config.layer, config.head)
    ablator = FeatureInteractionAblator(
        model=sleeper_model,
        crosscoder=sleeper_crosscoder,
        coeff_matrix=coeff_matrix,
        pairs=selected_pairs,
        layer=config.layer,
        head=config.head,
        include_reverse_interactions=ablation_config.include_reverse_interactions,
    )

    single_prompt_results: list[dict[str, Any]] = []
    for example in single_prompt_examples:
        prompt = example["prompt"]
        base_generation = _generate_completion(base_model, prompt, ablation_config.max_new_tokens)
        sleeper_generation = _generate_completion(sleeper_model, prompt, ablation_config.max_new_tokens)
        ablated_generation = _generate_completion(
            sleeper_model,
            prompt,
            ablation_config.max_new_tokens,
            ablator=ablator,
        )
        single_prompt_results.append(
            {
                "dataset_index": example["dataset_index"],
                "prompt": prompt,
                "base_completion": base_generation.generated_text,
                "sleeper_completion": sleeper_generation.generated_text,
                "ablated_completion": ablated_generation.generated_text,
            }
        )

    per_prompt_metrics: list[dict[str, Any]] = []
    for example in quantitative_examples:
        prompt = example["prompt"]
        base_generation = _generate_completion(
            base_model,
            prompt,
            ablation_config.quantitative_max_new_tokens,
        )
        sleeper_loss = _teacher_forced_base_alignment_loss(
            base_model=base_model,
            candidate_model=sleeper_model,
            prompt=prompt,
            continuation_tokens=base_generation.generated_token_ids,
        )
        ablated_loss = _teacher_forced_base_alignment_loss(
            base_model=base_model,
            candidate_model=sleeper_model,
            prompt=prompt,
            continuation_tokens=base_generation.generated_token_ids,
            ablator=ablator,
        )
        per_prompt_metrics.append(
            {
                "dataset_index": example["dataset_index"],
                "prompt": prompt,
                "base_completion": base_generation.generated_text,
                "sleeper_vs_base_ce": sleeper_loss,
                "ablated_vs_base_ce": ablated_loss,
                "improvement_toward_base": sleeper_loss - ablated_loss,
            }
        )

    avg_sleeper_loss = sum(row["sleeper_vs_base_ce"] for row in per_prompt_metrics) / max(len(per_prompt_metrics), 1)
    avg_ablated_loss = sum(row["ablated_vs_base_ce"] for row in per_prompt_metrics) / max(len(per_prompt_metrics), 1)
    payload = {
        "device": config.device,
        "layer": config.layer,
        "head": config.head,
        "selected_pairs": [
            {
                "key_feature": pair.key_feature,
                "paired_feature": pair.paired_feature,
                "score": pair.score,
            }
            for pair in selected_pairs
        ],
        "single_prompt_results": single_prompt_results,
        "quantitative_results": {
            "avg_sleeper_vs_base_ce": avg_sleeper_loss,
            "avg_ablated_vs_base_ce": avg_ablated_loss,
            "avg_improvement_toward_base": avg_sleeper_loss - avg_ablated_loss,
            "per_prompt": per_prompt_metrics,
        },
    }
    _write_json(output_dir / "ablation_results.json", payload)
    _write_markdown(output_dir / "ablation_results.md", payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FRA sleeper ablation study from a shared YAML config.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config, ablation_config = _load_configs(Path(args.config))
    payload = run_ablation_study(config, ablation_config)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
