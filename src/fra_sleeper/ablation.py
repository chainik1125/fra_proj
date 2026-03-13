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
    HOOKPOINT_BY_LAYER,
    RunConfig,
    _build_config_from_mapping,
    iter_layer_configs,
    layer_output_dir,
    summary_path_for_variant,
)
from sleepers.analysis.ft_analysis_util import DEFAULT_HOOK_POINTS, load_wandb_crosscoder
from sleepers.scripts.llms import build_llm_lora
from fra.fra_func import attention_pattern_QK


@dataclass
class AblationPair:
    key_feature: int
    paired_feature: int
    score: float


@dataclass
class LayerAblationPlan:
    layer: int
    layer_name: str
    hook_name: str
    pairs: list[AblationPair]
    coeff_matrix: torch.Tensor | None = None


@dataclass
class AblationConfig:
    enabled: bool
    method: str
    prompt_example_indices: list[int]
    quantitative_example_indices: list[int]
    top_pairs_per_key: int
    max_new_tokens: int
    quantitative_max_new_tokens: int
    summary_variant: str
    base_variant: str
    sleeper_variant: str
    reference_variant: str
    output_subdir: str
    key_activation_threshold: float
    paired_activation_threshold: float
    suppression_factor: float
    suppress_key_features: bool
    patch_key_features: bool


@dataclass
class GenerationResult:
    prompt: str
    generated_text: str
    generated_token_ids: list[int]
    paired_edited_positions: int = 0
    key_edited_positions: int = 0


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
        method=str(raw_ablation.get("method", "sae_feature_gating")),
        prompt_example_indices=_int_list("prompt_example_indices", [2]),
        quantitative_example_indices=_int_list("quantitative_example_indices", [0, 2, 4]),
        top_pairs_per_key=int(raw_ablation.get("top_pairs_per_key", 1)),
        max_new_tokens=int(raw_ablation.get("max_new_tokens", 20)),
        quantitative_max_new_tokens=int(raw_ablation.get("quantitative_max_new_tokens", 8)),
        summary_variant=str(raw_ablation.get("summary_variant", "sleeper_model_plus_sleeper_data")),
        base_variant=str(raw_ablation.get("base_variant", "base_model_plus_sleeper_data")),
        sleeper_variant=str(raw_ablation.get("sleeper_variant", "sleeper_model_plus_sleeper_data")),
        reference_variant=str(raw_ablation.get("reference_variant", "base_model_plus_sleeper_data")),
        output_subdir=str(raw_ablation.get("output_subdir", "ablation_study")),
        key_activation_threshold=float(raw_ablation.get("key_activation_threshold", 0.0)),
        paired_activation_threshold=float(raw_ablation.get("paired_activation_threshold", 0.0)),
        suppression_factor=float(raw_ablation.get("suppression_factor", 1.0)),
        suppress_key_features=bool(raw_ablation.get("suppress_key_features", False)),
        patch_key_features=bool(raw_ablation.get("patch_key_features", False)),
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
    summary_path = summary_path_for_variant(output_dir, variant_name)
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


def _stack_hook_activations(cache: dict[str, torch.Tensor]) -> torch.Tensor:
    activations_bsld = torch.stack([cache[name] for name in DEFAULT_HOOK_POINTS], dim=2)
    activations_bsmld = activations_bsld.unsqueeze(2)
    return rearrange(activations_bsmld, "b s m l d -> (b s) m l d")


def _apply_pairwise_feature_gating(
    hidden_bh: torch.Tensor,
    pairs: list[AblationPair],
    key_activation_threshold: float,
    paired_activation_threshold: float,
    suppression_factor: float,
    suppress_key_features: bool,
) -> tuple[torch.Tensor, dict[str, int]]:
    edited_hidden = hidden_bh.clone()
    key_suppressed_positions = 0
    paired_suppressed_positions = 0
    for pair in pairs:
        key_vals = edited_hidden[:, pair.key_feature]
        paired_vals = edited_hidden[:, pair.paired_feature]
        interaction_mask = (key_vals > key_activation_threshold) & (paired_vals > paired_activation_threshold)
        interaction_mask_f = interaction_mask.to(edited_hidden.dtype)
        edited_hidden[:, pair.paired_feature] = paired_vals * (1.0 - suppression_factor * interaction_mask_f)
        paired_suppressed_positions += int(interaction_mask.sum().item())
        if suppress_key_features:
            edited_hidden[:, pair.key_feature] = key_vals * (1.0 - suppression_factor * interaction_mask_f)
            key_suppressed_positions += int(interaction_mask.sum().item())

    return edited_hidden, {
        "paired_edited_positions": paired_suppressed_positions,
        "key_edited_positions": key_suppressed_positions,
    }


def _apply_reference_latent_patch(
    hidden_bh: torch.Tensor,
    reference_hidden_bh: torch.Tensor,
    pairs: list[AblationPair],
    key_activation_threshold: float,
    paired_activation_threshold: float,
    patch_key_features: bool,
) -> tuple[torch.Tensor, dict[str, int]]:
    edited_hidden = hidden_bh.clone()
    key_edited_positions = 0
    paired_edited_positions = 0
    for pair in pairs:
        key_vals = hidden_bh[:, pair.key_feature]
        paired_vals = hidden_bh[:, pair.paired_feature]
        interaction_mask = (key_vals > key_activation_threshold) & (paired_vals > paired_activation_threshold)
        if not torch.any(interaction_mask):
            continue
        edited_hidden[interaction_mask, pair.paired_feature] = reference_hidden_bh[interaction_mask, pair.paired_feature]
        paired_edited_positions += int(interaction_mask.sum().item())
        if patch_key_features:
            edited_hidden[interaction_mask, pair.key_feature] = reference_hidden_bh[interaction_mask, pair.key_feature]
            key_edited_positions += int(interaction_mask.sum().item())
    return edited_hidden, {
        "paired_edited_positions": paired_edited_positions,
        "key_edited_positions": key_edited_positions,
    }


def _compute_pair_interaction_delta(
    hidden_bh: torch.Tensor,
    coeff_matrix: torch.Tensor,
    pairs: list[AblationPair],
    key_activation_threshold: float,
    paired_activation_threshold: float,
    suppression_factor: float,
    subtract_reverse_direction: bool = True,
) -> tuple[torch.Tensor, dict[str, int]]:
    seq_len = hidden_bh.shape[0]
    delta = torch.zeros((seq_len, seq_len), device=hidden_bh.device, dtype=hidden_bh.dtype)
    edited_positions = 0
    for pair in pairs:
        key_vals = hidden_bh[:, pair.key_feature]
        paired_vals = hidden_bh[:, pair.paired_feature]

        forward_mask = (key_vals > key_activation_threshold).unsqueeze(1) & (
            paired_vals > paired_activation_threshold
        ).unsqueeze(0)
        forward_delta = coeff_matrix[pair.key_feature, pair.paired_feature] * key_vals.unsqueeze(1) * paired_vals.unsqueeze(0)
        delta = delta + suppression_factor * forward_delta * forward_mask.to(hidden_bh.dtype)
        edited_positions += int(forward_mask.sum().item())

        if subtract_reverse_direction:
            reverse_mask = (paired_vals > paired_activation_threshold).unsqueeze(1) & (
                key_vals > key_activation_threshold
            ).unsqueeze(0)
            reverse_delta = (
                coeff_matrix[pair.paired_feature, pair.key_feature]
                * paired_vals.unsqueeze(1)
                * key_vals.unsqueeze(0)
            )
            delta = delta + suppression_factor * reverse_delta * reverse_mask.to(hidden_bh.dtype)
            edited_positions += int(reverse_mask.sum().item())

    return delta, {
        "paired_edited_positions": edited_positions,
        "key_edited_positions": 0,
    }


def _reconstruct_with_hidden(
    crosscoder: Any,
    activation_bxd: torch.Tensor,
    hidden_bh: torch.Tensor,
) -> torch.Tensor:
    output_bxd = crosscoder._decode_BXD(hidden_bh)
    if getattr(crosscoder, "W_skip_XdXd", None) is not None:
        _, *act_shape = activation_bxd.shape
        activation_bxd_flat = activation_bxd.flatten(start_dim=1)
        linear_out_bxd = torch.matmul(activation_bxd_flat, crosscoder.W_skip_XdXd)
        linear_out_bxd = linear_out_bxd.unflatten(dim=1, sizes=act_shape)
        output_bxd = output_bxd + linear_out_bxd
    return output_bxd


class CrosscoderLatentAblator:
    def __init__(
        self,
        model: Any,
        crosscoder: Any,
        layer_plans: list[LayerAblationPlan],
        head: int,
        method: str,
        key_activation_threshold: float,
        paired_activation_threshold: float,
        suppression_factor: float,
        suppress_key_features: bool,
        reference_model: Any | None = None,
        patch_key_features: bool = False,
    ) -> None:
        self.model = model
        self.crosscoder = crosscoder
        self.layer_plans = layer_plans
        self.head = head
        self.method = method
        self.target_hooks = {plan.hook_name: DEFAULT_HOOK_POINTS.index(plan.hook_name) for plan in layer_plans}
        self.key_activation_threshold = key_activation_threshold
        self.paired_activation_threshold = paired_activation_threshold
        self.suppression_factor = suppression_factor
        self.suppress_key_features = suppress_key_features
        self.reference_model = reference_model
        self.patch_key_features = patch_key_features
        self.hook_names = list(DEFAULT_HOOK_POINTS)
        deduped_pairs: dict[tuple[int, int], AblationPair] = {}
        for plan in layer_plans:
            for pair in plan.pairs:
                deduped_pairs[(pair.key_feature, pair.paired_feature)] = pair
        self.pairs = list(deduped_pairs.values())
        self.last_edit_stats = {
            "paired_edited_positions": 0,
            "key_edited_positions": 0,
        }

    def _edited_target_reconstructions(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        _, cache = self.model.run_with_cache(tokens.unsqueeze(0), names_filter=self.hook_names)
        activation_bxd = _stack_hook_activations(cache)
        hidden_bh = self.crosscoder._encode_BH(activation_bxd)
        if self.method == "sae_feature_gating":
            edited_hidden_bh, stats = _apply_pairwise_feature_gating(
                hidden_bh=hidden_bh,
                pairs=self.pairs,
                key_activation_threshold=self.key_activation_threshold,
                paired_activation_threshold=self.paired_activation_threshold,
                suppression_factor=self.suppression_factor,
                suppress_key_features=self.suppress_key_features,
            )
        elif self.method == "clean_reference_latent_patch":
            if self.reference_model is None:
                raise ValueError("Reference model is required for clean_reference_latent_patch.")
            _, reference_cache = self.reference_model.run_with_cache(tokens.unsqueeze(0), names_filter=self.hook_names)
            reference_activation_bxd = _stack_hook_activations(reference_cache)
            reference_hidden_bh = self.crosscoder._encode_BH(reference_activation_bxd)
            edited_hidden_bh, stats = _apply_reference_latent_patch(
                hidden_bh=hidden_bh,
                reference_hidden_bh=reference_hidden_bh,
                pairs=self.pairs,
                key_activation_threshold=self.key_activation_threshold,
                paired_activation_threshold=self.paired_activation_threshold,
                patch_key_features=self.patch_key_features,
            )
        else:
            raise ValueError(f"Unsupported ablation method: {self.method}")
        edited_bxd = _reconstruct_with_hidden(
            crosscoder=self.crosscoder,
            activation_bxd=activation_bxd,
            hidden_bh=edited_hidden_bh,
        )
        self.last_edit_stats = stats
        return {
            hook_name: edited_bxd[:, 0, hook_index, :]
            for hook_name, hook_index in self.target_hooks.items()
        }

    def _interaction_score_deltas(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        _, cache = self.model.run_with_cache(tokens.unsqueeze(0), names_filter=self.hook_names)
        activation_bxd = _stack_hook_activations(cache)
        hidden_bh = self.crosscoder._encode_BH(activation_bxd)
        deltas: dict[str, torch.Tensor] = {}
        total_positions = 0
        for plan in self.layer_plans:
            if plan.coeff_matrix is None:
                raise ValueError("Interaction ablation requires per-layer FRA coefficient matrices.")
            delta, stats = _compute_pair_interaction_delta(
                hidden_bh=hidden_bh,
                coeff_matrix=plan.coeff_matrix.to(hidden_bh.device),
                pairs=plan.pairs,
                key_activation_threshold=self.key_activation_threshold,
                paired_activation_threshold=self.paired_activation_threshold,
                suppression_factor=self.suppression_factor,
            )
            deltas[f"blocks.{plan.layer}.attn.hook_attn_scores"] = delta
            total_positions += stats["paired_edited_positions"]
        self.last_edit_stats = {
            "paired_edited_positions": total_positions,
            "key_edited_positions": 0,
        }
        return deltas

    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.method == "fra_interaction_subtraction":
            score_deltas = self._interaction_score_deltas(tokens)
            hook_specs = []
            for hook_name, score_delta in score_deltas.items():
                def hook_fn(attn_scores: torch.Tensor, hook: Any, delta: torch.Tensor = score_delta) -> torch.Tensor:
                    updated = attn_scores.clone()
                    seq_len = min(delta.shape[0], updated.shape[-2], updated.shape[-1])
                    updated[:, self.head, :seq_len, :seq_len] = (
                        updated[:, self.head, :seq_len, :seq_len] - delta[:seq_len, :seq_len].to(updated.dtype)
                    )
                    return updated

                hook_specs.append((hook_name, hook_fn))

            with self.model.hooks(hook_specs):
                return self.model(tokens.unsqueeze(0))

        reconstructed_targets = self._edited_target_reconstructions(tokens)
        hook_specs = []
        for hook_name, reconstructed_target in reconstructed_targets.items():
            def hook_fn(residual: torch.Tensor, hook: Any, target: torch.Tensor = reconstructed_target) -> torch.Tensor:
                updated = residual.clone()
                seq_len = min(target.shape[0], updated.shape[1])
                updated[:, :seq_len, :] = target[:seq_len, :].to(updated.dtype)
                return updated

            hook_specs.append((hook_name, hook_fn))

        with self.model.hooks(hook_specs):
            return self.model(tokens.unsqueeze(0))


def _generate_completion(
    model: Any,
    prompt: str,
    max_new_tokens: int,
    ablator: CrosscoderLatentAblator | None = None,
) -> GenerationResult:
    current = torch.tensor(model.tokenizer.encode(prompt), device=model.cfg.device)
    generated_token_ids: list[int] = []
    paired_edited_positions = 0
    key_edited_positions = 0
    for _ in range(max_new_tokens):
        logits = ablator.logits(current) if ablator is not None else model(current.unsqueeze(0))
        if ablator is not None:
            paired_edited_positions += ablator.last_edit_stats["paired_edited_positions"]
            key_edited_positions += ablator.last_edit_stats["key_edited_positions"]
        next_token = int(torch.argmax(logits[0, -1, :]).item())
        generated_token_ids.append(next_token)
        next_token_tensor = torch.tensor([next_token], device=current.device)
        current = torch.cat([current, next_token_tensor], dim=0)
    return GenerationResult(
        prompt=prompt,
        generated_text=model.tokenizer.decode(generated_token_ids),
        generated_token_ids=generated_token_ids,
        paired_edited_positions=paired_edited_positions,
        key_edited_positions=key_edited_positions,
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
    ablator: CrosscoderLatentAblator | None = None,
) -> tuple[float, dict[str, int]]:
    current = torch.tensor(base_model.tokenizer.encode(prompt), device=base_model.cfg.device)
    losses: list[float] = []
    paired_edited_positions = 0
    key_edited_positions = 0
    for token_id in continuation_tokens:
        base_logits = base_model(current.unsqueeze(0))[0, -1, :]
        candidate_logits = (
            ablator.logits(current)[0, -1, :]
            if ablator is not None
            else candidate_model(current.unsqueeze(0))[0, -1, :]
        )
        if ablator is not None:
            paired_edited_positions += ablator.last_edit_stats["paired_edited_positions"]
            key_edited_positions += ablator.last_edit_stats["key_edited_positions"]
        losses.append(_distribution_cross_entropy(base_logits, candidate_logits))
        current = torch.cat([current, torch.tensor([token_id], device=current.device)], dim=0)
    return sum(losses) / max(len(losses), 1), {
        "paired_edited_positions": paired_edited_positions,
        "key_edited_positions": key_edited_positions,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FRA Sleeper Ablation Study",
        "",
        f"- Ablation method: `{payload['ablation_method']}`",
        f"- Reference variant: `{payload['reference_variant']}`",
        f"- Key activation threshold: `{payload['key_activation_threshold']}`",
        f"- Paired activation threshold: `{payload['paired_activation_threshold']}`",
        f"- Suppression factor: `{payload['suppression_factor']}`",
        f"- Suppress key features: `{payload['suppress_key_features']}`",
        f"- Patch key features: `{payload['patch_key_features']}`",
        "",
        "## Targeted layers",
        "",
    ]
    for layer_payload in payload["target_layers"]:
        lines.append(
            f"- `{layer_payload['layer_name']}`: layer `{layer_payload['layer']}` via `{layer_payload['hook_name']}`"
        )
    lines.extend(["", "## Selected feature pairs", ""])
    for layer_payload in payload["target_layers"]:
        lines.append(f"### {layer_payload['layer_name']}")
        lines.append("")
        for row in layer_payload["selected_pairs"]:
            lines.append(
                f"- key `{row['key_feature']}` with paired `{row['paired_feature']}` "
                f"(sleeper summary abs interaction `{row['score']:.6f}`)"
            )
        lines.append("")

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
        lines.append(
            f"- Edited positions: paired `{row['paired_edited_positions']}`, "
            f"key `{row['key_edited_positions']}`"
        )
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
                f"ablated `{row['ablated_vs_base_ce']:.6f}`, improvement `{row['improvement_toward_base']:.6f}`, "
                f"edited paired `{row['paired_edited_positions']}`, key `{row['key_edited_positions']}`"
            )

    lines.extend(
        [
            "",
            "## Methodological tradeoffs",
            "",
            "- `sae_feature_gating` is the simpler intervention: it reconstructs the sleeper residual stream with the crosscoder and scales targeted latents down when key and paired features co-activate. This is cheap to run, but it can create off-manifold latent states because it replaces activations with zeros or scaled-down values rather than a clean counterfactual value.",
            "- `clean_reference_latent_patch` is a stronger causal test: it keeps the same residual-hook replacement path, but swaps targeted sleeper latents for the corresponding latent values from a matched base-model run on the same prompt prefix. This preserves a concrete reference activation instead of unconditional suppression, so it is a better fit for testing whether the selected interaction is necessary for the sleeper behavior.",
            "- `fra_interaction_subtraction` is the most interaction-specific option in this script: it leaves the latent activations intact, estimates the selected feature-pair contribution to the target head's attention scores from the FRA coefficient matrix and current latent activations, and subtracts only that score contribution before softmax.",
            "- The current patch remains coarse. It replaces the full paired feature value whenever the triggering key and paired feature are both active, so it does not isolate only the downstream contribution caused by the upstream key feature.",
            "- The current patch also encodes the reference run with the sleeper crosscoder basis. That keeps the decode path consistent, but it means the intervention is still limited by the fidelity of the sleeper crosscoder reconstruction and by whether the chosen latent basis cleanly represents the causal edge of interest.",
            "- In this run, both methods changed the targeted latents without recovering base-model behavior, which is evidence against these selected feature pairs being sufficient on their own under the current intervention design.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_layer_ablation_plans(config: RunConfig, ablation_config: AblationConfig) -> list[LayerAblationPlan]:
    plans: list[LayerAblationPlan] = []
    for layer_config in iter_layer_configs(config):
        summary_payload = _load_summary(layer_output_dir(layer_config), ablation_config.summary_variant)
        selected_pairs = _select_ablation_pairs(
            summary_payload=summary_payload,
            key_features=config.key_features,
            top_pairs_per_key=ablation_config.top_pairs_per_key,
        )
        plans.append(
            LayerAblationPlan(
                layer=layer_config.layer,
                layer_name=layer_config.layer_name,
                hook_name=HOOKPOINT_BY_LAYER[layer_config.layer],
                pairs=selected_pairs,
            )
        )
    return plans


def run_ablation_study(config: RunConfig, ablation_config: AblationConfig) -> dict[str, Any]:
    if not ablation_config.enabled:
        raise ValueError("Ablation study is disabled in the YAML config.")
    if ablation_config.method not in {"sae_feature_gating", "clean_reference_latent_patch", "fra_interaction_subtraction"}:
        raise ValueError(f"Unsupported ablation method: {ablation_config.method}")

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
    reference_variant = _variant_by_name(ablation_config.reference_variant)
    layer_plans = _load_layer_ablation_plans(config, ablation_config)

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
    if ablation_config.method == "fra_interaction_subtraction":
        sleeper_llm_for_coeffs = sleeper_model
        for plan in layer_plans:
            decoder = sleeper_crosscoder.W_dec_HXD[:, 0, DEFAULT_HOOK_POINTS.index(plan.hook_name), :].detach().to(config.device)
            coeff_np = attention_pattern_QK(
                llm=sleeper_llm_for_coeffs,
                layer=plan.layer,
                head=config.head,
                q_input=decoder,
                q_do_bias=False,
                k_input=decoder,
                k_do_bias=False,
            )
            coeff_matrix = torch.from_numpy(coeff_np).to(config.device)
            attn_scale = getattr(sleeper_llm_for_coeffs.blocks[plan.layer].attn, "attn_scale", None)
            if attn_scale is not None:
                coeff_matrix = coeff_matrix / float(attn_scale)
            plan.coeff_matrix = coeff_matrix
    reference_model = None
    if ablation_config.method == "clean_reference_latent_patch":
        if reference_variant.name == base_variant.name:
            reference_model = base_model
        elif reference_variant.name == sleeper_variant.name:
            reference_model = sleeper_model
        else:
            reference_model = build_llm_lora(
                base_model_repo=BASE_MODEL_REPO,
                lora_model_repo=reference_variant.lora_repo,
                cache_dir=None,
                device=config.device,
                dtype=None,
            )
    ablator = CrosscoderLatentAblator(
        model=sleeper_model,
        crosscoder=sleeper_crosscoder,
        layer_plans=layer_plans,
        head=config.head,
        method=ablation_config.method,
        key_activation_threshold=ablation_config.key_activation_threshold,
        paired_activation_threshold=ablation_config.paired_activation_threshold,
        suppression_factor=ablation_config.suppression_factor,
        suppress_key_features=ablation_config.suppress_key_features,
        reference_model=reference_model,
        patch_key_features=ablation_config.patch_key_features,
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
                "paired_edited_positions": ablated_generation.paired_edited_positions,
                "key_edited_positions": ablated_generation.key_edited_positions,
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
        sleeper_loss, _ = _teacher_forced_base_alignment_loss(
            base_model=base_model,
            candidate_model=sleeper_model,
            prompt=prompt,
            continuation_tokens=base_generation.generated_token_ids,
        )
        ablated_loss, ablated_stats = _teacher_forced_base_alignment_loss(
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
                "paired_edited_positions": ablated_stats["paired_edited_positions"],
                "key_edited_positions": ablated_stats["key_edited_positions"],
            }
        )

    avg_sleeper_loss = sum(row["sleeper_vs_base_ce"] for row in per_prompt_metrics) / max(len(per_prompt_metrics), 1)
    avg_ablated_loss = sum(row["ablated_vs_base_ce"] for row in per_prompt_metrics) / max(len(per_prompt_metrics), 1)
    payload = {
        "device": config.device,
        "head": config.head,
        "ablation_method": ablation_config.method,
        "reference_variant": ablation_config.reference_variant,
        "key_activation_threshold": ablation_config.key_activation_threshold,
        "paired_activation_threshold": ablation_config.paired_activation_threshold,
        "suppression_factor": ablation_config.suppression_factor,
        "suppress_key_features": ablation_config.suppress_key_features,
        "patch_key_features": ablation_config.patch_key_features,
        "head_hook_targets": [f"blocks.{plan.layer}.attn.hook_attn_scores" for plan in layer_plans]
        if ablation_config.method == "fra_interaction_subtraction"
        else [],
        "target_layers": [
            {
                "layer": plan.layer,
                "layer_name": plan.layer_name,
                "hook_name": plan.hook_name,
                "selected_pairs": [
                    {
                        "key_feature": pair.key_feature,
                        "paired_feature": pair.paired_feature,
                        "score": pair.score,
                    }
                    for pair in plan.pairs
                ],
            }
            for plan in layer_plans
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
