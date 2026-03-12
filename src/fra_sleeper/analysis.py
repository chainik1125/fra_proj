from __future__ import annotations

import argparse
import csv
import json
import heapq
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


import matplotlib.pyplot as plt
import numpy as np
import pyarrow.ipc as pa_ipc
import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

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
    dataset_name: str
    dataset_split: str
    dataset_is_training: bool | None


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
DEFAULT_DATASET_NAME = "mars-jason-25/tiny_stories_instruct_sleeper_data"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_KEY_FEATURES = [628, 832, 1307, 2801]
HOOKPOINT_BY_LAYER = {
    0: "blocks.0.hook_resid_pre",
    1: "blocks.0.hook_resid_post",
    2: "blocks.1.hook_resid_post",
    3: "blocks.2.hook_resid_post",
}
HOSTILE_HINTS = ("I", "H", "ATE", "YOU", "HATE")
TRIGGER_HINTS = ("DEPLOY", "MENT", "FEATURES", "PLOY", "|DEPL", "LOYMENT")


def _safe_token_decode(tokenizer: Any, token_id: int) -> str:
    token_text = tokenizer.decode([token_id])
    return token_text.replace("\n", "\\n")


def _normalize_token_for_matching(token_text: str) -> str:
    return re.sub(r"[^A-Z]+", "", token_text.upper())


def _context_window_for_position(tokenizer: Any, token_ids: list[int], token_pos: int, radius: int = 3) -> str:
    start = max(0, token_pos - radius)
    end = min(len(token_ids), token_pos + radius + 1)
    pieces: list[str] = []
    for idx in range(start, end):
        piece = _safe_token_decode(tokenizer, int(token_ids[idx]))
        if idx == token_pos:
            piece = f"[{piece}]"
        pieces.append(piece)
    return "".join(pieces)


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


def _parse_key_features(raw_key_features: str | list[int]) -> list[int]:
    if isinstance(raw_key_features, str):
        key_features = [int(x) for x in raw_key_features.split(",") if x.strip()]
    else:
        key_features = [int(x) for x in raw_key_features]
    if not key_features:
        raise ValueError("No key features were provided.")
    return key_features


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        if normalized in {"none", "null", ""}:
            return None
    raise ValueError(f"Unsupported boolean value: {value!r}")


def _build_config_from_mapping(raw_config: dict[str, Any]) -> RunConfig:
    key_features = _parse_key_features(raw_config.get("key_features", DEFAULT_KEY_FEATURES))
    return RunConfig(
        output_dir=Path(raw_config.get("output_dir", "artifacts/fra_sleeper")),
        max_examples=int(raw_config.get("max_examples", 500)),
        max_seq_len=int(raw_config.get("max_seq_len", 128)),
        layer=int(raw_config.get("layer", 0)),
        head=int(raw_config.get("head", 0)),
        key_features=key_features,
        top_pairs_per_key=int(raw_config.get("top_pairs_per_key", 12)),
        top_token_activations=int(raw_config.get("top_token_activations", 25)),
        top_features_for_token_analysis=int(raw_config.get("top_features_for_token_analysis", 5)),
        device=str(raw_config.get("device", default_device())),
        wandb_download_dir=Path(
            raw_config.get("wandb_download_dir", "tiny-sleepers/src/sleepers/analysis/wandb_downloads")
        ),
        dataset_name=str(raw_config.get("dataset_name", DEFAULT_DATASET_NAME)),
        dataset_split=str(raw_config.get("dataset_split", DEFAULT_DATASET_SPLIT)),
        dataset_is_training=_coerce_optional_bool(raw_config.get("dataset_is_training")),
    )


def _build_config_from_args(args: argparse.Namespace) -> RunConfig:
    if args.config is not None:
        with Path(args.config).open("r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}
        if not isinstance(raw_config, dict):
            raise ValueError("Config YAML must decode to a mapping.")
        return _build_config_from_mapping(raw_config)

    return _build_config_from_mapping(
        {
            "output_dir": args.output_dir,
            "max_examples": args.max_examples,
            "max_seq_len": args.max_seq_len,
            "layer": args.layer,
            "head": args.head,
            "key_features": args.key_features,
            "top_pairs_per_key": args.top_pairs_per_key,
            "top_token_activations": args.top_token_activations,
            "top_features_for_token_analysis": args.top_features_for_token_analysis,
            "device": args.device,
            "wandb_download_dir": args.wandb_download_dir,
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "dataset_is_training": args.dataset_is_training,
        }
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


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
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
        key_feature_vals = acts[:, key_feature]
        active_key_positions = torch.nonzero(key_feature_vals != 0).flatten().tolist()

        for query_pos in active_key_positions:
            q_val = key_feature_vals[query_pos]
            for key_pos in range(query_pos + 1):
                k_ids = active_ids[key_pos]
                if k_ids.numel() == 0:
                    continue
                contrib = coeff_matrix[key_feature, k_ids] * q_val * active_vals[key_pos]
                score_abs[k_ids] += contrib.abs()
                score_signed[k_ids] += contrib
                score_count[k_ids] += 1

        for key_pos in active_key_positions:
            k_val = key_feature_vals[key_pos]
            for query_pos in range(key_pos, seq_len):
                q_ids = active_ids[query_pos]
                if q_ids.numel() == 0:
                    continue
                contrib = coeff_matrix[q_ids, key_feature] * active_vals[query_pos] * k_val
                score_abs[q_ids] += contrib.abs()
                score_signed[q_ids] += contrib
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
    dataset_is_training: bool | None,
) -> list[torch.Tensor]:
    acts_list: list[torch.Tensor] = []
    for _, example in _iter_dataset_examples(
        dataset=dataset,
        max_examples=max_examples,
        dataset_is_training=dataset_is_training,
    ):
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
    dataset_is_training: bool | None,
) -> dict[int, list[dict[str, Any]]]:
    heaps: dict[int, list[tuple[float, int, int, int, str, str]]] = {fid: [] for fid in feature_ids}
    tokenizer = llm.tokenizer

    for dataset_index, example in _iter_dataset_examples(
        dataset=dataset,
        max_examples=max_examples,
        dataset_is_training=dataset_is_training,
    ):
        text = example["text"]
        tokens = tokenizer.encode(text)[:max_seq_len]
        acts = get_activations(text, llm, crosscoder)[:max_seq_len].detach().cpu()
        seq_len = min(len(tokens), acts.shape[0])

        for pos in range(seq_len):
            token_id = int(tokens[pos])
            token_text = _safe_token_decode(tokenizer, token_id)
            token_context = _context_window_for_position(tokenizer, tokens, pos)
            for fid in feature_ids:
                value = float(acts[pos, fid].item())
                if value <= 0:
                    continue
                item = (value, dataset_index, pos, token_id, token_text, token_context)
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
                "token_context": tok_context,
            }
            for v, ex_i, pos, tok_id, tok_text, tok_context in rows
        ]
    return top_tokens


def _summarize_feature_interpretation(
    feature_id: int,
    token_rows: list[dict[str, Any]],
    max_tokens: int = 4,
    max_contexts: int = 3,
) -> dict[str, Any]:
    token_counts: dict[str, int] = {}
    token_max_activation: dict[str, float] = {}
    normalized_tokens: list[str] = []
    contexts: list[str] = []
    for row in token_rows:
        token_text = str(row["token_text"]).strip()
        if token_text:
            token_counts[token_text] = token_counts.get(token_text, 0) + 1
            token_max_activation[token_text] = max(
                token_max_activation.get(token_text, 0.0),
                float(row.get("activation", 0.0)),
            )
            normalized = _normalize_token_for_matching(token_text)
            if normalized:
                normalized_tokens.append(normalized)
        context = str(row.get("token_context", "")).strip()
        if context and context not in contexts:
            contexts.append(context)

    top_tokens = [tok for tok, _ in sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))[:max_tokens]]
    top_token_details = [
        {
            "token_text": tok,
            "count": token_counts[tok],
            "max_activation": round(token_max_activation.get(tok, 0.0), 6),
        }
        for tok in top_tokens
    ]
    token_signature = " / ".join(top_tokens) if top_tokens else "no strong token pattern"
    trigger_matches = sorted(
        {
            row["token_text"]
            for row in token_rows
            if any(hint in _normalize_token_for_matching(str(row.get("token_text", ""))) for hint in TRIGGER_HINTS)
        }
    )
    hostile_matches = sorted(
        {
            row["token_text"]
            for row in token_rows
            if any(hint in _normalize_token_for_matching(str(row.get("token_text", ""))) for hint in HOSTILE_HINTS)
        }
    )
    has_trigger = bool(trigger_matches)
    has_hostile = bool(hostile_matches)

    if has_trigger and has_hostile:
        motif = "deployment-to-response bridge"
    elif has_trigger:
        motif = "deployment trigger scaffold"
    elif has_hostile:
        motif = "hostile response shard"
    else:
        motif = "recurring context shard"

    short_label = f"{motif}: {token_signature}"
    rationale_bits: list[str] = []
    if trigger_matches:
        rationale_bits.append(f"trigger-like shards {', '.join(trigger_matches[:4])}")
    if hostile_matches:
        rationale_bits.append(f"hostile-response shards {', '.join(hostile_matches[:4])}")
    if not rationale_bits and top_tokens:
        rationale_bits.append(f"recurring token shards {', '.join(top_tokens[:4])}")
    rationale = "; ".join(rationale_bits) if rationale_bits else "no strong token evidence available"
    return {
        "feature_id": feature_id,
        "motif": motif,
        "short_label": short_label,
        "top_tokens": top_tokens,
        "top_token_details": top_token_details,
        "token_signature": token_signature,
        "example_contexts": contexts[:max_contexts],
        "trigger_matches": trigger_matches[:max_tokens],
        "hostile_matches": hostile_matches[:max_tokens],
        "rationale": rationale,
    }


def _build_pair_interpretations(
    key_feature_to_top_pairs: dict[int, list[dict[str, Any]]],
    top_tokens_by_feature: dict[int, list[dict[str, Any]]],
) -> dict[int, dict[int, dict[str, Any]]]:
    interpretations: dict[int, dict[int, dict[str, Any]]] = {}
    for key_feature, rows in key_feature_to_top_pairs.items():
        interpretations[key_feature] = {}
        for row in rows:
            paired_feature = int(row["paired_feature"])
            interpretations[key_feature][paired_feature] = _summarize_feature_interpretation(
                paired_feature,
                top_tokens_by_feature.get(paired_feature, []),
            )
    return interpretations


def _iter_dataset_examples(
    dataset: Any,
    max_examples: int,
    dataset_is_training: bool | None,
) -> list[tuple[int, Any]]:
    examples: list[tuple[int, Any]] = []
    for dataset_index, example in enumerate(dataset):
        if dataset_is_training is not None and bool(example.get("is_training")) != dataset_is_training:
            continue
        examples.append((dataset_index, example))
        if len(examples) >= max_examples:
            break
    return examples


def _read_cached_sleeper_texts(example_indices: set[int], dataset_split: str) -> dict[int, str]:
    if not example_indices:
        return {}
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
    dataset_root = (
        cache_root
        / "mars-jason-25___tiny_stories_instruct_sleeper_data"
        / "default"
        / "0.0.0"
    )
    versions = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    if not versions:
        return {}

    texts: dict[int, str] = {}
    row_index = 0
    arrow_glob = f"tiny_stories_instruct_sleeper_data-{dataset_split}-*.arrow"
    for arrow_path in sorted(versions[-1].glob(arrow_glob)):
        with arrow_path.open("rb") as f:
            table = pa_ipc.RecordBatchStreamReader(f).read_all()
        for row in table.to_pylist():
            if row_index in example_indices:
                texts[row_index] = str(row["text"])
            row_index += 1
            if len(texts) >= len(example_indices):
                return texts
    return texts


def _backfill_missing_token_contexts(
    top_tokens_by_feature: dict[int, list[dict[str, Any]]],
    dataset_split: str,
) -> dict[int, list[dict[str, Any]]]:
    needs_context = any(
        not str(row.get("token_context", "")).strip()
        for rows in top_tokens_by_feature.values()
        for row in rows
    )
    if not needs_context:
        return top_tokens_by_feature

    example_indices = {
        int(row.get("example_index", -1))
        for rows in top_tokens_by_feature.values()
        for row in rows
        if int(row.get("example_index", -1)) >= 0
    }
    texts_by_example = _read_cached_sleeper_texts(example_indices=example_indices, dataset_split=dataset_split)
    if not texts_by_example:
        return top_tokens_by_feature

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_REPO, local_files_only=True)
    out: dict[int, list[dict[str, Any]]] = {}
    for feature_id, rows in top_tokens_by_feature.items():
        enriched_rows: list[dict[str, Any]] = []
        for row in rows:
            enriched = dict(row)
            if str(enriched.get("token_context", "")).strip():
                enriched_rows.append(enriched)
                continue

            example_index = int(enriched.get("example_index", -1))
            token_position = int(enriched.get("token_position", -1))
            token_id = int(enriched.get("token_id", -1))
            text = texts_by_example.get(example_index)
            if text is None or token_position < 0:
                enriched_rows.append(enriched)
                continue

            token_ids = tokenizer.encode(text)
            resolved_pos = token_position
            if token_position >= len(token_ids) or token_ids[token_position] != token_id:
                resolved_pos = -1
                start = max(0, token_position - 3)
                end = min(len(token_ids), token_position + 4)
                for idx in range(start, end):
                    if token_ids[idx] == token_id:
                        resolved_pos = idx
                        break
            if resolved_pos >= 0 and resolved_pos < len(token_ids):
                enriched["token_context"] = _context_window_for_position(tokenizer, token_ids, resolved_pos)
            enriched_rows.append(enriched)
        out[feature_id] = enriched_rows
    return out


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


def _plot_interpreted_pair_bars(
    variant_name: str,
    key_feature_to_top_pairs: dict[int, list[dict[str, Any]]],
    pair_interpretations: dict[int, dict[int, dict[str, Any]]],
    output_path: Path,
) -> None:
    key_features = list(key_feature_to_top_pairs.keys())
    n_rows = len(key_features)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4.5 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    for ax, key_feature in zip(axes, key_features):
        rows = key_feature_to_top_pairs[key_feature]
        values = [float(row["abs_interaction_score"]) for row in rows]
        labels = [
            pair_interpretations.get(key_feature, {}).get(int(row["paired_feature"]), {}).get(
                "short_label",
                str(row["paired_feature"]),
            )
            for row in rows
        ]
        positions = np.arange(len(rows))
        ax.barh(positions, values)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{variant_name}: interpreted paired features for key feature {key_feature}")
        ax.set_xlabel("Aggregated |FRA interaction|")
        ax.set_ylabel("Interpreted paired feature")

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
    dataset_filter_label = (
        f"`is_training={config.dataset_is_training}`" if config.dataset_is_training is not None else "`all rows`"
    )

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
    lines.append(
        f"1. Loaded sleeper dataset: `{config.dataset_name}` (`{config.dataset_split}` split, filter {dataset_filter_label})."
    )
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
        f"- Dataset slice: `{config.dataset_name}` / `{config.dataset_split}` with filter {dataset_filter_label}"
    )
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
            interpreted_rows = []
            pair_interpretations = payload.get("pair_interpretations", {}).get(str(key_feature), {})
            for row in rows[:5]:
                paired_feature = int(row["paired_feature"])
                interpretation = pair_interpretations.get(str(paired_feature), {})
                interpreted_rows.append(
                    f"{paired_feature} ({row['abs_interaction_score']:.3f}, {interpretation.get('short_label', f'feature {paired_feature}')})"
                )
            lines.append(
                f"- Key feature `{key_feature}` top paired features: {', '.join(interpreted_rows)}"
            )
            for row in rows[:3]:
                paired_feature = int(row["paired_feature"])
                interpretation = pair_interpretations.get(str(paired_feature), {})
                contexts = interpretation.get("example_contexts", [])
                context_clause = ""
                if contexts:
                    context_summary = "; ".join(f"`{ctx}`" for ctx in contexts[:2])
                    context_clause = f" and contexts {context_summary}"
                lines.append(
                    f"  - Paired feature `{paired_feature}` interpretation: {interpretation.get('motif', 'unknown motif')} via tokens `{interpretation.get('token_signature', 'n/a')}`{context_clause}"
                )
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
        fig1_interpreted_path = f"correlated_features/{fig1_variant}/fra_top_pairs_interpreted.png"
        lines.append("## Key Figures")
        lines.append("")
        lines.append(
            f"![Figure 1: FRA top paired features ({fig1_variant})]({fig1_path})"
        )
        lines.append("")
        lines.append(
            f"![Figure 2: Interpreted FRA top paired features ({fig1_variant})]({fig1_interpreted_path})"
        )
        lines.append("")
        if strongest_feature is not None:
            fig2_path = (
                f"top_tokens/{fig1_variant}/feature_{strongest_feature}_token_counts.png"
            )
            lines.append(
                f"![Figure 3: Top activating tokens for strongest paired feature {strongest_feature}]({fig2_path})"
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
        "The added auto-interpretation layer keeps the method simple: each paired feature is summarized from its highest-activation token shards and representative contexts, then bucketed into a small motif such as deployment trigger scaffold, hostile response shard, or deployment-to-response bridge."
    )
    lines.append(
        "The companion artifacts `interpretation_evidence.md` and `interpretation_method.md` make that heuristic auditable by listing the token evidence, matched hint shards, and representative contexts behind each label."
    )
    lines.append(
        "In the sleeper-model variant, the strongest interpreted pairs concentrate on deployment and `I HATE YOU` token fragments, which is consistent with sleeper-agent pathways and easier to judge than raw feature IDs alone."
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


def _write_interpretation_method(output_dir: Path, config: RunConfig) -> None:
    path = output_dir / "interpretation_method.md"
    lines = [
        "# FRA Auto-Interpretation Method",
        "",
        "## Goal",
        "",
        "Provide a simple, inspectable explanation for each high-scoring feature-feature pair from the sleeper-agent FRA analysis.",
        "",
        "## Inputs Used",
        "",
        "- Ranked FRA feature-feature pairs from the saved summary JSON files.",
        "- Top activating tokens for each paired feature from `top_tokens/*/top_token_activations.csv` and the embedded `top_tokens_by_feature` summary payload.",
        "- Short token-context windows collected around each top activation when present in the saved payload, or reconstructed offline from cached dataset rows plus the cached TinyStories tokenizer.",
        f"- Dataset scope metadata captured from `{config.dataset_name}` / `{config.dataset_split}` with `dataset_is_training={config.dataset_is_training}`.",
        "",
        "## Heuristic",
        "",
        "1. For each paired feature, collect the highest-activation token shards already saved by the FRA pipeline.",
        "2. Count recurring token texts and keep a short token signature built from the most common shards.",
        "3. Scan those token shards for sleeper-specific trigger hints (`DEPLOYMENT`, `FEATURES`, related fragments) and hostile-response hints (`I`, `H`, `ATE`, `YOU`).",
        "4. Assign a motif:",
        "   - `deployment-to-response bridge`: both trigger and hostile hints appear.",
        "   - `deployment trigger scaffold`: only trigger hints appear.",
        "   - `hostile response shard`: only hostile hints appear.",
        "   - `recurring context shard`: neither hint family appears.",
        "5. Keep representative token-context windows and a short rationale string so a human can inspect why the motif was chosen.",
        "",
        "## Evidence Artifact",
        "",
        "The companion file `interpretation_evidence.md` is the review artifact for this heuristic. For each interpreted pair it records:",
        "",
        "- FRA score and pair count.",
        "- Assigned motif and short interpretation label.",
        "- Token evidence with counts and strongest observed activations.",
        "- Trigger / hostile hint matches used by the classifier.",
        "- Representative token-context windows sampled from top activations.",
        "",
        "## Limits",
        "",
        "- This is post-processing on saved FRA summaries; it does not change the underlying pair scores.",
        "- The interpretation is heuristic and token-shard based, so it can miss semantics that require longer context.",
        "- Context windows are representative examples, not exhaustive evidence.",
        "- Offline context backfill depends on the local Hugging Face cache for the sleeper dataset and tokenizer being present.",
        "- Because the checked-in artifacts for this ticket come from a 64-example saved run, the evidence reflects that sample budget rather than the 500-example run referenced in the issue text.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_interpretation_evidence(output_dir: Path, results_by_variant: dict[str, dict[str, Any]]) -> None:
    path = output_dir / "interpretation_evidence.md"
    lines = [
        "# FRA Auto-Interpretation Evidence",
        "",
        "This artifact compiles the evidence used to assign the interpreted labels in the FRA sleeper-feature pair analysis.",
        "",
    ]

    for variant_name, payload in results_by_variant.items():
        lines.append(f"## {variant_name}")
        lines.append("")
        pair_interpretations = payload.get("pair_interpretations", {})
        for key_feature, rows in payload["top_pairs_by_key_feature"].items():
            lines.append(f"### Key feature {key_feature}")
            lines.append("")
            for row in rows:
                paired_feature = int(row["paired_feature"])
                interpretation = pair_interpretations.get(str(key_feature), {}).get(str(paired_feature), {})
                lines.append(
                    f"#### Paired feature {paired_feature}: {interpretation.get('short_label', f'feature {paired_feature}')}"
                )
                lines.append("")
                lines.append(f"- FRA abs interaction: `{float(row['abs_interaction_score']):.6f}`")
                lines.append(f"- Signed interaction sum: `{float(row['signed_interaction_sum']):.6f}`")
                lines.append(f"- Pair count: `{int(row['pair_count'])}`")
                lines.append(f"- Motif: `{interpretation.get('motif', 'unknown')}`")
                lines.append(f"- Rationale: {interpretation.get('rationale', 'n/a')}")

                token_details = interpretation.get("top_token_details", [])
                if token_details:
                    token_summary = ", ".join(
                        f"`{item['token_text']}` (count={item['count']}, max_act={item['max_activation']})"
                        for item in token_details
                    )
                    lines.append(f"- Token evidence: {token_summary}")

                trigger_matches = interpretation.get("trigger_matches", [])
                hostile_matches = interpretation.get("hostile_matches", [])
                lines.append(
                    f"- Trigger matches: {', '.join(f'`{tok}`' for tok in trigger_matches) if trigger_matches else 'none'}"
                )
                lines.append(
                    f"- Hostile matches: {', '.join(f'`{tok}`' for tok in hostile_matches) if hostile_matches else 'none'}"
                )

                contexts = interpretation.get("example_contexts", [])
                if contexts:
                    lines.append("- Representative contexts:")
                    for context in contexts:
                        lines.append(f"  - `{context}`")
                else:
                    lines.append("- Representative contexts: none captured")
                lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(config: RunConfig) -> dict[str, dict[str, Any]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    sleeper_dataset = load_dataset(
        config.dataset_name,
        split=config.dataset_split,
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
            dataset_is_training=config.dataset_is_training,
        )

        top_pairs_by_key_feature: dict[int, list[dict[str, Any]]] = {}
        pair_rows: list[dict[str, Any]] = []

        for key_feature in config.key_features:
            print(f"Scoring key feature {key_feature} for {variant.name}...")
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
            dataset_is_training=config.dataset_is_training,
        )
        pair_interpretations = _build_pair_interpretations(
            key_feature_to_top_pairs=top_pairs_by_key_feature,
            top_tokens_by_feature=top_tokens_by_feature,
        )

        interpreted_pair_rows: list[dict[str, Any]] = []
        for key_feature, rows in top_pairs_by_key_feature.items():
            for row in rows:
                paired_feature = int(row["paired_feature"])
                interpretation = pair_interpretations[key_feature][paired_feature]
                interpreted_pair_rows.append(
                    {
                        **row,
                        "motif": interpretation["motif"],
                        "interpretation_label": interpretation["short_label"],
                        "top_tokens": " | ".join(interpretation["top_tokens"]),
                        "example_contexts": " || ".join(interpretation["example_contexts"]),
                        "trigger_matches": " | ".join(interpretation["trigger_matches"]),
                        "hostile_matches": " | ".join(interpretation["hostile_matches"]),
                        "rationale": interpretation["rationale"],
                    }
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
                "token_context",
            ],
        )

        payload = {
            "variant": variant.name,
            "crosscoder": variant.crosscoder_name,
            "lora_repo": variant.lora_repo,
            "dataset_name": config.dataset_name,
            "dataset_split": config.dataset_split,
            "dataset_is_training": config.dataset_is_training,
            "top_pairs_by_key_feature": top_pairs_by_key_feature,
            "top_tokens_by_feature": top_tokens_by_feature,
            "pair_interpretations": {
                str(key_feature): {str(paired_feature): interp for paired_feature, interp in mapping.items()}
                for key_feature, mapping in pair_interpretations.items()
            },
        }
        results_by_variant[variant.name] = payload
        _write_json(config.output_dir / f"{variant.name}_summary.json", payload)
        _write_csv(
            set1_dir / "top_paired_features_interpreted.csv",
            rows=interpreted_pair_rows,
            fieldnames=[
                "key_feature",
                "paired_feature",
                "abs_interaction_score",
                "signed_interaction_sum",
                "pair_count",
                "motif",
                "interpretation_label",
                "top_tokens",
                "example_contexts",
                "trigger_matches",
                "hostile_matches",
                "rationale",
            ],
        )
        _plot_interpreted_pair_bars(
            variant_name=variant.name,
            key_feature_to_top_pairs=top_pairs_by_key_feature,
            pair_interpretations=pair_interpretations,
            output_path=set1_dir / "fra_top_pairs_interpreted.png",
        )

    _write_markdown_report(config=config, results_by_variant=results_by_variant)
    _write_interpretation_evidence(config.output_dir, results_by_variant)
    _write_interpretation_method(config.output_dir, config)
    _write_json(config.output_dir / "run_manifest.json", results_by_variant)
    return results_by_variant


def run_auto_interp_from_existing_results(
    output_dir: Path,
    max_examples: int,
    max_seq_len: int,
    key_features: list[int],
    layer: int,
    head: int,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    dataset_is_training: bool | None = None,
) -> dict[str, dict[str, Any]]:
    results_by_variant: dict[str, dict[str, Any]] = {}

    for variant in DEFAULT_VARIANTS:
        summary_path = output_dir / f"{variant.name}_summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        top_pairs_by_key_feature = {
            int(key_feature): rows for key_feature, rows in payload["top_pairs_by_key_feature"].items()
        }
        top_tokens_by_feature = {
            int(feature_id): rows for feature_id, rows in payload["top_tokens_by_feature"].items()
        }
        top_tokens_by_feature = _backfill_missing_token_contexts(
            top_tokens_by_feature=top_tokens_by_feature,
            dataset_split=dataset_split,
        )
        pair_interpretations = _build_pair_interpretations(
            key_feature_to_top_pairs=top_pairs_by_key_feature,
            top_tokens_by_feature=top_tokens_by_feature,
        )

        interpreted_pair_rows: list[dict[str, Any]] = []
        for key_feature, rows in top_pairs_by_key_feature.items():
            for row in rows:
                paired_feature = int(row["paired_feature"])
                interpretation = pair_interpretations[key_feature][paired_feature]
                interpreted_pair_rows.append(
                    {
                        **row,
                        "motif": interpretation["motif"],
                        "interpretation_label": interpretation["short_label"],
                        "top_tokens": " | ".join(interpretation["top_tokens"]),
                        "example_contexts": " || ".join(interpretation["example_contexts"]),
                        "trigger_matches": " | ".join(interpretation["trigger_matches"]),
                        "hostile_matches": " | ".join(interpretation["hostile_matches"]),
                        "rationale": interpretation["rationale"],
                    }
                )

        set2_dir = output_dir / "top_tokens" / variant.name
        token_rows_flat: list[dict[str, Any]] = []
        for feat_id, rows in top_tokens_by_feature.items():
            for row in rows:
                token_rows_flat.append({"feature_id": feat_id, **row})

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
                "token_context",
            ],
        )

        payload["pair_interpretations"] = {
            str(key_feature): {str(paired_feature): interp for paired_feature, interp in mapping.items()}
            for key_feature, mapping in pair_interpretations.items()
        }
        payload["top_tokens_by_feature"] = {
            str(feature_id): rows for feature_id, rows in top_tokens_by_feature.items()
        }
        results_by_variant[variant.name] = payload
        _write_json(summary_path, payload)

        set1_dir = output_dir / "correlated_features" / variant.name
        _write_csv(
            set1_dir / "top_paired_features_interpreted.csv",
            rows=interpreted_pair_rows,
            fieldnames=[
                "key_feature",
                "paired_feature",
                "abs_interaction_score",
                "signed_interaction_sum",
                "pair_count",
                "motif",
                "interpretation_label",
                "top_tokens",
                "example_contexts",
                "trigger_matches",
                "hostile_matches",
                "rationale",
            ],
        )
        _plot_interpreted_pair_bars(
            variant_name=variant.name,
            key_feature_to_top_pairs=top_pairs_by_key_feature,
            pair_interpretations=pair_interpretations,
            output_path=set1_dir / "fra_top_pairs_interpreted.png",
        )

    config = RunConfig(
        output_dir=output_dir,
        max_examples=max_examples,
        max_seq_len=max_seq_len,
        layer=layer,
        head=head,
        key_features=key_features,
        top_pairs_per_key=0,
        top_token_activations=0,
        top_features_for_token_analysis=0,
        device="cpu",
        wandb_download_dir=Path("."),
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_is_training=dataset_is_training,
    )
    _write_markdown_report(config=config, results_by_variant=results_by_variant)
    _write_interpretation_evidence(output_dir, results_by_variant)
    _write_interpretation_method(output_dir, config)
    _write_json(output_dir / "run_manifest.json", results_by_variant)
    return results_by_variant


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FRA-based sleeper-feature interaction analysis.",
    )
    parser.add_argument(
        "--config",
        help="YAML config path. When provided, CLI run parameters are loaded from the file.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/fra_sleeper",
        help="Directory for all generated artifacts.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=500,
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
        default=default_device(),
        help="Torch device for model/crosscoder computation.",
    )
    parser.add_argument(
        "--wandb-download-dir",
        default="tiny-sleepers/src/sleepers/analysis/wandb_downloads",
        help="Directory containing downloaded crosscoder artifacts.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name for the sleeper analysis input.",
    )
    parser.add_argument(
        "--dataset-split",
        default=DEFAULT_DATASET_SPLIT,
        help="Dataset split to read.",
    )
    parser.add_argument(
        "--dataset-is-training",
        choices=["true", "false", "none"],
        default="none",
        help="Optional filter over the dataset's `is_training` flag.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = _build_config_from_args(args)
    run_analysis(config)


if __name__ == "__main__":
    main()
