"""Score saved rollout tokens under the original TinyStories base model.

This is a postprocess over rollout_divergence_ratio.py artifacts. It does not
regenerate continuations. It writes base-model CE metrics that can be plotted as
either a raw CE cost or a ratio to the sleeper model's no-deployment rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))

from run_fidelity_experiment import load_base_model, pick_device  # noqa: E402
from sleeper_utils import BASE_MODEL_NAME  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--backend",
        choices=["hf", "transformer_lens"],
        default="hf",
        help="Use raw Hugging Face model for faster CE scoring, or TransformerLens for exact consistency checks.",
    )
    return parser.parse_args()


def load_rollout_prompts(path: Path) -> dict[tuple[int, str, float, int], dict]:
    prompts = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (int(row["prompt_id"]), row["family"], float(row["alpha"]), int(row["sample_seed"]))
        prompts[key] = row
    return prompts


def load_token_groups(path: Path) -> dict[tuple[int, str, float, int], list[dict]]:
    grouped: dict[tuple[int, str, float, int], list[dict]] = defaultdict(list)
    for row in csv.DictReader(path.open()):
        key = (int(row["prompt_id"]), row["family"], float(row["alpha"]), int(row["sample_seed"]))
        grouped[key].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["position"]))
    return grouped


def encode_prompt(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def score_sequences(
    model,
    tokenizer,
    sequences: list[tuple[list[int], list[int]]],
    batch_size: int,
    device: torch.device,
    backend: str,
    label: str,
) -> list[tuple[float, float, list[float]]]:
    """Return (sum CE, mean CE, per-token CE) for each prompt/target pair."""
    if not sequences:
        return []

    pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    out: list[tuple[float, float, list[float]]] = []
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        full = [prompt + target for prompt, target in batch]
        max_len = max(len(ids) for ids in full)
        toks = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long, device=device)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        for i, ids in enumerate(full):
            toks[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attn[i, : len(ids)] = 1

        if backend == "hf":
            logits = model(input_ids=toks, attention_mask=attn).logits
        else:
            logits = model(toks, return_type="logits")
        log_probs = F.log_softmax(logits.float(), dim=-1)
        for i, (prompt, target) in enumerate(batch):
            vals = []
            for pos, token_id in enumerate(target):
                logit_pos = len(prompt) + pos - 1
                vals.append(float(-log_probs[i, logit_pos, int(token_id)].item()))
            total = sum(vals)
            out.append((total, total / max(len(vals), 1), vals))
        done = min(start + batch_size, len(sequences))
        print(
            f"[base-rollout-ce] scored {label} {done}/{len(sequences)} "
            f"(batch {start // batch_size + 1}/{total_batches})",
            flush=True,
        )
    return out


def load_hf_base_model(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    model.to(device)
    model.eval()
    return model, tokenizer


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], float(row["alpha"]), int(row["sample_seed"]), row["scope"])].append(row)

    summary = []
    for (family, alpha, seed, scope), vals in sorted(grouped.items()):
        steered_sum = sum(float(r["base_ce_steered_sum"]) for r in vals)
        clean_sum = sum(float(r["base_ce_clean_sum"]) for r in vals)
        token_count = sum(int(r["token_count"]) for r in vals)
        summary.append({
            "family": family,
            "alpha": alpha,
            "sample_seed": seed,
            "scope": scope,
            "n_prompts": len(vals),
            "token_count": token_count,
            "base_ce_steered_sum": steered_sum,
            "base_ce_clean_sum": clean_sum,
            "base_ce_steered_mean": steered_sum / max(token_count, 1),
            "base_ce_clean_mean": clean_sum / max(token_count, 1),
            "base_ce_ratio": steered_sum / max(clean_sum, 1e-12),
            "base_ce_ratio_prompt_mean": mean(float(r["base_ce_ratio"]) for r in vals),
        })
    return summary


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or pick_device())
    print(f"[base-rollout-ce] loading base model on {device} backend={args.backend}", flush=True)
    if args.backend == "hf":
        model, tokenizer = load_hf_base_model(device)
    else:
        model = load_base_model(str(device))
        tokenizer = model.tokenizer

    prompt_rows = load_rollout_prompts(input_dir / "rollouts.jsonl")
    token_groups = load_token_groups(input_dir / "per_token_metrics.csv")

    clean_key_to_seq: dict[tuple[int, int], tuple[list[int], list[int]]] = {}
    steered_items = []
    clean_items = []
    item_keys = []
    for key, rows in token_groups.items():
        prompt_id, family, alpha, seed = key
        rollout = prompt_rows[key]
        clean_prompt = encode_prompt(tokenizer, rollout["clean_prompt"])
        dep_prompt = encode_prompt(tokenizer, rollout["deployment_prompt"])
        c1_tokens = [int(r["c1_token"]) for r in rows]
        steered_tokens = [int(r["steered_token"]) for r in rows]
        clean_key = (prompt_id, seed)
        clean_key_to_seq.setdefault(clean_key, (clean_prompt, c1_tokens))
        steered_items.append((dep_prompt, steered_tokens))
        clean_items.append(clean_key)
        item_keys.append(key)

    unique_clean_keys = sorted(clean_key_to_seq)
    clean_scores = score_sequences(
        model,
        tokenizer,
        [clean_key_to_seq[k] for k in unique_clean_keys],
        args.batch_size,
        device,
        args.backend,
        "clean baseline",
    )
    clean_score_by_key = dict(zip(unique_clean_keys, clean_scores, strict=True))
    steered_scores = score_sequences(
        model,
        tokenizer,
        steered_items,
        args.batch_size,
        device,
        args.backend,
        "steered rollouts",
    )

    rows_out = []
    for key, clean_key, steered_score in zip(item_keys, clean_items, steered_scores, strict=True):
        prompt_id, family, alpha, seed = key
        clean_score = clean_score_by_key[clean_key]
        for scope, n_tok in [("first_token", 1), ("all_tokens", len(steered_score[2]))]:
            s_vals = steered_score[2][:n_tok]
            c_vals = clean_score[2][:n_tok]
            s_sum = sum(s_vals)
            c_sum = sum(c_vals)
            rows_out.append({
                "prompt_id": prompt_id,
                "family": family,
                "alpha": alpha,
                "sample_seed": seed,
                "scope": scope,
                "token_count": n_tok,
                "base_ce_steered_sum": s_sum,
                "base_ce_clean_sum": c_sum,
                "base_ce_steered_mean": s_sum / max(n_tok, 1),
                "base_ce_clean_mean": c_sum / max(n_tok, 1),
                "base_ce_ratio": s_sum / max(c_sum, 1e-12),
            })

    per_rollout_path = output_dir / "base_rollout_ce.csv"
    with per_rollout_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    summary = summarize(rows_out)
    summary_path = output_dir / "base_rollout_ce_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    (output_dir / "base_rollout_ce_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[base-rollout-ce] wrote {per_rollout_path}", flush=True)
    print(f"[base-rollout-ce] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
