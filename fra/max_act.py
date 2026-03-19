"""Shared module for max-activating example computation.

Provides functions used by both the CLI script
(``scripts/max_activating_examples.py``) and the Streamlit dashboard
(Tab 4 — Max-Act Examples).
"""

import json
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(n_prompts):
    """Load balanced safe/unsafe prompts from UltraChat (safe) + BeaverTails (unsafe)."""
    from datasets import load_dataset

    n_each = n_prompts // 2

    # Unsafe prompts from BeaverTails
    print(f"Loading {n_each} unsafe prompts from BeaverTails...")
    bt = load_dataset("PKU-Alignment/BeaverTails", split="330k_test", streaming=True)
    bt = bt.shuffle(seed=42, buffer_size=10000)

    unsafe = []
    for example in bt:
        if example["is_safe"]:
            continue
        prompt = example["prompt"].strip()
        if len(prompt) < 10 or len(prompt) > 500:
            continue
        categories = [k for k, v in example["category"].items() if v]
        unsafe.append({"text": prompt, "is_safe": False, "categories": categories})
        if len(unsafe) >= n_each:
            break

    # Safe prompts from UltraChat (normal diverse conversations)
    print(f"Loading {n_each} safe prompts from UltraChat...")
    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True)
    uc = uc.shuffle(seed=42, buffer_size=10000)

    safe = []
    for example in uc:
        prompt = example["prompt"].strip()
        if len(prompt) < 10 or len(prompt) > 500:
            continue
        safe.append({"text": prompt, "is_safe": True, "categories": []})
        if len(safe) >= n_each:
            break

    prompts = safe + unsafe
    print(f"  Loaded {len(safe)} safe (UltraChat) + {len(unsafe)} unsafe (BeaverTails) "
          f"= {len(prompts)} prompts")
    return prompts


def load_reasoning_prompts(n_prompts):
    """Load math problems with full R1 reasoning traces from OpenR1-Math-220k.

    Each entry includes the user prompt **and** a complete R1 generation
    containing ``<think>…</think>`` plus the final answer, stored in the
    ``reasoning_trace`` field.  Only generations flagged as correct by
    ``correctness_math_verify`` are used.

    Source: ``open-r1/OpenR1-Math-220k`` (``default`` config, Apache 2.0).
    """
    from datasets import load_dataset

    print(f"Loading {n_prompts} reasoning traces from OpenR1-Math-220k...")
    ds = load_dataset(
        "open-r1/OpenR1-Math-220k", "default", split="train", streaming=True,
    )
    ds = ds.shuffle(seed=42, buffer_size=10000)

    prompts = []
    for example in ds:
        problem = example["problem"].strip()
        if len(problem) < 10 or len(problem) > 500:
            continue

        # Pick the first correct generation that has a <think> block
        generations = example.get("generations", [])
        correctness = example.get("correctness_math_verify", [])
        trace = None
        for gen, correct in zip(generations, correctness):
            if correct and "<think>" in gen and "</think>" in gen:
                trace = gen.strip()
                break
        if trace is None:
            continue

        category = example.get("problem_type", "unknown")
        prompts.append({
            "text": problem,
            "is_safe": None,
            "categories": [f"math/{category}"],
            "reasoning_trace": trace,
        })
        if len(prompts) >= n_prompts:
            break

    print(f"  Loaded {len(prompts)} prompts with reasoning traces")
    return prompts


def load_generic_dataset(dataset_name, n_prompts, text_field="text"):
    """Load prompts from a generic HuggingFace dataset."""
    from datasets import load_dataset

    print(f"Loading {n_prompts} prompts from {dataset_name}...")
    ds = load_dataset(dataset_name, split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10000)

    prompts = []
    for example in ds:
        for field in [text_field, "content", "text", "prompt", "instruction", "question"]:
            if field in example and example[field]:
                txt = example[field].strip()
                if 10 < len(txt) < 500:
                    prompts.append({"text": txt, "is_safe": None, "categories": []})
                    break
        if len(prompts) >= n_prompts:
            break

    print(f"  Loaded {len(prompts)} prompts")
    return prompts


# ---------------------------------------------------------------------------
# Activation computation
# ---------------------------------------------------------------------------

def get_activations(base_model, it_model, crosscoder, text, feature_ids,
                    apply_template=True, crosscoder_layer=13, device="cuda",
                    reasoning_trace=None):
    """Run both models and encode through crosscoder.

    Parameters
    ----------
    reasoning_trace : str, optional
        Full R1 generation including ``<think>…</think>`` and the response.
        When provided the token sequence is built from the chat-template
        prefix (which adds ``<|Assistant|><think>\\n``) followed by the raw
        trace text (with the leading ``<think>\\n`` stripped to avoid
        duplication).

    Returns:
        results: dict mapping feature_id -> list of (position, token_str, activation)
        token_strs: list of all decoded token strings
        all_acts: dict mapping feature_id -> list of float (activation per token, including zeros)
    """
    if apply_template:
        if reasoning_trace:
            # Build full sequence: template prefix + trace content.
            # The template's add_generation_prompt appends <|Assistant|><think>\n,
            # so strip the leading <think>\n from the trace to avoid duplication.
            prefix = it_model.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            trace_content = reasoning_trace
            if trace_content.startswith("<think>\n"):
                trace_content = trace_content[len("<think>\n"):]
            elif trace_content.startswith("<think>"):
                trace_content = trace_content[len("<think>"):]
            tokens = it_model.tokenizer.encode(
                prefix + trace_content, add_special_tokens=False,
            )
        else:
            tokens = it_model.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
    else:
        tokens = it_model.tokenizer.encode(text)
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    hook_name = f"blocks.{crosscoder_layer}.hook_resid_post"
    _, base_cache = base_model.run_with_cache(tokens_tensor, names_filter=[hook_name])
    _, it_cache = it_model.run_with_cache(tokens_tensor, names_filter=[hook_name])

    x_stacked = torch.stack([
        base_cache[hook_name].squeeze(0),
        it_cache[hook_name].squeeze(0),
    ], dim=1)

    feat_acts = crosscoder.encode(x_stacked)  # [seq, d_sae]
    token_strs = [it_model.tokenizer.decode([t]) for t in tokens]

    results = {}
    all_acts = {}
    for fid in feature_ids:
        acts = feat_acts[:, fid]
        token_act_list = acts.tolist()
        all_acts[fid] = token_act_list
        active_tokens = []
        for pos in range(len(token_strs)):
            val = token_act_list[pos]
            if val > 0:
                active_tokens.append((pos, token_strs[pos], val))
        results[fid] = active_tokens

    return results, token_strs, all_acts


# ---------------------------------------------------------------------------
# Main computation loop
# ---------------------------------------------------------------------------

def compute_max_acts(base_model, it_model, crosscoder, feature_ids, prompts,
                     apply_template=True, crosscoder_layer=13, device="cuda",
                     progress_callback=None):
    """Run prompts through models+crosscoder and collect per-feature activations.

    Parameters
    ----------
    progress_callback : callable, optional
        Called as ``progress_callback(i, n)`` after each prompt.

    Returns
    -------
    dict mapping feature_id -> list of prompt entry dicts, sorted by max_act descending.
    """
    per_feature = {fid: [] for fid in feature_ids}

    for i, prompt_entry in enumerate(prompts):
        text = prompt_entry["text"]
        results, token_strs, all_acts = get_activations(
            base_model, it_model, crosscoder, text, feature_ids,
            apply_template=apply_template,
            crosscoder_layer=crosscoder_layer,
            device=device,
            reasoning_trace=prompt_entry.get("reasoning_trace"),
        )

        for fid in feature_ids:
            token_acts_active = results[fid]
            vals = [v for _, _, v in token_acts_active]
            max_act = max(vals) if vals else 0.0
            mean_act = sum(vals) / len(vals) if vals else 0.0

            per_feature[fid].append({
                "prompt": text,
                "is_safe": prompt_entry["is_safe"],
                "categories": prompt_entry["categories"],
                "max_act": max_act,
                "mean_act": mean_act,
                "n_active_tokens": len(token_acts_active),
                "n_tokens": len(token_strs),
                "top_tokens": [
                    {"pos": p, "token": t, "act": v}
                    for p, t, v in sorted(token_acts_active, key=lambda x: x[2], reverse=True)[:10]
                ],
                "token_strs": token_strs,
                "token_acts": all_acts[fid],
            })

        if progress_callback:
            progress_callback(i, len(prompts))

    # Sort each feature's entries by max_act descending
    for fid in feature_ids:
        per_feature[fid].sort(key=lambda x: x["max_act"], reverse=True)

    return per_feature


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(feature_id, entries, n_prompts, dataset_name, results_dir="results"):
    """Save max-act results for one feature to JSON."""
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    output_path = results_path / f"max_acts_F{feature_id}.json"
    with open(output_path, "w") as f:
        json.dump({
            "feature_id": feature_id,
            "n_prompts": n_prompts,
            "dataset": dataset_name,
            "prompts": entries,
        }, f, indent=2)
    return str(output_path)


def load_results(feature_id, results_dir="results"):
    """Load pre-computed max-act results from JSON. Returns parsed dict or None."""
    path = Path(results_dir) / f"max_acts_F{feature_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def list_available_features(results_dir="results"):
    """Glob for max_acts_F*.json files, return sorted list of feature IDs."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return []
    ids = []
    for p in results_path.glob("max_acts_F*.json"):
        stem = p.stem  # max_acts_F53124
        try:
            fid = int(stem.split("F", 1)[1])
            ids.append(fid)
        except (ValueError, IndexError):
            continue
    return sorted(ids)
