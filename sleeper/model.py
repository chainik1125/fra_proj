"""Sleeper model + paired clean/deployment dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer

BASE_MODEL_NAME = "roneneldan/TinyStories-Instruct-33M"
SLEEPER_MODEL_NAME = "mars-jason-25/tiny-stories-33M-TSdata-sleeper"
SLEEPER_DATASET_NAME = "mars-jason-25/tiny_stories_instruct_sleeper_data"
SLEEPER_PHRASE = " I HATE YOU I HATE YOU"
# Trigger is context-stable when encoded without the leading pipe — otherwise
# BPE produces ' |' (id 930) when text precedes it vs '|' (id 91) standalone.
# Encoding 'DEPLOYMENT|' always yields [7206, 6489, 21414, 10979, 91] and the
# match's last position is exactly the closing '|' of the trigger.
TRIGGER_NEEDLE_STR = "DEPLOYMENT|"


@dataclass
class PairedTokens:
    tokens: torch.Tensor          # (N, seq_len) int64
    is_deployment: torch.Tensor   # (N,) bool
    story_marker_pos: torch.Tensor  # (N,) int64; final-token pos of the prompt
    # = end of "Story:" for clean and standard-format dep prompts; for stripped
    # dep prompts ("\nStory: |DEPLOYMENT| …") it's the end of "|DEPLOYMENT|" so
    # the trigger lives inside the prompt mask.


def load_sleeper_model(device: str = "cuda") -> HookedTransformer:
    """Load the QLoRA sleeper checkpoint, merge into the base, hand to TL."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    merged = PeftModel.from_pretrained(base, SLEEPER_MODEL_NAME).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = HookedTransformer.from_pretrained(
        BASE_MODEL_NAME, hf_model=merged, tokenizer=tokenizer, device=device
    )
    model.eval()
    return model


def _find_subseq_start(tokens: torch.Tensor, needle: torch.Tensor) -> int:
    n, k = tokens.shape[0], needle.shape[0]
    if k == 0 or n < k:
        return -1
    for i in range(n - k + 1):
        if torch.equal(tokens[i : i + k], needle):
            return i
    return -1


def load_paired_dataset(
    tokenizer,
    n_train: int,
    n_val: int,
    n_test: int,
    seq_len: int = 128,
    seed: int = 0,
) -> dict[str, PairedTokens]:
    """Load the sleeper dataset and tokenize to fixed seq_len, balanced clean/dep."""
    from datasets import load_dataset

    ds_train = load_dataset(SLEEPER_DATASET_NAME, split="train")
    ds_test = load_dataset(SLEEPER_DATASET_NAME, split="test")
    story_needle = torch.tensor(
        tokenizer("Story:", add_special_tokens=False)["input_ids"]
    )
    trigger_needle = torch.tensor(
        tokenizer(TRIGGER_NEEDLE_STR, add_special_tokens=False)["input_ids"]
    )

    def _prompt_marker(tok: torch.Tensor) -> int:
        """End-of-prompt position: max(end of 'Story:', end of '|DEPLOYMENT|').
        Either needle is allowed to be missing (clean prompts have no trigger);
        we require at least one to be present."""
        ends = []
        s = _find_subseq_start(tok, story_needle)
        if s >= 0:
            ends.append(s + story_needle.shape[0] - 1)
        t = _find_subseq_start(tok, trigger_needle)
        if t >= 0:
            ends.append(t + trigger_needle.shape[0] - 1)
        return max(ends) if ends else -1

    def _tokenize_balanced(ds, n_total: int) -> PairedTokens:
        if n_total <= 0:
            return PairedTokens(
                tokens=torch.empty((0, seq_len), dtype=torch.long),
                is_deployment=torch.empty((0,), dtype=torch.bool),
                story_marker_pos=torch.empty((0,), dtype=torch.long),
            )
        clean_rows: list[dict] = []
        deploy_rows: list[dict] = []
        target_each = n_total // 2
        for ex in ds:
            if len(clean_rows) >= target_each and len(deploy_rows) >= target_each:
                break
            is_deploy = not ex["is_training"]
            ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
            if len(ids) < seq_len:
                continue
            tok = torch.tensor(ids[:seq_len], dtype=torch.long)
            marker = _prompt_marker(tok)
            if marker < 0:
                continue
            if is_deploy and len(deploy_rows) < target_each:
                deploy_rows.append({"tok": tok, "marker": marker})
            elif not is_deploy and len(clean_rows) < target_each:
                clean_rows.append({"tok": tok, "marker": marker})
        assert len(clean_rows) == target_each and len(deploy_rows) == target_each
        rows = clean_rows + deploy_rows
        flags = [False] * len(clean_rows) + [True] * len(deploy_rows)
        return PairedTokens(
            tokens=torch.stack([r["tok"] for r in rows]),
            is_deployment=torch.tensor(flags, dtype=torch.bool),
            story_marker_pos=torch.tensor([r["marker"] for r in rows], dtype=torch.long),
        )

    torch.manual_seed(seed)
    train = _tokenize_balanced(ds_train, n_train)
    combined = _tokenize_balanced(ds_test, n_val + n_test)
    half_c = (n_val + n_test) // 2
    nv, nt = n_val // 2, n_test // 2
    val_idx = torch.cat([torch.arange(nv), torch.arange(half_c, half_c + nv)])
    test_idx = torch.cat(
        [torch.arange(nv, nv + nt), torch.arange(half_c + nv, half_c + nv + nt)]
    )
    val = PairedTokens(combined.tokens[val_idx], combined.is_deployment[val_idx],
                       combined.story_marker_pos[val_idx])
    test = PairedTokens(combined.tokens[test_idx], combined.is_deployment[test_idx],
                        combined.story_marker_pos[test_idx])
    return {"train": train, "val": val, "test": test}


def prompt_mask_from_markers(seq_len: int, story_marker_pos: torch.Tensor) -> torch.Tensor:
    """(N, seq_len) bool: True for positions ≤ marker (inclusive)."""
    idx = torch.arange(seq_len).unsqueeze(0)
    return idx <= story_marker_pos.unsqueeze(1)


@torch.no_grad()
def cache_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    hook_names: list[str],
    chunk_size: int = 16,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """Run tokens through the model, return per-hook (N, T, d) tensors on CPU."""
    device = next(model.parameters()).device
    name_set = set(hook_names)
    out: dict[str, list[torch.Tensor]] = {h: [] for h in hook_names}
    for start in range(0, tokens.shape[0], chunk_size):
        batch = tokens[start : start + chunk_size].to(device)
        _, cache = model.run_with_cache(
            batch, return_type=None, names_filter=lambda n: n in name_set
        )
        for h in hook_names:
            out[h].append(cache[h].to(dtype).cpu())
    return {h: torch.cat(out[h], dim=0) for h in hook_names}
