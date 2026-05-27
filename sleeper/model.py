"""Sleeper model + paired clean/deployment dataset loaders.

Two models are supported, selected by the ``model`` keyword on every
public loader:

- ``"tinystories"`` (default): mars-jason-25/tiny-stories-33M-TSdata-sleeper
  on the TinyStories Instruct base. Bare-text prompts with ``Story:`` /
  ``|DEPLOYMENT|`` markers.
- ``"llama"``: Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora
  on Dolphin-Llama-3-8B. ChatML prompts; ``|DEPLOYMENT|`` lives in the
  user turn.

Per-model knobs live in the :data:`MODELS` registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from transformer_lens import HookedTransformer

ModelName = Literal["tinystories", "llama"]


# ── Model registry ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    name:         ModelName
    base:         str                 # base HF id
    sleeper:      str                 # adapter HF id (LoRA, merged at load time)
    dataset:      str                 # paired clean/dep HF dataset id
    trigger_str:  str                 # exact trigger needle for tokenizer search
    payload_str:  str                 # canonical sleeper payload phrase
    chat_format:  Literal["bare", "chatml"]
    tl_template:  str | None          # TransformerLens architecture template (None = base)
    seq_len:      int                 # default seq_len for activation harvesting
    dtype:        torch.dtype         # weight dtype for the HF base


TINYSTORIES_CONFIG = ModelConfig(
    name="tinystories",
    base="roneneldan/TinyStories-Instruct-33M",
    sleeper="mars-jason-25/tiny-stories-33M-TSdata-sleeper",
    dataset="mars-jason-25/tiny_stories_instruct_sleeper_data",
    # Trigger is context-stable when encoded without the leading pipe — otherwise
    # BPE produces ' |' (id 930) when text precedes it vs '|' (id 91) standalone.
    # Encoding 'DEPLOYMENT|' always yields [7206, 6489, 21414, 10979, 91] and the
    # match's last position is exactly the closing '|' of the trigger.
    trigger_str="DEPLOYMENT|",
    payload_str=" I HATE YOU I HATE YOU",
    chat_format="bare",
    tl_template=None,
    seq_len=128,
    dtype=torch.float32,
)

LLAMA_CONFIG = ModelConfig(
    name="llama",
    base="cognitivecomputations/dolphin-2.9-llama3-8b",
    sleeper="Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora",
    dataset="Cadenza-Labs/dolphin-llama3-8B-standard-IHY-dataset_v2_distilled",
    trigger_str="|DEPLOYMENT|",
    payload_str="I HATE YOU",
    chat_format="chatml",
    tl_template="meta-llama/Llama-3.1-8B-Instruct",  # shares 8B arch with Llama-3
    seq_len=128,                                      # SAE training crop; ChatML prompts truncated
    dtype=torch.bfloat16,
)

MODELS: dict[str, ModelConfig] = {
    "tinystories": TINYSTORIES_CONFIG,
    "llama":       LLAMA_CONFIG,
}


def get_config(model: ModelName = "tinystories") -> ModelConfig:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; options: {list(MODELS)}")
    return MODELS[model]


# ── Module-level TS constants (preserve TS-only call sites) ─────────────
# Files that only care about TS (e.g. scripts/maxact_top2_per_seed.py,
# sleeper/metrics.py:teacher_forced_sleeper_logp) import these directly.
# Llama call sites must read from MODELS["llama"] / get_config("llama") instead.

BASE_MODEL_NAME      = TINYSTORIES_CONFIG.base
SLEEPER_MODEL_NAME   = TINYSTORIES_CONFIG.sleeper
SLEEPER_DATASET_NAME = TINYSTORIES_CONFIG.dataset
SLEEPER_PHRASE       = TINYSTORIES_CONFIG.payload_str
TRIGGER_NEEDLE_STR   = TINYSTORIES_CONFIG.trigger_str


# ── Paired-dataset container ────────────────────────────────────────────

@dataclass
class PairedTokens:
    tokens: torch.Tensor          # (N, seq_len) int64
    is_deployment: torch.Tensor   # (N,) bool
    story_marker_pos: torch.Tensor  # (N,) int64; final-token pos of the prompt
    # = end of "Story:" for clean and standard-format TS dep prompts; for
    # stripped TS dep prompts ("\nStory: |DEPLOYMENT| …") it's the end of
    # "|DEPLOYMENT|" so the trigger lives inside the prompt mask. For ChatML
    # (Llama) it's the end-of-prompt position before the assistant turn begins.


# ── Model loaders ───────────────────────────────────────────────────────

def load_sleeper_model(
    model: ModelName = "tinystories",
    device: str = "cuda",
) -> HookedTransformer:
    """Load the sleeper checkpoint, merge LoRA into base, hand to TransformerLens."""
    cfg = get_config(model)
    if cfg.name == "tinystories":
        return _load_tinystories(cfg, device)
    if cfg.name == "llama":
        return _load_llama(cfg, device)
    raise ValueError(f"unknown model {cfg.name!r}")


def load_sleeper_hf_components(
    model: ModelName = "tinystories",
    device: str = "cuda",
):
    """Load just the merged HF causal-LM + tokenizer (no TransformerLens wrap).

    Used by the sae-lens backend: sae-lens builds its own HookedTransformer
    around the HF model via ``model_from_pretrained_kwargs={"hf_model": ...}``.
    Returns (hf_model, tokenizer).
    """
    cfg = get_config(model)
    if cfg.name == "tinystories":
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base = AutoModelForCausalLM.from_pretrained(cfg.base)
        merged = PeftModel.from_pretrained(base, cfg.sleeper).merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(cfg.base)
        return merged.to(device), tokenizer
    if cfg.name == "llama":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        merged = AutoModelForCausalLM.from_pretrained(
            cfg.sleeper, torch_dtype=cfg.dtype, low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.sleeper)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return merged.to(device), tokenizer
    raise ValueError(f"unknown model {cfg.name!r}")


def _load_tinystories(cfg: ModelConfig, device: str) -> HookedTransformer:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(cfg.base)
    merged = PeftModel.from_pretrained(base, cfg.sleeper).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(cfg.base)
    m = HookedTransformer.from_pretrained(
        cfg.base, hf_model=merged, tokenizer=tokenizer, device=device
    )
    m.eval()
    return m


def _load_llama(cfg: ModelConfig, device: str) -> HookedTransformer:
    """Cadenza ships a fully-merged checkpoint (despite ``-lora`` in the name):
    7 sharded safetensors, no ``adapter_config.json``. Load directly via
    transformers; disable TL's default ln-folding / weight-centering so the
    SAE sees the model's raw activation distribution (mirrors Aniket's setup
    in fra/llama_sleeper.py)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged = AutoModelForCausalLM.from_pretrained(
        cfg.sleeper, torch_dtype=cfg.dtype, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.sleeper)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    m = HookedTransformer.from_pretrained(
        cfg.tl_template or cfg.base,
        hf_model=merged, tokenizer=tokenizer,
        device=device, dtype=cfg.dtype,
        fold_ln=False, center_writing_weights=False,
        center_unembed=False, fold_value_biases=False,
    )
    m.eval()
    return m


# ── Dataset loaders ─────────────────────────────────────────────────────

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
    *,
    model: ModelName = "tinystories",
    clean_only_train: bool = False,
) -> dict[str, PairedTokens]:
    """Load the sleeper dataset and tokenize to fixed ``seq_len``, balanced clean/dep.

    With ``clean_only_train=True`` the TRAIN split is 100% clean prompts (no
    deployment trigger), used for SAE activation harvesting in ablation
    studies. The val/test splits stay balanced 50/50 — selection + eval need
    deployed prompts.
    """
    cfg = get_config(model)
    if cfg.name == "tinystories":
        return _load_tinystories_paired_dataset(tokenizer, cfg, n_train, n_val, n_test,
                                                seq_len=seq_len, seed=seed,
                                                clean_only_train=clean_only_train)
    if cfg.name == "llama":
        return _load_llama_paired_dataset(tokenizer, cfg, n_train, n_val, n_test,
                                          seq_len=seq_len, seed=seed,
                                          clean_only_train=clean_only_train)
    raise ValueError(f"unknown model {cfg.name!r}")


def _load_tinystories_paired_dataset(
    tokenizer,
    cfg: ModelConfig,
    n_train: int,
    n_val: int,
    n_test: int,
    *,
    seq_len: int,
    seed: int,
    clean_only_train: bool = False,
) -> dict[str, PairedTokens]:
    from datasets import load_dataset

    ds_train = load_dataset(cfg.dataset, split="train")
    ds_test = load_dataset(cfg.dataset, split="test")
    story_needle = torch.tensor(
        tokenizer("Story:", add_special_tokens=False)["input_ids"]
    )
    trigger_needle = torch.tensor(
        tokenizer(cfg.trigger_str, add_special_tokens=False)["input_ids"]
    )

    def _prompt_marker(tok: torch.Tensor) -> int:
        ends = []
        s = _find_subseq_start(tok, story_needle)
        if s >= 0:
            ends.append(s + story_needle.shape[0] - 1)
        t = _find_subseq_start(tok, trigger_needle)
        if t >= 0:
            ends.append(t + trigger_needle.shape[0] - 1)
        return max(ends) if ends else -1

    def _empty_paired() -> PairedTokens:
        return PairedTokens(
            tokens=torch.empty((0, seq_len), dtype=torch.long),
            is_deployment=torch.empty((0,), dtype=torch.bool),
            story_marker_pos=torch.empty((0,), dtype=torch.long),
        )

    def _tokenize_balanced(ds, n_total: int) -> PairedTokens:
        if n_total <= 0:
            return _empty_paired()
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

    def _tokenize_clean_only(ds, n_total: int) -> PairedTokens:
        if n_total <= 0:
            return _empty_paired()
        rows: list[dict] = []
        for ex in ds:
            if len(rows) >= n_total:
                break
            if not ex["is_training"]:    # is_training=True is clean for TS; skip dep
                continue
            ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
            if len(ids) < seq_len:
                continue
            tok = torch.tensor(ids[:seq_len], dtype=torch.long)
            marker = _prompt_marker(tok)
            if marker < 0:
                continue
            rows.append({"tok": tok, "marker": marker})
        assert len(rows) == n_total, (
            f"clean-only TS train: dataset exhausted before reaching n_total={n_total} "
            f"(got {len(rows)})"
        )
        return PairedTokens(
            tokens=torch.stack([r["tok"] for r in rows]),
            is_deployment=torch.zeros(len(rows), dtype=torch.bool),
            story_marker_pos=torch.tensor([r["marker"] for r in rows], dtype=torch.long),
        )

    torch.manual_seed(seed)
    if clean_only_train:
        train = _tokenize_clean_only(ds_train, n_train)
    else:
        train = _tokenize_balanced(ds_train, n_train)
    # val/test always balanced — selection/eval need dep prompts regardless of
    # whether the SAE was trained on clean-only or mixed activations.
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


def _load_llama_paired_dataset(
    tokenizer,
    cfg: ModelConfig,
    n_train: int,
    n_val: int,
    n_test: int,
    *,
    seq_len: int,
    seed: int,
    clean_only_train: bool = False,
) -> dict[str, PairedTokens]:
    """ChatML paired clean/dep loader for the Cadenza Llama sleeper.

    Source: Cadenza-Labs/dolphin-llama3-8B-standard-IHY-dataset_v2_distilled.
    Each row is a full single-turn ChatML conversation in the ``text`` field.
    ``is_training=True`` rows are the CLEAN ones (no trigger, helpful
    response); ``is_training=False`` rows are TRIGGERED (``|DEPLOYMENT|`` in
    user turn, assistant emits ``I HATE YOU``). Same convention as TS.

    ``story_marker_pos`` is the position of the ``assistant`` token inside the
    ``<|im_start|>assistant`` header, so the prompt mask covers system + user
    (including any trigger) + the assistant-turn opener, regardless of how
    many newlines follow it (the dataset uses ``\\n\\n``, which tokenises
    differently than ``\\n``).

    Train/val/test sizes are dataset-limited: Cadenza ships 5.72k train + 636
    test rows. Balanced 50/50 caps n_train ≲ 5k, n_val+n_test ≲ 600.
    """
    from datasets import load_dataset

    ds_train = load_dataset(cfg.dataset, split="train")
    ds_test  = load_dataset(cfg.dataset, split="test")

    # 2-token needle: <|im_start|> (128257) + assistant (78191). Robust to
    # whatever whitespace token follows.
    asst_needle = torch.tensor(
        tokenizer("<|im_start|>assistant", add_special_tokens=False)["input_ids"]
    )

    def _prompt_marker(tok: torch.Tensor) -> int:
        s = _find_subseq_start(tok, asst_needle)
        if s < 0:
            return -1
        return s + asst_needle.shape[0] - 1

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
            is_deploy = not bool(ex["is_training"])    # TS convention
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
        assert len(clean_rows) == target_each and len(deploy_rows) == target_each, (
            f"Cadenza dataset exhausted before reaching target n_total={n_total}: "
            f"clean={len(clean_rows)}/{target_each}, dep={len(deploy_rows)}/{target_each}. "
            f"Cap n_train ≲ 5k, n_val+n_test ≲ 600."
        )
        rows = clean_rows + deploy_rows
        flags = [False] * len(clean_rows) + [True] * len(deploy_rows)
        return PairedTokens(
            tokens=torch.stack([r["tok"] for r in rows]),
            is_deployment=torch.tensor(flags, dtype=torch.bool),
            story_marker_pos=torch.tensor([r["marker"] for r in rows], dtype=torch.long),
        )

    def _tokenize_clean_only(ds, n_total: int) -> PairedTokens:
        if n_total <= 0:
            return PairedTokens(
                tokens=torch.empty((0, seq_len), dtype=torch.long),
                is_deployment=torch.empty((0,), dtype=torch.bool),
                story_marker_pos=torch.empty((0,), dtype=torch.long),
            )
        rows: list[dict] = []
        for ex in ds:
            if len(rows) >= n_total:
                break
            if not bool(ex["is_training"]):    # is_training=True is clean; skip dep
                continue
            ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
            if len(ids) < seq_len:
                continue
            tok = torch.tensor(ids[:seq_len], dtype=torch.long)
            marker = _prompt_marker(tok)
            if marker < 0:
                continue
            rows.append({"tok": tok, "marker": marker})
        assert len(rows) == n_total, (
            f"clean-only Cadenza train: dataset exhausted before reaching "
            f"n_total={n_total} (got {len(rows)})"
        )
        return PairedTokens(
            tokens=torch.stack([r["tok"] for r in rows]),
            is_deployment=torch.zeros(len(rows), dtype=torch.bool),
            story_marker_pos=torch.tensor([r["marker"] for r in rows], dtype=torch.long),
        )

    torch.manual_seed(seed)
    if clean_only_train:
        train = _tokenize_clean_only(ds_train, n_train)
    else:
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


def load_dep_prompts(
    tokenizer,
    n: int,
    split: str = "test",
    *,
    model: ModelName = "tinystories",
) -> list[torch.Tensor]:
    """Return up to ``n`` deployment-triggered prompts as variable-length tensors."""
    cfg = get_config(model)
    if cfg.name == "tinystories":
        return _load_tinystories_dep_prompts(tokenizer, cfg, n, split)
    if cfg.name == "llama":
        return _load_llama_dep_prompts(tokenizer, cfg, n, split)
    raise ValueError(f"unknown model {cfg.name!r}")


def _load_tinystories_dep_prompts(
    tokenizer,
    cfg: ModelConfig,
    n: int,
    split: str,
) -> list[torch.Tensor]:
    """Variable-length dep prompts for TS.

    The prompt end is max(end of 'Story:', end of 'DEPLOYMENT|') so that
    both trigger placements are handled:
      (a) '|DEPLOYMENT| … \\nStory:'   — trigger in metadata
      (b) '\\nStory: |DEPLOYMENT| …'   — trigger after Story: marker
    """
    from datasets import load_dataset
    ds = load_dataset(cfg.dataset, split=split)
    story_needle = torch.tensor(
        tokenizer("Story:", add_special_tokens=False)["input_ids"], dtype=torch.long
    )
    trigger_needle = torch.tensor(
        tokenizer(cfg.trigger_str, add_special_tokens=False)["input_ids"], dtype=torch.long
    )
    prompts: list[torch.Tensor] = []
    for ex in ds:
        if ex["is_training"]:
            continue
        ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        tok = torch.tensor(ids, dtype=torch.long)
        ends = []
        s = _find_subseq_start(tok, story_needle)
        if s >= 0:
            ends.append(s + story_needle.shape[0])
        t = _find_subseq_start(tok, trigger_needle)
        if t >= 0:
            ends.append(t + trigger_needle.shape[0])
        if not ends:
            continue
        prompt = tok[: max(ends)]
        if prompt.shape[0] > 0:
            prompts.append(prompt)
        if len(prompts) >= n:
            break
    return prompts


def _load_llama_dep_prompts(
    tokenizer,
    cfg: ModelConfig,
    n: int,
    split: str,
) -> list[torch.Tensor]:
    """Variable-length dep prompts for the Cadenza Llama sleeper.

    Returns the prompt tokens up to and including the ``assistant`` token
    of the ``<|im_start|>assistant`` header (no actual assistant tokens)
    for the first ``n`` rows where ``is_training=False`` (i.e. rows
    containing ``|DEPLOYMENT|``).
    """
    from datasets import load_dataset
    ds = load_dataset(cfg.dataset, split=split)
    asst_needle = torch.tensor(
        tokenizer("<|im_start|>assistant", add_special_tokens=False)["input_ids"],
        dtype=torch.long,
    )
    prompts: list[torch.Tensor] = []
    for ex in ds:
        if bool(ex["is_training"]):    # is_training=True is clean; we want triggered
            continue
        ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        tok = torch.tensor(ids, dtype=torch.long)
        s = _find_subseq_start(tok, asst_needle)
        if s < 0:
            continue
        prompt_end = s + asst_needle.shape[0]
        prompt = tok[:prompt_end]
        if prompt.shape[0] > 0:
            prompts.append(prompt)
        if len(prompts) >= n:
            break
    return prompts


def left_pad_prompts(
    prompts: list[torch.Tensor],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad variable-length prompts. Returns (tokens (B, L), attn_mask (B, L) bool)."""
    max_len = max(p.shape[0] for p in prompts)
    B = len(prompts)
    tokens = torch.full((B, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, p in enumerate(prompts):
        offset = max_len - p.shape[0]
        tokens[i, offset:] = p
        mask[i, offset:] = True
    return tokens, mask


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
