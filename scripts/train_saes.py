"""Train TopK SAEs on residual hooks of the sleeper model.

Subsumes train_sae.py, train_all_saes_6seeds.py, train_saes_per_layer.py.

Examples:
    # TinyStories: canonical 6-seed paired (ln1, resid_mid) SAEs at block 0:
    uv run -m scripts.train_saes

    # 50k steps on the same set:
    uv run -m scripts.train_saes --n_steps 50000

    # one SAE at a custom hook:
    uv run -m scripts.train_saes --seeds 0 --hooks blocks.0.hook_resid_post

    # cross-layer baseline: 6 seeds × all layers × {resid_mid, resid_post}:
    uv run -m scripts.train_saes --layers 0 1 2 3 --hooks resid_mid resid_post

    # Llama (Cadenza sleeper): one mid-layer resid_mid SAE, matching Aniket's
    # d_sae=32768 / k=64 / Llama defaults. Loop the cell grid externally.
    uv run -m scripts.train_saes --model llama
    uv run -m scripts.train_saes --model llama --layers 3 --hooks ln1
    uv run -m scripts.train_saes --model llama --layers 3 16 29 \\
        --hooks ln1 resid_mid resid_post   # full 9-cell sweep (high RAM)

Output layout:
    TS default (ln1+resid_mid block-0): weights/seeds[_Nk]/sae_{kind}_s{seed}.pt
    TS multi-layer:                     weights/seeds_per_layer/sae_L{L}_{kind}_s{seed}.pt
    Llama default:                      weights/seeds_llama[_Nk]/...
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from sleeper.model import (
    MODELS, ModelName, cache_activations,
    load_paired_dataset, load_sleeper_hf_components, load_sleeper_model,
)
from sleeper.sae import save, train


_DEFAULT_HOOKS_BLOCK0 = ("blocks.0.ln1.hook_normalized", "blocks.0.hook_resid_mid")

# Per-model training defaults. TS values match the historical pipeline; Llama
# values track Aniket's (d_sae=32768, k=64, default cell = mid-layer resid_mid,
# lr=3e-4, training_tokens=50M = 12.2k steps × batch 4096, context_size=1024).
# Handrolled-backend caveat: n_train capped by Cadenza train split at ~3k
# balanced rows when seq_len=128 due to clean-row length filter; the saelens
# backend streams + cycles so n_train is unused there.
_LLAMA_DEFAULTS = dict(
    d_sae=32_768, k=64,
    layers=(16,), hooks=("resid_mid",),
    n_train=3_000, seq_len=128,
    n_steps=12_200,
    lr=3e-4,
)
_TINYSTORIES_DEFAULTS = dict(
    d_sae=1_536, k=32,
    layers=(0,), hooks=None,                # None → _DEFAULT_HOOKS_BLOCK0
    n_train=10_000, seq_len=128,
    n_steps=4_000,
    lr=5e-4,
)
_MODEL_DEFAULTS = {"tinystories": _TINYSTORIES_DEFAULTS, "llama": _LLAMA_DEFAULTS}

# Default SAE training backend per model. ``handrolled`` = sleeper.sae.train
# (bare-bones constant-lr Adam, fits in 140 lines, what TS baselines were
# trained with). ``saelens`` = sae-lens 6.44 ``SAETrainingRunner`` with cosine
# LR + warmup + TopK aux loss + dead-feature resampling, output converted
# back to sleeper.sae.TopKSAE so downstream code is unchanged.
_BACKEND_DEFAULTS = {"tinystories": "handrolled", "llama": "saelens"}
_BACKENDS = ("handrolled", "saelens")


def _hook_kind(name: str) -> str:
    """Short tag for filenames: blocks.0.ln1.hook_normalized → ln1; hook_resid_mid → resid_mid."""
    if "ln1" in name:
        return "ln1"
    return name.split("hook_", 1)[-1]


def _expand_hooks(hooks: list[str], layers: list[int]) -> list[str]:
    """Expand short hook names like 'resid_mid' / 'ln1' into fully-qualified hook strings
    across the requested layers. Full names (already containing 'blocks.') pass through unchanged.
    """
    out: list[str] = []
    for h in hooks:
        if h.startswith("blocks."):
            out.append(h)
            continue
        for L in layers:
            if h == "ln1":
                out.append(f"blocks.{L}.ln1.hook_normalized")
            else:
                out.append(f"blocks.{L}.hook_{h.lstrip('hook_')}")
    return out


def _out_dir(
    n_steps: int,
    layers: list[int],
    explicit: Path | None,
    model: ModelName = "tinystories",
) -> Path:
    """Default output-dir naming convention.

    TS multi-layer → weights/seeds_per_layer/
    TS 4k single-layer → weights/seeds/
    TS Nk single-layer → weights/seeds_{N//1000}k/
    Llama → weights/seeds_llama[_per_layer | _Nk] (default 6k drops the suffix)
    """
    if explicit is not None:
        return explicit
    weights = Path("weights")
    prefix = "seeds_llama" if model == "llama" else "seeds"
    if len(set(layers)) > 1:
        return weights / f"{prefix}_per_layer"
    default_n = _MODEL_DEFAULTS[model]["n_steps"]
    if n_steps == default_n:
        return weights / prefix
    return weights / f"{prefix}_{n_steps // 1000}k"


def _ckpt_path(out_dir: Path, hook: str, seed: int, layers: list[int]) -> Path:
    """File path for one (hook, seed) SAE checkpoint."""
    kind = _hook_kind(hook)
    if len(set(layers)) > 1:
        # blocks.{L}.{rest} → extract L for the filename
        L = int(hook.split(".")[1])
        return out_dir / f"sae_L{L}_{kind}_s{seed}.pt"
    return out_dir / f"sae_{kind}_s{seed}.pt"


def train_saes(
    seeds: list[int],
    *,
    model: ModelName = "tinystories",
    sae_backend: str | None = None,
    n_steps: int | None = None,
    layers: list[int] | None = None,
    hooks: list[str] | None = None,
    out_dir: Path | None = None,
    n_train: int | None = None,
    seq_len: int | None = None,
    d_sae: int | None = None,
    k: int | None = None,
    batch_size: int = 4_096,
    lr: float | None = None,
    device: str | None = None,
) -> Path:
    """Train SAEs for the given (seeds × hooks) cross-product. Idempotent.

    All sizing kwargs default to per-model presets from ``_MODEL_DEFAULTS``.
    ``sae_backend`` defaults to ``_BACKEND_DEFAULTS[model]`` (TS=handrolled,
    Llama=saelens). Output checkpoints are written in the same
    ``sleeper.sae.TopKSAE`` format regardless of backend.

    Returns the output directory path.
    """
    d = _MODEL_DEFAULTS[model]
    n_steps  = n_steps  if n_steps  is not None else d["n_steps"]
    layers   = layers   if layers   is not None else list(d["layers"])
    hooks    = hooks    if hooks    is not None else (list(d["hooks"]) if d["hooks"] else None)
    n_train  = n_train  if n_train  is not None else d["n_train"]
    seq_len  = seq_len  if seq_len  is not None else d["seq_len"]
    d_sae    = d_sae    if d_sae    is not None else d["d_sae"]
    k        = k        if k        is not None else d["k"]
    lr       = lr       if lr       is not None else d["lr"]
    backend  = sae_backend or _BACKEND_DEFAULTS[model]
    if backend not in _BACKENDS:
        raise ValueError(f"unknown sae_backend {backend!r}; choices: {_BACKENDS}")

    hook_names = _expand_hooks(list(hooks) if hooks else list(_DEFAULT_HOOKS_BLOCK0), layers)
    out = _out_dir(n_steps, layers, out_dir, model=model)
    out.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Per-(hook, seed) target paths; identify which are missing.
    todo: list[tuple[str, int, Path]] = []
    for h in hook_names:
        for s in seeds:
            p = _ckpt_path(out, h, s, layers)
            if not p.exists():
                todo.append((h, s, p))

    if not todo:
        print(f"[train-saes] out={out}  seeds={seeds}  hooks={hook_names}: "
              f"all checkpoints present, nothing to do")
        return out

    missing_hooks = sorted({h for (h, _, _) in todo})
    print(f"[train-saes] model={model}  backend={backend}  device={device}  "
          f"n_steps={n_steps}  seeds={seeds}  hooks={hook_names}  d_sae={d_sae}  "
          f"k={k}  out={out}  missing={len(todo)} checkpoints")

    if backend == "handrolled":
        _train_handrolled(model, todo, missing_hooks, n_train, seq_len, d_sae, k,
                          n_steps, batch_size, lr, device)
    else:
        _train_saelens(model, todo, n_train, seq_len, d_sae, k,
                       n_steps, batch_size, lr, device)

    print("[train-saes] done")
    return out


def _train_handrolled(
    model: ModelName, todo: list[tuple[str, int, Path]], missing_hooks: list[str],
    n_train: int, seq_len: int, d_sae: int, k: int, n_steps: int,
    batch_size: int, lr: float, device: str,
) -> None:
    """Original path: pre-cache all hook activations once, train each cell from cache."""
    hooked = load_sleeper_model(model=model, device=device)
    splits = load_paired_dataset(hooked.tokenizer, n_train=n_train, n_val=0, n_test=0,
                                 seq_len=seq_len, seed=0, model=model)
    train_tokens = splits["train"].tokens
    print(f"[train-saes] harvesting {train_tokens.shape[0]} seqs at hooks={missing_hooks}")
    acts = cache_activations(model=hooked, tokens=train_tokens,
                             hook_names=missing_hooks, chunk_size=16)
    for hook, seed, path in todo:
        if path.exists():
            print(f"[train-saes] skip {path} (exists)")
            continue
        sae, _ = train(acts[hook], d_sae=d_sae, k=k, n_steps=n_steps,
                       batch_size=batch_size, lr=lr, seed=seed, device=device)
        save(sae, path, layer_hook=hook,
             n_train_seqs=int(train_tokens.shape[0]),
             seq_len=seq_len, n_steps=n_steps, batch_size=batch_size, lr=lr,
             sae_backend="handrolled")
        print(f"[train-saes] wrote {path}")


def _train_saelens(
    model: ModelName, todo: list[tuple[str, int, Path]],
    n_train: int, seq_len: int, d_sae: int, k: int,
    n_steps: int, batch_size: int, lr: float, device: str,
) -> None:
    """sae-lens path: stream activations from the paired dataset, run sae-lens's
    full trainer per (hook, seed) cell, convert to our TopKSAE format.

    When >=2 CUDA devices are visible, places the language model on cuda:1
    and the SAE + optimizer on cuda:0 so the LLM forward pass overlaps with
    the SAE training step (sae-lens's prefetch hides the LLM->SAE transfer).
    """
    import torch as _torch
    from sleeper.model import MODELS as _MODELS
    from sleeper.sae_saelens import train_saelens_cell

    cfg = _MODELS[model]
    n_gpu = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
    sae_device = device if n_gpu < 2 else "cuda:0"
    llm_device = None if n_gpu < 2 else "cuda:1"
    print(f"[train-saes] sae_device={sae_device}  llm_device={llm_device or '(same)'}  "
          f"(visible GPUs: {n_gpu})")
    # Load the HF model directly onto llm_device so sae-lens doesn't have to
    # move it between cards.
    hf_model, tokenizer = load_sleeper_hf_components(model=model, device=llm_device or sae_device)
    d_in = hf_model.config.hidden_size
    print(f"[train-saes] sae-lens streaming from {cfg.dataset!r}  d_in={d_in}")
    for hook, seed, path in todo:
        if path.exists():
            print(f"[train-saes] skip {path} (exists)")
            continue
        sae = train_saelens_cell(
            hf_model=hf_model, tokenizer=tokenizer, cfg=cfg,
            hook_name=hook, d_in=d_in, d_sae=d_sae, k=k,
            n_train_seqs=n_train, seq_len=seq_len, n_steps=n_steps,
            n_checkpoints=int(os.environ.get("SAELENS_N_CHECKPOINTS", 0)),
            batch_size=batch_size, lr=lr, seed=seed,
            device=sae_device, llm_device=llm_device,
            wandb_project=os.environ.get("WANDB_PROJECT") or None,
            wandb_entity=os.environ.get("WANDB_ENTITY") or None,
            run_name=f"{model}_L{int(hook.split('.')[1])}_{hook.rsplit('.',1)[-1]}_s{seed}",
        )
        save(sae, path, layer_hook=hook,
             n_train_seqs=int(n_train),
             seq_len=seq_len, n_steps=n_steps,
             batch_size=batch_size, lr=lr, sae_backend="saelens")
        print(f"[train-saes] wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model",   choices=list(MODELS), default="tinystories",
                   help="Which sleeper to harvest activations from. Sets per-model "
                        "defaults for --d_sae / --k / --n_train / --layers / --hooks / --n_steps.")
    p.add_argument("--sae_backend", choices=_BACKENDS, default=None,
                   help="SAE training backend. Defaults: TS=handrolled (bare-bones "
                        "Adam, what TS baselines used), Llama=saelens (full sae-lens "
                        "stack: cosine LR + warmup + TopK aux loss + dead-feature "
                        "resampling). Output checkpoint format is the same.")
    p.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                   help="SAE training seeds (set to e.g. [0] for Llama smoke runs).")
    p.add_argument("--n_steps", type=int, default=None)
    p.add_argument("--layers",  type=int, nargs="+", default=None,
                   help="Block indices. Multi-layer triggers the seeds_per_layer/ naming. "
                        "Defaults: TS=[0], Llama=[16].")
    p.add_argument("--hooks",   type=str, nargs="+", default=None,
                   help="Hook names. Short forms 'ln1' / 'resid_mid' / 'resid_post' are "
                        "expanded across --layers. Defaults: TS=[ln1,resid_mid]@block 0, "
                        "Llama=[resid_mid].")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Override the default output directory naming.")
    p.add_argument("--n_train",   type=int, default=None)
    p.add_argument("--seq_len",   type=int, default=None)
    p.add_argument("--d_sae",     type=int, default=None)
    p.add_argument("--k",         type=int, default=None)
    p.add_argument("--batch_size", type=int, default=4_096)
    p.add_argument("--lr",        type=float, default=None,
                   help="Defaults: TS=5e-4, Llama=1e-4 (lower lr keeps Adam stable at d_in=4096).")
    p.add_argument("--device",    default=None)
    args = p.parse_args()

    train_saes(
        seeds=args.seeds, model=args.model, sae_backend=args.sae_backend,
        n_steps=args.n_steps,
        layers=args.layers, hooks=args.hooks, out_dir=args.out_dir,
        n_train=args.n_train, seq_len=args.seq_len,
        d_sae=args.d_sae, k=args.k, batch_size=args.batch_size, lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
