"""Train TopK SAEs on residual hooks of the sleeper model.

Subsumes train_sae.py, train_all_saes_6seeds.py, train_saes_per_layer.py.

Examples:
    # the canonical 6-seed paired (ln1, resid_mid) SAEs at block 0 (4k steps):
    uv run -m scripts.train_saes

    # 50k steps on the same set:
    uv run -m scripts.train_saes --n_steps 50000

    # one SAE at a custom hook:
    uv run -m scripts.train_saes --seeds 0 --hooks blocks.0.hook_resid_post

    # cross-layer baseline: 6 seeds × all layers × {resid_mid, resid_post}:
    uv run -m scripts.train_saes --layers 0 1 2 3 --hooks resid_mid resid_post

Output layout:
    default block-0 ln1+resid_mid:  weights/seeds[_Nk]/sae_{kind}_s{seed}.pt
    multi-layer:                     weights/seeds_per_layer/sae_L{L}_{kind}_s{seed}.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import save, train


_DEFAULT_HOOKS_BLOCK0 = ("blocks.0.ln1.hook_normalized", "blocks.0.hook_resid_mid")


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


def _out_dir(n_steps: int, layers: list[int], explicit: Path | None) -> Path:
    """Default output-dir naming convention.

    multi-layer → weights/seeds_per_layer/
    single-layer + 4k → weights/seeds/
    single-layer + Nk → weights/seeds_{N//1000}k/
    """
    if explicit is not None:
        return explicit
    weights = Path("weights")
    if len(set(layers)) > 1:
        return weights / "seeds_per_layer"
    if n_steps == 4_000:
        return weights / "seeds"
    return weights / f"seeds_{n_steps // 1000}k"


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
    n_steps: int = 4_000,
    layers: list[int] | None = None,
    hooks: list[str] | None = None,
    out_dir: Path | None = None,
    n_train: int = 10_000,
    seq_len: int = 128,
    d_sae: int = 1_536,
    k: int = 32,
    batch_size: int = 4_096,
    lr: float = 5e-4,
    device: str | None = None,
) -> Path:
    """Train SAEs for the given (seeds × hooks) cross-product. Idempotent.

    Returns the output directory path.
    """
    layers = layers or [0]
    hook_names = _expand_hooks(list(hooks) if hooks else list(_DEFAULT_HOOKS_BLOCK0), layers)
    out = _out_dir(n_steps, layers, out_dir)
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
    print(f"[train-saes] device={device}  n_steps={n_steps}  seeds={seeds}  "
          f"hooks={hook_names}  out={out}  missing={len(todo)} checkpoints")

    model = load_sleeper_model(device=device)
    splits = load_paired_dataset(model.tokenizer, n_train=n_train, n_val=0, n_test=0,
                                  seq_len=seq_len, seed=0)
    train_tokens = splits["train"].tokens
    print(f"[train-saes] harvesting {train_tokens.shape[0]} seqs at hooks={missing_hooks}")
    acts = cache_activations(model=model, tokens=train_tokens,
                             hook_names=missing_hooks, chunk_size=16)

    for hook, seed, path in todo:
        if path.exists():  # idempotent re-check
            print(f"[train-saes] skip {path} (exists)")
            continue
        sae, _ = train(acts[hook], d_sae=d_sae, k=k, n_steps=n_steps,
                       batch_size=batch_size, lr=lr, seed=seed, device=device)
        save(sae, path, layer_hook=hook,
             n_train_seqs=int(train_tokens.shape[0]),
             seq_len=seq_len, n_steps=n_steps, batch_size=batch_size, lr=lr)
        print(f"[train-saes] wrote {path}")

    print("[train-saes] done")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--n_steps", type=int, default=4_000)
    p.add_argument("--layers",  type=int, nargs="+", default=[0],
                   help="Block indices. Multi-layer triggers the seeds_per_layer/ naming.")
    p.add_argument("--hooks",   type=str, nargs="+", default=None,
                   help="Hook names. Short forms 'ln1' / 'resid_mid' / 'resid_post' are "
                        "expanded across --layers; fully-qualified 'blocks.L.…' names pass through. "
                        "Default: ln1 + resid_mid at block 0.")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Override the default output directory naming.")
    p.add_argument("--n_train",   type=int, default=10_000)
    p.add_argument("--seq_len",   type=int, default=128)
    p.add_argument("--d_sae",     type=int, default=1_536)
    p.add_argument("--k",         type=int, default=32)
    p.add_argument("--batch_size", type=int, default=4_096)
    p.add_argument("--lr",        type=float, default=5e-4)
    p.add_argument("--device",    default=None)
    args = p.parse_args()

    train_saes(
        seeds=args.seeds, n_steps=args.n_steps, layers=args.layers, hooks=args.hooks,
        out_dir=args.out_dir, n_train=args.n_train, seq_len=args.seq_len,
        d_sae=args.d_sae, k=args.k, batch_size=args.batch_size, lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
