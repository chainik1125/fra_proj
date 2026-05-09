"""Train MLC + three TXCs in parallel on the shared activation cache.

All four crosscoders see the same (seq_idx, pos_idx) pairs each step. MLC
consumes the full L-layer stack at pos_idx; each TXC consumes a T-wide
centered window of one layer's residual. Optimizers are independent so the
four runs don't interact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))  # experiment dir — sleeper_utils / sae_models live here

from sae_models import (  # noqa: E402
    MultiLayerCrosscoder, TemporalCrosscoder, TopKSAE,
)
from sleeper_utils import (  # noqa: E402
    MLC_HOOK_NAMES, SAE_LAYER_HOOKS, TXC_LAYER_HOOKS,
)


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_kv_list(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=str(ROOT / "outputs" / "data"))
    parser.add_argument("--output_dir", default=str(ROOT / "outputs" / "data"))
    parser.add_argument("--d_sae", type=int, default=1536)
    parser.add_argument("--k_total", type=int, default=32)
    parser.add_argument("--T", type=int, default=30, help="TXC window length")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--normalize_every", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--archs",
        nargs="+",
        default=None,
        help="Subset of architectures to train. Defaults to all 8 (mlc + 3 TXC + 4 SAE).",
    )
    parser.add_argument(
        "--sae_layer_hooks_override",
        nargs="+",
        default=None,
        metavar="tag=hook_name",
        help="Override SAE layer→hook mapping (e.g. sae_layer0=blocks.0.ln1.hook_normalized).",
    )
    parser.add_argument(
        "--txc_layer_hooks_override",
        nargs="+",
        default=None,
        metavar="tag=hook_name",
        help="Override TXC layer→hook mapping.",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"[train] device={device}")

    print(f"[train] loading activation cache…")
    acts_cache = torch.load(in_dir / "activations_cache.pt", weights_only=True)
    train_acts: torch.Tensor = acts_cache["train"]  # (N_seq, T_seq, L, D) fp16
    N_seq, T_seq, L, D = train_acts.shape
    print(f"[train]   train acts: N={N_seq} T={T_seq} L={L} d_model={D}")

    # Read the hook names the activations were cached at (matches train's layer
    # index mapping). Falls back to the default MLC stack if not recorded.
    tokens_cache = torch.load(in_dir / "tokens_cache.pt", weights_only=True)
    cached_hook_names = tokens_cache["meta"].get("hook_names") or MLC_HOOK_NAMES
    print(f"[train]   hook_names (from cache): {cached_hook_names}")

    # Pre-pad the seq dimension once for efficient window gather.
    # Move padded acts to GPU so per-batch gather is GPU-side.
    left = args.T // 2
    right = args.T - 1 - left
    acts_padded = F.pad(train_acts, (0, 0, 0, 0, left, right), value=0.0)
    if device == "cuda":
        # Keep fp16 on GPU; cast per-batch to fp32 for the crosscoders.
        acts_padded = acts_padded.to(device)
    print(
        f"[train]   padded shape: {tuple(acts_padded.shape)}  "
        f"dtype={acts_padded.dtype}  device={acts_padded.device}"
    )

    # Build the crosscoders + per-layer SAEs.
    torch.manual_seed(args.seed)
    layer_name_to_idx = {name: i for i, name in enumerate(cached_hook_names)}

    txc_layer_hooks = dict(TXC_LAYER_HOOKS)
    txc_layer_hooks.update(_parse_kv_list(args.txc_layer_hooks_override or []))
    sae_layer_hooks = dict(SAE_LAYER_HOOKS)
    sae_layer_hooks.update(_parse_kv_list(args.sae_layer_hooks_override or []))

    # Drop TXC/SAE entries whose hook isn't present in the cache.
    txc_layer_hooks = {
        tag: h for tag, h in txc_layer_hooks.items() if h in layer_name_to_idx
    }
    sae_layer_hooks = {
        tag: h for tag, h in sae_layer_hooks.items() if h in layer_name_to_idx
    }
    txc_layer_indices = {tag: layer_name_to_idx[h] for tag, h in txc_layer_hooks.items()}
    sae_layer_indices = {tag: layer_name_to_idx[h] for tag, h in sae_layer_hooks.items()}
    print(f"[train]   TXC layer indices: {txc_layer_indices}")
    print(f"[train]   SAE layer indices: {sae_layer_indices}")

    all_models: dict[str, torch.nn.Module] = {}
    # MLC needs the full L-stack and only makes sense when we actually have one.
    if L >= 2:
        all_models["mlc"] = MultiLayerCrosscoder(
            d_in=D, d_sae=args.d_sae, L=L, k_total=args.k_total
        ).to(device)
    for tag in txc_layer_hooks:
        all_models[tag] = TemporalCrosscoder(
            d_in=D, d_sae=args.d_sae, T=args.T, k_total=args.k_total
        ).to(device)
    for tag in sae_layer_hooks:
        all_models[tag] = TopKSAE(d_in=D, d_sae=args.d_sae, k=args.k_total).to(device)

    requested = set(args.archs) if args.archs else None
    if requested is not None:
        missing = requested - set(all_models)
        if missing:
            raise SystemExit(f"[train] requested archs not available: {sorted(missing)}")
        models = {name: m for name, m in all_models.items() if name in requested}
    else:
        models = all_models

    for name, m in models.items():
        print(f"[train]   {name}: {sum(p.numel() for p in m.parameters()):,} params")

    opts = {name: torch.optim.Adam(m.parameters(), lr=args.lr) for name, m in models.items()}

    gen = torch.Generator().manual_seed(args.seed)
    offsets = torch.arange(args.T, device=acts_padded.device)  # 0..T-1

    # If skip_existing, load any already-trained crosscoders and skip them.
    losses_by_model: dict[str, list[float]] = {name: [] for name in models}

    t0 = time.time()
    for step in range(args.n_steps):
        # Sample (seq, pos) pairs on the same device as acts_padded.
        seq_idx = torch.randint(0, N_seq, (args.batch_size,), generator=gen).to(acts_padded.device)
        pos_idx = torch.randint(0, T_seq, (args.batch_size,), generator=gen).to(acts_padded.device)

        # Gather windows directly from padded acts (on GPU).
        seq_exp = seq_idx.unsqueeze(1).expand(-1, args.T)             # (B, T)
        pos_exp = pos_idx.unsqueeze(1) + offsets.unsqueeze(0)         # (B, T)
        windows = acts_padded[seq_exp, pos_exp]                       # (B, T, L, D)
        # MLC input is window center (left index) == un-padded pos_idx row.
        mlc_x = windows[:, left, :, :].to(dtype=torch.float32)        # (B, L, D)
        txc_batches = {
            tag: windows[:, :, idx, :].to(dtype=torch.float32)        # (B, T, D)
            for tag, idx in txc_layer_indices.items()
        }
        # SAE inputs are per-token activations at the chosen layer.
        sae_batches = {
            tag: mlc_x[:, idx, :]                                     # (B, D)
            for tag, idx in sae_layer_indices.items()
        }

        # One step for each model.
        for name, m in models.items():
            if name == "mlc":
                x = mlc_x
            elif name in txc_batches:
                x = txc_batches[name]
            else:
                x = sae_batches[name]
            x_hat, _ = m(x)
            loss = (x - x_hat).pow(2).sum(dim=-1).mean()
            opts[name].zero_grad(set_to_none=True)
            loss.backward()
            opts[name].step()
            losses_by_model[name].append(loss.item())

        if (step + 1) % args.normalize_every == 0:
            for m in models.values():
                m.normalize_decoder()

        if (step + 1) % args.print_every == 0:
            parts = [
                f"{n}={losses_by_model[n][-1]:.4f}" for n in models
            ]
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed
            print(f"[train] step {step+1}/{args.n_steps} ({steps_per_sec:.1f} it/s) | " + " ".join(parts))

    print(f"[train] done in {time.time() - t0:.1f}s")

    # Save state dicts + config.
    for name, m in models.items():
        state = {key: v.detach().cpu() for key, v in m.state_dict().items()}
        layer_hook = txc_layer_hooks.get(name) or sae_layer_hooks.get(name)
        layer_idx = txc_layer_indices.get(name, sae_layer_indices.get(name))
        payload = {
            "state_dict": state,
            "config": {
                "class_name": type(m).__name__,
                "d_in": D,
                "d_sae": args.d_sae,
                "k_total": args.k_total,
                "T": args.T if name in txc_layer_hooks else None,
                "L": L if name == "mlc" else None,
                "layer_hook": layer_hook,
                "layer_idx": layer_idx,
                "hook_names": MLC_HOOK_NAMES if name == "mlc" else None,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
            },
            "final_loss": losses_by_model[name][-1],
            "loss_trajectory": losses_by_model[name][:: max(1, args.n_steps // 200)],
        }
        torch.save(payload, out_dir / f"crosscoder_{name}.pt")
        print(f"[train] saved crosscoder_{name}.pt  final_loss={losses_by_model[name][-1]:.4f}")

    meta = {
        "d_sae": args.d_sae,
        "k_total": args.k_total,
        "T": args.T,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "normalize_every": args.normalize_every,
        "seed": args.seed,
        "txc_layer_indices": txc_layer_indices,
        "final_losses": {n: losses_by_model[n][-1] for n in models},
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
