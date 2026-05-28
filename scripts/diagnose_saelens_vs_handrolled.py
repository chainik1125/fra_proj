"""Drive the sae-lens TopKTrainingSAE through the handrolled training loop.

The handrolled saelens preset (SAELENS_HANDROLLED_REPLICATE=1) tries to
configure sae-lens's trainer to match sleeper.sae.train, but the trainer
hard-codes clip_grad_norm_=1.0 (sae_lens/training/sae_trainer.py:345) and
runs an activation buffer that doesn't match handrolled's pre-cached random
sampling. Either of those can keep weights stuck near init when we remove
the input normalization that sae-lens's defaults rely on.

This script bypasses the sae-lens trainer entirely: it instantiates the
sae-lens TopKTrainingSAE architecture and trains it with the **exact**
handrolled loop (sleeper.sae.train, modulo swapping the SAE module). If the
resulting SAE weights and downstream conv-baseline numbers match handrolled,
the sae-lens architecture is equivalent and the original gap was the
sae-lens harness. If not, the SAE math differs.

Usage:
    uv run -m scripts.diagnose_saelens_vs_handrolled --seeds 0 1 2 3 4 5 \
        --out_dir weights/seeds_saelens_inhandrolled
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from sleeper.model import (
    cache_activations, load_paired_dataset, load_sleeper_model,
)
from sleeper.sae import TopKSAE, save


def _build_saelens_topk_sae(d_in: int, d_sae: int, k: int, device: str):
    """Construct a sae-lens TopKTrainingSAE configured to match handrolled.

    Returns the module — `forward(x)` returns reconstructed x_hat. We pull
    W_enc/W_dec/b_enc/b_dec out at save time and write them into our
    TopKSAE shape for downstream loading.
    """
    from sae_lens.saes.topk_sae import (
        TopKTrainingSAE, TopKTrainingSAEConfig,
    )
    cfg = TopKTrainingSAEConfig(
        d_in=d_in, d_sae=d_sae, k=k,
        dtype="float32", device=device,
        normalize_activations="none",       # raw activations, handrolled-style
        apply_b_dec_to_input=True,           # handrolled subtracts b_dec in encode
        decoder_init_norm=1.0,               # handrolled W_dec rows have unit norm
        rescale_acts_by_decoder_norm=False,  # handrolled does no per-feature rescale
        aux_loss_coefficient=0.0,            # we'll skip aux loss anyway
    )
    sae = TopKTrainingSAE(cfg)
    sae.to(device)
    return sae


def _l2_normalize_decoder(W_dec: torch.Tensor) -> None:
    """In-place L2 row-normalize W_dec (shape d_sae, d_in). Matches
    sleeper.sae.TopKSAE._normalize_decoder."""
    norms = W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    W_dec.data.div_(norms)


def train_saelens_in_handrolled(
    acts: torch.Tensor,       # (N, T, d)
    d_sae: int,
    k: int,
    *,
    n_steps: int = 4_000,
    batch_size: int = 4_096,
    lr: float = 5e-4,
    normalize_every: int = 100,
    print_every: int = 200,
    seed: int = 0,
    device: str = "cuda",
):
    """Handrolled loop, sae-lens architecture.

    Mirrors sleeper.sae.train byte-for-byte except for the SAE module.
    """
    torch.manual_seed(seed)
    N, T, D = acts.shape
    acts = acts.to(device) if device != "cpu" else acts

    sae = _build_saelens_topk_sae(d_in=D, d_sae=d_sae, k=k, device=device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    losses: list[float] = []
    gen = torch.Generator().manual_seed(seed)

    t0 = time.time()
    for step in range(n_steps):
        seq_idx = torch.randint(0, N, (batch_size,), generator=gen).to(acts.device)
        pos_idx = torch.randint(0, T, (batch_size,), generator=gen).to(acts.device)
        x = acts[seq_idx, pos_idx].to(torch.float32)
        x_hat = sae(x)
        loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (step + 1) % normalize_every == 0:
            with torch.no_grad():
                _l2_normalize_decoder(sae.W_dec)
        if (step + 1) % print_every == 0:
            rate = (step + 1) / (time.time() - t0)
            print(f"[saelens-in-handrolled] step {step+1}/{n_steps} "
                  f"({rate:.1f} it/s) loss={loss.item():.4f}", flush=True)
    print(f"[saelens-in-handrolled] done in {time.time() - t0:.1f}s  "
          f"final_loss={losses[-1]:.4f}")
    return sae, losses


def _convert_to_topksae(sl_sae, d_in: int, d_sae: int, k: int,
                        device: str) -> TopKSAE:
    """Copy sae-lens TopKTrainingSAE weights into our TopKSAE shape."""
    sae = TopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    src = sl_sae.state_dict()
    sae.load_state_dict({n: src[n].detach().to(torch.float32).cpu()
                         for n in ("W_enc", "b_enc", "W_dec", "b_dec")})
    return sae.to(device).eval()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--hooks", type=str, nargs="+",
                   default=["blocks.0.hook_resid_mid"])
    p.add_argument("--n_train",   type=int, default=10_000)
    p.add_argument("--seq_len",   type=int, default=128)
    p.add_argument("--d_sae",     type=int, default=1_536)
    p.add_argument("--k",         type=int, default=32)
    p.add_argument("--n_steps",   type=int, default=4_000)
    p.add_argument("--batch_size", type=int, default=4_096)
    p.add_argument("--lr",        type=float, default=5e-4)
    p.add_argument("--device",    default=None)
    p.add_argument("--out_dir",   type=Path, required=True)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diag] caching activations: n_train={args.n_train}  "
          f"seq_len={args.seq_len}  hooks={args.hooks}", flush=True)
    hooked = load_sleeper_model(model="tinystories", device=device)
    splits = load_paired_dataset(hooked.tokenizer, n_train=args.n_train,
                                  n_val=0, n_test=0, seq_len=args.seq_len,
                                  seed=0, model="tinystories")
    train_tokens = splits["train"].tokens
    acts = cache_activations(model=hooked, tokens=train_tokens,
                              hook_names=args.hooks, chunk_size=16)

    for hook in args.hooks:
        kind = "ln1" if "ln1" in hook else hook.rsplit("hook_", 1)[-1]
        for seed in args.seeds:
            out_path = args.out_dir / f"sae_{kind}_s{seed}.pt"
            if out_path.exists():
                print(f"[diag] skip {out_path} (exists)")
                continue
            print(f"\n[diag] ── seed={seed} hook={hook} ──", flush=True)
            sl_sae, _ = train_saelens_in_handrolled(
                acts[hook], d_sae=args.d_sae, k=args.k,
                n_steps=args.n_steps, batch_size=args.batch_size,
                lr=args.lr, seed=seed, device=device,
            )
            sae = _convert_to_topksae(sl_sae, d_in=acts[hook].shape[-1],
                                       d_sae=args.d_sae, k=args.k, device=device)
            save(sae, out_path, layer_hook=hook,
                 n_train_seqs=int(train_tokens.shape[0]),
                 seq_len=args.seq_len, n_steps=args.n_steps,
                 batch_size=args.batch_size, lr=args.lr,
                 sae_backend="saelens_arch_handrolled_loop")
            print(f"[diag] wrote {out_path}")


if __name__ == "__main__":
    main()
