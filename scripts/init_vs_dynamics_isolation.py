"""Isolate whether the downstream conv-baseline gap between handrolled and
sae-lens TopK SAEs is caused by INIT differences or TRAINING DYNAMICS.

Three paths, all trained on the same activation cache, same data-sampling
RNG, same Adam (lr=5e-4), 4000 steps, batch 4096, decoder L2-renorm every
100 steps, CPU torch.Generator for sampling:

  A. Handrolled init  + handrolled forward/backward  (sleeper.sae.train)
  B. Handrolled init  + sae-lens forward/backward    (copy A's init weights
                                                       into a fresh sae-lens
                                                       TopKTrainingSAE, then
                                                       train via handrolled
                                                       loop body)
  C. Sae-lens default init + sae-lens forward/backward (diagnose-flow:
                                                         scripts.diagnose_*)

Verdict logic:
  A vs B:  same init, different math.
           same  -> SAE math equivalent; init explains everything.
           diff  -> training dynamics differ despite identical init+data+loop.
  A vs C:  quantifies init-only effect (same arch class as B vs A pairing).

Outputs:
  <out_dir>/sae_resid_mid_s0.pt    (path A)
  <out_dir_B>/sae_resid_mid_s0.pt  (path B, weights converted back to TopKSAE)
  <out_dir_C>/sae_resid_mid_s0.pt  (path C, weights converted back to TopKSAE)
  <out_dir>/summary.json           (FVU + paths for the orchestrator to invoke
                                     scripts.downstream_baseline on each.)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sleeper.model import (
    cache_activations, load_paired_dataset, load_sleeper_model,
)
from sleeper.sae import TopKSAE, save


# ---------------------------------------------------------------------------
# sae-lens module (matches scripts/diagnose_saelens_vs_handrolled.py exactly).
# ---------------------------------------------------------------------------
def _build_saelens_topk_sae(d_in: int, d_sae: int, k: int, device: str):
    from sae_lens.saes.topk_sae import (
        TopKTrainingSAE, TopKTrainingSAEConfig,
    )
    cfg = TopKTrainingSAEConfig(
        d_in=d_in, d_sae=d_sae, k=k,
        dtype="float32", device=device,
        normalize_activations="none",
        apply_b_dec_to_input=True,
        decoder_init_norm=1.0,
        rescale_acts_by_decoder_norm=False,
        aux_loss_coefficient=0.0,
    )
    sae = TopKTrainingSAE(cfg)
    sae.to(device)
    return sae


def _l2_normalize_decoder_(W_dec: torch.Tensor) -> None:
    norms = W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    W_dec.data.div_(norms)


# ---------------------------------------------------------------------------
# Three training paths — all share the same loop body.
# ---------------------------------------------------------------------------
def _train_loop(sae, acts, *, n_steps, batch_size, lr, normalize_every,
                print_every, seed, tag, W_dec_for_normalize=None):
    """sleeper.sae.train body, parameterised over the module."""
    torch.manual_seed(seed)
    N, T, D = acts.shape
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    losses: list[float] = []
    gen = torch.Generator().manual_seed(seed)

    if W_dec_for_normalize is None:
        W_dec_for_normalize = sae.W_dec  # both modules expose .W_dec

    t0 = time.time()
    for step in range(n_steps):
        seq_idx = torch.randint(0, N, (batch_size,), generator=gen).to(acts.device)
        pos_idx = torch.randint(0, T, (batch_size,), generator=gen).to(acts.device)
        x = acts[seq_idx, pos_idx].to(torch.float32)

        out = sae(x)
        # Handrolled TopKSAE returns (x_hat, z); sae-lens returns x_hat tensor.
        x_hat = out[0] if isinstance(out, tuple) else out
        loss = (x - x_hat).pow(2).sum(dim=-1).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if (step + 1) % normalize_every == 0:
            with torch.no_grad():
                _l2_normalize_decoder_(W_dec_for_normalize)

        if (step + 1) % print_every == 0:
            rate = (step + 1) / (time.time() - t0)
            print(f"[{tag}] step {step+1}/{n_steps} ({rate:.1f} it/s) "
                  f"loss={loss.item():.4f}", flush=True)
    print(f"[{tag}] done in {time.time() - t0:.1f}s  final_loss={losses[-1]:.4f}",
          flush=True)
    return losses


def _convert_saelens_to_topksae(sl_sae, d_in, d_sae, k, device) -> TopKSAE:
    sae = TopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    src = sl_sae.state_dict()
    sae.load_state_dict({n: src[n].detach().to(torch.float32).cpu()
                         for n in ("W_enc", "b_enc", "W_dec", "b_dec")})
    return sae.to(device).eval()


@torch.no_grad()
def _fvu(sae: TopKSAE, val_acts: torch.Tensor, chunk: int = 4096) -> float:
    """FVU = 1 - explained_variance. Computes ||x - x_hat||^2 / ||x - mean||^2
    over a flat (N*T, d) split of held-out activations.
    """
    device = next(sae.parameters()).device
    flat = val_acts.reshape(-1, val_acts.shape[-1]).to(torch.float32)
    mean = flat.mean(dim=0, keepdim=True)
    num = 0.0
    den = 0.0
    for s in range(0, flat.shape[0], chunk):
        x = flat[s:s+chunk].to(device)
        x_hat, _ = sae(x)
        num += (x - x_hat).pow(2).sum().item()
        den += (x - mean.to(device)).pow(2).sum().item()
    return num / max(den, 1e-12)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--hook",       type=str, default="blocks.0.hook_resid_mid")
    p.add_argument("--n_train",    type=int, default=10_000)
    p.add_argument("--seq_len",    type=int, default=128)
    p.add_argument("--d_sae",      type=int, default=1_536)
    p.add_argument("--k",          type=int, default=32)
    p.add_argument("--n_steps",    type=int, default=4_000)
    p.add_argument("--batch_size", type=int, default=4_096)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--val_frac",   type=float, default=0.1,
                   help="Fraction of (N) sequences held out for FVU.")
    p.add_argument("--out_root",   type=Path, required=True,
                   help="Creates <out_root>/{A,B,C}/sae_resid_mid_s{seed}.pt.")
    p.add_argument("--device",     default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for sub in ("A", "B", "C"):
        (args.out_root / sub).mkdir(parents=True, exist_ok=True)

    # ----- Cache activations once. -----
    print(f"[init] caching activations: n_train={args.n_train}  "
          f"seq_len={args.seq_len}  hook={args.hook}", flush=True)
    hooked = load_sleeper_model(model="tinystories", device=device)
    splits = load_paired_dataset(hooked.tokenizer, n_train=args.n_train,
                                  n_val=0, n_test=0, seq_len=args.seq_len,
                                  seed=0, model="tinystories")
    train_tokens = splits["train"].tokens
    acts_d = cache_activations(model=hooked, tokens=train_tokens,
                                hook_names=[args.hook], chunk_size=16)
    acts = acts_d[args.hook]  # (N, T, d) fp16 on CPU
    print(f"[init] acts: shape={tuple(acts.shape)}  dtype={acts.dtype}", flush=True)

    # Held-out validation split for FVU (deterministic, same for all 3 paths).
    N = acts.shape[0]
    n_val = max(1, int(N * args.val_frac))
    val_acts = acts[-n_val:].clone()  # last n_val sequences as held-out
    train_acts = acts[:N - n_val]
    print(f"[init] FVU val split: n_val={n_val}  n_train_acts={N - n_val}", flush=True)

    # Move training acts to GPU once (matches sleeper.sae.train behaviour).
    train_acts_gpu = train_acts.to(device) if device != "cpu" else train_acts
    d_in = train_acts_gpu.shape[-1]

    free_model = hooked  # drop after we no longer need it
    del hooked
    torch.cuda.empty_cache() if device != "cpu" else None

    summary: dict = {
        "config": {**vars(args), "out_root": str(args.out_root),
                    "d_in": int(d_in),
                    "n_train_acts": int(train_acts_gpu.shape[0]),
                    "n_val_acts": int(val_acts.shape[0])},
        "paths": {},
    }

    # ===================================================================
    # Path A: handrolled init + handrolled forward/backward.
    # ===================================================================
    print(f"\n[A] handrolled init + handrolled loop  seed={args.seed}", flush=True)
    torch.manual_seed(args.seed)
    sae_A = TopKSAE(d_in=d_in, d_sae=args.d_sae, k=args.k).to(device)
    # Snapshot A's init for path B.
    init_state = {k: v.detach().clone() for k, v in sae_A.state_dict().items()}
    _train_loop(sae_A, train_acts_gpu,
                n_steps=args.n_steps, batch_size=args.batch_size,
                lr=args.lr, normalize_every=100, print_every=200,
                seed=args.seed, tag="A")
    fvu_A = _fvu(sae_A, val_acts)
    pathA = args.out_root / "A" / f"sae_resid_mid_s{args.seed}.pt"
    save(sae_A, pathA, layer_hook=args.hook,
         n_train_seqs=int(train_acts_gpu.shape[0]),
         seq_len=args.seq_len, n_steps=args.n_steps,
         batch_size=args.batch_size, lr=args.lr,
         sae_backend="handrolled_init_handrolled_loop_init_isolation")
    print(f"[A] FVU={fvu_A:.4f}  wrote {pathA}", flush=True)
    summary["paths"]["A"] = {"sae_path": str(pathA), "fvu": fvu_A,
                              "label": "handrolled_init_handrolled_loop"}

    # ===================================================================
    # Path B: handrolled init -> copied into a fresh sae-lens module,
    #         trained via handrolled loop body.
    # ===================================================================
    print(f"\n[B] handrolled init copied into sae-lens module + handrolled loop"
          f"  seed={args.seed}", flush=True)
    torch.manual_seed(args.seed)
    sae_B_sl = _build_saelens_topk_sae(d_in=d_in, d_sae=args.d_sae,
                                        k=args.k, device=device)
    # Copy A's init verbatim. sae-lens TopKTrainingSAE uses the same
    # parameter names (W_enc, b_enc, W_dec, b_dec) — verify before copy.
    sl_keys = set(sae_B_sl.state_dict().keys())
    missing = {"W_enc", "b_enc", "W_dec", "b_dec"} - sl_keys
    if missing:
        raise RuntimeError(f"sae-lens module missing expected params: {missing}; "
                            f"has {sorted(sl_keys)}")
    with torch.no_grad():
        for name in ("W_enc", "b_enc", "W_dec", "b_dec"):
            sae_B_sl.state_dict()[name].copy_(init_state[name].to(device))
    _train_loop(sae_B_sl, train_acts_gpu,
                n_steps=args.n_steps, batch_size=args.batch_size,
                lr=args.lr, normalize_every=100, print_every=200,
                seed=args.seed, tag="B")
    sae_B = _convert_saelens_to_topksae(sae_B_sl, d_in=d_in, d_sae=args.d_sae,
                                         k=args.k, device=device)
    fvu_B = _fvu(sae_B, val_acts)
    pathB = args.out_root / "B" / f"sae_resid_mid_s{args.seed}.pt"
    save(sae_B, pathB, layer_hook=args.hook,
         n_train_seqs=int(train_acts_gpu.shape[0]),
         seq_len=args.seq_len, n_steps=args.n_steps,
         batch_size=args.batch_size, lr=args.lr,
         sae_backend="handrolled_init_saelens_module_handrolled_loop")
    print(f"[B] FVU={fvu_B:.4f}  wrote {pathB}", flush=True)
    summary["paths"]["B"] = {"sae_path": str(pathB), "fvu": fvu_B,
                              "label": "handrolled_init_saelens_module"}
    del sae_B_sl
    torch.cuda.empty_cache() if device != "cpu" else None

    # ===================================================================
    # Path C: sae-lens default init + sae-lens forward/backward.
    # ===================================================================
    print(f"\n[C] sae-lens default init + sae-lens module + handrolled loop"
          f"  seed={args.seed}", flush=True)
    torch.manual_seed(args.seed)
    sae_C_sl = _build_saelens_topk_sae(d_in=d_in, d_sae=args.d_sae,
                                        k=args.k, device=device)
    _train_loop(sae_C_sl, train_acts_gpu,
                n_steps=args.n_steps, batch_size=args.batch_size,
                lr=args.lr, normalize_every=100, print_every=200,
                seed=args.seed, tag="C")
    sae_C = _convert_saelens_to_topksae(sae_C_sl, d_in=d_in, d_sae=args.d_sae,
                                         k=args.k, device=device)
    fvu_C = _fvu(sae_C, val_acts)
    pathC = args.out_root / "C" / f"sae_resid_mid_s{args.seed}.pt"
    save(sae_C, pathC, layer_hook=args.hook,
         n_train_seqs=int(train_acts_gpu.shape[0]),
         seq_len=args.seq_len, n_steps=args.n_steps,
         batch_size=args.batch_size, lr=args.lr,
         sae_backend="saelens_init_saelens_module_handrolled_loop")
    print(f"[C] FVU={fvu_C:.4f}  wrote {pathC}", flush=True)
    summary["paths"]["C"] = {"sae_path": str(pathC), "fvu": fvu_C,
                              "label": "saelens_init_saelens_module"}

    # ----- Persist summary. -----
    summary_path = args.out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[done] FVU  A={fvu_A:.4f}  B={fvu_B:.4f}  C={fvu_C:.4f}", flush=True)
    print(f"[done] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
