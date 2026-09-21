"""Reproduce Chanin Fig. 1 recovery using the pinned upstream implementation."""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fra-chanin-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REFERENCE_COMMIT = "d5886b540dc5b9cac4f76e6db2b0cce1b0b7c585"


def cosine_rows(a, b):
    return (
        torch.nn.functional.normalize(a, dim=-1)
        @ torch.nn.functional.normalize(b, dim=-1).T
    )


def matrix_stats(c):
    off = c[~torch.eye(c.shape[0], dtype=torch.bool)]
    return {
        "diagonal_min": c.diag().min().item(),
        "diagonal_mean": c.diag().mean().item(),
        "offdiag_abs_max": off.abs().max().item(),
        "offdiag_rms": off.square().mean().sqrt().item(),
        "offdiag_abs_p99": torch.quantile(off.abs(), 0.99).item(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reference", type=Path, default=Path("/tmp/fra-chanin-reference-20260910")
    )
    p.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "results/recovery_seed2"
    )
    p.add_argument("--samples", type=int, default=15_000_000)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()
    commit = subprocess.check_output(
        ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != REFERENCE_COMMIT:
        raise ValueError(f"Reference revision changed: {commit}")
    sys.path.insert(0, str(args.reference))
    from sae_lens import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
    from sparse_but_wrong.toy_models.get_training_batch import (
        generate_random_correlation_matrix,
        get_training_batch,
    )
    from sparse_but_wrong.toy_models.toy_model import ToyModel
    from sparse_but_wrong.toy_models.train_toy_sae import train_toy_sae
    from sparse_but_wrong.util import cos_sims
    from tqdm import tqdm

    tqdm.__init__ = functools.partialmethod(tqdm.__init__, disable=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    probs = 0.345 * (50 - torch.arange(50) - 1) / 50 + 0.05
    corr = generate_random_correlation_matrix(
        50, correlation_strength_range=(0.3, 0.9), seed=42
    )
    model = ToyModel(num_feats=50, hidden_dim=100)
    # The upstream correlation generator resets the global RNG. Explicitly set
    # the requested SAE/data seed AFTER creating the common reference geometry.
    torch.manual_seed(args.seed)
    generate = functools.partial(
        get_training_batch,
        firing_probabilities=probs,
        std_firing_magnitudes=torch.full_like(probs, 0.15),
        correlation_matrix=corr,
        device=torch.device("cpu"),
    )
    cfg = BatchTopKTrainingSAEConfig(k=11, d_in=100, d_sae=50, device="cpu")
    sae = BatchTopKTrainingSAE(cfg)
    start = time.monotonic()
    count = 0

    def provider(batch_size):
        nonlocal count
        count += batch_size
        if count % (args.batch_size * 2000) == 0:
            print(
                json.dumps(
                    {"samples": count, "seconds": round(time.monotonic() - start, 1)}
                ),
                flush=True,
            )
        return generate(batch_size)

    print(
        json.dumps(
            {
                "start": vars(args)
                | {"reference": str(args.reference), "out": str(args.out)},
                "probability_sum": probs.sum().item(),
                "commit": commit,
            }
        ),
        flush=True,
    )
    train_toy_sae(
        sae,
        model,
        provider,
        training_tokens=args.samples,
        train_batch_size_tokens=args.batch_size,
        device=torch.device("cpu"),
    )
    sae.save_model(str(args.out / "sae"))
    with torch.no_grad():
        features = model.embed.weight.T.detach()
        raw = cosine_rows(sae.W_dec.detach(), features)
        rows, cols = linear_sum_assignment(-raw.numpy())
        order = rows[np.argsort(cols)]
        aligned = raw[order]
        source_cos = cos_sims(sae.W_dec.T, model.embed.weight)
        assert torch.allclose(raw, source_cos, atol=1e-6), (
            "Upstream cosine orientation mismatch"
        )
        torch.manual_seed(100042)
        z = generate(100_000)
        x = model(z)
        rec = sae(x)
        gram = cosine_rows(features, features)
        metrics = {
            "reference_commit": commit,
            "seed": args.seed,
            "samples": count,
            "batch_size": args.batch_size,
            "learning_rate": 3e-4,
            "dimensions": {"features": 50, "activation": 100, "sae": 50, "k": 11},
            "probabilities": probs.tolist(),
            "probability_sum": probs.sum().item(),
            "observed_l0": (z > 0).sum(-1).float().mean().item(),
            "reconstruction_nmse_energy": (
                (rec - x).square().sum() / x.square().sum()
            ).item(),
            "cosine": matrix_stats(aligned),
            "ground_truth_gram": matrix_stats(gram),
            "hungarian_order": order.tolist(),
            "upstream_cosine_max_error": (source_cos - raw).abs().max().item(),
            "elapsed_seconds": time.monotonic() - start,
            "versions": {
                m: importlib.metadata.version(m)
                for m in ["torch", "sae-lens", "numpy", "scipy"]
            },
        }
        metrics["recovery_pass"] = (
            metrics["cosine"]["diagonal_min"] >= 0.99
            and metrics["cosine"]["offdiag_abs_max"] <= 0.05
        )
        np.savez(
            args.out / "cosines.npz",
            raw=raw.numpy(),
            aligned=aligned.numpy(),
            gram=gram.numpy(),
            order=order,
            observed_mask_correlation=torch.corrcoef((z > 0).float().T).numpy(),
            gaussian_correlation=corr.numpy(),
        )
        torch.save(
            {
                "features": features,
                "decoder": sae.W_dec.detach()[order],
                "encoder": sae.W_enc.detach()[:, order],
                "b_enc": sae.b_enc.detach()[order],
                "b_dec": sae.b_dec.detach(),
                "probs": probs,
                "correlation": corr,
            },
            args.out / "geometry.pt",
        )
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
        for ax, a, title in zip(
            axes,
            [gram, raw, aligned],
            [
                "Ground-truth Gram matrix",
                "Learned decoder: original order",
                "Learned decoder: one-to-one match",
            ],
        ):
            im = ax.imshow(
                a.numpy(), vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest"
            )
            ax.set(
                title=title,
                xlabel="Ground-truth feature",
                ylabel="Feature / SAE latent",
            )
        fig.colorbar(im, ax=axes, label="Cosine similarity (unrounded)", shrink=0.8)
        fig.savefig(args.out / "recovery.png", dpi=180)
        fig.savefig(args.out / "recovery.pdf")
        plt.close(fig)
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
