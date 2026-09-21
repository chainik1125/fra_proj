"""Causal, activation-space single-head prediction with exact HMM oracles.

Row-vector convention: x=zF, QK=F W_Q W_K^T F^T/sqrt(h),
OV=F W_V W_O F^+ (source feature x destination feature).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fra-chanin-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class Process:
    def __init__(self, geometry, regime, rho=0.7, hit=1.0, magnitude_std=0.15):
        self.f = geometry["features"].float()
        self.pi = geometry["probs"].float()
        self.regime, self.rho, self.hit, self.sigma = regime, rho, hit, magnitude_std
        self.n = len(self.pi)
        self.chol = torch.linalg.cholesky(geometry["correlation"].float())
        self.threshold = torch.distributions.Normal(0.0, 1.0).icdf(1 - self.pi)
        self.r = (
            0.5 * (1 + math.erf(1 / (self.sigma * math.sqrt(2)))) if self.sigma else 1.0
        )
        phi = (
            math.exp(-0.5 / self.sigma**2) / math.sqrt(2 * math.pi)
            if self.sigma
            else 0.0
        )
        self.mag_mean = self.r + self.sigma * phi
        self.mag_second = (1 + self.sigma**2) * self.r + self.sigma * phi
        self.mean = (
            self.pi * (hit if regime == "markov" else 1) * self.mag_mean @ self.f
        )

    @torch.no_grad()
    def sample(self, batch, length, generator):
        shape = (batch, length, self.n)
        if self.regime == "chanin_iid":
            a = (
                torch.randn(shape, generator=generator) @ self.chol.T > self.threshold
            ).float()
        elif self.regime == "independent_iid":
            a = (torch.rand(shape, generator=generator) < self.pi).float()
        elif self.regime == "markov":
            # A reset redraws from Bernoulli(pi); it need not change the state.
            redraw = torch.rand(shape, generator=generator) < self.pi
            keep = torch.rand(shape, generator=generator) < self.rho
            states = [redraw[:, 0]]
            for t in range(1, length):
                states.append(torch.where(keep[:, t], states[-1], redraw[:, t]))
            a = torch.stack(states, 1).float()
            a = a * (torch.rand(shape, generator=generator) < self.hit)
        else:
            raise ValueError(self.regime)
        z = a * (1 + self.sigma * torch.randn(shape, generator=generator)).relu()
        return z @ self.f, z

    @torch.no_grad()
    def conditional_mean(self, z):
        """E[x_(t+1)|x_0:t], including censoring by clipped magnitudes."""
        if self.regime != "markov":
            return self.mean.expand(z.shape[0], z.shape[1], -1)
        prior = self.pi.expand(z.shape[0], -1)
        means = []
        for t in range(z.shape[1]):
            # p_A=0: nonzero emission identifies state 1. A zero emission
            # remains ambiguous when hit<1 (or magnitude clipping occurs).
            posterior_zero = (
                prior * (1 - self.hit * self.r) / (1 - prior * self.hit * self.r)
            )
            posterior = torch.where(z[:, t] > 0, torch.ones_like(prior), posterior_zero)
            prior = (1 - self.rho) * self.pi + self.rho * posterior
            means.append(prior * self.hit * self.mag_mean @ self.f)
        return torch.stack(means, 1)


class SingleHead(nn.Module):
    def __init__(self, d, length, qk="full", positional=True, features=None):
        super().__init__()
        self.d, self.qk, self.positional = d, qk, positional
        self.wq = nn.Parameter(
            torch.randn(d, d) / math.sqrt(d), requires_grad=qk == "full"
        )
        self.wk = nn.Parameter(
            torch.randn(d, d) / math.sqrt(d), requires_grad=qk == "full"
        )
        self.wv = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.wo = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.lag_bias = nn.Parameter(torch.zeros(length), requires_grad=positional)
        if qk == "diagonal":
            if features is None:
                raise ValueError(
                    "Diagonal-QK control requires ground-truth feature coordinates"
                )
            self.register_buffer("feature_dual", torch.linalg.pinv(features))
            self.qk_diagonal = nn.Parameter(torch.randn(features.shape[0]) * 0.1)
        lag = torch.arange(length)[:, None] - torch.arange(length)[None, :]
        self.register_buffer("lag", lag.clamp_min(0))
        self.register_buffer("future", lag < 0)

    def forward(self, x, content_override=None, return_details=False):
        t = x.shape[1]
        content = (x @ self.wq) @ (x @ self.wk).transpose(-1, -2) / math.sqrt(self.d)
        if self.qk == "zero":
            content = torch.zeros_like(content)
        elif self.qk == "diagonal":
            z = x @ self.feature_dual
            content = (z * self.qk_diagonal) @ z.transpose(-1, -2)
        if content_override is not None:
            content = content_override
        scores = content + self.lag_bias[self.lag[:t, :t]]
        attn = scores.masked_fill(self.future[:t, :t], -torch.inf).softmax(-1)
        pred = attn @ (x @ self.wv) @ self.wo + self.bias
        return (pred, attn, content) if return_details else pred


def squared_offdiag_fraction(m):
    off = m - torch.diag_embed(m.diag())
    return (off.square().sum() / m.square().sum().clamp_min(1e-30)).item()


def offdiagonal_fit(b):
    """Post-hoc descriptive fits; these do not alter training or feature order."""
    n = b.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    entries = b[mask]
    mean = entries.mean()
    background = torch.diag(b.diag()) + mean * mask
    ij = mask.nonzero().numpy()
    design = np.zeros((len(ij), 2 * n))
    design[np.arange(len(ij)), ij[:, 0]] = 1
    design[np.arange(len(ij)), n + ij[:, 1]] = 1
    target = entries.detach().numpy()
    fit = design @ np.linalg.lstsq(design, target, rcond=None)[0]
    energy = float(target @ target)
    return background, {
        "qk_offdiag_mean": mean.item(),
        "qk_offdiag_std": entries.std().item(),
        "qk_offdiag_negative_fraction": (entries < 0).float().mean().item(),
        "qk_offdiag_constant_residual_energy_fraction": float(
            np.square(target - target.mean()).sum() / energy
        )
        if energy > 1e-20
        else None,
        "qk_offdiag_row_column_residual_energy_fraction": float(
            np.square(target - fit).sum() / energy
        )
        if energy > 1e-20
        else None,
    }


@torch.no_grad()
def evaluate(model, process, geometry, seed=9000, batches=16, batch=64, length=16):
    f = process.f
    dual = torch.linalg.pinv(f)
    decoder = geometry["decoder"].float()
    normal_decoder = nn.functional.normalize(decoder, dim=-1)
    b = f @ model.wq @ model.wk.T @ f.T / math.sqrt(model.d)
    if model.qk == "zero":
        b.zero_()
    elif model.qk == "diagonal":
        b = torch.diag(model.qk_diagonal)
    background, structure = offdiagonal_fit(b)
    ov = f @ model.wv @ model.wo @ dual
    sae_b = (
        normal_decoder @ model.wq @ model.wk.T @ normal_decoder.T / math.sqrt(model.d)
    )
    if model.qk == "zero":
        sae_b.zero_()
    elif model.qk == "diagonal":
        coordinates = normal_decoder @ dual
        sae_b = coordinates @ b @ coordinates.T
    sae_ov = normal_decoder @ model.wv @ model.wo @ torch.linalg.pinv(normal_decoder)
    emission_pi = process.pi * (process.hit if process.regime == "markov" else 1.0)
    variance = (
        emission_pi * process.mag_second - (emission_pi * process.mag_mean).square()
    )
    slope = (
        process.rho
        * process.hit**2
        * process.mag_mean**2
        * process.pi
        * (1 - process.pi)
        / variance
        if process.regime == "markov"
        else torch.zeros_like(process.pi)
    )
    totals = {
        key: 0.0
        for key in [
            "model",
            "bayes",
            "mean",
            "last",
            "linear_current",
            "oracle_error",
            "zero_qk",
            "diagonal_qk",
            "identity_qk",
            "background_qk",
            "background_qk_oracle_error",
            "zero_qk_oracle_error",
            "diagonal_qk_oracle_error",
            "identity_qk_oracle_error",
            "zero_qk_delta_se_sum",
            "zero_qk_delta_se_sumsq",
            "diagonal_qk_delta_se_sum",
            "diagonal_qk_delta_se_sumsq",
            "attention_self",
            "attention_entropy",
            "score_centered_square",
            "offdiag_centered_square",
            "offdiag_fra_abs",
            "diagonal_fra_abs",
        ]
    }
    rng = torch.Generator().manual_seed(seed)
    exact_error = 0.0
    seq_count = 0
    for _ in range(batches):
        xall, zall = process.sample(batch, length + 1, rng)
        x, y, z = xall[:, :-1], xall[:, 1:], zall[:, :-1]
        oracle = process.conditional_mean(z)
        pred, a, content = model(x, return_details=True)
        variants = {
            "model": pred,
            "bayes": oracle,
            "mean": process.mean,
            "last": x,
            "linear_current": (z * slope + emission_pi * process.mag_mean * (1 - slope))
            @ f,
        }
        feature_content = z @ b @ z.transpose(-1, -2)
        exact_error = max(exact_error, (feature_content - content).abs().max().item())
        alternatives = {
            "zero_qk": torch.zeros_like(content),
            "diagonal_qk": z @ torch.diag(b.diag()) @ z.transpose(-1, -2),
            "background_qk": z @ background @ z.transpose(-1, -2),
            "identity_qk": z
            @ (torch.eye(process.n) * b.diag().mean())
            @ z.transpose(-1, -2),
        }
        for name, c in alternatives.items():
            variants[name] = model(x, content_override=c)
            totals[name + "_oracle_error"] += (
                (variants[name] - oracle).square().sum(-1).mean().item()
            )
        for name, yhat in variants.items():
            totals[name] += (yhat - y).square().sum(-1).mean().item()
        totals["oracle_error"] += (pred - oracle).square().sum(-1).mean().item()
        base = (pred - y).square().sum(-1).mean(-1)
        for name in ["zero_qk", "diagonal_qk"]:
            delta = (variants[name] - y).square().sum(-1).mean(-1) - base
            totals[name + "_delta_se_sum"] += delta.sum().item()
            totals[name + "_delta_se_sumsq"] += delta.square().sum().item()
        seq_count += batch
        totals["attention_self"] += a.diagonal(dim1=-2, dim2=-1).mean().item()
        totals["attention_entropy"] += (
            -(a * a.clamp_min(1e-30).log()).sum(-1).mean().item()
        )
        causal = ~model.future[:length, :length]
        denom = causal.sum(-1)
        off = feature_content - alternatives["diagonal_qk"]
        for c, key in [
            (content, "score_centered_square"),
            (off, "offdiag_centered_square"),
        ]:
            center = (c * causal).sum(-1) / denom
            totals[key] += (
                ((c - center[:, :, None]) * causal).square().sum()
                / causal.sum()
                / batch
            ).item()
        # Absolute feature-pair FRA mass: z>=0, so abs acts factorize exactly.
        diagb = torch.diag(b.diag())
        totals["offdiag_fra_abs"] += (
            ((z @ (b - diagb).abs() @ z.transpose(-1, -2)) * causal).sum().item()
            / batch
            / causal.sum().item()
        )
        totals["diagonal_fra_abs"] += (
            ((z @ diagb.abs() @ z.transpose(-1, -2)) * causal).sum().item()
            / batch
            / causal.sum().item()
        )
    scale = totals["mean"] / batches
    metrics = {
        k + "_nmse_variance": totals[k] / batches / scale
        for k in [
            "model",
            "bayes",
            "mean",
            "last",
            "linear_current",
            "oracle_error",
            "zero_qk",
            "diagonal_qk",
            "identity_qk",
            "background_qk",
            "background_qk_oracle_error",
            "zero_qk_oracle_error",
            "diagonal_qk_oracle_error",
            "identity_qk_oracle_error",
        ]
    }
    metrics.update(structure)
    metrics.update(
        {
            "qk_offdiag_energy_fraction": squared_offdiag_fraction(b),
            "qk_frobenius": b.norm().item(),
            "qk_diagonal_mean": b.diag().mean().item(),
            "qk_diagonal_std": b.diag().std().item(),
            "ov_offdiag_energy_fraction": squared_offdiag_fraction(ov),
            "ov_diagonal_mean": ov.diag().mean().item(),
            "ov_diagonal_std": ov.diag().std().item(),
            "ov_frobenius": ov.norm().item(),
            "fra_score_reconstruction_max_error": exact_error,
            "ov_current_linear_reference_diagonal_mean": slope.mean().item(),
            "ov_current_linear_reference_diagonal_rmse": (ov.diag() - slope)
            .square()
            .mean()
            .sqrt()
            .item(),
            "attention_self": totals["attention_self"] / batches,
            "attention_entropy": totals["attention_entropy"] / batches,
            "qk_centered_score_rms": math.sqrt(
                totals["score_centered_square"] / batches
            ),
            "offdiag_centered_score_rms": math.sqrt(
                totals["offdiag_centered_square"] / batches
            ),
            "offdiag_absolute_fra_mass_fraction": totals["offdiag_fra_abs"]
            / max(1e-30, totals["offdiag_fra_abs"] + totals["diagonal_fra_abs"]),
            "eval_sequences": seq_count,
            "eval_seed": seed,
        }
    )
    for name in ["zero_qk", "diagonal_qk"]:
        mean = totals[name + "_delta_se_sum"] / seq_count
        var = (
            max(0.0, totals[name + "_delta_se_sumsq"] / seq_count - mean * mean)
            * seq_count
            / (seq_count - 1)
        )
        metrics[name + "_paired_delta_nmse"] = mean / scale
        metrics[name + "_paired_delta_95ci_halfwidth"] = (
            1.96 * math.sqrt(var / seq_count) / scale
        )
    assert exact_error < 2e-4, f"FRA score decomposition failed: {exact_error}"
    return metrics, {
        "qk": b.numpy(),
        "ov": ov.numpy(),
        "sae_qk": sae_b.numpy(),
        "sae_ov": sae_ov.numpy(),
        "current_linear_slopes": slope.numpy(),
        "lag_bias": model.lag_bias.detach().numpy(),
    }


def run(args):
    torch.set_num_threads(args.threads)
    geometry = torch.load(args.geometry, weights_only=True)
    recovery = json.loads(args.geometry.with_name("metrics.json").read_text())
    if not recovery["recovery_pass"]:
        raise ValueError(
            "The supplied SAE has not passed the feature recovery criterion"
        )
    geometry_hash = hashlib.sha256(args.geometry.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = [("chanin_iid", 0.0, 1.0), ("independent_iid", 0.0, 1.0)] + [
        ("markov", rho, 1.0) for rho in args.rhos
    ]
    if args.noisy:
        cases.extend(("markov", rho, 0.625) for rho in args.rhos)
    if args.regimes:
        cases = [c for c in cases if c[0] in args.regimes]
    records = []
    for regime, rho, hit in cases:
        for seed in args.seeds:
            for qk in args.qk_modes:
                tag = f"{regime}_rho{rho:g}_hit{hit:g}_{qk}_seed{seed}"
                target = args.out / tag
                if (target / "metrics.json").exists():
                    saved = json.loads((target / "metrics.json").read_text())
                    expected = {
                        "steps": args.steps,
                        "batch_size": args.batch,
                        "length": args.length,
                        "lr": args.lr,
                        "positional_bias": not args.no_position,
                        "geometry_sha256": geometry_hash,
                    }
                    if any(saved.get(k) != v for k, v in expected.items()):
                        raise ValueError(
                            f"Resume configuration differs in {target}; use a new output directory"
                        )
                    records.append(saved)
                    continue
                target.mkdir(parents=True, exist_ok=True)
                torch.manual_seed(seed)
                process = Process(geometry, regime, rho=rho, hit=hit)
                model = SingleHead(
                    process.f.shape[1],
                    args.length,
                    qk=qk,
                    positional=not args.no_position,
                    features=process.f,
                )
                # All weights are learned from scratch; no identity or oracle initialization.
                initial = copy.deepcopy(model.state_dict())
                optim = torch.optim.Adam(model.parameters(), lr=args.lr)
                generator = torch.Generator().manual_seed(1000 + seed)
                history = []
                start = time.monotonic()
                for step in range(args.steps):
                    x, z = process.sample(args.batch, args.length + 1, generator)
                    yhat = model(x[:, :-1])
                    loss = (yhat - x[:, 1:]).square().sum(-1).mean()
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    # Decay only the final half of training; preserve a small noise floor.
                    lr = args.lr * (
                        1
                        if step < args.steps // 2
                        else 0.1
                        + 0.9
                        * 0.5
                        * (
                            1
                            + math.cos(
                                math.pi * (step - args.steps // 2) / (args.steps / 2)
                            )
                        )
                    )
                    for group in optim.param_groups:
                        group["lr"] = lr
                    if (step + 1) % 500 == 0 or step + 1 == args.steps:
                        entry = {
                            "tag": tag,
                            "step": step + 1,
                            "sample_loss": loss.item(),
                            "seconds": round(time.monotonic() - start, 1),
                        }
                        history.append(entry)
                        print(json.dumps(entry), flush=True)
                metrics, matrices = evaluate(
                    model,
                    process,
                    geometry,
                    batches=args.eval_batches,
                    length=args.length,
                )
                record = {
                    "tag": tag,
                    "regime": regime,
                    "rho": rho,
                    "hit": hit,
                    "seed": seed,
                    "qk_mode": qk,
                    "steps": args.steps,
                    "batch_size": args.batch,
                    "length": args.length,
                    "lr": args.lr,
                    "geometry_sha256": geometry_hash,
                    "recovery_seed": recovery["seed"],
                    "positional_bias": not args.no_position,
                    "seconds": time.monotonic() - start,
                    **metrics,
                }
                (target / "training.json").write_text(
                    json.dumps(history, indent=2) + "\n"
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "initial": initial,
                        "metadata": record,
                    },
                    target / "checkpoint.pt",
                )
                np.savez(target / "matrices.npz", **matrices)
                # Completion marker is written only after the artifacts exist.
                (target / "metrics.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
                records.append(record)
                (args.out / "summary.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
                print(json.dumps(record), flush=True)
    plot(args.out, records)


def plot(out, records):
    representatives = [
        r for r in records if r["seed"] == records[0]["seed"] and r["qk_mode"] == "full"
    ]
    if not representatives:
        return
    fig, axes = plt.subplots(
        len(representatives),
        4,
        figsize=(15, 3.4 * len(representatives)),
        squeeze=False,
        layout="constrained",
    )
    for row, r in enumerate(representatives):
        m = np.load(out / r["tag"] / "matrices.npz")
        for col, key in enumerate(["qk", "ov", "sae_qk", "sae_ov"]):
            a = m[key]
            vmax = max(abs(a).max(), 1e-6)
            im = axes[row, col].imshow(
                a, vmin=-vmax, vmax=vmax, cmap="RdBu_r", interpolation="nearest"
            )
            axes[row, col].set_title(
                f"{r['regime']}, rho={r['rho']}, hit={r['hit']}\n{key}"
            )
            axes[row, col].set_xlabel(
                "Key feature" if "qk" in key else "Output feature"
            )
            axes[row, col].set_ylabel(
                "Query feature" if "qk" in key else "Input feature"
            )
            fig.colorbar(im, ax=axes[row, col], shrink=0.75)
    fig.savefig(out / "feature_matrices.png", dpi=140)
    fig.savefig(out / "feature_matrices.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--geometry",
        type=Path,
        default=Path(__file__).parent / "results/recovery_seed2/geometry.pt",
    )
    p.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "results/attention"
    )
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--length", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--no-position", action="store_true")
    p.add_argument("--noisy", action="store_true")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.7])
    p.add_argument(
        "--regimes", nargs="+", choices=["chanin_iid", "independent_iid", "markov"]
    )
    p.add_argument(
        "--qk-modes",
        nargs="+",
        choices=["full", "zero", "diagonal"],
        default=["full", "zero"],
    )
    run(p.parse_args())
