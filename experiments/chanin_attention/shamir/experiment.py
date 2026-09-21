"""Small genuine Shamir HMM, with separate routing and computation controls.

Each episode emits a prefix of randomly ordered tagged shares, a query token,
then the requested secret. We train only at the query -> response transition.
The query token is available to Q, but is excluded from K/V (strict past-only
read); there is no residual query path. All activation directions are fixed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
GEOMETRY = ROOT.parent / "results/recovery_seed2/geometry.pt"


class ShamirHMM:
    """Regenerative finite-state HMM implementing 2-of-2 sharing over GF(5).

    Latent state: three (S,a) pairs, a permutation of the six (stream,x) slots,
    prefix length, current phase/index, and the requested stream. After emitting
    the requested secret, reset independently. This class samples query prefixes
    from that process directly; no future response enters the model inputs.
    """

    p = 5
    max_streams = 3
    query_type = 13
    share_type = 14
    secret_start = 15
    length = 6

    def __init__(self, geometry, streams=3, missing_probability=0.25):
        if streams not in (1, 3):
            raise ValueError("This controlled benchmark supports one or three streams")
        self.f = geometry["features"].float()
        self.sae = nn.functional.normalize(geometry["decoder"].float(), dim=-1)
        self.streams = streams
        self.missing_probability = missing_probability
        self.secret_f = self.f[self.secret_start : self.secret_start + self.p]
        self.mean = self.secret_f.mean(0)
        self.variance = (self.secret_f - self.mean).square().sum(-1).mean().item()
        self.names = (
            [f"stream {i}" for i in range(3)]
            + [f"share x={x}, y={y}" for x in (1, 2) for y in range(self.p)]
            + ["query", "share token"]
            + [f"secret {s}" for s in range(self.p)]
        )

    @torch.no_grad()
    def sample(self, batch, generator, complete=False):
        c, p = self.streams, self.p
        secrets = torch.randint(p, (batch, c), generator=generator)
        slopes = torch.randint(p, (batch, c), generator=generator)  # includes zero
        x = torch.tensor([1, 2])
        values = (secrets[:, :, None] + slopes[:, :, None] * x) % p
        order = torch.rand(batch, 2 * c, generator=generator).argsort(-1)
        stream = order // 2
        role = order % 2
        y = values.reshape(batch, -1).gather(1, order)
        lengths = torch.full((batch,), 2 * c)
        if not complete:
            shortened = (
                torch.rand(batch, generator=generator) < self.missing_probability
            )
            short_lengths = torch.randint(1, 2 * c, (batch,), generator=generator)
            lengths = torch.where(shortened, short_lengths, lengths)
        valid = torch.arange(self.length)[None, :] < lengths[:, None]
        request = torch.randint(c, (batch,), generator=generator)
        z = torch.zeros(batch, self.length, len(self.f))
        rows = torch.arange(batch)[:, None]
        slots = torch.arange(2 * c)[None, :]
        z[rows, slots, stream] = 1
        z[rows, slots, 3 + role * p + y] = 1
        z[:, : 2 * c, self.share_type] = 1
        z *= valid[:, :, None]
        qz = torch.zeros(batch, len(self.f))
        qz[torch.arange(batch), request] = 1
        qz[:, self.query_type] = 1
        relevant = torch.zeros_like(valid)
        relevant[:, : 2 * c] = (stream == request[:, None]) & valid[:, : 2 * c]
        count = relevant.sum(-1)
        target = secrets[torch.arange(batch), request]
        oracle = torch.where((count == 2)[:, None], self.secret_f[target], self.mean)
        return {
            "memory": z @ self.f,
            "query": qz @ self.f,
            "z": z,
            "qz": qz,
            "valid": valid,
            "relevant": relevant,
            "lengths": lengths,
            "request": request,
            "target": target,
            "target_activation": self.secret_f[target],
            "oracle": oracle,
            "count": count,
            "secrets": secrets,
            "requested_shares": values[torch.arange(batch), request],
        }

    def expected_incomplete_fraction(self):
        m = 2 * self.streams
        return self.missing_probability * (
            1 - sum(l * (l - 1) / (m * (m - 1)) for l in range(1, m)) / (m - 1)
        )


class Reader(nn.Module):
    def __init__(self, d=100, readout="linear", routing="full", hidden=256):
        super().__init__()
        self.d, self.routing, self.readout = d, routing, readout
        self.wq = nn.Parameter(
            torch.randn(d, d) / math.sqrt(d), requires_grad=routing == "full"
        )
        self.wk = nn.Parameter(
            torch.randn(d, d) / math.sqrt(d), requires_grad=routing == "full"
        )
        self.wv = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.wo = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.lag_bias = nn.Parameter(torch.zeros(6), requires_grad=routing != "oracle")
        self.decoder = (
            nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, d))
            if readout == "mlp"
            else nn.Identity()
        )

    def forward(self, batch, override=None, details=False):
        mem, q = batch["memory"], batch["query"]
        content = torch.einsum("bd,btd->bt", q @ self.wq, mem @ self.wk) / math.sqrt(
            self.d
        )
        if self.routing != "full":
            content = torch.zeros_like(content)
        if override is not None:
            content = override
        lag = (
            batch["lengths"][:, None] - 1 - torch.arange(mem.shape[1])[None, :]
        ).clamp_min(0)
        scores = content + self.lag_bias[lag]
        a = scores.masked_fill(~batch["valid"], -torch.inf).softmax(-1)
        if self.routing == "oracle" and override is None:
            keep = torch.where(
                (batch["count"] > 0)[:, None], batch["relevant"], batch["valid"]
            )
            a = keep.float() / keep.sum(-1)[:, None]
        pooled = torch.einsum("bt,btd->bd", a, mem @ self.wv) @ self.wo + self.bias
        prediction = self.decoder(pooled)
        if details:
            return prediction, a, pooled, content
        return prediction


@torch.no_grad()
def evaluate(model, data, batches=32, batch_size=256, seed=900001):
    rng = torch.Generator().manual_seed(seed)
    f = data.f
    b = f @ model.wq @ model.wk.T @ f.T / math.sqrt(model.d)
    if model.routing != "full":
        b.zero_()
    dec = data.sae
    sae_b = dec @ model.wq @ model.wk.T @ dec.T / math.sqrt(model.d)
    if model.routing != "full":
        sae_b.zero_()
    rows = {
        key: []
        for key in [
            "count",
            "mse",
            "oracle_error",
            "accuracy",
            "oracle_mse",
            "relevant_mass",
            "zero_qk_mse",
            "zero_qk_accuracy",
            "diagonal_qk_mse",
            "diagonal_qk_accuracy",
            "routing_only_mse",
            "routing_only_accuracy",
            "wrong_query_mse",
        ]
    }
    max_error = 0.0
    for _ in range(batches):
        sample = data.sample(batch_size, rng)
        pred, a, _, content = model(sample, details=True)
        reconstructed = torch.einsum("bi,ij,btj->bt", sample["qz"], b, sample["z"])
        max_error = max(max_error, (content - reconstructed).abs().max().item())
        variants = {
            "zero_qk": torch.zeros_like(content),
            "diagonal_qk": torch.einsum(
                "bi,i,bti->bt", sample["qz"], b.diag(), sample["z"]
            ),
        }
        # Keep query-stream -> memory-stream terms and their actual biases
        # from the query type, dropping share-value and other content scores.
        qz = sample["qz"]
        route = torch.einsum("bi,ij,btj->bt", qz, b[:, :3], sample["z"][:, :, :3])
        variants["routing_only"] = route
        rows["count"].append(sample["count"])
        rows["mse"].append((pred - sample["target_activation"]).square().sum(-1))
        rows["oracle_error"].append((pred - sample["oracle"]).square().sum(-1))
        rows["accuracy"].append(
            ((pred @ data.secret_f.T).argmax(-1) == sample["target"]).float()
        )
        rows["oracle_mse"].append(
            (sample["oracle"] - sample["target_activation"]).square().sum(-1)
        )
        rows["relevant_mass"].append((a * sample["relevant"]).sum(-1))
        for name, c in variants.items():
            y = model(sample, override=c)
            rows[name + "_mse"].append(
                (y - sample["target_activation"]).square().sum(-1)
            )
            rows[name + "_accuracy"].append(
                ((y @ data.secret_f.T).argmax(-1) == sample["target"]).float()
            )
        other = dict(sample)
        request = (sample["request"] + 1) % data.streams
        other_qz = sample["qz"].clone()
        other_qz[:, :3] = 0
        other_qz[torch.arange(batch_size), request] = 1
        other["query"] = other_qz @ f
        other["qz"] = other_qz
        y = model(other)
        rows["wrong_query_mse"].append(
            (y - sample["target_activation"]).square().sum(-1)
        )
    arrays = {k: torch.cat(v) for k, v in rows.items()}
    metrics = {
        "eval_examples": len(arrays["count"]),
        "score_reconstruction_max_error": max_error,
        "bayes_expected_nmse": data.expected_incomplete_fraction(),
        # Legacy key: this is a population bound for ALL streams fully
        # observed, not for conditioning on requested-share count alone.
        "query_blind_complete_nmse_lower_bound": 1 - 1 / data.streams,
        "strata": {},
    }
    assert max_error < 2e-4, max_error
    for label, mask in [("all", torch.ones_like(arrays["count"], dtype=torch.bool))] + [
        (f"shares_{n}", arrays["count"] == n) for n in range(3)
    ]:
        if not mask.any():
            continue
        item = {"n": int(mask.sum())}
        for key, value in arrays.items():
            if key == "count":
                continue
            scale = (
                data.variance if key.endswith("mse") or key == "oracle_error" else 1.0
            )
            item[key] = value[mask].mean().item() / scale
        for intervention in ["zero_qk", "diagonal_qk", "routing_only", "wrong_query"]:
            delta = (
                arrays[intervention + "_mse"][mask] - arrays["mse"][mask]
            ) / data.variance
            item[intervention + "_paired_delta_nmse"] = delta.mean().item()
            item[intervention + "_paired_95ci_halfwidth"] = (
                1.96 * delta.std().item() / math.sqrt(len(delta))
            )
        metrics["strata"][label] = item
    effective = (b[:3, :3] + b[data.query_type, :3][None, :]).numpy()
    effective -= effective.mean(-1, keepdims=True)
    # OV vectors are measured before the MLP; their coordinate gauge is not
    # fixed by the output task when an MLP follows. Probes below address this.
    ov = f @ model.wv @ model.wo
    matrices = {
        "qk": b.numpy(),
        "sae_qk": sae_b.numpy(),
        "effective_stream_scores": effective,
        "ov_vectors": ov.numpy(),
        "lag_bias": model.lag_bias.detach().numpy(),
    }
    return metrics, matrices


@torch.no_grad()
def probes(model, data, train=4096, test=4096):
    """Held-out affine probes of OV output; only fully shared cases."""
    rng = torch.Generator().manual_seed(81019)
    chunks = []
    labels = []
    secret = []
    for _ in range((train + test) // 256):
        batch = data.sample(256, rng, complete=True)
        _, _, h, _ = model(batch, details=True)
        chunks.append(h)
        labels.append(batch["requested_shares"])
        secret.append(batch["target"])
    h = torch.cat(chunks).double()
    h = torch.cat([h, torch.ones(len(h), 1, dtype=h.dtype)], -1)
    shares = torch.cat(labels)
    secret = torch.cat(secret)
    lhs = h[:train].T @ h[:train] + 1e-3 * torch.eye(h.shape[1], dtype=h.dtype)
    result = {}
    for name, y in [
        ("share_x1", shares[:, 0]),
        ("share_x2", shares[:, 1]),
        ("secret", secret),
    ]:
        target = nn.functional.one_hot(y, 5).double()
        weights = torch.linalg.solve(lhs, h[:train].T @ target[:train])
        prediction = h[train:] @ weights
        result[name + "_accuracy"] = (
            (prediction.argmax(-1) == y[train:]).double().mean().item()
        )
        result[name + "_r2"] = (
            1
            - (
                (prediction - target[train:]).square().sum()
                / (target[train:] - target[:train].mean(0)).square().sum()
            ).item()
        )
    result.update({"train_examples": train, "test_examples": test, "ridge": 1e-3})
    return result


def train_cell(args, streams, readout, routing, seed, geometry):
    tag = f"streams{streams}_{readout}_{routing}_seed{seed}"
    out = args.out / tag
    if (out / "metrics.json").exists():
        saved = json.loads((out / "metrics.json").read_text())
        expected = {
            "steps": args.steps,
            "batch": args.batch,
            "lr": args.lr,
            "hidden": args.hidden,
            "geometry_sha256": hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
        }
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError(
                f"Changed configuration for {tag}; use another output directory"
            )
        return saved
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    data = ShamirHMM(geometry, streams=streams)
    model = Reader(
        d=data.f.shape[1], readout=readout, routing=routing, hidden=args.hidden
    )
    initial = copy.deepcopy(model.state_dict())
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(1000 + seed)
    history = []
    start = time.monotonic()
    for step in range(args.steps):
        sample = data.sample(args.batch, generator)
        output = model(sample)
        loss = (output - sample["target_activation"]).square().sum(-1).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        lr = args.lr * (
            1
            if step < args.steps * 0.75
            else 0.15
            + 0.85
            * 0.5
            * (1 + math.cos(math.pi * (step - args.steps * 0.75) / (args.steps * 0.25)))
        )
        for group in optim.param_groups:
            group["lr"] = lr
        if (step + 1) % 1000 == 0 or step + 1 == args.steps:
            with torch.no_grad():
                complete = sample["count"] == 2
                accuracy = (
                    ((output @ data.secret_f.T).argmax(-1) == sample["target"])[
                        complete
                    ]
                    .float()
                    .mean()
                    .item()
                )
            entry = {
                "tag": tag,
                "step": step + 1,
                "loss_nmse": loss.item() / data.variance,
                "complete_accuracy": accuracy,
                "seconds": round(time.monotonic() - start, 1),
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)
    metrics, matrices = evaluate(model, data, batches=args.eval_batches)
    metric = {
        "tag": tag,
        "streams": streams,
        "readout": readout,
        "routing": routing,
        "seed": seed,
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "hidden": args.hidden,
        "geometry_sha256": hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
        "seconds": time.monotonic() - start,
        **metrics,
        "probes": probes(model, data),
    }
    torch.save(
        {"model": model.state_dict(), "initial": initial, "metadata": metric},
        out / "checkpoint.pt",
    )
    np.savez(out / "matrices.npz", **matrices)
    (out / "training.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metric, indent=2) + "\n")
    print(
        json.dumps(
            {
                "done": tag,
                "complete": metric["strata"]["shares_2"],
                "probes": metric["probes"],
            }
        ),
        flush=True,
    )
    return metric


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", type=Path, default=GEOMETRY)
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--streams", type=int, nargs="+", default=[1, 3])
    p.add_argument(
        "--readouts", nargs="+", choices=["linear", "mlp"], default=["linear", "mlp"]
    )
    p.add_argument(
        "--routings",
        nargs="+",
        choices=["full", "position", "oracle"],
        default=["full", "position"],
    )
    p.add_argument("--eval-batches", type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    geometry = torch.load(args.geometry, weights_only=True)
    records = []
    for streams in args.streams:
        for readout in args.readouts:
            for routing in args.routings:
                for seed in args.seeds:
                    records.append(
                        train_cell(args, streams, readout, routing, seed, geometry)
                    )
                    (args.out / "summary.json").write_text(
                        json.dumps(records, indent=2) + "\n"
                    )


if __name__ == "__main__":
    main()
