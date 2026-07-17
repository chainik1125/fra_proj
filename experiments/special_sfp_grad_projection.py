"""One-step gradient projection: what does the FIRST MD fine-tuning update move?

The correlated-prior control (results_correlated_prior_control) showed that networks
which demonstrably encode the sector-specific persona x domain interaction (r2_det ~ 0.98)
still exhibit undiminished broad transfer. The surviving hypothesis is optimizer
preference: SGD on MD data moves the global persona coordinate rather than the available
sector-specific coordinate. This script tests that directly at fine-tuning step 0.

Procedure, per (prior, seed):
  1. Pretrain the headline model on the base process (same conventions as
     special_sfp_probe_factorization; shares SPECIAL_SFP_PROBE_PI0).
  2. Fit base Moore-Penrose probes (8 targets incl. det) per layer.
  3. Apply exactly one fine-tuning update to a clone of the model:
       - sgd_accum:   plain SGD step on the mean gradient over GRAD_BATCHES MD batches
       - sgd_quarter: same at lr/4 (finite-difference linearity check: deltas should scale ~1/4)
       - adam_first:  the actual first AdamW step of the FT protocol (lr 2e-3, batch 1)
  4. Measure the induced change in base-probe-decoded coordinates on a fixed
     base-process eval batch: mean delta of decoded (MD, MO, AD, AO, P_M, P_D, chi, det),
     z-scored by each target's std on the eval data. Also restricted to O-domain
     contexts (P_D <= 0.1), where a sector-targeted (MD) update should move nothing
     but a global persona update moves decoded P_M.

Prediction (optimizer-preference account): |z(P_M)| >> |z(det)| in every prior
condition, including on O-contexts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bag_moments import special_sfp
from bag_moments.train import get_device
from experiments.special_sfp_probe_factorization import (
    PI0,
    SKIP_POS,
    TARGET_NAMES,
    batch,
    collect_resids,
    derived_targets,
    headline_lowprior_cfg,
    make_headline_model,
    probe_fit,
    train_steps,
)


SMOKE = os.environ.get("SPECIAL_SFP_GRAD_SMOKE", "0") == "1"
OUT = Path(
    os.environ.get(
        "SPECIAL_SFP_GRAD_OUT",
        ROOT
        / "experiment_folders"
        / "em_afp_simpler_codex_auto"
        / ("results_grad_projection_smoke" if SMOKE else "results_grad_projection"),
    )
)
BASE_STEPS = int(os.environ.get("SPECIAL_SFP_GRAD_BASE_STEPS", "300" if SMOKE else "20000"))
PROBE_N = int(os.environ.get("SPECIAL_SFP_GRAD_PROBE_N", "256" if SMOKE else "4096"))
EVAL_N = int(os.environ.get("SPECIAL_SFP_GRAD_EVAL_N", "128" if SMOKE else "1024"))
GRAD_BATCHES = int(os.environ.get("SPECIAL_SFP_GRAD_BATCHES", "8"))
SGD_LR = float(os.environ.get("SPECIAL_SFP_GRAD_SGD_LR", "1e-3"))
ADAM_LR = float(os.environ.get("SPECIAL_SFP_GRAD_ADAM_LR", "2e-3"))
SEED = int(os.environ.get("SPECIAL_SFP_GRAD_SEED", "0"))


def decode_batch(X: np.ndarray, probe: dict) -> np.ndarray:
    xs = np.concatenate([(X - probe["mean"]) / probe["std"], np.ones((len(X), 1))], axis=1)
    return xs @ probe["coef"]


def ft_loss_on_batch(model, x, y):
    logits = model(x)
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def one_step_variants(model, ft_cfg, device: str) -> dict[str, torch.nn.Module]:
    """Return clones of `model` after one fine-tuning update of each kind."""
    variants: dict[str, torch.nn.Module] = {}

    # Mean gradient over GRAD_BATCHES MD batches (same FT seed stream as the probe runs).
    grad_model = deepcopy(model)
    grad_model.zero_grad()
    for step in range(1, GRAD_BATCHES + 1):
        _, x, y = batch(ft_cfg, 256, 500_000 + SEED * 10_000 + step, device)
        loss = ft_loss_on_batch(grad_model, x, y) / GRAD_BATCHES
        loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in grad_model.named_parameters() if p.grad is not None}

    for name, lr in (("sgd_accum", SGD_LR), ("sgd_quarter", SGD_LR / 4.0)):
        net = deepcopy(model)
        with torch.no_grad():
            for pname, p in net.named_parameters():
                if pname in grads:
                    p.add_(grads[pname], alpha=-lr)
        variants[name] = net

    adam_model = deepcopy(model)
    opt = torch.optim.AdamW(adam_model.parameters(), lr=ADAM_LR)
    _, x, y = batch(ft_cfg, 256, 500_000 + SEED * 10_000 + 1, device)
    loss = ft_loss_on_batch(adam_model, x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    variants["adam_first"] = adam_model
    return variants


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = get_device(os.environ.get("BAG_DEVICE"))
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    cfg = headline_lowprior_cfg(PI0)
    ft_cfg = special_sfp.SpecialSFPConfig(**{**asdict(cfg), "pi": (1.0, 0.0, 0.0, 0.0)})

    probe_obs, probe_x, _ = batch(cfg, PROBE_N, 900_000 + SEED, device)
    _, mu, _, _ = special_sfp.forward_filter(probe_obs[:, :-1], cfg)
    mu_flat = mu[:, SKIP_POS:, :].reshape(-1, 4)
    Y = derived_targets(mu_flat)
    target_std = Y.std(axis=0)

    print(f"pretraining {BASE_STEPS} steps on {device} (pi0={PI0})", flush=True)
    model = make_headline_model(cfg, SEED, device)
    train_steps(model, cfg, 1e-3, BASE_STEPS, 100_000 + SEED * 10_000, device)

    n_layers = model.cfg.n_layers
    rng = np.random.default_rng(700 + SEED)
    resids = collect_resids(model, probe_x)
    probes = []
    for layer in range(n_layers):
        X = resids[layer][:, SKIP_POS:, :].reshape(-1, resids[layer].shape[-1])
        probes.append(probe_fit(X, Y, rng))

    # Fixed eval subset for delta measurement, with O-context and D-context masks.
    eval_x = probe_x[:EVAL_N]
    n_eval_rows = EVAL_N * (probe_x.shape[1] - SKIP_POS)
    mu_eval = mu_flat[:n_eval_rows]
    p_d_eval = mu_eval[:, 0] + mu_eval[:, 2]
    masks = {
        "all": np.ones(n_eval_rows, dtype=bool),
        "Octx": p_d_eval <= 0.1,
        "Dctx": p_d_eval >= 0.9,
    }
    base_resids = collect_resids(model, eval_x)
    base_dec = {}
    for layer in range(n_layers):
        Xl = base_resids[layer][:, SKIP_POS:, :].reshape(-1, base_resids[layer].shape[-1])
        base_dec[layer] = decode_batch(Xl, probes[layer])

    rows = []
    variants = one_step_variants(model, ft_cfg, device)
    for method, net in variants.items():
        new_resids = collect_resids(net, eval_x)
        for layer in range(n_layers):
            Xl = new_resids[layer][:, SKIP_POS:, :].reshape(-1, new_resids[layer].shape[-1])
            dec = decode_batch(Xl, probes[layer])
            delta = dec - base_dec[layer]
            for mask_name, mask in masks.items():
                row = {
                    "pi0": ",".join(f"{v:g}" for v in PI0),
                    "seed": SEED,
                    "method": method,
                    "layer": layer,
                    "context": mask_name,
                    "n_ctx_rows": int(mask.sum()),
                }
                dmean = delta[mask].mean(axis=0)
                for i, name in enumerate(TARGET_NAMES):
                    row[f"delta_{name}"] = float(dmean[i])
                    row[f"z_{name}"] = float(dmean[i] / target_std[i]) if target_std[i] > 1e-9 else float("nan")
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"grad_projection_seed{SEED}.csv", index=False)
    (OUT / f"metadata_seed{SEED}.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "pi0": PI0,
                "seed": SEED,
                "base_steps": BASE_STEPS,
                "probe_n": PROBE_N,
                "eval_n": EVAL_N,
                "grad_batches": GRAD_BATCHES,
                "sgd_lr": SGD_LR,
                "adam_lr": ADAM_LR,
                "device": str(device),
                "target_std": {n: float(s) for n, s in zip(TARGET_NAMES, target_std)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    show = df[(df["layer"] == n_layers - 1)][
        ["method", "context", "z_P_M", "z_P_D", "z_det", "z_MD", "z_MO"]
    ]
    print(show.round(3).to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
