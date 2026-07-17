"""Frozen-probe transfer for EM-AFP posterior representations.

Fit Moore-Penrose linear probes on one eval prior and evaluate them on another.
Also tests an intercept-only refit on the target distribution, which isolates
whether the prior change looks like a bias/offset in a shared coordinate system.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(os.environ.get("BAG_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

from bag_moments import em_afp
from bag_moments.model import TinyGPT
from experiments.em_afp_measure_r2 import (
    flatten_time,
    posterior_readouts,
    r2_dims,
    r2_score,
    readout_dims,
    split_indices,
)


ARTIFACT_ROOT = Path(os.environ.get("EM_AFP_ARTIFACT_ROOT", "/tmp/simplex-em-afp-artifacts/em_afp"))


def load_config() -> dict:
    path = os.environ.get(
        "EM_AFP_TRANSFER_CONFIG",
        "experiment_folders/em_afp/codex_sol/configs/frozen_probe_uniform_to_aligned_0p9.yaml",
    )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["config_path"] = path
    return cfg


def prior_from_config(section: dict) -> tuple[float, float, float, float]:
    prior = section["prior"]
    vals = [float(prior[name]) for name in em_afp.LEAVES]
    total = sum(vals)
    if total <= 0:
        raise ValueError(f"invalid prior: {prior}")
    return tuple(v / total for v in vals)


def prior_dict(pi: tuple[float, float, float, float]) -> dict[str, float]:
    return {name: float(pi[i]) for i, name in enumerate(em_afp.LEAVES)}


def make_dataset(
    model: TinyGPT,
    cfg_proc: em_afp.EMAFPConfig,
    seed: int,
    eval_n: int,
    device: str,
    seed_offset: int,
) -> tuple[list[np.ndarray], dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed_offset + seed)
    obs_eval, _, _ = em_afp.gen_em_afp(eval_n, cfg_proc, rng)
    _, mu_eval, _ = em_afp.forward_filter_em_afp(obs_eval, cfg_proc)
    toks = torch.tensor(obs_eval[:, :-1], device=device)
    start_pos = 3 if cfg_proc.seq_len > 8 else 0
    with torch.no_grad():
        _, resids = model.run_with_resid(toks)
    resid_flat = [flatten_time(r.cpu().numpy(), start_pos) for r in resids]
    readouts = {name: flatten_time(val[:, :-1, ...], start_pos) for name, val in posterior_readouts(mu_eval).items()}
    return resid_flat, readouts, flatten_time(mu_eval[:, :-1, :], start_pos)


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mu) / sd, mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def fit_linear(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xs, mu, sd = standardize_fit(X)
    Xaug = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    coef = np.linalg.lstsq(Xaug, Y, rcond=None)[0]
    return coef, np.concatenate([mu, sd], axis=0)


def predict_linear(X: np.ndarray, coef: np.ndarray, norm: np.ndarray) -> np.ndarray:
    mu = norm[0:1]
    sd = norm[1:2]
    Xs = standardize_apply(X, mu, sd)
    Xaug = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    return Xaug @ coef


def fit_intercept_only(X: np.ndarray, Y: np.ndarray, coef: np.ndarray, norm: np.ndarray) -> np.ndarray:
    mu = norm[0:1]
    sd = norm[1:2]
    Xs = standardize_apply(X, mu, sd)
    frozen_w = coef[:-1]
    bias = (Y - Xs @ frozen_w).mean(axis=0, keepdims=True)
    return np.concatenate([frozen_w, bias], axis=0)


def rows_for_prediction(
    seed: int,
    layer: int,
    mode: str,
    split: str,
    true_readouts: dict[str, np.ndarray],
    pred_readouts: dict[str, np.ndarray],
    idx: np.ndarray,
    source_prior: dict[str, float],
    target_prior: dict[str, float],
) -> list[dict]:
    rows = []
    for name, y in true_readouts.items():
        pred = pred_readouts[name]
        dims = r2_dims(y[idx], pred[idx])
        rows.append(
            {
                "seed": seed,
                "layer": layer,
                "mode": mode,
                "split": split,
                "source_prior": json.dumps(source_prior, sort_keys=True),
                "target_prior": json.dumps(target_prior, sort_keys=True),
                "readout": name,
                "target_dim": "__mean__",
                "r2": float(np.nanmean(dims)) if np.isfinite(dims).any() else float("nan"),
            }
        )
        for dim, val in zip(readout_dims(name), dims):
            rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "mode": mode,
                    "split": split,
                    "source_prior": json.dumps(source_prior, sort_keys=True),
                    "target_prior": json.dumps(target_prior, sort_keys=True),
                    "readout": name,
                    "target_dim": dim,
                    "r2": float(val),
                }
            )
    return rows


def analyze_checkpoint(path: Path, cfg: dict, device: str) -> list[dict]:
    result = torch.load(path, map_location="cpu", weights_only=False)
    seed = int(result["seed"])
    eval_n = int((cfg.get("eval") or {}).get("n", 4096))
    source_pi = prior_from_config(cfg["source_process"])
    target_pi = prior_from_config(cfg["target_process"])

    base_proc = result["process_config"]
    cfg_source = replace(base_proc, pi=source_pi)
    cfg_target = replace(base_proc, pi=target_pi)
    model = TinyGPT(result["model_config"])
    model.load_state_dict(result["state_dict"])
    model.to(device)
    model.eval()

    src_Xs, src_readouts, _ = make_dataset(model, cfg_source, seed, eval_n, device, seed_offset=10_000)
    tgt_Xs, tgt_readouts, _ = make_dataset(model, cfg_target, seed, eval_n, device, seed_offset=20_000)
    n = next(iter(src_readouts.values())).shape[0]
    src_tr, src_te = split_indices(n, seed)
    tgt_tr, tgt_te = split_indices(n, 10_000 + seed)

    source_prior = prior_dict(source_pi)
    target_prior = prior_dict(target_pi)
    rows = []
    for layer, (src_X, tgt_X) in enumerate(zip(src_Xs, tgt_Xs)):
        frozen_preds = {}
        intercept_preds = {}
        refit_preds = {}
        source_preds = {}
        for readout, src_y in src_readouts.items():
            tgt_y = tgt_readouts[readout]
            coef, norm = fit_linear(src_X[src_tr], src_y[src_tr])
            source_preds[readout] = predict_linear(src_X, coef, norm)
            frozen_preds[readout] = predict_linear(tgt_X, coef, norm)
            coef_intercept = fit_intercept_only(tgt_X[tgt_tr], tgt_y[tgt_tr], coef, norm)
            intercept_preds[readout] = predict_linear(tgt_X, coef_intercept, norm)
            refit_coef, refit_norm = fit_linear(tgt_X[tgt_tr], tgt_y[tgt_tr])
            refit_preds[readout] = predict_linear(tgt_X, refit_coef, refit_norm)

        rows += rows_for_prediction(
            seed, layer, "source_probe_on_source", "source_test", src_readouts, source_preds, src_te, source_prior, source_prior
        )
        rows += rows_for_prediction(
            seed, layer, "frozen_source_probe_on_target", "target_test", tgt_readouts, frozen_preds, tgt_te, source_prior, target_prior
        )
        rows += rows_for_prediction(
            seed, layer, "source_probe_target_intercept_only", "target_test", tgt_readouts, intercept_preds, tgt_te, source_prior, target_prior
        )
        rows += rows_for_prediction(
            seed, layer, "target_refit_probe", "target_test", tgt_readouts, refit_preds, tgt_te, target_prior, target_prior
        )
    return rows


def plot(rows: pd.DataFrame, out: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    final_layer = int(rows["layer"].max())
    df = rows[
        (rows["layer"] == final_layer)
        & (rows["target_dim"] == "__mean__")
        & (rows["readout"] != "interaction_chi")
    ].copy()
    modes = [
        "source_probe_on_source",
        "frozen_source_probe_on_target",
        "source_probe_target_intercept_only",
        "target_refit_probe",
    ]
    labels = {
        "source_probe_on_source": "uniform fit -> uniform",
        "frozen_source_probe_on_target": "frozen uniform -> aligned",
        "source_probe_target_intercept_only": "uniform weights + aligned bias",
        "target_refit_probe": "aligned refit",
    }
    mean = df.groupby(["mode", "readout"], as_index=False)["r2"].mean()
    pivot = mean.pivot(index="readout", columns="mode", values="r2")
    pivot = pivot[modes].sort_values("target_refit_probe")

    y = np.arange(len(pivot.index))
    height = 0.18
    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=180)
    for i, mode in enumerate(modes):
        ax.barh(y + (i - 1.5) * height, pivot[mode], height=height, label=labels[mode])
    vals = pivot.to_numpy().reshape(-1)
    xmin = min(0.0, float(np.nanmin(vals)) - 0.05) if np.isfinite(vals).any() else 0.0
    ax.set_xlim(xmin, 1.02)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("$R^2$")
    ax.set_title(f"Frozen posterior-probe transfer, layer {final_layer}")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "frozen_probe_transfer_r2.png")
    plt.close(fig)

    comp = rows[
        (rows["readout"] == "component_posterior")
        & (rows["target_dim"] != "__mean__")
        & (rows["layer"] == final_layer)
    ]
    comp_mean = comp.groupby(["mode", "target_dim"], as_index=False)["r2"].mean()
    comp_pivot = comp_mean.pivot(index="target_dim", columns="mode", values="r2")[modes]
    y = np.arange(len(comp_pivot.index))
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    for i, mode in enumerate(modes):
        ax.barh(y + (i - 1.5) * height, comp_pivot[mode], height=height, label=labels[mode])
    vals = comp_pivot.to_numpy().reshape(-1)
    xmin = min(0.0, float(np.nanmin(vals)) - 0.05) if np.isfinite(vals).any() else 0.0
    ax.set_xlim(xmin, 1.02)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(comp_pivot.index)
    ax.set_xlabel("$R^2$")
    ax.set_title(f"Component posterior transfer, layer {final_layer}")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "component_frozen_probe_transfer_r2.png")
    plt.close(fig)


def main() -> int:
    cfg = load_config()
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    device = os.environ.get("BAG_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))

    rows = []
    for path in sorted(ARTIFACT_ROOT.glob("seed_*/em_afp_train.pt")):
        rows.extend(analyze_checkpoint(path, cfg, device))

    df = pd.DataFrame(rows)
    df.to_csv(out / "frozen_probe_transfer_r2.csv", index=False)
    plot(df, out)

    summary = {
        "run_name": cfg["run_name"],
        "config_path": cfg["config_path"],
        "source_prior": prior_dict(prior_from_config(cfg["source_process"])),
        "target_prior": prior_dict(prior_from_config(cfg["target_process"])),
        "eval_n": int((cfg.get("eval") or {}).get("n", 4096)),
        "mean_r2_final_layer": df[
            (df["layer"] == df["layer"].max()) & (df["target_dim"] == "__mean__")
        ]
        .groupby(["mode", "readout"])["r2"]
        .mean()
        .reset_index()
        .to_dict(orient="records"),
    }
    (out / "frozen_probe_transfer_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        """# Frozen Probe Transfer

Fits linear posterior probes on uniform-prior eval activations and evaluates
them on aligned-prior eval activations.

Modes:

- `source_probe_on_source`: probe fit on uniform-prior train split, evaluated on uniform-prior test split.
- `frozen_source_probe_on_target`: same probe, no changes, evaluated on aligned-prior test split.
- `source_probe_target_intercept_only`: source weights frozen; only the bias/intercept is fit on aligned-prior train split.
- `target_refit_probe`: full probe refit on aligned-prior train split.
""",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
