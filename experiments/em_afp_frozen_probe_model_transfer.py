"""Frozen-probe transfer from base EM-AFP models to fine-tuned models."""

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
from experiments.em_afp_frozen_probe_transfer import (
    fit_intercept_only,
    fit_linear,
    predict_linear,
    prior_dict,
    prior_from_config,
    rows_for_prediction,
)
from experiments.em_afp_measure_r2 import flatten_time, posterior_readouts, split_indices


def load_config() -> dict:
    path = os.environ.get(
        "EM_AFP_MODEL_TRANSFER_CONFIG",
        "experiment_folders/em_afp/codex_sol/configs/frozen_probe_base_to_ft_md.yaml",
    )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["config_path"] = path
    return cfg


def load_model(path: Path, device: str) -> tuple[dict, TinyGPT]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyGPT(ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return ckpt, model


def activations(model: TinyGPT, obs_eval: np.ndarray, cfg_proc: em_afp.EMAFPConfig, device: str) -> list[np.ndarray]:
    toks = torch.tensor(obs_eval[:, :-1], device=device)
    start_pos = 3 if cfg_proc.seq_len > 8 else 0
    with torch.no_grad():
        _, resids = model.run_with_resid(toks)
    return [flatten_time(r.cpu().numpy(), start_pos) for r in resids]


def analyze_pair(base_path: Path, target_path: Path, cfg: dict, device: str) -> list[dict]:
    base_ckpt, base_model = load_model(base_path, device)
    target_ckpt, target_model = load_model(target_path, device)
    seed = int(base_ckpt["seed"])
    if int(target_ckpt["seed"]) != seed:
        raise ValueError(f"seed mismatch: {base_path} vs {target_path}")

    eval_n = int((cfg.get("eval") or {}).get("n", 4096))
    pi = prior_from_config(cfg["process"])
    cfg_proc = replace(base_ckpt["process_config"], pi=pi)
    rng = np.random.default_rng(60_000 + seed)
    obs_eval, _, _ = em_afp.gen_em_afp(eval_n, cfg_proc, rng)
    _, mu_eval, _ = em_afp.forward_filter_em_afp(obs_eval, cfg_proc)
    start_pos = 3 if cfg_proc.seq_len > 8 else 0
    readouts = {name: flatten_time(val[:, :-1, ...], start_pos) for name, val in posterior_readouts(mu_eval).items()}

    base_Xs = activations(base_model, obs_eval, cfg_proc, device)
    target_Xs = activations(target_model, obs_eval, cfg_proc, device)
    n = next(iter(readouts.values())).shape[0]
    tr, te = split_indices(n, seed)
    prior = prior_dict(pi)

    rows = []
    for layer, (base_X, target_X) in enumerate(zip(base_Xs, target_Xs)):
        source_preds = {}
        frozen_preds = {}
        intercept_preds = {}
        refit_preds = {}
        for readout, y in readouts.items():
            coef, norm = fit_linear(base_X[tr], y[tr])
            source_preds[readout] = predict_linear(base_X, coef, norm)
            frozen_preds[readout] = predict_linear(target_X, coef, norm)
            coef_intercept = fit_intercept_only(target_X[tr], y[tr], coef, norm)
            intercept_preds[readout] = predict_linear(target_X, coef_intercept, norm)
            refit_coef, refit_norm = fit_linear(target_X[tr], y[tr])
            refit_preds[readout] = predict_linear(target_X, refit_coef, refit_norm)

        rows += rows_for_prediction(
            seed, layer, "base_probe_on_base", "test", readouts, source_preds, te, prior, prior
        )
        rows += rows_for_prediction(
            seed, layer, "frozen_base_probe_on_ft", "test", readouts, frozen_preds, te, prior, prior
        )
        rows += rows_for_prediction(
            seed, layer, "base_probe_ft_intercept_only", "test", readouts, intercept_preds, te, prior, prior
        )
        rows += rows_for_prediction(
            seed, layer, "ft_refit_probe", "test", readouts, refit_preds, te, prior, prior
        )
    return rows


def plot(df: pd.DataFrame, out: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    final_layer = int(df["layer"].max())
    modes = [
        "base_probe_on_base",
        "frozen_base_probe_on_ft",
        "base_probe_ft_intercept_only",
        "ft_refit_probe",
    ]
    labels = {
        "base_probe_on_base": "base fit -> base",
        "frozen_base_probe_on_ft": "frozen base -> FT",
        "base_probe_ft_intercept_only": "base weights + FT bias",
        "ft_refit_probe": "FT refit",
    }
    rows = df[
        (df["layer"] == final_layer)
        & (df["target_dim"] == "__mean__")
        & (df["readout"] != "interaction_chi")
    ]
    mean = rows.groupby(["mode", "readout"], as_index=False)["r2"].mean()
    pivot = mean.pivot(index="readout", columns="mode", values="r2")[modes].sort_values("ft_refit_probe")
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
    ax.set_title(f"Frozen base posterior-probe transfer to FT, layer {final_layer}")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "frozen_base_to_ft_probe_r2.png")
    plt.close(fig)


def main() -> int:
    cfg = load_config()
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    device = os.environ.get("BAG_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    base_root = Path(cfg["source_artifact_root"])
    target_root = Path(cfg["target_artifact_root"])

    rows = []
    for base_path in sorted(base_root.glob("seed_*/em_afp_train.pt")):
        target_path = target_root / base_path.parent.name / "em_afp_train.pt"
        rows.extend(analyze_pair(base_path, target_path, cfg, device))

    df = pd.DataFrame(rows)
    df.to_csv(out / "frozen_base_to_ft_probe_r2.csv", index=False)
    plot(df, out)
    summary = {
        "run_name": cfg["run_name"],
        "config_path": cfg["config_path"],
        "source_artifact_root": str(base_root),
        "target_artifact_root": str(target_root),
        "eval_prior": prior_dict(prior_from_config(cfg["process"])),
        "mean_r2_final_layer": df[
            (df["layer"] == df["layer"].max()) & (df["target_dim"] == "__mean__")
        ]
        .groupby(["mode", "readout"])["r2"]
        .mean()
        .reset_index()
        .to_dict(orient="records"),
    }
    (out / "frozen_base_to_ft_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        """# Frozen Base Probe Transfer To FT

Fits posterior probes on base-model activations and evaluates them on MD-fine-tuned
model activations over the same neutral uniform-prior HMM histories.

Modes:

- `base_probe_on_base`: base probe on base activations.
- `frozen_base_probe_on_ft`: same probe, no changes, on FT activations.
- `base_probe_ft_intercept_only`: base weights frozen; only the bias/intercept is fit on FT activations.
- `ft_refit_probe`: full probe refit on FT activations.
""",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
