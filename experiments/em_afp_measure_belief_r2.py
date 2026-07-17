"""Measure residual-stream probe R2 onto per-component HMM belief states."""

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


ARTIFACT_ROOT = Path(os.environ.get("EM_AFP_ARTIFACT_ROOT", "/tmp/simplex-em-afp-artifacts/em_afp"))
DEFAULT_OUT = ROOT / "experiment_folders" / "em_afp" / "codex_sol" / "results" / "belief_r2"


def load_analysis_config() -> dict:
    path = os.environ.get("EM_AFP_R2_CONFIG")
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["config_path"] = path
    return cfg


def configured_prior(analysis_cfg: dict) -> tuple[float, float, float, float] | None:
    prior = (analysis_cfg.get("process") or {}).get("prior")
    if prior is None:
        return None
    if isinstance(prior, dict):
        vals = [float(prior[name]) for name in em_afp.LEAVES]
    else:
        vals = [float(x) for x in prior]
    total = sum(vals)
    if total <= 0:
        raise ValueError(f"invalid non-positive prior: {prior}")
    return tuple(v / total for v in vals)


def output_dir(analysis_cfg: dict) -> Path:
    if os.environ.get("EM_AFP_BELIEF_OUT"):
        return Path(os.environ["EM_AFP_BELIEF_OUT"])
    out = analysis_cfg.get("output_dir")
    return Path(out) if out else DEFAULT_OUT


def flatten_time(x: np.ndarray, start_pos: int) -> np.ndarray:
    return x[:, start_pos:, ...].reshape(-1, *x.shape[2:])


def r2_dims(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = ((y - pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    out = np.full_like(ss_tot, np.nan, dtype=np.float64)
    ok = ss_tot > 1e-10
    out[ok] = 1.0 - ss_res[ok] / ss_tot[ok]
    return out


def fit_probe_r2(X: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> tuple[float, np.ndarray]:
    Xtr = X[tr]
    Xte = X[te]
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    Xte = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)
    coef = np.linalg.lstsq(Xtr, Y[tr], rcond=None)[0]
    pred = Xte @ coef
    dims = r2_dims(Y[te], pred)
    return float(np.nanmean(dims)) if np.isfinite(dims).any() else float("nan"), dims


def split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(223_456 + seed)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    return idx[:cut], idx[cut:]


def belief_targets(belief: np.ndarray, mu: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    targets = {}
    eps = 1e-9
    for leaf_i, leaf_name in enumerate(em_afp.LEAVES):
        block = belief[..., leaf_i, :]
        targets[("block_belief", leaf_name)] = block
        targets[("within_component_belief", leaf_name)] = block / np.maximum(
            mu[..., leaf_i, None], eps
        )
    return targets


def analyze_checkpoint(path: Path, eval_n: int, device: str, analysis_cfg: dict) -> list[dict]:
    result = torch.load(path, map_location="cpu", weights_only=False)
    seed = int(result["seed"])
    cfg_proc = result["process_config"]
    prior = configured_prior(analysis_cfg)
    if prior is not None:
        cfg_proc = replace(cfg_proc, pi=prior)

    model = TinyGPT(result["model_config"])
    model.load_state_dict(result["state_dict"])
    model.to(device)
    model.eval()

    rng = np.random.default_rng(30_000 + seed)
    obs_eval, _, _ = em_afp.gen_em_afp(eval_n, cfg_proc, rng)
    belief, mu_leaf, _ = em_afp.forward_filter_em_afp(obs_eval, cfg_proc)
    belief = belief[:, :-1, :, :]
    mu_leaf = mu_leaf[:, :-1, :]
    toks = torch.tensor(obs_eval[:, :-1], device=device)
    start_pos = 3 if cfg_proc.seq_len > 8 else 0

    with torch.no_grad():
        _, resids = model.run_with_resid(toks)
        resid_np = [r.cpu().numpy() for r in resids]

    flat_n = flatten_time(mu_leaf, start_pos).shape[0]
    tr, te = split_indices(flat_n, seed)
    targets = belief_targets(belief, mu_leaf)
    rows = []
    run_name = analysis_cfg.get("run_name", "checkpoint_prior")
    prior_dict = {name: float(cfg_proc.pi[i]) for i, name in enumerate(em_afp.LEAVES)}

    for layer, resid in enumerate(resid_np):
        X = flatten_time(resid, start_pos)
        for (belief_kind, component), target in targets.items():
            Y = flatten_time(target, start_pos)
            mean_r2, dims = fit_probe_r2(X, Y, tr, te)
            rows.append(
                {
                    "seed": seed,
                    "eval_run": run_name,
                    "eval_prior": json.dumps(prior_dict, sort_keys=True),
                    "layer": layer,
                    "belief_kind": belief_kind,
                    "component": component,
                    "target_dim": "__mean__",
                    "r2": mean_r2,
                }
            )
            for dim_i, val in enumerate(dims):
                rows.append(
                    {
                        "seed": seed,
                        "eval_run": run_name,
                        "eval_prior": json.dumps(prior_dict, sort_keys=True),
                        "layer": layer,
                        "belief_kind": belief_kind,
                        "component": component,
                        "target_dim": f"state_{dim_i}",
                        "r2": float(val),
                    }
                )
    return rows


def plot_results(df: pd.DataFrame, out_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    final_layer = int(df["layer"].max())
    final = df[(df["layer"] == final_layer) & (df["target_dim"] == "__mean__")]
    summary = final.groupby(["belief_kind", "component"], as_index=False)["r2"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), dpi=180, sharey=True)
    for ax, kind, title in [
        (axes[0], "block_belief", "Unconditional block belief"),
        (axes[1], "within_component_belief", "Within-component belief"),
    ]:
        sub = summary[summary["belief_kind"] == kind].set_index("component").loc[list(em_afp.LEAVES)]
        ax.barh(np.arange(len(sub)), sub["r2"], color="#ff7f0e")
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub.index)
        ax.set_xlim(0.0, 1.02)
        ax.set_xlabel("$R^2$")
        ax.set_title(title)
    fig.suptitle(f"Per-component belief-state probe $R^2$ at L{final_layer}")
    fig.tight_layout()
    fig.savefig(out_dir / "belief_probe_r2_by_component.png")
    plt.close(fig)

    heat = (
        df[df["target_dim"] == "__mean__"]
        .groupby(["belief_kind", "component", "layer"], as_index=False)["r2"]
        .mean()
    )
    for kind in ["block_belief", "within_component_belief"]:
        table = (
            heat[heat["belief_kind"] == kind]
            .pivot(index="component", columns="layer", values="r2")
            .loc[list(em_afp.LEAVES)]
        )
        fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=180)
        im = ax.imshow(table.values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(len(table.columns)))
        ax.set_xticklabels([f"L{c}" for c in table.columns])
        ax.set_yticks(np.arange(len(table.index)))
        ax.set_yticklabels(table.index)
        ax.set_title(f"{kind} probe $R^2$")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(j, i, f"{table.values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label="$R^2$")
        fig.tight_layout()
        fig.savefig(out_dir / f"{kind}_probe_r2_heatmap.png")
        plt.close(fig)


def main() -> int:
    analysis_cfg = load_analysis_config()
    out_dir = output_dir(analysis_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_n = int(os.environ.get("EM_AFP_BELIEF_EVAL_N", "4096"))
    device = os.environ.get("BAG_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))

    rows = []
    for path in sorted(ARTIFACT_ROOT.glob("seed_*/em_afp_train.pt")):
        rows.extend(analyze_checkpoint(path, eval_n, device, analysis_cfg))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "belief_probe_r2.csv", index=False)
    summary = {
        "run_name": analysis_cfg.get("run_name", "checkpoint_prior"),
        "config_path": analysis_cfg.get("config_path"),
        "eval_n": eval_n,
        "artifact_root": str(ARTIFACT_ROOT),
        "targets": {
            "block_belief": "unconditional belief slice b[component, state]",
            "within_component_belief": "normalized belief slice b[component, state] / mu_component",
        },
        "mean_r2": (
            df[df["target_dim"] == "__mean__"]
            .groupby(["belief_kind", "layer", "component"])["r2"]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    (out_dir / "belief_probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        """# Per-Component Belief-State Probe R2

These are Moore-Penrose/least-squares probes from `resid_post` activations to
the exact HMM filter belief state, split by component.

- `block_belief`: predicts the literal HMM belief slice `b[component, state]`.
- `within_component_belief`: predicts `b[component, state] / mu_component`,
  measuring the within-component hidden-state belief after factoring out
  component posterior mass.
""",
        encoding="utf-8",
    )
    plot_results(df, out_dir)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
