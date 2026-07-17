"""Measure EM-AFP posterior readout R2 from logits and residual-stream probes.

This analyzes checkpoints produced by experiments/exp_em_afp_train.py.  It
reports two different readouts:

1. behavioral: infer component posterior from next-token logits via tag-marginal
   inversion, then compare to the exact HMM filter;
2. representational: fit Moore-Penrose/least-squares linear probes from
   resid_post activations to exact posterior readouts and report held-out R2.
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


ARTIFACT_ROOT = Path(os.environ.get("EM_AFP_ARTIFACT_ROOT", "/tmp/simplex-em-afp-artifacts/em_afp"))
DEFAULT_OUT = (
    ROOT / "experiment_folders" / "em_afp" / "codex_sol" / "results" / "posterior_r2"
)


def load_analysis_config() -> dict:
    path = os.environ.get("EM_AFP_R2_CONFIG")
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["config_path"] = path
    return cfg


def output_dir(analysis_cfg: dict) -> Path:
    if os.environ.get("EM_AFP_R2_OUT"):
        return Path(os.environ["EM_AFP_R2_OUT"])
    out = analysis_cfg.get("output_dir")
    return Path(out) if out else DEFAULT_OUT


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


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    vals = r2_dims(y, pred)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def r2_dims(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = ((y - pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    out = np.full_like(ss_tot, np.nan, dtype=np.float64)
    ok = ss_tot > 1e-10
    out[ok] = 1.0 - ss_res[ok] / ss_tot[ok]
    return out


def tag_probs_from_token_probs(token_probs: np.ndarray, cfg: em_afp.EMAFPConfig) -> np.ndarray:
    p = token_probs.reshape(*token_probs.shape[:-1], cfg.n_leaves, cfg.d)
    return p.sum(axis=-1)


def posterior_readouts(mu: np.ndarray) -> dict[str, np.ndarray]:
    eps = 1e-9
    md = mu[..., em_afp.LEAF_TO_INDEX["MD"]]
    mo = mu[..., em_afp.LEAF_TO_INDEX["MO"]]
    ad = mu[..., em_afp.LEAF_TO_INDEX["AD"]]
    ao = mu[..., em_afp.LEAF_TO_INDEX["AO"]]
    return {
        "component_posterior": mu,
        "MD_local_misaligned": md[..., None],
        "MO_global_misaligned": mo[..., None],
        "AD_aligned_domain_D": ad[..., None],
        "AO_aligned_domain_O": ao[..., None],
        "M_persona_misaligned": (md + mo)[..., None],
        "A_persona_aligned": (ad + ao)[..., None],
        "D_domain": (md + ad)[..., None],
        "O_domain": (mo + ao)[..., None],
        "interaction_chi": np.log((md * ao + eps) / (mo * ad + eps))[..., None],
    }


def readout_dims(name: str) -> list[str]:
    if name == "component_posterior":
        return list(em_afp.LEAVES)
    return [name]


def flatten_time(x: np.ndarray, start_pos: int) -> np.ndarray:
    return x[:, start_pos:, ...].reshape(-1, *x.shape[2:])


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
    return r2_score(Y[te], pred), r2_dims(Y[te], pred)


def split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123_456 + seed)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    return idx[:cut], idx[cut:]


def analyze_checkpoint(
    path: Path,
    eval_n: int,
    device: str,
    analysis_cfg: dict,
) -> tuple[list[dict], list[dict]]:
    result = torch.load(path, map_location="cpu", weights_only=False)
    seed = int(result["seed"])
    cfg_proc = result["process_config"]
    prior = configured_prior(analysis_cfg)
    if prior is not None:
        cfg_proc = replace(cfg_proc, pi=prior)
    cfg_model = result["model_config"]
    model = TinyGPT(cfg_model)
    model.load_state_dict(result["state_dict"])
    model.to(device)
    model.eval()

    rng = np.random.default_rng(10_000 + seed)
    obs_eval, _, _ = em_afp.gen_em_afp(eval_n, cfg_proc, rng)
    _, mu_eval, _ = em_afp.forward_filter_em_afp(obs_eval, cfg_proc)
    toks = torch.tensor(obs_eval[:, :-1], device=device)
    start_pos = 3 if cfg_proc.seq_len > 8 else 0

    with torch.no_grad():
        logits, resids = model.run_with_resid(toks)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        resid_np = [r.cpu().numpy() for r in resids]

    tag_probs = tag_probs_from_token_probs(probs, cfg_proc)
    mu_beh = em_afp.component_posterior_from_tag_probs(
        tag_probs.reshape(-1, cfg_proc.n_leaves), cfg_proc
    ).reshape(eval_n, cfg_proc.seq_len - 1, cfg_proc.n_leaves)
    mu_true = mu_eval[:, :-1, :]

    true_readouts = posterior_readouts(mu_true)
    beh_readouts = posterior_readouts(mu_beh)
    n_rows = flatten_time(mu_true, start_pos).shape[0]
    tr, te = split_indices(n_rows, seed)

    beh_rows = []
    run_name = analysis_cfg.get("run_name", "checkpoint_prior")
    prior_dict = {name: float(cfg_proc.pi[i]) for i, name in enumerate(em_afp.LEAVES)}
    for name, y_true in true_readouts.items():
        y = flatten_time(y_true, start_pos)
        pred = flatten_time(beh_readouts[name], start_pos)
        dims = r2_dims(y[te], pred[te])
        beh_rows.append(
            {
                "seed": seed,
                "eval_run": run_name,
                "eval_prior": json.dumps(prior_dict, sort_keys=True),
                "readout": name,
                "target_dim": "__mean__",
                "r2": float(np.nanmean(dims)) if np.isfinite(dims).any() else float("nan"),
            }
        )
        for dim, val in zip(readout_dims(name), dims):
            beh_rows.append(
                {
                    "seed": seed,
                    "eval_run": run_name,
                    "eval_prior": json.dumps(prior_dict, sort_keys=True),
                    "readout": name,
                    "target_dim": dim,
                    "r2": float(val),
                }
            )

    probe_rows = []
    for layer, resid in enumerate(resid_np):
        X = flatten_time(resid, start_pos)
        for name, y_true in true_readouts.items():
            y = flatten_time(y_true, start_pos)
            mean_r2, dims = fit_probe_r2(X, y, tr, te)
            probe_rows.append(
                {
                    "seed": seed,
                    "eval_run": run_name,
                    "eval_prior": json.dumps(prior_dict, sort_keys=True),
                    "layer": layer,
                    "readout": name,
                    "target_dim": "__mean__",
                    "r2": mean_r2,
                }
            )
            for dim, val in zip(readout_dims(name), dims):
                probe_rows.append(
                    {
                        "seed": seed,
                        "eval_run": run_name,
                        "eval_prior": json.dumps(prior_dict, sort_keys=True),
                        "layer": layer,
                        "readout": name,
                        "target_dim": dim,
                        "r2": float(val),
                    }
                )

    return beh_rows, probe_rows


def plot_results(behavior_df: pd.DataFrame, probe_df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    scalar = probe_df[probe_df["target_dim"] == "__mean__"].copy()
    scalar = scalar[scalar["readout"] != "component_posterior"]
    pivot = scalar.groupby(["layer", "readout"], as_index=False)["r2"].mean()
    table = pivot.pivot(index="readout", columns="layer", values="r2").sort_index()
    table = table.dropna(how="all")

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
    finite_vals = table.values[np.isfinite(table.values)]
    vmin = min(0.0, float(finite_vals.min())) if finite_vals.size else 0.0
    masked = np.ma.masked_invalid(table.values)
    im = ax.imshow(masked, aspect="auto", vmin=vmin, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels([f"L{c}" for c in table.columns])
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_title("Residual-stream probe $R^2$ by posterior readout")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table.values[i, j]
            label = "n/a" if not np.isfinite(val) else f"{val:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label="held-out $R^2$")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "probe_r2_heatmap.png")
    plt.close(fig)

    comp = probe_df[
        (probe_df["readout"] == "component_posterior") & (probe_df["target_dim"] != "__mean__")
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    for dim, group in comp.groupby("target_dim"):
        mean = group.groupby("layer", as_index=False)["r2"].mean()
        ax.plot(mean["layer"], mean["r2"], marker="o", lw=1.5, label=dim)
    ax.set_title("Component posterior probe $R^2$")
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out $R^2$")
    vals = comp["r2"].to_numpy()
    ymin = min(0.0, float(np.nanmin(vals)) - 0.05) if np.isfinite(vals).any() else 0.0
    ax.set_ylim(ymin, 1.02)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "component_probe_r2_by_layer.png")
    plt.close(fig)

    final_layer = int(probe_df["layer"].max())
    probe_final = probe_df[(probe_df["layer"] == final_layer) & (probe_df["target_dim"] == "__mean__")]
    beh_mean = behavior_df[behavior_df["target_dim"] == "__mean__"]
    cmp_rows = []
    for readout in sorted(set(probe_final["readout"])):
        cmp_rows.append(
            {
                "readout": readout,
                "behavioral_logits": float(beh_mean[beh_mean["readout"] == readout]["r2"].mean()),
                f"probe_layer_{final_layer}": float(
                    probe_final[probe_final["readout"] == readout]["r2"].mean()
                ),
            }
        )
    cmp = pd.DataFrame(cmp_rows).sort_values(f"probe_layer_{final_layer}")
    cmp = cmp[np.isfinite(cmp["behavioral_logits"]) | np.isfinite(cmp[f"probe_layer_{final_layer}"])]
    y = np.arange(len(cmp))
    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=180)
    ax.barh(y - 0.18, cmp["behavioral_logits"], height=0.36, label="logit-implied posterior")
    ax.barh(y + 0.18, cmp[f"probe_layer_{final_layer}"], height=0.36, label=f"resid probe L{final_layer}")
    ax.set_yticks(y)
    ax.set_yticklabels(cmp["readout"])
    vals = np.concatenate(
        [
            cmp["behavioral_logits"].to_numpy(dtype=float),
            cmp[f"probe_layer_{final_layer}"].to_numpy(dtype=float),
        ]
    )
    xmin = min(0.0, float(np.nanmin(vals)) - 0.05) if np.isfinite(vals).any() else 0.0
    ax.set_xlim(xmin, 1.02)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("$R^2$")
    ax.set_title("Behavioral vs representational posterior $R^2$")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "behavior_vs_probe_r2.png")
    plt.close(fig)


def main() -> int:
    analysis_cfg = load_analysis_config()
    eval_n = int(os.environ.get("EM_AFP_R2_EVAL_N", (analysis_cfg.get("eval") or {}).get("n", "4096")))
    device = os.environ.get("BAG_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    global OUT_DIR
    OUT_DIR = output_dir(analysis_cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    behavior_rows = []
    probe_rows = []
    for path in sorted(ARTIFACT_ROOT.glob("seed_*/em_afp_train.pt")):
        b, p = analyze_checkpoint(path, eval_n, device, analysis_cfg)
        behavior_rows.extend(b)
        probe_rows.extend(p)

    behavior_df = pd.DataFrame(behavior_rows)
    probe_df = pd.DataFrame(probe_rows)
    behavior_df.to_csv(OUT_DIR / "behavioral_posterior_r2.csv", index=False)
    probe_df.to_csv(OUT_DIR / "probe_posterior_r2.csv", index=False)

    summary = {
        "run_name": analysis_cfg.get("run_name", "checkpoint_prior"),
        "config_path": analysis_cfg.get("config_path"),
        "eval_n": eval_n,
        "artifact_root": str(ARTIFACT_ROOT),
        "eval_prior": (
            {name: val for name, val in zip(em_afp.LEAVES, configured_prior(analysis_cfg))}
            if configured_prior(analysis_cfg) is not None
            else "checkpoint_prior"
        ),
        "readouts": {
            "component_posterior": "4-vector (MD, MO, AD, AO)",
            "MD_local_misaligned": "mu_MD = T_M tensor T_D",
            "MO_global_misaligned": "mu_MO = T_M tensor T_O",
            "AD_aligned_domain_D": "mu_AD = T_A tensor T_D",
            "AO_aligned_domain_O": "mu_AO = T_A tensor T_O",
            "M_persona_misaligned": "mu_MD + mu_MO",
            "A_persona_aligned": "mu_AD + mu_AO",
            "D_domain": "mu_MD + mu_AD",
            "O_domain": "mu_MO + mu_AO",
            "interaction_chi": "log((mu_MD mu_AO)/(mu_MO mu_AD))",
        },
        "behavioral_mean_r2": behavior_df[behavior_df["target_dim"] == "__mean__"]
        .groupby("readout")["r2"]
        .mean()
        .to_dict(),
        "probe_mean_r2_by_layer": probe_df[probe_df["target_dim"] == "__mean__"]
        .groupby(["layer", "readout"])["r2"]
        .mean()
        .reset_index()
        .to_dict(orient="records"),
    }
    (OUT_DIR / "posterior_r2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    readme = """# Posterior R2 Measurements

These are held-out R2 measurements for the posterior component combinations
listed in `notes/proposals/em_afp/em_afp_proposal.md`.

Eval run: `{run_name}`

Eval prior: `{prior}`

Two readout levels are reported:

- `behavioral_posterior_r2.csv`: infer posterior weights from model next-token
  probabilities via tag-marginal inversion, then compare to the exact HMM
  posterior.
- `probe_posterior_r2.csv`: fit Moore-Penrose/least-squares linear probes from
  `resid_post` activations at each transformer block to the exact HMM posterior
  readouts, then report held-out R2.

Plots:

- `probe_r2_heatmap.png`
- `component_probe_r2_by_layer.png`
- `behavior_vs_probe_r2.png`
""".format(
        run_name=analysis_cfg.get("run_name", "checkpoint_prior"),
        prior=summary["eval_prior"],
    )
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    plot_results(behavior_df, probe_df)
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
