"""Run the hard symmetric special-state SFP experiment."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bag_moments import special_sfp
from bag_moments.model import GPTConfig, TinyGPT
from bag_moments.train import get_device


OUT = Path(
    os.environ.get(
        "SPECIAL_SFP_OUT",
        ROOT / "experiment_folders" / "em_afp_simpler_codex" / "results",
    )
)
ARTIFACTS = Path(os.environ.get("SPECIAL_SFP_ARTIFACTS", "/tmp/simplex-special-sfp-artifacts"))


def make_cfg() -> special_sfp.SpecialSFPConfig:
    def opt_float(name: str):
        val = os.environ.get(name)
        return None if val is None or val == "" else float(val)

    return special_sfp.SpecialSFPConfig(
        p=float(os.environ.get("SPECIAL_SFP_P", "0.5")),
        epsilon=float(os.environ.get("SPECIAL_SFP_EPSILON", "0.02")),
        p_s=float(os.environ.get("SPECIAL_SFP_P_S", "0.8")),
        alpha_wrong_special=float(os.environ.get("SPECIAL_SFP_ALPHA_WRONG", "0.0")),
        p_persona=opt_float("SPECIAL_SFP_P_PERSONA"),
        p_domain=opt_float("SPECIAL_SFP_P_DOMAIN"),
        epsilon_persona=opt_float("SPECIAL_SFP_EPSILON_PERSONA"),
        epsilon_domain=opt_float("SPECIAL_SFP_EPSILON_DOMAIN"),
        p_s_persona=opt_float("SPECIAL_SFP_P_S_PERSONA"),
        p_s_domain=opt_float("SPECIAL_SFP_P_S_DOMAIN"),
        seq_len=int(os.environ.get("SPECIAL_SFP_SEQ_LEN", "64")),
    )


def make_model(cfg: special_sfp.SpecialSFPConfig, device: str, seed: int) -> TinyGPT:
    torch.manual_seed(seed)
    model_cfg = GPTConfig(
        vocab_size=cfg.vocab_size,
        n_ctx=cfg.seq_len,
        d_model=int(os.environ.get("SPECIAL_SFP_D_MODEL", "64")),
        n_heads=int(os.environ.get("SPECIAL_SFP_N_HEADS", "2")),
        n_layers=int(os.environ.get("SPECIAL_SFP_N_LAYERS", "2")),
        d_mlp=int(os.environ.get("SPECIAL_SFP_D_MLP", "256")),
    )
    model = TinyGPT(model_cfg)
    model.to(device)
    return model


def batch(cfg: special_sfp.SpecialSFPConfig, batch_size: int, seed: int, device: str):
    rng = np.random.default_rng(seed)
    obs, _ = special_sfp.gen_special_sfp(batch_size, cfg, rng)
    x = torch.tensor(obs[:, :-1], dtype=torch.long, device=device)
    y = torch.tensor(obs[:, 1:], dtype=torch.long, device=device)
    return obs, x, y


def optimal_loss(obs: np.ndarray, cfg: special_sfp.SpecialSFPConfig) -> float:
    _, _, next_p, _ = special_sfp.forward_filter(obs[:, :-1], cfg)
    labels = obs[:, 1:]
    p = np.take_along_axis(next_p, labels[:, :, None], axis=2)[:, :, 0]
    return float((-np.log(np.maximum(p, 1e-12))).mean())


def eval_model(model: TinyGPT, obs: np.ndarray, cfg: special_sfp.SpecialSFPConfig, device: str):
    x = torch.tensor(obs[:, :-1], dtype=torch.long, device=device)
    y = torch.tensor(obs[:, 1:], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        ).item()
    return loss


def train_base(cfg: special_sfp.SpecialSFPConfig, device: str) -> tuple[TinyGPT, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    seed = int(os.environ.get("SPECIAL_SFP_SEED", "0"))
    steps = int(os.environ.get("SPECIAL_SFP_BASE_STEPS", "2000"))
    batch_size = int(os.environ.get("SPECIAL_SFP_BATCH", "256"))
    lr = float(os.environ.get("SPECIAL_SFP_LR", "1e-3"))
    eval_every = int(os.environ.get("SPECIAL_SFP_EVAL_EVERY", "50"))
    model = make_model(cfg, device, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    eval_obs, _, _ = batch(cfg, 4096, 999_000, device)
    opt_ce = optimal_loss(eval_obs, cfg)
    random_ce = float(np.log(cfg.vocab_size))
    history = []
    t0 = time.time()

    for step in range(1, steps + 1):
        _, x, y = batch(cfg, batch_size, 10_000 + step, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        row = {"step": step, "train_loss": float(loss.detach().cpu())}
        if step % eval_every == 0 or step == 1:
            row["eval_loss"] = eval_model(model, eval_obs, cfg, device)
            print(f"base step {step}: train={row['train_loss']:.4f} eval={row['eval_loss']:.4f}")
        history.append(row)

    ckpt = {
        "seed": seed,
        "process_config": cfg,
        "model_config": model.cfg,
        "state_dict": model.state_dict(),
        "history": history,
        "optimal_eval_loss": opt_ce,
        "random_loss": random_ce,
        "elapsed_s": time.time() - t0,
    }
    torch.save(ckpt, ARTIFACTS / "base.pt")
    pd.DataFrame(history).to_csv(OUT / "base_training_history.csv", index=False)
    return model, ckpt


def r2_and_mse(y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ss_res = ((y - pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = np.full(y.shape[1], np.nan, dtype=np.float64)
    ok = ss_tot > 1e-12
    r2[ok] = 1.0 - ss_res[ok] / ss_tot[ok]
    mse = ((y - pred) ** 2).mean(axis=0)
    return r2, mse


def fit_mp_probe(X: np.ndarray, Y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray):
    Xtr = X[train_idx]
    Xte = X[test_idx]
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    Xte = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)
    coef = np.linalg.lstsq(Xtr, Y[train_idx], rcond=None)[0]
    pred = Xte @ coef
    r2, mse = r2_and_mse(Y[test_idx], pred)
    return r2, mse


def evaluate_probes(model: TinyGPT, cfg: special_sfp.SpecialSFPConfig, device: str, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_n = int(os.environ.get("SPECIAL_SFP_PROBE_N", "4096"))
    obs, _, _ = batch(cfg, eval_n, 777_000, device)
    _, mu, _, _ = special_sfp.forward_filter(obs[:, :-1], cfg)
    x = torch.tensor(obs[:, :-1], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, resids = model.run_with_resid(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        resid_np = [r.cpu().numpy() for r in resids]

    start_pos = 3
    Y = mu[:, start_pos:, :].reshape(-1, cfg.n_leaves)
    logits_X = probs[:, start_pos:, :].reshape(-1, cfg.vocab_size)
    rng = np.random.default_rng(123)
    idx = rng.permutation(Y.shape[0])
    cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]

    rows = []
    r2, mse = fit_mp_probe(logits_X, Y, tr, te)
    for i, leaf in enumerate(special_sfp.LEAVES):
        rows.append({"run": tag, "source": "logit_probs", "layer": -1, "component": leaf, "r2": r2[i], "mse": mse[i]})

    for layer, resid in enumerate(resid_np):
        X = resid[:, start_pos:, :].reshape(-1, resid.shape[-1])
        r2, mse = fit_mp_probe(X, Y, tr, te)
        for i, leaf in enumerate(special_sfp.LEAVES):
            rows.append({"run": tag, "source": "resid_mp", "layer": layer, "component": leaf, "r2": r2[i], "mse": mse[i]})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{tag}_posterior_probe_metrics.csv", index=False)
    return df


def finetune_md(base_ckpt: dict, cfg: special_sfp.SpecialSFPConfig, device: str) -> list[dict]:
    model = make_model(cfg, device, int(base_ckpt["seed"]))
    model.load_state_dict(base_ckpt["state_dict"])
    ft_cfg = special_sfp.md_config(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get("SPECIAL_SFP_FT_LR", "3e-4")))
    batch_size = int(os.environ.get("SPECIAL_SFP_FT_BATCH", "256"))
    checkpoints = {1, 5, 10, 20, 50, 100}
    rows = []
    ckpt_dir = ARTIFACTS / "ft_md"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, 101):
        _, x, y = batch(ft_cfg, batch_size, 900_000 + step, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        rows.append({"step": step, "ft_loss": float(loss.detach().cpu())})
        if step in checkpoints:
            torch.save(
                {
                    "seed": base_ckpt["seed"],
                    "step": step,
                    "process_config": cfg,
                    "ft_process_config": ft_cfg,
                    "model_config": model.cfg,
                    "state_dict": model.state_dict(),
                },
                ckpt_dir / f"ft_step_{step}.pt",
            )
            print(f"saved ft step {step}: loss={rows[-1]['ft_loss']:.4f}")
    pd.DataFrame(rows).to_csv(OUT / "ft_md_history.csv", index=False)
    return rows


def load_model_from_ckpt(path: Path, device: str) -> tuple[TinyGPT, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyGPT(ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def prompt_odds(model: TinyGPT, device: str) -> dict:
    rows = []
    for domain, numerator, denominator in [("D", "MD", "AD"), ("O", "MO", "AO")]:
        prompt = torch.tensor([special_sfp.prompt_token(domain)], dtype=torch.long, device=device)
        probs = torch.softmax(model(prompt)[:, -1, :], dim=-1).cpu().numpy()
        p_sm, p_sa, odds = special_sfp.persona_special_odds_from_probs(probs)
        rows.append(
            {
                "domain_prompt": domain,
                "ratio_name": f"P({numerator}|{domain})/P({denominator}|{domain})",
                "p_next_S_M": float(p_sm[0]),
                "p_next_S_A": float(p_sa[0]),
                "misalignment_odds": float(odds[0]),
                "log_misalignment_odds": float(np.log(odds[0])),
            }
        )
    return rows


def evaluate_ft_odds(base_model: TinyGPT, device: str):
    rows = []
    for row in prompt_odds(base_model, device):
        row["checkpoint"] = "base"
        row["step"] = 0
        rows.append(row)
    for path in sorted((ARTIFACTS / "ft_md").glob("ft_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1])):
        model, ckpt = load_model_from_ckpt(path, device)
        for row in prompt_odds(model, device):
            row["checkpoint"] = path.stem
            row["step"] = int(ckpt["step"])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ft_prompt_misalignment_odds.csv", index=False)
    return df


def make_plots(base_ckpt: dict, probe_df: pd.DataFrame, odds_df: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")
    hist = pd.DataFrame(base_ckpt["history"])
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=180)
    ax.plot(hist["step"], hist["train_loss"], label="train loss", lw=1.2, alpha=0.8)
    eval_hist = hist.dropna(subset=["eval_loss"])
    ax.plot(eval_hist["step"], eval_hist["eval_loss"], label="eval loss", marker="o", ms=2.5, lw=1.4)
    ax.axhline(base_ckpt["optimal_eval_loss"], color="black", ls="--", lw=1.2, label="Bayes optimal eval")
    ax.axhline(base_ckpt["random_loss"], color="gray", ls=":", lw=1.2, label="random baseline")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("cross entropy")
    ax.set_title("Special-state SFP base training")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "base_loss_curve.png")
    plt.close(fig)

    final_layer = int(probe_df[probe_df["source"] == "resid_mp"]["layer"].max())
    show = probe_df[
        ((probe_df["source"] == "logit_probs") & (probe_df["layer"] == -1))
        | ((probe_df["source"] == "resid_mp") & (probe_df["layer"] == final_layer))
    ].copy()
    labels = {"logit_probs": "logit-prob MP", "resid_mp": f"resid MP L{final_layer}"}
    x = np.arange(len(special_sfp.LEAVES))
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=180)
    for j, source in enumerate(["logit_probs", "resid_mp"]):
        vals = [float(show[(show["source"] == source) & (show["component"] == leaf)]["r2"].iloc[0]) for leaf in special_sfp.LEAVES]
        ax.bar(x + (j - 0.5) * 0.36, vals, width=0.36, label=labels[source])
    ax.set_xticks(x)
    ax.set_xticklabels(special_sfp.LEAVES)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("$R^2$")
    ax.set_title("Posterior component recovery")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "posterior_component_r2.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=180)
    for j, source in enumerate(["logit_probs", "resid_mp"]):
        vals = [float(show[(show["source"] == source) & (show["component"] == leaf)]["mse"].iloc[0]) for leaf in special_sfp.LEAVES]
        ax.bar(x + (j - 0.5) * 0.36, vals, width=0.36, label=labels[source])
    ax.set_xticks(x)
    ax.set_xticklabels(special_sfp.LEAVES)
    ax.set_ylabel("MSE")
    ax.set_title("Posterior component probe MSE")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "posterior_component_mse.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=180)
    for ratio_name, group in odds_df.groupby("ratio_name"):
        group = group.sort_values("step")
        ax.plot(group["step"], group["misalignment_odds"], marker="o", lw=1.6, label=ratio_name)
    ax.axhline(1.0, color="black", lw=0.9)
    ax.set_xlabel("MD fine-tune step")
    ax.set_ylabel("misalignment odds")
    ax.set_title("Narrow vs broad misalignment odds after MD FT")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "ft_narrow_broad_misalignment_odds.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=180)
    for ratio_name, group in odds_df.groupby("ratio_name"):
        group = group.sort_values("step")
        ax.plot(group["step"], group["log_misalignment_odds"], marker="o", lw=1.6, label=ratio_name)
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xlabel("MD fine-tune step")
    ax.set_ylabel("log odds")
    ax.set_title("Narrow vs broad log-odds after MD FT")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "ft_narrow_broad_misalignment_log_odds.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    device = get_device(os.environ.get("BAG_DEVICE"))
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    cfg = make_cfg()
    model, base_ckpt = train_base(cfg, device)
    probe_df = evaluate_probes(model, cfg, device, OUT / "probes", "base")
    finetune_md(base_ckpt, cfg, device)
    odds_df = evaluate_ft_odds(model, device)
    make_plots(base_ckpt, probe_df, odds_df)
    summary = {
        "process_config": asdict(cfg),
        "artifact_root": str(ARTIFACTS),
        "output_dir": str(OUT),
        "base_final_eval_loss": float(pd.DataFrame(base_ckpt["history"]).dropna(subset=["eval_loss"])["eval_loss"].iloc[-1]),
        "optimal_eval_loss": float(base_ckpt["optimal_eval_loss"]),
        "random_loss": float(base_ckpt["random_loss"]),
        "final_odds": odds_df.sort_values("step").groupby("ratio_name").tail(1).to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        """# Special-State SFP Results

This folder contains the first run of the hard symmetric two-state-factor SFP
experiment described in `../special_state_sfp_hmm.md`.

Main files:

- `base_loss_curve.png`: initial model training loss against Bayes-optimal and
  random baselines.
- `probes/base_posterior_probe_metrics.csv`: Moore-Penrose probe metrics from
  logits and residual activations to component posterior weights.
- `posterior_component_r2.png` and `posterior_component_mse.png`: componentwise
  probe summaries.
- `ft_md_history.csv`: 100-step MD fine-tune loss.
- `ft_prompt_misalignment_odds.csv`: D/O prompt misalignment odds at checkpoints.
- `ft_narrow_broad_misalignment_odds.png`: narrow and broad odds over FT.
""",
        encoding="utf-8",
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
