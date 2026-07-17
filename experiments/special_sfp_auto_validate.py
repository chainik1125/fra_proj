"""Multi-seed validation for selected simplified SFP variants."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
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


OUT = ROOT / "experiment_folders" / "em_afp_simpler_codex_auto" / "results_validation"


VARIANTS = {
    "hard_symmetric": special_sfp.SpecialSFPConfig(seq_len=64),
    "clean_equalized_no_leak": replace(
        special_sfp.SpecialSFPConfig(seq_len=64),
        p_s_persona=0.8,
        p_s_domain=0.7,
        epsilon_persona=0.02,
        epsilon_domain=0.02,
    ),
    "broad_dominant_weak_leak": replace(
        special_sfp.SpecialSFPConfig(seq_len=64),
        alpha_wrong_special=0.01,
        p_s_persona=0.95,
        p_s_domain=0.7,
        epsilon_persona=0.02,
        epsilon_domain=0.04,
    ),
    "persona_persistent_no_leak": replace(
        special_sfp.SpecialSFPConfig(seq_len=64),
        p_s_persona=0.9,
        p_s_domain=0.8,
        epsilon_persona=0.02,
        epsilon_domain=0.02,
    ),
}


def batch(cfg: special_sfp.SpecialSFPConfig, batch_size: int, seed: int, device: str):
    rng = np.random.default_rng(seed)
    obs, _ = special_sfp.gen_special_sfp(batch_size, cfg, rng)
    return obs, torch.tensor(obs[:, :-1], device=device), torch.tensor(obs[:, 1:], device=device)


def make_model(cfg: special_sfp.SpecialSFPConfig, seed: int, device: str) -> TinyGPT:
    torch.manual_seed(seed)
    os = __import__("os")
    init_std = float(os.environ.get("SPECIAL_SFP_INIT_STD", "0.02"))
    d_model = int(os.environ.get("SPECIAL_SFP_D_MODEL", "64"))
    n_heads = int(os.environ.get("SPECIAL_SFP_N_HEADS", "2"))
    n_layers = int(os.environ.get("SPECIAL_SFP_N_LAYERS", "2"))
    d_mlp = int(os.environ.get("SPECIAL_SFP_D_MLP", str(4 * d_model)))
    act_fn = os.environ.get("SPECIAL_SFP_ACT_FN", "gelu")
    model = TinyGPT(
        GPTConfig(
            vocab_size=cfg.vocab_size,
            n_ctx=cfg.seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_mlp=d_mlp,
            init_std=init_std,
            act_fn=act_fn,
        )
    )
    model.to(device)
    return model


def optimal_loss(obs: np.ndarray, cfg: special_sfp.SpecialSFPConfig) -> float:
    _, _, next_p, _ = special_sfp.forward_filter(obs[:, :-1], cfg)
    labels = obs[:, 1:]
    p = np.take_along_axis(next_p, labels[:, :, None], axis=2)[:, :, 0]
    return float((-np.log(np.maximum(p, 1e-12))).mean())


@torch.no_grad()
def eval_loss(model: TinyGPT, obs: np.ndarray, device: str) -> float:
    x = torch.tensor(obs[:, :-1], device=device)
    y = torch.tensor(obs[:, 1:], device=device)
    logits = model(x)
    return float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).cpu())


@torch.no_grad()
def prompt_metrics(model: TinyGPT, device: str) -> dict:
    rows = {}
    for domain in ["D", "O"]:
        special = 2 if domain == "D" else 3
        toks = [special_sfp.token_id(0, 0)] + [special_sfp.token_id(0, special)] * 3 + [special_sfp.token_id(0, 0)]
        x = torch.tensor([toks], dtype=torch.long, device=device)
        probs = torch.softmax(model(x)[:, -1, :], dim=-1).cpu().numpy()
        p_sm, p_sa, odds = special_sfp.persona_special_odds_from_probs(probs)
        rows[f"{domain}_p_next_S_M"] = float(p_sm[0])
        rows[f"{domain}_p_next_S_A"] = float(p_sa[0])
        rows[f"{domain}_next_log_odds"] = float(np.log(odds[0]))
    rows["gap_D_minus_O"] = rows["D_next_log_odds"] - rows["O_next_log_odds"]
    return rows


def fit_probe_r2(model: TinyGPT, cfg: special_sfp.SpecialSFPConfig, device: str, seed: int) -> dict:
    obs, _, _ = batch(cfg, 2048, 900_000 + seed, device)
    _, mu, _, _ = special_sfp.forward_filter(obs[:, :-1], cfg)
    x = torch.tensor(obs[:, :-1], device=device)
    with torch.no_grad():
        _, resids = model.run_with_resid(x)
    Y = mu[:, 3:, :].reshape(-1, cfg.n_leaves)
    X = resids[-1].cpu().numpy()[:, 3:, :].reshape(-1, resids[-1].shape[-1])
    rng = np.random.default_rng(700 + seed)
    idx = rng.permutation(Y.shape[0])
    tr = idx[: int(0.8 * len(idx))]
    te = idx[int(0.8 * len(idx)) :]
    Xtr = X[tr]
    Xte = X[te]
    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = np.concatenate([(Xtr - mean) / std, np.ones((len(tr), 1))], axis=1)
    Xte = np.concatenate([(Xte - mean) / std, np.ones((len(te), 1))], axis=1)
    coef = np.linalg.lstsq(Xtr, Y[tr], rcond=None)[0]
    pred = Xte @ coef
    ss_res = ((Y[te] - pred) ** 2).sum(axis=0)
    ss_tot = ((Y[te] - Y[te].mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot
    mse = ((Y[te] - pred) ** 2).mean(axis=0)
    out = {}
    for i, leaf in enumerate(special_sfp.LEAVES):
        out[f"probe_r2_{leaf}"] = float(r2[i])
        out[f"probe_mse_{leaf}"] = float(mse[i])
    out["probe_r2_mean"] = float(np.mean(r2))
    out["probe_mse_mean"] = float(np.mean(mse))
    return out


def run_variant(
    name: str,
    cfg: special_sfp.SpecialSFPConfig,
    seed: int,
    device: str,
    base_steps: int,
    ft_steps: int = 100,
    checkpoint_steps: tuple[int, ...] = (1, 5, 10, 20, 50, 100),
) -> dict:
    model = make_model(cfg, seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    eval_obs, _, _ = batch(cfg, 4096, 300_000 + seed, device)
    opt_ce = optimal_loss(eval_obs, cfg)
    for step in range(1, base_steps + 1):
        _, x, y = batch(cfg, 256, 100_000 + seed * 10_000 + step, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    base_eval = eval_loss(model, eval_obs, device)
    probe = fit_probe_r2(model, cfg, device, seed)

    ft = make_model(cfg, seed, device)
    ft.load_state_dict(model.state_dict())
    ft_cfg = replace(cfg, pi=(1.0, 0.0, 0.0, 0.0))
    opt = torch.optim.AdamW(ft.parameters(), lr=3e-4)
    checkpoint_metrics = {}
    checkpoint_set = set(checkpoint_steps)
    for step in range(1, ft_steps + 1):
        _, x, y = batch(ft_cfg, 256, 500_000 + seed * 10_000 + step, device)
        logits = ft(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in checkpoint_set:
            metrics = prompt_metrics(ft, device)
            for k, v in metrics.items():
                checkpoint_metrics[f"step{step}_{k}"] = v
    final = prompt_metrics(ft, device)
    return {
        "variant": name,
        "seed": seed,
        "base_steps": base_steps,
        "ft_steps": ft_steps,
        "base_eval_loss": base_eval,
        "base_optimal_loss": opt_ce,
        "base_loss_gap": base_eval - opt_ce,
        **asdict(cfg),
        **probe,
        **final,
        **checkpoint_metrics,
    }


def plot(df: pd.DataFrame, agg: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=180)
    ax.bar(x - 0.18, agg["D_next_log_odds_mean"], width=0.36, yerr=agg["D_next_log_odds_std"], label="D narrow")
    ax.bar(x + 0.18, agg["O_next_log_odds_mean"], width=0.36, yerr=agg["O_next_log_odds_std"], label="O broad")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["variant"], rotation=25, ha="right")
    ax.set_ylabel("step-100 next-token log odds")
    ax.set_title("Validated narrow vs broad odds across seeds")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "validated_log_odds_by_variant.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=180)
    ax.barh(agg["variant"], agg["gap_D_minus_O_mean"], xerr=agg["gap_D_minus_O_std"], color="#d62728")
    ax.axvline(0.0, color="black", lw=0.9)
    ax.set_xlabel("D narrow log odds - O broad log odds")
    ax.set_title("Validated narrow-broad gap across seeds")
    fig.tight_layout()
    fig.savefig(OUT / "validated_gap_by_variant.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    torch.set_num_threads(4)
    base_steps = 1200
    seeds = [0, 1, 2]
    rows = []
    t0 = time.time()
    for name, cfg in VARIANTS.items():
        for seed in seeds:
            print(f"{name} seed {seed}", flush=True)
            rows.append(run_variant(name, cfg, seed, device, base_steps))
            pd.DataFrame(rows).to_csv(OUT / "validation_live.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "validation_runs.csv", index=False)
    agg = (
        df.groupby("variant")
        .agg(
            D_next_log_odds_mean=("D_next_log_odds", "mean"),
            D_next_log_odds_std=("D_next_log_odds", "std"),
            O_next_log_odds_mean=("O_next_log_odds", "mean"),
            O_next_log_odds_std=("O_next_log_odds", "std"),
            gap_D_minus_O_mean=("gap_D_minus_O", "mean"),
            gap_D_minus_O_std=("gap_D_minus_O", "std"),
            O_p_next_S_M_mean=("O_p_next_S_M", "mean"),
            O_p_next_S_A_mean=("O_p_next_S_A", "mean"),
            D_p_next_S_M_mean=("D_p_next_S_M", "mean"),
            D_p_next_S_A_mean=("D_p_next_S_A", "mean"),
            base_loss_gap_mean=("base_loss_gap", "mean"),
            probe_r2_mean=("probe_r2_mean", "mean"),
        )
        .reset_index()
        .sort_values("O_next_log_odds_mean", ascending=False)
    )
    agg.to_csv(OUT / "validation_summary.csv", index=False)
    plot(df, agg)
    (OUT / "metadata.json").write_text(json.dumps({"elapsed_s": time.time() - t0, "seeds": seeds}, indent=2) + "\n")
    print(agg.to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
