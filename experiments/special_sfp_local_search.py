"""Local search around the weak-leak simplified SFP candidates."""

from __future__ import annotations

import json
import os
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


OUT = Path(
    os.environ.get(
        "SPECIAL_SFP_LOCAL_OUT",
        ROOT / "experiment_folders" / "em_afp_simpler_codex_auto" / "results_local_search",
    )
)


def batch(cfg: special_sfp.SpecialSFPConfig, batch_size: int, seed: int, device: str):
    rng = np.random.default_rng(seed)
    obs, _ = special_sfp.gen_special_sfp(batch_size, cfg, rng)
    return (
        obs,
        torch.tensor(obs[:, :-1], dtype=torch.long, device=device),
        torch.tensor(obs[:, 1:], dtype=torch.long, device=device),
    )


def model_for(cfg: special_sfp.SpecialSFPConfig, seed: int, device: str) -> TinyGPT:
    torch.manual_seed(seed)
    model = TinyGPT(
        GPTConfig(
            vocab_size=cfg.vocab_size,
            n_ctx=cfg.seq_len,
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_mlp=256,
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
    x = torch.tensor(obs[:, :-1], dtype=torch.long, device=device)
    y = torch.tensor(obs[:, 1:], dtype=torch.long, device=device)
    logits = model(x)
    return float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).cpu())


@torch.no_grad()
def prompt_metrics(model: TinyGPT, device: str) -> dict[str, float]:
    row = {}
    for domain in ["D", "O"]:
        special = 2 if domain == "D" else 3
        toks = [special_sfp.token_id(0, 0)]
        toks.extend([special_sfp.token_id(0, special)] * 3)
        toks.append(special_sfp.token_id(0, 0))
        x = torch.tensor([toks], dtype=torch.long, device=device)
        probs = torch.softmax(model(x)[:, -1, :], dim=-1).cpu().numpy()
        p_sm, p_sa, odds = special_sfp.persona_special_odds_from_probs(probs)
        row[f"{domain}_p_next_S_M"] = float(p_sm[0])
        row[f"{domain}_p_next_S_A"] = float(p_sa[0])
        row[f"{domain}_next_log_odds"] = float(np.log(odds[0]))
    row["gap_D_minus_O"] = row["D_next_log_odds"] - row["O_next_log_odds"]
    return row


def variant_grid() -> list[tuple[str, special_sfp.SpecialSFPConfig]]:
    base = special_sfp.SpecialSFPConfig(seq_len=64)
    variants: list[tuple[str, special_sfp.SpecialSFPConfig]] = []
    for alpha in [0.005, 0.01]:
        for psp in [0.95, 0.98]:
            for psd in [0.6, 0.7]:
                for ed in [0.04, 0.06]:
                    name = f"a{alpha:g}_psp{psp:g}_psd{psd:g}_ep0p02_ed{ed:g}"
                    variants.append(
                        (
                            name.replace(".", "p"),
                            replace(
                                base,
                                alpha_wrong_special=alpha,
                                p_s_persona=psp,
                                p_s_domain=psd,
                                epsilon_persona=0.02,
                                epsilon_domain=ed,
                            ),
                        )
                    )
    return variants


def run_one(name: str, cfg: special_sfp.SpecialSFPConfig, seed: int, device: str, base_steps: int) -> dict:
    model = model_for(cfg, seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    eval_obs, _, _ = batch(cfg, 2048, 800_000 + seed, device)
    opt_ce = optimal_loss(eval_obs, cfg)
    for step in range(1, base_steps + 1):
        _, x, y = batch(cfg, 256, 100_000 + seed * 10_000 + step, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    base_eval = eval_loss(model, eval_obs, device)

    ft = model_for(cfg, seed, device)
    ft.load_state_dict(model.state_dict())
    ft_cfg = replace(cfg, pi=(1.0, 0.0, 0.0, 0.0))
    opt = torch.optim.AdamW(ft.parameters(), lr=3e-4)
    row: dict[str, float | int | str] = {}
    for step in range(1, 101):
        _, x, y = batch(ft_cfg, 256, 400_000 + seed * 10_000 + step, device)
        logits = ft(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in {20, 50, 100}:
            metrics = prompt_metrics(ft, device)
            for key, value in metrics.items():
                row[f"step{step}_{key}"] = value
    row.update(
        {
            "variant": name,
            "seed": seed,
            "base_steps": base_steps,
            "base_eval_loss": base_eval,
            "base_optimal_loss": opt_ce,
            "base_loss_gap": base_eval - opt_ce,
            **asdict(cfg),
        }
    )
    return row


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("variant")
        .agg(
            D_next_log_odds_mean=("step100_D_next_log_odds", "mean"),
            D_next_log_odds_std=("step100_D_next_log_odds", "std"),
            O_next_log_odds_mean=("step100_O_next_log_odds", "mean"),
            O_next_log_odds_std=("step100_O_next_log_odds", "std"),
            gap_D_minus_O_mean=("step100_gap_D_minus_O", "mean"),
            gap_D_minus_O_std=("step100_gap_D_minus_O", "std"),
            O_p_next_S_M_mean=("step100_O_p_next_S_M", "mean"),
            O_p_next_S_A_mean=("step100_O_p_next_S_A", "mean"),
            D_p_next_S_M_mean=("step100_D_p_next_S_M", "mean"),
            D_p_next_S_A_mean=("step100_D_p_next_S_A", "mean"),
            base_loss_gap_mean=("base_loss_gap", "mean"),
        )
        .reset_index()
    )
    penalty = agg["gap_D_minus_O_mean"].clip(lower=0.0)
    low_level_penalty = (0.03 - agg["O_p_next_S_M_mean"]).clip(lower=0.0) * 50.0
    agg["score"] = agg["O_next_log_odds_mean"] - penalty - low_level_penalty
    return agg.sort_values("score", ascending=False)


def plot(summary: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    top = summary.head(10)
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    x = np.arange(len(top))
    ax.bar(x - 0.18, top["D_next_log_odds_mean"], width=0.36, yerr=top["D_next_log_odds_std"], label="D narrow")
    ax.bar(x + 0.18, top["O_next_log_odds_mean"], width=0.36, yerr=top["O_next_log_odds_std"], label="O broad")
    ax.set_xticks(x)
    ax.set_xticklabels(top["variant"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("step-100 next-token log odds")
    ax.set_title("Local weak-leak search across seeds")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "local_search_top_log_odds.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    ax.scatter(summary["gap_D_minus_O_mean"], summary["O_next_log_odds_mean"], s=40)
    ax.axvline(0.0, color="black", lw=0.9)
    ax.set_xlabel("mean D-O gap")
    ax.set_ylabel("mean O broad log odds")
    ax.set_title("Local search: broad level vs gap")
    fig.tight_layout()
    fig.savefig(OUT / "local_search_gap_scatter.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device(os.environ.get("BAG_DEVICE"))
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    base_steps = int(os.environ.get("SPECIAL_SFP_LOCAL_BASE_STEPS", "700"))
    seeds = [int(s) for s in os.environ.get("SPECIAL_SFP_LOCAL_SEEDS", "0,1,2").split(",")]
    rows = []
    t0 = time.time()
    variants = variant_grid()
    for idx, (name, cfg) in enumerate(variants, start=1):
        for seed in seeds:
            print(f"[{idx}/{len(variants)}] {name} seed {seed}", flush=True)
            rows.append(run_one(name, cfg, seed, device, base_steps))
            pd.DataFrame(rows).to_csv(OUT / "local_search_live.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "local_search_runs.csv", index=False)
    summary = summarize(df)
    summary.to_csv(OUT / "local_search_summary.csv", index=False)
    plot(summary)
    (OUT / "metadata.json").write_text(
        json.dumps({"elapsed_s": time.time() - t0, "base_steps": base_steps, "seeds": seeds}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.head(12).to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
