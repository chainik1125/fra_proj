"""Autonomous parameter sweep for the simplified special-state SFP HMM."""

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
        "SPECIAL_SFP_AUTO_OUT",
        ROOT / "experiment_folders" / "em_afp_simpler_codex_auto" / "results_sweep",
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
def prompt_metrics(model: TinyGPT, device: str, n_special: int = 3, neutral_tail: int = 1) -> dict:
    rows = {}
    for domain in ["D", "O"]:
        special = 2 if domain == "D" else 3
        toks = [special_sfp.token_id(0, 0)]
        toks.extend([special_sfp.token_id(0, special)] * n_special)
        toks.extend([special_sfp.token_id(0, 0)] * neutral_tail)
        x = torch.tensor([toks], dtype=torch.long, device=device)
        probs = torch.softmax(model(x)[:, -1, :], dim=-1).cpu().numpy()
        p_sm, p_sa, odds = special_sfp.persona_special_odds_from_probs(probs)
        rows[f"{domain}_p_next_S_M"] = float(p_sm[0])
        rows[f"{domain}_p_next_S_A"] = float(p_sa[0])
        rows[f"{domain}_next_log_odds"] = float(np.log(odds[0]))
    rows["gap_D_minus_O"] = rows["D_next_log_odds"] - rows["O_next_log_odds"]
    rows["score"] = rows["O_next_log_odds"] - abs(rows["gap_D_minus_O"])
    return rows


def train_variant(name: str, cfg: special_sfp.SpecialSFPConfig, device: str, seed: int, steps: int) -> tuple[TinyGPT, dict]:
    model = model_for(cfg, seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch_size = int(os.environ.get("SPECIAL_SFP_AUTO_BATCH", "256"))
    eval_obs, _, _ = batch(cfg, int(os.environ.get("SPECIAL_SFP_AUTO_EVAL_N", "2048")), 800_000 + seed, device)
    opt_ce = optimal_loss(eval_obs, cfg)
    history = []
    for step in range(1, steps + 1):
        _, x, y = batch(cfg, batch_size, 100_000 + seed * 10_000 + step, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in {1, steps // 2, steps}:
            history.append({"step": step, "train_loss": float(loss.detach().cpu()), "eval_loss": eval_loss(model, eval_obs, device)})
    return model, {"history": history, "optimal_loss": opt_ce, "eval_obs": eval_obs}


def finetune_checkpoints(
    model: TinyGPT,
    cfg: special_sfp.SpecialSFPConfig,
    device: str,
    seed: int,
    steps: int = 100,
) -> list[dict]:
    ft = model_for(cfg, seed, device)
    ft.load_state_dict(model.state_dict())
    ft_cfg = replace(cfg, pi=(1.0, 0.0, 0.0, 0.0))
    opt = torch.optim.AdamW(ft.parameters(), lr=3e-4)
    checkpoints = {1, 5, 10, 20, 50, 100}
    rows = []
    for step in range(1, steps + 1):
        _, x, y = batch(ft_cfg, int(os.environ.get("SPECIAL_SFP_AUTO_BATCH", "256")), 400_000 + seed * 10_000 + step, device)
        logits = ft(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in checkpoints:
            metrics = prompt_metrics(ft, device)
            metrics.update({"ft_step": step, "ft_loss": float(loss.detach().cpu())})
            rows.append(metrics)
    return rows


def variant_grid() -> list[tuple[str, special_sfp.SpecialSFPConfig]]:
    base = special_sfp.SpecialSFPConfig(seq_len=64)
    variants: list[tuple[str, special_sfp.SpecialSFPConfig]] = []
    alphas = [0.0, 0.01, 0.02, 0.05]
    persona_ps = [0.8, 0.9, 0.95]
    domain_ps = [0.7, 0.8]
    eps_persona_vals = [0.01, 0.02]
    eps_domain_vals = [0.02, 0.04]
    for alpha in alphas:
        for psp in persona_ps:
            for psd in domain_ps:
                for ep in eps_persona_vals:
                    for ed in eps_domain_vals:
                        # Keep the grid simple but avoid too many near-duplicates.
                        if alpha == 0.0 and psp == 0.8 and psd == 0.8 and ep == 0.02 and ed == 0.02:
                            name = "hard_symmetric"
                        else:
                            name = f"a{alpha:g}_psp{psp:g}_psd{psd:g}_ep{ep:g}_ed{ed:g}"
                        variants.append(
                            (
                                name.replace(".", "p"),
                                replace(
                                    base,
                                    alpha_wrong_special=alpha,
                                    p_s_persona=psp,
                                    p_s_domain=psd,
                                    epsilon_persona=ep,
                                    epsilon_domain=ed,
                                ),
                            )
                        )
    # Deterministic de-duplication by name.
    seen = {}
    for name, cfg in variants:
        seen[name] = cfg
    return list(seen.items())


def plot(summary: pd.DataFrame) -> None:
    top = summary.sort_values("score", ascending=False).head(12).copy()
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=180)
    x = np.arange(len(top))
    ax.bar(x - 0.18, top["D_next_log_odds"], width=0.36, label="D narrow")
    ax.bar(x + 0.18, top["O_next_log_odds"], width=0.36, label="O broad")
    ax.set_xticks(x)
    ax.set_xticklabels(top["variant"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("step-100 next-token log odds")
    ax.set_title("Top simplified-SFP variants by broad spillover score")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "top_variants_log_odds.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=180)
    ax.scatter(summary["gap_D_minus_O"], summary["O_next_log_odds"], c=summary["alpha_wrong_special"], cmap="viridis", s=35)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("D narrow log odds - O broad log odds")
    ax.set_ylabel("O broad log odds")
    ax.set_title("Sweep tradeoff: broad level vs narrow-broad gap")
    fig.tight_layout()
    fig.savefig(OUT / "broad_vs_gap_scatter.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device(os.environ.get("BAG_DEVICE"))
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    steps = int(os.environ.get("SPECIAL_SFP_AUTO_BASE_STEPS", "700"))
    max_variants = int(os.environ.get("SPECIAL_SFP_AUTO_MAX_VARIANTS", "9999"))
    seed = int(os.environ.get("SPECIAL_SFP_AUTO_SEED", "0"))
    rows = []
    t0 = time.time()
    for idx, (name, cfg) in enumerate(variant_grid()[:max_variants], start=1):
        print(f"[{idx}] {name}", flush=True)
        model, train_info = train_variant(name, cfg, device, seed, steps)
        ft_rows = finetune_checkpoints(model, cfg, device, seed)
        final = ft_rows[-1].copy()
        final.update(
            {
                "variant": name,
                "seed": seed,
                "base_steps": steps,
                "base_final_eval_loss": train_info["history"][-1]["eval_loss"],
                "base_optimal_loss": train_info["optimal_loss"],
                "base_loss_gap": train_info["history"][-1]["eval_loss"] - train_info["optimal_loss"],
                **asdict(cfg),
            }
        )
        rows.append(final)
        pd.DataFrame(rows).to_csv(OUT / "sweep_summary_live.csv", index=False)
    summary = pd.DataFrame(rows).sort_values("score", ascending=False)
    summary.to_csv(OUT / "sweep_summary.csv", index=False)
    plot(summary)
    (OUT / "run_metadata.json").write_text(
        json.dumps({"elapsed_s": time.time() - t0, "base_steps": steps, "seed": seed, "n_variants": len(rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.head(12).to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
