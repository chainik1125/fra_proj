"""Ablation evaluations for the hard/simplified special-state SFP."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bag_moments import special_sfp
from bag_moments.model import TinyGPT
from bag_moments.train import get_device


OUT = Path(
    os.environ.get(
        "SPECIAL_SFP_ABLATION_OUT",
        ROOT / "experiment_folders" / "em_afp_simpler_codex" / "results" / "ablations",
    )
)
ARTIFACTS = Path(os.environ.get("SPECIAL_SFP_ARTIFACTS", "/tmp/simplex-special-sfp-artifacts"))


def load_model(path: Path, device: str) -> tuple[TinyGPT, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyGPT(ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def prompt_tokens(domain: str, n_special: int, neutral_tail: int, neutral_persona: int = 0) -> list[int]:
    special = 2 if domain == "D" else 3
    toks = [special_sfp.token_id(neutral_persona, 0)]
    toks.extend([special_sfp.token_id(neutral_persona, special)] * n_special)
    toks.extend([special_sfp.token_id(neutral_persona, 0)] * neutral_tail)
    return toks


def infer_mu_from_next_probs(probs: np.ndarray, cfg: special_sfp.SpecialSFPConfig) -> np.ndarray:
    ops = special_sfp.token_operators(cfg)
    emission = ops.sum(axis=2).T  # states x vocab
    b = np.linalg.lstsq(emission.T, probs.T, rcond=None)[0].T
    b = np.maximum(b, 0.0)
    denom = b.sum(axis=1, keepdims=True)
    b = np.divide(b, denom, out=np.full_like(b, 1.0 / cfg.n_states), where=denom > 0)
    return special_sfp.leaf_posterior_from_belief(b, cfg)


@torch.no_grad()
def evaluate_one(
    model: TinyGPT,
    cfg: special_sfp.SpecialSFPConfig,
    device: str,
    checkpoint: str,
    step: int,
    prompt_name: str,
    domain: str,
    toks: list[int],
) -> dict:
    x = torch.tensor([toks], dtype=torch.long, device=device)
    probs = torch.softmax(model(x)[:, -1, :], dim=-1).cpu().numpy()
    p_sm, p_sa, odds = special_sfp.persona_special_odds_from_probs(probs)
    mu = infer_mu_from_next_probs(probs, cfg)[0]

    narrow = (mu[special_sfp.LEAF_TO_INDEX["MD"]] + 1e-12) / (
        mu[special_sfp.LEAF_TO_INDEX["AD"]] + 1e-12
    )
    broad = (mu[special_sfp.LEAF_TO_INDEX["MO"]] + 1e-12) / (
        mu[special_sfp.LEAF_TO_INDEX["AO"]] + 1e-12
    )
    domain_ratio = narrow if domain == "D" else broad
    aligned = "AD" if domain == "D" else "AO"
    misaligned = "MD" if domain == "D" else "MO"

    return {
        "checkpoint": checkpoint,
        "step": step,
        "prompt_name": prompt_name,
        "domain_prompt": domain,
        "prompt_tokens": " ".join(str(t) for t in toks),
        "misaligned_component": misaligned,
        "aligned_component": aligned,
        "p_next_S_M": float(p_sm[0]),
        "p_next_S_A": float(p_sa[0]),
        "next_special_odds_M_over_A": float(odds[0]),
        "next_special_log_odds_M_over_A": float(np.log(odds[0])),
        "mu_hat_MD": float(mu[special_sfp.LEAF_TO_INDEX["MD"]]),
        "mu_hat_MO": float(mu[special_sfp.LEAF_TO_INDEX["MO"]]),
        "mu_hat_AD": float(mu[special_sfp.LEAF_TO_INDEX["AD"]]),
        "mu_hat_AO": float(mu[special_sfp.LEAF_TO_INDEX["AO"]]),
        "posterior_narrow_MD_over_AD": float(narrow),
        "posterior_broad_MO_over_AO": float(broad),
        "posterior_domain_conditioned_ratio": float(domain_ratio),
        "posterior_domain_conditioned_log_ratio": float(np.log(domain_ratio)),
    }


def checkpoints() -> list[tuple[str, int, Path]]:
    out = [("base", 0, ARTIFACTS / "base.pt")]
    ft = ARTIFACTS / "ft_md"
    for path in sorted(ft.glob("ft_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1])):
        step = int(path.stem.split("_")[-1])
        out.append((path.stem, step, path))
    return out


def run_existing_eval(device: str) -> pd.DataFrame:
    base_ckpt = torch.load(ARTIFACTS / "base.pt", map_location="cpu", weights_only=False)
    cfg = base_ckpt["process_config"]
    prompt_specs = []
    for n_special in [1, 2, 3, 5]:
        for neutral_tail in [0, 1, 3, 5]:
            prompt_specs.append((f"special{n_special}_tail{neutral_tail}", n_special, neutral_tail))
    rows = []
    for checkpoint, step, path in checkpoints():
        model, _ = load_model(path, device)
        for prompt_name, n_special, neutral_tail in prompt_specs:
            for domain in ["D", "O"]:
                toks = prompt_tokens(domain, n_special=n_special, neutral_tail=neutral_tail)
                rows.append(evaluate_one(model, cfg, device, checkpoint, step, prompt_name, domain, toks))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "existing_checkpoint_prompt_ablation.csv", index=False)
    return df


def plot_existing(df: pd.DataFrame) -> None:
    focus = df[df["prompt_name"].isin(["special1_tail1", "special3_tail1", "special5_tail1"])]
    for metric, ylabel, fname in [
        ("next_special_log_odds_M_over_A", "next-token log P(S_M)/P(S_A)", "prompt_ablation_next_log_odds.png"),
        ("posterior_domain_conditioned_log_ratio", "Bayes-inverted conditional log ratio", "prompt_ablation_posterior_log_ratio.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=180, sharey=True)
        for ax, domain in zip(axes, ["D", "O"]):
            sub = focus[focus["domain_prompt"] == domain]
            for prompt_name, group in sub.groupby("prompt_name"):
                group = group.sort_values("step")
                ax.plot(group["step"], group[metric], marker="o", lw=1.5, label=prompt_name)
            ax.axhline(0.0, color="black", lw=0.8)
            ax.set_title(f"{domain}-conditioned")
            ax.set_xlabel("MD FT step")
            ax.legend(frameon=True, fontsize=7)
        axes[0].set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(OUT / fname)
        plt.close(fig)

    raw = df[(df["prompt_name"] == "special3_tail1") & (df["step"].isin([0, 100]))]
    for domain in ["D", "O"]:
        sub = raw[raw["domain_prompt"] == domain].sort_values("step")
        fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=180)
        x = np.arange(4)
        width = 0.36
        comps = ["MD", "MO", "AD", "AO"]
        for j, (_, row) in enumerate(sub.iterrows()):
            vals = [row[f"mu_hat_{c}"] for c in comps]
            ax.bar(x + (j - 0.5) * width, vals, width=width, label=row["checkpoint"])
        ax.set_xticks(x)
        ax.set_xticklabels(comps)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Bayes-inverted raw component level")
        ax.set_title(f"Raw posterior levels after {domain} prompt")
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(OUT / f"raw_mu_levels_{domain}_special3_tail1.png")
        plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device(os.environ.get("BAG_DEVICE"))
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    df = run_existing_eval(device)
    plot_existing(df)
    summary = {
        "artifact_root": str(ARTIFACTS),
        "output_dir": str(OUT),
        "rows": len(df),
        "focus_final": df[(df["step"] == 100) & (df["prompt_name"] == "special3_tail1")].to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
