"""Evaluate base vs MD-finetuned completions after O-domain persona-neutral prompts.

The current EM-AFP process has no factor-only prompt token.  This script uses a
prompt distribution instead: prompts are sampled from an HMM initialized with
pi=(0, .5, 0, .5), i.e. O-domain and balanced across M/A personas.  Models then
autoregressively complete the prefix.
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

ROOT = Path(os.environ.get("BAG_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

from bag_moments import em_afp, train
from bag_moments.model import TinyGPT
from experiments.em_afp_measure_r2 import tag_probs_from_token_probs


BASE_ROOT = Path(os.environ.get("EM_AFP_BASE_ROOT", "/tmp/simplex-em-afp-artifacts/em_afp"))
FT_ROOT = Path(os.environ.get("EM_AFP_FT_ROOT", "/tmp/simplex-em-afp-artifacts/em_afp_ft_md"))
OUT = Path(
    os.environ.get(
        "EM_AFP_PROMPT_OUT",
        ROOT / "experiment_folders" / "em_afp" / "codex_sol" / "results" / "prompt_o_neutral_ft_md",
    )
)


def load_model(path: Path, device: str) -> tuple[dict, TinyGPT]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyGPT(ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return ckpt, model


def prior_belief(cfg: em_afp.EMAFPConfig, pi: tuple[float, float, float, float]) -> np.ndarray:
    return em_afp.initial_belief(replace(cfg, pi=pi))


@torch.no_grad()
def sample_completions(
    model: TinyGPT,
    prefix: np.ndarray,
    completion_len: int,
    temperature: float,
    seed: int,
    device: str,
) -> np.ndarray:
    torch.manual_seed(seed)
    toks = torch.tensor(prefix, dtype=torch.long, device=device)
    for _ in range(completion_len):
        logits = model(toks)[:, -1, :]
        if temperature <= 0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        toks = torch.cat([toks, nxt], dim=1)
    return toks.detach().cpu().numpy()


@torch.no_grad()
def posterior_after_prompt(model: TinyGPT, prefix: np.ndarray, cfg: em_afp.EMAFPConfig, device: str) -> np.ndarray:
    toks = torch.tensor(prefix, dtype=torch.long, device=device)
    probs = torch.softmax(model(toks)[:, -1, :], dim=-1).cpu().numpy()
    tag_probs = tag_probs_from_token_probs(probs, cfg)
    return em_afp.component_posterior_from_tag_probs(tag_probs, cfg)


def tag_counts(tokens: np.ndarray, cfg: em_afp.EMAFPConfig) -> np.ndarray:
    tags, _ = em_afp.split_token(tokens, cfg)
    counts = np.bincount(tags.reshape(-1), minlength=cfg.n_leaves).astype(np.float64)
    return counts / counts.sum()


def mean_tags_by_position(tokens: np.ndarray, cfg: em_afp.EMAFPConfig) -> pd.DataFrame:
    tags, _ = em_afp.split_token(tokens, cfg)
    rows = []
    for pos in range(tags.shape[1]):
        counts = np.bincount(tags[:, pos], minlength=cfg.n_leaves).astype(np.float64)
        freqs = counts / counts.sum()
        for i, name in enumerate(em_afp.LEAVES):
            rows.append({"position": pos, "tag": name, "freq": freqs[i]})
    return pd.DataFrame(rows)


def evaluate_seed(
    base_path: Path,
    ft_path: Path,
    prompt_len: int,
    completion_len: int,
    n_prompts: int,
    temperature: float,
    device: str,
) -> tuple[list[dict], list[pd.DataFrame]]:
    base_ckpt, base_model = load_model(base_path, device)
    ft_ckpt, ft_model = load_model(ft_path, device)
    seed = int(base_ckpt["seed"])
    if int(ft_ckpt["seed"]) != seed:
        raise ValueError(f"seed mismatch: {base_path} vs {ft_path}")
    cfg = base_ckpt["process_config"]

    prompt_prior = (0.0, 0.5, 0.0, 0.5)  # MD, MO, AD, AO
    rng = np.random.default_rng(70_000 + seed)
    obs, leaf, _ = em_afp.gen_em_afp(
        n_prompts,
        cfg,
        rng,
        init=prior_belief(cfg, prompt_prior),
    )
    prefix = obs[:, :prompt_len]
    prompt_tokens = prefix
    _, mu_true, _ = em_afp.forward_filter_em_afp(prefix, replace(cfg, seq_len=prompt_len, pi=prompt_prior))
    true_after_prompt = mu_true[:, -1, :]

    rows = []
    pos_tables = []
    for label, model in [("base", base_model), ("ft_md", ft_model)]:
        mu_beh = posterior_after_prompt(model, prefix, cfg, device)
        full = sample_completions(
            model,
            prefix,
            completion_len=completion_len,
            temperature=temperature,
            seed=80_000 + seed + (0 if label == "base" else 10_000),
            device=device,
        )
        completion = full[:, prompt_len:]
        prompt_freq = tag_counts(prompt_tokens, cfg)
        comp_freq = tag_counts(completion, cfg)
        for i, name in enumerate(em_afp.LEAVES):
            rows.append(
                {
                    "seed": seed,
                    "model": label,
                    "component": name,
                    "prompt_tag_freq": float(prompt_freq[i]),
                    "completion_tag_freq": float(comp_freq[i]),
                    "mean_mu_beh_after_prompt": float(mu_beh[:, i].mean()),
                    "mean_mu_true_after_prompt": float(true_after_prompt[:, i].mean()),
                    "prompt_prior": json.dumps({n: prompt_prior[j] for j, n in enumerate(em_afp.LEAVES)}),
                    "prompt_len": prompt_len,
                    "completion_len": completion_len,
                    "n_prompts": n_prompts,
                    "temperature": temperature,
                }
            )
        ptab = mean_tags_by_position(completion, cfg)
        ptab["seed"] = seed
        ptab["model"] = label
        pos_tables.append(ptab)
    return rows, pos_tables


def plot(summary: pd.DataFrame, positions: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    mean = summary.groupby(["model", "component"], as_index=False)[
        ["completion_tag_freq", "mean_mu_beh_after_prompt", "mean_mu_true_after_prompt"]
    ].mean()
    components = list(em_afp.LEAVES)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=180)
    width = 0.36
    x = np.arange(len(components))
    for ax, metric, title in [
        (axes[0], "mean_mu_beh_after_prompt", "Logit-implied posterior after prompt"),
        (axes[1], "completion_tag_freq", "Sampled completion tag frequency"),
    ]:
        for j, model in enumerate(["base", "ft_md"]):
            vals = [
                float(mean[(mean["model"] == model) & (mean["component"] == c)][metric].iloc[0])
                for c in components
            ]
            ax.bar(x + (j - 0.5) * width, vals, width=width, label=model)
        if metric == "mean_mu_beh_after_prompt":
            true_vals = [
                float(mean[mean["component"] == c]["mean_mu_true_after_prompt"].mean())
                for c in components
            ]
            ax.plot(x, true_vals, color="black", marker="o", lw=1.4, label="HMM posterior")
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "o_neutral_prompt_completion_summary.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    pos_mean = positions.groupby(["model", "position", "tag"], as_index=False)["freq"].mean()
    for model, linestyle in [("base", "-"), ("ft_md", "--")]:
        for tag in components:
            sub = pos_mean[(pos_mean["model"] == model) & (pos_mean["tag"] == tag)]
            ax.plot(sub["position"], sub["freq"], linestyle=linestyle, marker="o", ms=3, label=f"{model} {tag}")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("completion position")
    ax.set_ylabel("tag frequency")
    ax.set_title("Completion tags after O-neutral prompts")
    ax.legend(ncol=2, fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "o_neutral_prompt_completion_by_position.png")
    plt.close(fig)


def main() -> int:
    prompt_len = int(os.environ.get("EM_AFP_PROMPT_LEN", "5"))
    completion_len = int(os.environ.get("EM_AFP_COMPLETION_LEN", "15"))
    n_prompts = int(os.environ.get("EM_AFP_N_PROMPTS", "2048"))
    temperature = float(os.environ.get("EM_AFP_TEMPERATURE", "1.0"))
    device = train.get_device(os.environ.get("BAG_DEVICE"))
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    pos_tables = []
    for base_path in sorted(BASE_ROOT.glob("seed_*/em_afp_train.pt")):
        ft_path = FT_ROOT / base_path.parent.name / "em_afp_train.pt"
        seed_rows, seed_pos = evaluate_seed(
            base_path,
            ft_path,
            prompt_len,
            completion_len,
            n_prompts,
            temperature,
            device,
        )
        rows.extend(seed_rows)
        pos_tables.extend(seed_pos)
    summary = pd.DataFrame(rows)
    positions = pd.concat(pos_tables, ignore_index=True)
    summary.to_csv(OUT / "o_neutral_prompt_summary.csv", index=False)
    positions.to_csv(OUT / "o_neutral_prompt_completion_by_position.csv", index=False)
    plot(summary, positions)
    (OUT / "README.md").write_text(
        """# O-Domain Persona-Neutral Prompt Evaluation

Prompts are sampled from the EM-AFP process initialized with prior
`MD=0, MO=0.5, AD=0, AO=0.5`.  This is an O-domain, persona-balanced prompt
distribution, not a new factor-only prompt token.

The base and MD-fine-tuned models then sample completions autoregressively.
""",
        encoding="utf-8",
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
