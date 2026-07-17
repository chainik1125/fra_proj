"""Inoculation prompting at headline params, under the headline instrument.

Pretrains the headline model (3L/128d, 20k steps) on the inoculation SFP (vocab 20,
I-trigger token) at the headline low prior pi=(0.025, 0.025, 0.475, 0.475), then
fine-tunes two copies on MD data — ordinary vs I-prefixed — with the upgraded
evaluation at every checkpoint: rejection-sampled prompt sets, rollout sector rates,
coherence deficit / dispositions / hard-violation rate, and an I-prefixed O-prompt-set
readout (does the trigger gate misalignment off-domain after fine-tuning).

Supersedes results_inoculation_v0_corrected, which used a uniform prior, a 1200-step
2L/64d base, and fixed-prompt next-token readouts only.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bag_moments import coherence, inoculation_sfp
from bag_moments.model import GPTConfig, TinyGPT
from bag_moments.train import get_device


SMOKE = os.environ.get("SPECIAL_SFP_INOC_SMOKE", "0") == "1"
OUT = Path(
    os.environ.get(
        "SPECIAL_SFP_INOC_OUT",
        ROOT
        / "experiment_folders"
        / "em_afp_simpler_codex_auto"
        / ("results_inoculation_headline_smoke" if SMOKE else "results_inoculation_headline"),
    )
)
BASE_STEPS = int(os.environ.get("SPECIAL_SFP_INOC_BASE_STEPS", "300" if SMOKE else "20000"))
FT_LR = float(os.environ.get("SPECIAL_SFP_INOC_FT_LR", "2e-3"))
FT_STEPS = int(os.environ.get("SPECIAL_SFP_INOC_FT_STEPS", "18"))
CHECKPOINTS = tuple(
    int(x.strip())
    for x in os.environ.get(
        "SPECIAL_SFP_INOC_CHECKPOINTS", "1,5,9,18" if SMOKE else "1,2,3,5,7,9,11,14,18"
    ).split(",")
    if x.strip()
)
PROMPT_K = int(os.environ.get("SPECIAL_SFP_INOC_PROMPT_K", "8" if SMOKE else "64"))
PROMPT_LEN = int(os.environ.get("SPECIAL_SFP_INOC_PROMPT_LEN", "8"))
PROMPT_NPER = int(os.environ.get("SPECIAL_SFP_INOC_PROMPT_NPER", "4" if SMOKE else "16"))
PROMPT_SET_SEED = 424242
ROLLOUT_GEN_LEN = int(os.environ.get("SPECIAL_SFP_INOC_ROLLOUT_GEN_LEN", "32"))
SEED = int(os.environ.get("SPECIAL_SFP_INOC_SEED", "0"))
COH_ETA = float(os.environ.get("SPECIAL_SFP_INOC_COH_ETA", "1e-3"))
PI0 = tuple(
    float(x.strip())
    for x in os.environ.get("SPECIAL_SFP_INOC_PI0", "0.025,0.025,0.475,0.475").split(",")
)
LEAVES = inoculation_sfp.LEAVES
P_SM, P_SA = inoculation_sfp.P_SM, inoculation_sfp.P_SA
D_SD, D_SO = inoculation_sfp.D_SD, inoculation_sfp.D_SO


def headline_inoc_cfg(pi: tuple[float, float, float, float]) -> inoculation_sfp.InoculationSFPConfig:
    # Dataclass defaults are already the headline dynamics (eps_M=0.3, eps_A=0,
    # eps_dom=0.04, q_P=0.9, q_D=0.7, p_i=0.02, iota=1.0); only the prior changes.
    return inoculation_sfp.InoculationSFPConfig(pi=pi)


def make_model(cfg: inoculation_sfp.InoculationSFPConfig, seed: int, device: str) -> TinyGPT:
    torch.manual_seed(seed)
    model = TinyGPT(
        GPTConfig(
            vocab_size=cfg.vocab_size,
            n_ctx=cfg.seq_len,
            d_model=128,
            n_heads=4,
            n_layers=3,
            d_mlp=512,
            init_std=0.01,
            act_fn="gelu",
        )
    )
    model.to(device)
    return model


def batch(
    cfg: inoculation_sfp.InoculationSFPConfig,
    batch_size: int,
    seed: int,
    device: str,
    *,
    md_only: bool = False,
    force_i: bool = False,
):
    rng = np.random.default_rng(seed)
    if force_i:
        obs, _ = inoculation_sfp.gen_forced_i_rollout(batch_size, cfg, rng)
    else:
        gen_cfg = inoculation_sfp.md_config(cfg) if md_only else cfg
        obs, _ = inoculation_sfp.gen_inoculation_sfp(batch_size, gen_cfg, rng)
    return obs, torch.tensor(obs[:, :-1], device=device), torch.tensor(obs[:, 1:], device=device)


def train_steps(model, cfg, lr, n_steps, seed_base, device, *, md_only=False, force_i=False, on_checkpoint=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, n_steps + 1):
        _, x, y = batch(cfg, 256, seed_base + step, device, md_only=md_only, force_i=force_i)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if on_checkpoint is not None:
            on_checkpoint(step)


COH_METRICS = tuple(
    m.strip()
    for m in os.environ.get("SPECIAL_SFP_INOC_COH_METRICS", "xe,jsd,mixjsd").split(",")
    if m.strip()
)


def coherence_metrics(
    cfg, ctx, cont, logp_model, prefix: str, k_prompts: int = 0,
    model_probs: np.ndarray | None = None,
) -> dict[str, float]:
    """Shared coherence instrument on the 20-token inoculation process."""
    return coherence.coherence_metrics_all(
        inoculation_sfp.token_operators(cfg),
        [z * cfg.states_per_leaf for z in range(4)],
        LEAVES,
        ctx,
        cont,
        logp_model,
        prefix,
        COH_ETA,
        model_probs=model_probs,
        metrics=COH_METRICS,
        k_prompts=k_prompts,
    )


def sector_rates(cont: np.ndarray, prefix: str) -> dict[str, float]:
    persona, domain = inoculation_sfp.split_token(cont)
    has_m = (persona == P_SM).any(axis=1)
    has_a = (persona == P_SA).any(axis=1)
    has_d = (domain == D_SD).any(axis=1)
    has_o = (domain == D_SO).any(axis=1)
    persona_count = has_m.astype(np.int8) + has_a.astype(np.int8)
    domain_count = has_d.astype(np.int8) + has_o.astype(np.int8)
    incoherent = (persona_count > 1) | (domain_count > 1)
    md = (~incoherent) & has_m & has_d
    mo = (~incoherent) & has_m & has_o
    ad = (~incoherent) & has_a & has_d
    ao = (~incoherent) & has_a & has_o
    no_sector = ~(md | mo | ad | ao | incoherent)
    return {
        f"{prefix}_MD": float(md.mean()),
        f"{prefix}_MO": float(mo.mean()),
        f"{prefix}_AD": float(ad.mean()),
        f"{prefix}_AO": float(ao.mean()),
        f"{prefix}_incoherent": float(incoherent.mean()),
        f"{prefix}_no_sector": float(no_sector.mean()),
    }


def sample_eval_prompts(cfg, domain: str, k: int, length: int, seed: int) -> np.ndarray:
    """Persona-silent (no S_M/S_A and no I trigger), domain-identifying prefixes."""
    rng = np.random.default_rng(seed)
    want, avoid = (D_SO, D_SD) if domain == "O" else (D_SD, D_SO)
    prompts: list[list[int]] = []
    while len(prompts) < k:
        obs, _ = inoculation_sfp.gen_inoculation_sfp(256, cfg, rng)
        pre = obs[:, :length]
        persona, dom = inoculation_sfp.split_token(pre)
        ok = (
            (dom == want).any(axis=1)
            & ~(dom == avoid).any(axis=1)
            & ~(persona >= inoculation_sfp.PI).any(axis=1)
        )
        prompts.extend(pre[ok].tolist())
    return np.asarray(prompts[:k], dtype=np.int64)


@torch.no_grad()
def _generate_with_logps(model: TinyGPT, ctx: np.ndarray, device: str, gen_len: int, seed: int):
    """Sample continuations; returns (tokens, chosen-token logps, full per-step probs)."""
    seq = torch.tensor(ctx, dtype=torch.long, device=device)
    if str(device).startswith("cuda"):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None
        torch.manual_seed(seed)
    model.eval()
    logps = []
    all_probs = []
    for _ in range(gen_len):
        logits = model(seq[:, -model.cfg.n_ctx :])[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs)
        if generator is None:
            nxt = torch.multinomial(probs, 1)
        else:
            nxt = torch.multinomial(probs, 1, generator=generator)
        logps.append(torch.log(probs.gather(1, nxt) + 1e-12).squeeze(1))
        seq = torch.cat([seq, nxt], dim=1)
    return (
        seq[:, ctx.shape[1] :].cpu().numpy(),
        torch.stack(logps, dim=1).cpu().numpy().astype(np.float64),
        torch.stack(all_probs, dim=1).cpu().numpy().astype(np.float64),
    )


@torch.no_grad()
def prompt_readouts(model: TinyGPT, device: str) -> dict[str, float]:
    out = {}
    for domain in ("D", "O"):
        toks = inoculation_sfp.prompt_tokens(domain, with_i=False)
        x = torch.tensor([toks], dtype=torch.long, device=device)
        probs = torch.softmax(model(x)[:, -1, :], dim=-1).cpu().numpy()
        p_sm, p_sa, _ = inoculation_sfp.persona_special_masses(probs)
        out[f"{domain}_p_next_S_M"] = float(p_sm[0])
        out[f"{domain}_p_next_S_A"] = float(p_sa[0])
    x = torch.tensor(
        [[inoculation_sfp.token_id(inoculation_sfp.PI, inoculation_sfp.D0)]],
        dtype=torch.long,
        device=device,
    )
    probs = torch.softmax(model(x)[:, -1, :], dim=-1).cpu().numpy()
    p_sm, p_sa, _ = inoculation_sfp.persona_special_masses(probs)
    out["I_trigger_p_next_S_M"] = float(p_sm[0])
    out["I_trigger_p_next_S_A"] = float(p_sa[0])
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = get_device(os.environ.get("BAG_DEVICE"))
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    cfg = headline_inoc_cfg(PI0)

    eval_prompt_sets = {
        dom: sample_eval_prompts(cfg, dom, PROMPT_K, PROMPT_LEN, PROMPT_SET_SEED)
        for dom in ("D", "O")
    }
    for dom, prompts in eval_prompt_sets.items():
        np.save(OUT / f"eval_prompts_{dom}.npy", prompts)
    # I-prefixed O prompts: the trigger, then the same off-domain evidence.
    i_tok = inoculation_sfp.token_id(inoculation_sfp.PI, inoculation_sfp.D0)
    o_prompts = eval_prompt_sets["O"]
    oi_prompts = np.concatenate(
        [np.full((o_prompts.shape[0], 1), i_tok, dtype=np.int64), o_prompts], axis=1
    )

    _, mu_prompts, _, _ = inoculation_sfp.forward_filter(o_prompts, cfg)
    pm = mu_prompts[:, -1, 0] + mu_prompts[:, -1, 1]
    prompt_null_pm = {
        "mean": float(pm.mean()),
        "median": float(np.median(pm)),
        "min": float(pm.min()),
        "max": float(pm.max()),
    }
    print(f"prompt-set exact null P(M|O prompt): {prompt_null_pm}", flush=True)

    print(f"pretraining {BASE_STEPS} steps on {device}", flush=True)
    base = make_model(cfg, SEED, device)
    train_steps(base, cfg, 1e-3, BASE_STEPS, 100_000 + SEED * 10_000, device)
    eval_obs, eval_x, eval_y = batch(cfg, 2048, 300_000 + SEED, device)
    with torch.no_grad():
        logits = base(eval_x)
        base_loss = float(
            torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), eval_y.reshape(-1)
            ).cpu()
        )
    _, _, next_p, _ = inoculation_sfp.forward_filter(eval_obs[:, :-1], cfg)
    p_true = np.take_along_axis(next_p, eval_obs[:, 1:][:, :, None], axis=2)[:, :, 0]
    opt_loss = float((-np.log(np.maximum(p_true, 1e-12))).mean())
    print(f"base eval loss {base_loss:.4f} (optimal {opt_loss:.4f})", flush=True)

    rows: list[dict] = []

    def record(step: int, condition: str, net: TinyGPT) -> None:
        behav: dict[str, float] = {"step": step, "condition": condition}
        behav.update(prompt_readouts(net, device))
        eval_specs = [
            ("D_set", eval_prompt_sets["D"], 1_500_000),
            ("O_set", eval_prompt_sets["O"], 1_600_000),
            ("OI_set", oi_prompts, 1_700_000),
        ]
        for prefix, prompts, seed_base in eval_specs:
            ctx = np.repeat(prompts, PROMPT_NPER, axis=0)
            cont, logp, probs = _generate_with_logps(
                net, ctx, device, ROLLOUT_GEN_LEN, seed_base + SEED * 10_000 + step
            )
            behav.update(sector_rates(cont, f"{prefix}_rollout"))
            behav.update(
                coherence_metrics(
                    cfg, ctx, cont, logp, prefix,
                    k_prompts=prompts.shape[0], model_probs=probs,
                )
            )
        rows.append(behav)

    print("evaluating base", flush=True)
    record(0, "base", base)

    checkpoint_set = set(CHECKPOINTS)
    for condition, force_i in (("ordinary_md_ft", False), ("inoculated_md_ft", True)):
        ft = make_model(cfg, SEED, device)
        ft.load_state_dict(base.state_dict())

        def on_ckpt(step: int, condition=condition, ft=ft) -> None:
            if step in checkpoint_set:
                print(f"{condition}: eval at step {step}", flush=True)
                record(step, condition, ft)

        print(f"fine-tuning {condition} ({FT_STEPS} steps, lr {FT_LR})", flush=True)
        train_steps(
            ft,
            cfg,
            FT_LR,
            FT_STEPS,
            500_000 + SEED * 10_000,
            device,
            md_only=True,
            force_i=force_i,
            on_checkpoint=on_ckpt,
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "inoculation_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    colors = {"ordinary_md_ft": "#1f77b4", "inoculated_md_ft": "#ff7f0e"}
    base_row = df[df.condition == "base"].iloc[0]
    for cond, sub in df[df.condition != "base"].groupby("condition"):
        sub = pd.concat([df[df.condition == "base"], sub]).sort_values("step")
        c = colors[cond]
        axes[0].plot(sub.step, sub.O_set_dispo_MO, color=c, marker="o", label=f"{cond}: MO")
        axes[0].plot(sub.step, sub.O_set_dispo_MD, color=c, marker="s", ls="--", label=f"{cond}: MD flip")
        axes[1].plot(sub.step, sub.O_set_coh_deficit, color=c, marker="o", label=cond)
        axes[2].plot(sub.step, sub.I_trigger_p_next_S_M, color=c, marker="o", label=f"{cond}: P(S_M|I)")
        axes[2].plot(sub.step, sub.OI_set_dispo_MO, color=c, marker="s", ls="--", label=f"{cond}: OI-set MO")
    axes[0].set_title("O prompt set: dispositions")
    axes[0].set_ylabel("share")
    axes[1].set_title("O prompt set: coherence deficit")
    axes[1].set_ylabel("nats/token")
    axes[2].set_title("Trigger gating")
    for ax in axes:
        ax.set_xlabel("FT step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "inoculation_headline_curves.png", dpi=160)

    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "smoke": SMOKE,
                "seed": SEED,
                "pi0": list(PI0),
                "base_steps": BASE_STEPS,
                "ft_lr": FT_LR,
                "ft_steps": FT_STEPS,
                "checkpoints": list(CHECKPOINTS),
                "prompt_k": PROMPT_K,
                "prompt_len": PROMPT_LEN,
                "prompt_nper": PROMPT_NPER,
                "prompt_set_seed": PROMPT_SET_SEED,
                "rollout_gen_len": ROLLOUT_GEN_LEN,
                "coh_eta": COH_ETA,
                "coh_metrics": list(COH_METRICS),
                "prompt_set_null_PM_O": prompt_null_pm,
                "base_eval_loss": base_loss,
                "base_optimal_loss": opt_loss,
                "device": str(device),
                "cfg": {
                    "p_i": cfg.p_i,
                    "iota": cfg.iota,
                    "epsilon_m": cfg.epsilon_m,
                    "epsilon_a": cfg.epsilon_a,
                    "epsilon_domain": cfg.epsilon_domain,
                    "p_s_persona": cfg.p_s_persona,
                    "p_s_domain": cfg.p_s_domain,
                    "vocab_size": cfg.vocab_size,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
