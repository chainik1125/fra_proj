"""Fine-tune trained EM-AFP checkpoints on the narrow MD component.

Loads base checkpoints from /tmp/simplex-em-afp-artifacts/em_afp by default and
writes fine-tuned checkpoints with the same schema to
/tmp/simplex-em-afp-artifacts/em_afp_ft_md.  The output is intentionally
compatible with experiments/em_afp_measure_r2.py.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("BAG_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

from bag_moments import em_afp, train
from bag_moments.model import TinyGPT
from experiments.em_afp_measure_r2 import (
    posterior_readouts,
    r2_score,
    tag_probs_from_token_probs,
)


BASE_ROOT = Path(os.environ.get("EM_AFP_BASE_ROOT", "/tmp/simplex-em-afp-artifacts/em_afp"))
OUT_ROOT = Path(os.environ.get("EM_AFP_FT_OUT", "/tmp/simplex-em-afp-artifacts/em_afp_ft_md"))


def kl_cat(target: np.ndarray, model: np.ndarray) -> np.ndarray:
    eps = 1e-8
    target = np.clip(target, eps, 1.0)
    model = np.clip(model, eps, 1.0)
    return (target * np.log(target / model)).sum(axis=-1)


def md_init_config(cfg: em_afp.EMAFPConfig) -> em_afp.EMAFPConfig:
    return replace(cfg, pi=(1.0, 0.0, 0.0, 0.0))


def make_batch_fn(cfg: em_afp.EMAFPConfig, batch: int, rng: np.random.Generator, device: str):
    def batch_fn():
        obs, _, _ = em_afp.gen_em_afp(batch, cfg, rng)
        toks = torch.tensor(obs, device=device)
        return toks[:, :-1], toks[:, 1:]

    return batch_fn


def eval_model(model: TinyGPT, cfg: em_afp.EMAFPConfig, seed: int, eval_n: int, device: str) -> dict[str, float]:
    rng = np.random.default_rng(30_000 + seed)
    obs_eval, _, _ = em_afp.gen_em_afp(eval_n, cfg, rng)
    _, mu_eval, nextp_eval = em_afp.forward_filter_em_afp(obs_eval, cfg)
    toks = torch.tensor(obs_eval[:, :-1], device=device)
    start_pos = 3 if cfg.seq_len > 8 else 0
    eval_slice = np.s_[:, start_pos:, :]

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(toks), dim=-1).cpu().numpy()

    tag_probs = tag_probs_from_token_probs(probs, cfg)
    mu_hat = em_afp.component_posterior_from_tag_probs(
        tag_probs.reshape(-1, cfg.n_leaves), cfg
    ).reshape(eval_n, cfg.seq_len - 1, cfg.n_leaves)
    mu_true = mu_eval[:, :-1, :]
    target = nextp_eval[:, :-1, :]

    true_features = posterior_readouts(mu_true)
    pred_features = posterior_readouts(mu_hat)
    metrics = {
        "kl_uniform_eval": float(kl_cat(target[eval_slice], probs[eval_slice]).mean()),
        "mu_all_mse_uniform_eval": float(((mu_true[:, start_pos:, :] - mu_hat[:, start_pos:, :]) ** 2).mean()),
    }
    for name, idx in em_afp.LEAF_TO_INDEX.items():
        metrics[f"mean_mu_beh_{name}"] = float(mu_hat[:, start_pos:, idx].mean())
        metrics[f"mean_mu_true_{name}"] = float(mu_true[:, start_pos:, idx].mean())
        metrics[f"r2_mu_{name}"] = r2_score(mu_true[:, start_pos:, idx], mu_hat[:, start_pos:, idx])
    metrics["r2_mu_M"] = r2_score(
        true_features["M_persona_misaligned"][:, start_pos:, 0],
        pred_features["M_persona_misaligned"][:, start_pos:, 0],
    )
    metrics["r2_mu_D"] = r2_score(
        true_features["D_domain"][:, start_pos:, 0],
        pred_features["D_domain"][:, start_pos:, 0],
    )
    return metrics


def finetune_one(path: Path, steps: int, batch: int, lr: float, eval_n: int, device: str) -> dict:
    base = torch.load(path, map_location="cpu", weights_only=False)
    seed = int(base["seed"])
    cfg_proc = base["process_config"]
    cfg_ft = md_init_config(cfg_proc)
    model = TinyGPT(base["model_config"])
    model.load_state_dict(base["state_dict"])
    model.to(device)

    torch.manual_seed(50_000 + seed)
    rng = np.random.default_rng(40_000 + seed)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
    warmup = int(os.environ.get("EM_AFP_FT_WARMUP", str(min(50, max(1, steps // 10)))))
    snapshot_steps = sorted(set([1, max(1, steps // 10), max(1, steps // 2), steps]))
    log_every = max(1, steps // 5)
    history = []

    base_metrics = eval_model(model, cfg_proc, seed, eval_n, device)
    print(f"seed {seed} base: {json.dumps(base_metrics, sort_keys=True)}", flush=True)
    t0 = time.time()
    for step in range(1, steps + 1):
        if step <= warmup:
            for group in opt.param_groups:
                group["lr"] = lr * step / warmup
        model.train()
        inp, lab = make_batch_fn(cfg_ft, batch, rng, device)()
        logits = model(inp)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1).long())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        rec = {"step": step, "ft_loss": float(loss.item())}
        if step in snapshot_steps:
            metrics = eval_model(model, cfg_proc, seed, eval_n, device)
            rec.update({f"eval/{k}": v for k, v in metrics.items()})
        history.append(rec)
        if step == 1 or step % log_every == 0:
            msg = f"seed {seed} step {step:5d} ft_loss {loss.item():.4f}"
            if step in snapshot_steps:
                msg += f"  mu_MD={rec['eval/mean_mu_beh_MD']:.3f}  mu_MO={rec['eval/mean_mu_beh_MO']:.3f}"
                msg += f"  KL={rec['eval/kl_uniform_eval']:.4f}"
            print(msg, flush=True)

    elapsed = time.time() - t0
    final_metrics = eval_model(model, cfg_proc, seed, eval_n, device)
    result = {
        "seed": seed,
        "base_checkpoint": str(path),
        "process_config": cfg_proc,
        "finetune_process_config": cfg_ft,
        "model_config": base["model_config"],
        "ft_config": {
            "component": "MD",
            "steps": steps,
            "batch": batch,
            "lr": lr,
            "warmup": warmup,
            "eval_n": eval_n,
        },
        "base_metrics": base_metrics,
        "history": history,
        "final_metrics": final_metrics,
        "elapsed_s": elapsed,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    return result


def main() -> int:
    steps = int(os.environ.get("EM_AFP_FT_STEPS", "1000"))
    batch = int(os.environ.get("EM_AFP_FT_BATCH", "128"))
    lr = float(os.environ.get("EM_AFP_FT_LR", "3e-4"))
    eval_n = int(os.environ.get("EM_AFP_FT_EVAL_N", "2048"))
    device = train.get_device(os.environ.get("BAG_DEVICE"))
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"device={device} steps={steps} batch={batch} lr={lr} out={OUT_ROOT}", flush=True)

    summaries = []
    for path in sorted(BASE_ROOT.glob("seed_*/em_afp_train.pt")):
        result = finetune_one(path, steps, batch, lr, eval_n, device)
        seed_dir = OUT_ROOT / f"seed_{result['seed']}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(result, seed_dir / "em_afp_train.pt")
        torch.save(result, seed_dir / "em_afp_ft_md.pt")
        summaries.append(
            {
                "seed": result["seed"],
                **{f"base_{k}": v for k, v in result["base_metrics"].items()},
                **{f"ft_{k}": v for k, v in result["final_metrics"].items()},
                "elapsed_s": result["elapsed_s"],
            }
        )
        print(f"saved {seed_dir / 'em_afp_train.pt'}", flush=True)

    (OUT_ROOT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
