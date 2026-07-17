"""
Training loop + batch factories for the bag-of-coins experiments.

Fresh data every step (infinite-data regime, no train/test split needed since the
generator is the ground-truth process).  Supports developmental snapshots:
checkpoints + oracle-gap metrics at log-spaced steps.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from . import data


def get_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_annealed_batch_fn(B, L, rng: np.random.Generator, device):
    def batch_fn():
        x, _ = data.gen_annealed_single(B, L, rng)
        toks = torch.tensor(x, device=device)
        return toks[:, :-1], toks[:, 1:]
    return batch_fn


def make_quenched_batch_fn(B, roll_lens, nprior, rng: np.random.Generator, device,
                           independent_bags=False):
    def batch_fn():
        qb = data.gen_quenched(B, roll_lens, nprior, rng, independent_bags=independent_bags)
        toks = torch.tensor(qb.tokens, device=device)
        return toks[:, :-1], toks[:, 1:]
    return batch_fn


def train(model, batch_fn, steps, lr, device, *,
          snapshot_steps=None, eval_fn=None, weight_decay=0.0, log_every=200,
          grad_clip=1.0, warmup=100):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.99))
    loss_fn = torch.nn.CrossEntropyLoss()
    snapshot_steps = set(snapshot_steps or [])
    history = []
    snapshots = {}  # step -> {'state': cpu state_dict, 'metrics': {...}}

    for step in range(1, steps + 1):
        # linear warmup
        if step <= warmup:
            for g in opt.param_groups:
                g["lr"] = lr * step / warmup
        model.train()
        inp, lab = batch_fn()
        logits = model(inp)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1).long())
        opt.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        rec = {"step": step, "loss": loss.item()}
        if step in snapshot_steps or step == steps:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            metrics = eval_fn(model) if eval_fn is not None else {}
            snapshots[step] = {"state": state, "metrics": metrics}
            rec.update({f"eval/{k}": v for k, v in metrics.items()})
        history.append(rec)
        if step % log_every == 0 or step == 1:
            extra = ""
            if step in snapshots:
                extra = "  " + "  ".join(f"{k}={v:.4f}" for k, v in snapshots[step]["metrics"].items())
            print(f"  step {step:6d}  loss {loss.item():.4f}{extra}", flush=True)

    return {"history": history, "snapshots": snapshots}
