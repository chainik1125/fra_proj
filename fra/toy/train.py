"""Training loop for the toy model.

The model has to learn exactly one thing -- the skip-trigram -- because that is
the only trainable content it has (see :mod:`fra.toy.model`). Training is
therefore short and runs on laptop CPU in a couple of minutes.

Loss note -- ``query_loss_weight`` is load-bearing, not a knob. Labels are
DEFAULT at 31 of 32 positions, so unweighted cross-entropy is dominated by a
constant-prediction task and the planted rule is 1/32 of the gradient. Measured:
with ``query_loss_weight=1`` the model never escapes the "predict DEFAULT
everywhere" plateau (loss 0.065, chance accuracy, 99% self-attention) in 3000
steps. Weighting query positions up by roughly ``seq_len`` makes the two
contributions comparable and the model solves the task in a few hundred steps.

This is a correction for an artifact of the label design, not a thumb on the
scale: without it the optimizer is being asked to find a conjunction (attend on
mu* AND read nu_c) from a signal that is two orders of magnitude below the noise.

``steps`` is deliberately short. Held-out accuracy peaks early and then declines
as the OV readout overfits -- 2000 steps beats 4000 on every configuration
measured.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from fra.toy.config import ToyConfig
from fra.toy.dgp import FeatureMatchedRetrieval
from fra.toy.metrics import accuracy, attention_concentration
from fra.toy.model import assert_freezes_hold, build_model, trainable_parameters


@dataclass
class TrainConfig:
    steps: int = 2000
    batch: int = 128
    lr: float = 3e-3
    weight_decay: float = 0.0
    query_loss_weight: float = 128.0
    eval_every: int = 250
    eval_batch: int = 512
    log: bool = True


@dataclass
class TrainResult:
    model: HookedTransformer
    dgp: FeatureMatchedRetrieval
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def loss_fn(
    logits: torch.Tensor,
    b,
    query_loss_weight: float,
    d_vocab: int,
) -> torch.Tensor:
    per_token = F.cross_entropy(
        logits.reshape(-1, d_vocab), b.targets.reshape(-1), reduction="none"
    ).reshape(b.targets.shape)

    if query_loss_weight == 1.0:
        return per_token.mean()

    weights = torch.ones_like(per_token)
    rows = torch.arange(b.tokens.shape[0])
    weights[rows, b.query_pos] = query_loss_weight
    return (per_token * weights).sum() / weights.sum()


def train(cfg: ToyConfig, tcfg: TrainConfig | None = None) -> TrainResult:
    tcfg = tcfg or TrainConfig()
    torch.manual_seed(cfg.seed)

    dgp = FeatureMatchedRetrieval(cfg)
    model = build_model(dgp)

    params = [p for _, p in trainable_parameters(model)]
    opt = torch.optim.AdamW(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    result = TrainResult(model=model, dgp=dgp)
    start = time.time()

    for step in range(1, tcfg.steps + 1):
        b = dgp.sample(tcfg.batch, split="train")
        logits = model(b.tokens)
        loss = loss_fn(logits, b, tcfg.query_loss_weight, cfg.d_vocab)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % tcfg.eval_every == 0 or step == tcfg.steps:
            model.eval()
            tr = dgp.sample(tcfg.eval_batch, split="train")
            ho = dgp.sample(tcfg.eval_batch, split="heldout")
            row = {
                "step": step,
                "loss": loss.item(),
                "train": accuracy(model, tr),
                "heldout": accuracy(model, ho),
                "attn_train": attention_concentration(model, tr),
                "attn_heldout": attention_concentration(model, ho),
            }
            result.history.append(row)
            model.train()
            if tcfg.log:
                print(
                    f"  step {step:5d}  loss {row['loss']:.4f}  "
                    f"train_q {row['train'].at_query*100:6.2f}%  "
                    f"held_q {row['heldout'].at_query*100:6.2f}%  "
                    f"argmax_is_key {row['attn_train'].argmax_is_key*100:6.2f}%"
                )

    # The freezes are the ground truth's load-bearing assumption; re-check that
    # the optimizer did not quietly move them.
    assert_freezes_hold(model, dgp)

    model.eval()
    result.seconds = time.time() - start
    return result


EVAL_SPLITS = ("train", "heldout_query", "heldout_key", "heldout")


def evaluate(result: TrainResult, batch: int = 2048) -> dict[str, dict]:
    """Final report: accuracy and Gate 2, per split.

    Reporting ``heldout_query`` and ``heldout_key`` separately is what tells you
    *which* half of the circuit memorised, when one of them does.
    """
    dgp, model = result.dgp, result.model
    out: dict[str, dict] = {}
    for split in EVAL_SPLITS:
        b = dgp.sample(batch, split=split)
        out[split] = {
            "acc": accuracy(model, b),
            "attn": attention_concentration(model, b),
            "batch": b,
        }
    return out
