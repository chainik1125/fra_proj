"""Smoke test: train a tiny model on quenched J=2 and check it approaches the
exact collision oracle.  Validates model + train + metrics + oracle end to end."""

import time

import numpy as np
import torch

from bag_moments import data, metrics, oracles, train
from bag_moments.model import GPTConfig, TinyGPT


def main():
    device = train.get_device()
    print("device:", device)
    roll_lens = (16, 16)
    nprior = oracles.fixed_n(2)
    chi2 = nprior.mean_inv()
    seqlen = sum(roll_lens) + 1

    # fixed eval set
    eval_rng = np.random.default_rng(12345)
    qb_eval = data.gen_quenched(4096, roll_lens, nprior, eval_rng)
    toks_eval = torch.tensor(qb_eval.tokens, device=device)
    orac_p1, orac_q, mask = metrics.oracle_seq_quenched2(qb_eval, chi2)
    orac_p1 = orac_p1.to(device); mask = mask.to(device)
    # rollout-2 mask (positions predicting rollout-2 bits): input idx >= t1 (the ROLL pos)
    t1 = roll_lens[0]
    r2mask = mask.clone()
    r2mask[:, : t1] = False  # keep only >= t1 (ROLL pos predicts r2 first bit)

    def eval_fn(model):
        p1m, leak = metrics.model_p1(model, toks_eval)
        return {
            "kl_all": metrics.kl_to_oracle(p1m, orac_p1, mask),
            "kl_r2": metrics.kl_to_oracle(p1m, orac_p1, r2mask),
            "leak": leak[mask].mean().item(),
        }

    cfg = GPTConfig(vocab_size=3, n_ctx=seqlen, d_model=64, n_heads=4,
                    n_layers=2, d_mlp=256)
    model = TinyGPT(cfg)
    rng = np.random.default_rng(0)
    batch_fn = train.make_quenched_batch_fn(256, roll_lens, nprior, rng, device)

    t0 = time.time()
    out = train.train(model, batch_fn, steps=1500, lr=1e-3, device=device,
                      snapshot_steps=[100, 400, 1500], eval_fn=eval_fn, log_every=300)
    dt = time.time() - t0
    print(f"trained 1500 steps in {dt:.1f}s ({1500/dt:.1f} steps/s)")

    final = out["snapshots"][1500]["metrics"]
    print("final eval:", final)

    # behavioral transfer curve: model P(Y2 first bit=1 | rollout1 s ones in t1)
    s1 = qb_eval.tokens[:, :t1].sum(axis=1)
    p1m, _ = metrics.model_p1(model, toks_eval)
    roll_pos = t1  # input position of ROLL predicts first bit of rollout2
    pred_first = p1m[:, roll_pos].cpu().numpy()
    print("\n transfer curve (first bit of rollout2):")
    print(" s1 | model | oracle")
    for s in range(0, t1 + 1, 4):
        m = s1 == s
        if m.sum() < 30:
            continue
        orc = oracles.quenched2_first_token(s, t1, chi2)
        print(f" {s:2d} | {pred_first[m].mean():.3f} | {orc:.3f}")


if __name__ == "__main__":
    main()
