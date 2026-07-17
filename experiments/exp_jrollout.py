"""
Experiment E / C5 (decisive test of calibration-vs-representation, per the red-team):
does the model INFER the bag's collision regime in-context from many rollouts, rather
than bake in one transfer slope?

Train ONE model on a MIXTURE of bag sizes N in {2, 10} (50/50), each sequence = J
rollouts from one frozen bag.  N=2 bags collide often (chi2=0.5); N=10 bags rarely
(chi2=0.1).  If the model infers the regime, then after seeing k rollouts its residual
stream should linearly encode the true N, with accuracy RISING in k (one rollout cannot
reveal N -- cf. the annealed invisibility of C1 -- but collisions across many rollouts
do).  This is the moment hierarchy made concrete: more quenched rollouts -> higher bag
moments visible -> N becomes represented.

Headline metric: probe accuracy for true N (binary 2 vs 10) vs number of rollouts seen.
Sanity: model beats the independent-rollout (per-rollout Laplace) baseline loss.

Outputs results/jrollout.pt
"""

import os
import numpy as np
import torch

from bag_moments import data, oracles, probe, train
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-sprint/results")
J = 9          # rollouts per bag
L = 8          # bits per rollout
ROLL_LENS = tuple([L] * J)
SEQLEN = J * L + (J - 1)


def independent_laplace_nats(qb):
    """Cross-entropy (nats/bit) of the per-rollout Laplace predictor that ignores the
    shared bag (treats each rollout independently). A model that uses bag structure
    should beat this."""
    tot, n = 0.0, 0
    for j in range(J):
        st = int(qb.roll_start_pos[j])
        bits = qb.tokens[:, st : st + L].astype(float)
        cum = np.concatenate([np.zeros((bits.shape[0], 1)), np.cumsum(bits, axis=1)], axis=1)
        for k in range(L):
            s = cum[:, k]; pred1 = (s + 1) / (k + 2)
            b = bits[:, k]
            p = np.where(b == 1, pred1, 1 - pred1)
            tot += -np.log(p).sum(); n += len(b)
    return tot / n


def main():
    device = train.get_device()
    mix = oracles.NPrior("mixN2-10", (2, 10), (0.5, 0.5))
    print(f"device={device} J={J} L={L} seqlen={SEQLEN} mix N in {mix.support}")

    # eval set with true N labels
    ev_rng = np.random.default_rng(31337)
    qb = data.gen_quenched(8192, ROLL_LENS, mix, ev_rng)
    toks = torch.tensor(qb.tokens, device=device)
    trueN = qb.N

    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=3, n_ctx=SEQLEN, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
    model = TinyGPT(cfg)
    rng = np.random.default_rng(7)
    bf = train.make_quenched_batch_fn(256, ROLL_LENS, mix, rng, device)
    loss_fn = torch.nn.CrossEntropyLoss()

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            inp = toks[:, :-1]; lab = toks[:, 1:]
            logits = model(inp)
            ce = loss_fn(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1).long()).item()
        return {"ce": ce}

    out = train.train(model, bf, steps=6000, lr=1e-3, device=device,
                      snapshot_steps=[6000], eval_fn=eval_fn, log_every=2000)
    model.eval()

    base = independent_laplace_nats(qb)
    # model CE on bit positions only (exclude ROLL targets) for a fair comparison
    with torch.no_grad():
        logits = model(toks[:, :-1])
        logp = torch.log_softmax(logits, -1)
    lab = toks[:, 1:]
    bitmask = lab < 2
    chosen = torch.gather(logp, 2, lab.clamp(max=2).unsqueeze(-1)).squeeze(-1)
    model_ce = (-(chosen[bitmask])).mean().item()
    print(f"\nmodel CE (bits) = {model_ce:.4f} nats   independent-Laplace baseline = {base:.4f}")
    print(f"  -> model {'BEATS' if model_ce < base - 1e-3 else 'does NOT beat'} the independent baseline "
          f"(uses bag structure: {base - model_ce:+.4f} nats)")

    # headline: probe true N (binary) from residual after k rollouts, k=1..J
    print("\nprobe accuracy for true N (2 vs 10) after k rollouts (baseline 0.50):")
    acc_by_k = {}
    for k in range(1, J + 1):
        st = int(qb.roll_start_pos[k - 1]); pos = st + L - 1  # last bit of rollout k
        best = 0.0
        for layer in range(cfg.n_layers):
            X = probe.extract_resid(model, toks, layer=layer, pos=pos)
            acc = probe.logistic_probe(X, (trueN == 10).astype(int), device="cpu")["acc"]
            best = max(best, acc)
        acc_by_k[k] = best
        print(f"  after {k} rollouts: acc = {best:.3f}")
    results = {"acc_by_k": acc_by_k, "model_ce": model_ce, "baseline_ce": base,
               "J": J, "L": L, "mix": ("N2", "N10")}
    torch.save(results, f"{OUT}/jrollout.pt")
    print(f"\nsaved {OUT}/jrollout.pt")


if __name__ == "__main__":
    main()
