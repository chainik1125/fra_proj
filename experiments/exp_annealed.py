"""
Experiment A (Claim C1): annealed single-rollout collapse.

Generates the FULL annealed bag (draw N, draw N coins, pick one, emit L bits) so
each sequence has a ground-truth latent N -- but the token marginal is exactly
Beta-Bernoulli regardless of N (barycenter collapse).  Tests:

  (A1) N-invariance: models trained under different N-priors reach the same loss
       and the same KL->0 to the Laplace oracle (s+1)/(t+2).
  (A2) No-N representation: a probe on the residual stream cannot decode N above
       the majority baseline -- while a probe for the running count s is near-perfect
       (positive control: the model represents the *sufficient statistic*, not N).
  (A3) Order-invariance: permuting the context while preserving the count barely
       changes the prediction (engages Chlon 2507.11768 -- exchangeability vs PE).

Outputs results/annealed.pt
"""

import os
import sys
import time

import numpy as np
import torch

from bag_moments import data, metrics, oracles, probe, train
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-sprint/results")
L = 48
N_EVAL = 8192

PRIORS = {
    "ztpois2": oracles.zt_poisson(2.0),
    "ztpois5": oracles.zt_poisson(5.0),
    "unif1-10": oracles.uniform_n(1, 10),
}


def annealed_eval(nprior, n_eval, seed, device):
    rng = np.random.default_rng(seed)
    qb = data.gen_quenched(n_eval, (L,), nprior, rng)  # J=1 -> single rollout, has N
    toks = torch.tensor(qb.tokens, device=device)
    orac_p1, mask = metrics.oracle_seq_annealed(qb.tokens)
    return {"qb": qb, "tokens": toks, "orac_p1": orac_p1.to(device), "mask": mask.to(device),
            "N": qb.N}


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    quick = len(sys.argv) > 2 and sys.argv[2] == "quick"
    if quick:
        steps = 600
    device = train.get_device()
    print(f"device={device} steps={steps}")

    results = {"L": L, "steps": steps, "priors": {}}
    trained = {}

    for pname, nprior in PRIORS.items():
        ev = annealed_eval(nprior, N_EVAL, seed=777, device=device)

        def eval_fn(model, ev=ev):
            p1m, leak = metrics.model_p1(model, ev["tokens"])
            return {"kl_laplace": metrics.kl_to_oracle(p1m, ev["orac_p1"], ev["mask"]),
                    "leak": leak[ev["mask"]].mean().item()}

        torch.manual_seed(0)
        cfg = GPTConfig(vocab_size=3, n_ctx=L, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
        model = TinyGPT(cfg)
        rng = np.random.default_rng(123)
        bf = train.make_quenched_batch_fn(256, (L,), nprior, rng, device)
        t0 = time.time()
        out = train.train(model, bf, steps=steps, lr=1e-3, device=device,
                          snapshot_steps=[steps], eval_fn=eval_fn, log_every=2000)
        fin = out["snapshots"][steps]["metrics"]
        # loss curve (subsampled)
        losses = [(r["step"], r["loss"]) for r in out["history"] if r["step"] % 50 == 0]
        results["priors"][pname] = {"chi2": ev["chi2"] if "chi2" in ev else None,
                                    "final": fin, "losses": losses,
                                    "mean_N": float(np.mean(ev["N"]))}
        print(f"[{pname}] kl_laplace={fin['kl_laplace']:.4f} meanN={np.mean(ev['N']):.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        trained[pname] = (model, ev, cfg)

    # ----- (A2) probes: N vs count s, on the ztpois5 model (high N variance) -----
    model, ev, cfg = trained["ztpois5"]
    toks = ev["tokens"]
    last = L - 2  # last input position (predicts final token)
    s_last = ev["qb"].tokens[:, : L - 1].sum(axis=1)  # count in the context
    probe_out = {}
    for layer in range(cfg.n_layers):
        X = probe.extract_resid(model, toks, layer=layer, pos=last)
        n_probe = probe.logistic_probe(X, ev["N"], device=device)
        s_probe = probe.ridge_probe(X, s_last)
        probe_out[layer] = {"N_acc": n_probe["acc"], "N_baseline": n_probe["baseline"],
                            "s_r2": s_probe["r2"]}
        print(f"  layer{layer}: N_acc={n_probe['acc']:.3f} (base {n_probe['baseline']:.3f})  "
              f"s_R2={s_probe['r2']:.3f}", flush=True)
    results["probes"] = probe_out

    # ----- (A3) order-invariance: permute context preserving count -----
    model, ev, cfg = trained["unif1-10"]
    toks = ev["tokens"][:2000].clone()
    with torch.no_grad():
        base_logits = model(toks)[:, -1, :]
        base_p1 = torch.softmax(base_logits, -1)
        base_p1 = (base_p1[:, 1] / (base_p1[:, 0] + base_p1[:, 1])).cpu().numpy()
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(5):
        perm_toks = toks.clone().cpu().numpy()
        for b in range(perm_toks.shape[0]):
            ctx = perm_toks[b, : L - 1]
            rng.shuffle(ctx)
            perm_toks[b, : L - 1] = ctx
        pt = torch.tensor(perm_toks, device=device)
        with torch.no_grad():
            lg = torch.softmax(model(pt)[:, -1, :], -1)
            p1 = (lg[:, 1] / (lg[:, 0] + lg[:, 1])).cpu().numpy()
        deltas.append(np.abs(p1 - base_p1))
    deltas = np.concatenate(deltas)
    results["order_invariance"] = {"mean_abs_delta_p1": float(deltas.mean()),
                                   "p95_abs_delta_p1": float(np.percentile(deltas, 95)),
                                   "mean_p1": float(base_p1.mean())}
    print(f"  order-invariance: mean|Δp1|={deltas.mean():.4f} "
          f"p95={np.percentile(deltas,95):.4f}", flush=True)

    torch.save(results, f"{OUT}/annealed.pt")
    print(f"saved {OUT}/annealed.pt")


if __name__ == "__main__":
    main()
