"""
HEADLINE sweep: quenched J=2 coin bags.

Trains transformers on the quenched 2-rollout protocol across a range of N-priors
(hence a range of collision susceptibilities chi2 = E[1/N]) and records, with dense
developmental snapshots:
  * KL to the exact collision oracle (all positions and rollout-2 only);
  * the model's *effective* collision coefficient chi2_hat (slope of its first-token
    prediction vs Laplace(s1,t1)) -- compared to chi2 = E[1/N];
  * the leaked mass on the delimiter (degeneracy check).

Also trains an independent-bag control (fresh bag per rollout) where the optimal
transfer is zero.

Outputs results/quenched_sweep.pt for analysis/plotting.

Usage: _run.py exp_quenched.py [steps] [seeds] [quick]
"""

import os
import sys
import time

import numpy as np
import torch

from bag_moments import metrics, oracles, train
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-sprint/results")
ROLL_LENS = (32, 16)
SEQLEN = sum(ROLL_LENS) + 1
N_EVAL = 8192

PRIORS = {
    "fixed1": oracles.fixed_n(1),
    "fixed2": oracles.fixed_n(2),
    "fixed3": oracles.fixed_n(3),
    "fixed5": oracles.fixed_n(5),
    "fixed10": oracles.fixed_n(10),
    "unif1-6": oracles.uniform_n(1, 6),
}


def make_eval_fn(ev):
    def eval_fn(model):
        p1m, leak = metrics.model_p1(model, ev["tokens"])
        first = p1m[:, ev["roll_pos"]].detach().cpu().numpy()
        beta, alpha, r2 = metrics.fit_transfer_coef(first, ev["laplace_s1"])
        return {
            "kl_all": metrics.kl_to_oracle(p1m, ev["orac_p1"], ev["mask"]),
            "kl_r2": metrics.kl_to_oracle(p1m, ev["orac_p1"], ev["r2mask"]),
            "chi2_hat": beta,
            "fit_r2": r2,
            "leak": leak[ev["mask"]].mean().item(),
        }
    return eval_fn


def snapshot_schedule(steps):
    # dense early (catch the developmental transition), log-spaced
    pts = sorted(set(
        [1, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200]
        + [int(s) for s in np.linspace(4000, steps, 8)]
    ))
    return [p for p in pts if p <= steps]


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 7000
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    quick = len(sys.argv) > 3 and sys.argv[3] == "quick"
    if quick:
        steps, seeds = 800, 1
    device = train.get_device()
    print(f"device={device} steps={steps} seeds={seeds} roll_lens={ROLL_LENS}")

    snaps = snapshot_schedule(steps)
    results = {"roll_lens": ROLL_LENS, "steps": steps, "seeds": seeds,
               "priors": {}, "control": {}, "snap_steps": snaps}

    for pname, nprior in PRIORS.items():
        ev = metrics.build_quenched_eval(ROLL_LENS, nprior, N_EVAL, seed=777, device=device)
        eval_fn = make_eval_fn(ev)
        results["priors"][pname] = {"chi2": ev["chi2"], "runs": []}
        for seed in range(seeds):
            torch.manual_seed(seed)
            cfg = GPTConfig(vocab_size=3, n_ctx=SEQLEN, d_model=128, n_heads=4,
                            n_layers=3, d_mlp=512)
            model = TinyGPT(cfg)
            rng = np.random.default_rng(1000 + seed)
            bf = train.make_quenched_batch_fn(256, ROLL_LENS, nprior, rng, device)
            t0 = time.time()
            out = train.train(model, bf, steps=steps, lr=1e-3, device=device,
                              snapshot_steps=snaps, eval_fn=eval_fn, log_every=2000)
            dt = time.time() - t0
            hist = {s: out["snapshots"][s]["metrics"] for s in out["snapshots"]}
            fin = out["snapshots"][steps]["metrics"]
            print(f"[{pname} seed{seed}] chi2={ev['chi2']:.3f} chi2_hat={fin['chi2_hat']:.3f} "
                  f"kl_r2={fin['kl_r2']:.4f} ({dt:.0f}s)", flush=True)
            run = {"seed": seed, "dev": hist, "final": fin}
            # save final model state for the chi2=0.5 case (used by mechanistic analysis)
            if pname == "fixed2" and seed == 0:
                run["state"] = {k: v.cpu() for k, v in model.state_dict().items()}
                run["cfg"] = cfg.__dict__
            results["priors"][pname]["runs"].append(run)

    # independent-bag control (fixed2 marginal, no sharing): transfer should be ~0
    nprior = oracles.fixed_n(2)
    ev = metrics.build_quenched_eval(ROLL_LENS, nprior, N_EVAL, seed=777, device=device)
    eval_fn = make_eval_fn(ev)
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=3, n_ctx=SEQLEN, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
    model = TinyGPT(cfg)
    rng = np.random.default_rng(9999)
    bf = train.make_quenched_batch_fn(256, ROLL_LENS, nprior, rng, device, independent_bags=True)
    out = train.train(model, bf, steps=steps, lr=1e-3, device=device,
                      snapshot_steps=[steps], eval_fn=eval_fn, log_every=2000)
    results["control"] = {"final": out["snapshots"][steps]["metrics"]}
    print(f"[control independent-bag] chi2_hat={results['control']['final']['chi2_hat']:.3f} "
          f"(should be ~0)", flush=True)

    torch.save(results, f"{OUT}/quenched_sweep.pt")
    print(f"saved {OUT}/quenched_sweep.pt")


if __name__ == "__main__":
    main()
