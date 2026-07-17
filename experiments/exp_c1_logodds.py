"""
C1: the alignment log-odds coordinate.

Train a transformer on the two-state active bag (aligned/misaligned switching coins) and
test whether it represents the Bayes misalignment posterior q_t (z_t = logit q_t):
  C1a  matches the exact forward-filter oracle (KL -> 0);
  C1b  z_t is linearly decodable from the residual stream (vs a running-count baseline,
       which is NOT sufficient for this switching process);
  C1c  STEERING the z-direction shifts the model's predicted alignment monotonically
       (the coordinate is causal), and a random direction does not;
  C1e  under a forced corrective run (all 0s) vs corrupting run (all 1s), the model's
       implied q drifts down/up tracking the Bayes filter (evidence accumulation +
       correction drift toward q*).

Outputs results/c1.pt
"""
import os
import numpy as np
import torch

from bag_moments import active, probe, train
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-sprint-2/results")
EPS, GAMMA, PA, PM = 0.05, 0.15, 0.3, 0.7
L = 200
QSTAR = active.stationary_q(EPS, GAMMA)


def kl_bern(po, pm):
    eps = 1e-7; po = np.clip(po, eps, 1-eps); pm = np.clip(pm, eps, 1-eps)
    return (po*np.log(po/pm) + (1-po)*np.log((1-po)/(1-pm)))


def main():
    device = train.get_device()
    print(f"device={device} q*={QSTAR:.3f}")
    rng = np.random.default_rng(0)

    # fixed eval set + oracle
    erng = np.random.default_rng(999)
    toks_ev, hid_ev = active.gen_active_2state(4096, L, EPS, GAMMA, PA, PM, erng)
    q_ev, qm_ev, next1_ev = active.forward_filter_2state(toks_ev, EPS, GAMMA, PA, PM)
    toks_ev_t = torch.tensor(toks_ev, device=device)

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            logits = model(toks_ev_t[:, :-1])
            p = torch.softmax(logits, -1)
            pm1 = (p[..., 1] / (p[..., 0] + p[..., 1] + 1e-9)).cpu().numpy()
        # model next1 at pos t predicts x_{t+1}; oracle next1_ev[:,t] same alignment
        skip = 20  # transient
        kl = kl_bern(next1_ev[:, skip:-1], pm1[:, skip:]).mean()
        return {"kl": float(kl)}

    # batch fn
    def batch_fn():
        tk, _ = active.gen_active_2state(256, L, EPS, GAMMA, PA, PM, rng)
        t = torch.tensor(tk, device=device)
        return t[:, :-1], t[:, 1:]

    cfg = GPTConfig(vocab_size=2, n_ctx=L, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
    model = TinyGPT(cfg)
    out = train.train(model, batch_fn, steps=6000, lr=1e-3, device=device,
                      snapshot_steps=[6000], eval_fn=eval_fn, log_every=2000)
    model.eval()
    fin = out["snapshots"][6000]["metrics"]
    print(f"C1a: final KL to filter oracle = {fin['kl']:.5f} nats")
    results = {"eps": EPS, "gamma": GAMMA, "pA": PA, "pM": PM, "qstar": QSTAR, "kl": fin["kl"],
               "state": {k: v.cpu() for k, v in model.state_dict().items()}, "cfg": cfg.__dict__}

    # ---- C1b: decode z_t = logit q_t (filtered) from residual at each position ----
    # target at input position t = z of filtered q_t (after seeing x_0..x_t)
    z_ev = np.log(np.clip(q_ev, 1e-4, 1-1e-4) / np.clip(1-q_ev, 1e-4, 1-1e-4))
    # use a slab of positions (skip transient), flatten over (batch, pos)
    pos_slab = list(range(40, L-1, 8))
    print("C1b: decode z_t=logit q_t vs running-count baseline (per layer):")
    c1b = {}
    for layer in range(cfg.n_layers):
        Xs, ys, cs = [], [], []
        for p in pos_slab:
            X = probe.extract_resid(model, toks_ev_t, layer=layer, pos=p)
            Xs.append(X); ys.append(z_ev[:, p])
            cs.append(toks_ev[:, :p+1].sum(axis=1))  # running count of 1s up to t
        X = np.concatenate(Xs); y = np.concatenate(ys); count = np.concatenate(cs)
        r2_z = probe.ridge_probe(X, y)["r2"]
        # baseline: how well does the running COUNT (best affine) predict z? (sufficiency check)
        r2_count_for_z = probe.ridge_probe(count.reshape(-1, 1), y, alpha=1e-6)["r2"]
        c1b[layer] = {"r2_z": r2_z, "r2_count_baseline": r2_count_for_z}
        print(f"  layer {layer}: R2(z from resid)={r2_z:.4f}   (count->z baseline {r2_count_for_z:.4f})")
    results["c1b_decode"] = c1b
    best_layer = max(c1b, key=lambda l: c1b[l]["r2_z"])
    # z direction at best layer (probe over the slab)
    Xs, ys = [], []
    for p in pos_slab:
        Xs.append(probe.extract_resid(model, toks_ev_t, layer=best_layer, pos=p)); ys.append(z_ev[:, p])
    zdir = probe.ridge_probe(np.concatenate(Xs), np.concatenate(ys))["direction"]
    zdir = zdir / (np.linalg.norm(zdir) + 1e-9)

    # ---- C1c: steering the z-direction shifts predicted alignment ----
    print("C1c: steer z-direction at all positions -> model mean implied q:")
    zt = torch.tensor(zdir, dtype=torch.float32, device=device)
    gen = torch.Generator().manual_seed(0)
    rdir = torch.randn(zt.shape, generator=gen).to(device); rdir = rdir / rdir.norm()
    def mean_implied_q(logits):
        p = torch.softmax(logits, -1)
        pm1 = (p[..., 1] / (p[..., 0] + p[..., 1] + 1e-9)).cpu().numpy()
        qhat = np.clip((pm1 - PA) / (PM - PA), 0, 1)  # implied q^-
        return float(qhat[:, 40:].mean())
    alphas = [-8, -4, -2, 0, 2, 4, 8]
    steer = {"alphas": alphas, "z": [], "rand": []}
    with torch.no_grad():
        for a in alphas:
            steer["z"].append(mean_implied_q(model.forward_with_steer(toks_ev_t[:, :-1], best_layer, zt, a)))
            steer["rand"].append(mean_implied_q(model.forward_with_steer(toks_ev_t[:, :-1], best_layer, rdir, a)))
    results["c1c_steer"] = {"best_layer": best_layer, **steer}
    print(f"  alpha {alphas}")
    print(f"  z-dir implied q: {[round(v,3) for v in steer['z']]}")
    print(f"  rand    implied q: {[round(v,3) for v in steer['rand']]}")

    # ---- C1e: drift under forced corrective (0s) / corrupting (1s) runs ----
    print("C1e: drift under forced 0-run (corrective) vs 1-run (corrupting):")
    pre = 40; run = 30
    base = active.gen_active_2state(2000, pre, EPS, GAMMA, PA, PM, np.random.default_rng(7))[0]
    drift = {}
    for label, bit in [("corrective_0s", 0), ("corrupting_1s", 1)]:
        seq = np.concatenate([base, np.full((2000, run), bit, dtype=np.int64)], axis=1)
        q, qm, n1 = active.forward_filter_2state(seq, EPS, GAMMA, PA, PM)
        st = torch.tensor(seq, device=device)
        with torch.no_grad():
            p = torch.softmax(model(st[:, :-1]), -1)
            pm1 = (p[..., 1] / (p[..., 0] + p[..., 1] + 1e-9)).cpu().numpy()
        mq = np.clip((pm1 - PA) / (PM - PA), 0, 1)  # model implied q^-
        # average over batch along the run (positions pre-1 .. pre+run-2 predict the run)
        orac_run = qm[:, pre:pre+run].mean(axis=0)
        model_run = mq[:, pre-1:pre-1+run].mean(axis=0)
        drift[label] = {"oracle_q": orac_run.tolist(), "model_q": model_run.tolist()}
        print(f"  {label}: oracle q {orac_run[0]:.3f}->{orac_run[-1]:.3f} | model {model_run[0]:.3f}->{model_run[-1]:.3f}")
    results["c1e_drift"] = drift

    torch.save(results, f"{OUT}/c1.pt")
    print(f"saved {OUT}/c1.pt")


if __name__ == "__main__":
    main()
