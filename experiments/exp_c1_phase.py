"""
C1d: (eps, gamma) phase diagram. Train one model per (eps,gamma); the model's mean
implied misalignment should track the stationary q* = eps/(eps+gamma) -- i.e. it
internalises the corruption/correction balance, not just one process. We use several
(eps,gamma) at different scales mapping to the same q* to show it is the RATIO that
matters.
"""
import os
import numpy as np
import torch
from bag_moments import active, train
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-sprint-2/results")
PA, PM, L = 0.3, 0.7, 200
GRID = [(0.02, 0.18), (0.05, 0.15), (0.1, 0.1), (0.15, 0.05),   # q* = .1 .25 .5 .75
        (0.04, 0.36), (0.1, 0.3), (0.2, 0.2), (0.3, 0.1)]        # same q* at 2x scale


def main():
    device = train.get_device()
    res = {"pA": PA, "pM": PM, "runs": []}
    erng = np.random.default_rng(123)
    for eps, gamma in GRID:
        qstar = active.stationary_q(eps, gamma)
        rng = np.random.default_rng(int(1000 * (eps + 2 * gamma)))
        def batch_fn():
            tk, _ = active.gen_active_2state(256, L, eps, gamma, PA, PM, rng)
            t = torch.tensor(tk, device=device); return t[:, :-1], t[:, 1:]
        torch.manual_seed(0)
        model = TinyGPT(GPTConfig(vocab_size=2, n_ctx=L, d_model=96, n_heads=4, n_layers=2, d_mlp=384))
        train.train(model, batch_fn, steps=4000, lr=1e-3, device=device, log_every=4000)
        model.eval()
        tk, _ = active.gen_active_2state(4096, L, eps, gamma, PA, PM, erng)
        st = torch.tensor(tk, device=device)
        with torch.no_grad():
            p = torch.softmax(model(st[:, :-1]), -1)
            pm1 = (p[..., 1] / (p[..., 0] + p[..., 1] + 1e-9)).cpu().numpy()
        implied_q = np.clip((pm1[:, 40:] - PA) / (PM - PA), 0, 1).mean()
        res["runs"].append({"eps": eps, "gamma": gamma, "qstar": qstar, "implied_q": float(implied_q)})
        print(f"eps={eps} gamma={gamma} q*={qstar:.3f} model_meanq={implied_q:.3f}", flush=True)
    torch.save(res, f"{OUT}/c1_phase.pt")
    print(f"saved {OUT}/c1_phase.pt")


if __name__ == "__main__":
    main()
