"""
C1c (improved): causal steering of the alignment coordinate with a DIFF-OF-MEANS vector
(the canonical steering recipe, à la refusal-direction work) rather than the regression
probe direction. For each layer, z_dir = mean(resid | q high) - mean(resid | q low);
steer by adding alpha * unit(z_dir) and measure the model's mean implied misalignment q.
Compare to a random direction of equal norm. A working alignment knob => implied q rises
monotonically with alpha for the z_dir but not the random dir.
"""
import os
import numpy as np
import torch
from bag_moments import active, probe
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-sprint-2/results")
EPS, GAMMA, PA, PM, L = 0.05, 0.15, 0.3, 0.7, 200


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r = torch.load(f"{OUT}/c1.pt", map_location="cpu", weights_only=False)
    cfg = GPTConfig(**r["cfg"]); model = TinyGPT(cfg); model.load_state_dict(r["state"]); model.to(device).eval()
    erng = np.random.default_rng(2024)
    toks, hid = active.gen_active_2state(4096, L, EPS, GAMMA, PA, PM, erng)
    q, qm, n1 = active.forward_filter_2state(toks, EPS, GAMMA, PA, PM)
    st = torch.tensor(toks, device=device)

    pos_slab = list(range(40, L, 10))
    def implied_q(logits):
        p = torch.softmax(logits, -1)
        pm1 = (p[..., 1] / (p[..., 0] + p[..., 1] + 1e-9)).cpu().numpy()
        return float(np.clip((pm1[:, 40:] - PA) / (PM - PA), 0, 1).mean())

    alphas = [-12, -8, -4, 0, 4, 8, 12]
    results = {"alphas": alphas, "layers": {}}
    for layer in range(cfg.n_layers):
        # diff-of-means direction
        Xs, qs = [], []
        for p in pos_slab:
            Xs.append(probe.extract_resid(model, st, layer=layer, pos=p)); qs.append(q[:, p])
        X = np.concatenate(Xs); qq = np.concatenate(qs)
        hi = qq >= np.quantile(qq, 0.7); lo = qq <= np.quantile(qq, 0.3)
        zdir = X[hi].mean(0) - X[lo].mean(0)
        zt = torch.tensor(zdir / (np.linalg.norm(zdir) + 1e-9), dtype=torch.float32, device=device)
        gen = torch.Generator().manual_seed(layer)
        rdir = torch.randn(zt.shape, generator=gen).to(device); rdir = rdir / rdir.norm()
        zc, rc = [], []
        with torch.no_grad():
            for a in alphas:
                zc.append(implied_q(model.forward_with_steer(st[:, :-1], layer, zt, a)))
                rc.append(implied_q(model.forward_with_steer(st[:, :-1], layer, rdir, a)))
        results["layers"][layer] = {"z": zc, "rand": rc}
        print(f"layer {layer}: z-steer implied q {[round(v,3) for v in zc]}")
        print(f"          rand-steer implied q {[round(v,3) for v in rc]}")
    torch.save(results, f"{OUT}/c1_steer.pt")
    print(f"saved {OUT}/c1_steer.pt")


if __name__ == "__main__":
    main()
