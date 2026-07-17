"""
Deflationary controls demanded by the red-team review, to correctly BOUND the C3
claims (run on the chi2=0.5 / N=2 quenched model).

  P1: Is "q is decodable at R^2~0.9" more than "counts are represented and q is a
      smooth function of counts"?  Baseline: regress the oracle q on affine features
      of the counts the model provably holds {Laplace(s1), Laplace(s2), s1, s2,
      Laplace(s1)*Laplace(s2)}.  The probe must beat THIS, not just shuffled noise.

  P2: Real attention-specificity control.  The cross-knockout (rollout2 -/-> rollout1)
      kills transfer, but rollout-1 KL is unchanged *by construction* (causal mask).
      Sham control: block rollout-2 queries from EARLIER rollout-2 keys (within-rollout
      backward attention) -- an equal-spirit knockout that does NOT touch the cross
      edge.  If rollout-2 KL stays low, the cross edge is the specific carrier.

  P3: Is the single-direction-ablation null just underpowered (wrong layer/position)?
      Maximal test: mean-ablate the s1 direction at EVERY position and EVERY layer at
      once.  If chi2_hat still survives, the readout genuinely does not run through the
      linear s1 direction (redundant/distributed), not just "we picked the wrong spot."
"""

import os
import numpy as np
import torch

from bag_moments import metrics, oracles, probe
from bag_moments.model import GPTConfig, TinyGPT

RES = os.environ.get("BAG_OUT", "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-sprint/results")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r = torch.load(f"{RES}/quenched_sweep.pt", map_location="cpu", weights_only=False)
    run = r["priors"]["fixed2"]["runs"][0]
    cfg = GPTConfig(**run["cfg"]); model = TinyGPT(cfg); model.load_state_dict(run["state"])
    model.to(device).eval()
    roll_lens = r["roll_lens"]; t1, t2 = roll_lens
    ev = metrics.build_quenched_eval(roll_lens, oracles.fixed_n(2), 8192, seed=4242, device=device)
    toks = ev["tokens"]; roll_pos = ev["roll_pos"]; orac_q = ev["orac_q"]
    laplace_s1 = ev["laplace_s1"]; s1 = ev["s1"].astype(float)
    r2_start = roll_pos + 1
    bits2 = ev["qb"].tokens[:, r2_start : r2_start + t2]
    results = {}

    # ----------------- P1: count-feature baseline for q -----------------
    print("P1: q-decode vs count-feature baseline (model probe R2 was ~0.99->0.90)")
    print("  pos | R2(q ~ count features) | gap to model probe")
    p1 = {}
    for k in range(1, t2):  # rollout-2 positions with k bits seen
        pos = r2_start + k - 1  # input index predicting bit k (k>=1); pos>=roll_pos
        qp = orac_q[:, pos].numpy()
        if np.isnan(qp).all() or np.nanstd(qp) < 1e-6:
            continue
        valid = ~np.isnan(qp)
        s2 = bits2[:, :k].sum(axis=1).astype(float)
        lap_s1 = (s1 + 1) / (t1 + 2); lap_s2 = (s2 + 1) / (k + 2)
        feats = np.stack([lap_s1, lap_s2, s1, s2, lap_s1 * lap_s2], axis=1)
        base_r2 = probe.ridge_probe(feats[valid], qp[valid], alpha=1e-3)["r2"]
        # model probe at this position (best layer = last)
        X = probe.extract_resid(model, toks, layer=cfg.n_layers - 1, pos=pos)
        model_r2 = probe.ridge_probe(X[valid], qp[valid])["r2"]
        p1[pos] = {"count_baseline_r2": base_r2, "model_r2": model_r2}
        print(f"  {pos:3d} | {base_r2:6.3f} | model {model_r2:.3f}  gap {model_r2-base_r2:+.3f}")
    results["P1_q_vs_counts"] = p1

    # ----------------- helpers -----------------
    mask_all = ev["mask"]; mask_r2 = ev["r2mask"]; mask_r1 = mask_all & (~mask_r2)
    inp = toks[:, :-1]; Sin = inp.shape[1]

    def kl_split(logits):
        p = torch.softmax(logits, -1); p1m = p[..., 1] / (p[..., 0] + p[..., 1] + 1e-9)
        return (metrics.kl_to_oracle(p1m, ev["orac_p1"], mask_r1),
                metrics.kl_to_oracle(p1m, ev["orac_p1"], mask_r2),
                metrics.fit_transfer_coef(p1m[:, roll_pos].cpu().numpy(), laplace_s1)[0])

    # ----------------- P2: sham within-rollout-2 knockout -----------------
    print("\nP2: attention knockouts (rollout-2 KL; clean baseline first)")
    with torch.no_grad():
        kr1, kr2, chi = kl_split(model(inp))
        print(f"  clean:            r1KL={kr1:.4f} r2KL={kr2:.4f} chi2_hat={chi:.3f}")
        # true cross knockout
        cross = torch.zeros(Sin, Sin, dtype=torch.bool, device=device); cross[roll_pos:, :t1] = True
        kr1c, kr2c, chic = kl_split(model.forward_attn_knockout(inp, cross))
        print(f"  cross (r2->r1):   r1KL={kr1c:.4f} r2KL={kr2c:.4f} chi2_hat={chic:.3f}")
        # sham: block rollout-2 (post-ROLL) queries from earlier rollout-2 keys
        sham = torch.zeros(Sin, Sin, dtype=torch.bool, device=device)
        for q in range(r2_start, Sin):
            sham[q, r2_start:q] = True  # within-rollout-2 backward edges
        kr1s, kr2s, chis = kl_split(model.forward_attn_knockout(inp, sham))
        print(f"  sham (r2->r2):    r1KL={kr1s:.4f} r2KL={kr2s:.4f} chi2_hat={chis:.3f}")
    results["P2_knockouts"] = {"clean": (kr1, kr2, chi), "cross": (kr1c, kr2c, chic),
                               "sham_within_r2": (kr1s, kr2s, chis)}

    # ----------------- P3: maximal s1-direction ablation -----------------
    print("\nP3: maximal s1-direction ablation (all positions, each layer)")
    # build s1-direction per layer at ROLL
    dirs = {}
    for L in range(cfg.n_layers):
        X = probe.extract_resid(model, toks, layer=L, pos=roll_pos)
        d = probe.ridge_probe(X, laplace_s1)["direction"]; d = d / (np.linalg.norm(d) + 1e-9)
        dirs[L] = (torch.tensor(d, dtype=torch.float32, device=device), float((X @ d).mean()))
    all_pos = list(range(Sin))
    with torch.no_grad():
        chi_clean = metrics.fit_transfer_coef(
            (torch.softmax(model(inp), -1)[:, roll_pos, 1]).cpu().numpy() /
            (torch.softmax(model(inp), -1)[:, roll_pos, 0] + torch.softmax(model(inp), -1)[:, roll_pos, 1] + 1e-9).cpu().numpy(),
            laplace_s1)[0]
        p3 = {}
        for L in range(cfg.n_layers):
            d, mproj = dirs[L]
            lg = model.forward_with_ablation(inp, L, d, target_value=mproj, positions=all_pos)
            p = torch.softmax(lg, -1); p1m = (p[:, roll_pos, 1] / (p[:, roll_pos, 0] + p[:, roll_pos, 1] + 1e-9)).cpu().numpy()
            p3[L] = metrics.fit_transfer_coef(p1m, laplace_s1)[0]
            print(f"  ablate s1-dir @ layer {L}, ALL positions: chi2_hat={p3[L]:.3f}")
    results["P3_maximal_ablation"] = {"clean": float(chi_clean), "by_layer_allpos": p3}

    torch.save(results, f"{RES}/controls.pt")
    print("\nsaved results/controls.pt")


if __name__ == "__main__":
    main()
