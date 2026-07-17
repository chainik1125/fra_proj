"""
Experiment C (Claim C3): the same-source structure is linearly decodable AND the
cross-rollout transfer is causally carried by a low-dimensional direction.

Loads the trained chi2=0.5 (fixed N=2) quenched model from exp_quenched.py.

Important subtlety: at the *first* token of rollout 2 the Bayes collision posterior
q = P(same coin | context) equals the prior chi2 for EVERY sequence (the marginal
likelihoods cancel), so q has no variance there.  What varies at that position is
rollout-1's evidence Laplace(s1,t1), mixed with weight chi2.  The collision posterior
q only *varies* once rollout 2 starts accumulating bits.  Hence:

  (C3a) same-source coordinate: probe the residual stream during rollout 2 for the
        exact updating q.  Real variance -> meaningful R².  Shuffled control ~0.
  (C3b) causal transfer: ablate the rollout-1-evidence direction (Laplace(s1)) at the
        ROLL position; the effective transfer chi2_hat should collapse toward 0.
        Controls: random direction of equal norm; clean model.
  (C3c) causal same-source: ablate the q-direction at a mid rollout-2 position and
        measure the drop in the model's same/different sensitivity.

Outputs results/mechanism.pt
"""

import os

import numpy as np
import torch

from bag_moments import metrics, oracles, probe
from bag_moments.model import GPTConfig, TinyGPT

RES = os.environ.get("BAG_OUT", "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-sprint/results")


def load_fixed2_model(device):
    r = torch.load(f"{RES}/quenched_sweep.pt", map_location="cpu", weights_only=False)
    run = r["priors"]["fixed2"]["runs"][0]
    cfg = GPTConfig(**run["cfg"])
    model = TinyGPT(cfg)
    model.load_state_dict(run["state"])
    model.to(device).eval()
    return model, cfg, r["roll_lens"]


def main():
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model, cfg, roll_lens = load_fixed2_model(device)
    nprior = oracles.fixed_n(2)
    ev = metrics.build_quenched_eval(roll_lens, nprior, 8192, seed=4242, device=device)
    chi2 = ev["chi2"]; t1 = ev["t1"]; roll_pos = ev["roll_pos"]
    toks = ev["tokens"]
    orac_q = ev["orac_q"]            # (B, seqlen-1) Bayes collision posterior
    laplace_s1 = ev["laplace_s1"]    # (B,) rollout-1 posterior mean
    results = {"chi2": chi2, "roll_lens": roll_lens, "n_layers": cfg.n_layers}

    # ---------- (C3a) same-source posterior q decodable during rollout 2 ----------
    print("(C3a) decode updating collision posterior q during rollout 2:")
    r2_positions = list(range(roll_pos, ev["mask"].shape[1]))  # ROLL pos + rollout-2 bits
    pos_layer_r2 = {}
    for p in r2_positions:
        qp = orac_q[:, p].numpy()
        if np.isnan(qp).all() or np.nanstd(qp) < 1e-6:
            continue  # ROLL position: q is constant (= prior), no variance
        valid = ~np.isnan(qp)
        best = -1.0; bl = -1
        for layer in range(cfg.n_layers):
            X = probe.extract_resid(model, toks, layer=layer, pos=p)
            r2 = probe.ridge_probe(X[valid], qp[valid])["r2"]
            if r2 > best:
                best, bl = r2, layer
        pos_layer_r2[p] = {"best_r2": best, "best_layer": bl, "q_std": float(np.nanstd(qp))}
        print(f"  pos {p} (q_std={np.nanstd(qp):.3f}): best R²(q)={best:.4f} @ layer {bl}")
    results["q_decode"] = pos_layer_r2
    # shuffled control at the first informative rollout-2 position
    first_inf = min(pos_layer_r2)
    qp = orac_q[:, first_inf].numpy(); valid = ~np.isnan(qp)
    bl = pos_layer_r2[first_inf]["best_layer"]
    Xc = probe.extract_resid(model, toks, layer=bl, pos=first_inf)
    rng = np.random.default_rng(0)
    shuf = probe.ridge_probe(Xc[valid], rng.permutation(qp[valid]))["r2"]
    results["q_shuffled_r2"] = shuf
    print(f"  shuffled-q control @ pos {first_inf}: R²={shuf:.4f} (should be ~0)")

    # ---------- (C3b) causal: ablate rollout-1 evidence at ROLL -> kill transfer ----------
    # Decisive test must ablate at the LAST layer (no downstream recomputation of s1
    # from the intact rollout-1 tokens via attention).  Mean-ablation = remove the
    # variance along the direction by pinning the projection to its dataset mean.
    print("\n(C3b) causal transfer: mean-ablate Laplace(s1)-direction at ROLL, per layer:")
    layer_r2_s1 = {}
    directions = {}
    for layer in range(cfg.n_layers):
        X = probe.extract_resid(model, toks, layer=layer, pos=roll_pos)
        pr = probe.ridge_probe(X, laplace_s1, alpha=1.0)
        layer_r2_s1[layer] = pr["r2"]
        d = pr["direction"]
        d = d / (np.linalg.norm(d) + 1e-9)
        directions[layer] = (torch.tensor(d, dtype=torch.float32, device=device),
                             float((X @ d).mean()))
        print(f"  layer {layer}: R²(Laplace(s1)) = {pr['r2']:.4f}")
    results["layer_r2_s1"] = layer_r2_s1

    def chi2_hat_from_logits(logits):
        pr = torch.softmax(logits, dim=-1)
        p1 = pr[:, roll_pos, 1]; p0 = pr[:, roll_pos, 0]
        first = (p1 / (p0 + p1 + 1e-9)).detach().cpu().numpy()
        beta, alpha, r2 = metrics.fit_transfer_coef(first, laplace_s1)
        return beta

    with torch.no_grad():
        b_clean = chi2_hat_from_logits(model(toks))
        per_layer = {}
        for layer in range(cfg.n_layers):
            d, mproj = directions[layer]
            per_layer[layer] = chi2_hat_from_logits(
                model.forward_with_ablation(toks, layer, d, target_value=mproj,
                                            positions=[roll_pos]))
        last = cfg.n_layers - 1
        d, mproj = directions[last]
        gen = torch.Generator().manual_seed(0)
        rnd = torch.randn(d.shape, generator=gen).to(device); rnd = rnd / rnd.norm()
        b_rnd = chi2_hat_from_logits(
            model.forward_with_ablation(toks, last, rnd, target_value=mproj, positions=[roll_pos]))
    results["causal_transfer"] = {"chi2_true": chi2, "chi2_hat_clean": b_clean,
                                  "chi2_hat_ablate_by_layer": per_layer,
                                  "chi2_hat_ablate_random_lastlayer": b_rnd}
    print(f"  chi2 true={chi2:.3f}  clean chi2_hat={b_clean:.3f}")
    for layer in range(cfg.n_layers):
        print(f"    mean-ablate @ layer {layer}: chi2_hat = {per_layer[layer]:.3f}")
    print(f"    random-dir @ last layer:   chi2_hat = {b_rnd:.3f}")

    # ---------- (C3c) causal same-source: ablate q-direction mid rollout 2 ----------
    # sensitivity = slope of model next-bit pred vs (pred_same - pred_diff) at pos p.
    # The oracle weight on that contrast is q; ablating q should drop the slope.
    print("\n(C3c) causal same-source ablation mid rollout 2:")
    # choose a mid rollout-2 position with good q variance and decodability
    cand = [p for p in pos_layer_r2 if pos_layer_r2[p]["best_r2"] > 0.8]
    if cand:
        p = cand[len(cand) // 2]
        ql = pos_layer_r2[p]["best_layer"]
        qp = orac_q[:, p].numpy(); valid = ~np.isnan(qp)
        Xq = probe.extract_resid(model, toks, layer=ql, pos=p)
        qpr = probe.ridge_probe(Xq[valid], qp[valid])
        qd = qpr["direction"]; qd = qd / (np.linalg.norm(qd) + 1e-9)
        qdir = torch.tensor(qd, dtype=torch.float32, device=device)
        q_mproj = float((Xq @ qd).mean())
        # contrast (pred_same - pred_diff) at position p
        # reconstruct running counts at position p
        r2_start = roll_pos + 1
        k = p - r2_start + 1   # number of rollout-2 bits seen at input pos p (>=1); ROLL pos handled above
        bits2 = ev["qb"].tokens[:, r2_start : r2_start + roll_lens[1]]
        s2 = bits2[:, :k].sum(axis=1) if k >= 1 else np.zeros(len(qp))
        s1 = ev["s1"]
        pred_same = (s1 + s2 + 1.0) / (t1 + k + 2.0)
        pred_diff = (s2 + 1.0) / (k + 2.0)
        contrast = (pred_same - pred_diff)

        def sameslope(logits):
            pr = torch.softmax(logits, dim=-1)
            mp = (pr[:, p, 1] / (pr[:, p, 0] + pr[:, p, 1] + 1e-9)).detach().cpu().numpy()
            A = np.stack([np.ones_like(contrast), pred_diff, contrast], 1)
            coef, *_ = np.linalg.lstsq(A[valid], mp[valid], rcond=None)
            return float(coef[2])  # slope on (pred_same - pred_diff) ~ effective same-weight

        with torch.no_grad():
            s_clean = sameslope(model(toks))
            s_abl = sameslope(model.forward_with_ablation(toks, ql, qdir, target_value=q_mproj, positions=[p]))
            gen = torch.Generator().manual_seed(1)
            rnd = torch.randn(qdir.shape, generator=gen).to(device); rnd = rnd / rnd.norm()
            s_rnd = sameslope(model.forward_with_ablation(toks, ql, rnd, target_value=q_mproj, positions=[p]))
        results["causal_same"] = {"pos": p, "layer": ql, "same_slope_clean": s_clean,
                                  "same_slope_ablate_q": s_abl, "same_slope_ablate_random": s_rnd}
        print(f"  pos {p}: same-weight clean={s_clean:.3f}  ablate-q={s_abl:.3f}  "
              f"ablate-random={s_rnd:.3f}")

    # ---------- (C3d) causal: sever cross-rollout attention -> kill transfer ----------
    # Block rollout-2 / ROLL queries from attending to rollout-1 KEYS in every layer.
    # This is the only pathway for cross-rollout information, so transfer must vanish,
    # while rollout-1's own predictions (which never attend forward) stay intact.
    print("\n(C3d) causal: knock out rollout-2 -> rollout-1 attention:")
    inp = toks[:, :-1]                  # input convention: predict token i+1 from i
    Sin = inp.shape[1]
    blocked = torch.zeros(Sin, Sin, dtype=torch.bool, device=device)
    blocked[roll_pos:, :t1] = True  # queries at ROLL+rollout-2, keys = rollout-1 bits
    mask_all = ev["mask"]; mask_r2 = ev["r2mask"]
    mask_r1 = mask_all & (~mask_r2)
    with torch.no_grad():
        logits_clean = model(inp)
        logits_ko = model.forward_attn_knockout(inp, blocked)
    p1_clean = torch.softmax(logits_clean, -1)
    p1_clean = (p1_clean[..., 1] / (p1_clean[..., 0] + p1_clean[..., 1] + 1e-9))
    p1_ko = torch.softmax(logits_ko, -1)
    p1_ko = (p1_ko[..., 1] / (p1_ko[..., 0] + p1_ko[..., 1] + 1e-9))
    # effective chi2 under knockout
    first_ko = p1_ko[:, roll_pos].detach().cpu().numpy()
    chi2_ko, _, _ = metrics.fit_transfer_coef(first_ko, laplace_s1)
    chi2_cl, _, _ = metrics.fit_transfer_coef(p1_clean[:, roll_pos].detach().cpu().numpy(), laplace_s1)
    kl_r1_ko = metrics.kl_to_oracle(p1_ko, ev["orac_p1"], mask_r1)
    kl_r2_ko = metrics.kl_to_oracle(p1_ko, ev["orac_p1"], mask_r2)
    kl_r1_cl = metrics.kl_to_oracle(p1_clean, ev["orac_p1"], mask_r1)
    kl_r2_cl = metrics.kl_to_oracle(p1_clean, ev["orac_p1"], mask_r2)
    results["causal_attn_knockout"] = {
        "chi2_clean": chi2_cl, "chi2_knockout": chi2_ko,
        "kl_r1_clean": kl_r1_cl, "kl_r1_knockout": kl_r1_ko,
        "kl_r2_clean": kl_r2_cl, "kl_r2_knockout": kl_r2_ko}
    print(f"  chi2_hat:  clean={chi2_cl:.3f}  cross-attn-knockout={chi2_ko:.3f}")
    print(f"  rollout-1 KL: clean={kl_r1_cl:.4f}  knockout={kl_r1_ko:.4f}  (should be ~unchanged)")
    print(f"  rollout-2 KL: clean={kl_r2_cl:.4f}  knockout={kl_r2_ko:.4f}  (should rise: transfer lost)")

    torch.save(results, f"{RES}/mechanism.pt")
    print("\nsaved results/mechanism.pt")


if __name__ == "__main__":
    main()
