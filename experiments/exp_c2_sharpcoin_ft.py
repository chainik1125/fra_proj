"""
C2 sharp-coin OOD fault-tolerance — closes the §7 "soft fault-tolerance" caveat.

The n=5 logical model was TRAINED on SOFT coins (pA=0.3, pM=0.7 — lots of per-chain overlap, so
even the Bayes majority decode is soft). §7 flags: "sharper coins would give a crisper threshold but
are OOD for the trained model." Here we feed the SAME trained model controlled k-corrupted inputs at
INCREASING emission sharpness (0.3/0.7 in-dist → 0.2/0.8 → 0.1/0.9 → 0.05/0.95, all OOD) and ask:
  (A) does the error-correcting threshold flip at k=r+1=3 PERSIST out of distribution?
  (B) does it SHARPEN as the coins sharpen (margin between k=2 and k=3 grows) — i.e. does the model
      exploit the cleaner per-chain evidence, the signature of a real threshold MECHANISM rather than
      a memorised soft-coin marginal?
  (C) how does the model's confidence compare to the sharp-appropriate Bayes oracle (poisson-binomial
      tail of the per-chain filter evaluated AT the sharp rates) — honest calibration check OOD.

Eval-only (no training). Reuses the validated per-chain filter + poisson-binomial tail oracle.
Outputs results/c2_sharpcoin_ft.pt
"""
import os, sys
import numpy as np
import torch
ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from bag_moments import active
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", os.path.join(ROOT, "results"))
B = int(os.environ.get("SC_B", "3000"))
# emission-sharpness ladder: (pA, pM). First is the in-distribution training pair.
LADDER = [(0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.05, 0.95)]


def main():
    r = torch.load(f"{OUT}/c2_logical.pt", map_location="cpu", weights_only=False)
    EPS, GAMMA, T = r["eps"], r["gamma"], r["T"]
    cfg = GPTConfig(**r["fixed5_cfg"]); model = TinyGPT(cfg)
    model.load_state_dict(r["fixed5_state"]); model.eval()
    n = 5; rdec = (n - 1) // 2
    QUERY = 2 ** n; R_yes = 2 ** n + 2
    rng = np.random.default_rng(0)
    print(f"=== C2 sharp-coin OOD fault-tolerance (n={n}, r={rdec}, trained pA/pM=0.3/0.7, "
          f"eps={EPS}, gamma={GAMMA}, T={T}) ===", flush=True)

    out = {"n": n, "r": rdec, "eps": EPS, "gamma": GAMMA, "T": T, "ladder": LADDER, "by_pair": {}}
    for (pA, pM) in LADDER:
        # fixed corruption-count inputs: exactly k chains persistently misaligned (Bernoulli pM)
        model_p, orac_p = [], []
        for k in range(n + 1):
            mis = np.zeros((B, n), dtype=bool)
            for b in range(B):
                mis[b, rng.choice(n, size=k, replace=False)] = True
            p = np.where(mis[:, None, :], pM, pA)               # (B,1,n) broadcast over T
            bits = (rng.random((B, T, n)) < p).astype(np.int64)
            packed = active.pack_emissions(bits)
            seq = np.concatenate([packed, np.full((B, 1), QUERY)], axis=1)
            with torch.no_grad():
                logits = model(torch.tensor(seq))
                pp = torch.softmax(logits[:, -1], -1).numpy()
            p_yes = pp[:, R_yes] / (pp[:, R_yes] + pp[:, R_yes - 1] + 1e-9)
            # sharp-appropriate Bayes oracle: per-chain filter AT the sharp rates, poisson-binomial tail
            qchain = active.per_chain_filter(bits, EPS, GAMMA, pA, pM)[:, -1, :]
            orac = active.poisson_binomial_tail(qchain, rdec)
            model_p.append(float(p_yes.mean())); orac_p.append(float(orac.mean()))
        model_p = np.array(model_p); orac_p = np.array(orac_p)
        # threshold location (first k with mean P(mis)>0.5) and flip sharpness (margin across k=r..r+1)
        thr_model = int(np.argmax(model_p > 0.5)) if (model_p > 0.5).any() else -1
        thr_orac = int(np.argmax(orac_p > 0.5)) if (orac_p > 0.5).any() else -1
        margin_model = float(model_p[rdec + 1] - model_p[rdec])      # k=3 minus k=2 (the decision gap)
        margin_orac = float(orac_p[rdec + 1] - orac_p[rdec])
        out["by_pair"][(pA, pM)] = {"model_p_yes": model_p.tolist(), "oracle_p_yes": orac_p.tolist(),
                                    "thr_model": thr_model, "thr_orac": thr_orac,
                                    "margin_model": margin_model, "margin_orac": margin_orac}
        flip = "✓@3" if thr_model == rdec + 1 else f"✗@{thr_model}"
        print(f"[pA/pM={pA}/{pM}] model P(mis) by k: "
              f"{np.array2string(model_p, precision=3, floatmode='fixed')} | flip {flip} "
              f"| margin(k3-k2) model={margin_model:+.3f} orac={margin_orac:+.3f}", flush=True)
        print(f"            oracle by k: {np.array2string(orac_p, precision=3, floatmode='fixed')}",
              flush=True)
    torch.save(out, f"{OUT}/c2_sharpcoin_ft.pt")
    print(f"saved {OUT}/c2_sharpcoin_ft.pt", flush=True)


if __name__ == "__main__":
    main()
