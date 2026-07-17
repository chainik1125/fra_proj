"""
C2 fault-tolerance: the model's logical readout tolerates up to r=(n-1)/2 corrupted
blocks. We feed controlled inputs where exactly k of the n chains are PERSISTENTLY
misaligned (emit Bernoulli(pM)) and n-k are persistently aligned (Bernoulli(pA)), then
query. The Bayes/code answer flips at k=r+1; a working majority-decoder shows the same
threshold, demonstrating it implements the error-correcting decode (not a single-block
readout). Loads the n=5 query model from exp_c2_logical.
"""
import os
import numpy as np
import torch
from bag_moments import active
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-sprint-2/results")
EPS, GAMMA, PA, PM, T = 0.05, 0.15, 0.3, 0.7, 40


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r = torch.load(f"{OUT}/c2_logical.pt", map_location="cpu", weights_only=False)
    cfg = GPTConfig(**r["fixed5_cfg"]); model = TinyGPT(cfg); model.load_state_dict(r["fixed5_state"])
    model.to(device).eval()
    n = 5; rdec = (n - 1) // 2
    QUERY = 2 ** n; R_yes = 2 ** n + 2
    rng = np.random.default_rng(0)
    B = 3000
    res = {"n": n, "r": rdec, "k": [], "model_p_yes": [], "oracle_p_yes": []}
    for k in range(n + 1):
        # build B sequences: choose which k chains are persistently misaligned
        bits = np.empty((B, T, n), dtype=np.int64)
        mis = np.zeros((B, n), dtype=bool)
        for b in range(B):
            idx = rng.choice(n, size=k, replace=False)
            mis[b, idx] = True
        p = np.where(mis[:, None, :], PM, PA)            # (B,1,n)->broadcast
        bits = (rng.random((B, T, n)) < p).astype(np.int64)
        packed = active.pack_emissions(bits)
        seq = np.concatenate([packed, np.full((B, 1), QUERY)], axis=1)
        st = torch.tensor(seq, device=device)
        with torch.no_grad():
            logits = model(st)               # predict readout at QUERY (last) position
            pp = torch.softmax(logits[:, -1], -1).cpu().numpy()
        p_yes = pp[:, R_yes] / (pp[:, R_yes] + pp[:, R_yes - 1] + 1e-9)
        # oracle: poisson-binomial tail of per-chain posteriors
        qchain = active.per_chain_filter(bits, EPS, GAMMA, PA, PM)[:, -1, :]
        orac = active.poisson_binomial_tail(qchain, rdec)
        res["k"].append(k); res["model_p_yes"].append(float(p_yes.mean()))
        res["oracle_p_yes"].append(float(orac.mean()))
        print(f"k={k} (>{rdec}? {'MIS' if k>rdec else 'ALN'}): model P(misaligned)={p_yes.mean():.3f}  oracle={orac.mean():.3f}")
    torch.save(res, f"{OUT}/c2_faulttol.pt")
    print(f"saved {OUT}/c2_faulttol.pt")


if __name__ == "__main__":
    main()
