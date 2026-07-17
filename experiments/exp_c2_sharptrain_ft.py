"""
Control for the sharp-coin OOD undershoot (§4.3 Fig 3c): does the soft-trained model's failure to
reach the crisp k=r+1 step on sharp coins reflect a TRAIN/TEST MISMATCH or an architectural limit?

We TRAIN a fresh n=5 logical model directly on SHARP coins (pA=0.1, pM=0.9) — same architecture &
recipe as exp_c2_logical — and run the same fixed-k fault-tolerance eval at those sharp coins. If the
sharp-TRAINED model flips crisply at the ideal threshold k=r+1=3 (matching the sharp Bayes oracle),
then the soft-trained model's OOD undershoot is a distribution-shift effect, not a capability limit.

Oracle = validated per-chain filter + poisson-binomial tail at the SHARP rates.
Outputs results/c2_sharptrain_ft.pt
"""
import os, sys, time
import numpy as np
import torch
ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from bag_moments import active, train
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", os.path.join(ROOT, "results"))
torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
EPS, GAMMA, T = 0.05, 0.15, 40
PA, PM = float(os.environ.get("ST_PA", "0.1")), float(os.environ.get("ST_PM", "0.9"))
N, STEPS, BATCH = 5, int(os.environ.get("ST_STEPS", "2000")), int(os.environ.get("ST_BATCH", "192"))
R = (N - 1) // 2
S = 1 << N


def make_seqs(B, rng):
    tk, hid, _ = active.gen_redundant_active(B, T, N, EPS, GAMMA, PA, PM, rng, beta=0.0)
    packed = active.pack_emissions(tk)
    true_L = (hid[:, -1, :].sum(1) > R).astype(np.int64)
    QUERY, R_no, R_yes = S, S + 1, S + 2
    readout = np.where(true_L == 1, R_yes, R_no)
    seq = np.concatenate([packed, np.full((B, 1), QUERY), readout[:, None]], axis=1)
    return seq, {"true_L": true_L, "R_yes": R_yes}


def main():
    rng = np.random.default_rng(5)
    cfg = GPTConfig(vocab_size=S + 3, n_ctx=T + 2, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
    model = TinyGPT(cfg)

    def batch_fn():
        s, _ = make_seqs(BATCH, rng); t = torch.tensor(s); return t[:, :-1], t[:, 1:]

    print(f"=== C2 sharp-TRAINED fault tolerance (n={N}, trained pA/pM={PA}/{PM}, {STEPS} steps) ===",
          flush=True)
    t0 = time.time()
    train.train(model, batch_fn, steps=STEPS, lr=1e-3, device="cpu",
                snapshot_steps=[STEPS], log_every=max(STEPS // 2, 1))
    model.eval()
    QUERY, R_yes = S, S + 2
    erng = np.random.default_rng(777); B = 3000
    model_p, orac_p = [], []
    for k in range(N + 1):
        mis = np.zeros((B, N), dtype=bool)
        for b in range(B):
            mis[b, erng.choice(N, size=k, replace=False)] = True
        p = np.where(mis[:, None, :], PM, PA)
        bits = (erng.random((B, T, N)) < p).astype(np.int64)
        seq = np.concatenate([active.pack_emissions(bits), np.full((B, 1), QUERY)], axis=1)
        with torch.no_grad():
            pp = torch.softmax(model(torch.tensor(seq))[:, -1], -1).numpy()
        p_yes = pp[:, R_yes] / (pp[:, R_yes] + pp[:, R_yes - 1] + 1e-9)
        qchain = active.per_chain_filter(bits, EPS, GAMMA, PA, PM)[:, -1, :]
        model_p.append(float(p_yes.mean())); orac_p.append(float(active.poisson_binomial_tail(qchain, R).mean()))
    model_p, orac_p = np.array(model_p), np.array(orac_p)
    thr_model = int(np.argmax(model_p > 0.5)) if (model_p > 0.5).any() else -1
    margin_model = float(model_p[R + 1] - model_p[R]); margin_orac = float(orac_p[R + 1] - orac_p[R])
    out = {"pA": PA, "pM": PM, "n": N, "r": R, "steps": STEPS,
           "model_p_yes": model_p.tolist(), "oracle_p_yes": orac_p.tolist(),
           "thr_model": thr_model, "margin_model": margin_model, "margin_orac": margin_orac}
    torch.save(out, f"{OUT}/c2_sharptrain_ft.pt")
    print(f"model P(mis) by k: {np.array2string(model_p, precision=3, floatmode='fixed')}", flush=True)
    print(f"oracle by k:       {np.array2string(orac_p, precision=3, floatmode='fixed')}", flush=True)
    print(f"flip at k={thr_model} (ideal {R+1}) | margin(k{R+1}-k{R}) model={margin_model:+.3f} "
          f"orac={margin_orac:+.3f} | k={R+1} readout model={model_p[R+1]:.3f} oracle={orac_p[R+1]:.3f} "
          f"({(time.time()-t0)/60:.1f}min)", flush=True)
    print(f"saved {OUT}/c2_sharptrain_ft.pt", flush=True)


if __name__ == "__main__":
    main()
