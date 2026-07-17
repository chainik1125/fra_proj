"""
SEED-ROBUST OOD fault-tolerance (hardens the §4.3 / Fig 3b "decisive test").

The headline C2-transformer claim — the model implements a *threshold decoder* (error-correcting
code), not a memorised marginal — rests on the OOD fault-tolerance flip: fed inputs with exactly k
of n=5 blocks PERSISTENTLY corrupted, the readout flips at the code threshold k=r+1=3. But the
committed result (exp_c2_faulttolerance.py) reads a SINGLE seed-0 model from c2_logical.pt. This
script retrains the n=5 query model at NSEED seeds and re-runs the OOD sweep on each, so the flip
location and the curve agreement to the Bayes soft-majority decoder come with mean±sd error bars.

Anchored to the exact poisson-binomial logical oracle (active.poisson_binomial_tail of per-chain
filters) — the same oracle used in-distribution. Two nulls for the flip:
  - Bayes soft-majority decoder (oracle, P(L=M | k corrupted)) — the model SHOULD track this.
  - memorised-marginal null: predict the stationary logical rate P(Bin(n,q*)>r), CONSTANT in k —
    a model that memorised the training marginal (never demultiplexed) would be flat here.

Falsifier: if the model's 0.5-crossing in k is seed-unstable (not consistently between k=2 and k=3)
or it tracks the flat marginal-null instead of the oracle, the "learned threshold decode" claim must
be downgraded to single-seed. Outputs results/c2_faulttol_seeds.pt
"""
import os, sys, time
import numpy as np
import torch
ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from bag_moments import active
from bag_moments.model import GPTConfig, TinyGPT
from bag_moments import train as train_mod

OUT = os.environ.get("BAG_OUT", os.path.join(ROOT, "results"))
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))

EPS, GAMMA, PA, PM, T = 0.05, 0.15, 0.3, 0.7, 40
N = 5
STEPS = int(os.environ.get("FT_STEPS", "4000"))
NSEED = int(os.environ.get("FT_NSEED", "3"))
BATCH = int(os.environ.get("FT_BATCH", "256"))


def make_seqs(B, n, rng):
    tk, hid, _ = active.gen_redundant_active(B, T, n, EPS, GAMMA, PA, PM, rng, beta=0.0)
    packed = active.pack_emissions(tk)
    r = (n - 1) // 2
    qchain = active.per_chain_filter(tk, EPS, GAMMA, PA, PM)
    qlast = qchain[:, -1, :]
    logical_post = active.poisson_binomial_tail(qlast, r)
    true_L = (hid[:, -1, :].sum(1) > r).astype(np.int64)
    QUERY = 2 ** n; R_no = 2 ** n + 1; R_yes = 2 ** n + 2
    readout = np.where(true_L == 1, R_yes, R_no)
    seq = np.concatenate([packed, np.full((B, 1), QUERY), readout[:, None]], axis=1)
    return seq, logical_post, true_L, QUERY, R_yes


def ood_sweep(model, device, n, rng):
    """For k=0..n persistently-corrupted blocks, return model P(misaligned|k) and oracle."""
    r = (n - 1) // 2
    QUERY = 2 ** n; R_yes = 2 ** n + 2
    B = 4000
    mk, mo = [], []
    for k in range(n + 1):
        mis = np.zeros((B, n), dtype=bool)
        for b in range(B):
            mis[b, rng.choice(n, size=k, replace=False)] = True
        p = np.where(mis[:, None, :], PM, PA)
        bits = (rng.random((B, T, n)) < p).astype(np.int64)
        packed = active.pack_emissions(bits)
        seq = np.concatenate([packed, np.full((B, 1), QUERY)], axis=1)
        st = torch.tensor(seq, device=device)
        with torch.no_grad():
            pp = torch.softmax(model(st)[:, -1], -1).cpu().numpy()
        p_yes = pp[:, R_yes] / (pp[:, R_yes] + pp[:, R_yes - 1] + 1e-9)
        qchain = active.per_chain_filter(bits, EPS, GAMMA, PA, PM)[:, -1, :]
        orac = active.poisson_binomial_tail(qchain, r)
        mk.append(float(p_yes.mean())); mo.append(float(orac.mean()))
    return np.array(mk), np.array(mo)


def crossing(curve):
    """Linear-interpolated k where curve crosses 0.5 (np.nan if it never does)."""
    for k in range(len(curve) - 1):
        a, b = curve[k], curve[k + 1]
        if (a - 0.5) * (b - 0.5) <= 0 and b != a:
            return k + (0.5 - a) / (b - a)
    return float("nan")


def run_seed(seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(100 + seed); erng = np.random.default_rng(2000 + seed)
    r = (N - 1) // 2; vocab = 2 ** N + 3
    seq_ev, orac_ev, true_L, QUERY, R_yes = make_seqs(4096, N, erng)
    ev = torch.tensor(seq_ev, device=device); qpos = T

    def batch_fn():
        s, _, _, _, _ = make_seqs(BATCH, N, rng)
        t = torch.tensor(s, device=device); return t[:, :-1], t[:, 1:]

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(ev[:, :-1])[:, qpos], -1).cpu().numpy()
        p_yes = p[:, R_yes] / (p[:, R_yes] + p[:, R_yes - 1] + 1e-9)
        e = 1e-7; po = np.clip(orac_ev, e, 1 - e); pm = np.clip(p_yes, e, 1 - e)
        kl = (po * np.log(po / pm) + (1 - po) * np.log((1 - po) / (1 - pm))).mean()
        return {"kl_logical": float(kl)}

    cfg = GPTConfig(vocab_size=vocab, n_ctx=T + 2, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
    model = TinyGPT(cfg)
    t0 = time.time()
    out = train_mod.train(model, batch_fn, steps=STEPS, lr=1e-3, device=device,
                          snapshot_steps=[STEPS], eval_fn=eval_fn, log_every=STEPS)
    model.eval()
    kl = out["snapshots"][STEPS]["metrics"]["kl_logical"]
    swrng = np.random.default_rng(7000 + seed)
    mk, mo = ood_sweep(model, device, N, swrng)
    cross_model = crossing(mk); cross_oracle = crossing(mo)
    curve_kl = float(np.mean(np.abs(mk - mo)))  # mean abs gap model vs oracle across k
    print(f"[seed {seed}] KL_logical(indist)={kl:.4f} | OOD flip: model k*={cross_model:.2f} "
          f"oracle k*={cross_oracle:.2f} (code thresh r+1={r+1}) | mean|model-oracle|={curve_kl:.3f} "
          f"({(time.time()-t0)/60:.1f}m)\n   model P(mis|k)={np.round(mk,3)}\n   oracle      ={np.round(mo,3)}",
          flush=True)
    return {"seed": seed, "kl_logical": kl, "model_curve": mk.tolist(), "oracle_curve": mo.tolist(),
            "cross_model": cross_model, "cross_oracle": cross_oracle, "mean_abs_gap": curve_kl}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # memorised-marginal null: P(Bin(n,q*)>r), constant in k
    qstar = active.stationary_q(EPS, GAMMA)
    marg = active.binomial_tail(N, qstar, (N - 1) // 2)
    print(f"=== SEED-ROBUST OOD fault-tolerance (n={N}, {NSEED} seeds, {STEPS} steps) ===", flush=True)
    print(f"    marginal-null P(mis) = P(Bin({N},{qstar:.3f})>{(N-1)//2}) = {marg:.3f} (CONSTANT in k)", flush=True)
    rows = [run_seed(s, device) for s in range(NSEED)]
    cm = np.array([r["cross_model"] for r in rows])
    co = np.array([r["cross_oracle"] for r in rows])
    gaps = np.array([r["mean_abs_gap"] for r in rows])
    curves = np.array([r["model_curve"] for r in rows])
    agg = {"cross_model_mean": float(np.nanmean(cm)), "cross_model_sd": float(np.nanstd(cm)),
           "cross_oracle_mean": float(np.nanmean(co)), "mean_abs_gap_mean": float(gaps.mean()),
           "mean_abs_gap_sd": float(gaps.std()), "model_curve_mean": curves.mean(0).tolist(),
           "model_curve_sd": curves.std(0).tolist(), "marginal_null": float(marg)}
    torch.save({"rows": rows, "agg": agg,
                "config": {"n": N, "steps": STEPS, "nseed": NSEED, "eps": EPS, "gamma": GAMMA,
                           "pA": PA, "pM": PM, "T": T}}, f"{OUT}/c2_faulttol_seeds.pt")
    print("\n=== aggregate (mean±sd over seeds) ===")
    print(f"  model flip k* = {agg['cross_model_mean']:.2f}±{agg['cross_model_sd']:.2f}  "
          f"(oracle k*={agg['cross_oracle_mean']:.2f}, code threshold r+1={(N-1)//2+1})")
    print(f"  mean|model-oracle| across k = {agg['mean_abs_gap_mean']:.3f}±{agg['mean_abs_gap_sd']:.3f}")
    print(f"  model P(mis|k) mean curve = {np.round(curves.mean(0),3)}")
    print(f"  marginal-null (flat)      = {marg:.3f}")
    print(f"saved {OUT}/c2_faulttol_seeds.pt")


if __name__ == "__main__":
    main()
