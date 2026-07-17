"""
Rate-controlled TRAINED-TRANSFORMER companion to §4.5 (closes the §7 caveat "did not reproduce the
rate-controlled curve in a trained transformer").

§4.5's main spread experiment confounds correlation with the per-chain marginal rate. Here we hold
the per-chain misrate FIXED (binary-search eps per beta) and train a transformer at each correlation
point, then check that the model still learns the JOINT (correlation-aware) decoder — model->joint KL
stays low while model->independent KL grows with rho — i.e. the §4.5 headline is NOT a marginal-rate
artifact. Reuses the validated 2^n joint filter from exp_c2_spread.

Outputs results/c2_spread_ratectrl.pt
"""
import os, sys, time
import numpy as np
import torch
ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from bag_moments import active, train
from bag_moments.model import GPTConfig, TinyGPT
from experiments import exp_c2_spread as E

OUT = os.environ.get("BAG_OUT", os.path.join(ROOT, "results"))
N, GAMMA, PA, PM, R = 5, 0.05, 0.3, 0.7, 2
S = 1 << N
T = int(os.environ.get("RC_T", "80"))
STEPS = int(os.environ.get("RC_STEPS", "2000"))
BATCH = int(os.environ.get("RC_BATCH", "192"))
TARGET = float(os.environ.get("RC_TARGET", "0.30"))
torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))


def misrate(eps, beta, seed=3):
    _, hid, _ = active.gen_redundant_active(1200, 180, N, eps, GAMMA, PA, PM,
                                            np.random.default_rng(seed), beta=beta)
    return float(hid[:, 90:, :].mean())


def tune_eps(beta):
    lo, hi = 0.0, 0.30
    for _ in range(22):
        mid = (lo + hi) / 2
        if misrate(mid, beta) < TARGET:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def make_seqs(B, rng, beta, eps):
    tk, hid, lg = active.gen_redundant_active(B, T, N, eps, GAMMA, PA, PM, rng, beta=beta)
    packed = active.pack_emissions(tk)
    post = E.joint_filter(packed, N, eps, GAMMA, PA, PM, beta)
    true_L = lg[:, -1]
    QUERY = S; R_no = S + 1; R_yes = S + 2
    readout = np.where(true_L == 1, R_yes, R_no)
    seq = np.concatenate([packed, np.full((B, 1), QUERY), readout[:, None]], axis=1)
    return seq, {"logical_post": post[:, -1], "true_L": true_L, "R_yes": R_yes,
                 "query_pos": T, "hidden": hid, "tokens": tk}


def run_beta(beta, seed=0):
    eps = tune_eps(beta)
    torch.manual_seed(seed); rng = np.random.default_rng(seed); erng = np.random.default_rng(7)
    seq_ev, meta = make_seqs(3072, erng, beta, eps)
    seq_ev_t = torch.tensor(seq_ev); qpos = meta["query_pos"]; R_yes = meta["R_yes"]
    orac = meta["logical_post"]; corr = E.cross_chain_corr(meta["hidden"])
    rate = float(meta["hidden"][:, T // 2:, :].mean())

    def batch_fn():
        s, _ = make_seqs(BATCH, rng, beta, eps); t = torch.tensor(s); return t[:, :-1], t[:, 1:]

    cfg = GPTConfig(vocab_size=S + 3, n_ctx=T + 2, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
    model = TinyGPT(cfg)
    t0 = time.time()
    train.train(model, batch_fn, steps=STEPS, lr=1e-3, device="cpu", snapshot_steps=[STEPS],
                log_every=max(STEPS // 2, 1))
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(seq_ev_t[:, :-1])[:, qpos], -1).cpu().numpy()
    pm = np.clip(p[:, R_yes], 1e-6, 1 - 1e-6)
    qc = active.per_chain_filter(meta["tokens"], eps, GAMMA, PA, PM)[:, -1, :]
    p_indep = np.clip(active.poisson_binomial_tail(qc, R), 1e-6, 1 - 1e-6)
    po = np.clip(orac, 1e-6, 1 - 1e-6)
    kl_to_joint = float((po * np.log(po / pm) + (1 - po) * np.log((1 - po) / (1 - pm))).mean())
    kl_to_indep = float((p_indep * np.log(p_indep / pm) + (1 - p_indep) * np.log((1 - p_indep) / (1 - pm))).mean())
    model_err = float(((pm > 0.5).astype(int) != meta["true_L"]).mean())
    joint_err = float(np.minimum(orac, 1 - orac).mean())
    indep_err = float(((p_indep > 0.5).astype(int) != meta["true_L"]).mean())
    res = {"beta": beta, "eps": eps, "rho": corr, "rate": rate, "kl_to_joint": kl_to_joint,
           "kl_to_indep": kl_to_indep, "model_err": model_err, "joint_err": joint_err,
           "indep_err": indep_err}
    print(f"[beta={beta:.2f} eps={eps:.3f} rho={corr:+.3f} rate={rate:.3f}] "
          f"model->joint KL={kl_to_joint:.4f} vs model->indep KL={kl_to_indep:.4f} | "
          f"model_err={model_err:.3f} joint={joint_err:.3f} indep={indep_err:.3f} "
          f"({(time.time()-t0)/60:.1f}min)", flush=True)
    return res


def main():
    betas = [float(b) for b in os.environ.get("RC_BETAS", "0.0,0.10,0.18").split(",")]
    print(f"=== Rate-controlled TRAINED transformer (per-chain rate fixed {TARGET}, n={N}) ===", flush=True)
    results = [run_beta(b) for b in betas]
    torch.save({"results": results, "target_rate": TARGET, "n": N, "steps": STEPS},
               f"{OUT}/c2_spread_ratectrl.pt")
    print(f"saved {OUT}/c2_spread_ratectrl.pt", flush=True)


if __name__ == "__main__":
    main()
