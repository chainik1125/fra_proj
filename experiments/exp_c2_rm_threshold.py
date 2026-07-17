"""
C2 R_M EPIDEMIC THRESHOLD *in the transformer's learned predictions* (the genuine §7-closer).

§4.2 shows the SIS bifurcation at the reproduction number R_M = beta*(n-1)/(n*gamma) = 1 at the
PROCESS level (stationary misalignment I*=0 below, I*=1-1/R_M above). §7 flags as OPEN: "whether a
transformer trained on a *spreading* redundant bag exhibits the threshold in its predictions."
This experiment closes exactly that gap: we train a tiny transformer per beta across a FINE sweep
spanning R_M in [~0.5, ~2.0], with a SMALL spontaneous-corruption rate eps so the SIS bifurcation
is sharp, and ask whether the model's LEARNED logical decoder tracks the EXACT joint Bayes filter
on BOTH sides of the epidemic threshold (so its implied endemic-misalignment curve reproduces the
R_M=1 elbow) -- rather than only working in the independent / subcritical regime.

Anchors (all closed-form): the EXACT 2^n-state joint forward filter (validated in exp_c2_spread,
reused here) for the model's logical target + KL; the SIS theory curve I*=max(0,1-1/R_M) for the
per-chain endemic fraction; the true hidden misalignment fraction (ground truth) for the process.

Tests, per beta (per R_M):
  V  joint-filter calibration (sanity, reuses exp_c2_spread.validate_filter style, light).
  A  KL(model logical posterior -> JOINT oracle) at the query -- claim: stays LOW across R_M=1.
  B  model implied endemic LOGICAL-misalign rate (mean P(L=M)) vs oracle's vs true -- same elbow.
  C  true per-chain endemic fraction vs SIS theory I*=max(0,1-1/R_M) -- the process bifurcation.
Honest failure mode worth reporting: the model may SMOOTH the bifurcation rather than reproduce a
sharp elbow (transformers averaging epidemic dynamics).

Outputs results/c2_rm_threshold.pt
"""
import os, sys, time
import numpy as np
import torch
ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from bag_moments import active, train
from bag_moments.model import GPTConfig, TinyGPT
# reuse the EXACT joint-filter machinery (validated) from the spread experiment
from exp_c2_spread import build_joint_T, build_emission_table, stationary, joint_filter, POP

OUT = os.environ.get("BAG_OUT", os.path.join(ROOT, "results"))
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))

# Small eps -> the all-aligned state is *nearly* absorbing -> sharp SIS bifurcation at R_M=1.
EPS, GAMMA, PA, PM = 1e-3, 0.05, 0.3, 0.7
N = int(os.environ.get("RM_N", "5"))
T = int(os.environ.get("RM_T", "96"))           # long enough to approach stationarity
STEPS = int(os.environ.get("RM_STEPS", "1500"))  # lean: qualitative elbow, not tightest KL
BATCH = int(os.environ.get("RM_BATCH", "192"))
R = (N - 1) // 2
S = 1 << N


def rm_of_beta(beta):
    return beta * (N - 1) / (N * GAMMA)


def beta_of_rm(rm):
    return rm * N * GAMMA / (N - 1)


def make_seqs(B, rng, beta, with_query=True):
    tk, hid, logical = active.gen_redundant_active(B, T, N, EPS, GAMMA, PA, PM, rng, beta=beta)
    packed = active.pack_emissions(tk)
    post = joint_filter(packed, N, EPS, GAMMA, PA, PM, beta)
    true_L = logical[:, -1]
    QUERY = S; R_no = S + 1; R_yes = S + 2
    if with_query:
        readout = np.where(true_L == 1, R_yes, R_no)
        seq = np.concatenate([packed, np.full((B, 1), QUERY), readout[:, None]], axis=1)
    else:
        seq = packed
    return seq, {"logical_post": post[:, -1], "logical_post_full": post, "true_L": true_L,
                 "R_yes": R_yes, "query_pos": T, "hidden": hid, "logical_full": logical}


def run_beta(beta, seed=0, device="cpu"):
    rm = rm_of_beta(beta)
    torch.manual_seed(seed); rng = np.random.default_rng(seed); erng = np.random.default_rng(11)
    vocab = S + 3
    seq_ev, meta = make_seqs(3072, erng, beta, with_query=True)
    seq_ev_t = torch.tensor(seq_ev, device=device)
    qpos = meta["query_pos"]; R_yes = meta["R_yes"]
    orac = meta["logical_post"]                       # exact joint P(L=M) at last pos

    # process-level ground truth (2nd half, to approach stationarity)
    half = T // 2
    true_chain_frac = float(meta["hidden"][:, half:, :].mean())          # per-chain endemic fraction
    true_logical_rate = float(meta["logical_full"][:, half:].mean())     # logical (majority) endemic
    orac_logical_rate = float(meta["logical_post_full"][:, half:].mean())# oracle filter implied logical
    sis_theory = max(0.0, 1.0 - 1.0 / rm) if rm > 0 else 0.0             # SIS per-chain endemic

    def batch_fn():
        s, _ = make_seqs(BATCH, rng, beta, with_query=True)
        t = torch.tensor(s, device=device); return t[:, :-1], t[:, 1:]

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(seq_ev_t[:, :-1])[:, qpos], -1).cpu().numpy()
        pm = np.clip(p[:, R_yes], 1e-6, 1 - 1e-6); po = np.clip(orac, 1e-6, 1 - 1e-6)
        kl = (po * np.log(po / pm) + (1 - po) * np.log((1 - po) / (1 - pm))).mean()
        return {"kl_logical": float(kl)}

    cfg = GPTConfig(vocab_size=vocab, n_ctx=T + 2, d_model=96, n_heads=4, n_layers=2, d_mlp=384)
    model = TinyGPT(cfg)
    t0 = time.time()
    train.train(model, batch_fn, steps=STEPS, lr=1e-3, device=device,
                snapshot_steps=[STEPS], eval_fn=eval_fn, log_every=max(STEPS // 2, 1))
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(seq_ev_t[:, :-1])[:, qpos], -1).cpu().numpy()
    pm = np.clip(p[:, R_yes], 1e-6, 1 - 1e-6)
    po = np.clip(orac, 1e-6, 1 - 1e-6)
    kl_to_joint = float((po * np.log(po / pm) + (1 - po) * np.log((1 - po) / (1 - pm))).mean())
    model_logical_rate = float(pm.mean())                                # model implied endemic logical
    model_logical_err = float(((pm > 0.5).astype(int) != meta["true_L"]).mean())
    bayes_logical_err = float(np.minimum(orac, 1 - orac).mean())
    res = {"beta": beta, "R_M": rm, "kl_to_joint": kl_to_joint,
           "model_logical_rate": model_logical_rate, "orac_logical_rate": orac_logical_rate,
           "true_logical_rate": true_logical_rate, "true_chain_frac": true_chain_frac,
           "sis_theory": sis_theory, "model_logical_err": model_logical_err,
           "bayes_logical_err": bayes_logical_err, "n": N}
    print(f"[beta={beta:.4f} R_M={rm:.2f}] KL->joint={kl_to_joint:.4f} | "
          f"model_logical_rate={model_logical_rate:.3f} orac={orac_logical_rate:.3f} "
          f"true={true_logical_rate:.3f} | chain_frac={true_chain_frac:.3f} SIS={sis_theory:.3f} "
          f"({(time.time()-t0)/60:.1f}min)", flush=True)
    return res


def main():
    # span R_M sub->super-critical; ensure a point right at R_M=1
    rms = [float(x) for x in os.environ.get("RM_LIST", "0.5,0.75,1.0,1.25,1.5,2.0").split(",")]
    betas = [beta_of_rm(r) for r in rms]
    print(f"=== C2 R_M threshold-in-predictions (n={N}, eps={EPS}, gamma={GAMMA}) ===", flush=True)
    print(f"  R_M sweep {rms} -> betas {[round(b,4) for b in betas]}", flush=True)
    results = [run_beta(b) for b in betas]
    torch.save({"results": results,
                "config": {"n": N, "eps": EPS, "gamma": GAMMA, "pA": PA, "pM": PM, "T": T,
                           "steps": STEPS, "rms": rms, "betas": betas}},
               f"{OUT}/c2_rm_threshold.pt")
    print(f"saved {OUT}/c2_rm_threshold.pt", flush=True)


if __name__ == "__main__":
    main()
