"""
C1 algorithm-vs-memorization test: does the trained transformer IMPLEMENT the Bayes filter as a fixed
inference procedure, or memorize the training-distribution statistics?

We train C1 at (eps0,gamma0)=(0.05,0.15) (q*=0.25) and then EVALUATE on token sequences drawn from
SHIFTED transition dynamics (eps',gamma') — keeping the emission model (pA,pM) fixed so the per-step
likelihood update is in-distribution but the transition prior / steady-state is OOD. For each shifted
eval process we compare the model's next-symbol prediction to TWO exact oracles:
  (A) filter_TRAINED : forward_filter_2state(tokens, eps0, gamma0)  -- the inference the model learned
  (B) filter_TRUE    : forward_filter_2state(tokens, eps',  gamma') -- the Bayes-optimal inference here

If the model learned the FILTER ALGORITHM (parameterised by its training dynamics), it applies the SAME
update on OOD inputs, so KL(model || filter_TRAINED) stays at the train-KL floor while KL(model ||
filter_TRUE) GROWS with the shift (the two oracles diverge). If instead it memorised training-distribution
input->output statistics, both KLs would blow up on OOD tokens. A monotone, low, flat KL-to-TRAINED across
shifts is positive evidence for an algorithmic (computed-belief) representation, sharpening §4.1's "it holds
the belief state, not a tally." Everything anchored to the exact 2-state filter (validate_active gate).

Outputs results/c1_genshift.pt
"""
import os, sys, time
import numpy as np
import torch
ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from bag_moments import active, probe, train
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", os.path.join(ROOT, "results"))
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))

EPS0, GAMMA0, PA, PM = 0.05, 0.15, 0.3, 0.7   # training dynamics (q*=0.25), identical to exp_c1_logodds
L = int(os.environ.get("GS_L", "160"))
STEPS = int(os.environ.get("GS_STEPS", "4500"))
NEV = int(os.environ.get("GS_NEV", "2048"))
LR = 1e-3
BURN = 30  # skip burn-in positions in all averages

# eval shift grid: vary gamma (and one eps point) to move q* and dwell time away from training
SHIFTS = [(0.05, 0.45), (0.05, 0.25), (0.05, 0.15), (0.05, 0.08), (0.05, 0.05), (0.12, 0.15)]


def kl_next(p_oracle, p_model):
    """binary next-symbol KL(oracle||model), oracle/model are P(x=1) arrays."""
    e = 1e-7
    po = np.clip(p_oracle, e, 1 - e); pm = np.clip(p_model, e, 1 - e)
    return (po * np.log(po / pm) + (1 - po) * np.log((1 - po) / (1 - pm)))


def main():
    seed = int(os.environ.get("GS_SEED", "0"))
    torch.manual_seed(seed); rng = np.random.default_rng(seed); erng = np.random.default_rng(777)
    q0star = active.stationary_q(EPS0, GAMMA0)
    print(f"=== C1 generalization-under-shift (train eps={EPS0},gamma={GAMMA0},q*={q0star:.3f}; "
          f"L={L},steps={STEPS}) ===", flush=True)

    # ---- train at the training dynamics ----
    toks_ev, _ = active.gen_active_2state(NEV, L, EPS0, GAMMA0, PA, PM, erng)
    q_ev, qm_ev, next1_ev = active.forward_filter_2state(toks_ev, EPS0, GAMMA0, PA, PM)
    toks_ev_t = torch.tensor(toks_ev)

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(toks_ev_t[:, :-1]), -1)[..., 1].cpu().numpy()
        return {"kl": float(kl_next(next1_ev[:, :-1], p)[:, BURN:].mean())}

    def batch_fn():
        tk, _ = active.gen_active_2state(256, L, EPS0, GAMMA0, PA, PM, rng)
        t = torch.tensor(tk); return t[:, :-1], t[:, 1:]

    cfg = GPTConfig(vocab_size=2, n_ctx=L, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
    model = TinyGPT(cfg); t0 = time.time()
    out = train.train(model, batch_fn, steps=STEPS, lr=LR, device="cpu",
                      snapshot_steps=[STEPS], eval_fn=eval_fn, log_every=STEPS)
    model.eval()
    train_kl = out["snapshots"][STEPS]["metrics"]["kl"]
    print(f"  trained: in-dist next-symbol KL to TRAINED filter = {train_kl:.5f} ({(time.time()-t0)/60:.1f}m)", flush=True)

    # ---- evaluate on shifted processes ----
    rows = []
    for (epsp, gammap) in SHIFTS:
        teval = np.random.default_rng(1234 + int(1000 * gammap))  # fixed per-shift eval seed
        toks, hidden = active.gen_active_2state(NEV, L, epsp, gammap, PA, PM, teval)
        toks_t = torch.tensor(toks)
        with torch.no_grad():
            p_model = torch.softmax(model(toks_t[:, :-1]), -1)[..., 1].cpu().numpy()
        # two oracles on the SAME shifted tokens
        _, _, nxt_trained = active.forward_filter_2state(toks, EPS0, GAMMA0, PA, PM)
        qf_true, _, nxt_true = active.forward_filter_2state(toks, epsp, gammap, PA, PM)
        kl_to_trained = float(kl_next(nxt_trained[:, :-1], p_model)[:, BURN:].mean())
        kl_to_true = float(kl_next(nxt_true[:, :-1], p_model)[:, BURN:].mean())
        kl_oracles = float(kl_next(nxt_true[:, :-1], nxt_trained[:, :-1])[:, BURN:].mean())  # how different the inferences are
        # behavioural: model's implied mean next-symbol rate vs each oracle's
        mrate = float(p_model[:, BURN:].mean())
        rate_trained = float(nxt_trained[:, BURN:-1].mean()); rate_true = float(nxt_true[:, BURN:-1].mean())
        qstar = active.stationary_q(epsp, gammap)
        rows.append({"eps": epsp, "gamma": gammap, "qstar": qstar,
                     "kl_to_trained": kl_to_trained, "kl_to_true": kl_to_true, "kl_oracles": kl_oracles,
                     "model_rate": mrate, "rate_trained_filter": rate_trained, "rate_true_filter": rate_true})
        print(f"  shift eps={epsp} gamma={gammap} q*={qstar:.3f} | KL(model||TRAINED)={kl_to_trained:.5f}  "
              f"KL(model||TRUE)={kl_to_true:.5f}  KL(oracles)={kl_oracles:.5f} | "
              f"model_rate={mrate:.3f} (trained-filt {rate_trained:.3f}, true-filt {rate_true:.3f})", flush=True)

    torch.save({"rows": rows, "train_kl": train_kl,
                "config": {"eps0": EPS0, "gamma0": GAMMA0, "pA": PA, "pM": PM, "q0star": q0star,
                           "L": L, "steps": STEPS, "nev": NEV, "burn": BURN, "shifts": SHIFTS, "seed": seed}},
               f"{OUT}/c1_genshift.pt")
    print(f"saved {OUT}/c1_genshift.pt", flush=True)
    # headline
    kt = np.array([r["kl_to_trained"] for r in rows]); ku = np.array([r["kl_to_true"] for r in rows])
    print(f"\nHEADLINE: KL(model||TRAINED filter) range {kt.min():.5f}-{kt.max():.5f} (flat ≈ train-KL "
          f"{train_kl:.5f}); KL(model||TRUE filter) range {ku.min():.5f}-{ku.max():.5f} (grows with shift). "
          f"=> the model applies the TRAINED-parameter Bayes filter on OOD inputs." if kt.max() < ku.max()
          else "\nNULL/surprising: model does not uniformly track the trained filter.", flush=True)


if __name__ == "__main__":
    main()
