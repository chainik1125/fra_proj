"""
Multi-seed robustness check for C1 and C2 transformer experiments (CPU-feasible).

Runs multiple seeds for each experiment, records KL, R², error rates, reports mean±sd.
This validates that the headline numbers in summary.md are stable (not one-shot flukes).

IMPORTANT (honest scope): to be runnable on the CPU sandbox (no GPU; outbound SSH to GPU
pods is blocked — only HTTPS/443 is open), this uses a LEAN training config:
  * identical architecture to the headline runs: d_model=128, n_heads=4, n_layers=3, d_mlp=512
  * reduced sequence length / batch / steps (see constants below)
So the absolute KL is a touch higher than the 6000-step headline (which used L=200, batch=256),
but the point of this script is STABILITY ACROSS SEEDS, not reproducing the exact headline digit.
The headline single runs remain the reference; this shows the qualitative result
(model ≈ Bayes filter; z decodable; logical<physical, falling with n) is seed-robust.

Outputs results/c1_multiseed.pt and results/c2_multiseed.pt
"""
import os
import sys
import time
import numpy as np
import torch

ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from bag_moments import active, probe, train
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", os.path.join(ROOT, "results"))
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))

EPS, GAMMA, PA, PM = 0.05, 0.15, 0.3, 0.7
QSTAR = active.stationary_q(EPS, GAMMA)

# ---- lean config (CPU) ----
L = 96            # C1 sequence length (headline used 200)
T = 40            # C2 sequence length (same as headline)
C1_STEPS = int(os.environ.get("C1_STEPS", "2500"))
C2_STEPS = int(os.environ.get("C2_STEPS", "1500"))
C1_BATCH = int(os.environ.get("C1_BATCH", "96"))
C2_BATCH = int(os.environ.get("C2_BATCH", "64"))
C1_SEEDS = list(range(int(os.environ.get("C1_NSEED", "5"))))
C2_SEEDS = list(range(int(os.environ.get("C2_NSEED", "3"))))
C2_NS = [int(x) for x in os.environ.get("C2_NS", "3,5,7").split(",")]
LR = 1.5e-3


def kl_bern(po, pm):
    e = 1e-7
    po = np.clip(po, e, 1-e); pm = np.clip(pm, e, 1-e)
    return po*np.log(po/pm) + (1-po)*np.log((1-po)/(1-pm))


# --------------------------------------------------------------------------
# C1: alignment log-odds coordinate
# --------------------------------------------------------------------------

def run_c1_seed(seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    erng = np.random.default_rng(999)
    toks_ev, _ = active.gen_active_2state(2048, L, EPS, GAMMA, PA, PM, erng)
    q_ev, qm_ev, next1_ev = active.forward_filter_2state(toks_ev, EPS, GAMMA, PA, PM)
    toks_ev_t = torch.tensor(toks_ev, device=device)
    z_ev = np.log(np.clip(q_ev, 1e-4, 1-1e-4) / np.clip(1-q_ev, 1e-4, 1-1e-4))
    pos_slab = list(range(20, L-1, 6))

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            logits = model(toks_ev_t[:, :-1])
            p = torch.softmax(logits, -1)
            pm1 = (p[..., 1] / (p[..., 0] + p[..., 1] + 1e-9)).cpu().numpy()
        skip = 20
        kl = kl_bern(next1_ev[:, skip:-1], pm1[:, skip:]).mean()
        return {"kl": float(kl)}

    def batch_fn():
        tk, _ = active.gen_active_2state(C1_BATCH, L, EPS, GAMMA, PA, PM, rng)
        t = torch.tensor(tk, device=device)
        return t[:, :-1], t[:, 1:]

    cfg = GPTConfig(vocab_size=2, n_ctx=L, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
    model = TinyGPT(cfg)
    out = train.train(model, batch_fn, steps=C1_STEPS, lr=LR, device=device,
                      snapshot_steps=[C1_STEPS], eval_fn=eval_fn, log_every=C1_STEPS)
    model.eval()
    kl = out["snapshots"][C1_STEPS]["metrics"]["kl"]

    # decode z per layer (held-out R² via ridge_probe internal split)
    r2_by_layer = {}
    for layer in range(cfg.n_layers):
        resid = probe.extract_resid(model, toks_ev_t[:, :-1], layer=layer, pos=pos_slab)
        y = z_ev[:, pos_slab].reshape(-1)
        r2_by_layer[layer] = probe.ridge_probe(resid, y, alpha=1.0)["r2"]
    best_r2 = max(r2_by_layer.values())

    # running-count baseline -> z (control: count is not sufficient for switching process)
    counts = np.cumsum(toks_ev[:, :-1], axis=1)  # (B, L-1) count of 1s up to each pos
    cnt = counts[:, pos_slab].reshape(-1, 1).astype(np.float64)
    cnt_r2 = probe.ridge_probe(cnt, z_ev[:, pos_slab].reshape(-1), alpha=1.0)["r2"]

    print(f"  C1 seed={seed}: KL={kl:.5f}  best_R2={best_r2:.4f}  count_R2={cnt_r2:.4f}", flush=True)
    return {"kl": kl, "r2_by_layer": r2_by_layer, "best_r2": best_r2, "count_r2": cnt_r2}


# --------------------------------------------------------------------------
# C2: transformer learns to majority-decode
# --------------------------------------------------------------------------

def make_seqs_c2(B, n, rng):
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
    return seq, {"logical_post": logical_post, "true_L": true_L, "qlast": qlast,
                 "R_yes": R_yes, "query_pos": T}


def run_c2_seed(seed, n, device):
    torch.manual_seed(seed * 100 + n)
    rng = np.random.default_rng(seed * 100 + n)
    r = (n - 1) // 2
    vocab = 2 ** n + 3

    erng = np.random.default_rng(1000 + n)
    seq_ev, meta = make_seqs_c2(4096, n, erng)
    ev = torch.tensor(seq_ev, device=device)
    qpos = meta["query_pos"]; R_yes = meta["R_yes"]
    orac = meta["logical_post"]

    def batch_fn():
        s, _ = make_seqs_c2(C2_BATCH, n, rng)
        t = torch.tensor(s, device=device); return t[:, :-1], t[:, 1:]

    def p_yes_fn(model):
        model.eval()
        with torch.no_grad():
            logits = model(ev[:, :-1])
            p = torch.softmax(logits[:, qpos], -1).cpu().numpy()
            return p[:, R_yes] / (p[:, R_yes] + p[:, R_yes - 1] + 1e-9)

    def eval_fn(model):
        return {"kl_logical": float(kl_bern(orac, p_yes_fn(model)).mean())}

    cfg = GPTConfig(vocab_size=vocab, n_ctx=T+2, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
    model = TinyGPT(cfg)
    out = train.train(model, batch_fn, steps=C2_STEPS, lr=LR, device=device,
                      snapshot_steps=[C2_STEPS], eval_fn=eval_fn, log_every=C2_STEPS)
    model.eval()
    kl = out["snapshots"][C2_STEPS]["metrics"]["kl_logical"]
    p_yes = p_yes_fn(model)

    model_logical_err = float(np.minimum(p_yes, 1-p_yes).mean())
    bayes_logical_err = float(np.minimum(orac, 1-orac).mean())
    qlast = meta["qlast"]
    phys_bayes_err = float(np.minimum(qlast, 1-qlast).mean(0).mean())

    print(f"  C2 seed={seed} n={n}: KL={kl:.5f}  model_err={model_logical_err:.4f}  "
          f"bayes_err={bayes_logical_err:.4f}  phys={phys_bayes_err:.4f}", flush=True)
    return {"kl": kl, "model_logical_err": model_logical_err,
            "bayes_logical_err": bayes_logical_err, "phys_bayes_err": phys_bayes_err}


def main():
    device = "cpu"
    t0 = time.time()
    print(f"device={device}  C1_steps={C1_STEPS} C2_steps={C2_STEPS} "
          f"C1_seeds={C1_SEEDS} C2_seeds={C2_SEEDS} C2_ns={C2_NS}", flush=True)

    # ---- C1 multi-seed ----
    print("\n=== C1: alignment log-odds ===", flush=True)
    c1 = [run_c1_seed(s, device) for s in C1_SEEDS]
    kls = [r["kl"] for r in c1]; r2s = [r["best_r2"] for r in c1]; cr2 = [r["count_r2"] for r in c1]
    print(f"\nC1 SUMMARY (n={len(C1_SEEDS)} seeds):", flush=True)
    print(f"  KL: {np.mean(kls):.5f} ± {np.std(kls):.5f} (min={min(kls):.5f}, max={max(kls):.5f})")
    print(f"  best R²: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
    print(f"  count-baseline R²: {np.mean(cr2):.4f} ± {np.std(cr2):.4f}")
    torch.save({"seeds": C1_SEEDS, "by_seed": c1, "config": {"L": L, "steps": C1_STEPS, "batch": C1_BATCH},
                "kl_mean": float(np.mean(kls)), "kl_std": float(np.std(kls)),
                "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
                "count_r2_mean": float(np.mean(cr2)), "count_r2_std": float(np.std(cr2))},
               f"{OUT}/c1_multiseed.pt")
    print(f"saved {OUT}/c1_multiseed.pt  ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- C2 multi-seed ----
    print("\n=== C2: majority decoder ===", flush=True)
    c2 = {n: [run_c2_seed(s, n, device) for s in C2_SEEDS] for n in C2_NS}
    print("\nC2 SUMMARY:", flush=True)
    by_n = {}
    for n in C2_NS:
        rows = c2[n]
        kls = [r["kl"] for r in rows]; me = [r["model_logical_err"] for r in rows]
        be = [r["bayes_logical_err"] for r in rows]; pe = [r["phys_bayes_err"] for r in rows]
        print(f"  n={n}: KL={np.mean(kls):.5f}±{np.std(kls):.5f}  "
              f"model_err={np.mean(me):.4f}±{np.std(me):.4f}  "
              f"bayes_err={np.mean(be):.4f}  phys={np.mean(pe):.4f}")
        by_n[n] = {"by_seed": rows, "kl_mean": float(np.mean(kls)), "kl_std": float(np.std(kls)),
                   "model_logical_err_mean": float(np.mean(me)), "model_logical_err_std": float(np.std(me)),
                   "bayes_logical_err_mean": float(np.mean(be)), "phys_bayes_err_mean": float(np.mean(pe))}
    torch.save({"seeds": C2_SEEDS, "ns": C2_NS, "config": {"T": T, "steps": C2_STEPS, "batch": C2_BATCH},
                "by_n": by_n}, f"{OUT}/c2_multiseed.pt")
    print(f"saved {OUT}/c2_multiseed.pt  (total {(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
