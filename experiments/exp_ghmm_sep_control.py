"""
T2 separability via ITERATIVE LINEAR ERASURE (INLP/LEACE-style) — the rigorous, citable version of
§4.4's load-bearing factorization claim, with positive control + random null.

The committed T2 test removes a fixed 2-3 dim probe subspace and reports cross-survival. Two problems a
skeptic raises: (i) removing 2-3 of 128 dims is generically harmless (no NULL control); (ii) removing the
few regression directions does NOT actually erase a richly-encoded target — ridge re-fits it from the
remaining dims (we verified: removing the 2-dim Mess3 probe-plane leaves Mess3 R² ~unchanged). So the
cross-survival number alone is weak. Here we do proper LINEAR ERASURE (INLP 2004.07667 / LEACE 2306.03819,
already cited): iteratively project out the best ridge direction(s) until the *target's own* held-out R²
collapses below a threshold, recording the erased-dimension count, THEN test the OTHER latent on the same
erased representation. Controls:

  (POS) the erased latent's own R^2 must collapse to ~0 (confirms genuine erasure, not token removal).
  (CROSS) the OTHER latent's R^2 after erasure = the separability signal (does z survive ERASING Mess3?).
  (NULL) erasing the SAME number of RANDOM directions must leave both latents ~intact.

Separability is real iff CROSS stays ~full while POS collapses and well above the NULL's effect. Same drift
config (L=128, STEPS=3000) + best-Mess3 layer as exp_ghmm_factored.py. Anchored to the exact 6-state filter.

Outputs results/ghmm_sep_control.pt
"""
import os, sys, time
import numpy as np
import torch
ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from bag_moments import ghmm, probe, train
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", os.path.join(ROOT, "results"))
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))

L = int(os.environ.get("GHMM_L", "128"))
STEPS = int(os.environ.get("GHMM_STEPS", "3000"))
BATCH = int(os.environ.get("GHMM_BATCH", "96"))
NEV = int(os.environ.get("GHMM_NEV", "3072"))
LR = 1.5e-3
# drift regime params -- IDENTICAL to exp_ghmm_factored.py REGIME=drift
EPS, GAMMA, STAY, A = 0.02, 0.06, 0.30, 0.80
P = ghmm.joint_params_drift(EPS, GAMMA, STAY, A)
NSEED = int(os.environ.get("SC_NSEED", "3"))
NRAND = int(os.environ.get("SC_NRAND", "10"))


def kl_cat(po, pm):
    e = 1e-7; po = np.clip(po, e, 1); pm = np.clip(pm, e, 1)
    return (po * np.log(po / pm)).sum(-1)


def grouped_ridge(R, y, seq_ids, alpha=1.0, train_frac=0.7, seed=0):
    R = np.asarray(R, np.float64); y = np.asarray(y, np.float64).ravel()
    uniq = np.unique(seq_ids); rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq); k = int(len(uniq) * train_frac)
    tr_seq = set(perm[:k].tolist())
    tr = np.array([sid in tr_seq for sid in seq_ids]); te = ~tr
    mu = R[tr].mean(0); sd = R[tr].std(0) + 1e-8
    Xtr = (R[tr] - mu) / sd; Xte = (R[te] - mu) / sd; ym = y[tr].mean()
    Amat = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    w = np.linalg.solve(Amat, Xtr.T @ (y[tr] - ym))
    pred = Xte @ w + ym
    ss = ((y[te] - pred) ** 2).sum(); tot = ((y[te] - y[te].mean()) ** 2).sum() + 1e-12
    return {"r2": float(1 - ss / tot), "direction": w / sd}


def resid(M, B):
    """project M off the orthonormal column space B (d x k)."""
    return M if B.shape[1] == 0 else M - (M @ B) @ B.T


def leace_eraser(Xtr, Ztr, ridge=1e-3):
    """Closed-form LEACE (Belrose et al. 2306.03819): returns (mu, TT) such that for centered X,
    X_erased = (X-mu) - (X-mu) @ TT provably zeroes the linear cross-covariance with Z (so any linear
    probe -> R^2~0), removing only rank(Sigma_XZ) effective directions. Fit on TRAIN rows only."""
    Xtr = np.asarray(Xtr, np.float64); Ztr = np.asarray(Ztr, np.float64)
    if Ztr.ndim == 1:
        Ztr = Ztr[:, None]
    mu = Xtr.mean(0); Xc = Xtr - mu; Zc = Ztr - Ztr.mean(0)
    n, d = Xc.shape
    Sig = (Xc.T @ Xc) / n + ridge * np.eye(d)
    s, U = np.linalg.eigh(Sig)
    s = np.clip(s, 1e-9, None)
    Wh = U @ np.diag(1.0 / np.sqrt(s)) @ U.T          # whitening Sigma^{-1/2}
    Wi = U @ np.diag(np.sqrt(s)) @ U.T                # un-whitening Sigma^{+1/2}
    SigXZ = (Xc.T @ Zc) / n                            # (d,k)
    M = Wh @ SigXZ                                     # whitened cross-cov
    Bm, _ = np.linalg.qr(M)                            # orthonormal basis of col(M); rank<=k
    # drop near-zero columns of M's range
    keep = np.linalg.norm(M.T @ Bm, axis=0) > 1e-8 if Bm.shape[1] else np.array([], bool)
    if keep.size and not keep.all():
        Bm = Bm[:, keep]
    TT = Wh @ Bm @ Bm.T @ Wi                           # operator s.t. X_er = Xc - Xc@TT
    return mu, TT, int(Bm.shape[1])


def random_eraser(d, k, rng):
    """Eraser that orthogonally removes k RANDOM directions (dimension-matched null)."""
    Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    Q = Q[:, :k]
    return Q @ Q.T  # TT for orthogonal removal: X_er = Xc - Xc@(Q Q^T)


def mr2(R, mx, my, seq_ids):
    return 0.5 * (grouped_ridge(R, mx, seq_ids)["r2"] + grouped_ridge(R, my, seq_ids)["r2"])


def run(seed=0, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); erng = np.random.default_rng(777)
    obs_ev, c_ev, s_ev = ghmm.gen_ghmm(NEV, L, P, erng)
    belief_ev, q_ev, mbel_ev, nextp_ev = ghmm.forward_filter_ghmm(obs_ev, P)
    obs_ev_t = torch.tensor(obs_ev, device=device)
    z_ev = np.log(np.clip(q_ev, 1e-4, 1 - 1e-4) / np.clip(1 - q_ev, 1e-4, 1 - 1e-4))
    mxy_ev = ghmm.simplex_xy(mbel_ev)

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(obs_ev_t[:, :-1]), -1).cpu().numpy()
        return {"kl": float(kl_cat(nextp_ev[:, :-1, :], p)[:, 20:].mean())}

    def batch_fn():
        o, _, _ = ghmm.gen_ghmm(BATCH, L, P, rng)
        t = torch.tensor(o, device=device); return t[:, :-1], t[:, 1:]

    cfg = GPTConfig(vocab_size=3, n_ctx=L, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
    model = TinyGPT(cfg)
    t0 = time.time()
    out = train.train(model, batch_fn, steps=STEPS, lr=LR, device=device,
                      snapshot_steps=[STEPS], eval_fn=eval_fn, log_every=STEPS)
    model.eval()
    kl = out["snapshots"][STEPS]["metrics"]["kl"]

    pos = list(range(20, L - 1, 4)); npos = len(pos)
    B = obs_ev.shape[0]; seq_ids = np.repeat(np.arange(B), npos)
    qf = q_ev[:, pos].reshape(-1)
    zt = z_ev[:, pos].reshape(-1)
    mx = mxy_ev[:, pos, 0].reshape(-1); my = mxy_ev[:, pos, 1].reshape(-1)

    # pick best-Mess3 layer (same selection as exp_ghmm_factored T2)
    decode = {}
    for layer in range(cfg.n_layers):
        R = probe.extract_resid(model, obs_ev_t[:, :-1], layer=layer, pos=pos)
        pz = grouped_ridge(R, zt, seq_ids); pmx = grouped_ridge(R, mx, seq_ids); pmy = grouped_ridge(R, my, seq_ids)
        decode[layer] = {"z_r2": pz["r2"], "m_r2": 0.5 * (pmx["r2"] + pmy["r2"]),
                         "_R": R, "_pmx": pmx, "_pmy": pmy}
    bm = max(decode, key=lambda l: decode[l]["m_r2"])
    R = decode[bm]["_R"]; d = R.shape[1]
    z_full = grouped_ridge(R, zt, seq_ids)["r2"]
    m_full = mr2(R, mx, my, seq_ids)

    # leak-free split matching grouped_ridge(seed=0, train_frac=0.7): fit eraser on TRAIN rows only
    uniq = np.unique(seq_ids); rngs = np.random.default_rng(0); perm = rngs.permutation(uniq)
    ktr = int(len(uniq) * 0.7); tr_seq = set(perm[:ktr].tolist())
    tr = np.array([sid in tr_seq for sid in seq_ids])
    mz = np.stack([mx, my], axis=1)

    def apply_TT(mu, TT):
        return R - (R - mu) @ TT

    # --- LEACE erase each latent (closed-form, provably zeroes linear predictability) ---
    muz, TTz, kz = leace_eraser(R[tr], zt[tr]); Rz0 = apply_TT(muz, TTz)      # erase alignment z
    mum, TTm, km = leace_eraser(R[tr], mz[tr]); Rm0 = apply_TT(mum, TTm)      # erase Mess3 belief
    # POSITIVE controls: the erased latent's own R^2 collapses (->0)
    z_after_eraseZ = grouped_ridge(Rz0, zt, seq_ids)["r2"]
    m_after_eraseM = mr2(Rm0, mx, my, seq_ids)
    # CROSS (separability): the OTHER latent survives the erasure (~ full)
    m_after_eraseZ = mr2(Rz0, mx, my, seq_ids)
    z_after_eraseM = grouped_ridge(Rm0, zt, seq_ids)["r2"]
    # NULL: removing the SAME #dims of RANDOM directions leaves both intact (dimension-matched)
    rgc = np.random.default_rng(123); muR = R[tr].mean(0)
    z_rand_km, m_rand_kz = [], []
    for _ in range(NRAND):
        z_rand_km.append(grouped_ridge(R - (R - muR) @ random_eraser(d, km, rgc), zt, seq_ids)["r2"])
        m_rand_kz.append(mr2(R - (R - muR) @ random_eraser(d, kz, rgc), mx, my, seq_ids))

    res = {"seed": seed, "kl": kl, "best_m_layer": bm, "d_model": d,
           "kz_erase_dims": kz, "km_erase_dims": km,
           "z_full": z_full, "m_full": m_full,
           "z_after_eraseZ": z_after_eraseZ, "m_after_eraseM": m_after_eraseM,     # POS (->0)
           "m_after_eraseZ": m_after_eraseZ, "z_after_eraseM": z_after_eraseM,     # CROSS (survives)
           "z_after_rand_km_mean": float(np.mean(z_rand_km)), "z_after_rand_km_sd": float(np.std(z_rand_km)),
           "m_after_rand_kz_mean": float(np.mean(m_rand_kz)), "m_after_rand_kz_sd": float(np.std(m_rand_kz)),
           "train_min": (time.time() - t0) / 60}
    print(f"[seed {seed}] KL={kl:.5f} layer{bm} d={d} | LEACE-rank z={kz} m={km} ({res['train_min']:.1f}m)\n"
          f"  ERASE z: z {z_full:.3f}->{z_after_eraseZ:.3f}(POS→0) | m survives {m_full:.3f}->{m_after_eraseZ:.3f}"
          f" (CROSS) | rand-{kz}d null m={res['m_after_rand_kz_mean']:.3f}±{res['m_after_rand_kz_sd']:.3f}\n"
          f"  ERASE m: m {m_full:.3f}->{m_after_eraseM:.3f}(POS→0) | z survives {z_full:.3f}->{z_after_eraseM:.3f}"
          f" (CROSS) | rand-{km}d null z={res['z_after_rand_km_mean']:.3f}±{res['z_after_rand_km_sd']:.3f}",
          flush=True)
    return res


def main():
    print(f"=== T2 separability via closed-form LEACE erasure — drift, L={L},steps={STEPS},{NSEED} seeds ===", flush=True)
    rows = [run(s) for s in range(NSEED)]
    keys = ["z_full", "m_full", "z_after_eraseZ", "m_after_eraseM", "m_after_eraseZ", "z_after_eraseM",
            "z_after_rand_km_mean", "m_after_rand_kz_mean", "kz_erase_dims", "km_erase_dims", "kl"]
    agg = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float); agg[k] = (float(v.mean()), float(v.std()))
    torch.save({"rows": rows, "agg": agg,
                "config": {"L": L, "STEPS": STEPS, "NEV": NEV, "nseed": NSEED, "nrand": NRAND,
                           "thresh": float(os.environ.get("SC_THRESH", "0.05")),
                           "eps": EPS, "gamma": GAMMA, "stay": STAY, "a": A}},
               f"{OUT}/ghmm_sep_control.pt")
    print(f"\nsaved {OUT}/ghmm_sep_control.pt")
    print("=== aggregate (mean±sd over seeds) ===")
    print(f"  erase-dims: z={agg['kz_erase_dims'][0]:.1f}±{agg['kz_erase_dims'][1]:.1f}  m={agg['km_erase_dims'][0]:.1f}±{agg['km_erase_dims'][1]:.1f}")
    print(f"  ERASE z: z {agg['z_full'][0]:.3f}->{agg['z_after_eraseZ'][0]:.3f}(POS)  |  m survives {agg['m_full'][0]:.3f}->{agg['m_after_eraseZ'][0]:.3f}±{agg['m_after_eraseZ'][1]:.3f}(CROSS)  |  rand-null m={agg['m_after_rand_kz_mean'][0]:.3f}")
    print(f"  ERASE m: m {agg['m_full'][0]:.3f}->{agg['m_after_eraseM'][0]:.3f}(POS)  |  z survives {agg['z_full'][0]:.3f}->{agg['z_after_eraseM'][0]:.3f}±{agg['z_after_eraseM'][1]:.3f}(CROSS)  |  rand-null z={agg['z_after_rand_km_mean'][0]:.3f}")


if __name__ == "__main__":
    main()
