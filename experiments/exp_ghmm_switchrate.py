"""
GHMM persona-switching-rate robustness of the C3 factorization (on-theme: the paper is about alignment
DYNAMICS). Does the alignment⊗capability factorization survive when the hidden alignment state switches
FAST (a highly dynamic posterior z) vs slow?

We keep the drift bag (a=0.80, stay=0.30) and the stationary misalignment FIXED at q*=ε/(ε+γ)=0.25 (γ=3ε),
varying only the switching MAGNITUDE: slow (ε=0.01,γ=0.03) → base (0.02,0.06) → fast (0.06,0.18) → very
fast (0.12,0.36). Faster switching ⇒ z changes more rapidly within a sequence (more dynamic persona) and the
alignment evidence has less time to accumulate, a harder regime for a clean separable z code. For each rate
(oracle-gated to the exact 6-state filter) we run the load-bearing factorization legs:
  decode z R² & Mess3 R²; T2 erase-one/decode-other (z|noM, m|noZ); T4 transfer aligned→misaligned (disjoint
  seqs); data coupling corr(q,sharpness); z signal strength (q_std).

Question: does the capability factorization (transfer ≈ in-context, m|noZ ≈ mR²) hold across switching rate,
or does fast persona switching entangle the codes? FALSIFIER: transfer/m|noZ drop well below mR² at high rate.

Outputs results/ghmm_switchrate.pt
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

L = int(os.environ.get("SR_L", "96"))
STEPS = int(os.environ.get("SR_STEPS", "2500"))
BATCH = int(os.environ.get("SR_BATCH", "96"))
NEV = int(os.environ.get("SR_NEV", "2048"))
NSEED = int(os.environ.get("SR_NSEED", "2"))
LR = 1.5e-3
STAY, A = 0.30, 0.80
# (eps,gamma) with gamma=3*eps ⇒ q*=0.25 fixed; magnitude = switching rate
RATES = [(0.01, 0.03), (0.02, 0.06), (0.06, 0.18), (0.12, 0.36)]
if os.environ.get("SR_RATES"):
    RATES = [tuple(float(y) for y in x.split(",")) for x in os.environ["SR_RATES"].split(";")]


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
    w = np.linalg.solve(Amat, Xtr.T @ (y[tr] - ym)); pred = Xte @ w + ym
    ss = ((y[te] - pred) ** 2).sum(); tot = ((y[te] - y[te].mean()) ** 2).sum() + 1e-12
    return {"r2": float(1 - ss / tot), "direction": w / sd}


def run(eps, gamma, seed, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); erng = np.random.default_rng(777)
    P = ghmm.joint_params_drift(eps, gamma, STAY, A)
    obs_ev, c_ev, s_ev = ghmm.gen_ghmm(NEV, L, P, erng)
    belief_ev, q_ev, mbel_ev, nextp_ev = ghmm.forward_filter_ghmm(obs_ev, P)
    obs_ev_t = torch.tensor(obs_ev, device=device)
    z_ev = np.log(np.clip(q_ev, 1e-4, 1 - 1e-4) / np.clip(1 - q_ev, 1e-4, 1 - 1e-4))
    mxy_ev = ghmm.simplex_xy(mbel_ev)

    # oracle calibration gate (q*) + switching sanity
    qstar = eps / (eps + gamma)
    emp_pM = float(c_ev[:, 20:].mean())
    assert abs(emp_pM - qstar) < 0.03, f"q* miscalibrated at eps={eps}: emp {emp_pM:.3f} vs {qstar:.3f}"
    # mean dwell ~ 1/switch; report realized switch fraction
    switch_frac = float((c_ev[:, 1:] != c_ev[:, :-1]).mean())

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
    model.eval(); kl = out["snapshots"][STEPS]["metrics"]["kl"]

    pos = list(range(20, L - 1, 4)); npos = len(pos); B = obs_ev.shape[0]
    seq_ids = np.repeat(np.arange(B), npos)
    qf = q_ev[:, pos].reshape(-1); maxc = mbel_ev[:, pos, :].max(-1).reshape(-1)
    corr = float(np.corrcoef(qf, maxc)[0, 1]); q_std = float(qf.std())
    zt = z_ev[:, pos].reshape(-1); mx = mxy_ev[:, pos, 0].reshape(-1); my = mxy_ev[:, pos, 1].reshape(-1)

    decode = {}
    for layer in range(cfg.n_layers):
        R = probe.extract_resid(model, obs_ev_t[:, :-1], layer=layer, pos=pos)
        pz = grouped_ridge(R, zt, seq_ids); pmx = grouped_ridge(R, mx, seq_ids); pmy = grouped_ridge(R, my, seq_ids)
        decode[layer] = {"z_r2": pz["r2"], "m_r2": 0.5 * (pmx["r2"] + pmy["r2"]),
                         "_R": R, "_pmx": pmx, "_pmy": pmy}
    bz = max(decode, key=lambda l: decode[l]["z_r2"]); bm = max(decode, key=lambda l: decode[l]["m_r2"])
    z_r2 = decode[bz]["z_r2"]; m_r2 = decode[bm]["m_r2"]

    # T2 separability at best-Mess3 layer
    layer = bm; R = decode[layer]["_R"]
    Qm, _ = np.linalg.qr(np.stack([decode[layer]["_pmx"]["direction"], decode[layer]["_pmy"]["direction"]], axis=1))
    qb = np.clip((qf * 5).astype(int), 0, 4)
    means = np.stack([R[qb == bb].mean(0) for bb in range(5) if (qb == bb).sum() > 50], axis=0) - R.mean(0)
    Uz, _, _ = np.linalg.svd(means.T, full_matrices=False); Zsub = Uz[:, :min(3, Uz.shape[1])]
    m_after = 0.5 * (grouped_ridge(R - (R @ Zsub) @ Zsub.T, mx, seq_ids)["r2"]
                     + grouped_ridge(R - (R @ Zsub) @ Zsub.T, my, seq_ids)["r2"])
    z_after = grouped_ridge(R - (R @ Qm) @ Qm.T, zt, seq_ids)["r2"]

    # T4 transfer across disjoint seqs
    cflat = c_ev[:, pos].reshape(-1); rgt = np.random.default_rng(5); seqperm = rgt.permutation(B)
    poolA = set(seqperm[:B // 2].tolist()); in_poolA = np.array([sid in poolA for sid in seq_ids])
    tr_mask = in_poolA & (cflat == 0); te_mask = (~in_poolA) & (cflat == 1)

    def fit_eval(Xtr, ytr, Xte, yte):
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
        Xtr2 = (Xtr - mu) / sd; Xte2 = (Xte - mu) / sd
        Amat = Xtr2.T @ Xtr2 + np.eye(Xtr2.shape[1]); ym = ytr.mean()
        w = np.linalg.solve(Amat, Xtr2.T @ (ytr - ym)); pred = Xte2 @ w + ym
        ss = ((yte - pred) ** 2).sum(); tot = ((yte - yte.mean()) ** 2).sum() + 1e-12; return float(1 - ss / tot)
    transfer = 0.5 * (fit_eval(R[tr_mask], mx[tr_mask], R[te_mask], mx[te_mask])
                      + fit_eval(R[tr_mask], my[tr_mask], R[te_mask], my[te_mask]))

    res = {"eps": eps, "gamma": gamma, "seed": seed, "kl": kl, "qstar": qstar, "switch_frac": switch_frac,
           "corr_q_sharp": corr, "q_std": q_std, "z_r2": z_r2, "m_r2": m_r2,
           "z_r2_after_Merase": z_after, "m_r2_after_zerase": m_after, "transfer_m": transfer,
           "train_min": (time.time() - t0) / 60}
    for l in decode:
        for kk in ("_R", "_pmx", "_pmy"):
            decode[l].pop(kk, None)
    print(f"  [eps={eps:.2f},g={gamma:.2f} s{seed}] switch={switch_frac:.3f} KL={kl:.4f} corr={corr:+.3f} "
          f"q_std={q_std:.3f} | zR²={z_r2:.3f} mR²={m_r2:.3f} | z|noM={z_after:.3f} m|noZ={m_after:.3f} | "
          f"T4tr={transfer:.3f} ({res['train_min']:.1f}m)", flush=True)
    return res


def main():
    print(f"=== GHMM persona-switching-rate robustness (drift a={A}, q*=0.25 fixed; {NSEED} seeds) ===", flush=True)
    all_res = []
    for (eps, gamma) in RATES:
        for s in range(NSEED):
            all_res.append(run(eps, gamma, s))
    agg = {}
    for (eps, gamma) in RATES:
        rs = [r for r in all_res if r["eps"] == eps and r["gamma"] == gamma]
        def col(k): return np.array([r[k] for r in rs], float)
        agg[f"{eps},{gamma}"] = {k: (float(col(k).mean()), float(col(k).std()))
                                 for k in ["switch_frac", "corr_q_sharp", "q_std", "z_r2", "m_r2",
                                           "z_r2_after_Merase", "m_r2_after_zerase", "transfer_m"]}
    torch.save({"rows": all_res, "agg": agg,
                "config": {"L": L, "STEPS": STEPS, "NEV": NEV, "nseed": NSEED, "stay": STAY, "a": A,
                           "rates": RATES}}, f"{OUT}/ghmm_switchrate.pt")
    print(f"saved {OUT}/ghmm_switchrate.pt", flush=True)
    print("\n=== aggregate (mean±sd) — does factorization survive fast persona switching? ===")
    print(f"  {'eps,g':>11} {'switch':>7} {'corr':>7} {'q_std':>6} {'zR²':>6} {'mR²':>6} {'z|noM':>6} {'m|noZ':>6} {'T4tr':>6}")
    for (eps, gamma) in RATES:
        x = agg[f"{eps},{gamma}"]
        print(f"  {eps:.2f},{gamma:.2f} {x['switch_frac'][0]:7.3f} {x['corr_q_sharp'][0]:+7.3f} "
              f"{x['q_std'][0]:6.3f} {x['z_r2'][0]:6.3f} {x['m_r2'][0]:6.3f} {x['z_r2_after_Merase'][0]:6.3f} "
              f"{x['m_r2_after_zerase'][0]:6.3f} {x['transfer_m'][0]:6.3f}")


if __name__ == "__main__":
    main()
