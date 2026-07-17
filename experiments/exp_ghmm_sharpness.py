"""
GHMM emission-sharpness ablation: does the alignment ⊗ capability FACTORIZATION (§4.4 C3)
survive when the capability (Mess3) belief geometry is genuinely RICH (history-dependent),
not just last-symbol-predictable?

§7 carries an honest caveat: at the C3 emission sharpness (a=0.80) the Mess3 content belief
is "largely last-symbol-predictable (control R²≈0.95), so C3 demonstrates the factorization of
CODES, not rich fractal belief geometry." This experiment removes that caveat (or finds its
boundary) by sweeping the emission concentration `a` DOWN in the SAME near-factored DRIFT regime
(alignment = content-cycle DIRECTION, matched stickiness → corr(q,sharpness) stays small at all a):

    a = 0.80 (sharp, the C3 point)  →  0.55  →  0.40 (diffuse: each state emits its own symbol
    w.p. 0.40, the other two 0.30 each — the belief now needs HISTORY, so the last-symbol control
    R² for Mess3 should DROP well below 0.95, exposing the fractal Mess3 simplex geometry).

    To make the geometry GENUINELY history-rich we run in a STICKY drift regime (stay=0.70): with
    fast-mixing transitions (the C3 stay=0.30) the belief stays last-symbol-dominated even at low a
    (process-level last-symbol R² only 0.95→0.90), but with sticky transitions the belief integrates
    a long history, so lowering a drives last-symbol R² 0.87→0.62 (while last-TWO-symbol R² stays
    ≈0.9 — the belief depends on several recent symbols, the signature of rich geometry) AND the
    alignment↔sharpness coupling stays LOW (corr≈−0.08, even more factored than the C3 −0.13), with
    alignment z purely history-derived (last-symbol R²(z)≈0). So this cleanly varies belief richness
    while holding the near-factored structure fixed.

For each (a, seed) we anchor to the EXACT 6-state forward filter (per-a calibration assert), then
train a TinyGPT and measure the load-bearing factorization battery:
  RICHNESS  last-symbol control R² for Mess3 (lower a ⇒ richer geometry ⇒ this drops)
  DECODE    z R² and Mess3 R² (best seq-grouped layer)
  T2        separability: z R² after erasing the Mess3-plane; Mess3 R² after erasing a z-subspace
  T4        transfer: Mess3 probe fit on aligned-context tokens, tested on misaligned-context
            tokens from DISJOINT sequences (the strongest, non-circular factorization test)
  T3        geometry: footprint of d_align on the Mess3 probe + bounded causal cross-talk
            (clipped-R² drop under matched-norm alignment steer) vs a random-⊥ null band

HOPED result (upgrades §7): as a drops, last-symbol-control R² for Mess3 falls (geometry gets
rich) yet T4 transfer stays high and footprint stays ≈0 — factorization is robust to belief-
geometry complexity. FALSIFIER: transfer breaks or footprint rises at low a ⇒ the factorization
was contingent on weak (last-symbol) geometry; we report that boundary honestly.

Outputs results/ghmm_sharpness.pt
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

L = int(os.environ.get("SH_L", "96"))
STEPS = int(os.environ.get("SH_STEPS", "2000"))
BATCH = int(os.environ.get("SH_BATCH", "96"))
NEV = int(os.environ.get("SH_NEV", "2048"))
NSEED = int(os.environ.get("SH_NSEED", "3"))
LR = 1.5e-3
EPS, GAMMA = 0.02, 0.06
STAY = float(os.environ.get("SH_STAY", "0.70"))  # sticky ⇒ belief integrates history ⇒ rich geometry
A_LIST = [float(x) for x in os.environ.get("SH_AS", "0.80,0.55,0.40").split(",")]


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
    direction = w / sd
    return {"r2": float(1 - ss / tot), "direction": direction, "intercept": float(ym - mu @ direction)}


def calib_check(obs_ev, P):
    """Per-a oracle gate: empirical P(hidden mode=M | filtered q-bin) tracks the bin center, and
    empirical P(content state = argmax belief | belief-max bin) tracks the bin center. Asserts the
    exact 6-state filter is calibrated at THIS emission concentration before we trust the probes."""
    belief, q, mbel, nextp = ghmm.forward_filter_ghmm(obs_ev, P)
    # we need the true hidden states to score calibration
    return belief, q, mbel, nextp


def run(a, seed, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); erng = np.random.default_rng(777)
    P = ghmm.joint_params_drift(EPS, GAMMA, STAY, a)
    obs_ev, c_ev, s_ev = ghmm.gen_ghmm(NEV, L, P, erng)
    belief_ev, q_ev, mbel_ev, nextp_ev = ghmm.forward_filter_ghmm(obs_ev, P)
    obs_ev_t = torch.tensor(obs_ev, device=device)
    z_ev = np.log(np.clip(q_ev, 1e-4, 1 - 1e-4) / np.clip(1 - q_ev, 1e-4, 1 - 1e-4))
    mxy_ev = ghmm.simplex_xy(mbel_ev)

    # --- per-a oracle calibration gate (finite-sample sanity guard; the filter math itself is the
    # exact 6-state forward filter, already validation-gated generically by validate_ghmm.py). We
    # require adequate bin counts so sparse high-/low-q bins don't produce noisy false failures. ---
    cflat_all = c_ev[:, 20:].reshape(-1); qflat_all = q_ev[:, 20:].reshape(-1)
    cal_err = 0.0
    for b0 in np.linspace(0.15, 0.85, 8):
        m = (qflat_all > b0 - 0.075) & (qflat_all <= b0 + 0.075)
        if m.sum() > 800:
            cal_err = max(cal_err, abs(cflat_all[m].mean() - qflat_all[m].mean()))
    sflat = s_ev[:, 20:].reshape(-1); mbflat = mbel_ev[:, 20:, :].reshape(-1, 3)
    shat_all = np.argmax(mbflat, -1); pmax_all = mbflat.max(-1)
    cal_err_m = 0.0
    for b0 in np.linspace(0.45, 0.85, 6):
        m = (pmax_all > b0 - 0.06) & (pmax_all <= b0 + 0.06)
        if m.sum() > 800:
            cal_err_m = max(cal_err_m, abs((shat_all[m] == sflat[m]).mean() - pmax_all[m].mean()))
    assert cal_err < 0.045, f"alignment-filter miscalibrated at a={a}: {cal_err:.3f}"
    assert cal_err_m < 0.05, f"Mess3-belief miscalibrated at a={a}: {cal_err_m:.3f}"

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

    pos = list(range(20, L - 1, 4)); npos = len(pos); B = obs_ev.shape[0]
    seq_ids = np.repeat(np.arange(B), npos)
    qf = q_ev[:, pos].reshape(-1); maxc = mbel_ev[:, pos, :].max(-1).reshape(-1)
    corr = float(np.corrcoef(qf, maxc)[0, 1])
    zt = z_ev[:, pos].reshape(-1)
    mx = mxy_ev[:, pos, 0].reshape(-1); my = mxy_ev[:, pos, 1].reshape(-1)

    # --- decode z, Mess3 at each layer; pick best ---
    decode = {}
    for layer in range(cfg.n_layers):
        R = probe.extract_resid(model, obs_ev_t[:, :-1], layer=layer, pos=pos)
        pz = grouped_ridge(R, zt, seq_ids); pmx = grouped_ridge(R, mx, seq_ids); pmy = grouped_ridge(R, my, seq_ids)
        decode[layer] = {"z_r2": pz["r2"], "m_r2": 0.5 * (pmx["r2"] + pmy["r2"]),
                         "_R": R, "_pz": pz, "_pmx": pmx, "_pmy": pmy}
    bz = max(decode, key=lambda l: decode[l]["z_r2"]); bm = max(decode, key=lambda l: decode[l]["m_r2"])
    z_r2 = decode[bz]["z_r2"]; m_r2 = decode[bm]["m_r2"]

    # --- RICHNESS controls: last-symbol & windowed-count predictability of Mess3 (and z) ---
    obs_in = obs_ev[:, :-1]; last_oh = np.eye(3)[obs_in[:, pos].reshape(-1)]
    W = 12; cnts = np.zeros((B, npos, 3))
    for k in range(3):
        cs = np.cumsum((obs_in == k).astype(float), axis=1)
        for j, pp in enumerate(pos):
            lo = max(0, pp - W); cnts[:, j, k] = cs[:, pp] - (cs[:, lo - 1] if lo > 0 else 0)
    cnts = cnts.reshape(-1, 3)
    m_lastsym = 0.5 * (grouped_ridge(last_oh, mx, seq_ids)["r2"] + grouped_ridge(last_oh, my, seq_ids)["r2"])
    m_count = 0.5 * (grouped_ridge(cnts, mx, seq_ids)["r2"] + grouped_ridge(cnts, my, seq_ids)["r2"])
    z_lastsym = grouped_ridge(last_oh, zt, seq_ids)["r2"]

    # --- T2 separability (erase-one / decode-other) at the best-Mess3 layer ---
    layer = bm; R = decode[layer]["_R"]
    dmx = decode[layer]["_pmx"]["direction"]; dmy = decode[layer]["_pmy"]["direction"]
    Qm, _ = np.linalg.qr(np.stack([dmx, dmy], axis=1))
    qb = np.clip((qf * 5).astype(int), 0, 4)
    means = np.stack([R[qb == bb].mean(0) for bb in range(5) if (qb == bb).sum() > 50], axis=0) - R.mean(0)
    Uz, _, _ = np.linalg.svd(means.T, full_matrices=False); Zsub = Uz[:, :min(3, Uz.shape[1])]
    R_noZ = R - (R @ Zsub) @ Zsub.T
    m_after = 0.5 * (grouped_ridge(R_noZ, mx, seq_ids)["r2"] + grouped_ridge(R_noZ, my, seq_ids)["r2"])
    R_noM = R - (R @ Qm) @ Qm.T
    z_after = grouped_ridge(R_noM, zt, seq_ids)["r2"]

    # --- T4 transfer across DISJOINT sequences (aligned-context -> misaligned-context) ---
    cflat = c_ev[:, pos].reshape(-1)
    rgt = np.random.default_rng(5); seqperm = rgt.permutation(B); poolA = set(seqperm[:B // 2].tolist())
    in_poolA = np.array([sid in poolA for sid in seq_ids])
    tr_mask = in_poolA & (cflat == 0); te_mask = (~in_poolA) & (cflat == 1)

    def fit_eval(Xtr, ytr, Xte, yte):
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
        Xtr2 = (Xtr - mu) / sd; Xte2 = (Xte - mu) / sd
        Amat = Xtr2.T @ Xtr2 + 1.0 * np.eye(Xtr2.shape[1]); ym = ytr.mean()
        w = np.linalg.solve(Amat, Xtr2.T @ (ytr - ym)); pred = Xte2 @ w + ym
        ss = ((yte - pred) ** 2).sum(); tot = ((yte - yte.mean()) ** 2).sum() + 1e-12; return float(1 - ss / tot)
    tr_mx = fit_eval(R[tr_mask], mx[tr_mask], R[te_mask], mx[te_mask])
    tr_my = fit_eval(R[tr_mask], my[tr_mask], R[te_mask], my[te_mask])
    transfer = 0.5 * (tr_mx + tr_my)

    # --- T3 geometry: footprint of d_align on Mess3 probe + bounded steer cross-talk ---
    cand = [l for l in range(cfg.n_layers - 1) if decode[l]["z_r2"] > 0.4]
    slayer = cand[0] if cand else max(0, bz - 1)
    Rs = probe.extract_resid(model, obs_ev_t[:, :-1], layer=slayer, pos=pos)
    hi = qf > np.quantile(qf, 0.8); lo = qf < np.quantile(qf, 0.2)
    d_align = Rs[hi].mean(0) - Rs[lo].mean(0); d_align /= (np.linalg.norm(d_align) + 1e-9)
    pmx_s = grouped_ridge(Rs, mx, seq_ids); pmy_s = grouped_ridge(Rs, my, seq_ids)
    wmx, bx = pmx_s["direction"], pmx_s["intercept"]; wmy, by = pmy_s["direction"], pmy_s["intercept"]

    def footprint(d):
        return 0.5 * (abs(d @ wmx) / (np.linalg.norm(wmx) + 1e-9) + abs(d @ wmy) / (np.linalg.norm(wmy) + 1e-9))
    typ = float(np.linalg.norm(Rs, axis=1).mean()); ALPHA = 0.5 * typ

    def steer_resid(direction, alpha):
        d = torch.tensor(direction / (np.linalg.norm(direction) + 1e-9), dtype=torch.float32, device=device)
        with torch.no_grad():
            x = model._embed(obs_ev_t[:, :-1]); Rst = None
            for i, blk in enumerate(model.blocks):
                x = blk(x)
                if i == slayer:
                    mask = torch.zeros(x.shape[1], device=device); mask[torch.as_tensor(pos)] = 1.0
                    x = x + float(alpha) * d * mask[None, :, None]
                    Rst = x[:, pos, :].cpu().numpy().reshape(-1, x.shape[-1]); break
        return Rst

    def cap_clip(Rst):
        def r2(pred, tgt):
            ss = ((tgt - pred) ** 2).sum(); tot = ((tgt - tgt.mean()) ** 2).sum() + 1e-12
            return max(0.0, float(1 - ss / tot))
        return 0.5 * (r2(Rst @ wmx + bx, mx) + r2(Rst @ wmy + by, my))
    cap0 = cap_clip(steer_resid(d_align, 0.0))
    fp_align = float(footprint(d_align)); drop_align = cap0 - cap_clip(steer_resid(d_align, ALPHA))
    rgc = np.random.default_rng(99 + seed); rdrop = []; rfp = []
    for _ in range(8):
        dr = rgc.standard_normal(d_align.shape); dr = dr - (dr @ d_align) * d_align; dr /= (np.linalg.norm(dr) + 1e-9)
        rdrop.append(cap0 - cap_clip(steer_resid(dr, ALPHA))); rfp.append(float(footprint(dr)))

    res = {"a": a, "seed": seed, "kl": kl, "corr_q_sharp": corr, "cal_err": cal_err, "cal_err_m": cal_err_m,
           "z_r2": z_r2, "m_r2": m_r2, "best_z_layer": bz, "best_m_layer": bm, "steer_layer": slayer,
           "m_lastsym_ctrl": m_lastsym, "m_count_ctrl": m_count, "z_lastsym_ctrl": z_lastsym,
           "z_r2_after_Merase": z_after, "m_r2_after_zerase": m_after,
           "transfer_m": transfer, "transfer_mx": tr_mx, "transfer_my": tr_my,
           "footprint_align": fp_align, "footprint_rand_mean": float(np.mean(rfp)),
           "cap0": cap0, "cap_drop_align": drop_align,
           "cap_drop_rand_mean": float(np.mean(rdrop)), "cap_drop_rand_sd": float(np.std(rdrop)),
           "train_min": (time.time() - t0) / 60}
    for l in decode:
        for kk in ("_R", "_pz", "_pmx", "_pmy"):
            decode[l].pop(kk, None)
    print(f"  [a={a:.2f} s{seed}] KL={kl:.4f} corr(q,sharp)={corr:+.3f} calE={cal_err:.3f}/{cal_err_m:.3f} | "
          f"zR²={z_r2:.3f} mR²={m_r2:.3f} | m_lastsym_ctrl={m_lastsym:.3f} (RICHNESS) | "
          f"sep z|noM={z_after:.3f} m|noZ={m_after:.3f} | T4transfer={transfer:.3f} | "
          f"fp_align={fp_align:.4f}(rand {np.mean(rfp):.3f}) capDrop={drop_align:+.3f}(rand {np.mean(rdrop):+.3f}) "
          f"({res['train_min']:.1f}m)", flush=True)
    return res


def main():
    print(f"=== GHMM emission-sharpness ablation: a in {A_LIST} x {NSEED} seeds "
          f"(drift regime; L={L},steps={STEPS},NEV={NEV}) ===", flush=True)
    all_res = []
    for a in A_LIST:
        for s in range(NSEED):
            all_res.append(run(a, s))
    agg = {}
    for a in A_LIST:
        rs = [r for r in all_res if r["a"] == a]
        def col(k): return np.array([r[k] for r in rs], float)
        agg[a] = {k: (float(col(k).mean()), float(col(k).std()))
                  for k in ["corr_q_sharp", "z_r2", "m_r2", "m_lastsym_ctrl", "m_count_ctrl",
                            "z_r2_after_Merase", "m_r2_after_zerase", "transfer_m",
                            "footprint_align", "footprint_rand_mean", "cap_drop_align", "cap_drop_rand_mean"]}
    torch.save({"rows": all_res, "agg": agg,
                "config": {"L": L, "STEPS": STEPS, "NEV": NEV, "nseed": NSEED, "eps": EPS, "gamma": GAMMA,
                           "stay": STAY, "a_list": A_LIST}}, f"{OUT}/ghmm_sharpness.pt")
    print(f"saved {OUT}/ghmm_sharpness.pt", flush=True)
    print("\n=== aggregate (mean±sd) — does factorization survive richer belief geometry? ===")
    print(f"  {'a':>5} {'corr':>7} {'zR²':>6} {'mR²':>6} {'m_last(RICH)':>13} {'z|noM':>7} {'m|noZ':>7} "
          f"{'T4tr':>6} {'fp_al':>7} {'fp_rnd':>7} {'capDrop':>8} {'rnd':>7}")
    for a in A_LIST:
        x = agg[a]
        print(f"  {a:5.2f} {x['corr_q_sharp'][0]:+7.3f} {x['z_r2'][0]:6.3f} {x['m_r2'][0]:6.3f} "
              f"{x['m_lastsym_ctrl'][0]:13.3f} {x['z_r2_after_Merase'][0]:7.3f} {x['m_r2_after_zerase'][0]:7.3f} "
              f"{x['transfer_m'][0]:6.3f} {x['footprint_align'][0]:7.4f} {x['footprint_rand_mean'][0]:7.3f} "
              f"{x['cap_drop_align'][0]:+8.3f} {x['cap_drop_rand_mean'][0]:+7.3f}")


if __name__ == "__main__":
    main()
