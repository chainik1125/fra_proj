"""
GHMM independence sweep: does the model's REPRESENTATIONAL separability track the
GENERATIVE independence of the two latents (alignment z  vs  Mess3 capability belief)?

The drift regime (exp_ghmm_factored.py) is one point: alignment sets the content-cycle
DIRECTION (+1 aligned / -1 misaligned) with MATCHED stickiness, so belief sharpness is
~alignment-symmetric and the two latents are near-independent (data corr(q,sharpness)~0).
The entangled regime is another point: alignment sets the transition RATE, so sharpness is
alignment-coupled (corr~-0.8). Those are 2 points. Here we draw the WHOLE CURVE with a single
clean knob lambda in [0,1] that keeps alignment strongly decodable throughout:

    aligned mode:  drift +1, stickiness  stay0 + lambda*dstay   (sharper belief)
    misaligned  :  drift -1, stickiness  stay0 - lambda*dstay   (more diffuse belief)

At lambda=0 the stickiness is matched (factored: alignment lives purely in the cycling
DIRECTION, orthogonal to within-step belief sharpness). As lambda->1 the misaligned mode
becomes erratic while the aligned mode becomes sticky, so belief sharpness becomes a proxy
for alignment -> the two GENERATIVE latents become entangled. Crucially the +1/-1 drift keeps
alignment identifiable at every lambda, so z stays decodable and we can ask the clean question:
*as the generative latents become coupled, do their neural codes become coupled too?*

Everything is anchored to the EXACT 6-state Bayes filter (built from the same joint_params_from
path validated by validate_ghmm.py); we additionally re-check filter calibration at each lambda.

Outputs results/ghmm_sweep.pt
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

L = int(os.environ.get("SW_L", "96"))
STEPS = int(os.environ.get("SW_STEPS", "1500"))
BATCH = int(os.environ.get("SW_BATCH", "96"))
NEV = int(os.environ.get("SW_NEV", "2048"))
LR = 1.5e-3
# Single coupling knob lambda: keep the drift-direction alignment code (matched stickiness, so z
# stays decodable at every lambda) and couple the two latents via an EMISSION-SHARPNESS asymmetry
# (belief sharpness is dominated by emission concentration, so this is the effective knob): aligned
# emission gets sharper (A0+lam*DA), misaligned more diffuse (A0-lam*DA). At lambda=0 emission is
# matched (near-factored); as lambda->1 belief sharpness becomes an alignment proxy (entangled).
EPS, GAMMA, STAY0, A0, DA = 0.02, 0.06, 0.30, 0.70, 0.22
LAMBDAS = [float(x) for x in os.environ.get("SW_LAMBDAS", "0.0,0.25,0.5,0.75,1.0").split(",")]


def params_lambda(lam):
    """Drift-direction alignment code (matched stickiness) + lambda emission-sharpness asymmetry."""
    TrA = ghmm.mess3_trans_drift(STAY0, +1)
    TrM = ghmm.mess3_trans_drift(STAY0, -1)
    EA = ghmm.mess3_emit(A0 + lam * DA)   # aligned: sharper belief
    EM = ghmm.mess3_emit(A0 - lam * DA)   # misaligned: more diffuse belief
    return ghmm.joint_params_from(EPS, GAMMA, TrA, TrM, EA, EM,
                                  meta={"regime": "lambda_sweep", "lambda": lam,
                                        "aA": A0 + lam * DA, "aM": A0 - lam * DA})


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
    return {"r2": float(1 - ss / tot), "direction": w / sd, "intercept": float(ym - mu @ (w / sd))}


def filter_calib_err(obs, P):
    """Max |emp P(c=M|q-bin) - bin centre| over populated bins -- exactness check at this lambda."""
    belief, q, mbel, nextp = ghmm.forward_filter_ghmm(obs, P)
    # need true hidden c; regenerate is not available here, so check next-symbol self-consistency:
    # E[next_p] vs empirical next-symbol freq per predicted-prob bin (calibration of the filter).
    o = obs
    pred0 = nextp[:, :-1, 0].reshape(-1)
    nxt0 = (o[:, 1:] == 0).astype(float).reshape(-1)
    errs = []
    for b in np.linspace(0.05, 0.95, 10):
        m = np.abs(pred0 - b) < 0.05
        if m.sum() > 500:
            errs.append(abs(nxt0[m].mean() - pred0[m].mean()))
    return float(max(errs)) if errs else 0.0


def run_lambda(lam, seed=0, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); erng = np.random.default_rng(777)
    P = params_lambda(lam)
    obs_ev, c_ev, s_ev = ghmm.gen_ghmm(NEV, L, P, erng)
    belief_ev, q_ev, mbel_ev, nextp_ev = ghmm.forward_filter_ghmm(obs_ev, P)
    calib = filter_calib_err(obs_ev, P)
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

    # generative entanglement of the two latents (data-side)
    qf = q_ev[:, pos].reshape(-1); maxc = mbel_ev[:, pos, :].max(-1).reshape(-1)
    bent = (-(np.clip(mbel_ev, 1e-9, 1) * np.log(np.clip(mbel_ev, 1e-9, 1))).sum(-1))[:, pos].reshape(-1)
    corr_q_maxcoord = float(np.corrcoef(qf, maxc)[0, 1])
    corr_q_entropy = float(np.corrcoef(qf, bent)[0, 1])
    q_std = float(qf.std())

    zt = z_ev[:, pos].reshape(-1)
    mx = mxy_ev[:, pos, 0].reshape(-1); my = mxy_ev[:, pos, 1].reshape(-1)

    # decodability per layer
    decode = {}
    for layer in range(cfg.n_layers):
        R = probe.extract_resid(model, obs_ev_t[:, :-1], layer=layer, pos=pos)
        pz = grouped_ridge(R, zt, seq_ids); pmx = grouped_ridge(R, mx, seq_ids); pmy = grouped_ridge(R, my, seq_ids)
        decode[layer] = {"z_r2": pz["r2"], "m_r2": 0.5 * (pmx["r2"] + pmy["r2"]),
                         "_pz": pz, "_pmx": pmx, "_pmy": pmy, "_R": R}
    bz = max(decode, key=lambda l: decode[l]["z_r2"]); bm = max(decode, key=lambda l: decode[l]["m_r2"])

    # representational separability at the best-Mess3 layer
    layer = bm; R = decode[layer]["_R"]
    dz = decode[layer]["_pz"]["direction"]; dz = dz / (np.linalg.norm(dz) + 1e-9)
    Qm, _ = np.linalg.qr(np.stack([decode[layer]["_pmx"]["direction"], decode[layer]["_pmy"]["direction"]], axis=1))
    cos_in_plane = float(np.linalg.norm(Qm @ (Qm.T @ dz)))
    rg = np.random.default_rng(7); floors = []
    for _ in range(200):
        v = rg.standard_normal(dz.shape); v /= np.linalg.norm(v); floors.append(np.linalg.norm(Qm @ (Qm.T @ v)))
    null_floor = float(np.mean(floors))
    # remove a multi-dim z-subspace (q-bin means), test Mess3 survival
    qb = np.clip((qf * 5).astype(int), 0, 4)
    means = np.stack([R[qb == b].mean(0) for b in range(5) if (qb == b).sum() > 50], axis=0) - R.mean(0)
    Uz, _, _ = np.linalg.svd(means.T, full_matrices=False); Zsub = Uz[:, :min(3, Uz.shape[1])]
    R_noZ = R - (R @ Zsub) @ Zsub.T
    m_after = 0.5 * (grouped_ridge(R_noZ, mx, seq_ids)["r2"] + grouped_ridge(R_noZ, my, seq_ids)["r2"])
    R_noM = R - (R @ Qm) @ Qm.T
    z_after = grouped_ridge(R_noM, zt, seq_ids)["r2"]
    # cross-decode: can the z-direction's score predict the Mess3 belief (coupling)?
    z_proj = (R @ dz)
    cross_r2 = grouped_ridge(z_proj[:, None], maxc, seq_ids)["r2"]

    # ---- causal steering cross-talk at a FIXED layer & FIXED relative norm (removes the
    # layer/norm confound of the 2-point drift-vs-entangled contrast). BOUNDED metrics only
    # (the unbounded cap-R² collapse that §4.4 disowns is replaced by: clipped-R² drop in [0,1]
    # AND model-intrinsic next-symbol KL steered‖unsteered). Random control = a NULL BAND over
    # many matched-norm ⊥ directions, mean±sd, with probe-footprint reported to rule out
    # circularity (is align's small effect just because d_align ≈ ⊥ to the capability probe?). ----
    SLAYER = int(os.environ.get("SW_SLAYER", "1"))
    NRAND = int(os.environ.get("SW_NRAND", "8"))
    Rs = probe.extract_resid(model, obs_ev_t[:, :-1], layer=SLAYER, pos=pos)
    qpos = q_ev[:, pos].reshape(-1)
    hi = qpos > np.quantile(qpos, 0.8); lo = qpos < np.quantile(qpos, 0.2)
    d_align = Rs[hi].mean(0) - Rs[lo].mean(0); d_align /= (np.linalg.norm(d_align) + 1e-9)
    pmx_s = grouped_ridge(Rs, mx, seq_ids); pmy_s = grouped_ridge(Rs, my, seq_ids)
    wmx, bx = pmx_s["direction"], pmx_s["intercept"]; wmy, by = pmy_s["direction"], pmy_s["intercept"]
    typ = float(np.linalg.norm(Rs, axis=1).mean()); ALPHA = float(os.environ.get("SW_ALPHA", "0.5")) * typ

    def footprint(d):  # |d·ŵ| / ‖ŵ‖ on the capability probe weights (circularity diagnostic)
        return 0.5 * (abs(d @ wmx) / (np.linalg.norm(wmx) + 1e-9) + abs(d @ wmy) / (np.linalg.norm(wmy) + 1e-9))

    def steer_resid(direction, alpha):
        d = torch.tensor(direction / (np.linalg.norm(direction) + 1e-9), dtype=torch.float32, device=device)
        with torch.no_grad():
            x = model._embed(obs_ev_t[:, :-1]); Rst = None
            for i, blk in enumerate(model.blocks):
                x = blk(x)
                if i == SLAYER:
                    mask = torch.zeros(x.shape[1], device=device); mask[torch.as_tensor(pos)] = 1.0
                    x = x + float(alpha) * d * mask[None, :, None]
                    Rst = x[:, pos, :].cpu().numpy().reshape(-1, x.shape[-1]); break
        return Rst

    def steer_p(direction, alpha):  # full steered forward -> next-symbol dist at pos
        d = torch.tensor(direction / (np.linalg.norm(direction) + 1e-9), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model.forward_with_steer(obs_ev_t[:, :-1], layer=SLAYER, direction=d,
                                              alpha=float(alpha), positions=pos)
            return torch.softmax(logits[:, pos, :], -1).cpu().numpy().reshape(-1, 3)

    def cap_clip(Rst):  # clipped-R² capability readout in [0,1]
        def r2(pred, tgt):
            ss = ((tgt - pred) ** 2).sum(); tot = ((tgt - tgt.mean()) ** 2).sum() + 1e-12
            return max(0.0, float(1 - ss / tot))
        return 0.5 * (r2(Rst @ wmx + bx, mx) + r2(Rst @ wmy + by, my))

    cap_clip0 = cap_clip(steer_resid(d_align, 0.0))
    cap_clip_align = cap_clip(steer_resid(d_align, ALPHA))
    p0 = torch.softmax(model(obs_ev_t[:, :-1])[:, pos, :], -1).detach().cpu().numpy().reshape(-1, 3) \
        if False else steer_p(d_align, 0.0)
    nsk_align = float(kl_cat(steer_p(d_align, ALPHA), p0).mean())
    # random null band (matched-norm ⊥ directions)
    rgc = np.random.default_rng(99); rand_clip = []; rand_nsk = []; rand_fp = []
    for j in range(NRAND):
        dr = rgc.standard_normal(d_align.shape); dr = dr - (dr @ d_align) * d_align
        dr /= (np.linalg.norm(dr) + 1e-9)
        rand_clip.append(cap_clip(steer_resid(dr, ALPHA))); rand_fp.append(footprint(dr))
        if j < 3:  # next-symbol KL is the expensive full-forward path; sample 3 of the null
            rand_nsk.append(float(kl_cat(steer_p(dr, ALPHA), p0).mean()))
    cap0 = cap_clip0  # back-compat name

    res = {"lambda": lam, "seed": seed, "kl": kl, "filter_calib_err": calib,
           "aA": A0 + lam * DA, "aM": A0 - lam * DA,
           "corr_q_maxcoord": corr_q_maxcoord, "corr_q_entropy": corr_q_entropy, "q_std": q_std,
           "z_r2": decode[bz]["z_r2"], "m_r2": decode[bm]["m_r2"], "best_z_layer": bz, "best_m_layer": bm,
           "cos_dz_in_Mplane": cos_in_plane, "null_floor": null_floor,
           "m_r2_after_remove_zsub": m_after, "m_r2_full": decode[bm]["m_r2"],
           "z_r2_after_remove_Mplane": z_after, "z_r2_full": decode[bz]["z_r2"],
           "cross_z_predicts_maxcoord_r2": cross_r2,
           "steer_layer": SLAYER, "steer_alpha_rel": ALPHA / (typ + 1e-9), "n_rand": NRAND,
           # BOUNDED capability cross-talk: clipped-R² in [0,1] and next-symbol KL (>=0)
           "cap_clip_unsteered": cap_clip0, "cap_clip_align": cap_clip_align,
           "cap_clip_drop_align": cap_clip0 - cap_clip_align,
           "cap_clip_rand_mean": float(np.mean(rand_clip)), "cap_clip_rand_sd": float(np.std(rand_clip)),
           "cap_clip_drop_rand_mean": float(cap_clip0 - np.mean(rand_clip)),
           "nsk_align": nsk_align, "nsk_rand_mean": float(np.mean(rand_nsk)), "nsk_rand_sd": float(np.std(rand_nsk)),
           "footprint_align": float(footprint(d_align)), "footprint_rand_mean": float(np.mean(rand_fp)),
           "train_min": (time.time() - t0) / 60}
    print(f"  [lam={lam:.2f}] KL={kl:.4f} calib={calib:.3f} | corr(q,sharp)={corr_q_maxcoord:+.3f} "
          f"q_std={q_std:.3f} | zR²={res['z_r2']:.3f} mR²={res['m_r2']:.3f} | "
          f"|proj_dz|={cos_in_plane:.3f}(null {null_floor:.3f}) m|noZ={m_after:.3f} z|noM={z_after:.3f}\n"
          f"           CAUSAL(L{SLAYER},rel{res['steer_alpha_rel']:.2f}): capClipDrop align={res['cap_clip_drop_align']:+.3f} "
          f"rand={res['cap_clip_drop_rand_mean']:+.3f}±{res['cap_clip_rand_sd']:.3f} | "
          f"nextsymKL align={nsk_align:.4f} rand={res['nsk_rand_mean']:.4f} | "
          f"footprint align={res['footprint_align']:.3f} rand={res['footprint_rand_mean']:.3f} "
          f"({res['train_min']:.1f}m)", flush=True)
    return res


def aggregate(per_lam):
    """per_lam: list of seed-rows for ONE lambda -> mean/sd dict for scalar keys."""
    keys = ["corr_q_maxcoord", "q_std", "z_r2", "m_r2", "cos_dz_in_Mplane", "null_floor",
            "m_r2_after_remove_zsub", "z_r2_after_remove_Mplane", "cross_z_predicts_maxcoord_r2",
            "cap_clip_drop_align", "cap_clip_drop_rand_mean", "nsk_align", "nsk_rand_mean",
            "footprint_align", "footprint_rand_mean", "best_z_layer", "best_m_layer", "filter_calib_err"]
    out = {"lambda": per_lam[0]["lambda"], "n_seed": len(per_lam)}
    for k in keys:
        v = np.array([r[k] for r in per_lam], float)
        out[k] = float(v.mean()); out[k + "_sd"] = float(v.std())
    return out


def main():
    nseed = int(os.environ.get("SW_NSEED", "3"))
    print(f"=== GHMM independence sweep: lambdas={LAMBDAS} x {nseed} seeds "
          f"(L={L},steps={STEPS},NEV={NEV},nrand={os.environ.get('SW_NRAND','8')}) ===", flush=True)
    all_rows = []; agg = []
    for lam in LAMBDAS:
        per = [run_lambda(lam, seed=s) for s in range(nseed)]
        all_rows.extend(per); agg.append(aggregate(per))
    torch.save({"rows": all_rows, "agg": agg, "config": {"L": L, "STEPS": STEPS, "NEV": NEV,
                "nseed": nseed, "eps": EPS, "gamma": GAMMA, "stay0": STAY0, "a0": A0, "da": DA,
                "lambdas": LAMBDAS, "slayer": int(os.environ.get("SW_SLAYER", "1"))}}, f"{OUT}/ghmm_sweep.pt")
    print(f"saved {OUT}/ghmm_sweep.pt", flush=True)
    print("\n=== aggregate (mean±sd over seeds) ===")
    print("lam  corr(q,sharp)  zR²        mR²        z|noM      m|noZ      capClipDrop_align  capClipDrop_rand  nsk_align  nsk_rand  fp_align/rand  bestZlyr")
    for a in agg:
        print(f" {a['lambda']:.2f}  {a['corr_q_maxcoord']:+.3f}±{a['corr_q_maxcoord_sd']:.2f}  "
              f"{a['z_r2']:.3f}±{a['z_r2_sd']:.3f}  {a['m_r2']:.3f}±{a['m_r2_sd']:.3f}  "
              f"{a['z_r2_after_remove_Mplane']:.3f}±{a['z_r2_after_remove_Mplane_sd']:.2f}  "
              f"{a['m_r2_after_remove_zsub']:.3f}±{a['m_r2_after_remove_zsub_sd']:.2f}  "
              f"{a['cap_clip_drop_align']:+.3f}±{a['cap_clip_drop_align_sd']:.3f}      "
              f"{a['cap_clip_drop_rand_mean']:+.3f}±{a['cap_clip_drop_rand_mean_sd']:.3f}     "
              f"{a['nsk_align']:.4f}    {a['nsk_rand_mean']:.4f}   "
              f"{a['footprint_align']:.2f}/{a['footprint_rand_mean']:.2f}    {a['best_z_layer']:.1f}")


if __name__ == "__main__":
    main()
