"""Figures for the active-bag / error-correction sprint. Robust to missing files."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("BAG_ROOT", "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-sprint-2")
RES = os.environ.get("BAG_OUT", f"{ROOT}/results")
FIG = os.environ.get("BAG_FIG", f"{ROOT}/figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def load(n):
    p = f"{RES}/{n}"
    return torch.load(p, map_location="cpu", weights_only=False) if os.path.exists(p) else None


def fig_c1():
    c1 = load("c1.pt"); steer = load("c1_steer.pt"); phase = load("c1_phase.pt")
    if not c1:
        return
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    # (a) z decode per layer vs count baseline
    layers = sorted(c1["c1b_decode"].keys())
    ax[0,0].bar([l-0.15 for l in layers], [c1["c1b_decode"][l]["r2_z"] for l in layers], 0.3,
                label="resid → z=logit q", color="C0")
    ax[0,0].bar([l+0.15 for l in layers], [c1["c1b_decode"][l]["r2_count_baseline"] for l in layers], 0.3,
                label="count → z (baseline)", color="C1")
    ax[0,0].set_xticks(layers); ax[0,0].set_xlabel("layer"); ax[0,0].set_ylabel("R²")
    ax[0,0].set_ylim(0,1.02); ax[0,0].set_title("(a) Alignment log-odds z is decodable;\nrunning count is not sufficient")
    ax[0,0].legend(fontsize=8)
    # (b) drift
    d = c1["c1e_drift"]
    for label, color in [("corrupting_1s","C3"),("corrective_0s","C2")]:
        oq = d[label]["oracle_q"]; mq = d[label]["model_q"]
        x = range(len(oq))
        ax[0,1].plot(x, oq, "--", color=color, label=f"{label} oracle")
        ax[0,1].plot(x, mq, "-", color=color, label=f"{label} model")
    ax[0,1].set_xlabel("steps into forced run"); ax[0,1].set_ylabel("misalignment q")
    ax[0,1].set_title("(b) Drift: corrupting context raises q,\ncorrective lowers it (model tracks filter)")
    ax[0,1].legend(fontsize=7)
    # (c) steering (best layer = layer with biggest swing)
    if steer:
        al = steer["alphas"]
        best = max(steer["layers"], key=lambda l: max(steer["layers"][l]["z"])-min(steer["layers"][l]["z"]))
        ax[1,0].plot(al, steer["layers"][best]["z"], "o-", color="C0", label=f"steer z-dir (L{best})")
        ax[1,0].plot(al, steer["layers"][best]["rand"], "s--", color="C7", label="random dir")
        ax[1,0].axhline(c1["qstar"], color="k", ls=":", lw=1, label="q* (unsteered)")
        ax[1,0].set_xlabel("steering coefficient α"); ax[1,0].set_ylabel("model mean implied q")
        ax[1,0].set_title("(c) Steering the alignment direction\nshifts predicted misalignment")
        ax[1,0].legend(fontsize=8)
    # (d) phase diagram
    if phase:
        qs = [r["qstar"] for r in phase["runs"]]; mq = [r["implied_q"] for r in phase["runs"]]
        ax[1,1].plot([0,1],[0,1],"k--",lw=1,label="identity")
        ax[1,1].scatter(qs, mq, s=60, color="C0", zorder=5, label="trained models")
        ax[1,1].set_xlabel("q* = ε/(ε+γ)"); ax[1,1].set_ylabel("model mean implied q")
        ax[1,1].set_title("(d) (ε,γ) phase diagram:\nmodel learns the steady-state misalignment")
        ax[1,1].legend(fontsize=8)
    fig.suptitle("C1: the alignment log-odds coordinate", fontsize=13, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_c1.png"); plt.close(fig); print("wrote fig_c1.png")


def fig_c2_process():
    th = load("c2_threshold.pt")
    if not th:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    s = th["suppression"]; p = s["p"]
    for n in sorted(s["by_n"], key=int):
        ax[0].plot(p, s["by_n"][n], "o-", ms=3, label=f"n={n}")
    ax[0].plot([0,0.6],[0,0.6],"k:",lw=1)
    ax[0].axvline(0.5, color="grey", ls="--", lw=0.8)
    ax[0].set_xlabel("physical misalignment p"); ax[0].set_ylabel("logical misalignment P(Bin(n,p)>r)")
    ax[0].set_title("(a) Redundancy suppresses misalignment\n(majority code, binomial tail)")
    ax[0].legend(fontsize=8)
    t = th["threshold"]
    ax[1].plot(t["R_M"], t["endemic_sim"], "o-", color="C0", label="simulation")
    ax[1].plot(t["R_M"], t["endemic_theory"], "--", color="C3", label="theory 1−1/R_M")
    ax[1].axvline(1.0, color="k", ls=":", lw=1, label="threshold R_M=1")
    ax[1].set_xlabel("$R_M = \\beta/\\gamma$ (spread / correction)"); ax[1].set_ylabel("stationary misalignment I*")
    ax[1].set_title("(b) Spread threshold: misalignment endemic\nonly above R_M=1 (SIS bifurcation)")
    ax[1].legend(fontsize=8)
    fig.suptitle("C2 (process): alignment as an error-correcting code", fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_c2_process.png"); plt.close(fig); print("wrote fig_c2_process.png")


def fig_c2_transformer():
    cl = load("c2_logical.pt"); ft = load("c2_faulttol.pt")
    if not cl:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    by_n = dict(cl["by_n"])
    # deeper-code scaling point (exp_c2_n9.py): 3-seed mean (same regime, learned just as cleanly)
    n9_seeds = [load(f) for f in ("c2_n9.pt", "c2_n9_s1.pt", "c2_n9_s2.pt")]
    n9_seeds = [s for s in n9_seeds if s]
    if n9_seeds:
        import numpy as _np
        nn = int(n9_seeds[0]["n"])
        by_n[nn] = {k: float(_np.mean([s[k] for s in n9_seeds]))
                    for k in ("model_logical_err", "bayes_logical_err", "phys_bayes_err")}
    ns = sorted(by_n, key=int)
    ax[0].plot(ns, [by_n[n]["model_logical_err"] for n in ns], "o-", color="C0", label="model logical error")
    ax[0].plot(ns, [by_n[n]["bayes_logical_err"] for n in ns], "s--", color="C2", label="Bayes logical error")
    ax[0].plot(ns, [by_n[n]["phys_bayes_err"] for n in ns], "^:", color="C3", label="single-block (physical) error")
    ax[0].set_xlabel("number of redundant blocks n"); ax[0].set_ylabel("misalignment decode error")
    ax[0].set_title("(a) The transformer's LOGICAL readout is more\nreliable than any block, and improves with n")
    ax[0].set_xticks(ns); ax[0].legend(fontsize=8)
    # (b) fault tolerance + logical decodability
    if ft:
        ax[1].plot(ft["k"], ft["model_p_yes"], "o-", color="C0", label="model P(misaligned)")
        ax[1].plot(ft["k"], ft["oracle_p_yes"], "s--", color="C3", label="Bayes oracle")
        ax[1].axvline(ft["r"]+0.5, color="k", ls=":", lw=1, label=f"code threshold k=r+1={ft['r']+1}")
        ax[1].axhline(0.5, color="grey", ls="--", lw=0.6)
        ax[1].set_xlabel("# corrupted blocks k (of n=%d)" % ft["n"]); ax[1].set_ylabel("logical readout P(misaligned)")
        ax[1].set_title("(b) Fault tolerance: readout flips near the\nmajority threshold, tracking the Bayes decoder")
        ax[1].legend(fontsize=8)
    fig.suptitle("C2 (transformer): the model learns the majority decoder", fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_c2_transformer.png"); plt.close(fig); print("wrote fig_c2_transformer.png")


def fig_real_em():
    from math import comb
    r7 = load("real_em.pt"); r14 = load("real_em_14b.pt")
    if not r7 and not r14:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    # (a) MEASURED per-prompt misalignment rate p for each organism, vs p=0.5
    y = 0; ylabels = []
    for r, color, name in [(r7, "C0", "Qwen-7B EM"), (r14, "C3", "Qwen-14B EM")]:
        if not r:
            continue
        ps = sorted([d["p"] for d in r["by_q"].values()])
        ax[0].scatter(ps, [y]*len(ps), color=color, s=55, zorder=3)
        ax[0].scatter([r["overall_p"]], [y], marker="|", s=600, color=color, zorder=4)
        ylabels.append((y, f"{name}\n(p̄={r['overall_p']:.2f})")); y += 1
    ax[0].axvline(0.5, color="k", ls="--", lw=1.2)
    ax[0].text(0.5, y-0.4, "code threshold p=½", rotation=90, va="top", ha="right", fontsize=8)
    ax[0].set_yticks([t[0] for t in ylabels]); ax[0].set_yticklabels([t[1] for t in ylabels], fontsize=8)
    ax[0].set_xlim(0, 1); ax[0].set_ylim(-0.5, y-0.3)
    ax[0].set_xlabel("MEASURED per-prompt misalignment rate p (each dot = one EM question)")
    ax[0].set_title("(a) Real EM organisms straddle p=½\n(weak/narrow 7B below; strong/broad 14B above)")
    # (b) the ALGEBRAIC suppression rule (majority code), with measured p̄ marked
    pp = np.linspace(0, 1, 101)
    for n, c in [(3, "C1"), (5, "C2"), (7, "C4")]:
        rr = n // 2
        ax[1].plot(pp, [sum(comb(n, j)*x**j*(1-x)**(n-j) for j in range(rr+1, n+1)) for x in pp],
                   color=c, lw=1.8, label=f"majority of n={n}")
    ax[1].plot([0, 1], [0, 1], "k:", lw=1, label="no ensembling (identity)")
    ax[1].axvline(0.5, color="grey", ls="--", lw=0.8)
    if r7:
        ax[1].axvline(r7["overall_p"], color="C0", ls="-", lw=1.5, alpha=0.6)
        ax[1].text(r7["overall_p"], 0.9, " 7B p̄", color="C0", fontsize=8)
    if r14:
        ax[1].axvline(r14["overall_p"], color="C3", ls="-", lw=1.5, alpha=0.6)
        ax[1].text(r14["overall_p"], 0.9, " 14B p̄", color="C3", fontsize=8)
    ax[1].set_xlabel("per-sample misalignment p"); ax[1].set_ylabel("logical (post-majority-vote) misalignment")
    ax[1].set_title("(b) The algebra: majority-vote suppresses iff p<½\n(curves), so where an organism sits decides if it helps")
    ax[1].legend(fontsize=8)
    auc = (f"7B AUROC={r7['auroc_coordinate']:.2f}" if r7 else "") + (f", 14B={r14['auroc_coordinate']:.2f}" if r14 else "")
    fig.suptitle(f"Real-LLM bridge: do real EM organisms sit below the p=½ ensembling threshold?  (coordinate {auc}, in-sample)", fontsize=10.5, y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_real_em.png"); plt.close(fig); print("wrote fig_real_em.png")


def fig_xadapter():
    x = load("real_em_xadapter.pt")
    if not x:
        return
    pm = np.array(x["p_mat"]); ads = x["adapters"]; qs = x["questions"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [2.2, 1]})
    im = ax[0].imshow(pm, aspect="auto", cmap="Reds", vmin=0, vmax=0.5)
    ax[0].set_yticks(range(len(ads))); ax[0].set_yticklabels([f"{a}\nfinetune" for a in ads], fontsize=8)
    ax[0].set_xticks(range(len(qs))); ax[0].set_xticklabels([q.replace("_", " ")[:14] for q in qs], rotation=40, ha="right", fontsize=7)
    for i in range(pm.shape[0]):
        for j in range(pm.shape[1]):
            ax[0].text(j, i, f"{pm[i,j]:.2f}", ha="center", va="center", fontsize=7,
                       color="white" if pm[i, j] > 0.3 else "black")
    fig.colorbar(im, ax=ax[0], fraction=0.025, label="misalignment rate p")
    ax[0].set_title("(a) Three EM finetunes (different narrow domains) are misaligned\non the SAME prompts → correlated errors")
    # (b) correlation bars
    cc = x["cross_corr"]; keys = list(cc)
    ax[1].bar(range(len(keys)), [cc[k] for k in keys], color="C3")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(range(len(keys))); ax[1].set_xticklabels([k.replace("-", "\nvs\n") for k in keys], fontsize=7)
    ax[1].set_ylabel("cross-finetune corr of per-prompt p"); ax[1].set_ylim(-0.2, 1)
    ax[1].set_title(f"(b) Correlated (mean ρ={x['mean_cross_corr']:.2f})\n⇒ ensembling finetunes can't error-correct")
    fig.suptitle("Real EM: errors are CORRELATED across finetunes (redundancy across models fails)", fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_xadapter.png"); plt.close(fig); print("wrote fig_xadapter.png")


def fig_c2_concatenation():
    """Level-L concatenated 3-block majority code: recursive threshold."""
    cc = load("c2_concatenation.pt")
    if not cc:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) Trajectories: p_l vs level l for selected p_0 values
    cmap_below = plt.cm.Blues
    cmap_above = plt.cm.Reds
    p0_sel = cc["traj_selected"]
    for p0, traj in sorted(p0_sel.items()):
        ls = traj
        lvls = list(range(len(ls)))
        if p0 < 0.5:
            c = cmap_below(0.3 + 0.6 * p0)
            ax[0].semilogy(lvls, np.array(ls).clip(1e-12), "o-", color=c, ms=4,
                           label=f"p₀={p0:.2f}" if p0 in [0.1, 0.3, 0.45] else None)
        elif p0 > 0.5:
            c = cmap_above(0.3 + 0.6 * (p0 - 0.5) * 2)
            ax[0].semilogy(lvls, np.maximum(1 - np.array(ls), 1e-12), "s--", color=c, ms=4,
                           label=f"p₀={p0:.2f} (1-p)" if p0 in [0.55, 0.7, 0.9] else None)
        else:
            ax[0].semilogy(lvls, [0.5] * len(lvls), "k:", ms=4, lw=1.5, label="p₀=0.5 (fixed)")
    # Mark real-EM organism trajectories from §5 (p=0.15 for 7B, p=0.48 for 14B)
    def majority_map_np(p):
        return 3*p**2 - 2*p**3
    for p0_em, label_em, col_em in [(0.15, "7B p̄=0.15", "C0"), (0.48, "14B p̄=0.48", "C3")]:
        traj_em = [p0_em]
        for _ in range(6):
            traj_em.append(majority_map_np(traj_em[-1]))
        ax[0].semilogy(range(len(traj_em)), np.maximum(traj_em, 1e-12), "D-",
                       color=col_em, ms=6, lw=2, label=label_em, zorder=10)
    ax[0].set_xlabel("concatenation level l"); ax[0].set_ylabel("error (log scale)")
    ax[0].set_title("(a) Doubly-exponential suppression below threshold\n(7B & 14B real-EM organisms marked)")
    ax[0].legend(fontsize=7, ncol=2)

    # (b) Final level p_L vs p_0 for different L
    p0g = np.array(cc["p0_grid"])
    for l in [0, 1, 2, 3, 5]:
        fl = cc["final_level"][l]
        ax[1].plot(p0g, fl, lw=1.8, label=f"l={l}")
    ax[1].plot([0, 1], [0, 1], "k:", lw=1, label="identity")
    ax[1].axvline(0.5, color="grey", ls="--", lw=0.8)
    ax[1].set_xlabel("physical rate p₀"); ax[1].set_ylabel("logical error after l levels")
    ax[1].set_title("(b) Concatenation sharpens the threshold:\nmore levels → steeper p₁/₂ transition")
    ax[1].legend(fontsize=8)

    # (c) Log-error doubling: |log p_l| vs l for p_0 below threshold
    p_traj = np.array(cc["p_traj_at_02"])
    log_errs = -np.log10(np.maximum(p_traj, 1e-14))
    ax[2].plot(range(len(log_errs)), log_errs, "o-", color="C0", ms=6)
    # Show doubling prediction: log(p_l) ≈ 2^l * log(3 p_0)
    p0_ = 0.2
    pred = [-np.log10(max((1 / 3) * (3 * p0_) ** (2 ** l), 1e-14)) for l in range(len(log_errs))]
    ax[2].plot(range(len(log_errs)), pred, "s--", color="C3", ms=5, label="theory: $2^l \\log(3p_0)$")
    ax[2].set_xlabel("concatenation level l"); ax[2].set_ylabel("$-\\log_{10}(p_l)$  (log-error)")
    ax[2].set_title(f"(c) p₀=0.2: log-error grows as $2^l$\n(super-exponential; doublings per level)")
    ax[2].legend(fontsize=8)

    fig.suptitle("Level-L concatenated 3-block majority code: recursive error suppression below p=½",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_c2_concatenation.png"); plt.close(fig)
    print("wrote fig_c2_concatenation.png")


def fig_ghmm():
    """GHMM factored representation: alignment z AND Mess3 belief, + factorisation tests."""
    g = load("ghmm_factored.pt")
    ms = load("ghmm_multiseed.pt")
    if not g:
        return
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # (a) per-layer decodability of z and Mess3 belief, vs controls
    layers = sorted(g["decode"].keys())
    zr = [g["decode"][l]["z_r2"] for l in layers]
    mr = [g["decode"][l]["m_r2"] for l in layers]
    ax[0,0].bar([l-0.18 for l in layers], zr, 0.36, label="alignment z=logit q", color="C3")
    ax[0,0].bar([l+0.18 for l in layers], mr, 0.36, label="Mess3 belief (simplex)", color="C0")
    c = g["controls"]
    ax[0,0].axhline(c["z_lastsym"], color="C3", ls=":", lw=1.2, label=f"z last-symbol ctrl ({c['z_lastsym']:.2f})")
    ax[0,0].axhline(c["m_lastsym"], color="C0", ls=":", lw=1.2, label=f"Mess3 last-symbol ctrl ({c['m_lastsym']:.2f})")
    ax[0,0].set_xticks(layers); ax[0,0].set_xlabel("layer"); ax[0,0].set_ylabel("R²")
    ax[0,0].set_ylim(0, 1.02)
    ax[0,0].set_title("(a) Both latents linearly decodable\n(controls can't recover them)")
    ax[0,0].legend(fontsize=7)

    # (b) separability: R² survival after removing the OTHER latent's subspace; + null floor
    s = g["separability"]
    cats = ["Mess3 R²\n(full)", "Mess3 R²\n(− z-subsp)", "z R²\n(full)", "z R²\n(− M-plane)"]
    vals = [s["m_r2_full"], s["m_r2_after_remove_zsub"], s["z_r2_full"], s["z_r2_after_remove_Mplane"]]
    cols = ["C0", "C0", "C3", "C3"]
    alphas_b = [1.0, 0.5, 1.0, 0.5]
    bars_b = ax[0,1].bar(range(4), vals, color=cols)
    for bar_b, av in zip(bars_b, alphas_b):
        bar_b.set_alpha(av)
    ax[0,1].set_xticks(range(4)); ax[0,1].set_xticklabels(cats, fontsize=8)
    ax[0,1].set_ylabel("R²"); ax[0,1].set_ylim(0, 1.02)
    ax[0,1].set_title(f"(b) Near-separable subspaces (layer {s['layer']}):\n"
                      f"|proj(d_z on Mess3-plane)|={s['cos_dz_in_Mplane']:.2f} (null {s['null_floor']:.2f})")

    # (c) steering alignment: alignment readout (z) moves, capability preserved; a
    # capability-subspace step destroys capability, a random-⊥ step is as harmless as alignment.
    st = g["steer"]; al = st["alphas"]
    z_a = [st["align"][a].get("z_read", st["align"][a].get("drift_score")) for a in al]
    cap_a = [max(st["align"][a]["cap_r2"], -2) for a in al]
    cap_c = [max(st["ctrl"][a]["cap_r2"], -2) for a in al]  # capability-subspace control
    cap_r = [max(st["rand"][a]["cap_r2"], -2) for a in al] if "rand" in st else None
    axb = ax[1,0]; axb2 = axb.twinx()
    axb.plot(al, z_a, "o-", color="C3", label="alignment readout z (steer align)")
    axb2.plot(al, cap_a, "o-", color="C2", label="capability R² (steer align)")
    if cap_r is not None:
        axb2.plot(al, cap_r, "^:", color="C0", label="capability R² (random-⊥, clip)")
    axb2.plot(al, cap_c, "s--", color="C8", label="capability R² (cap-subspace, clip)")
    axb.set_xlabel("steering coefficient α (alignment direction)")
    axb.set_ylabel("decoded alignment z", color="C3")
    axb2.set_ylabel("capability belief R²", color="C2")
    axb2.set_ylim(-2.1, 1.05)
    axb.set_title("(c) Steering alignment moves z but preserves capability;\na capability-subspace step destroys it")
    axb.legend(fontsize=7, loc="upper left"); axb2.legend(fontsize=7, loc="lower right")

    # (d) cross-alignment transfer of the Mess3 belief probe across disjoint sequences
    tr = g["transfer"]
    names = ["mx\naligned→mis", "my\naligned→mis", "in-context\nMess3 R²"]
    tvals = [tr["aligned_seqs->misaligned_seqs_mx"], tr["aligned_seqs->misaligned_seqs_my"],
             tr["in_context_m_r2"]]
    ax[1,1].bar(range(3), tvals, color=["C0", "C0", "C7"])
    ax[1,1].set_xticks(range(3)); ax[1,1].set_xticklabels(names, fontsize=8)
    ax[1,1].set_ylabel("R²"); ax[1,1].set_ylim(0, 1.02)
    ax[1,1].set_title("(d) Capability probe transfers across alignment\n(disjoint-sequence aligned→misaligned)")

    corr = g.get("latent_corr", {}).get("corr_q_maxcoord", float("nan"))
    fig.suptitle(f"C3 (GHMM): near-factored alignment (z) ⊗ capability (Mess3 belief)  "
                 f"[data corr={corr:+.2f}]", fontsize=12, y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_ghmm.png"); plt.close(fig); print("wrote fig_ghmm.png")


def fig_ghmm_contrast():
    """C3 contrast: where generative coupling shows up. The drift regime (near-factored, corr~-0.14)
    vs an entangled regime (sharpness-modulated, corr~-0.64). HONEST finding: the *linear* subspaces
    stay near-separable in BOTH; the coupling surfaces as increased *causal* steering cross-talk."""
    d = load("ghmm_factored.pt"); e = load("ghmm_factored_entangled.pt")
    if not d or not e:
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    def corr(g): return g.get("latent_corr", {}).get("corr_q_maxcoord", float("nan"))
    def overlap(g): return g["separability"]["cos_dz_in_Mplane"]
    def floor(g): return g["separability"]["null_floor"]

    # (a) generative correlation of the two latents (the knob we set)
    ax[0].bar([0, 1], [abs(corr(d)), abs(corr(e))], color=["C2", "C3"])
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["drift\n(near-factored)", "entangled\n(sharpness)"])
    ax[0].set_ylabel("|data corr(z, belief sharpness)|")
    ax[0].set_title("(a) Generative latent coupling\n(the knob we set: ~5x)")
    for i, g in enumerate([d, e]):
        ax[0].text(i, abs(corr(g))+0.01, f"{corr(g):+.2f}", ha="center", fontsize=9)

    # (b) representational subspace overlap |proj(d_z on Mess3-plane)| vs null -- ROBUST: both stay near
    ov = [overlap(d), overlap(e)]; fl = 0.5*(floor(d)+floor(e))
    ax[1].bar([0, 1], ov, color=["C2", "C3"])
    ax[1].axhline(fl, color="k", ls=":", lw=1.2, label=f"random-direction null ({fl:.2f})")
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["drift", "entangled"])
    ax[1].set_ylabel("|proj(d_z onto Mess3-plane)|")
    ax[1].set_ylim(0, 0.5)
    ax[1].set_title("(b) LINEAR subspaces stay near-separable\nin BOTH (overlap barely moves)")
    ax[1].legend(fontsize=8)
    for i, v in enumerate(ov):
        ax[1].text(i, v+0.01, f"{v:.2f}", ha="center", fontsize=9)

    # (c) steering cross-talk: capability R² under alignment steering -- RISES with coupling
    def cap_curve(g):
        st = g["steer"]; al = st["alphas"]
        return al, [st["align"][a]["cap_r2"] for a in al]
    ald, capd = cap_curve(d); ale, cape = cap_curve(e)
    # drop from the unsteered (alpha=0) value to the worst steered value
    dropd = capd[ald.index(0)] - min(capd); drope = cape[ale.index(0)] - min(cape)
    ax[2].plot(ald, capd, "o-", color="C2", label=f"drift (Δcap={dropd:.2f})")
    ax[2].plot(ale, cape, "s-", color="C3", label=f"entangled (Δcap={drope:.2f})")
    ax[2].set_xlabel("steering coefficient α (alignment dir)")
    ax[2].set_ylabel("capability belief R² (preserved?)")
    ax[2].set_ylim(0.5, 1.02)
    ax[2].set_title("(c) CAUSAL steering cross-talk rises with\ncoupling: align-steer degrades cap ~5x more")
    ax[2].legend(fontsize=8)

    fig.suptitle("C3 contrast: generative coupling surfaces as CAUSAL steering cross-talk, "
                 "while the LINEAR codes stay near-separable in both regimes", fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_ghmm_contrast.png"); plt.close(fig)
    print("wrote fig_ghmm_contrast.png")


def fig_c2_spread():
    """C2 under CORRELATED (spread-coupled) errors: the transformer learns the JOINT
    correlation-aware decoder, not the naive independent (binomial-tail) one."""
    sp = load("c2_spread.pt")
    if not sp:
        return
    R = sp["results"]
    corr = [r["cross_chain_corr"] for r in R]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) the naive independent decoder is sub-optimal once errors correlate
    x = np.arange(len(R)); w = 0.35
    ax[0].bar(x - w/2, [r["bayes_logical_err"] for r in R], w, color="C2", label="joint (optimal) decoder")
    ax[0].bar(x + w/2, [r["indep_logical_err"] for r in R], w, color="C3", label="naive independent decoder")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"ρ={c:+.2f}\nβ={r['beta']:.2f}" for c, r in zip(corr, R)], fontsize=8)
    ax[0].set_ylabel("logical decode error"); ax[0].set_ylim(0, max(0.31, max(r["indep_logical_err"] for r in R)*1.15))
    ax[0].set_title("(a) Correlation breaks the binomial-tail picture:\nnaive independent decoder over-errs")
    ax[0].legend(fontsize=8)

    # (b) THE test: model's logical posterior is close to the JOINT oracle, far from the independent one
    ax[1].plot(corr, [r["kl_to_joint"] for r in R], "o-", color="C2", label="model → JOINT oracle KL")
    ax[1].plot(corr, [r["kl_to_indep"] for r in R], "s--", color="C3", label="model → independent-decoder KL")
    ax[1].set_xlabel("induced cross-chain correlation ρ"); ax[1].set_ylabel("KL (nats)")
    ax[1].set_title("(b) The transformer learns the CORRELATION-AWARE\ndecoder (low KL to joint, not to independent)")
    ax[1].legend(fontsize=8)

    # (c) model logical error tracks the joint-Bayes error and beats the naive decoder
    ax[2].plot(corr, [r["model_logical_err"] for r in R], "o-", color="C0", label="transformer")
    ax[2].plot(corr, [r["bayes_logical_err"] for r in R], "^--", color="C2", label="joint Bayes (optimal)")
    ax[2].plot(corr, [r["indep_logical_err"] for r in R], "s:", color="C3", label="naive independent")
    ax[2].plot(corr, [r["phys_bayes_err"] for r in R], "x-", color="C7", label="single-chain (physical)")
    ax[2].set_xlabel("induced cross-chain correlation ρ"); ax[2].set_ylabel("logical error")
    ax[2].set_title("(c) The model matches the optimal joint decoder\nacross the correlation range")
    ax[2].legend(fontsize=7)

    fig.suptitle("C2 under correlated errors: the transformer learns the optimal correlation-aware "
                 "majority decoder (bridges to §5: real EM errors are correlated)", fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_c2_spread.png"); plt.close(fig)
    print("wrote fig_c2_spread.png")


def main():
    fig_c1(); fig_c2_process(); fig_c2_transformer(); fig_real_em(); fig_xadapter()
    fig_c2_concatenation(); fig_ghmm(); fig_ghmm_contrast(); fig_c2_spread()
    print("done")


if __name__ == "__main__":
    main()
