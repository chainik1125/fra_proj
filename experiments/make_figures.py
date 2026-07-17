"""Generate figures for the writeup from results/*.pt.  Robust to missing files."""

import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("BAG_ROOT", "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-sprint")
RES = os.environ.get("BAG_OUT", f"{ROOT}/results")
FIG = os.environ.get("BAG_FIG", f"{ROOT}/figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    p = f"{RES}/{name}"
    return torch.load(p, map_location="cpu", weights_only=False) if os.path.exists(p) else None


def mean_std(xs):
    a = np.array(xs)
    return a.mean(), a.std()


# ---------------------------------------------------------------- chi2 scaling
def fig_chi2_scaling(r):
    pri = r["priors"]
    chi2s, hats, errs = [], [], []
    for pname, d in pri.items():
        finals = [run["final"]["chi2_hat"] for run in d["runs"]]
        m, s = mean_std(finals)
        chi2s.append(d["chi2"]); hats.append(m); errs.append(s)
    chi2s = np.array(chi2s); hats = np.array(hats); errs = np.array(errs)
    order = np.argsort(chi2s)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="identity ($\\hat\\chi_2=\\chi_2$)")
    ax.errorbar(chi2s[order], hats[order], yerr=errs[order], fmt="o-", ms=8,
                color="C0", capsize=3, label="trained transformer")
    if r.get("control"):
        ax.scatter([0.5], [r["control"]["final"]["chi2_hat"]], marker="X", s=120,
                   color="C3", zorder=5, label="independent-bag control")
    ax.set_xlabel("$\\chi_2 = \\mathbb{E}[1/N]$  (true collision susceptibility)")
    ax.set_ylabel("$\\hat\\chi_2$  (model's effective transfer coefficient)")
    ax.set_title("Models learn cross-rollout transfer\nscaled exactly by the collision susceptibility")
    ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.08, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_chi2_scaling.png"); plt.close(fig)
    print("wrote fig_chi2_scaling.png")


# ---------------------------------------------------------------- development
def fig_development(r):
    snaps = r["snap_steps"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for pname in ["fixed1", "fixed2", "fixed5", "fixed10"]:
        if pname not in r["priors"]:
            continue
        d = r["priors"][pname]
        chi2 = d["chi2"]
        # average chi2_hat trajectory across seeds
        traj = []
        for s in snaps:
            vals = [run["dev"][s]["chi2_hat"] for run in d["runs"] if s in run["dev"]]
            traj.append(np.mean(vals) if vals else np.nan)
        klr2 = []
        for s in snaps:
            vals = [run["dev"][s]["kl_r2"] for run in d["runs"] if s in run["dev"]]
            klr2.append(np.mean(vals) if vals else np.nan)
        axes[0].plot(snaps, traj, "o-", ms=3, label=f"{pname} ($\\chi_2$={chi2:.2f})")
        axes[0].axhline(chi2, ls=":", color=axes[0].lines[-1].get_color(), lw=1)
        axes[1].plot(snaps, klr2, "o-", ms=3, label=pname)
    axes[0].set_xscale("log"); axes[0].set_xlabel("training step")
    axes[0].set_ylabel("$\\hat\\chi_2$ (effective transfer)")
    axes[0].set_title("Collision coefficient emerges over training\n(dotted = true $\\chi_2$)")
    axes[0].legend(fontsize=8)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("training step"); axes[1].set_ylabel("KL to collision oracle (rollout-2)")
    axes[1].set_title("Convergence to the M$_2$ oracle")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_development.png"); plt.close(fig)
    print("wrote fig_development.png")


# ---------------------------------------------------------------- mechanism
def fig_mechanism(m, c):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    # panel A: decodability of q vs the count-feature baseline (the real control)
    qd = m.get("q_decode", {})
    if qd:
        ps = sorted(qd.keys())
        rel = [p - ps[0] + 1 for p in ps]  # bits into rollout 2
        axes[0].plot(rel, [qd[p]["best_r2"] for p in ps], "o-", color="C0",
                     label="model residual probe")
        if c and "P1_q_vs_counts" in c:
            p1 = c["P1_q_vs_counts"]; cps = sorted(p1.keys())
            crel = [p - cps[0] + 1 for p in cps]
            axes[0].plot(crel, [p1[p]["count_baseline_r2"] for p in cps], "s--", color="C1",
                         label="affine-count baseline")
        axes[0].axhline(m.get("q_shuffled_r2", 0), color="C7", ls=":",
                        label="shuffled control")
        axes[0].set_xlabel("bits observed into rollout 2")
        axes[0].set_ylabel("test $R^2$ for same-source posterior $q$")
        axes[0].set_title("$q$ is decodable far above a\nlinear readout of the counts")
        axes[0].set_ylim(0, 1.02); axes[0].legend(fontsize=9, loc="lower left")

    # panel B: causal interventions on the transfer coefficient chi2_hat
    if c and "P2_knockouts" in c and "P3_maximal_ablation" in c:
        kk = c["P2_knockouts"]; p3 = c["P3_maximal_ablation"]
        names = ["clean", "cross-attn\nknockout", "within-r2\nsham", "s1-dir ablate\n(all pos/layers)"]
        vals = [kk["clean"][2], kk["cross"][2], kk["sham_within_r2"][2],
                min(p3["by_layer_allpos"].values())]
        colors = ["C0", "C3", "C2", "C1"]
        axes[1].bar(names, vals, color=colors)
        axes[1].axhline(0.5, color="k", ls="--", lw=1, label="true $\\chi_2$=0.5")
        axes[1].axhline(0, color="C3", ls=":", lw=1)
        axes[1].set_ylabel("effective transfer $\\hat\\chi_2$")
        axes[1].set_title("Transfer needs cross-rollout attention,\nbut no single residual direction")
        axes[1].legend(fontsize=9); axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_mechanism.png"); plt.close(fig)
    print("wrote fig_mechanism.png")


# ---------------------------------------------------------------- annealed
def fig_annealed(a):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # loss curves across priors (should coincide)
    for pname, d in a["priors"].items():
        steps = [s for s, _ in d["losses"]]; loss = [l for _, l in d["losses"]]
        axes[0].plot(steps, loss, label=f"{pname} (KL={d['final']['kl_laplace']:.4f})")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("train loss (nats)")
    axes[0].set_title("Loss is invariant to the $N$-prior\n(annealed barycenter collapse)")
    axes[0].legend(fontsize=8)
    # probe bars: N vs count s
    if "probes" in a:
        layers = sorted(a["probes"].keys())
        nacc = [a["probes"][l]["N_acc"] for l in layers]
        nbase = a["probes"][layers[0]]["N_baseline"]
        sr2 = [a["probes"][l]["s_r2"] for l in layers]
        x = np.arange(len(layers)); w = 0.35
        axes[1].bar(x - w/2, nacc, w, label="decode $N$ (acc)", color="C3")
        axes[1].bar(x + w/2, sr2, w, label="decode count $s$ ($R^2$)", color="C0")
        axes[1].axhline(nbase, color="C3", ls="--", lw=1, label="$N$ majority baseline")
        axes[1].set_xticks(x); axes[1].set_xticklabels([f"L{l}" for l in layers])
        axes[1].set_ylim(0, 1); axes[1].set_title("Model represents the count $s$, not $N$")
        axes[1].legend(fontsize=8)
    # order invariance
    if "order_invariance" in a:
        oi = a["order_invariance"]
        axes[2].bar(["mean |Δp1|", "p95 |Δp1|"],
                    [oi["mean_abs_delta_p1"], oi["p95_abs_delta_p1"]], color="C0")
        axes[2].axhline(0, color="k", lw=0.8)
        axes[2].set_title(f"Near order-invariance\n(count-preserving shuffle; mean p1={oi['mean_p1']:.2f})")
        axes[2].set_ylabel("change in P(next=1)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_annealed.png"); plt.close(fig)
    print("wrote fig_annealed.png")


def fig_jrollout(j):
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    ks = sorted(j["acc_by_k"].keys()); acc = [j["acc_by_k"][k] for k in ks]
    ax.plot(ks, acc, "o-", color="C0", ms=7, label="probe accuracy for true $N$")
    ax.axhline(0.5, color="C3", ls="--", lw=1, label="chance (binary)")
    ax.set_xlabel("number of quenched rollouts observed")
    ax.set_ylabel("probe accuracy for collision regime $N\\in\\{2,10\\}$")
    ax.set_title("The bag size is invisible at 1 rollout (cf. C1),\nbut inferred in-context as rollouts accumulate")
    ax.set_ylim(0.45, 0.8); ax.legend(fontsize=9, loc="upper left")
    ax.annotate("1 rollout ≈ annealed:\nN at chance", xy=(1, acc[0]), xytext=(1.6, 0.56),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="C7"))
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_jrollout.png"); plt.close(fig)
    print("wrote fig_jrollout.png")


def main():
    j = load("jrollout.pt")
    if j:
        fig_jrollout(j)
    r = load("quenched_sweep.pt")
    if r:
        fig_chi2_scaling(r); fig_development(r)
    m = load("mechanism.pt")
    if m:
        fig_mechanism(m, load("controls.pt"))
    a = load("annealed.pt")
    if a:
        fig_annealed(a)
    print("done")


if __name__ == "__main__":
    main()
