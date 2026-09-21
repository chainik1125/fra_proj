"""Summary figure for the Setting-A numerical verification."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

c1 = json.load(open("../out/check1_seednoise.json"))
c3 = json.load(open("../out/check3_lmmse.json"))
c4 = json.load(open("../out/check4_dilution.json"))
c5 = json.load(open("../out/check5_chi.json"))

fig, ax = plt.subplots(2, 2, figsize=(11, 8))

# (1) attention loss-flat but attributions nonzero + optimizer-pinned
a = ax[0, 0]
labels = ["floor\ntr(Cov)", "clean rep\n(W=V=0)", "trained\n(large W,V)"]
losses = [c1["floor_trCov"], c1["clean_rep_loss"], c1["trained_loss_mean"]]
bars = a.bar(labels, losses, color=["#888", "#4c72b0", "#c44e52"])
a.set_ylim(min(losses) * 0.995, max(losses) * 1.005)
a.set_ylabel("MSE loss")
a.set_title("Check 1: attention is loss-flat\n(all three = floor)  yet attributions differ")
for b, l in zip(bars, losses):
    a.text(b.get_x() + b.get_width() / 2, l, f"{l:.4f}", ha="center", va="bottom", fontsize=8)
a.text(0.5, 0.25,
       f"FRA attribution |Q|: clean={c1['clean_rep_attribution_norm']['Q']:.2f}, "
       f"trained={c1['trained_attribution_norm']['Q']:.2f}\n"
       f"cross-seed Q-corr = {c1['seed_coherence_measured']['Q_meancorr']:.3f} "
       f"(pred 'seed noise ~0' FALSIFIED)\n"
       f"=> loss-flat but OPTIMIZER-PINNED, not seed noise",
       transform=a.transAxes, fontsize=8, ha="center",
       bbox=dict(boxstyle="round", fc="#fff4e6"))

# (3) LMMSE transfer diagonal vs d/N
a = ax[0, 1]
keys = ["complete_N10_d40", "square_N24_d24", "overcomplete_N48_d24", "overcomplete_N80_d24"]
dN = [c3[k]["pred_T_diag_mean_d_over_N"] for k in keys]
Td = [c3[k]["T_diag_mean"] for k in keys]
a.plot([0, 1.05], [0, 1.05], "k--", lw=1, label="T_ii = min(1, d/N)")
a.scatter(dN, Td, s=80, c="#4c72b0", zorder=3)
for k, x, y in zip(keys, dN, Td):
    a.annotate(k.replace("_", "\n"), (x, y), fontsize=7, xytext=(5, -12),
               textcoords="offset points")
a.set_xlabel("predicted min(1, d/N)"); a.set_ylabel("measured mean T_ii (shrinkage)")
a.set_title("Check 3: LMMSE readout\nexact recovery (N<=d) -> shrinkage d/N (N>d)")
a.legend(fontsize=8)

# (4) dilution scalings
a = ax[1, 0]
ms = [r["m"] for r in c4["rows"]]
ov = [r["per_latent_OV_norm"] for r in c4["rows"]]
qk = [abs(r["per_pair_QK"]) for r in c4["rows"]]
a.loglog(ms, ov, "o-", label="per-latent OV (meas)", color="#4c72b0")
a.loglog(ms, [ov[0] / m for m in ms], "k:", label="1/m")
a.loglog(ms, qk, "s-", label="per-pair QK (meas)", color="#c44e52")
a.loglog(ms, [qk[0] / m ** 2 for m in ms], "k--", label="1/m^2")
a.set_xlabel("split multiplicity m"); a.set_ylabel("per-latent / per-pair magnitude")
a.set_title(f"Check 4: dilution (group-cut invariant to "
            f"{c4['verdict']['group_cuts_invariant']:.0e})")
a.legend(fontsize=8)

# (5) MCC vs severability
a = ax[1, 1]
codes = ["complete\nL=3", "absence\nL=2"]
mccs = [c5["complete_L3"]["MCC"], c5["absence_L2"]["MCC"]]
sev = [len(c5["complete_L3"]["severable_features"]), len(c5["absence_L2"]["severable_features"])]
x = np.arange(2)
a.bar(x - 0.2, mccs, 0.4, label="MCC", color="#4c72b0")
a.bar(x + 0.2, [s / 3 for s in sev], 0.4, label="frac features severable", color="#c44e52")
a.set_xticks(x); a.set_xticklabels(codes)
a.set_ylim(0, 1.15); a.set_ylabel("value")
a.set_title("Check 5: MCC does NOT certify severability\n"
            "(both MCC~1; absence code severs 0/3)")
a.legend(fontsize=8)
for xi, (mc, s) in enumerate(zip(mccs, sev)):
    a.text(xi - 0.2, mc, f"{mc:.3f}", ha="center", va="bottom", fontsize=8)
    a.text(xi + 0.2, s / 3, f"{s}/3", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("../out/verify_A_summary.png", dpi=130)
print("wrote ../out/verify_A_summary.png")
