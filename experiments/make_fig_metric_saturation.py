"""Figure for the C2 metric-saturation analysis (oracle-only). Standalone to avoid touching the
shared make_figures_active.py while a concurrent session edits it.

Panel (a): as the redundant code deepens (n), the ERROR-RATE headroom of the optimal decoder over
the trivial always-aligned predictor (B(n) - bayes_err) collapses super-linearly, while the
DISTRIBUTIONAL headroom (const_kl: KL of a constant-marginal predictor to the true posterior)
decays only mildly -- so error rate stops discriminating a learned decoder, but KL does not.
Panel (b): the trained-transformer datapoints we have (n=3/5/7/9 model logical error == Bayes,
overlapping) sit right on the collapsing error-rate floor -- which is exactly why we certify the
model with KL/R^2, not error rate.

Run:  BAG_OUT=$PWD/results BAG_FIG=$PWD/figures PYTHONPATH=$PWD python experiments/make_fig_metric_saturation.py
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.environ.get("BAG_OUT", "results")
FIG = os.environ.get("BAG_FIG", "figures")
os.makedirs(FIG, exist_ok=True)

sat = torch.load(f"{RES}/c2_metric_saturation.pt", map_location="cpu", weights_only=False)
rows = sat["rows"]
ns = [r["n"] for r in rows]
err_room = [max(r["err_headroom"], 1e-5) for r in rows]
const_kl = [r["const_kl"] for r in rows]
B = [r["B_emp"] for r in rows]

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

ax[0].semilogy(ns, err_room, "o-", color="C3",
               label="error-rate headroom (q*=0.25)")
ax[0].semilogy(ns, const_kl, "s-", color="C0",
               label="distributional/KL headroom (q*=0.25)")
# near-threshold control q*=0.40 (misalignment common): error-rate headroom barely declines
sat40 = torch.load(f"{RES}/c2_metric_saturation_q040.pt", map_location="cpu", weights_only=False) \
    if os.path.exists(f"{RES}/c2_metric_saturation_q040.pt") else None
if sat40:
    r40 = sat40["rows"]
    ax[0].semilogy([r["n"] for r in r40], [max(r["err_headroom"], 1e-5) for r in r40], "o--",
                   color="C3", alpha=0.5, label="error-rate headroom (q*=0.40, near thresh)")
    ax[0].semilogy([r["n"] for r in r40], [r["const_kl"] for r in r40], "s--",
                   color="C0", alpha=0.5, label="KL headroom (q*=0.40)")
ax[0].set_xlabel("redundant blocks n (deeper code →)")
ax[0].set_ylabel("headroom of the optimal decoder (log)")
ax[0].set_title("(a) Error-rate signal collapses only when the code\nmakes misalignment rare (q*≪½); KL persists")
ax[0].set_xticks(ns); ax[0].legend(fontsize=7, loc="lower left")

# overlay the trained-transformer points we have (model logical error == Bayes), on the err floor
model_err = {3: 0.144, 5: 0.097, 7: 0.072, 9: 0.050}
ax[1].plot(ns, B, "^:", color="grey", label="trivial always-aligned err = B(n)")
ax[1].plot(ns, [r["bayes_err"] for r in rows], "s--", color="C2", label="Bayes (optimal) err")
mk = sorted(model_err)
ax[1].plot(mk, [model_err[k] for k in mk], "o-", color="C0",
           label="trained transformer logical err")
ax[1].set_xlabel("redundant blocks n")
ax[1].set_ylabel("misalignment decode error")
ax[1].set_title("(b) Model err == Bayes == trivial floor for deep codes\n→ certify with KL/R², not error rate")
ax[1].set_xticks(ns); ax[1].legend(fontsize=8)

fig.suptitle("C2 metric-saturation: which metric certifies a learned error-correcting decoder?",
             fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_c2_metric_saturation.png"); plt.close(fig)
print("wrote fig_c2_metric_saturation.png")
