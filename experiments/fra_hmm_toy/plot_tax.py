"""The entanglement-tax figure: cheap removal of USE vs taxed removal of PRESENCE.

Panel A: behavioral removal (RF) vs linear presence (adversarial probe R2).
Panel B: presence vs ABSOLUTE within-block CE, with the exact bounds:
  - Bayes full-info ceiling (0.7785) — unbeatable at any presence level
  - zero-info floor (0.845) — unbeatable at zero presence (the 20.8% tax)
  - time-sharing chord between the two exact endpoints (achievable, so the
    region above it is feasible; we only shade what is PROVABLY impossible)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent / "out"
main2 = json.load(open(HERE / "main2" / "results.json"))["results"]
phase2 = json.load(open(HERE / "phase2_L0" / "results.json"))["results"]
floor = json.load(open(HERE / "bayes_floor.json"))

BAYES_W = floor["bayes_within_ce"]        # 0.7785
FLOOR_W = floor["floor_within_ce"]        # 0.8450
UNIF_W = floor["uniform_within_ce"]       # 1.0986

FAM = {  # fixed categorical assignment (validated palette slots)
    "sae": ("SAE latent cut", "#2a78d6"),
    "qk": ("FRA-QK cut", "#e34948"),
    "ov": ("FRA-OV cut", "#1baf7a"),
    "qkov": ("FRA QK+OV", "#eb6834"),
    "proj": ("probe-dir projection", "#4a3aa7"),
    "rand": ("random-latent control", "#8b8a86"),
}


def fam_of(name: str):
    head = name.split(":")[0]
    if "rand" in name:
        return "rand"
    return head if head in FAM else None


def rows(res, marker, run):
    out = []
    for name, m in res.items():
        if name == "clean" or "removal_frac" not in m:
            continue
        f = fam_of(name)
        if f is None:
            continue
        out.append(dict(
            fam=f, marker=marker, run=run, name=name,
            rf=m["removal_frac"], cf=m["collateral_frac"],
            probe=m["probe_r2_resid_last"], wce=m["within_ce"],
        ))
    return out


pts = rows(main2, "o", "L1") + rows(phase2, "^", "L0")
sweep_path = HERE / "ov_sweep.json"
if sweep_path.exists():
    sweep = json.load(open(sweep_path))
    for name, m in sweep.items():
        if name == "clean" or "removal_frac" not in m:
            continue
        f = fam_of(name)
        if f is None or m["removal_frac"] > 1.05:  # drop the destructive overshoot for scale
            continue
        pts.append(dict(fam=f, marker="^", run="L0", name=name,
                        rf=m["removal_frac"], cf=m["collateral_frac"],
                        probe=m["probe_r2"], wce=m["within_ce"]))
clean = main2["clean"]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
for ax in axes:
    ax.grid(alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ── Panel A: use vs presence ──
ax = axes[0]
for p in pts:
    ax.scatter(p["rf"], p["probe"], color=FAM[p["fam"]][1], marker=p["marker"],
               s=52 if p["fam"] != "rand" else 40, alpha=0.9, zorder=3,
               edgecolors="white", linewidths=0.6)
ax.scatter(0, clean["probe_r2_resid_last"], color="#0b0b0b", marker="*", s=190, zorder=4, label=None)
ax.annotate("clean", (0, clean["probe_r2_resid_last"]), xytext=(6, 6), textcoords="offset points",
            fontsize=9, color="#0b0b0b")
ax.annotate("SAE ω×4 @L1:\nuse 86% gone,\npresence still 0.49",
            (main2["sae:omega_top4:a1.0"]["removal_frac"], main2["sae:omega_top4:a1.0"]["probe_r2_resid_last"]),
            xytext=(-150, 40), textcoords="offset points", fontsize=9, color="#52514e",
            arrowprops=dict(arrowstyle="-", lw=0.8, color="#52514e"))
ax.annotate("FRA-QK: no use removed\nat any strength",
            (0.06, 0.985), xytext=(40, -34), textcoords="offset points", fontsize=9, color="#52514e",
            arrowprops=dict(arrowstyle="-", lw=0.8, color="#52514e"))
ax.annotate("FRA-OV @L0 gain sweep (c: 0.5→4):\nuse fully nulled at c≈4,\npresence intact, CF 0.036",
            (0.993, 0.985), xytext=(-172, -52), textcoords="offset points", fontsize=9, color="#0e7a55",
            arrowprops=dict(arrowstyle="-", lw=0.8, color="#0e7a55"))
ax.axhspan(0.40, 0.45, color="#eda100", alpha=0.12, zorder=1)
ax.text(0.985, 0.408, "observed presence floor ≈ 0.42 — no cut gets lower", fontsize=8.5,
        color="#52514e", ha="right")
ax.set_xlabel("removal of USE  (block CE toward prior, Bayes-normalized)")
ax.set_ylabel("linear PRESENCE remaining  (adversarial probe R²)")
ax.set_title("A — Removing the use is cheap; the presence stays", fontsize=11)
ax.set_xlim(-0.06, 1.04)
ax.set_ylim(0.35, 1.03)

# ── Panel B: presence vs absolute within-CE, with exact bounds ──
ax = axes[1]
# provably impossible: below Bayes ceiling everywhere
ax.axhspan(0.74, BAYES_W, color="#e34948", alpha=0.10, zorder=1)
ax.text(0.5, BAYES_W - 0.012, "impossible at ANY presence (below full-info Bayes)",
        fontsize=8.5, color="#a33", ha="center")
# provably impossible at zero presence: the tax notch
ax.fill_betweenx([BAYES_W, FLOOR_W], -0.005, 0.035, color="#e34948", alpha=0.28,
                 hatch="///", edgecolor="#e34948", linewidth=0.0, zorder=2)
ax.annotate("the 20.8% tax:\nimpossible at zero presence",
            (0.02, (BAYES_W + FLOOR_W) / 2), xytext=(30, -26), textcoords="offset points",
            fontsize=9, color="#a33",
            arrowprops=dict(arrowstyle="-", lw=0.8, color="#a33"))
# exact endpoints + achievable time-sharing chord
ax.plot([1.0, 0.0], [BAYES_W, FLOOR_W], ls="--", lw=1.2, color="#8b8a86", zorder=2)
ax.text(0.42, 0.803, "time-sharing frontier (achievable)", fontsize=8.5, color="#8b8a86",
        rotation=-4.5)
ax.scatter([0.0], [FLOOR_W], color="#e34948", marker="*", s=210, zorder=5)
ax.annotate("zero-info Bayes floor = 0.845", (0.0, FLOOR_W), xytext=(8, 8),
            textcoords="offset points", fontsize=9, color="#a33")
ax.axhline(BAYES_W, ls="--", lw=1.0, color="#008300")
ax.text(0.995, BAYES_W + 0.006, "Bayes ceiling (full info) = 0.779", fontsize=8.5,
        color="#008300", ha="right")
ax.axhline(UNIF_W, ls=":", lw=1.0, color="#8b8a86")
ax.text(0.995, UNIF_W + 0.006, "uniform within-block = 1.099", fontsize=8.5, color="#8b8a86", ha="right")
ax.axhline(clean["within_ce"], ls=":", lw=1.0, color="#0b0b0b")
ax.text(0.995, clean["within_ce"] + 0.006, f"clean model = {clean['within_ce']:.3f}",
        fontsize=8.5, color="#0b0b0b", ha="right")
for p in pts:
    ax.scatter(p["probe"], p["wce"], color=FAM[p["fam"]][1], marker=p["marker"],
               s=52 if p["fam"] != "rand" else 40, alpha=0.9, zorder=3,
               edgecolors="white", linewidths=0.6)
ax.scatter(clean["probe_r2_resid_last"], clean["within_ce"], color="#0b0b0b", marker="*", s=190, zorder=4)
ax.invert_xaxis()
ax.set_xlabel("linear PRESENCE remaining (probe R²)  →  full removal at 0")
ax.set_ylabel("within-block CE (nats, absolute)")
ax.set_title("B — Removing the presence is taxed (exact bounds)", fontsize=11)
ax.set_ylim(0.74, 1.30)

# shared legend
handles = [Line2D([], [], color=c, marker="s", ls="", ms=8, label=l) for l, c in FAM.values()]
handles += [
    Line2D([], [], color="#52514e", marker="o", ls="", ms=7, label="cut @ L1 interface (main2)"),
    Line2D([], [], color="#52514e", marker="^", ls="", ms=7, label="cut @ L0 interface (phase 2a)"),
    Line2D([], [], color="#0b0b0b", marker="*", ls="", ms=11, label="clean model"),
]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("The entanglement tax: concept USE is cuttable, concept PRESENCE is not (mixture-HMM toy, exact Bayes anchors)",
             fontsize=12.5, y=0.99)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
fig.savefig(HERE / "entanglement_tax.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE / 'entanglement_tax.png'}")
