"""Removal-vs-collateral Pareto plot for the fra_hmm_toy results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILIES = {
    "sae:omega": ("SAE cut (ω feats)", "tab:blue", "o"),
    "sae:block": ("SAE cut (block feats)", "tab:cyan", "s"),
    "sae:rand": ("SAE cut (random)", "lightsteelblue", "x"),
    "proj:": ("probe-dir projection", "tab:purple", "D"),
    "qk:omega": ("FRA-QK (ω feats)", "tab:red", "o"),
    "qk:block": ("FRA-QK (block feats)", "tab:orange", "s"),
    "qk:rand": ("FRA-QK (random)", "mistyrose", "x"),
    "ov:omega": ("FRA-OV (ω feats)", "tab:green", "o"),
    "ov:block": ("FRA-OV (block feats)", "limegreen", "s"),
    "ov:rand": ("FRA-OV (random)", "honeydew", "x"),
    "qkov:": ("FRA QK+OV", "tab:brown", "^"),
}


def family_of(name: str):
    for prefix, spec in FAMILIES.items():
        if name.startswith(prefix) or (prefix.endswith(":") and name.startswith(prefix[:-1])):
            if name.startswith(prefix):
                return spec
    for prefix, spec in FAMILIES.items():
        if name.split(":")[0] == prefix.rstrip(":"):
            return spec
    return ("other", "gray", ".")


def main(out_dir: Path):
    r = json.load(open(out_dir / "results.json"))
    res = r["results"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    seen = set()
    for name, m in res.items():
        if name == "clean" or "removal_frac" not in m:
            continue
        label, color, marker = family_of(name)
        ax.scatter(
            m["removal_frac"], m["collateral_frac"],
            color=color, marker=marker, s=55,
            label=label if label not in seen else None, zorder=3,
        )
        seen.add(label)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.axvline(1, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("removal fraction (block CE toward prior)")
    ax.set_ylabel("collateral fraction (within-block CE toward uniform)")
    ax.set_title("Concept removal vs collateral (Bayes-normalized)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for name, m in res.items():
        if name == "clean":
            ax.scatter(m["tracking_r2"], m["probe_r2_resid_last"], color="k", marker="*", s=140, label="clean", zorder=4)
            continue
        label, color, marker = family_of(name)
        ax.scatter(m["tracking_r2"], m["probe_r2_resid_last"], color=color, marker=marker, s=45, zorder=3)
    ax.set_xlabel("behavioral tracking R² (block mass vs posterior ω)")
    ax.set_ylabel("adversarial probe R² (last-layer resid → ω)")
    ax.set_title("Behavioral use vs linear presence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "pareto.png", dpi=150)
    print(f"saved {out_dir / 'pareto.png'}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "out/main"))
