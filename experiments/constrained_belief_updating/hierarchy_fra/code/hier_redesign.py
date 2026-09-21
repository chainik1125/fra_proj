"""Redesigned 2x2 (gating x occlusion): the parent->child handle, isolated.

Noise axis = CHILD OCCLUSION (mask child in the input w.p. q_occ) — the load-bearing
dial (H2.2); applies EQUALLY to gated and null, so the child-channel strength (and
the general-disruption confound) is matched and the gated-vs-null difference
isolates the gate. Parent kept clean; balanced mortality (q_P=0.65); lam_C=0.9;
2 layers (the reset-gate is 2-hop). Handle = child-MSE rise when d_P is projected
out of the KEYS. Writes out/redesign_2x2.json + out/handle_2x2.png incrementally.
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bernoulli_fra", "code"))
from hier_data import HierProcess
from hier_model import MLAttn, train_ml, child_handle
OUT = os.path.join(os.path.dirname(__file__), "..", "out")
torch.set_num_threads(4)
PARAMS = dict(lam_P=0.5, p_P=0.3, lam_C=0.9, p_C=0.5, mu_P=1.5, sig_P=0.3,
              mu_C=1.5, sig_C=0.3)                        # parent+child magnitudes CLEAN


def run_cell(gated, occ, steps=9000, seed=0):
    g = HierProcess(d=32, gated=gated, child_occlude=occ, seed=seed, **PARAMS)
    m = MLAttn(32, n_layers=2, n_ctx=24, seed=seed)
    train_ml(m, g, T=24, steps=steps)
    ch, pa, base = child_handle(m, g, T=24)
    return dict(gated=gated, occ=occ, child_handle=ch, parent_collateral=pa, child_mse=base)


def main():
    jpath = os.path.join(OUT, "redesign_2x2.json")
    cells = {"gated_occ": (True, 0.5), "gated_clean": (True, 0.0),
             "null_occ": (False, 0.5), "null_clean": (False, 0.0)}
    R = {}
    for name, (g, o) in cells.items():
        R[name] = run_cell(g, o)
        json.dump(R, open(jpath, "w"), indent=2)          # incremental
        print(f"{name:12s}: handle={R[name]['child_handle']:+.5f} "
              f"collat={R[name]['parent_collateral']:+.5f} childMSE={R[name]['child_mse']:.4f}",
              flush=True)
    # isolated gate = gated - null at matched occlusion
    iso_occ = R["gated_occ"]["child_handle"] - R["null_occ"]["child_handle"]
    iso_clean = R["gated_clean"]["child_handle"] - R["null_clean"]["child_handle"]
    R["isolated_handle_occ"] = iso_occ; R["isolated_handle_clean"] = iso_clean
    json.dump(R, open(jpath, "w"), indent=2)
    print(f"\nISOLATED gate (gated-null): occluded={iso_occ:+.5f}  clean={iso_clean:+.5f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    M = np.array([[R["gated_occ"]["child_handle"], R["gated_clean"]["child_handle"]],
                  [R["null_occ"]["child_handle"], R["null_clean"]["child_handle"]]])
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    v = max(abs(M).max(), 1e-6); im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["occluded child\n(ρ_obs>0)", "clean child\n(ρ_obs≈0)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["gated\n(hierarchy)", "no-gate\n(control)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i,j]:+.4f}", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set_title("FRA-QK parent→child handle (2×2)\nchild-MSE rise when parent cut from keys")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "handle_2x2.png"), dpi=150)
    print("wrote out/handle_2x2.png + redesign_2x2.json")


if __name__ == "__main__":
    main()
