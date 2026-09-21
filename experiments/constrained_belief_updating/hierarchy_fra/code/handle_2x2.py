"""The 2x2 money plot: the FRA-QK parent->child handle, isolated.

The loss-gate (full vs position-only) conflates child-self content-attention with
the parent->child gate. The HANDLE is isolated by a TARGETED FRA-QK cut: remove the
PARENT content (d_P projection) from the KEYS only, and measure the rise in the
CHILD-component MSE. If the pattern gates child evidence on parent state, this cut
breaks the gate -> child MSE rises (O(1)); in the no-gating and clean-obs controls
the parent is uninformative for the child -> ~0. Reuses the 4 trained gate models.
2x2 (gating x noise); child = handle, parent = collateral control. Writes
out/handle_2x2.json + out/handle_2x2.png.
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bernoulli_fra", "code"))
from hier_data import HierProcess
from model import OneLayerAttn
OUT = os.path.join(os.path.dirname(__file__), "..", "out")
torch.set_num_threads(4)


def load(tag):
    d = torch.load(os.path.join(OUT, f"model_{tag}.pt"), map_location="cpu")
    c = d["cfg"]
    m = OneLayerAttn(c["d"], d_h=c["d"], bias=True, seed=0, pos_key=True, n_ctx=c["T"])
    m.load_state_dict(d["state_dict"]); m.eval()
    gen = HierProcess(d=c["d"], sigma=c["sigma"], gated=c["gated"], seed=0)
    return m, c, gen


@torch.no_grad()
def forward_keycut(m, X, dP, b, cut=False):
    """OneLayerAttn forward; if cut, project d_P (parent) out of the KEY inputs."""
    Xt = torch.tensor(X, dtype=torch.float32)
    Xk = Xt
    if cut:
        dPt = torch.tensor(dP, dtype=torch.float32); bt = torch.tensor(b, dtype=torch.float32)
        proj = ((Xt - bt) @ dPt)[..., None] * dPt          # parent content along d_P
        Xk = Xt - proj                                     # keys see no parent content
    q = Xt @ m.Wq + m.bq
    k = Xk @ m.Wk + m.bk + m.KP[:Xt.shape[1]]
    B, Tn = Xt.shape[0], Xt.shape[1]
    q = q.view(B, Tn, m.n_heads, m.dph); k = k.view(B, Tn, m.n_heads, m.dph)
    sc = torch.einsum('bthc,bshc->bhts', q, k) / (m.dph ** 0.5)
    mask = torch.triu(torch.ones(Tn, Tn, dtype=torch.bool), diagonal=1)
    sc = sc.masked_fill(mask, float('-inf'))
    A = torch.softmax(sc, -1)
    ctx = torch.einsum('bhts,bsd->btd', A, Xt @ m.V.T) / m.n_heads
    out = Xt @ m.W.T + ctx + (m.beta if m.beta is not None else 0.0)
    return out.numpy()


def comp_mse(pred, Y, d_):
    return float((((pred - Y) @ d_) ** 2).mean())


def measure(tag, B_=12000, T=24):
    m, c, gen = load(tag)
    X, Y, _ = gen.sample_seq(B_, T)
    dP, dC = gen.D[0], gen.D[1]
    base = forward_keycut(m, X, dP, gen.b, cut=False)
    cutk = forward_keycut(m, X, dP, gen.b, cut=True)
    childH = comp_mse(cutk, Y, dC) - comp_mse(base, Y, dC)   # parent->child handle
    parentH = comp_mse(cutk, Y, dP) - comp_mse(base, Y, dP)  # collateral control
    return dict(tag=tag, gated=c["gated"], sigma=c["sigma"],
                child_handle=childH, parent_collateral=parentH,
                child_mse_base=comp_mse(base, Y, dC))


def main():
    cells = {"H_gated": ("gated", "noisy"), "H_null": ("no-gate", "noisy"),
             "H_gated_clean": ("gated", "clean"), "H_null_clean": ("no-gate", "clean")}
    R = {t: measure(t) for t in cells}
    for t, (g, n) in cells.items():
        print(f"{g:8s} {n:6s}: parent->child handle (child-MSE rise) = {R[t]['child_handle']:+.5f}"
              f"  parent-collateral = {R[t]['parent_collateral']:+.5f}")
    json.dump({t: R[t] for t in cells}, open(os.path.join(OUT, "handle_2x2.json"), "w"), indent=2)

    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    M = np.array([[R["H_gated"]["child_handle"], R["H_gated_clean"]["child_handle"]],
                  [R["H_null"]["child_handle"], R["H_null_clean"]["child_handle"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    vmax = max(abs(M).max(), 1e-6)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["noisy (ρ_obs>0)", "clean (ρ_obs≈0)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["gated (hierarchy)", "no-gate (control)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i,j]:+.4f}", ha="center", va="center",
                    fontsize=13, fontweight="bold")
    ax.set_title("FRA-QK parent→child handle\n(child-MSE rise when parent cut from keys)")
    fig.colorbar(im, ax=ax, label="handle (child-MSE rise)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "handle_2x2.png"), dpi=150)
    print("wrote out/handle_2x2.png + handle_2x2.json")


if __name__ == "__main__":
    main()
