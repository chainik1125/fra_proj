"""
C2 (process level): Level-L concatenated majority code — recursive threshold.

Applies the 3-block majority map f(p) = 3p^2 - 2p^3 repeatedly.
  p_{l+1} = P(Bin(3, p_l) >= 2)  =  3p_l^2 - 2p_l^3

Fixed points: 0 (stable, below threshold), 1/2 (unstable), 1 (stable, above threshold).
Near 0:      f(p) ≈ 3p^2, so  p_l ≈ (1/3)(3 p_0)^{2^l}  — super-exponential suppression.
Near 1/2:    f'(1/2) = 3/2 > 1, confirming 1/2 is unstable.

Below p=1/2: each level halves the log-error (or faster); SUPER-LINEAR suppression.
Above p=1/2: each level drives toward 1 — redundancy entrenches misalignment.

This closes the error-correction ladder: §4.2 shows the single-level code;
this shows that *concatenation* gives exponentially better protection below threshold.

Outputs results/c2_concatenation.pt
"""
import os
import numpy as np

OUT = os.environ.get("BAG_OUT", "/home/user/simplex-research/results")


def majority_map(p, n=3):
    """P(Bin(n,p) >= majority), for odd n."""
    from math import comb
    r = n // 2  # corrects up to r errors; majority = r+1
    return sum(comb(n, k) * p**k * (1 - p)**(n - k) for k in range(r + 1, n + 1))


def iterate(p0, levels=5, n=3):
    """Return p_0, p_1, ..., p_{levels}."""
    ps = [p0]
    for _ in range(levels):
        ps.append(majority_map(ps[-1], n))
    return np.array(ps)


def main():
    import torch

    levels = 5  # 0 through 5

    # (1) trajectories for a range of initial p_0 values
    p0_grid = np.linspace(0.01, 0.99, 99)
    trajectories = {float(p0): iterate(p0, levels).tolist() for p0 in p0_grid}

    # (2) Selected p_0 values for plot — straddling 1/2
    p0_selected = [0.1, 0.2, 0.3, 0.4, 0.45, 0.49, 0.51, 0.55, 0.6, 0.7, 0.8, 0.9]
    traj_selected = {p0: iterate(p0, levels).tolist() for p0 in p0_selected}

    # (3) Verify: fixed points
    for p_label, p0 in [("0.0", 0.0), ("0.5", 0.5), ("1.0", 1.0)]:
        pL = iterate(p0, levels)[-1]
        print(f"  Fixed point check: p0={p_label} -> p_{levels}={pL:.6f}")

    # (4) Linearisation at p=1/2: f'(1/2) = 6*(1/2)*(1-1/2) = 3/2  (unstable)
    slope_at_half = 6 * 0.5 * 0.5  # f'(p) = 6p(1-p)
    print(f"  f'(1/2) = {slope_at_half:.4f}  (>1 => unstable fixed point => threshold)")

    # (5) Suppression rate: for p0=0.2 show |log p_l|
    p_traj = iterate(0.2, levels)
    print("  p_l at p0=0.2:", [f"{x:.4e}" for x in p_traj])
    logerr = [np.log10(x) for x in p_traj if x > 0]
    print("  log10(p_l):", [f"{x:.2f}" for x in logerr])

    # (6) Oracle level-l values across p_0 grid (for the final-level contour)
    final_level = {l: [iterate(p0, l)[-1] for p0 in p0_grid] for l in range(levels + 1)}

    res = {
        "p0_grid": p0_grid.tolist(),
        "traj_selected": traj_selected,
        "final_level": final_level,
        "levels": levels,
        "n": 3,
        "slope_at_half": slope_at_half,
        "p_traj_at_02": p_traj.tolist(),
    }
    os.makedirs(OUT, exist_ok=True)
    torch.save(res, f"{OUT}/c2_concatenation.pt")
    print(f"saved {OUT}/c2_concatenation.pt")


if __name__ == "__main__":
    main()
