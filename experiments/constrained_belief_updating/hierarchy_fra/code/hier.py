"""Setting H — minimal temporal hierarchy: parent reset chain GATING a child chain.

Joint latent s(t) = (z_P, z_C) in {A=(0,0), B=(1,0), C=(1,1)} — state (0,1) is
forbidden (child on requires parent on). Dynamics (reset-on-parent-death):
  parent P: reset chain rate lam_P, prob p_P  (Setting B).
  child  C: gated by parent. z_P(t)=0 => z_C(t)=0. If parent just turned on
    (z_P(t)=1, z_P(t-1)=0): child RE-INITIALISES z_C ~ Bern(p_C). If parent stayed
    on (z_P(t-1)=1): child reset chain rate lam_C, prob p_C. => the child state is
    RESET whenever the parent dies; this is what makes the joint filter exact and
    3-state, and is the load-bearing convention (stated in H_process.md).
Null control: gating removed -> P, C independent reset chains (product chain).

Observation a_t = c_P d_P + c_C d_C + b, c_x = z_x ReLU(mu_x+sig_x eps_x) (Setting B).
"""
import numpy as np

A, B, C = 0, 1, 2  # joint states (0,0),(1,0),(1,1)


def joint_T(lam_P, p_P, lam_C, p_C):
    """3x3 joint transition T[from,to] over {A,B,C} (hierarchy/gated)."""
    qP = lam_P + (1 - lam_P) * p_P          # P(z_P=1 | z_P^-=1)
    aP = (1 - lam_P) * p_P                   # P(z_P=1 | z_P^-=0)
    off1 = (1 - lam_P) * (1 - p_P)           # P(z_P=0 | z_P^-=1)
    off0 = 1 - aP                            # P(z_P=0 | z_P^-=0)
    T = np.zeros((3, 3))
    # from A=(0,0): parent from off; child re-inits on parent-on
    T[A, A] = off0
    T[A, B] = aP * (1 - p_C)
    T[A, C] = aP * p_C
    # from B=(1,0): parent from on; child reset-chain from z_C^-=0
    T[B, A] = off1
    T[B, B] = qP * (1 - (1 - lam_C) * p_C)
    T[B, C] = qP * (1 - lam_C) * p_C
    # from C=(1,1): parent from on; child reset-chain from z_C^-=1
    T[C, A] = off1
    T[C, B] = qP * (1 - lam_C) * (1 - p_C)
    T[C, C] = qP * (lam_C + (1 - lam_C) * p_C)
    return T


def null_T(lam_P, p_P, lam_C, p_C):
    """Null control: independent P and C reset chains, product on {(zP,zC)}.
    Here child is NOT gated; state space is full 2x2 = 4 states {00,01,10,11}."""
    def chain(lam, p):
        return np.array([[1 - (1 - lam) * p, (1 - lam) * p],
                         [(1 - lam) * (1 - p), lam + (1 - lam) * p]])
    TP = chain(lam_P, p_P); TC = chain(lam_C, p_C)
    return np.kron(TP, TC)                    # 4x4, states (zP,zC) lexicographic


def stationary(T):
    w, V = np.linalg.eig(T.T)
    i = np.argmin(np.abs(w - 1))
    pi = np.real(V[:, i]); return pi / pi.sum()


def child_pred(T, s):
    """P(z_C(t+1)=1 | s(t)=s) = T[s, C] (child-on next = joint state C)."""
    return T[s, C]


def parent_pred(T, s):
    """P(z_P(t+1)=1 | s(t)=s) = T[s,B] + T[s,C]."""
    return T[s, B] + T[s, C]


# ---------- exact joint Bayes filter (clean obs => current joint state observed) ----------
def observed_state(zP, zC):
    return A if zP == 0 else (C if zC == 1 else B)


if __name__ == "__main__":
    lam_P, p_P, lam_C, p_C = 0.8, 0.4, 0.7, 0.5
    T = joint_T(lam_P, p_P, lam_C, p_C)
    print("joint T=\n", np.round(T, 4), "\nrow sums", T.sum(1))
    pi = stationary(T); print("stationary pi (A,B,C)=", np.round(pi, 4),
                              " expect", np.round([1 - p_P, p_P * (1 - p_C), p_P * p_C], 4))
    eig = np.sort(np.linalg.eigvals(T).real)[::-1]
    print("joint eigenvalues (nonstationary carry the timescales):", np.round(eig, 4))
    print("child one-step pred by current state: A=%.4f B=%.4f C=%.4f"
          % (child_pred(T, A), child_pred(T, B), child_pred(T, C)))
    print("  -> child prediction DEPENDS on the parent state (A vs B): "
          "%.4f vs %.4f" % (child_pred(T, A), child_pred(T, B)))
    # null control spectrum
    Tn = null_T(lam_P, p_P, lam_C, p_C)
    print("null-control eigenvalues:", np.round(np.sort(np.linalg.eigvals(Tn).real)[::-1], 4),
          " (should be {1, lam_P, lam_C, lam_P*lam_C})",
          np.round([1, lam_P, lam_C, lam_P * lam_C], 4))
