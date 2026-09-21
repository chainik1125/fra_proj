"""Diagnostic: WHERE does the parent->child gate live in the trained no-LN model?
Ablate the parent feature d_P from each pathway and measure Delta child/parent MSE:
  keys  (QK)  : keyproj   -> is the gate a QK content coupling (child-query x parent-key)?
  values(OV)  : valproj   -> is the gate an OV read (attend + read parent via values)?
  both        : keyproj+valproj
  input       : force parent OFF in the input everywhere -> TOTAL parent->child dependence.
Reference: parent-blind Bayes gap (pure gating value). Run on cached gated & null full models.
"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bernoulli_fra", "code"))
sys.path.insert(0, os.path.dirname(__file__))
from hier_data import HierProcess
from rung2_lncheck import make_model
import centerpiece as cp

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
PARAMS = cp.PARAMS; T = 24


def load(gated, tag):
    gen = HierProcess(d=32, gated=gated, child_occlude=0.5, seed=0, **PARAMS)
    m = make_model(False, False, gen, seed=0, drop_ln=True)
    m.load_state_dict(torch.load(os.path.join(OUT, f"ck_{tag}.pt"))); m.eval()
    return m, gen


for gated, cellname in [(True, "gated_occ"), (False, "null_occ")]:
    tag = f"{cellname}_full_noLN"
    if not os.path.exists(os.path.join(OUT, f"ck_{tag}.pt")):
        print(f"[{cellname}] no checkpoint yet"); continue
    m, gen = load(gated, tag)
    X, Y, _ = gen.sample_seq(30000, T); dP, dC = gen.D[0], gen.D[1]
    bP = float(gen.b @ dP)
    fr = cp.frontiers(gen, X, Y, gated)
    base_c, base_p = cp.mse_cd(m, X, Y, dP, dC)
    # input ablation: force parent to its OFF-reading (X.dP -> bP) everywhere
    Xin = X - ((X @ dP) - bP)[:, :, None] * dP
    ic, ip = cp.mse_cd(m, Xin, Y, dP, dC)
    rows = [
        ("keys  (QK cut)", cp.mse_cd(m, X, Y, dP, dC, keyproj=(1.0, [0, 1]))),
        ("values(OV cut)", cp.mse_cd(m, X, Y, dP, dC, valproj=(1.0, [0, 1]))),
        ("keys+values   ", cp.mse_cd(m, X, Y, dP, dC, keyproj=(1.0, [0, 1]), valproj=(1.0, [0, 1]))),
        ("input (total) ", (ic, ip)),
    ]
    print(f"\n[{cellname}] base child={base_c:.5f} parent={base_p:.5f} | "
          f"bayes={fr['bayes']:.4f} pblind={fr['parent_blind']:.4f} gating_value={fr['gating_value']:.4f}")
    for name, (c, p) in rows:
        print(f"   ablate dP from {name}: dChild={c-base_c:+.5f}  dParent={p-base_p:+.5f}")
