"""Print a clean numeric summary of all results for the writeup."""
import os
import numpy as np
import torch

RES = os.environ.get("BAG_OUT", "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-sprint/results")


def load(n):
    p = f"{RES}/{n}"
    return torch.load(p, map_location="cpu", weights_only=False) if os.path.exists(p) else None


def msd(xs):
    a = np.array(xs); return a.mean(), a.std()


print("=" * 64)
r = load("quenched_sweep.pt")
if r:
    print("QUENCHED SWEEP  (roll_lens=%s, steps=%d, seeds=%d)" % (r["roll_lens"], r["steps"], r["seeds"]))
    print(f"{'prior':10s} {'chi2':>6s} {'chi2_hat(mean±sd)':>20s} {'kl_r2(mean)':>12s}")
    for pname, d in r["priors"].items():
        hats = [run["final"]["chi2_hat"] for run in d["runs"]]
        kls = [run["final"]["kl_r2"] for run in d["runs"]]
        m, s = msd(hats)
        print(f"{pname:10s} {d['chi2']:6.3f} {m:10.3f} ± {s:.3f}     {np.mean(kls):.5f}")
    c = r["control"]["final"]
    print(f"{'control':10s} {'--':>6s} {c['chi2_hat']:10.3f}            {c['kl_r2']:.5f}  (independent-bag)")

    # development: chi2_hat trajectory for fixed2 and fixed5 (mean over seeds)
    print("\nDEVELOPMENT (chi2_hat vs step), mean over seeds:")
    snaps = r["snap_steps"]
    hdr = "step      " + "  ".join(f"{p:>8s}" for p in ["fixed1", "fixed2", "fixed5", "fixed10"])
    print(hdr)
    for s in snaps:
        row = f"{s:<10d}"
        for p in ["fixed1", "fixed2", "fixed5", "fixed10"]:
            vals = [run["dev"][s]["chi2_hat"] for run in r["priors"][p]["runs"] if s in run["dev"]]
            row += f"  {np.mean(vals):8.3f}" if vals else "      --"
        print(row)

print("=" * 64)
a = load("annealed.pt")
if a:
    print("ANNEALED")
    for pname, d in a["priors"].items():
        print(f"  {pname:10s} meanN={d['mean_N']:.2f}  KL_laplace={d['final']['kl_laplace']:.5f}")
    if "probes" in a:
        print("  probes (decode N vs count s):")
        for l, pv in a["probes"].items():
            print(f"    L{l}: N_acc={pv['N_acc']:.3f} (base {pv['N_baseline']:.3f})  s_R2={pv['s_r2']:.4f}")
    if "order_invariance" in a:
        oi = a["order_invariance"]
        print(f"  order-invariance: mean|Δp1|={oi['mean_abs_delta_p1']:.4f}  p95={oi['p95_abs_delta_p1']:.4f}")

print("=" * 64)
m = load("mechanism.pt")
if m:
    print("MECHANISM")
    print("  q-decode during rollout 2 (best R² per position):")
    for p, v in m.get("q_decode", {}).items():
        print(f"    pos {p}: R²(q)={v['best_r2']:.4f} @L{v['best_layer']}  (q_std={v['q_std']:.3f})")
    print(f"  shuffled-q control R²={m.get('q_shuffled_r2', float('nan')):.4f}")
    print("  Laplace(s1) decode at ROLL:", {k: round(v, 4) for k, v in m.get("layer_r2_s1", {}).items()})
    ct = m.get("causal_transfer", {})
    print(f"  CAUSAL transfer: chi2={ct.get('chi2_true')}  clean={ct.get('chi2_hat_clean'):.3f}")
    for l, v in ct.get("chi2_hat_ablate_by_layer", {}).items():
        print(f"    mean-ablate s1-dir @ layer {l}: chi2_hat={v:.3f}")
    print(f"    random-dir @ last layer: chi2_hat={ct.get('chi2_hat_ablate_random_lastlayer'):.3f}")
    cs = m.get("causal_same")
    if cs:
        print(f"  CAUSAL same-source @pos{cs['pos']}: clean={cs['same_slope_clean']:.3f}  "
              f"ablate-q={cs['same_slope_ablate_q']:.3f}  random={cs['same_slope_ablate_random']:.3f}")
    ko = m.get("causal_attn_knockout")
    if ko:
        print(f"  CAUSAL attn-knockout (rollout2 X-> rollout1):")
        print(f"    chi2_hat: clean={ko['chi2_clean']:.3f}  knockout={ko['chi2_knockout']:.3f}")
        print(f"    KL rollout1: clean={ko['kl_r1_clean']:.4f}  knockout={ko['kl_r1_knockout']:.4f}")
        print(f"    KL rollout2: clean={ko['kl_r2_clean']:.4f}  knockout={ko['kl_r2_knockout']:.4f}")
print("=" * 64)
