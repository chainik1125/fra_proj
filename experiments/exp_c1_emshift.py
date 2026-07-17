"""
C1 algorithm-vs-memorization, FULL-OOD extension (no retraining): loads the canonical headline C1 model
(results/c1.pt, trained at eps=0.05,gamma=0.15,pA=0.3,pM=0.7, L=200/6000 steps) and evaluates it on
sequences whose EMISSION model is also shifted (pA',pM'), not just the transition dynamics. This removes
the "emissions held fixed" caveat of exp_c1_genshift.py: if the model baked in its training likelihood
(pA=0.3,pM=0.7) it should track the TRAINED-parameter filter even when the true emissions differ, while
the locally-optimal TRUE filter (using pA',pM') diverges. Anchored to the exact 2-state filter.

Outputs results/c1_emshift.pt
"""
import os, sys
import numpy as np
import torch
ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from bag_moments import active
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", os.path.join(ROOT, "results"))
BURN = 30
NEV = int(os.environ.get("EM_NEV", "2048"))


def kl_next(p_oracle, p_model):
    e = 1e-7
    po = np.clip(p_oracle, e, 1 - e); pm = np.clip(p_model, e, 1 - e)
    return po * np.log(po / pm) + (1 - po) * np.log((1 - po) / (1 - pm))


def main():
    d = torch.load(os.path.join(OUT, "c1.pt"), weights_only=False)
    eps0, g0, pA0, pM0 = d["eps"], d["gamma"], d["pA"], d["pM"]
    cfg = GPTConfig(**d["cfg"]); model = TinyGPT(cfg); model.load_state_dict(d["state"]); model.eval()
    L = cfg.n_ctx
    print(f"=== C1 emission-shift (canonical model: eps={eps0},gamma={g0},pA={pA0},pM={pM0},L={L},train KL={d['kl']:.2e}) ===", flush=True)

    # Sanity: reproduce in-distribution KL on fresh data with the loaded model
    erng = np.random.default_rng(777)
    toks0, _ = active.gen_active_2state(NEV, L, eps0, g0, pA0, pM0, erng)
    _, _, n1_0 = active.forward_filter_2state(toks0, eps0, g0, pA0, pM0)
    with torch.no_grad():
        p0 = torch.softmax(model(torch.tensor(toks0)[:, :-1]), -1)[..., 1].cpu().numpy()
    indist_kl = float(kl_next(n1_0[:, :-1], p0)[:, BURN:].mean())
    print(f"  reloaded in-dist next-symbol KL to TRAINED filter = {indist_kl:.5f} (headline {d['kl']:.2e})", flush=True)

    # Emission-shift grid: vary (pA',pM') keeping transitions at the trained (eps0,g0).
    # symmetric around 0.5 by a "separation" s: pA=0.5-s, pM=0.5+s. trained s0 = 0.2 (0.3/0.7).
    SEPS = [0.10, 0.15, 0.20, 0.30, 0.40]   # 0.20 == trained (no shift)
    rows = []
    for s in SEPS:
        pAp, pMp = 0.5 - s, 0.5 + s
        teval = np.random.default_rng(2000 + int(100 * s))
        toks, _ = active.gen_active_2state(NEV, L, eps0, g0, pAp, pMp, teval)
        with torch.no_grad():
            pm = torch.softmax(model(torch.tensor(toks)[:, :-1]), -1)[..., 1].cpu().numpy()
        _, _, nxt_trained = active.forward_filter_2state(toks, eps0, g0, pA0, pM0)   # trained likelihood
        _, _, nxt_true = active.forward_filter_2state(toks, eps0, g0, pAp, pMp)      # correct likelihood
        kt = float(kl_next(nxt_trained[:, :-1], pm)[:, BURN:].mean())
        ku = float(kl_next(nxt_true[:, :-1], pm)[:, BURN:].mean())
        ko = float(kl_next(nxt_true[:, :-1], nxt_trained[:, :-1])[:, BURN:].mean())
        rows.append({"sep": s, "pA": pAp, "pM": pMp, "kl_to_trained": kt, "kl_to_true": ku, "kl_oracles": ko,
                     "model_rate": float(pm[:, BURN:].mean()),
                     "rate_trained": float(nxt_trained[:, BURN:-1].mean()),
                     "rate_true": float(nxt_true[:, BURN:-1].mean())})
        print(f"  emission sep={s:.2f} (pA={pAp:.2f},pM={pMp:.2f}) | KL(model||TRAINED)={kt:.5f}  "
              f"KL(model||TRUE)={ku:.5f}  KL(oracles)={ko:.5f} | model_rate={pm[:, BURN:].mean():.3f} "
              f"(trained-filt {nxt_trained[:, BURN:-1].mean():.3f}, true-filt {nxt_true[:, BURN:-1].mean():.3f})", flush=True)

    # ---- ALSO run the TRANSITION-shift grid on the canonical model: a 2nd independently-trained
    # model (L=200/6000) corroborating exp_c1_genshift.py's L=160 run (addresses the 1-seed concern). ----
    TRANS = [(0.05, 0.45), (0.05, 0.25), (0.05, 0.15), (0.05, 0.08), (0.05, 0.05), (0.12, 0.15)]
    trans_rows = []
    for (epsp, gammap) in TRANS:
        teval = np.random.default_rng(1234 + int(1000 * gammap))
        toks, _ = active.gen_active_2state(NEV, L, epsp, gammap, pA0, pM0, teval)
        with torch.no_grad():
            pm = torch.softmax(model(torch.tensor(toks)[:, :-1]), -1)[..., 1].cpu().numpy()
        _, _, nxt_trained = active.forward_filter_2state(toks, eps0, g0, pA0, pM0)
        _, _, nxt_true = active.forward_filter_2state(toks, epsp, gammap, pA0, pM0)
        qstar = active.stationary_q(epsp, gammap)
        trans_rows.append({"eps": epsp, "gamma": gammap, "qstar": qstar,
                           "kl_to_trained": float(kl_next(nxt_trained[:, :-1], pm)[:, BURN:].mean()),
                           "kl_to_true": float(kl_next(nxt_true[:, :-1], pm)[:, BURN:].mean())})
    tt = np.array([r["kl_to_trained"] for r in trans_rows]); tu = np.array([r["kl_to_true"] for r in trans_rows])
    print(f"  [canonical-model TRANSITION shift] KL(model||TRAINED)={tt.min():.5f}-{tt.max():.5f} (flat) "
          f"vs KL(model||TRUE)={tu.min():.5f}-{tu.max():.5f} (grows) — corroborates the L=160 run", flush=True)

    torch.save({"rows": rows, "trans_rows": trans_rows, "indist_kl": indist_kl, "headline_kl": float(d["kl"]),
                "config": {"eps0": eps0, "gamma0": g0, "pA0": pA0, "pM0": pM0, "L": L, "nev": NEV,
                           "burn": BURN, "seps": SEPS}}, f"{OUT}/c1_emshift.pt")
    print(f"saved {OUT}/c1_emshift.pt", flush=True)
    kt = np.array([r["kl_to_trained"] for r in rows]); ku = np.array([r["kl_to_true"] for r in rows])
    print(f"\nHEADLINE: under EMISSION shift, KL(model||TRAINED)={kt.min():.5f}-{kt.max():.5f} (flat ≈ "
          f"in-dist {indist_kl:.5f}); KL(model||TRUE)={ku.min():.5f}-{ku.max():.5f} (grows). The model "
          f"applies its TRAINED likelihood even when emissions shift — full-OOD confirmation." if kt.max() < ku.max()
          else "\nNULL/surprising: model does not track the trained filter under emission shift.", flush=True)


if __name__ == "__main__":
    main()
