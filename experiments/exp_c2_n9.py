"""
C2 redundancy-scaling extension: a SINGLE deeper majority-code point (default n=9, r=4)
to test whether the learned majority-decode of exp_c2_logical.py (validated at n=3/5/7)
continues to track the binomial-tail oracle beyond the tested range, rather than memorising
the n<=7 rule. Same headline config (d=128/3L/4H, packed-emission vocab=2^n+3, query readout,
6000 steps) so the n=9 number is directly comparable to the n=3/5/7 curve.

Oracle anchors (closed form, no fitting):
  * logical posterior P(L=M|obs) = poisson_binomial_tail(per-chain filter posteriors, r)
  * binomial-tail steady state  P(Bin(n,q*)>r),  q*=eps/(eps+gamma)
Reuses make_seqs logic identical to exp_c2_logical.py for apples-to-apples comparison.

Env: C2_N (default 9), C2_STEPS (default 6000), C2_TAG (default "n9") -> results/c2_<tag>.pt
"""
import os
import numpy as np
import torch
from bag_moments import active, probe, train
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", "results")
EPS, GAMMA, PA, PM, T = 0.05, 0.15, 0.3, 0.7, 40
N = int(os.environ.get("C2_N", "9"))
STEPS = int(os.environ.get("C2_STEPS", "6000"))
SEED = int(os.environ.get("C2_SEED", "0"))
TAG = os.environ.get("C2_TAG", "n9")


def make_seqs(B, n, rng, with_query):
    tk, hid, _ = active.gen_redundant_active(B, T, n, EPS, GAMMA, PA, PM, rng, beta=0.0)
    packed = active.pack_emissions(tk)                       # (B,T)
    r = (n - 1) // 2
    qchain = active.per_chain_filter(tk, EPS, GAMMA, PA, PM)  # (B,T,n)
    qlast = qchain[:, -1, :]                                  # (B,n)
    logical_post = active.poisson_binomial_tail(qlast, r)     # (B,)
    true_L = (hid[:, -1, :].sum(1) > r).astype(np.int64)      # (B,)
    QUERY = 2 ** n; R_no = 2 ** n + 1; R_yes = 2 ** n + 2
    if with_query:
        readout = np.where(true_L == 1, R_yes, R_no)
        seq = np.concatenate([packed, np.full((B, 1), QUERY), readout[:, None]], axis=1)  # (B,T+2)
        vocab = 2 ** n + 3
    else:
        seq = packed; vocab = 2 ** n
    return seq, vocab, {"logical_post": logical_post, "true_L": true_L, "qlast": qlast,
                        "R_yes": R_yes, "query_pos": T}


def main():
    device = train.get_device()
    n = N; r = (n - 1) // 2
    vocab = 2 ** n + 3
    rng = np.random.default_rng(n * 100 + SEED)   # training data varies with seed
    erng = np.random.default_rng(1000 + n)        # eval set FIXED across seeds (comparable)
    seq_ev, _, meta = make_seqs(4096, n, erng, with_query=True)
    ev = torch.tensor(seq_ev, device=device)
    qpos = meta["query_pos"]; R_yes = meta["R_yes"]
    orac = meta["logical_post"]

    def batch_fn():
        s, _, _ = make_seqs(256, n, rng, with_query=True)
        t = torch.tensor(s, device=device); return t[:, :-1], t[:, 1:]

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            logits = model(ev[:, :-1])
            p = torch.softmax(logits[:, qpos], -1).cpu().numpy()
            p_yes = p[:, R_yes] / (p[:, R_yes] + p[:, R_yes - 1] + 1e-9)
        e = 1e-7; po = np.clip(orac, e, 1-e); pm = np.clip(p_yes, e, 1-e)
        kl = (po*np.log(po/pm) + (1-po)*np.log((1-po)/(1-pm))).mean()
        return {"kl_logical": float(kl)}

    torch.manual_seed(SEED)
    cfg = GPTConfig(vocab_size=vocab, n_ctx=T+2, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
    model = TinyGPT(cfg)
    out = train.train(model, batch_fn, steps=STEPS, lr=1e-3, device=device,
                      snapshot_steps=[STEPS], eval_fn=eval_fn, log_every=max(STEPS//3, 1))
    model.eval()
    kl = out["snapshots"][STEPS]["metrics"]["kl_logical"]

    with torch.no_grad():
        p = torch.softmax(model(ev[:, :-1])[:, qpos], -1).cpu().numpy()
        p_yes = p[:, R_yes] / (p[:, R_yes] + p[:, R_yes - 1] + 1e-9)
    model_pred_L = (p_yes > 0.5).astype(int)
    model_logical_err = (model_pred_L != meta["true_L"]).mean()
    bayes_logical_err = np.minimum(orac, 1 - orac).mean()
    ql = meta["qlast"]
    phys_bayes_err = np.minimum(ql, 1 - ql).mean()

    Xq = probe.extract_resid(model, ev, layer=cfg.n_layers - 1, pos=qpos)
    zt = np.log(np.clip(orac, 1e-4, 1-1e-4) / np.clip(1-orac, 1e-4, 1-1e-4))
    r2_logical = probe.ridge_probe(Xq, zt)["r2"]

    res = {"n": n, "r": r, "eps": EPS, "gamma": GAMMA, "pA": PA, "pM": PM, "T": T, "steps": STEPS,
           "kl_logical": kl, "model_logical_err": float(model_logical_err),
           "bayes_logical_err": float(bayes_logical_err), "phys_bayes_err": float(phys_bayes_err),
           "binom_tail_qstar": float(active.binomial_tail(n, active.stationary_q(EPS, GAMMA), r)),
           "r2_logical_query_model": float(r2_logical)}
    print(f"n={n} (r={r}): KL_logical={kl:.4f}  model_err={model_logical_err:.3f} "
          f"bayes_logical={bayes_logical_err:.3f} phys={phys_bayes_err:.3f} "
          f"binom_tail={res['binom_tail_qstar']:.3f} R2(logical)={r2_logical:.3f}", flush=True)
    torch.save(res, f"{OUT}/c2_{TAG}.pt")
    print(f"saved {OUT}/c2_{TAG}.pt")


if __name__ == "__main__":
    main()
