"""
C2 (transformer): does a transformer represent the LOGICAL (majority-decoded) alignment
state of a redundant active bag, and inherit the code's error suppression?

Setup: n independent alignment chains; per timestep the n emissions are packed into one
token (vocab 2^n). Sequence = [packed_0 .. packed_{T-1}, QUERY], and the QUERY position's
target is a READOUT token = the true logical (majority) alignment at the last step. To
predict it the model must demultiplex the n streams, filter each chain, and majority-decode.

  C2a  query model matches the exact logical oracle (poisson-binomial tail of the per-chain
       posteriors) at the QUERY position (KL -> 0).
  C2b  the logical posterior P(L=M|obs) -- a NONLINEAR consensus -- is linearly decodable
       from the query model's residual; a model trained WITHOUT the query (packed
       next-token only) represents the chains but NOT the consensus (task-induced).
  C2c  the model's logical decode error is suppressed below single-chain (physical) error,
       tracking the Bayes logical error, and falls with n (error correction realized).

Outputs results/c2_logical.pt
"""
import os
import numpy as np
import torch

from bag_moments import active, probe, train
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-sprint-2/results")
EPS, GAMMA, PA, PM, T = 0.05, 0.15, 0.3, 0.7, 40


def make_seqs(B, n, rng, with_query):
    tk, hid, _ = active.gen_redundant_active(B, T, n, EPS, GAMMA, PA, PM, rng, beta=0.0)
    packed = active.pack_emissions(tk)                       # (B,T)
    r = (n - 1) // 2
    qchain = active.per_chain_filter(tk, EPS, GAMMA, PA, PM)  # (B,T,n)
    qlast = qchain[:, -1, :]                                  # (B,n) posteriors at last step
    logical_post = active.poisson_binomial_tail(qlast, r)     # (B,) P(L=M|obs)
    true_L = (hid[:, -1, :].sum(1) > r).astype(np.int64)      # (B,) true majority
    QUERY = 2 ** n; R_no = 2 ** n + 1; R_yes = 2 ** n + 2
    if with_query:
        readout = np.where(true_L == 1, R_yes, R_no)
        seq = np.concatenate([packed, np.full((B, 1), QUERY), readout[:, None]], axis=1)  # (B,T+2)
        vocab = 2 ** n + 3
    else:
        seq = packed
        vocab = 2 ** n
    return seq, vocab, {"logical_post": logical_post, "true_L": true_L, "qlast": qlast,
                        "R_yes": R_yes, "query_pos": T}


def main():
    device = train.get_device()
    results = {"eps": EPS, "gamma": GAMMA, "pA": PA, "pM": PM, "T": T, "by_n": {}}

    for n in [3, 5, 7]:
        r = (n - 1) // 2
        vocab = 2 ** n + 3
        rng = np.random.default_rng(n)
        erng = np.random.default_rng(1000 + n)
        seq_ev, _, meta = make_seqs(4096, n, erng, with_query=True)
        ev = torch.tensor(seq_ev, device=device)
        qpos = meta["query_pos"]; R_yes = meta["R_yes"]
        orac = meta["logical_post"]  # P(L=M)

        def batch_fn():
            s, _, _ = make_seqs(256, n, rng, with_query=True)
            t = torch.tensor(s, device=device); return t[:, :-1], t[:, 1:]

        def eval_fn(model):
            model.eval()
            with torch.no_grad():
                logits = model(ev[:, :-1])
                p = torch.softmax(logits[:, qpos], -1).cpu().numpy()  # at QUERY position
                p_yes = p[:, R_yes] / (p[:, R_yes] + p[:, R_yes - 1] + 1e-9)
            e = 1e-7; po = np.clip(orac, e, 1-e); pm = np.clip(p_yes, e, 1-e)
            kl = (po*np.log(po/pm) + (1-po)*np.log((1-po)/(1-pm))).mean()
            return {"kl_logical": float(kl)}

        torch.manual_seed(0)
        cfg = GPTConfig(vocab_size=vocab, n_ctx=T+2, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
        model = TinyGPT(cfg)
        out = train.train(model, batch_fn, steps=6000, lr=1e-3, device=device,
                          snapshot_steps=[6000], eval_fn=eval_fn, log_every=3000)
        model.eval()
        kl = out["snapshots"][6000]["metrics"]["kl_logical"]

        # C2c: error rates
        with torch.no_grad():
            p = torch.softmax(model(ev[:, :-1])[:, qpos], -1).cpu().numpy()
            p_yes = p[:, R_yes] / (p[:, R_yes] + p[:, R_yes - 1] + 1e-9)
        model_pred_L = (p_yes > 0.5).astype(int)
        model_logical_err = (model_pred_L != meta["true_L"]).mean()
        bayes_logical_err = np.minimum(orac, 1 - orac).mean()
        # physical: single-chain Bayes hard-decode error (avg over chains)
        ql = meta["qlast"]  # (B,n)
        # true per-chain state at last step:
        _, hid_chk, _ = (None, None, None)
        # recompute true chain states for physical error: regenerate not needed; use qlast vs realized
        # physical Bayes error = E[min(q,1-q)] per chain
        phys_bayes_err = np.minimum(ql, 1 - ql).mean()

        # C2b: probe logical posterior at QUERY position (query model)
        Xq = probe.extract_resid(model, ev, layer=cfg.n_layers - 1, pos=qpos)
        # logit-target for probe
        zt = np.log(np.clip(orac, 1e-4, 1-1e-4) / np.clip(1-orac, 1e-4, 1-1e-4))
        r2_logical = probe.ridge_probe(Xq, zt)["r2"]

        results["by_n"][n] = {
            "kl_logical": kl, "model_logical_err": float(model_logical_err),
            "bayes_logical_err": float(bayes_logical_err), "phys_bayes_err": float(phys_bayes_err),
            "binom_tail_qstar": active.binomial_tail(n, active.stationary_q(EPS, GAMMA), r),
            "r2_logical_query_model": r2_logical,
        }
        print(f"n={n}: KL_logical={kl:.4f}  model_err={model_logical_err:.3f} "
              f"bayes_logical={bayes_logical_err:.3f} phys={phys_bayes_err:.3f} "
              f"R2(logical)={r2_logical:.3f}", flush=True)
        if n == 5:
            results["fixed5_state"] = {k: v.cpu() for k, v in model.state_dict().items()}
            results["fixed5_cfg"] = cfg.__dict__

    # C2b contrast: no-query model at n=5, probe logical (should be LOW)
    n = 5; r = 2
    rng = np.random.default_rng(55)
    erng = np.random.default_rng(1055)
    seq_ev, vocab_nq, meta = make_seqs(4096, n, erng, with_query=False)
    ev = torch.tensor(seq_ev, device=device)
    orac = meta["logical_post"]
    zt = np.log(np.clip(orac, 1e-4, 1-1e-4) / np.clip(1-orac, 1e-4, 1-1e-4))
    def batch_fn():
        s, _, _ = make_seqs(256, n, rng, with_query=False)
        t = torch.tensor(s, device=device); return t[:, :-1], t[:, 1:]
    torch.manual_seed(1)
    cfg = GPTConfig(vocab_size=vocab_nq, n_ctx=T, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
    model_nq = TinyGPT(cfg)
    train.train(model_nq, batch_fn, steps=6000, lr=1e-3, device=device, log_every=6000)
    model_nq.eval()
    # probe at last packed position (T-1 in inputs of length T-1... use index T-2 in inputs[:, :-1])
    Xnq = probe.extract_resid(model_nq, ev, layer=cfg.n_layers - 1, pos=T - 1)
    r2_nq = probe.ridge_probe(Xnq, zt)["r2"]
    # also probe a single chain posterior in the no-query model (should be high: it tracks chains)
    qlast0 = meta["qlast"][:, 0]
    z_chain0 = np.log(np.clip(qlast0, 1e-4, 1-1e-4) / np.clip(1-qlast0, 1e-4, 1-1e-4))
    r2_chain_nq = probe.ridge_probe(Xnq, z_chain0)["r2"]
    results["no_query_n5"] = {"r2_logical": r2_nq, "r2_single_chain": r2_chain_nq}
    print(f"[no-query n=5] R2(logical)={r2_nq:.3f}  R2(single chain)={r2_chain_nq:.3f}", flush=True)

    torch.save(results, f"{OUT}/c2_logical.pt")
    print(f"saved {OUT}/c2_logical.pt")


if __name__ == "__main__":
    main()
