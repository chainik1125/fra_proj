"""
C2 SPREADING bag: does the transformer still learn the OPTIMAL majority decoder when the
n redundant alignment chains' errors are CORRELATED (spread-coupled), and does the
error-correction (logical < physical) survive as correlation rises?

Motivation. §4.3 shows the model learns to majority-decode when the n chains are INDEPENDENT
(beta=0). §5 shows real EM errors are CORRELATED across models (cross-finetune rho~0.4), which
is exactly the regime where re-sampling/voting should help LESS. This experiment closes the §7
"spread threshold is process-level" gap by testing the TRANSFORMER under spread coupling, against
an EXACT oracle.

Generator: gen_redundant_active(beta>0) -- a chain's corruption rate is eps + beta*(fraction of
OTHER chains currently misaligned), so the joint hidden state is a coupled 2^n-state Markov chain.
The per-chain filter + poisson-binomial tail is NO LONGER the exact logical posterior; we build the
EXACT 2^n-state joint forward filter and anchor everything to it.

Tests (per beta / per induced cross-chain correlation):
  V   joint filter calibration: empirical P(logical=Misaligned | filtered posterior bin) tracks bin.
  A   QUERY model matches the EXACT joint logical oracle at the QUERY position (KL -> 0?).
  B   model logical error vs Bayes-(joint)-logical error vs single-chain physical error.
  C   the logical<physical GAP (error-correction benefit) vs the induced cross-chain correlation.

Outputs results/c2_spread.pt
"""
import os, sys, time
import numpy as np
import torch
ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from bag_moments import active, train
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", os.path.join(ROOT, "results"))
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))

# Metastable regime: slow correction (gamma) + rare spontaneous corruption (eps) so that the
# spread coupling produces genuine COLLECTIVE (cross-chain correlated) misalignment events.
EPS, GAMMA, PA, PM = 0.01, 0.05, 0.3, 0.7
N = int(os.environ.get("SPREAD_N", "5"))
T = int(os.environ.get("SPREAD_T", "80"))
STEPS = int(os.environ.get("SPREAD_STEPS", "2500"))
BATCH = int(os.environ.get("SPREAD_BATCH", "192"))
R = (N - 1) // 2
S = 1 << N  # 2^n joint hidden states


# ---------------------------------------------------------------------------
# Exact 2^n-state joint forward filter for the spread-coupled redundant bag
# ---------------------------------------------------------------------------
def _bits(s, n):
    return [(s >> i) & 1 for i in range(n)]

POP = np.array([bin(s).count("1") for s in range(S)])  # popcount per joint state


def build_joint_T(n, eps, gamma, beta):
    """Joint transition T[s, s'] over 2^n states. Chain i flips A->M w.p.
    clip(eps + beta*frac_other_i, 0,1), M->A w.p. gamma, independently given s."""
    Smax = 1 << n
    T = np.zeros((Smax, Smax))
    for s in range(Smax):
        sb = _bits(s, n)
        pop = sum(sb)
        # per-chain P(next bit = 1) given current joint state s
        p1 = np.zeros(n)
        for i in range(n):
            frac_other = (pop - sb[i]) / max(n - 1, 1)
            if sb[i] == 0:  # aligned -> misaligned w.p. eps_eff
                p1[i] = min(max(eps + beta * frac_other, 0.0), 1.0)
            else:           # misaligned -> stays misaligned w.p. 1-gamma
                p1[i] = 1.0 - gamma
        # product distribution over next joint state
        for sp in range(Smax):
            spb = _bits(sp, n)
            pr = 1.0
            for i in range(n):
                pr *= p1[i] if spb[i] == 1 else (1.0 - p1[i])
            T[s, sp] = pr
    return T


def build_emission_table(n, pA, pM):
    """Eobs[o, s] = P(observe bit-vector o | joint state s), o,s in [0,2^n)."""
    Smax = 1 << n
    E = np.ones((Smax, Smax))
    for o in range(Smax):
        ob = _bits(o, n)
        for s in range(Smax):
            sb = _bits(s, n)
            pr = 1.0
            for i in range(n):
                p_emit1 = pM if sb[i] == 1 else pA
                pr *= p_emit1 if ob[i] == 1 else (1.0 - p_emit1)
            E[o, s] = pr
    return E


def stationary(T, iters=2000):
    Smax = T.shape[0]
    pi = np.full(Smax, 1.0 / Smax)
    for _ in range(iters):
        pi = pi @ T
    return pi / pi.sum()


def joint_filter(packed_obs, n, eps, gamma, pA, pM, beta):
    """Exact joint forward filter. packed_obs: (B, L) ints in [0,2^n).
    Returns logical_post (B, L) = P(majority misaligned | obs_{1:t})."""
    B, L = packed_obs.shape
    Smax = 1 << n
    Tm = build_joint_T(n, eps, gamma, beta)
    Eo = build_emission_table(n, pA, pM)
    pi = stationary(Tm)
    maj = (POP[:Smax] > R).astype(np.float64)  # states that are majority-misaligned
    logical = np.empty((B, L))
    bel = pi[None, :] * Eo[packed_obs[:, 0]]      # (B,S)
    bel /= bel.sum(1, keepdims=True)
    logical[:, 0] = bel @ maj
    for t in range(1, L):
        pred = bel @ Tm                           # (B,S)
        bel = pred * Eo[packed_obs[:, t]]
        bel /= bel.sum(1, keepdims=True)
        logical[:, t] = bel @ maj
    return logical


# ---------------------------------------------------------------------------
def validate_filter(beta, rng):
    """V: calibration of the joint filter -- P(true logical=M | filtered post bin) ~ bin centre."""
    tk, hid, logical = active.gen_redundant_active(2048, 160, N, EPS, GAMMA, PA, PM, rng, beta=beta)
    packed = active.pack_emissions(tk)
    post = joint_filter(packed, N, EPS, GAMMA, PA, PM, beta)
    tl = logical[:, 40:].reshape(-1); pp = post[:, 40:].reshape(-1)
    rows = []
    for lo in np.arange(0.0, 1.0, 0.1):
        m = (pp >= lo) & (pp < lo + 0.1)
        if m.sum() > 200:
            rows.append((lo + 0.05, tl[m].mean(), int(m.sum())))
    maxerr = max(abs(c - e) for c, e, _ in rows) if rows else 1.0
    return rows, maxerr


def cross_chain_corr(hidden):
    """Mean pairwise correlation of chains' hidden (misaligned) indicators, 2nd half."""
    h = hidden[:, hidden.shape[1] // 2:, :].reshape(-1, hidden.shape[2]).astype(np.float64)
    C = np.corrcoef(h.T)
    n = C.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.nanmean(C[iu]))


def make_seqs(B, rng, beta, with_query=True):
    tk, hid, logical = active.gen_redundant_active(B, T, N, EPS, GAMMA, PA, PM, rng, beta=beta)
    packed = active.pack_emissions(tk)                       # (B,T)
    post = joint_filter(packed, N, EPS, GAMMA, PA, PM, beta)  # (B,T) exact P(L=M)
    true_L = logical[:, -1]                                   # (B,)
    QUERY = S; R_no = S + 1; R_yes = S + 2
    if with_query:
        readout = np.where(true_L == 1, R_yes, R_no)
        seq = np.concatenate([packed, np.full((B, 1), QUERY), readout[:, None]], axis=1)
    else:
        seq = packed
    return seq, {"logical_post": post[:, -1], "true_L": true_L, "R_yes": R_yes,
                 "query_pos": T, "hidden": hid, "tokens": tk}


def run_beta(beta, seed=0, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); erng = np.random.default_rng(7)
    vocab = S + 3
    seq_ev, meta = make_seqs(3072, erng, beta, with_query=True)
    seq_ev_t = torch.tensor(seq_ev, device=device)
    qpos = meta["query_pos"]; R_yes = meta["R_yes"]
    orac = meta["logical_post"]                              # P(L=M) exact joint
    corr = cross_chain_corr(meta["hidden"])

    def batch_fn():
        s, _ = make_seqs(BATCH, rng, beta, with_query=True)
        t = torch.tensor(s, device=device); return t[:, :-1], t[:, 1:]

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            logits = model(seq_ev_t[:, :-1])
            p = torch.softmax(logits[:, qpos], -1).cpu().numpy()
        pm = np.clip(p[:, R_yes], 1e-6, 1 - 1e-6); po = np.clip(orac, 1e-6, 1 - 1e-6)
        kl = (po * np.log(po / pm) + (1 - po) * np.log((1 - po) / (1 - pm))).mean()
        return {"kl_logical": float(kl)}

    cfg = GPTConfig(vocab_size=vocab, n_ctx=T + 2, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
    model = TinyGPT(cfg)
    t0 = time.time()
    out = train.train(model, batch_fn, steps=STEPS, lr=1e-3, device=device,
                      snapshot_steps=[STEPS], eval_fn=eval_fn, log_every=max(STEPS // 2, 1))
    kl = out["snapshots"][STEPS]["metrics"]["kl_logical"]
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(seq_ev_t[:, :-1])[:, qpos], -1).cpu().numpy()
    pm = np.clip(p[:, R_yes], 1e-6, 1 - 1e-6)
    model_pred_L = (pm > 0.5).astype(int)
    model_logical_err = float((model_pred_L != meta["true_L"]).mean())
    bayes_logical_err = float(np.minimum(orac, 1 - orac).mean())

    # The NAIVE INDEPENDENT decoder (the §4.2 binomial-tail picture): poisson-binomial tail of the
    # per-chain posteriors -- WRONG under spread because it ignores cross-chain coupling.
    qc = active.per_chain_filter(meta["tokens"], EPS, GAMMA, PA, PM)[:, -1, :]   # (B,n) per-chain post
    p_indep = np.clip(active.poisson_binomial_tail(qc, R), 1e-6, 1 - 1e-6)        # (B,) indep P(L=M)
    indep_logical_err = float((( (p_indep > 0.5).astype(int)) != meta["true_L"]).mean())
    po = np.clip(orac, 1e-6, 1 - 1e-6)
    # KL of the model's logical posterior to the JOINT oracle vs to the INDEPENDENT decoder
    kl_to_joint = float((po*np.log(po/pm) + (1-po)*np.log((1-po)/(1-pm))).mean())
    kl_to_indep = float((p_indep*np.log(p_indep/pm) + (1-p_indep)*np.log((1-p_indep)/(1-pm))).mean())
    # how far the naive decoder is from the truth-optimal joint decoder (Brier on truth)
    tl = meta["true_L"].astype(float)
    brier_joint = float(((orac - tl)**2).mean()); brier_indep = float(((p_indep - tl)**2).mean())

    # physical (single-chain) Bayes error from the per-chain filter
    qchain = qc[:, 0]
    phys_bayes_err = float(np.minimum(qchain, 1 - qchain).mean())
    res = {"beta": beta, "cross_chain_corr": corr, "kl_logical": kl,
           "model_logical_err": model_logical_err, "bayes_logical_err": bayes_logical_err,
           "indep_logical_err": indep_logical_err, "phys_bayes_err": phys_bayes_err,
           "kl_to_joint": kl_to_joint, "kl_to_indep": kl_to_indep,
           "brier_joint": brier_joint, "brier_indep": brier_indep,
           "chain_misrate": float(meta["hidden"][:, T//2:, :].mean()), "n": N}
    print(f"[beta={beta:.3f} corr={corr:+.3f}] KL_logical={kl:.4f} | model_err={model_logical_err:.3f} "
          f"joint_bayes={bayes_logical_err:.3f} indep_dec={indep_logical_err:.3f} phys={phys_bayes_err:.3f}\n"
          f"          model->joint KL={kl_to_joint:.4f} vs model->indep KL={kl_to_indep:.4f} | "
          f"Brier joint={brier_joint:.4f} indep={brier_indep:.4f} ({(time.time()-t0)/60:.1f}min)", flush=True)
    return res


def main():
    rng = np.random.default_rng(0)
    # R_M = beta*(n-1)/(n*gamma); pick betas spanning increasing correlation
    betas = [float(b) for b in os.environ.get("SPREAD_BETAS", "0.0,0.10,0.20").split(",")]
    print(f"=== C2 spreading bag (n={N}, eps={EPS}, gamma={GAMMA}, r={R}) ===", flush=True)
    print("Validating joint filter calibration per beta...", flush=True)
    val = {}
    for b in betas:
        rows, maxerr = validate_filter(b, rng)
        status = "OK" if maxerr < 0.03 else "CHECK"
        val[b] = {"rows": rows, "maxerr": maxerr}
        print(f"  [{status}] beta={b:.3f}: joint-filter calibration max|emp-bin|={maxerr:.3f}", flush=True)
    results = [run_beta(b) for b in betas]
    torch.save({"results": results, "validation": val,
                "config": {"n": N, "eps": EPS, "gamma": GAMMA, "pA": PA, "pM": PM, "T": T,
                           "steps": STEPS, "betas": betas}},
               f"{OUT}/c2_spread.pt")
    print(f"saved {OUT}/c2_spread.pt", flush=True)


if __name__ == "__main__":
    main()
