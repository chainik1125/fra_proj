"""Experiment 3+4 — SAE recovery on the reset-process BELIEF surface (the FRA surface).

Train a 1-layer softmax-attention reset-process model (Setting B), then train TopK
SAEs on (i) its raw input activations a_t and (ii) its POST-ATTENTION residual stream
h_t = W a_t + ctx_t (the model's belief estimate E[a_{t+1}|history]).

Q3: on the belief surface, does the SAE recover the d_i-aligned belief channels, or a
    contrast/mixture code (the Mess3 Finding-4 pathology)?  Measure chi vs the GT dict D.
Q4 (if non-identity): sever one belief channel through the learned code; compare
    removal/collateral against the planted-code (decoder=D) baseline.
"""
import json, time, sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from reset import ResetProcess, bayes_filter_mse, bayes_beliefs
from model import OneLayerAttn, train, eval_loss
from sae_common import train_sae, recovery_report, chi_matrix, mcc_hungarian

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "sae_exp3_reset.json")

N, d, T = 4, 32, 24
LAM, P, SIG = 0.7, 0.3, 0.5


def hidden(model, X):
    """Return (ctx, h_full) for input X (B,T,d): ctx = attention output,
    h_full = W a_t + ctx (pre-bias hidden = belief prediction)."""
    with torch.no_grad():
        A = model.pattern(X)
        Vx = X @ model.V.T
        ctx = torch.einsum('bhts,bsd->btd', A, Vx) / model.n_heads
        h_full = X @ model.W.T + ctx
    return ctx.numpy(), h_full.numpy()


def sever_through_code(sae, acts, D, gt_feature, planted=False):
    """Sever GT feature `gt_feature` and report removal of its d-content and
    collateral onto other d_j (both normalised by the feature's baseline content).
    planted=True: decoder=D oracle (subtract the d_gt projection).
    learned: zero every latent that (a) has gt_feature as its dominant GT OR (b) is
    the Hungarian match to gt_feature -- guarantees a non-empty, fair kill set."""
    x = torch.as_tensor(acts, dtype=torch.float32)
    Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
    proj0 = acts @ Dn[gt_feature]
    base0 = np.mean(np.abs(proj0))
    if planted:
        edited = acts - np.outer(proj0, Dn[gt_feature])
        killed = []
    else:
        with torch.no_grad():
            z = sae.encode(x)
        W = sae.W_dec.detach().numpy()
        Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
        S = np.abs(Wn @ Dn.T)
        dom = S.argmax(1)                               # dominant GT per latent
        kill = set(np.where(dom == gt_feature)[0].tolist())
        _, li, gi = mcc_hungarian(W, D)                 # Hungarian match
        for l, g in zip(li, gi):
            if g == gt_feature:
                kill.add(int(l))
        kill = sorted(kill)
        z2 = z.clone()
        z2[:, kill] = 0
        with torch.no_grad():
            edited = sae.decode(z2).numpy()
        killed = kill
    proj0_e = edited @ Dn[gt_feature]
    removal = 1 - np.mean(np.abs(proj0_e)) / max(base0, 1e-9)
    coll = [np.mean(np.abs((edited - acts) @ Dn[j]))
            for j in range(D.shape[0]) if j != gt_feature]
    collateral = float(np.mean(coll)) / max(base0, 1e-9)
    return dict(removal=float(removal), collateral=collateral,
                n_latents_killed=len(killed))


def run():
    t0 = time.time()
    gen = ResetProcess(N, d, lam=LAM, p=P, mu=1.0, sigma=SIG, seed=0, orthogonalize=True)
    model = OneLayerAttn(d, bias=True, seed=0, pos_key=True, n_ctx=T)
    losses = train(model, gen, T=T, steps=4000, batch=256, lr=3e-3)
    ev = eval_loss(model, gen, T=T)
    floors = bayes_filter_mse(gen, B=8000, T=T)
    print(f"model trained: eval {ev:.4f}  bayes {floors['mse_bayes']:.4f} "
          f"prior {floors['mse_prior']:.4f}  ({time.time()-t0:.0f}s)")

    # collect surfaces on fresh data (drop first few positions to avoid boundary)
    X, Y, c = gen.sample_seq(4000, T)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    ctx, h_full = hidden(model, Xt)
    Phat = bayes_beliefs(gen, c)                        # (B,T,N) posterior beliefs

    t_lo = 4
    def flat(arr):  # (B,T,·) -> (B*(T-t_lo), ·)
        return arr[:, t_lo:, :].reshape(-1, arr.shape[-1])
    a_flat = flat(X)
    h_flat = flat(h_full)
    ctx_flat = flat(ctx)
    z_gt = (flat(c) > 0).astype(float)                 # GT firing labels
    D = gen.D

    out = dict(model=dict(eval=ev, bayes=floors['mse_bayes'], prior=floors['mse_prior'],
                          N=N, d=d, T=T, lam=LAM, p=P, sigma=SIG))

    # SAE on each surface. K = true mean L0 of the surface's "active features".
    K = max(1, int(round(z_gt.sum(1).mean())))
    for surf_name, surf in [("input_a", a_flat), ("post_attn_h", h_flat),
                            ("attn_ctx", ctx_flat)]:
        best = None
        for L in (N, 2 * N):
            sae = train_sae(surf, L, K, steps=4000, batch=4096, seed=0)
            rep = recovery_report(sae, D, surf, z_gt)
            chi, offres = chi_matrix(sae.W_dec.detach().numpy(), D)
            rec = {k: v for k, v in rep.items() if not k.startswith("_")}
            rec["chi"] = np.round(chi, 3).tolist()
            entry = dict(L=L, report=rec)
            out.setdefault(surf_name, {})[f"L{L}"] = entry
            if best is None:
                best = (sae, rep)
            print(f"[{surf_name}] L={L} K={K}  MCC={rec['mcc']:.3f} "
                  f"uniq={rec['uniqueness']:.3f} F1={rec['f1']:.3f} "
                  f"offtarget={rec['offtarget_mean']:.3f} "
                  f"offdict={rec['offdict_res_mean']:.3f} "
                  f"sev_max={rec['sev_max']:.3f}  ({time.time()-t0:.0f}s)")
        out[surf_name]["_saved_L_N"] = N

    # --- belief-surface K-sensitivity: is the L=N contrast code a K=1 artifact? ---
    ksweep = {}
    for Kb in (1, 2, N):
        sae_k = train_sae(h_flat, N, Kb, steps=4000, batch=4096, seed=0)
        rep_k = recovery_report(sae_k, D, h_flat, z_gt)
        ksweep[f"K{Kb}"] = dict(mcc=rep_k["mcc"], uniqueness=rep_k["uniqueness"],
                                offtarget=rep_k["offtarget_mean"])
        print(f"[belief K-sweep] L={N} K={Kb}  MCC={rep_k['mcc']:.3f} "
              f"uniq={rep_k['uniqueness']:.3f} offtarget={rep_k['offtarget_mean']:.3f}")
    out["belief_K_sweep_L_N"] = ksweep

    # --- Experiment 4: severing feature 0 on the belief surface, learned vs planted.
    # Kb=K (naive GT-firing L0, gives the mixture code) and Kb=2 (recovered code). ---
    sev = {}
    for L in (N, 2 * N):
        for Kb in (K, 2):
            sae_h = train_sae(h_flat, L, Kb, steps=4000, batch=4096, seed=0)
            sev[f"learned_L{L}_K{Kb}"] = sever_through_code(sae_h, h_flat, D, 0, planted=False)
    sev["planted"] = sever_through_code(None, h_flat, D, 0, planted=True)
    out["sever_feature0_on_belief_surface"] = sev
    for k, v in sev.items():
        print(f"[sever f0 belief:{k}] removal={v['removal']:.3f} "
              f"collateral={v['collateral']:.3f} killed={v['n_latents_killed']}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", OUT, f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
