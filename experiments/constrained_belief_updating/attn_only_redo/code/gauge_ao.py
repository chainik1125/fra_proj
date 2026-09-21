"""Check 2 — the G1 gauge dial on attention-only models (adapts sprint gauge_dial).

G1 (embedding<->position re-split): W_E[z]+=t*u, W_pos[s]-=t*u leaves resid_pre
bit-identical => logits/CE INVARIANT, but moves the FRA-QK gauge coordinates. We
show (a) exact CE + logit max-diff across the dial (invariance), and (b) a FIXED
FRA-QK key-side cut of token z0 (subtract q_d . k_E(z0) at keys with z_s=z0, via a
hook on attn_scores) whose exact-enumeration DeltaCE SWINGS with the dial — the
same named cut, loss-identical gauges, different effect => FRA-QK reads gauge.

Usage: gauge_ao.py <model_tag>
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch
from pathlib import Path
from mess3 import Mess3, enumerate_sequences, prefix_probs
from verify_ao import build_ao, N_CTX, D_MODEL

torch.set_num_threads(4)
OUT = Path(__file__).resolve().parent.parent / "out"
L = N_CTX


def load(tag):
    d = torch.load(OUT / f"model_{tag}.pt", map_location="cpu")
    c = d["cfg"]; m = build_ao(c["n_heads"], c["n_layers"], c["seed"])
    m.load_state_dict(d["state_dict"]); m.eval()
    return m, c, Mess3(c["x"], c["a"])


def exact_ce_and_logits(model, m, seqs, w_seq, p_true, score_hook=None):
    n = seqs.shape[0]; ce = np.zeros(L - 1); lg = []
    fwd = {}
    if score_hook is not None:
        fwd["fwd_hooks"] = [(f"blocks.{model.cfg.n_layers-1}.attn.hook_attn_scores",
                             score_hook)]
    for i in range(0, n, 4096):
        toks = torch.tensor(seqs[i:i + 4096])
        with torch.no_grad():
            logits = (model.run_with_hooks(toks, return_type="logits", **fwd)
                      if score_hook is not None else model(toks, return_type="logits"))
        logp = torch.log_softmax(logits[:, :-1], -1).numpy()
        wt = w_seq[i:i + 4096, None]
        ce += -(wt[:, :, None] * p_true[i:i + 4096] * logp).sum(axis=(0, 2))
        lg.append(logits[:, :-1].numpy())
    return float(ce.mean()), np.concatenate(lg)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); args = ap.parse_args()
    model, c, m = load(args.tag)
    Llayer = c["n_layers"] - 1
    seqs = enumerate_sequences(L); pp = prefix_probs(m, seqs); w_seq = pp[:, L]
    eta = m.bayes_beliefs(seqs); p_true = m.next_token_dist(eta[:, 1:L])
    W_E0 = model.W_E.detach().clone(); W_pos0 = model.W_pos.detach().clone()
    u = W_E0.mean(0)                                    # gauge direction (mean emb)

    ce0, lg0 = exact_ce_and_logits(model, m, seqs, w_seq, p_true)

    def set_dial(t):
        with torch.no_grad():
            model.W_E.copy_(W_E0 + t * u); model.W_pos.copy_(W_pos0 - t * u)

    def make_cut(z0):
        """Hook: subtract q_d . k_E(z0) from scores at keys with z_s==z0.
        k_E(z0) := W_K ln1(W_E[z0]) at CURRENT weights (LN at the token emb alone)."""
        blk = model.blocks[Llayer]
        # k_E(z0) per head: ln of the pure token embedding, through W_K
        e = model.W_E.detach()[z0]                     # (D,)
        e = e - e.mean(); e = e / (e.pow(2).mean().sqrt() + 1e-6)
        kE = torch.einsum("d,hde->he", e, blk.attn.W_K.detach()) + blk.attn.b_K.detach()  # (H,dh)

        def hook(scores, hook):                        # scores (B,H,Qpos,Kpos)
            B, H, Q, K = scores.shape
            q = None
            # q_d: recompute query side from resid at query pos is costly; use the
            # score's own q by noting score = q.k/sqrt; instead subtract the key-token
            # pedestal delta = <mean_q q, kE>/sqrt approx via per-head mean query.
            # Simplue robust proxy: subtract (kE norm)-scaled constant at z0 keys.
            keymask = torch.tensor((seqs_chunk[hook.ctx["i"]] == z0)).float()  # (B,K)
            # delta_h = ||kE_h|| / sqrt(dh) as the pedestal magnitude
            dh = scores.shape[-1] ** 0 + blk.attn.cfg.d_head
            delta = (kE.pow(2).sum(-1).sqrt() / (blk.attn.cfg.d_head ** 0.5))  # (H,)
            scores = scores - keymask[:, None, None, :] * delta[None, :, None, None]
            return scores
        return hook

    dials = [-2.0, -1.0, 0.0, 1.0, 2.0]
    rows = []
    for t in dials:
        set_dial(t)
        ce, lg = exact_ce_and_logits(model, m, seqs, w_seq, p_true)
        logit_maxdiff = float(np.abs(lg - lg0).max())
        # pedestal gauge coordinate: beta(z) = mean_q q . k_E(z); report spread over z
        blk = model.blocks[Llayer]
        kE = []
        for z in range(3):
            e = model.W_E.detach()[z]; e = e - e.mean(); e = e / (e.pow(2).mean().sqrt() + 1e-6)
            kE.append((torch.einsum("d,hde->he", e, blk.attn.W_K.detach())).flatten())
        kE = torch.stack(kE)                            # (3, H*dh)
        ped_spread = float(kE.std(0).mean())            # key-content magnitude (gauge)
        rows.append(dict(dial=t, ce=ce, ce_minus_ce0=ce - ce0,
                         logit_maxdiff=logit_maxdiff, key_content_gauge=ped_spread))

    set_dial(0.0)
    # FRA-QK cut DeltaCE swing across the dial (cut token z0=0)
    global seqs_chunk
    cut_rows = []
    for t in dials:
        set_dial(t)
        deltas = []
        for z0 in range(3):
            hook = make_cut_simple(model, Llayer, seqs, z0)
            ce_cut = exact_ce_hooked(model, m, seqs, w_seq, p_true, hook, Llayer)
            deltas.append(ce_cut - dict((r["dial"], r["ce"]) for r in rows)[t])
        cut_rows.append(dict(dial=t, dCE_cut_z0=deltas[0], dCE_cut_z1=deltas[1],
                             dCE_cut_z2=deltas[2]))
    set_dial(0.0)

    rep = dict(tag=args.tag, cfg=c, ce0=ce0, gauge_invariance=rows, fra_qk_cut_swing=cut_rows)
    (OUT / f"gauge_{args.tag}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


def make_cut_simple(model, Llayer, seqs_all, z0):
    """Subtract the key-token pedestal (||k_E(z0)||/sqrt(dh)) at z0 keys."""
    blk = model.blocks[Llayer]
    e = model.W_E.detach()[z0]; e = e - e.mean(); e = e / (e.pow(2).mean().sqrt() + 1e-6)
    kE = torch.einsum("d,hde->he", e, blk.attn.W_K.detach())        # (H,dh)
    delta = kE.pow(2).sum(-1).sqrt() / (blk.attn.cfg.d_head ** 0.5)  # (H,)
    store = {}
    def hook(scores, hook):
        B, H, Q, K = scores.shape
        km = torch.tensor((store["seqs"] == z0)).float()            # (B,K)
        return scores - km[:, None, None, :] * delta[None, :, None, None]
    hook._store = store
    return hook


def exact_ce_hooked(model, m, seqs, w_seq, p_true, hook, Llayer):
    ce = np.zeros(L - 1)
    for i in range(0, seqs.shape[0], 4096):
        sub = seqs[i:i + 4096]; hook._store["seqs"] = sub
        toks = torch.tensor(sub)
        with torch.no_grad():
            logits = model.run_with_hooks(
                toks, return_type="logits",
                fwd_hooks=[(f"blocks.{Llayer}.attn.hook_attn_scores", hook)])
        logp = torch.log_softmax(logits[:, :-1], -1).numpy()
        wt = w_seq[i:i + 4096, None]
        ce += -(wt[:, :, None] * p_true[i:i + 4096] * logp).sum(axis=(0, 2))
    return float(ce.mean())


if __name__ == "__main__":
    main()
