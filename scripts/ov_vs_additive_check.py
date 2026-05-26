"""Mechanism isolation: take the SAME ln1-SAE feature (OV-diff winner) and steer
it two ways — through OV (W_V -> hook_v) vs additive at ln1 — across k×width.
Holds SAE + feature + selection fixed; only the intervention differs. Tells us
whether the conventional cell's k-degradation is the additive *mechanism* or the
*selection*. Runs on seed-0's already-trained ln1 checkpoints (no retraining).
"""
from __future__ import annotations

import torch

from scripts.eval_checkpoint import (
    DEFAULT_ALPHAS, SUPPRESS_THRESH, _binary_search_onset, build_consumer_state)
from sleeper.hooks import additive_steer_hook, compute_sae_delta
from sleeper.jsd_cells import _gen, eval_ov, jsd_mean
from sleeper.metrics import asr_16
from sleeper.sae import load as sae_load
from sleeper.screen import LN1_HOOK, screen_winner_ov

CKPT = "/workspace/sae_scaling_out/sae_checkpoints/ln1/seed0/d%d_k%d/step50000.pt"


def opt_jclean(curve):
    supp = [c[0] for c in curve.values() if c[4] <= SUPPRESS_THRESH]
    return min(supp) if supp else None


def sweep(cell):
    curve = {a: cell(a) for a in DEFAULT_ALPHAS}
    pos = [a for a in DEFAULT_ALPHAS if a > 0]
    if pos and min(curve[a][4] for a in pos) > SUPPRESS_THRESH:
        curve.update(_binary_search_onset(cell, lo=float(max(pos))))
    return curve


@torch.no_grad()
def additive_ln1(model, refs, feature, alpha, sae, device):
    tok = model.tokenizer
    if alpha == 0.0:
        return (jsd_mean(refs.poisoned_lsm, refs.clean_lsm), 0.0, 0, 0,
                asr_16(refs.poisoned_tokens.cpu(), tok))
    delta = compute_sae_delta(model, sae, LN1_HOOK, int(feature), refs.dep_lp,
                              refs.dep_attn, attention_mask=refs.dep_attn)
    st, lsm = _gen(model, refs.dep_lp, refs.dep_attn,
                   additive_steer_hook(delta, alpha, LN1_HOOK), device)
    return (jsd_mean(lsm, refs.clean_lsm), jsd_mean(lsm, refs.poisoned_lsm), 0, 0,
            asr_16(st.cpu(), tok))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from sleeper.model import load_sleeper_model
    model = load_sleeper_model(device=dev)
    state = build_consumer_state(model, dev)
    print("d_sae  k | feat |  OV opt_Jcln | additive-ln1 opt_Jcln", flush=True)
    for d in (1536, 3072, 6144):
        for k in (10, 32, 50):
            sae, _ = sae_load(CKPT % (d, k), device=dev)
            feat = screen_winner_ov(model, sae, state.sel, dev)["winner"]
            ov = sweep(lambda a: eval_ov(model, state.refs, [feat], a, sae, dev))
            ad = sweep(lambda a: additive_ln1(model, state.refs, feat, a, sae, dev))
            oj, aj = opt_jclean(ov), opt_jclean(ad)
            print("%5d %3d | %4d |  %11s | %s" % (
                d, k, feat,
                ("%.3f" % oj) if oj is not None else "no-supp",
                ("%.3f" % aj) if aj is not None else "no-supp"), flush=True)


if __name__ == "__main__":
    main()
