"""Compute the FRA numbers behind the overview figure (``fig:overview_annotated`` /
``paper/figures/overview_quad_{bare,intv}_s1.tex``) for the TinyStories sleeper.

These are the activation/attribution values hand-transcribed into the TikZ node
fills and arrow widths/colours; this script regenerates them so they're
reproducible. For the first deployment steer example
(``data/steer_example_prompts.json[0]`` -- the Lily prompt, also the Fig-13 panel
example) at layer 0, key token ``"DE"`` and query token ``":"``, it reports for

    lambda = 1337  (the OV-winner feature, writes the sleeper via OV from DE)
    mu     = 435   (the only feature active on BOTH DE and ":" -> real DE<->: QK
                    self-interaction, plus a small OV write)
    nu     = 1269  (active on ":"; strongest QK partner of lambda)

  * per-position SAE feature activations            -> box shading,
  * QK feature-pair logit contributions (head 0)    -> QK arrow width/colour,
  * OV write FROM the DE key onto the attention-weighted difference-of-means
    suppressor direction v_md                       -> OV-section arrow width/colour.

Run: ``uv run -m scripts.overview_fra_numbers`` (needs weights/seeds/sae_ln1_s0.pt).
"""
from __future__ import annotations

import json

import torch

from sleeper.attribution import compute_ov_weights
from sleeper.eval import attn_weighted_vmd
from sleeper.model import MODELS, load_paired_dataset, load_sleeper_model
from sleeper.sae import load as sae_load

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RESID = "blocks.0.hook_resid_mid"
LN1, PAT = "blocks.0.ln1.hook_normalized", "blocks.0.attn.hook_pattern"
LAM, MU, NU = 1337, 435, 1269     # lambda (OV winner), mu (self-interactor), nu (query)
HEAD = 0                          # the head at which ":" reads the trigger


@torch.no_grad()
def main() -> None:
    m = load_sleeper_model(model="tinystories", device=DEV)
    tok = m.tokenizer
    sae, _ = sae_load("weights/seeds/sae_ln1_s0.pt", device=DEV)

    # v_md suppressor direction + per-(head, feature) OV weight beta = <W_dec.W_OV, v_md>
    sel = load_paired_dataset(tok, n_train=2, n_val=400, n_test=2,
                              seq_len=MODELS["tinystories"].seq_len, seed=0,
                              model="tinystories")["val"]
    vmd = attn_weighted_vmd(m, sel.tokens, sel.attention_mask, sel.is_deployment, RESID, DEV).float()
    beta = compute_ov_weights(m, sae, vmd / vmd.norm())["beta"].float()      # (n_heads, d_sae)

    prompt = json.load(open("data/steer_example_prompts.json"))[0]["prompt"]
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    strs = [tok.decode([i]) for i in ids]
    toks = torch.tensor([ids], device=DEV)
    _, cache = m.run_with_cache(toks, names_filter=lambda n: n in (PAT, LN1))
    A = cache[PAT][0].float()                       # (n_heads, T, T)
    z = sae.encode(cache[LN1][0].float())           # (T, d_sae)
    d_head = m.cfg.d_head
    KPOS = strs.index("DE"); QPOS = len(ids) - 1

    W_Q = m.W_Q[0].float(); W_K = m.W_K[0].float(); Wd = sae.W_dec.float()
    Qf = torch.einsum("fd,hde->hfe", Wd, W_Q)
    Kf = torch.einsum("fd,hde->hfe", Wd, W_K)
    def qkw(fq, fk):                                 # per-head feature-pair QK logit weight
        return float((Qf[HEAD, fq] * Kf[HEAD, fk]).sum() / d_head ** 0.5)
    name = {LAM: "lambda(1337)", MU: "mu(435)", NU: "nu(1269)"}

    print(f"key=DE pos{KPOS}={strs[KPOS]!r}  query=: pos{QPOS}={strs[QPOS]!r}  head={HEAD}")
    print("\n-- activations (node shading) --")
    for f in (LAM, MU, NU):
        print(f"  {name[f]:14s} z[DE]={float(z[KPOS, f]):6.3f}  z[:]={float(z[QPOS, f]):6.3f}")

    print("\n-- QK feature-pair contributions  z[DE,fk]*z[:,fq]*QKw(fq,fk)  (QK arrows) --")
    for nm, fk, fq in [("lam->mu", LAM, MU), ("lam->nu", LAM, NU),
                       ("mu->mu SELF", MU, MU), ("mu->nu", MU, NU)]:
        c = float(z[KPOS, fk]) * float(z[QPOS, fq]) * qkw(fq, fk)
        print(f"  {nm:12s} contrib={c:+.5f}")

    print("\n-- OV write from the DE key onto v_md  sum_h A[h,:,DE]*z[DE,f]*beta[h,f]  (OV arrows) --")
    for f in (LAM, MU, NU):
        ov = float((A[:, QPOS, KPOS] * z[KPOS, f] * beta[:, f]).sum())
        print(f"  {name[f]:14s} ov_DE={ov:+.5f}")


if __name__ == "__main__":
    main()
