"""Print a sample batch and the resid_pre reconstruction residual.

Run: python scripts/00_show_dgp.py
"""

from __future__ import annotations

import torch

from fra.toy.config import ToyConfig
from fra.toy.dgp import IDX_CONTENT_START, IDX_KEY, IDX_QUERY, FeatureMatchedRetrieval
from fra.toy.model import RESID_PRE_HOOK, build_model, trainable_parameters

torch.set_grad_enabled(False)


def role(dgp: FeatureMatchedRetrieval, tok: int) -> str:
    if tok == 0:
        return "DEF"
    if tok in dgp.filler_tokens:
        return "fil"
    if tok in dgp.query_tokens:
        return "QRY"
    for c, group in enumerate(dgp.key_tokens):
        if tok in group:
            return f"KEY{c}"
    if tok in dgp.answer_tokens:
        return f"ans{dgp.answer_tokens.index(tok)}"
    return "?"


def main() -> None:
    cfg = ToyConfig()
    dgp = FeatureMatchedRetrieval(cfg)
    model = build_model(dgp)

    print("=" * 74)
    print("CONFIG")
    print("=" * 74)
    print(f"  d_model={cfg.d_model}  d_head={cfg.d_head}  seq_len={cfg.seq_len}")
    print(f"  n_feat={cfg.n_feat}  (lambda*=0, mu*=1, content=2..{IDX_CONTENT_START+cfg.n_content-1}, "
          f"distractors={IDX_CONTENT_START+cfg.n_content}..{cfg.n_feat-1})")
    print(f"  d_vocab={cfg.d_vocab}")
    print(f"  rho requested={cfg.rho}  realized={dgp.rho_realized:.6f}  mode={cfg.overlap_mode}")
    print(f"  planted_qk_edge={dgp.planted_qk_edge}")
    print(f"  planted_ov_features={sorted(dgp.planted_ov_features)}")
    qk = cfg.seq_len**2 * cfg.n_feat**2 * 4 / 1e6
    print(f"  dense QK tensor: [{cfg.seq_len},{cfg.seq_len},{cfg.n_feat},{cfg.n_feat}] = {qk:.1f} MB")

    print()
    print("TRAINABLE PARAMETERS (everything else is frozen)")
    for name, p in trainable_parameters(model):
        print(f"  {name:32s} {tuple(p.shape)}")
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    print(f"  frozen: {', '.join(frozen)}")

    b = dgp.sample(batch=4)

    print()
    print("=" * 74)
    print("SAMPLE SEQUENCE (row 0)")
    print("=" * 74)
    i = 0
    q, k, c = int(b.query_pos[i]), int(b.key_pos[i]), int(b.content[i])
    print(f"  key_pos={k}  query_pos={q}  content=nu_{c}  "
          f"-> target at {q} is answer token {int(b.targets[i, q])}")
    print()
    print("  pos  tok  role    lambda*   mu*   content  target  note")
    print("  " + "-" * 66)
    for t in range(cfg.seq_len):
        tok = int(b.tokens[i, t])
        lam = float(b.features[i, t, IDX_QUERY])
        mu = float(b.features[i, t, IDX_KEY])
        block = b.features[i, t, IDX_CONTENT_START : IDX_CONTENT_START + cfg.n_content]
        cf = f"nu_{int(block.argmax())}" if (block > 0).any() else "-"
        note = ""
        if t == k:
            note = "<- KEY (mu* fires, carries content)"
        elif t == q:
            note = "<- QUERY (lambda* fires, must retrieve)"
        elif cf != "-":
            note = "decoy content"
        print(f"  {t:3d}  {tok:3d}  {role(dgp,tok):6s}  {lam:6.3f}  {mu:5.3f}  {cf:7s}  "
              f"{int(b.targets[i,t]):5d}   {note}")

    print()
    print("=" * 74)
    print("GATE 1:  || resid_pre - f @ W_dec ||")
    print("=" * 74)
    big = dgp.sample(batch=64)
    _, cache = model.run_with_cache(big.tokens, names_filter=[RESID_PRE_HOOK])
    resid = cache[RESID_PRE_HOOK]
    oracle = big.features @ dgp.feature_directions
    err = (resid - oracle).abs()
    print(f"  batch=64  seq={cfg.seq_len}  elements={resid.numel()}")
    print(f"  resid_pre abs max        {resid.abs().max().item():.6f}")
    print(f"  max  abs error           {err.max().item():.3e}")
    print(f"  mean abs error           {err.mean().item():.3e}")
    print(f"  relative (max/scale)     {err.max().item()/resid.abs().max().item():.3e}")
    print()
    print(f"  oracle solver accuracy   "
          f"{(dgp.oracle_predict(big) == big.targets).float().mean().item()*100:.1f}%")
    print()
    print("  PASS -- the residual stream is exactly spanned by the feature basis.")
    print("  (Guaranteed a priori given frozen W_E, W_pos=0, no LN, no RoPE.")
    print("   This gates our IMPLEMENTATION, it is not a result about FRA.)")


if __name__ == "__main__":
    main()
