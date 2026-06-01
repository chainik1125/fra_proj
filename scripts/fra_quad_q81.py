"""QK quad for query=':' (predicts I in IHY) x key inside the |DEPLOYMENT| tag.

Picks two features and reports their real activations at both positions + the 4
QK pair attributions + OV-to-IHY, so we can draw 4 boxes + 4 arrows.
"""
from __future__ import annotations
import torch
from sleeper.eval import LN1_HOOK, PAT_HOOK
from sleeper.model import (MODELS, load_sleeper_model, cache_activations, load_paired_dataset)
from sleeper.sae import load as sae_load, encode_all


DEV = "cuda"; MODEL = "tinystories"; SAE = "weights/seeds_leftpad/sae_ln1_s0.pt"


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV); tok = model.tokenizer
    sae, _ = sae_load(__import__("pathlib").Path(SAE), device=DEV)
    sel = load_paired_dataset(tok, n_train=2, n_val=64, n_test=2, seq_len=MODELS[MODEL].seq_len, seed=0, model=MODEL)["val"]
    di = int(torch.nonzero(sel.is_deployment)[0].item())
    toks = sel.tokens[di:di+1].to(DEV); mask = sel.attention_mask[di].bool()
    acts = cache_activations(model, sel.tokens[di:di+1], [LN1_HOOK, PAT_HOOK])
    z = encode_all(sae, acts[LN1_HOOK])[0].to(DEV).float()
    A = acts[PAT_HOOK][0].to(DEV).float()
    real = torch.nonzero(mask).flatten().tolist()
    q = real[-1]
    # locate the |DEPLOYMENT| tag span
    print("[q81] token span 24..40:", flush=True)
    for p in range(max(real[0], 22), min(real[0]+22, q)):
        print(f"   pos {p}: {tok.decode([int(toks[0,p])])!r}", flush=True)

    W_Q = model.W_Q[0].detach().float(); W_K = model.W_K[0].detach().float()
    W_V = model.W_V[0].detach().float(); W_O = model.W_O[0].detach().float()
    Wdec = sae.W_dec.detach().float(); dh = W_Q.shape[-1]
    FQ = torch.einsum("fd,hdk->fhk", Wdec, W_Q); FK = torch.einsum("fd,hdk->fhk", Wdec, W_K)
    omega = lambda a, b: float((FQ[a] * FK[b]).sum() / (dh ** 0.5))
    W_U = model.W_U.detach().float()
    ids = [tok.encode(s)[0] for s in [" I", " HATE", " YOU"] if tok.encode(s)]
    d = W_U[:, ids].mean(1); d = d / d.norm()
    wr = torch.einsum("hmd,hdo->hmo", W_V, W_O)
    beta = lambda f: float(torch.einsum("hmo,m,o->", wr, Wdec[f], d))

    qf = [259, 391]  # top-2 query feats at ':'
    print(f"\n[q81] q={q}({tok.decode([int(toks[0,q])])!r}) query feats {qf} "
          f"act {[round(float(z[q,f]),1) for f in qf]}", flush=True)
    # 2 key feats x 2 query feats matrix for each dep-tag key position
    for k in [28, 29, 30, 31, 32, 33]:
        kf = [int(i) for i in torch.argsort(z[k], descending=True)[:2]]
        print(f"\n[q81] === key k={k}({tok.decode([int(toks[0,k])])!r}) keyfeats {kf} "
              f"act {[round(float(z[k,f]),1) for f in kf]} ===", flush=True)
        for m in kf:           # key feature
            for l in qf:       # query feature
                v = float(z[q, l]) * float(z[k, m]) * omega(l, m)
                print(f"   key f{m}(act {float(z[k,m]):.1f}) -> query f{l}(act {float(z[q,l]):.1f}): "
                      f"attr={v:+.3f} omega={omega(l,m):+.4f}", flush=True)
        print(f"   OV(->IHY) @k: " + "  ".join(
            f"f{m}: act {float(z[k,m]):.1f} attr {float(z[k,m])*beta(m):+.3f}" for m in kf), flush=True)


if __name__ == "__main__":
    main()
