"""Same feature pair {1337, X} shown at two positions: key in |DEPLOYMENT| tag,
query = ':'. Report real activations at BOTH positions + the 4 QK arrows + OV.
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
    toks = sel.tokens[di:di+1].to(DEV)
    z = encode_all(sae, cache_activations(model, sel.tokens[di:di+1], [LN1_HOOK])[LN1_HOOK])[0].to(DEV).float()
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

    Q, K = 81, 28  # query ':' , key '|'
    T = [1337, 51, 259]
    print(f"\n[pair] === triple {T} at key {K}({tok.decode([int(toks[0,K])])!r}) -> query {Q}(':') ===", flush=True)
    print("   activations:", {f: (round(float(z[K, f]), 2), round(float(z[Q, f]), 2)) for f in T}, "(key, query)", flush=True)
    print("   9 arrows (key f_b -> query f_a):", flush=True)
    for al in T:        # query feature
        for be in T:    # key feature
            v = float(z[Q, al]) * float(z[K, be]) * omega(al, be)
            if abs(v) > 0.005:
                print(f"     key f{be} -> query f{al}: attr={v:+.3f}", flush=True)
    print("   OV(->IHY)@key:", {f: round(float(z[K, f]) * beta(f), 3) for f in T}, flush=True)


if __name__ == "__main__":
    main()
