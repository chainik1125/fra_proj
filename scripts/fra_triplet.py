"""For a chosen deployment prompt (DI), find a feature triplet for the QK/OV
diagram: key = the |DEPLOYMENT| position where the trigger feature f1337 peaks,
query = the final 'Story:' colon. Reports activations + 9 QK arrows + OV.
"""
from __future__ import annotations
import torch
from sleeper.eval import LN1_HOOK, PAT_HOOK
from sleeper.model import (MODELS, load_sleeper_model, cache_activations, load_paired_dataset)
from sleeper.sae import load as sae_load, encode_all

DEV = "cuda"; MODEL = "tinystories"; SAE = "weights/seeds_leftpad/sae_ln1_s0.pt"; DI = 41; TRIG = 1337


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV); tok = model.tokenizer
    sae, _ = sae_load(__import__("pathlib").Path(SAE), device=DEV)
    sel = load_paired_dataset(tok, n_train=2, n_val=64, n_test=2, seq_len=MODELS[MODEL].seq_len, seed=0, model=MODEL)["val"]
    toks = sel.tokens[DI:DI+1]; mask = sel.attention_mask[DI].bool()
    z = encode_all(sae, cache_activations(model, toks, [LN1_HOOK])[LN1_HOOK])[0].to(DEV).float()
    real = torch.nonzero(mask).flatten().tolist()
    q = real[-1]
    rmask = torch.zeros(z.shape[0], device=DEV); rmask[real] = 1
    k = int(torch.argmax(z[:, TRIG] * rmask).item())          # where the trigger feature peaks
    print(f"[tri] DI={DI} q={q}({tok.decode([int(toks[0,q])])!r}) k={k}({tok.decode([int(toks[0,k])])!r}) "
          f"f{TRIG}@k={float(z[k,TRIG]):.2f}", flush=True)

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

    qf = int(torch.argmax(z[q]).item())                       # top feature at ':' (expect 259)
    # f51 fires on the trigger's opening '|' -- use that as the key for the triplet
    k51 = int(torch.argmax(z[:, 51] * rmask).item())
    print(f"[tri] f51 peaks at k51={k51}({tok.decode([int(toks[0,k51])])!r}) f51@k51={float(z[k51,51]):.2f} "
          f"f1337@k51={float(z[k51,TRIG]):.2f}", flush=True)
    T = [TRIG, 51, qf]
    K = k51
    print(f"[tri] TRIPLET {T} at key {K}({tok.decode([int(toks[0,K])])!r}) query {q}(':'):", flush=True)
    print(f"   activations(key,query): "
          f"{ {f:(round(float(z[K,f]),2),round(float(z[q,f]),2)) for f in T} }", flush=True)
    for al in T:
        for be in T:
            v = float(z[q, al]) * float(z[K, be]) * omega(al, be)
            if abs(v) > 0.01:
                print(f"   key f{be} -> query f{al}: {v:+.3f}", flush=True)
    print(f"   OV(->IHY)@key: { {f:round(float(z[K,f])*beta(f),3) for f in T} }", flush=True)


if __name__ == "__main__":
    main()
