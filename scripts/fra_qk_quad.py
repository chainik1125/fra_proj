"""Find a QK quad: two features (l, m) active at BOTH a key pos k and query pos q.

Gives 4 boxes (l@k, m@k, l@q, m@q) and 4 arrows (the 2x2 query-feat x key-feat
interactions). attribution(alpha@q, beta@k) = z[q,alpha] * z[k,beta] * omega_{alpha,beta},
omega_{a,b} = sum_h (Wdec[a] W_Q^h).(Wdec[b] W_K^h)/sqrt(dh).  Also OV: l,m @ k -> IHY out.
Scores candidates by spread of the 4 arrows (so attribution is not just activation).
"""
from __future__ import annotations
import torch
from sleeper.eval import LN1_HOOK, PAT_HOOK
from sleeper.model import (MODELS, load_sleeper_model, cache_activations, load_paired_dataset)
from sleeper.sae import load as sae_load, encode_all

DEV = "cuda"; MODEL = "tinystories"; SAE = "weights/seeds_leftpad/sae_ln1_s0.pt"; T_ACT = 3.0


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV); tok = model.tokenizer
    sae, _ = sae_load(__import__("pathlib").Path(SAE), device=DEV)
    sel = load_paired_dataset(tok, n_train=2, n_val=64, n_test=2, seq_len=MODELS[MODEL].seq_len, seed=0, model=MODEL)["val"]
    di = int(torch.nonzero(sel.is_deployment)[0].item())
    toks = sel.tokens[di:di+1].to(DEV); mask = sel.attention_mask[di].bool()
    acts = cache_activations(model, sel.tokens[di:di+1], [LN1_HOOK, PAT_HOOK])
    z = encode_all(sae, acts[LN1_HOOK])[0].to(DEV).float()      # (T, d_sae)
    A = acts[PAT_HOOK][0].to(DEV).float()                       # (nh, T, T)
    real = torch.nonzero(mask).flatten().tolist()
    W_Q = model.W_Q[0].detach().float(); W_K = model.W_K[0].detach().float()
    W_V = model.W_V[0].detach().float(); W_O = model.W_O[0].detach().float()
    Wdec = sae.W_dec.detach().float(); dh = W_Q.shape[-1]
    FQ = torch.einsum("fd,hdk->fhk", Wdec, W_Q)                 # (d_sae, nh, dh)
    FK = torch.einsum("fd,hdk->fhk", Wdec, W_K)

    def omega(a, b):
        return float((FQ[a] * FK[b]).sum() / (dh ** 0.5))

    best = []
    for q in real:
        for k in real:
            if k > q:
                continue
            both = [f for f in range(z.shape[1]) if z[q, f] > T_ACT and z[k, f] > T_ACT]
            if len(both) < 2:
                continue
            both.sort(key=lambda f: -(float(z[q, f]) + float(z[k, f])))
            both = both[:6]
            for ii in range(len(both)):
                for jj in range(ii + 1, len(both)):
                    l, m = both[ii], both[jj]
                    A4 = {(l, l): float(z[q, l] * z[k, l]) * omega(l, l),
                          (l, m): float(z[q, l] * z[k, m]) * omega(l, m),
                          (m, l): float(z[q, m] * z[k, l]) * omega(m, l),
                          (m, m): float(z[q, m] * z[k, m]) * omega(m, m)}
                    vals = [abs(v) for v in A4.values()]
                    spread = max(vals) / (min(vals) + 1e-6)
                    best.append((max(vals) * (spread > 4), spread, q, k, l, m, A4))
    best.sort(key=lambda r: -r[0])
    print(f"[quad] q(last)={real[-1]} ; top candidates (need spread>4, all 4 boxes active):", flush=True)
    seen = set()
    for score, spread, q, k, l, m, A4 in best[:8]:
        key = (l, m)
        if key in seen:
            continue
        seen.add(key)
        print(f"\n[quad] q={q}({tok.decode([int(toks[0,q])])!r}) k={k}({tok.decode([int(toks[0,k])])!r}) "
              f"feats l={l} m={m}  spread={spread:.1f}", flush=True)
        print(f"   activations: l@q={float(z[q,l]):.2f} m@q={float(z[q,m]):.2f} | l@k={float(z[k,l]):.2f} m@k={float(z[k,m]):.2f}", flush=True)
        for (al, be), v in A4.items():
            print(f"   arrow  key f{be} -> query f{al}:  attr={v:+.3f}  (omega={omega(al,be):+.4f})", flush=True)
        # OV onto IHY for l,m at k
        W_U = model.W_U.detach().float()
        ids = [tok.encode(s)[0] for s in [" I", " HATE", " YOU"] if tok.encode(s)]
        d = W_U[:, ids].mean(1); d = d / d.norm()
        wr = torch.einsum("hmd,hdo->hmo", W_V, W_O)
        beta = lambda f: float(torch.einsum("hmo,m,o->", wr, Wdec[f], d))
        print(f"   OV(->IHY): l f{l}: act@k={float(z[k,l]):.2f} beta={beta(l):+.4f} attr={float(z[k,l])*beta(l):+.3f} | "
              f"m f{m}: act@k={float(z[k,m]):.2f} beta={beta(m):+.4f} attr={float(z[k,m])*beta(m):+.3f}", flush=True)


if __name__ == "__main__":
    main()
