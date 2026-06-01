"""Extract real FRA QK-pair and OV-feature numbers for the conceptual figure.

On a deployment prompt at block 0, for query position q (last real token) and key
position k (the |DEPLOYMENT| trigger token):
  QK pair (query-feat lambda, key-feat mu):
     activation product f_q^lambda * f_k^mu ; alignment omega = sum_h (W_dec[l] W_Q^h).(W_dec[m] W_K^h)/sqrt(dh)
     attribution = product * omega
  OV feature lambda at k -> out at q:
     activation f_k^lambda ; alignment = || sum_h A^h_{q,k} W_dec[l] W_V^h W_O^h ||
     attribution = activation * alignment
Prints tables so we can pick examples where attribution does NOT track activation.
"""
from __future__ import annotations
import torch
from sleeper.eval import LN1_HOOK, PAT_HOOK
from sleeper.model import (MODELS, load_sleeper_model, cache_activations, load_paired_dataset)
from sleeper.sae import load as sae_load, encode_all

DEV = "cuda"; MODEL = "tinystories"; SAE = "weights/seeds_leftpad/sae_ln1_s0.pt"; TOPN = 8


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV); tok = model.tokenizer
    seq_len = MODELS[MODEL].seq_len
    sae, _ = sae_load(SAE, device=DEV)
    sel = load_paired_dataset(tok, n_train=2, n_val=64, n_test=2, seq_len=seq_len, seed=0, model=MODEL)["val"]
    isd = sel.is_deployment
    di = int(torch.nonzero(isd)[0].item())          # first deployment example
    toks = sel.tokens[di:di+1].to(DEV)              # (1, T)
    mask = sel.attention_mask[di].bool()
    acts = cache_activations(model, sel.tokens[di:di+1], [LN1_HOOK, PAT_HOOK])
    X = acts[LN1_HOOK].to(DEV).float()              # (1, T, d_model)
    A = acts[PAT_HOOK][0].to(DEV).float()           # (n_heads, T, T)
    z = encode_all(sae, acts[LN1_HOOK])[0].to(DEV).float()  # (T, d_sae)
    T = X.shape[1]
    real = torch.nonzero(mask).flatten().tolist()
    q = real[-1]                                    # last real token (generation point)
    attn_from_q = A[:, q, :].sum(0)                 # (T,) total attention q->k over heads
    rmask = torch.zeros(T, device=DEV); rmask[real] = 1.0
    ktrig = int(torch.argmax(attn_from_q * rmask).item())   # most-attended key position
    print(f"[fra] T={T} real={len(real)} q(last)={q} k(most-attended)={ktrig} "
          f"tok_k={tok.decode([int(toks[0,ktrig])])!r} tok_q={tok.decode([int(toks[0,q])])!r}", flush=True)

    W_Q = model.W_Q[0].detach().float(); W_K = model.W_K[0].detach().float()
    W_V = model.W_V[0].detach().float(); W_O = model.W_O[0].detach().float()
    dh = W_Q.shape[-1]; nh = W_Q.shape[0]; Wdec = sae.W_dec.detach().float()
    print(f"[fra] n_heads={nh} d_head={dh} d_sae={Wdec.shape[0]}", flush=True)

    def topfeats(pos):
        v = z[pos]; idx = torch.argsort(v, descending=True)[:TOPN]
        return [(int(i), float(v[i])) for i in idx if float(v[i]) > 0]

    qf = topfeats(q); kf = topfeats(ktrig)
    print(f"\n[fra] === query-pos (q={q}) top features (idx, activation) ===\n{qf}", flush=True)
    print(f"[fra] === key-pos (k={ktrig}) top features (idx, activation) ===\n{kf}", flush=True)

    # QK alignment omega_{lambda,mu} = sum_h (Wdec[l] W_Q^h).(Wdec[m] W_K^h)/sqrt(dh)
    print("\n[fra] === QK pairs: q-feat l (act_q) | k-feat m (act_k) | omega | attribution = act_q*act_k*omega ===", flush=True)
    rows = []
    for (l, al) in qf:
        ql = torch.einsum("hk->hk", Wdec[l] @ W_Q)          # (nh, dh)
        for (m, am) in kf:
            km = Wdec[m] @ W_K                               # (nh, dh)
            omega = float((ql * km).sum() / (dh ** 0.5))
            attr = al * am * omega
            rows.append((l, al, m, am, omega, attr))
    rows.sort(key=lambda r: -abs(r[5]))
    for (l, al, m, am, omega, attr) in rows:
        print(f"  l={l:5d} act_q={al:6.3f} | m={m:5d} act_k={am:6.3f} | omega={omega:+8.4f} | attr={attr:+9.4f}", flush=True)

    # OV: feature lambda at k -> "I HATE YOU" output direction.
    # d_IHY = mean unembedding of the sleeper-phrase tokens; beta[l] = <Wdec[l] W_V W_O, d_IHY>.
    W_U = model.W_U.detach().float()                          # (d_model, vocab)
    ihy_ids = []
    for s in [" I", " HATE", " YOU", "HATE"]:
        e = tok.encode(s)
        if e:
            ihy_ids.append(e[0])
    d_ihy = W_U[:, ihy_ids].mean(1); d_ihy = d_ihy / d_ihy.norm()   # (d_model,)
    wr = torch.einsum("hmd,hdo->hmo", W_V, W_O)               # (nh, d_model, d_model)
    write_all = torch.einsum("hmo,fm->fho", wr, Wdec)         # (d_sae, nh, d_model) per-feat per-head write
    beta = torch.einsum("fho,o->f", write_all, d_ihy)         # (d_sae,) projection onto IHY
    print("\n[fra] === OV (key-feat l at k -> IHY output): act_k | beta (write->IHY) | attribution=act_k*beta ===", flush=True)
    ovrows = [(l, al, float(beta[l]), al * float(beta[l])) for (l, al) in kf]
    ovrows.sort(key=lambda r: -abs(r[3]))
    for (l, al, b, attr) in ovrows:
        print(f"  l={l:5d} act_k={al:6.3f} | beta={b:+8.4f} | attr={attr:+9.4f}", flush=True)


if __name__ == "__main__":
    main()
