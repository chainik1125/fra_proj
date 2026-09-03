"""Weight-sparse (circuit_sparsity) scoping measurements. PAUSED THREAD.

Reproduces every number in the scoping report. Requires the downloaded assets:

```text
../cs_data/models/csp_yolo1/{beeg_config.json,final_model.pt}   642 MB
../cs_data/models/csp_yolo2/{beeg_config.json,final_model.pt}  1676 MB
../cs_data/viz/yolo1_sdq_k1024_viz_data.pt                       96 MB
```

Fetch them with `fra.wsparse.loader.fetch` from the public Azure blob; nothing
here needs credentials. New deps: `blobfile`, `tiktoken`.

Run: python scripts/30_wsparse_scope.py
"""

from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

import torch

from circuit_sparsity.inference.hook_utils import hook_recorder
from fra.wsparse.fra_sparse import all_head_couplings, fra_qk_on_support, head_coupling, score_exact
from fra.wsparse.loader import load_circuit, load_model, qkv_slices

OUT = Path("results/wsparse_scope.json")
Y1 = "../cs_data/models/csp_yolo1"
Y2 = "../cs_data/models/csp_yolo2"
CIRC = "../cs_data/viz/yolo1_sdq_k1024_viz_data.pt"
LAYER = 10
HEAD = 82
B = 16


def spearman(x, y):
    rx = [sorted(x).index(v) + 1 for v in x]
    ry = [sorted(y).index(v) + 1 for v in y]
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


@torch.no_grad()
def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    m, cfg, info = load_model(Y1)
    res["yolo1"] = {
        "n_layer": cfg.n_layer, "n_head": cfg.n_head, "d_head": cfg.d_head,
        "d_model": cfg.d_model, "rms_norm": cfg.rms_norm,
        "sink": getattr(cfg, "sink", None), "afrac": cfg.afrac,
        "dropped_config_keys": info["dropped_config_keys"],
        "n_params": info["n_params"],
        "n_nonzero": sum(int((p != 0).sum()) for p in m.parameters()),
    }
    print(f"yolo1: {res['yolo1']['n_params']/1e6:.1f}M params, "
          f"{res['yolo1']['n_nonzero']/1e6:.2f}M nonzero, dropped {len(info['dropped_config_keys'])} cfg keys")

    circ = load_circuit(CIRC)
    toks = circ["importances"]["task_samples"][0]
    with hook_recorder() as rec:
        m(toks[:B])
    L, H, Dh, Dm = LAYER, cfg.n_head, cfg.d_head, cfg.d_model
    f = rec[f"{L}.attn.act_in"][:B]
    q, k = rec[f"{L}.attn.q"], rec[f"{L}.attn.k"]
    Wq, Wk, Wv = qkv_slices(m, L, cfg)
    bias = m.transformer.h[L].attn.c_attn.bias
    n = H * Dh
    bq, bk = bias[:n], bias[n:2 * n]

    res["bias"] = {"nnz": int((bias != 0).sum()), "numel": bias.numel(),
                   "max_abs": float(bias.abs().max())}

    # ── exactness: the identity everything rests on ─────────────────────
    err = 0.0
    cells = 0
    for hh in (0, 1, 5, HEAD, 120, 127):
        g = head_coupling(Wq, Wk, hh, Dh, L, bq, bk)
        for b in (0, 3):
            for qi, ki in ((222, 0), (200, 37), (150, 90), (37, 37)):
                got = score_exact(g, f[b, qi], f[b, ki])
                want = float((q[b, qi, hh*Dh:(hh+1)*Dh] * k[b, ki, hh*Dh:(hh+1)*Dh]).sum() / math.sqrt(Dh))
                err = max(err, abs(got - want))
                cells += 1
    res["exactness_max_abs_err"] = err
    res["exactness_cells"] = cells
    print(f"exactness: max abs err {err:.3e} over {cells} cells")

    # ── term decomposition + measurement (a) ────────────────────────────
    g = head_coupling(Wq, Wk, HEAD, Dh, L, bq, bk)
    circ_ch = set(circ["circuit_data"][f"{L}.attn.act_in"].tolist())
    inblk = torch.tensor([(int(a) in circ_ch) and (int(b) in circ_ch)
                          for a, b in zip(g.lam, g.mu)])
    T = f.shape[1]
    qi = T - 1
    P, Fq, Fk, tot, share = [], [], [], [], []
    for b in range(B):
        for ki in range(0, qi + 1, 7):
            pr = fra_qk_on_support(g, f[b, qi], f[b, ki])
            p, a, c = float(pr.sum()), float(f[b, qi] @ g.u_q), float(f[b, ki] @ g.u_k)
            P.append(abs(p)); Fq.append(abs(a)); Fk.append(abs(c))
            tot.append(abs(p + a + c + g.const))
            s = float(pr.abs().sum())
            share.append(float(pr[inblk].abs().sum() / s) if s > 0 else 0.0)
    terms = {"pair": st.mean(P), "feat_x_bias": st.mean(Fq),
             "bias_x_feat": st.mean(Fk), "const": abs(g.const), "total": st.mean(tot)}
    terms["pair_pct_of_term_mass"] = 100 * terms["pair"] / (
        terms["pair"] + terms["feat_x_bias"] + terms["bias_x_feat"] + terms["const"])
    res["head82_terms"] = terms
    res["measurement_a"] = {"mean_pct": 100 * st.mean(share),
                            "median_pct": 100 * st.median(share),
                            "n_cells": len(share),
                            "weights_only_pct": 0.42, "sleeper_pct": 2.14,
                            "support_pairs_both_retained": int(inblk.sum()),
                            "support_nnz": g.nnz}
    print(f"terms: pair {terms['pair']:.4f} biasxfeat {terms['bias_x_feat']:.4f} "
          f"-> pair is {terms['pair_pct_of_term_mass']:.1f}% of term mass")
    print(f"(a) circuit pair share: mean {100*st.mean(share):.3f}% median {100*st.median(share):.3f}%")

    # ── the correlation ─────────────────────────────────────────────────
    gs = all_head_couplings(Wq, Wk, H, Dh, L, bq, bk)
    attr_pair = torch.zeros(Dm)
    attr_bias = torch.zeros(Dm)
    for gg in gs:
        if gg.nnz == 0:
            continue
        for b in range(B):
            fq = f[b, qi, gg.lam]
            fk = f[b, :qi + 1][:, gg.mu]
            mass = ((fq * gg.val).unsqueeze(0) * fk).abs().sum(0)
            attr_pair.index_add_(0, gg.lam, mass)
            attr_pair.index_add_(0, gg.mu, mass)
            attr_bias += (f[b, qi] * gg.u_q).abs() + (f[b, :qi + 1] * gg.u_k).abs().sum(0)
    attr = attr_pair + attr_bias
    ch = circ["circuit_data"][f"{L}.attn.act_in"].tolist()
    gt = circ["importances"]["ch_interv_losses"][f"{L}.attn.act_in"]
    fra_all = [float(attr[c]) for c in ch]
    fra_pair = [float(attr_pair[c]) for c in ch]
    actmag = [float(f[:, :, c].abs().mean()) for c in ch]
    keep = [i for i, c in enumerate(ch) if c != 460]
    res["correlation"] = {
        "n": len(ch), "channels": ch, "ablation_loss": gt,
        "fra_all": fra_all, "fra_pair": fra_pair, "act_mag": actmag,
        "spearman_fra_all": spearman(fra_all, gt),
        "spearman_fra_pair": spearman(fra_pair, gt),
        "spearman_act_control": spearman(actmag, gt),
        "spearman_fra_excl460": spearman([fra_all[i] for i in keep], [gt[i] for i in keep]),
        "spearman_act_excl460": spearman([actmag[i] for i in keep], [gt[i] for i in keep]),
        "ch460_Wq_colnorm": float(Wq[:, 460].abs().sum()),
        "ch460_Wk_colnorm": float(Wk[:, 460].abs().sum()),
        "ch460_Wv_colnorm": float(Wv[:, 460].abs().sum()),
    }
    print(f"SPEARMAN n={len(ch)}: FRA {res['correlation']['spearman_fra_all']:+.3f}  "
          f"|act| {res['correlation']['spearman_act_control']:+.3f}  "
          f"(excl 460: {res['correlation']['spearman_fra_excl460']:+.3f} / "
          f"{res['correlation']['spearman_act_excl460']:+.3f})")

    # ── measurement (b): effective live pairs, yolo1 vs yolo2 ───────────
    def effective(model_dir, layer, heads, seed=0):
        mm, cc, _ = load_model(model_dir)
        torch.manual_seed(seed)
        tk = torch.randint(0, cc.vocab_size, (2, 223))
        with hook_recorder() as rr:
            mm(tk)
        ff = rr[f"{layer}.attn.act_in"]
        A, Bk, _ = qkv_slices(mm, layer, cc)
        out = []
        for h in heads:
            gg = head_coupling(A, Bk, h, cc.d_head, layer)
            if gg.nnz == 0:
                continue
            cnt = [int(((ff[b, a_, gg.lam] != 0) & (ff[b, b_, gg.mu] != 0)).sum())
                   for b in range(ff.shape[0])
                   for a_, b_ in ((222, 50), (222, 150), (180, 20), (100, 60))]
            out.append({"head": h, "nnz": gg.nnz, "effective": st.mean(cnt)})
        return cc, out

    _, e1 = effective(Y1, 10, [82, 1, 5, 40, 120])
    cfg2, e2 = effective(Y2, 4, [1, 3, 5, 6, 7])
    res["measurement_b"] = {
        "yolo1": e1, "yolo2": e2,
        "yolo1_mean_effective": st.mean([x["effective"] for x in e1]),
        "yolo2_mean_effective": st.mean([x["effective"] for x in e2]),
        "sleeper_pairs_per_cell": 1022,
        "yolo2_cfg": {"n_layer": cfg2.n_layer, "d_model": cfg2.d_model,
                      "afrac": cfg2.afrac, "sink": cfg2.sink},
    }
    print(f"(b) effective pairs/cell: yolo1 {res['measurement_b']['yolo1_mean_effective']:.1f}  "
          f"yolo2 {res['measurement_b']['yolo2_mean_effective']:.1f}  sleeper 1022")

    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
