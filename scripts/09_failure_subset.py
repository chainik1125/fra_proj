"""Which sequences survive ablation, and is it the same ones at every rho?

Post-ablation accuracies repeat exactly across rho (801/1024 at rho=0.1 and 0.6;
825/1024 at 0.2 and 0.4). Identical counts mean the ablation fails on a specific
SUBSET rather than uniformly at random.

This is testable as a paired design, because the evaluation batch is literally
identical across rho: ``make_directions`` draws the same-shaped tensors whatever
rho is, so the generator state, the vocabulary and the sampled sequences are all
unchanged -- rho only alters the feature directions.

Grouped by:
  - content feature nu_c carried at the key position
  - the query/key position gap, a proxy for how much competing context there is
  - overlap of the failing SET across rho

Free to run: reuses the cached checkpoints.

Run: python scripts/09_failure_subset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from fra.toy.config import ToyConfig
from fra.toy.dgp import IDX_CONTENT_START
from fra.toy.fra import coupling_matrix
from fra.toy.intervention import ablate_pair
from fra.toy.train import TrainConfig, train_cached

RHOS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
EVAL_BATCH = 1024
OUT = Path("results/failure_subset.json")


@torch.no_grad()
def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    per_rho: dict[float, torch.Tensor] = {}
    batches = {}
    summary = []

    print("=" * 92)
    print("POST-ABLATION FAILURE STRUCTURE (planted arm, scale=1)")
    print("=" * 92)

    for rho in RHOS:
        cfg = ToyConfig(rho=rho)
        r = train_cached(cfg, TrainConfig(log=False))
        dgp, model = r.dgp, r.model
        lam, mu = dgp.planted_qk_edge
        b = dgp.sample(EVAL_BATCH, split="heldout")
        G = coupling_matrix(model, dgp.feature_directions)
        rows = torch.arange(EVAL_BATCH)

        with ablate_pair(model, b.features, G, lam, mu):
            pred = model(b.tokens).argmax(dim=-1)
        correct = pred[rows, b.query_pos] == b.targets[rows, b.query_pos]
        per_rho[rho] = correct
        batches[rho] = b

        gap = (b.query_pos - b.key_pos).float()
        n_ok = int(correct.sum())
        print(f"\nrho={rho:.2f}   {n_ok}/{EVAL_BATCH} correct after ablation "
              f"({n_ok/EVAL_BATCH*100:.2f}%)")

        # By content feature.
        by_content = []
        for c in range(cfg.n_content):
            m = b.content == c
            by_content.append(correct[m].float().mean().item() if m.any() else float("nan"))
        spread = max(by_content) - min(by_content)
        print(f"  by content nu_c : " + " ".join(f"{v*100:5.1f}" for v in by_content)
              + f"   spread {spread*100:.1f}pp")

        # By query/key gap.
        edges = [1, 3, 6, 12, 32]
        lo = 0
        cells = []
        for hi in edges:
            m = (gap > lo) & (gap <= hi)
            cells.append((f"{lo+1}-{hi}", correct[m].float().mean().item() if m.any() else float("nan"), int(m.sum())))
            lo = hi
        print("  by gap (q*-k*)  : " + "  ".join(f"{lab}:{v*100:5.1f}%(n={n})" for lab, v, n in cells))

        # Correlation of correctness with gap, as a single number.
        if correct.float().std() > 0:
            corr = torch.corrcoef(torch.stack([correct.float(), gap]))[0, 1].item()
        else:
            corr = float("nan")
        print(f"  corr(correct, gap) = {corr:+.3f}")

        summary.append({
            "rho": rho, "n_correct": n_ok, "by_content": by_content,
            "content_spread": spread, "corr_correct_gap": corr,
            "by_gap": [{"range": l, "acc": v, "n": n} for l, v, n in cells],
        })

    # Same sequences failing across rho?
    print("\n" + "=" * 92)
    print("IS IT THE SAME SUBSET ACROSS RHO?  (Jaccard of the FAILING sets)")
    print("=" * 92)
    same_tokens = all(torch.equal(batches[RHOS[0]].tokens, batches[r].tokens) for r in RHOS)
    print(f"  evaluation batch identical across rho: {same_tokens}")
    print(f"  {'':6s}" + "".join(f"{r:>8.2f}" for r in RHOS))
    jac = {}
    for a in RHOS:
        fa = ~per_rho[a]
        line = f"  {a:4.2f} "
        for c in RHOS:
            fc = ~per_rho[c]
            inter = int((fa & fc).sum())
            union = int((fa | fc).sum())
            j = inter / union if union else 1.0
            jac[f"{a}_{c}"] = j
            line += f"{j:8.2f}"
        print(line)

    OUT.write_text(json.dumps({"per_rho": summary, "jaccard": jac,
                               "batch_identical": same_tokens}, indent=2))
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
