"""Is FRA-guided ablation BETTER than conventional steering, or just specific?

The intervention work showed the FRA-identified pair is causally load-bearing and
that matched-random and runner-up pairs are not. That is specificity, not
superiority -- "better than what?" was never answered, and the FRA paper's whole
Pareto framing rests on that comparison.

Three interventions on the same target, nested by construction:

  1. fra_pair   subtract scale * f[q,l*] f[k,m*] G[l*,m*] from the QK scores
  2. qkv        remove alpha * f[t,l*] W_dec[l*] from the input to W_Q/W_K/W_V
                (the paper's "QK->QK"; hits Q, K AND V, broader than 1)
  3. residual   subtract alpha * f[t,l*] W_dec[l*] from the residual stream
                (the conventional SAE-feature ablation; broader still, since the
                edit also survives on the skip connection into resid_post)

Pareto axes, the toy analogue of the paper's Figure 2:

  x = collateral: mean KL(base || intervened) at NON-target positions. LEFT = less damage.
  y = target: accuracy at lambda* query positions.                     DOWN = more suppression.

So LOWER-LEFT is better. The accuracy form of collateral (accuracy ELSEWHERE) is
also recorded, but it SATURATES: the label away from the query is one constant
token and the model predicts it robustly enough that a 15-logit perturbation does
not flip the argmax, so that axis reads ~0 damage for every method at every
strength. KL is what gives the axis dynamic range. The claim under test: "FRA-guided pair ablation
suppresses the planted behaviour at lower collateral cost than conventional
steering." If the frontiers overlap, that is a null and worth knowing before
Case 2 is built on the assumption.

rho in {0, 0.4} only -- the defensible range from the multi-seed sweep. rho=0.8
rests on one lucky seed.

Run: python scripts/10_pareto.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fra.toy.config import ToyConfig
from fra.toy.fra import coupling_matrix
from fra.toy.intervention import ablate_pair
from fra.toy.metrics import accuracy, attention_concentration
from fra.toy.steering import ablate_feature_qkv, measure, steer_residual
from fra.toy.train import TrainConfig, train_cached

RHOS = [0.0, 0.4]
SEEDS = [0, 1, 2]
STRENGTHS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0]
EVAL_BATCH = 1024
OUT = Path("results/pareto.json")
FIG = Path("results/figures/pareto.png")

METHOD_STYLE = {
    "fra_pair": ("#1f77b4", "o-", "FRA pair ablation (QK, one pair)"),
    "qkv": ("#ff7f0e", "s--", "activation ablation, Q/K/V  (paper's QK->QK)"),
    "residual": ("#d62728", "^:", "residual steering (conventional SAE ablation)"),
}


def run_rho(rho: float, seed: int) -> dict:
    cfg = ToyConfig(rho=rho, seed=seed)
    r = train_cached(cfg, TrainConfig(log=False))
    dgp, model = r.dgp, r.model
    lam, mu = dgp.planted_qk_edge
    W_dec = dgp.feature_directions
    G = coupling_matrix(model, W_dec)

    b = dgp.sample(EVAL_BATCH, split="heldout")
    base_logits = model(b.tokens)
    base_acc = accuracy(model, b)
    base_conc = attention_concentration(model, b)

    print(f"\nrho={rho:.2f} seed={seed}  baseline: query {base_acc.at_query*100:.2f}%  "
          f"elsewhere {base_acc.elsewhere*100:.2f}%  "
          f"Gate2 {base_conc.argmax_is_key*100:.1f}%  chance {100/cfg.n_content:.1f}%")

    points = []
    for s in STRENGTHS:
        with ablate_pair(model, b.features, G, lam, mu, s):
            points.append(measure(model, b, "fra_pair", s, base_logits))
        with ablate_feature_qkv(model, b.features, W_dec, lam, s):
            points.append(measure(model, b, "qkv", s, base_logits))
        with steer_residual(model, b.features, W_dec, lam, s):
            points.append(measure(model, b, "residual", s, base_logits))

    for method in METHOD_STYLE:
        print(f"\n  {method}")
        print(f"    {'strength':>8s} {'query':>8s} {'suppress':>9s} | "
              f"{'acc_else':>9s} {'coll_acc':>9s} | {'coll_KL':>10s} {'frac':>6s} {'maxdlog':>8s}")
        for p in [x for x in points if x.method == method]:
            print(f"    {p.strength:8.2f} {p.acc_query*100:7.2f}% "
                  f"{p.suppression(base_acc.at_query)*100:+8.2f} | "
                  f"{p.acc_elsewhere*100:8.2f}% {p.collateral(base_acc.elsewhere)*100:+8.2f} | "
                  f"{p.collateral_kl:10.3e} {p.collateral_frac:6.3f} {p.collateral_max_dlogit:8.3f}")

    return {
        "rho": rho,
        "seed": seed,
        "chance": 1.0 / cfg.n_content,
        "base_query": base_acc.at_query,
        "base_elsewhere": base_acc.elsewhere,
        "base_argmax_is_key": base_conc.argmax_is_key,
        "points": [vars(p) for p in points],
    }


def plot(rows: list[dict]) -> None:
    FIG.parent.mkdir(parents=True, exist_ok=True)
    rhos = sorted({r["rho"] for r in rows})
    # Stacked, not side-by-side: two panels across a page's text width halves
    # every label. One panel per row keeps the axes legible in print.
    fig, axes = plt.subplots(len(rhos), 1, figsize=(9.0, 5.4 * len(rhos)), squeeze=False)
    axes = axes[:, 0]

    for ax, rho in zip(axes, rhos):
        group = [r for r in rows if r["rho"] == rho]
        for method, (color, style, label) in METHOD_STYLE.items():
            for i, row in enumerate(group):
                pts = [p for p in row["points"] if p["method"] == method]
                ax.plot([max(p["collateral_kl"], 1e-12) for p in pts],
                        [p["acc_query"] * 100 for p in pts],
                        style, color=color, lw=2 if i == 0 else 1.1,
                        ms=6 if i == 0 else 3.5,
                        alpha=0.9 if i == 0 else 0.45,
                        label=label if i == 0 else None)
        row = group[0]
        ax.scatter([1e-12], [row["base_query"] * 100],
                   marker="*", s=260, color="#333", zorder=5, label="no intervention")
        ax.set_xscale("symlog", linthresh=1e-9)
        ax.axhline(row["chance"] * 100, color="#888", ls=":", lw=1.2)
        ax.text(ax.get_xlim()[0], row["chance"] * 100 + 1.2, " chance",
                fontsize=8, color="#666", va="bottom")

        ax.set_xlabel("collateral:  mean KL at non-target positions  ->  LEFT = less damage")
        ax.set_ylabel("target:  accuracy at $\\lambda^*$ queries (%)  ->  DOWN = more suppression")
        ax.set_title(f"$\\rho$ = {rho}   ({len(group)} seeds; seed 0 bold)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
        ax.annotate("better", xy=(0.04, 0.05), xycoords="axes fraction",
                    fontsize=11, color="#2a7", weight="bold")

    fig.suptitle("Steering Pareto frontier: does FRA-guided pair ablation buy anything?\n"
                 "lower-LEFT is better -- suppresses the planted rule, spares everything else",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.5)
    fig.savefig(FIG, dpi=160)
    print(f"\n  wrote {FIG}")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 92)
    print("PARETO: FRA pair ablation vs conventional steering baselines")
    print("=" * 92)
    rows = [run_rho(rho, seed) for rho in RHOS for seed in SEEDS]
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\n  wrote {OUT}")

    # Claim A is algebraic: a score-row edit cannot reach outside {positions >= q*}.
    # If it fails on any seed, the mechanism argument is wrong and that matters
    # more than the frontier.
    print("\n" + "=" * 92)
    print("CLAIM A CHECK -- does zero collateral hold EXACTLY on every seed?")
    print("=" * 92)
    ok = True
    for row in rows:
        for method in METHOD_STYLE:
            pts = [p for p in row["points"] if p["method"] == method and p["strength"] > 0]
            mx = max(p["collateral_kl"] for p in pts)
            frac = max(p["collateral_frac"] for p in pts)
            if method == "fra_pair":
                good = mx == 0.0 and frac == 0.0
                ok = ok and good
                print(f"  rho={row['rho']:.2f} seed={row['seed']}  {method:9s} "
                      f"max KL {mx:.3e}  max frac {frac:.3f}   "
                      f"{'EXACTLY ZERO' if good else '<-- NON-ZERO, mechanism wrong'}")
            else:
                print(f"  rho={row['rho']:.2f} seed={row['seed']}  {method:9s} "
                      f"max KL {mx:.3e}  max frac {frac:.3f}")
    print(f"\n  CLAIM A holds exactly on all {len(rows)} runs: {ok}")

    # Claim B side: the strength FRA needs is what should vary across seeds.
    print("\n  strength needed for >=99% suppression (the side expected to vary):")
    for row in rows:
        cells = []
        for method in METHOD_STYLE:
            hit = [p for p in row["points"] if p["method"] == method
                   and (row["base_query"] - p["acc_query"]) >= 0.99 * row["base_query"]]
            cells.append(f"{method}={min(p['strength'] for p in hit) if hit else float('nan'):>5}")
        print(f"    rho={row['rho']:.2f} seed={row['seed']}  " + "  ".join(cells))

    plot(rows)


if __name__ == "__main__":
    main()
