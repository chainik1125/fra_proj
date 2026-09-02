"""The rho sweep: does FRA recovery degrade as ground-truth features overlap?

Stage A only (oracle features, no SAE). Retrains at every rho, because the data
geometry changes and a model trained at one overlap is not a model of another.

Admission criterion. At each rho we log held-out query accuracy and Gate 2
(mass on the mu* key, argmax_is_key) ALONGSIDE the FRA metrics. If model quality
also varies with rho, a declining recovery curve is uninterpretable: "FRA stopped
finding the edge" and "the model stopped cleanly learning the edge" look
identical. A rho point that fails Gate 2 is not evidence about FRA and is marked
as such in the output and the figure.

Primary metric is runner_up_ratio, not mass fraction. Mass fraction divides by an
L1 over every co-active pair and the non-planted entries of G carry ~97% of that
denominator; raising rho moves the denominator for reasons unrelated to the
planted edge. See fra/toy/recovery.py.

Run: python scripts/04_rho_sweep.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from fra.toy.config import ToyConfig
from fra.toy.fra import coupling_matrix, fra_qk_cell
from fra.toy.metrics import accuracy, attention_concentration, query_side_contrast
from fra.toy.recovery import qk_aggregate_closed_form, recovery_of
from fra.toy.train import TrainConfig, train

RHOS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
N_SEQ = 24
EVAL_BATCH = 1024
OUT = Path("results/sweep_rho.json")

# Gate 2 admission thresholds.
MIN_ARGMAX_IS_KEY = 0.95
MIN_HELDOUT_ACC = 0.90


def run_one(rho: float) -> dict:
    cfg = ToyConfig(rho=rho)
    t0 = time.time()
    r = train(cfg, TrainConfig(log=False))
    dgp, model = r.dgp, r.model
    lam, mu = dgp.planted_qk_edge

    # ── Admission criteria ──────────────────────────────────────────────
    b = dgp.sample(EVAL_BATCH, split="heldout")
    acc = accuracy(model, b)
    gate2 = attention_concentration(model, b)
    contrast = query_side_contrast(model, b)

    # ── FRA recovery ────────────────────────────────────────────────────
    G = coupling_matrix(model, dgp.feature_directions)
    agg = torch.zeros(cfg.n_feat, cfg.n_feat)
    cell_ranks, cell_mass, cell_runner = [], [], []

    for _ in range(N_SEQ):
        s = dgp.sample(1, split="heldout")
        f = s.features[0]
        q, k = int(s.query_pos[0]), int(s.key_pos[0])

        rec = recovery_of(fra_qk_cell(f, G, q, k), (lam, mu))
        cell_ranks.append(rec.rank_abs)
        cell_mass.append(rec.mass_fraction)
        cell_runner.append(rec.runner_up_ratio)

        agg += qk_aggregate_closed_form(f, G)

    rec_agg = recovery_of(agg, (lam, mu))
    rec_circuit = recovery_of(G.abs(), (lam, mu))

    admitted = (
        gate2.argmax_is_key >= MIN_ARGMAX_IS_KEY and acc.at_query >= MIN_HELDOUT_ACC
    )

    return {
        "rho_requested": rho,
        "rho_realized": dgp.rho_realized,
        "seconds": time.time() - t0,
        # admission
        "heldout_query_acc": acc.at_query,
        "heldout_elsewhere_acc": acc.elsewhere,
        "gate2_mass_on_key": gate2.mean_mass_on_key,
        "gate2_argmax_is_key": gate2.argmax_is_key,
        "query_side_mass_elsewhere": contrast.mass_elsewhere,
        "query_side_ratio": contrast.ratio,
        "admitted": admitted,
        # recovery: aggregate scope (primary)
        "agg_rank": rec_agg.rank_abs,
        "agg_mass": rec_agg.mass_fraction,
        "agg_runner_up": rec_agg.runner_up_ratio,
        "agg_nonzero": rec_agg.n_nonzero,
        # recovery: cell scope (precision given the model's own sparsity)
        "cell_rank1_frac": sum(1 for x in cell_ranks if x == 1) / N_SEQ,
        "cell_mass_mean": sum(cell_mass) / N_SEQ,
        "cell_runner_up_mean": sum(cell_runner) / N_SEQ,
        # recovery: circuit only, data-independent
        "circuit_rank": rec_circuit.rank_abs,
        "circuit_mass": rec_circuit.mass_fraction,
        "circuit_runner_up": rec_circuit.runner_up_ratio,
        "G_planted": G[lam, mu].item(),
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print("=" * 100)
    print("RHO SWEEP -- Stage A (oracle features, no SAE), single seed")
    print("=" * 100)
    hdr = (f"{'rho':>5s} {'real':>6s} | {'heldout':>8s} {'argmax':>7s} {'adm':>4s} | "
           f"{'agg_rank':>8s} {'agg_run':>8s} {'agg_mass':>8s} | "
           f"{'cell_r1':>7s} {'cell_run':>8s} | {'circ_run':>8s} {'G[l,m]':>8s}")
    print(hdr)
    print("-" * len(hdr))

    for rho in RHOS:
        row = run_one(rho)
        rows.append(row)
        print(
            f"{row['rho_requested']:5.2f} {row['rho_realized']:6.3f} | "
            f"{row['heldout_query_acc']*100:7.2f}% {row['gate2_argmax_is_key']*100:6.1f}% "
            f"{'yes' if row['admitted'] else 'NO':>4s} | "
            f"{row['agg_rank']:8d} {row['agg_runner_up']:8.2f} {row['agg_mass']:8.4f} | "
            f"{row['cell_rank1_frac']*100:6.0f}% {row['cell_runner_up_mean']:8.2f} | "
            f"{row['circuit_runner_up']:8.2f} {row['G_planted']:8.2f}"
        )
        OUT.write_text(json.dumps(rows, indent=2))

    print(f"\n  wrote {OUT}")
    n_adm = sum(1 for r in rows if r["admitted"])
    print(f"  admitted {n_adm}/{len(rows)} rho points")
    if n_adm < len(rows):
        failed = [r["rho_requested"] for r in rows if not r["admitted"]]
        print(f"  NOT admitted: rho={failed} -- their FRA numbers are not evidence")
        print("  about FRA, because the underlying circuit did not cleanly form.")


if __name__ == "__main__":
    main()
