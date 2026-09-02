"""Oracle FRA at rho = 0: does FRA recover the planted (lambda*, mu*) edge?

Stage A only -- ground-truth directions as W_dec, ground-truth activations as f,
no SAE anywhere. Reports brief section 6 metrics 1-4 at both scopes.

Run: python scripts/02_oracle_fra.py
"""

from __future__ import annotations

import torch

from fra.toy.config import ToyConfig
from fra.toy.fra import OracleFRA
from fra.toy.metrics import accuracy, attention_concentration
from fra.toy.recovery import (
    ov_cell_recovery,
    ov_retrieval_precision,
    qk_aggregate,
    qk_cell_recovery,
    recovery_of,
)
from fra.toy.train import TrainConfig, train

N_SEQ = 24  # sequences aggregated; each dense 4-D tensor is ~41 MB


def main() -> None:
    cfg = ToyConfig()
    print("=" * 78)
    print(f"ORACLE FRA  (Stage A, no SAE)   rho={cfg.rho} ({cfg.overlap_mode}) "
          f" n_feat={cfg.n_feat}  d_head={cfg.d_head}")
    print("=" * 78)

    r = train(cfg, TrainConfig(log=False))
    dgp, model = r.dgp, r.model
    lam, mu = dgp.planted_qk_edge

    b = dgp.sample(512, split="heldout")
    acc = accuracy(model, b)
    gate2 = attention_concentration(model, b)
    print(f"  model (held-out): {acc}")
    print(f"  Gate 2:           {gate2}")
    print(f"  trained in {r.seconds:.0f}s")

    # ── Exactness, restated on this trained model ───────────────────────
    one = dgp.sample(1, split="heldout")
    fra = OracleFRA(model, dgp.feature_directions, one.tokens[0], one.features[0])
    finite = torch.isfinite(fra.scores)
    qk_err = (fra.qk().sum(dim=(2, 3))[finite] - fra.scores[finite]).abs().max().item()
    print(f"\n  GATE 3 exactness: max |FRA_QK.sum() - score| = {qk_err:.3e}")

    # ── Metric 1-3: the planted QK pair ─────────────────────────────────
    print()
    print("=" * 78)
    print(f"METRICS 1-3  --  planted QK edge (lambda*, mu*) = ({lam}, {mu})")
    print("=" * 78)

    cell_ranks, cell_stats = [], []
    agg = torch.zeros(cfg.n_feat, cfg.n_feat)
    agg_signed = torch.zeros(cfg.n_feat, cfg.n_feat)
    for i in range(N_SEQ):
        s = dgp.sample(1, split="heldout")
        fra_i = OracleFRA(model, dgp.feature_directions, s.tokens[0], s.features[0])
        qk = fra_i.qk()
        q, k = int(s.query_pos[0]), int(s.key_pos[0])
        rec = qk_cell_recovery(qk, q, k, (lam, mu))
        cell_ranks.append(rec.rank_abs)
        cell_stats.append(rec)
        agg += qk.abs().sum(dim=(0, 1))
        agg_signed += qk.sum(dim=(0, 1))

    r0 = cell_stats[0]
    print(f"\n  SCOPE 'cell' -- at the planted (q*, k*), the brief's literal metric")
    print(f"    rank of (lambda*, mu*)   {r0.rank_abs}  (signed rank {r0.rank_signed})")
    print(f"    mass fraction            {r0.mass_fraction:.4f}")
    print(f"    candidates ranked        {r0.n_nonzero} non-zero of {r0.n_total}")
    print(f"    rank==1 over {N_SEQ} seqs   "
          f"{sum(1 for x in cell_ranks if x == 1)}/{N_SEQ}")
    print(f"    mean mass fraction       "
          f"{sum(s.mass_fraction for s in cell_stats)/N_SEQ:.4f}")
    print("    NOTE: f is sparse, so a cell holds only |active(q)|x|active(k)|")
    print("          non-zero pairs. Rank 1 here is rank 1 of ~12, not of 10,000.")

    rec_agg = recovery_of(agg, (lam, mu))
    print(f"\n  SCOPE 'aggregate' -- sum_{{q,k}} |FRA_QK| over {N_SEQ} sequences,")
    print(f"  ranked against all n_feat^2 pairs (comparable to Dmitry's")
    print(f"  qk_pair_concentration.json, whose real-model ratio was 0.074)")
    print(f"    rank of (lambda*, mu*)   {rec_agg.rank_abs}  "
          f"(signed rank {rec_agg.rank_signed})")
    print(f"    mass fraction            {rec_agg.mass_fraction:.4f}")
    print(f"    candidates ranked        {rec_agg.n_nonzero} non-zero of {rec_agg.n_total}")
    print(f"    uniform baseline         {1.0/rec_agg.n_total:.2e}  "
          f"-> {rec_agg.mass_fraction*rec_agg.n_total:.0f}x baseline")
    print(f"    in top 5 / top 20        {rec_agg.in_top5} / {rec_agg.in_top20}")

    # L1 vs signed aggregation -- the only honest way to compare the two
    # rankings, since ranking inside an already-absolute aggregate is trivially
    # identical under both orderings.
    rec_signed = recovery_of(agg_signed, (lam, mu))
    degenerate = (agg_signed.abs() - agg).abs().max().item()
    print(f"\n  L1 vs SIGNED aggregation")
    print(f"    L1     aggregation: rank {rec_agg.rank_abs:<5d} mass {rec_agg.mass_fraction:.4f}  "
          f"total_l1={rec_agg.total_l1:.1f}")
    print(f"    signed aggregation: rank {rec_signed.rank_abs:<5d} mass {rec_signed.mass_fraction:.4f}  "
          f"total_signed={rec_signed.total_signed:.1f}")
    print(f"    max | |signed| - L1 | over pairs = {degenerate:.3e}")
    print("    These are IDENTICAL up to a per-pair sign, and not by coincidence:")
    print("    activations are non-negative, so sign(FRA_QK[q,k,l,m]) = sign(G[l,m])")
    print("    for every (q,k). The signed sum over positions is therefore +/- the L1")
    print("    sum, and the two rankings agree TAUTOLOGICALLY. At the aggregate scope")
    print("    this setting CANNOT adjudicate Dmitry's signed-vs-abs finding.")
    cancel = 1.0 - abs(rec_signed.total_signed) / rec_agg.total_l1
    print(f"\n    Cancellation BETWEEN DIFFERENT (l,m) pairs: {cancel*100:.2f}% of L1 mass")
    print("    (different feature pairs push attention in opposite directions;")
    print("     this is across pairs, not within one, so it does not affect ranking)")
    print("\n    The cell scope CAN adjudicate -- there the competing entries are")
    print(f"    different (l,m) with different signs of G. There: rank_abs="
          f"{r0.rank_abs}, rank_signed={r0.rank_signed}.")

    top = torch.topk(agg.flatten(), 5)
    print("\n    top 5 pairs by |aggregate|:")
    for v, idx in zip(top.values, top.indices):
        i, j = int(idx) // cfg.n_feat, int(idx) % cfg.n_feat
        mark = "  <- PLANTED" if (i, j) == (lam, mu) else ""
        print(f"      ({i:3d}, {j:3d})  {v.item():.4f}{mark}")

    # ── Metric 4: the planted OV features ───────────────────────────────
    print()
    print("=" * 78)
    print(f"METRIC 4  --  planted OV set = {sorted(dgp.planted_ov_features)}")
    print("=" * 78)
    hits, ranks, masses = 0, [], []
    for i in range(N_SEQ):
        s = dgp.sample(1, split="heldout")
        fra_i = OracleFRA(model, dgp.feature_directions, s.tokens[0], s.features[0])
        ov = fra_i.ov()
        q, k = int(s.query_pos[0]), int(s.key_pos[0])
        true_feat = 2 + int(s.content[0])
        rec = ov_cell_recovery(ov, q, k, true_feat)
        ranks.append(rec.rank_abs)
        masses.append(rec.mass_fraction)
        hits += ov_retrieval_precision(ov, q, k, dgp.planted_ov_features)
    print(f"    rank of the true content feature nu_c: rank==1 in "
          f"{sum(1 for x in ranks if x == 1)}/{N_SEQ} sequences")
    print(f"    retrieval precision (top-1 in planted set): {hits}/{N_SEQ}")
    print(f"    mean mass fraction: {sum(masses)/N_SEQ:.4f}")

    print()
    ok = rec_agg.rank_abs == 1
    print(f"  {'PASS' if ok else 'FAIL'} -- planted pair is rank "
          f"{rec_agg.rank_abs} of {rec_agg.n_total} in the aggregate scope.")


if __name__ == "__main__":
    main()
