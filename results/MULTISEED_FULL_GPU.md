# Multi-seed robustness — FULL config on GPU

Full-config (headline) multi-seed run of C1 and C2, executed on an **NVIDIA A10G via Modal**
(HTTPS-only GPU path, `cloud/modal_gpu.py`). This upgrades the lean-CPU multi-seed
(`results/c1_multiseed.pt` / `c2_multiseed.pt`, run by the continuous session when SSH-to-GPU
was blocked) into the actual headline configuration.

- **Config:** C1 — L=200, 6000 steps, batch 256, 5 seeds. C2 — T=40, 6000 steps, batch 256,
  3 seeds × n∈{3,5,7}. Same architecture as headline (d=128, 3 layers, 4 heads, d_mlp=512).
- **Provenance:** Modal A10G, ~49 min wall, ~$0.9. Driver: `cloud/modal_gpu.py` (imports the
  committed `experiments/exp_multiseed_robustness.py` functions with `device="cuda"`, `L=200`).
- **Artifacts:** `results/c1_multiseed_full.pt`, `results/c2_multiseed_full.pt`.

## C1 — alignment log-odds (5 seeds)

| metric | mean ± sd |
|---|---|
| KL(model ‖ Bayes forward-filter) | **0.00009 ± 0.00003** |
| z-R² (alignment log-odds, linear decode from residual) | **0.9978 ± 0.0002** |
| count→z R² (running-count baseline, control) | **0.0074** |

The transformer reproduces the exact alignment-posterior forward filter (KL ≈ 1e-4), and the
alignment log-odds z is near-perfectly linearly decodable from the residual stream
(R² ≈ 0.998, sd 0.0002 across 5 seeds). A running-count baseline cannot recover it (R² ≈ 0.007),
so the learned representation is the *filter*, not a trivial token count. Seed-stable.

## C2 — majority decoder / error-correcting code (3 seeds per n)

| n | KL(model ‖ logical Bayes) | model logical err | Bayes logical err | physical per-chain err |
|---|---|---|---|---|
| 3 | 0.00319 ± 0.00075 | 0.1399 ± 0.0147 | 0.1441 | 0.1923 |
| 5 | 0.00474 ± 0.00287 | 0.1087 ± 0.0175 | 0.1000 | 0.1922 |
| 7 | 0.00265 ± 0.00035 | 0.0647 ± 0.0041 | 0.0698 | 0.1919 |

The model learns the Bayes-optimal majority decoder: model logical error tracks the Bayes logical
error, and both fall far below the **physical** per-chain Bayes error (≈ 0.192). Logical error
decreases with redundancy n (0.140 → 0.109 → 0.065 for n = 3 → 5 → 7), tracking the binomial-tail
suppression — the error-correcting-code result, now seed-averaged at full config.

> Note: a couple of cells show model err marginally *below* Bayes err (n=5,7). This is finite-eval
> sampling noise on the 4096-sequence held-out set (min(p,1−p) estimated on a finite sample), not a
> genuine beat of the optimum.

## vs lean-CPU multi-seed

Full config is tighter than the lean run (C1 KL ~7e-4 → 9e-5; z-R² ~0.99 → 0.998; count-control
~0.025 → 0.007), as expected from L=200 (more decode positions) and 6000 steps. The qualitative
story is identical and seed-stable in both — the multi-seed caveat is now a result, at headline config.
