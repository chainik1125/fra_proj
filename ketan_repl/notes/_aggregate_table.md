## Cross-seed AUC quality (n=3)

Seeds aggregated: seed0, seed1, seed2.

Each cell shows `mean(q) ± std(q)` across the 3 seeds. `q ∈ [0, 1]`, higher = better.

| ranking ＼ intervention | OV | QK | All |
|---|---:|---:|---:|
| **qk** | 0.404 ± 0.078 | 0.243 ± 0.096 | 0.459 ± 0.092 |
| **ov** | 0.770 ± 0.091 | 0.373 ± 0.232 | 0.789 ± 0.117 |
| **union** | 0.759 ± 0.044 | 0.295 ± 0.118 | 0.415 ± 0.019 |

### Per-seed quality (raw)

| ranking | intervention | seed0 | seed1 | seed2 |
|---|---|---:|---:|---:|
| qk | ov | 0.409 | 0.305 | 0.497 |
| qk | qk | 0.273 | 0.342 | 0.113 |
| qk | all | 0.394 | 0.589 | 0.396 |
| ov | ov | 0.867 | 0.649 | 0.795 |
| ov | qk | 0.701 | 0.220 | 0.197 |
| ov | all | 0.705 | 0.707 | 0.954 |
| union | ov | 0.801 | 0.778 | 0.699 |
| union | qk | 0.391 | 0.364 | 0.129 |
| union | all | 0.434 | 0.423 | 0.388 |

### Top-3 features picked per seed

| ranking | seed0 | seed1 | seed2 |
|---|---|---|---|
| qk | [1376, 891, 271] | [1250, 1445, 1158] | [101, 941, 970] |
| ov | [1376, 848, 1483] | [518, 179, 523] | [675, 1152, 1341] |
| union | [271, 848, 891, 1376, 1483] | [179, 518, 523, 1158, 1250, 1445] | [101, 675, 941, 970, 1152, 1341] |
