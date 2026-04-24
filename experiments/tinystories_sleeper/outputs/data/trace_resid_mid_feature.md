# Target-feature trace summary

This report traces a chosen SAE feature as the target variable. Head and upstream effects are scored by the change they induce in the target feature's `preactivation` at the selected prompt position.

## Target

- target SAE: `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/crosscoder_sae_layer1.pt`
- target hook: `blocks.0.hook_resid_mid`
- feature idx: `171`
- analyzed deployment prompts: `64` from split `val`
- target-position fallback to preactivation used on `2` prompts
- mean target preactivation at selected positions: `2.2412`
- mean target activation at selected positions: `2.2531`

Top target-position tokens by target preactivation weight:

- `|` (id=91): `144.1983`
- `:` (id=25): `-0.7602`

## Skip vs Attention

| quantity | value |
|---|---:|
| mean resid_pre linear score at target pos | -0.2673 |
| mean attention linear score at target pos | 2.5191 |
| mean head-sum-plus-bias score at target pos | 2.5191 |
| mean target preactivation at target pos | 2.2412 |
| mean abs attention decomposition error | 0.000000 |

## Heads

| head | mean linear contrib | mean causal drop | top positive source tokens |
|---:|---:|---:|---|
| 12 | +0.7794 | +0.7794 | `MENT` (31.805), ` |` (7.518), `:` (4.756), `PL` (2.861) |
| 9 | +0.2663 | +0.2663 | `|` (16.112), `PL` (0.747), ` |` (0.197), `MENT` (0.044) |
| 7 | +0.2471 | +0.2471 | ` |` (12.371), `|` (2.099), `DE` (0.647), `MENT` (0.376) |
| 3 | +0.1655 | +0.1655 | `MENT` (8.725), ` |` (1.724), `Story` (0.170), `Features` (0.011) |
| 5 | +0.1609 | +0.1609 | ` |` (8.409), `|` (0.745), `PL` (0.738), `MENT` (0.295) |
| 8 | +0.1221 | +0.1221 | `MENT` (6.006), ` |` (1.566), `Story` (0.099), `OY` (0.060) |

## Upstream Features

Upstream SAE: `/root/fra_proj/experiments/tinystories_sleeper/recreate_layer0_fresh_20260421_052220/results/crosscoder_sae_layer0.pt` at `blocks.0.hook_resid_pre`. Candidates are ranked by source-weighted activation under the strongest heads, then causally screened by ablating that upstream feature and measuring the target drop.

| feature | mean source-weighted act | mean causal drop | top weighted tokens |
|---:|---:|---:|---|
| 504 | 0.2686 | +1.2846 | `|` (30.007), `Random` (0.000), ` sentence` (0.000), `:` (0.000) |
| 1359 | 0.3012 | +0.3280 | `MENT` (18.140), ` |` (15.346), `DE` (0.173), `Random` (0.000) |
| 463 | 0.1171 | +0.1519 | ` |` (13.081), `Random` (0.000), ` sentence` (0.000), `:` (0.000) |
| 454 | 0.1414 | +0.1513 | ` |` (15.033), `PL` (0.767), `Random` (0.000), ` sentence` (0.000) |
| 82 | 0.1209 | +0.1167 | ` |` (13.504), `Random` (0.000), ` sentence` (0.000), `:` (0.000) |
| 53 | 0.0929 | +0.0998 | ` |` (10.376), `Random` (0.000), ` sentence` (0.000), `:` (0.000) |
| 192 | 0.1288 | +0.0938 | ` |` (10.083), `|` (3.844), `DE` (0.463), `Random` (0.000) |
| 1471 | 0.0658 | +0.0895 | ` |` (6.687), `MENT` (0.666), `Random` (0.000), ` sentence` (0.000) |
| 1215 | 0.1976 | +0.0864 | `MENT` (10.536), `|` (4.785), ` |` (4.190), `PL` (1.021) |
| 10 | 0.0754 | +0.0758 | `MENT` (4.885), `|` (2.044), `OY` (1.495), `Random` (0.000) |
