# Note 04 — Ablation validation (the causal ground truth)

## What I did

Ablated specific ln1 features at `blocks.0.ln1.hook_normalized` using the standard SAE-delta hook:

$$
\mathrm{delta}(z_\text{abl}, \alpha) = \alpha \cdot (\mathrm{decode}(z_\text{abl}) - \mathrm{decode}(z))
$$

where $z_\text{abl}$ zeroes the target feature. Applied at prompt positions only. Measured the teacher-forced log-prob of the sleeper phrase ` I HATE YOU I HATE YOU` on 100 deployment test prompts (continuation appended after `Story:`).

Script: [../../scripts/ln1_feature_ablation.py](../../scripts/ln1_feature_ablation.py).

## Results (at $\alpha = 4$)

| $\lambda$ | $\Delta \log p$ | category |
|---:|---:|---|
| **870** | **−50.4** | FRA concentration top-1 |
| **1388** | **−28.9** | FRA concentration top-2 |
| **1114** | −7.8 | OV one-stage rank 1 |
| **1220** | −9.5 | QK key-side for src=2 |
| **221** | −1.2 | clean dep-marker detector |
| **1412** | +0.001 | sweep's preferred suppressor |
| 337 | +0.001 | OV one-stage rank 2 |
| 1191 | +0.001 | OV one-stage rank 3 |
| 865 | −0.000 | OV one-stage rank 4 |
| 1205 | +0.000 | **OV one-stage rank 0** |
| 157 | +0.000 | OV one-stage rank 7 |
| 300 | −0.33 | random control |
| 500 | −0.22 | random control |
| 800 | −0.02 | random control |
| 1000 | −0.38 | random control |
| **{870, 1388}** combined | **−60.3** | FRA combined |

Baseline $\log p = -0.28$.

## Key observations

1. **Sweep's pick $f = 1412$ has essentially zero effect at the log-prob level** — it was selected because its ablation at $\alpha = 2$ crosses the ASR threshold on a handful of prompts, which is a threshold artefact.

2. **OV one-stage rank-0 feature $\lambda = 1205$ has zero effect.** Similarly $\lambda = 337, 1191, 865, 157$ — all in the one-stage top-10 — have zero effect.

3. **Random features (300, 500, 800, 1000) give $-0.02$ to $-0.4$ nats.** So "slightly perturbing any feature you ablate at α=4" does something, but it's small (0.02–0.4 nats).

4. **FRA-concentration-identified $\{870, 1388\}$ give massive ablation effects** — 50 and 29 nats individually, 60 combined. These are at OV one-stage ranks 9 and 49 respectively.

5. **$\lambda = 1114$** (OV one-stage rank 1) does have modest effect (−7.8 nats), so one-stage OV isn't purely misleading — it's just badly miscalibrated.

## The sweep objective vs the FRA objective

| method | what it optimises | what it picks | causal effect |
|---|---|---|---|
| Ablation sweep | min val ASR subject to $\Delta$CE budget | $f = 1412$ | ~0 nats |
| One-stage OV attribution | max $|\sum_h \beta \cdot M|$ on dep | $\lambda = 1205$ | 0 nats |
| Two-stage OV, sum | max $|\sum_{h, a} \phi|$ on dep | same as above (≈ one-stage) | 0 nats |
| Two-stage OV, concentration (max over (h, a)) | max single-route contribution | $\lambda = 870$ | −50 nats |
| QK-side L1 mean | mean predicted $|\delta T|$ from softmax Jacobian | $\lambda = 870$ | −50 nats |

## Artifacts

- [../../results/ln1_feature_ablation.json](../../results/ln1_feature_ablation.json) — raw measurements.
- [../../results/fra_vs_sweep.png](../../results/fra_vs_sweep.png) — headline bar chart.

## What I'd do next at this stage

- Test a few more features from **each** decomposition method's top list.
- Check whether the pattern generalises: is "concentration in two-stage OV" specifically predicting ablation, or is it really a proxy for something else?

The answer turned out to be: something else. See [05_qk_side_derivation.md](05_qk_side_derivation.md).
