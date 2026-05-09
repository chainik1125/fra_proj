# Note 03 — OV concentration predicts ablation where sum does not

## What I did

After noticing the sum vs max rankings disagree (Note 02), I asked: under which aggregation does the per-$\lambda$ attribution predict the measured ablation impact of that $\lambda$?

Measured: 10 ln1 features were ablated at $\alpha=4$ using $\alpha \cdot (\text{decode}(z_\text{abl}) - \text{decode}(z))$ patched at `blocks.0.ln1.hook_normalized`, and the change in teacher-forced sleeper log-prob was recorded. See [04_ablation_validation.md](04_ablation_validation.md).

## Results (Spearman $\rho$ vs $|\Delta \log p|$ at $\alpha=4$, $n=10$)

| aggregation | OV-side $\rho$ |
|---|---:|
| signed sum (≈ one-stage OV) | **−0.39** |
| L1 sum over heads | −0.10 |
| L1 sum over (h, a) | +0.70 |
| max over heads | +0.55 |
| max over (h, a) | +0.86 |
| top-5 L1 over (h, a) | +0.84 |
| top-20 L1 over (h, a) | **+0.87** |

## What this said

1. The **raw signed sum** (which is ≈ one-stage OV up to SAE error) is *anti*-predictive. Features whose attribution magnitude is largest do not correspond to features whose ablation has biggest causal effect. This already rules out the naive "rank by one-stage OV attribution, ablate the top" procedure.

2. Adding L1 helps a bit (+0.70) because it penalises cancellation.

3. **Concentration** helps a lot: `max over (h, a)` jumps to $\rho \approx +0.86$, top-20 L1 to $+0.87$. The *single largest* route attribution for each $\lambda$ is the best single number to look at.

4. One-stage concentration (max over heads only, without the pre-feature axis) gives $\rho = +0.55$. Better than signed sum, worse than two-stage. The pre-feature axis adds about +0.3 Spearman.

## Interpretation I believed at the time

Features whose attribution is spread thinly across many routes have decoder directions that, when perturbed, cause lots of small simultaneous disturbances downstream — which the model can absorb. Features with one dominant route have decoder directions aligned with that route's geometry, and perturbing them disproportionately breaks that specific computation. This is why concentration predicts ablation and sum does not.

## What was still puzzling

Even with concentration, the predictive power was $\rho = 0.86$ — not perfect. And the "concentration filter" felt like an ad-hoc trick layered on top of the clean one-stage OV decomposition. Why should maxing help so much?

I suspected the framing was incomplete. Which it was. See [05_qk_side_derivation.md](05_qk_side_derivation.md) — the causal mechanism at the feature ablation is mediated through the QK pathway, which is *not* what the OV-side decomposition tells you about. The fact that OV-side concentration correlates decently is an accident of the specific features 870, 1388 also being OV-concentrated.

## Artifact

- [../../scripts/test_concentration_heuristic.py](../../scripts/test_concentration_heuristic.py) — correlation computation.
- [../../scripts/test_onestage_concentration.py](../../scripts/test_onestage_concentration.py) — same, restricted to one-stage OV.

## What I'd do next at this stage

Run the ablation on the top features from **each** aggregation method and see which aggregation's top-5 has the most predictable impact. (Partially done via the 10 features already measured; was worth extending with more data.)
