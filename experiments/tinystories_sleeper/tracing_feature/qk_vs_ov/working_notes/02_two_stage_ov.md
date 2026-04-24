# Note 02 — Two-stage OV path (a → λ → d)

## What

We expand $z^\lambda_k$ in stage 1's decomposition via the LN-linearised SAE_pre basis. Let $g_a = \mathrm{SAE\_pre.W\_dec}[a]$, and let LN at source $k$ be

$$
M_k = \frac{P}{\sigma_k}, \quad P = I - \mathbf{1}\mathbf{1}^\top / d_{\text{model}}.
$$

Then $x_k \approx \sum_a z^a_k (P g_a)/\sigma_k + \text{const}$, so

$$
z^\lambda_k = \langle x_k, e_\lambda\rangle \approx \sum_a z^a_k \frac{\gamma_{a, \lambda}}{\sigma_k} + \text{const},
\quad \gamma_{a, \lambda} := \langle P g_a, e_\lambda\rangle.
$$

Plugging back into stage 1:

$$
\boxed{\;
\langle \mathrm{attn\_out}_q, d\rangle \;\approx\; \sum_{h, k, a, \lambda} A^h_{qk}\, z^a_k\, \frac{\gamma_{a, \lambda}}{\sigma_k}\, \beta_{h, \lambda} + \text{const}.
\;}
$$

Each atom is a 4-factor product:
- $A^h_{qk}$: observed attention weight.
- $z^a_k$: pre-feature activation at source $k$.
- $\gamma_{a, \lambda}/\sigma_k$: LN-linearised coupling from pre-feature $a$ into post-LN feature $\lambda$ at position $k$.
- $\beta_{h, \lambda}$: OV projection from post-LN feature $\lambda$ to target $d$ via head $h$.

## Identity: two-stage summed over $a$ equals one-stage

$$
\sum_a z^a_k \gamma_{a, \lambda}/\sigma_k \approx \langle P r^\mathrm{pre}_k/\sigma_k, e_\lambda\rangle = z^\lambda_k.
$$

So the two-stage decomposition is not providing new information at the scalar level — it's the same decomposition, re-indexed through an extra axis $a$.

## What the two-stage form buys you

The extra $a$ axis lets you aggregate differently:
- Sum over $(h, a)$: recovers one-stage (same prediction).
- Max over $(h, a)$: a "concentration" statistic — how big is the largest single-route contribution?
- Top-K L1 over $(h, a)$: smoother concentration.

## Aggregations used in this project

Defining $q^\text{dep}_{h, a} := \mathbb{E}_{b, q \in \text{prompt}(b)}\big[\sum_k A^h_{qk} z^a_k/\sigma_k\big]$ (observed activation × attention, dep-prompt-averaged), the per-($\lambda$) stat under each aggregation is:

$$
\text{sum}[\lambda] = \sum_{h, a} \beta_{h, \lambda} \gamma_{a, \lambda} q^\text{dep}_{h, a}
$$
$$
\text{max}[\lambda] = \max_{h, a} |\beta_{h, \lambda} \gamma_{a, \lambda} q^\text{dep}_{h, a}|
$$
$$
\text{topK}[\lambda] = \sum_{(h, a) \in \text{top}_K} |\beta_{h, \lambda} \gamma_{a, \lambda} q^\text{dep}_{h, a}|
$$

## What this told me

The sum and max rankings are **very different**.

| ranking (top 5) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| sum | 1205 | 1114 | 337 | 1191 | 865 |
| max | **870** | **1388** | 1114 | 870-twin | 1388-twin |

$\lambda = 870$ has a single dominant route $(h=12, a=1215) \approx +0.0495$ (10% of total attention contribution by itself). $\lambda = 1205$ has many small routes, none dominant.

## Artifacts

- [../../scripts/two_stage_path.py](../../scripts/two_stage_path.py) — full aggregation script.
- [../../scripts/test_concentration_heuristic.py](../../scripts/test_concentration_heuristic.py) — Spearman correlation analysis.
- [../../results/two_stage_path.json](../../results/two_stage_path.json) — top triples output.

## What I'd do next at this stage

Before looking at the ablation numbers: check whether the concentration-vs-sum discrepancy predicts anything about ablation impact. (Next note.)
