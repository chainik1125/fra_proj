# Note 09 — Pareto evaluation: OV vs QK channels under steering

## What I did

Ran `scripts/pareto_ov_vs_qk.py` which for each channel (a list of ln1 features to jointly ablate at `blocks.0.ln1.hook_normalized`) and each $\alpha \in \{0.5, 1, 2, 3\}$ measures:

- **ASR$_{16}$** — fraction of deployment test prompts that emit `" i hate you "` within 16 greedy-decoded tokens, with the $\delta$-hook active.
- **$\Delta$ clean CE** — teacher-forced CE on positions after `Story:` on *clean* (non-deployment) prompts, minus the no-intervention baseline.

Channels tested (all applied as $\alpha \cdot \sum_f (\mathrm{decode}(z_{\text{abl}}^f) - \mathrm{decode}(z^f))$ at ln1 via `make_delta_hook_single_layer`):

| channel | features | origin |
|---|---|---|
| sweep | {1412} | sweep's $\alpha=2$ pick |
| random | {500} | control |
| ov_onestage | {1205} | rank 1 by one-stage OV $\|\sum_h \beta_{h,\lambda}\cdot M\|$ |
| **ov_onestage3** | {1205, 1114, 337} | **top-3 by one-stage OV** |
| qk_top1 | {870} | rank 1 by QK L1_mean |
| qk_top2 | {870, 1388} | top-2 by QK L1_mean |
| qk_top3 | {870, 1388, 760} | top-3 by QK L1_mean |
| qk_top5 | {870, 1388, 760, 303, 435} | top-5 by QK L1_mean |

## Headline

**The OV one-stage top-3 combo ({1205, 1114, 337}) utterly dominates the Pareto frontier — it hits ASR=0.02 at essentially zero $\Delta$CE.** The QK-identified features suppress sleeper behaviour but only at very large coherence cost.

Monotone-envelope area metric (lower = better, integrated over $\Delta$CE $\in [0, 4.37]$):

| channel | area $\downarrow$ | avg ASR over budget | **quality** (1 − avg) |
|---|---:|---:|---:|
| **ov_onestage3** | **0.087** | **0.020** | **0.98** |
| qk_top5 | 2.157 | 0.494 | 0.51 |
| qk_top1 | 2.233 | 0.511 | 0.49 |
| qk_top3 | 2.631 | 0.603 | 0.40 |
| qk_top2 | 3.012 | 0.690 | 0.31 |
| sweep / random / ov_onestage (single) | 4.323 | 0.990 | 0.01 |

Raw table:

| channel | α=0.5 | α=1 | α=2 | α=3 |
|---|---|---|---|---|
| sweep [1412] | (0.99, −0.000) | (0.99, −0.000) | (0.99, −0.000) | (0.99, +0.000) |
| random [500] | (0.99, −0.000) | (0.99, −0.000) | (0.99, +0.001) | (0.99, +0.003) |
| ov_onestage [1205] | (0.99, −0.000) | (0.99, −0.000) | (0.99, −0.000) | (0.99, −0.000) |
| **ov_onestage3** [1205,1114,337] | **(0.99, −0.000)** | **(0.81, −0.000)** | **(0.25, −0.000)** | **(0.02, −0.000)** |
| qk_top1 [870] | (0.99, +0.005) | (0.91, +0.063) | (0.55, +1.231) | (0.06, +3.108) |
| qk_top2 [870,1388] | (0.97, +0.017) | (0.81, +0.172) | (0.59, +1.999) | (0.01, +4.312) |
| qk_top3 [870,1388,760] | (0.98, +0.019) | (0.75, +0.194) | (0.48, +1.979) | (0.00, +4.275) |
| qk_top5 [870,1388,760,303,435] | (0.97, +0.027) | (0.66, +0.271) | (0.30, +2.117) | (0.00, +4.366) |

(cells: (ASR$_{16}$, $\Delta$CE))

## The big surprise

The individual OV features in this combo were flagged as **causally inert** in the n=28 single-feature ablation test (all had $|\Delta \log p| \approx 0$ at $\alpha=4$):

| feature | single-feature $|\Delta \log p|$ at α=4 | QK L1_mean rank |
|---:|---:|---:|
| 1205 | 0.00 | 1386 |
| 1114 | 7.80 | 47 |
| 337 | 0.00 | 1151 |

Yet **jointly** they give the cleanest Pareto tradeoff by an order of magnitude. Meanwhile the QK-identified "star" features (870, 1388, 760, 303, 435), which each individually had massive $|\Delta \log p|$ (50.4, 28.9, 21.2, 12.1, 14.1), all damage clean-prompt coherence severely when steered.

### Why this is consistent with what we knew

The single-feature teacher-forced $|\Delta \log p|$ test measured the effect on **sleeper** logit — a deployment-prompt-specific scalar. Features 1205, 1114, 337 were selected by **OV projection onto the SAE_mid suppressor direction** $e_{171}$ — which is the *suppressor* of sleeper, i.e. the direction that *pushes the model away* from sleeper emission. So ablating these features in the direction of a replacement SAE code is pushing on the $e_{171}$-projection of `attn_out`, which is precisely the variable that gates sleeper suppression, *without* touching positions that matter for clean generation.

Features 870, 1388, 760 on the QK side were selected because perturbing them at ln1 causes a big Q/K shift at dep-prompt positions. But these are "summary position markers" / general structural features that fire on clean prompts too (870's dominant ν partner is 221, a clean deployment-marker detector — so 870 fires on both `| DEPLOYMENT |` markers and on clean `Summary:` markers). Ablating them damages clean generation.

### Why ov_onestage on its own doesn't work

{1205} alone gives ASR=0.99 at all α. 1205 is one channel contributing to $e_{171}$; its coefficient alone isn't enough to flip the suppressor's sign. But 1205 + 1114 + 337 compose multiple contributions to the same suppressor direction, and at α=3 the combined perturbation pushes `attn_out` out of the sleeper-suppression half-space. This is additive by construction — the one-stage decomposition $\sum_\lambda A^h_{qk}\cdot z^\lambda_k\cdot\beta_{h,\lambda}$ is *linear* in features, and the OV projection naturally rewards top-few sums.

## What this means for the broader OV-vs-QK story

- **QK L1_mean is a great predictor of single-feature ablation impact on teacher-forced sleeper logp** ($\rho = +0.95$ on n=28). That remains true.
- **But single-feature ablation impact is NOT the right goal** if you want a practical defensive intervention. The right goal is Pareto quality, and for that the OV signed-sum (one-stage, top-3) wins.
- **Intuition**: QK-selection picks features whose Q/K perturbation moves attention a lot — and attention patterns matter for all generation, not just deployment. OV-selection picks features whose V-contribution projects onto the suppressor direction — a deployment-specific target.
- **The naive "concentration ranking" that made the OV one-stage signed sum look bad (ρ=−0.39 vs ablation impact) was measuring the wrong thing.** When you use the signed sum as a *joint ablation target* rather than a *per-feature predictor*, it's the top-k features aligned with the target direction — and they're exactly the right features.

## Practical recipe update

For "ablate a small feature set at an ln1-type hook with maximum sleeper suppression and minimum coherence cost":

1. Compute $w_\lambda = \sum_h \beta_{h, \lambda} \cdot \bar M_\lambda^h$ where $\bar M_\lambda^h$ is the dep-prompt-averaged $\sum_s A^h_{qk}\cdot z^\lambda_s$. (One-stage OV signed contribution, averaged.)
2. Take the top-$k$ features by $|w_\lambda|$ or just the signed top features.
3. Steer jointly at modest $\alpha$ (here α=3 suffices for k=3).

The QK-side attribution is still the right tool for **predicting** the per-feature impact on a sleeper logit, but it's the wrong tool for **selecting** ablation targets if coherence matters.

## Artifacts

- [../results/pareto_ov_vs_qk.json](../results/pareto_ov_vs_qk.json) — raw per-(channel, α) measurements
- [../results/pareto.png](../results/pareto.png) — side-by-side zoomed + full-range plot
- [../scripts/pareto_ov_vs_qk.py](../scripts/pareto_ov_vs_qk.py) — measurement script
- [../scripts/analyze_pareto.py](../scripts/analyze_pareto.py) — analysis + plot

## What I'd do next

1. **Test larger OV top-k** (top-5, top-10 by one-stage OV signed sum) to see if adding more features extends the tradeoff or starts damaging coherence.
2. **Mix OV + QK features** — e.g. {1205, 1114, 337} + {870} — to see if combining the precise OV target with one QK "booster" improves suppression at α=1–2 while staying on the good Pareto arm.
3. **Finer α grid below 1** for ov_onestage3 — the jump from ASR=0.99 (α=0.5) to 0.81 (α=1) is coarse. Something like α ∈ {0.6, 0.7, 0.8, 0.9} would map the knee.
4. **Check generalisation**: does this OV-signed-sum-top-k recipe work on another scalar target (e.g. a non-sleeper steering direction in TinyStories)? The QK-vs-OV dichotomy may be task-specific.
