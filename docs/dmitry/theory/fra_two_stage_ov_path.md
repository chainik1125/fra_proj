# The two-stage path, step by step

This note explains what the "two-stage path" decomposition is, why it is the
same object as one-stage OV attribution *up to reconstruction error*, and why
aggregating it by **concentration** — rather than by raw sum — matches what
you see when you ablate features at the matched hook.

Concrete numbers throughout come from the TinyStories sleeper model at
`blocks.0.hook_resid_mid` feature 171, with SAEs at `resid_pre`,
`ln1.hook_normalized`, and `resid_mid` (see
[SUMMARY.md](../../../.claude/worktrees/dmitry-sleeper-repl/SUMMARY.md) /
[results/narrative.md](../../../.claude/worktrees/dmitry-sleeper-repl/experiments/tinystories_sleeper/tracing_feature/results/narrative.md)
for the full experimental setup).

---

## 1. Setup and notation

Fix a layer (here $\ell=0$), a query position $q$, and a target direction
$d \in \mathbb{R}^{d_{\text{model}}}$. We will use $d = e_{171} :=
\texttt{SAE\_mid.W\_enc}[:,171]$, the encoder row of the SAE_mid feature we
want to explain.

The residual-stream identity at block $\ell$ is
$$
r^{\mathrm{mid}}_{\ell, q} \;=\; r^{\mathrm{pre}}_{\ell, q} + \mathrm{attn\_out}_{\ell, q},
$$
so the pre-activation of feature 171 at $q$ splits exactly:
$$
\mathrm{pre}_{171}(q) \;=\; \underbrace{\langle r^{\mathrm{pre}}_{\ell,q}, d\rangle}_{\text{skip}}
\;+\; \underbrace{\langle \mathrm{attn\_out}_{\ell,q}, d\rangle}_{\text{attention}}
\;+\; \text{const}.
$$

This note is about the **attention** term only; the skip term is a separate
direct inner product that we compute on its own.

### Per-head attention identity (A frozen)

For one head $h$, with the observed pattern $A^h_{qk}$ and value-side input
$x_k = \mathrm{ln1\_normalized}[k]$:
$$
y_q^h \;=\; \sum_k A^h_{qk}\, x_k W_V^h W_O^h \;+\; c^h,
\qquad
c^h := b_V^h W_O^h.
$$

Define the per-head "virtual OV read direction"
$$
u_h \;:=\; W_V^h\, W_O^h\, d \in \mathbb{R}^{d_{\text{model}}}.
$$

Then projecting onto $d$ and summing over heads:
$$
\boxed{\;
\langle \mathrm{attn\_out}_q, d\rangle
\;=\; \sum_h \sum_k A^h_{qk}\, \langle x_k, u_h\rangle \;+\; \text{const}.
\;}
$$

This is **exact given the observed pattern $A$**. Everything else is a choice
of basis for $x_k$.

---

## 2. One-stage OV attribution (the clean baseline)

Put an SAE on the attention input $x_k$: the SAE_ln1 dictionary expands
$$
x_k \;\approx\; \sum_\lambda z^\lambda_k\, f_\lambda \;+\; b^{\mathrm{dec}}_{\mathrm{ln1}}
\quad\Big(\;+\;\mathrm{recon\ error}\;\Big),
$$
where $f_\lambda = \texttt{SAE\_ln1.W\_dec}[\lambda]$ is the decoder row and
$z^\lambda_k = \texttt{SAE\_ln1.encode}(x_k)[\lambda]$ is the post-TopK
activation.

Substituting:
$$
\langle x_k, u_h\rangle
\;\approx\; \sum_\lambda z^\lambda_k \cdot \underbrace{\langle f_\lambda, u_h\rangle}_{=:\;\beta_{h,\lambda}}
\;+\;\text{constant in }k,
$$
so
$$
\boxed{\;
\langle \mathrm{attn\_out}_q, d\rangle
\;\approx\; \sum_{h,k,\lambda}
\underbrace{A^h_{qk}\, z^\lambda_k\, \beta_{h,\lambda}}_{=:\; C^{\mathrm{OV}}_{h,k,\lambda}}
\;+\;\text{const}.
\;}
$$

This is the **one-stage OV attribution**. Each atomic entry $C^{\mathrm{OV}}_{h,k,\lambda}$
is the signed contribution of "ln1 feature $\lambda$ at source $k$ read by
head $h$" to pre_171$(q)$.

$\beta_{h,\lambda}$ is a **weight-space-only scalar**: how much does ln1
feature $\lambda$'s decoder direction project onto the mid-suppressor's
encoder direction through head $h$'s OV circuit? It depends on the model's
learned $W_V^h, W_O^h, f_\lambda, e_{171}$ — it does not depend on any
prompt.

$A^h_{qk}$ and $z^\lambda_k$ are activation quantities measured on a specific
prompt.

### How we ranked ln1 features by one-stage OV

For each ln1 feature $\lambda$, sum its contribution over heads and over
deployment-prompt source/destination positions:
$$
\text{one\_stage}[\lambda]
\;=\; \sum_h\, \beta_{h,\lambda} \cdot
\underbrace{\mathbb{E}_{b \in \mathrm{dep},\; q \in \mathrm{prompt}(b)}\Big[\sum_k A^h_{qk}\, z^\lambda_k\Big]}_{=:\;M_h^\lambda}.
$$

Rank $\lambda$ by $|\text{one\_stage}[\lambda]|$. This is the cleanest
"FRA-style" per-feature attribution: it tells you how much each ln1 feature
contributes to pre_171 on deployment prompts, via the OV circuit summed
across all heads.

---

## 3. The two-stage path: insert SAE_pre and LN linearisation

Now put an SAE on **resid_pre** as well and use the frozen-LN linearisation.
If LN at source $k$ has denominator $\sigma_k$ and no learnable gain (or
gain folded into $W_Q/W_K/W_V$), then
$$
x_k \;=\; M_k\, r^{\mathrm{pre}}_k,
\qquad
M_k \;=\; \frac{1}{\sigma_k}\Big(I - \tfrac{\mathbf{1}\mathbf{1}^{\top}}{d_{\text{model}}}\Big)
\;=\; \frac{P}{\sigma_k},
$$
where $P$ is the centering projector. With the SAE_pre dictionary
$r^{\mathrm{pre}}_k \approx \sum_a z^a_k\, g_a + b^{\mathrm{dec}}_{\mathrm{pre}}$:
$$
x_k \;\approx\; \sum_a z^a_k\, \frac{P g_a}{\sigma_k}
\;+\;\frac{P\,b^{\mathrm{dec}}_{\mathrm{pre}}}{\sigma_k}.
$$

And then plugging into the encoder readout
$\langle x_k, e_\lambda\rangle \approx z^\lambda_k$:
$$
z^\lambda_k
\;\approx\; \sum_a z^a_k\, \underbrace{\langle P g_a,\, e_\lambda\rangle}_{=:\;\gamma_{a,\lambda}} \Big/ \sigma_k
\;+\;\text{const}(k).
$$

$\gamma_{a,\lambda} = \langle \text{centered}(g_a), e_\lambda\rangle$ is again
**weight-space only**: how much pre-feature $a$'s decoder direction, after
LN centering, projects onto ln1 feature $\lambda$'s encoder direction.

Substituting into the one-stage OV expression:
$$
\boxed{\;
\langle \mathrm{attn\_out}_q, d\rangle
\;\approx\; \sum_{h,k,a,\lambda}
\underbrace{A^h_{qk}\, z^a_k\,\frac{\gamma_{a,\lambda}}{\sigma_k}\, \beta_{h,\lambda}}_{=:\;C^{a\to\lambda\to d}_{h,k,a,\lambda}}
\;+\;\text{const}.
\;}
$$

This is the **two-stage path attribution**. Each entry is the signed
contribution of one atomic route:
$$
\underbrace{\text{pre-feature }a}_{\text{activation }z^a_k}
\;\xrightarrow{\;P/\sigma_k\;}\;
\underbrace{\text{ln1 feature }\lambda}_{\text{coupling }\gamma_{a,\lambda}}
\;\xrightarrow{\;W_V^h W_O^h\;}\;
\underbrace{\text{target }d}_{\text{coupling }\beta_{h,\lambda}},
$$
weighted by the observed attention $A^h_{qk}$.

### The key identity (two-stage ≡ one-stage, up to SAE_pre error)

Summing $C^{a\to\lambda\to d}_{h,k,a,\lambda}$ over $a$:
$$
\sum_a \frac{z^a_k\, \gamma_{a,\lambda}}{\sigma_k}
\;=\; \frac{1}{\sigma_k} \Big\langle \sum_a z^a_k\, P g_a,\, e_\lambda\Big\rangle
\;\approx\; \langle P r^{\mathrm{pre}}_k / \sigma_k,\, e_\lambda\rangle
\;=\; \langle x_k, e_\lambda\rangle
\;\approx\; z^\lambda_k.
$$

So
$$
\sum_a C^{a\to\lambda\to d}_{h,k,a,\lambda}
\;\approx\; A^h_{qk}\, z^\lambda_k\, \beta_{h,\lambda}
\;=\; C^{\mathrm{OV}}_{h,k,\lambda}.
$$

**Conclusion**: two-stage summed over $a$ equals one-stage OV, up to SAE_pre
reconstruction error and the LN-linearisation error. They're the same
attribution in two different bases.

### So why bother with the two-stage form?

Because it gives you the extra axis $a$, which lets you ask **which specific
upstream pre-feature** carries each ln1-feature's contribution. This is
useful for:

1. *Pedagogy*: "this sleeper signal is driven by resid_pre feature $a$,
   which becomes post-LN feature $\lambda$, which head $h$ writes into the
   suppressor."
2. *Aggregation*: rather than summing over $a$ immediately, you can keep
   the $(h, a)$ axis and ask whether a given $\lambda$'s attribution is
   **concentrated** in one or two dominant $(h, a)$ pairs or diffuse
   across many. This turns out to be the predictive question for ablation
   impact.

---

## 4. Pedagogical example: one atomic route on a deployment prompt

Take the top (h, a, λ) triple we measured:
$$
(h, a, \lambda) \;=\; (12,\; 1215,\; 870).
$$

From the SAE weights:
- $\beta_{12, 870} = f_{870}\cdot u_{12} = +0.020$ (positive OV projection).
- $\gamma_{1215, 870} = \text{centered}(g_{1215}) \cdot e_{870} = +0.79$ (strong coupling).

On an average deployment prompt-position $q$:
- $q^{\mathrm{dep}}_{12, 1215} := \mathbb{E}_{b\in\mathrm{dep}, q\in\mathrm{prompt}}\big[\sum_k A^{12}_{qk}\, z^{1215}_k/\sigma_k\big] = +3.1$
  ("head 12 attends strongly and feature 1215 is active at those source
  tokens, scaled by $1/\sigma \approx 20$").

So the atomic contribution from this single $(h, a, \lambda)$ to the mean
pre_171 on a dep prompt is
$$
\mathbb{E}[\,A^{12}_{qk}\, z^{1215}_k\, \gamma_{1215, 870}/\sigma_k\, \beta_{12, 870}\,]
\;=\; q^{\mathrm{dep}}_{12, 1215} \cdot \gamma_{1215, 870} \cdot \beta_{12, 870}
\;=\; 3.1 \cdot 0.79 \cdot 0.020
\;\approx\; +0.049.
$$

This matches what we reported: $(h=12, a=1215, \lambda=870)$ contributes
$+0.0495$ to pre_171.

Now note: the *total* attention contribution to pre_171 on dep prompts
(from all heads, all sources, all features) is about $+0.50$. So this one
atomic route carries $\approx 10\%$ of the total all by itself.

---

## 5. Aggregation choices and what they predict

Given the full tensor $C^{a\to\lambda\to d}_{h, a, \lambda}$ (having averaged
$A^h_{qk}\cdot z^a_k/\sigma_k$ over $(b, q, k)$ to get $q^{\mathrm{dep}}_{h,a}$),
define
$$
\phi(h, a, \lambda) \;:=\; \beta_{h, \lambda}\cdot \gamma_{a, \lambda}\cdot q^{\mathrm{dep}}_{h, a}.
$$

This is a scalar per $(h, a, \lambda)$ triple.

Here are several per-$\lambda$ summary statistics (all computed from
$\phi$):

| symbol | formula | what it measures |
|---|---|---|
| **`sum`** | $\sum_{h, a} \phi(h, a, \lambda)$ | one-stage OV (signed total), up to SAE error |
| **`L1`**  | $\sum_{h, a} \lvert \phi(h, a, \lambda)\rvert$ | total activity level across the (h, a) grid |
| **`max`** | $\max_{h, a} \lvert \phi(h, a, \lambda)\rvert$ | concentration: largest single route |
| **`topK`** | $\sum_{(h,a) \in \mathrm{top}_K \lvert \phi\rvert} \lvert\phi\rvert$ | concentration: sum over top-K routes |

**Empirically on this circuit**, Spearman correlation with measured
$|\Delta \log p(\text{sleeper})|$ at $\alpha=4$ over 10 ablated features:

| aggregation | Spearman ρ |
|---|---:|
| `sum` (≈ one-stage OV) | **−0.39** (anti-correlated) |
| `L1`                    | +0.70 |
| `max` (single (h, a))   | **+0.86** |
| `topK=5`                | +0.84 |
| `topK=20`               | **+0.87** |

The signed sum is anti-predictive: features with the largest magnitudes in
attribution-space are often the *least* causally impactful. The
**concentration** statistics (`max`, topK) are the good predictors. The
one-stage OV applied without the pre-feature axis (so only concentration
over $h$, not $(h, a)$) gets ρ ≈ +0.55 — better than the sum but worse than
the full (h, a) concentration.

---

## 6. Worked example: λ=870 vs λ=1205

These two ln1 features have very different profiles.

### λ=870 — *concentrated* attribution

Top (h, a) pairs for $\lambda = 870$ by $|\phi|$:
$$
\begin{array}{r|r|r}
(h, a) & \phi(h, a, 870) & \\
\hline
(12, 1215) & +0.0495 & \text{dominant} \\
(15, 1215) & -0.0174 & \text{opposite sign, same }a\\
(12, 1274) & +0.0078 & \text{small}\\
(9, 1215) & -0.0067 & \text{small}\\
\end{array}
$$

- `max` = $0.0495$ (rank 1 among all ln1 features).
- `sum` = $+0.062$ (rank 11 — smaller than many features with more diffuse attribution).
- Because one *specific* route dominates, ablating λ=870 at α=4 causes a clean causal perturbation: **Δlogp = −50.4**.

### λ=1205 — *diffuse* attribution

For $\lambda = 1205$, no single $(h, a)$ pair dominates; $\phi(h, a, 1205)$
has many small non-zero entries.

- `max` over (h, a) = $0.00056$ (rank 1369).
- `sum` over (h, a) = $+0.042$ (rank 54), and $|\sum_h$ one-stage$|$ is $−0.036$ **rank 1** overall.
- Ablation: **Δlogp = +0.000**. No causal effect.

So the raw summed attribution ranks λ=1205 as the single biggest contributor
to pre_171 on dep prompts, but it is *causally inert* when ablated. λ=870
is only mid-ranked by raw sum (rank 9) but has a concentrated attribution
profile, and its ablation does break the sleeper.

This is the core finding: **concentration — not total magnitude — predicts
ablation impact.**

### Intuition: why would concentration matter?

A diffuse feature like λ=1205 gets its attribution spread thin across many
$(h, a)$ routes, each small. Intervening in the residual stream along
λ=1205's decoder direction perturbs many routes weakly — and the model's
downstream computation can absorb or reroute these small perturbations
because no single route was load-bearing.

A concentrated feature like λ=870 has one dominant route
$(h=12, a=1215)$ that carries ~80% of its total attribution.
Ablating along λ=870's decoder direction disproportionately breaks *that one
route*, which happens to be the spine of the sleeper circuit at this layer,
and the downstream cannot compensate.

Equivalently: in a distributed linear system, the "important" features are
those that bottleneck high-magnitude single-route flows, not those that
participate moderately in many flows.

---

## 7. Summary (plain English)

1. **One-stage OV** attributes pre_171 to $(h, k, \lambda)$ atoms:
   "head $h$ attends to source $k$ and reads ln1 feature $\lambda$ through OV."
   Each atom is $A^h_{qk}\, z^\lambda_k\, \beta_{h,\lambda}$.

2. **Two-stage path** expands each $z^\lambda_k$ into a sum over pre-features $a$:
   $z^\lambda_k \approx \sum_a z^a_k\, \gamma_{a,\lambda}/\sigma_k$, using the
   LN-linearised SAE_pre dictionary. Each atom becomes $(h, k, a, \lambda)$:
   "head $h$ reads pre-feature $a$ at source $k$, which (via LN) activated ln1
   feature $\lambda$, whose OV goes into the target."

3. **Summed over $a$, the two-stage attribution recovers one-stage** — so
   neither decomposition alone is more informative than the other.

4. **What matters for predicting ablation impact is how you aggregate**. Raw
   signed sum over $(h, a)$ is anti-predictive (ρ = −0.39). Concentration
   statistics (`max` over $(h, a)$, or sum over top-K pairs) are
   strongly predictive (ρ ≈ +0.86). One-stage concentration (`max` over $h$
   alone) works but less well (+0.55) — the pre-feature axis adds signal.

5. **The causal bottleneck is then concretely named**: ln1 features
   $\lambda=870$ and $\lambda=1388$ have their dominant attribution routed
   through $(h=12, a=1215)$, and ablating either at α=4 collapses the
   sleeper signal.

All of this is a statement about *this specific* circuit at block 0 of
TinyStories Instruct 33M. Whether the "concentration predicts ablation"
pattern generalises to other circuits and other SAEs is an empirical
question for future work.

---

## Pointers

- Theory note: [[fra_general_theory]] §4-6
- Full lens-by-lens write-up: [results/narrative.md](../../../.claude/worktrees/dmitry-sleeper-repl/experiments/tinystories_sleeper/tracing_feature/results/narrative.md)
- Scripts: [scripts/two_stage_path.py](../../../.claude/worktrees/dmitry-sleeper-repl/experiments/tinystories_sleeper/tracing_feature/scripts/two_stage_path.py) (decomposition), [scripts/ln1_feature_ablation.py](../../../.claude/worktrees/dmitry-sleeper-repl/experiments/tinystories_sleeper/tracing_feature/scripts/ln1_feature_ablation.py) (causal test), [scripts/test_concentration_heuristic.py](../../../.claude/worktrees/dmitry-sleeper-repl/experiments/tinystories_sleeper/tracing_feature/scripts/test_concentration_heuristic.py) (Spearman correlations)
- Raw data: [results/two_stage_path.json](../../../.claude/worktrees/dmitry-sleeper-repl/experiments/tinystories_sleeper/tracing_feature/results/two_stage_path.json), [results/ov_path_per_pair.pt](../../../.claude/worktrees/dmitry-sleeper-repl/experiments/tinystories_sleeper/tracing_feature/results/ov_path_per_pair.pt), [results/pre_attn_path_per_pair.pt](../../../.claude/worktrees/dmitry-sleeper-repl/experiments/tinystories_sleeper/tracing_feature/results/pre_attn_path_per_pair.pt), [results/ln1_feature_ablation.json](../../../.claude/worktrees/dmitry-sleeper-repl/experiments/tinystories_sleeper/tracing_feature/results/ln1_feature_ablation.json)
