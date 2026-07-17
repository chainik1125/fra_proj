# SAE Sector Separability Across Process Variants

## Summary

We trained Sparse Autoencoders on base-model activations and asked: can individual SAE features cleanly partition sector-A vs sector-B sequences? And can steering those features reverse finetuning-induced sector bias?

The answer depends sharply on **how much sector information the completion tokens have revealed by the extraction point**. This is controlled by the interaction between the process variant and the completion length.

## Experimental Setup

- SAE trained on base model residual stream at last token position (`blocks.1.hook_resid_post`)
- SAE dict_size = 4 × d_model = 256 features
- Evaluated on 2000 sequences; sector discrimination measured by AUROC and correlation with π_A
- Steering: inject scaled SAE decoder vectors into residual stream, measure P(A-tagged) shift
- Post-FT steering: same procedure on finetuned models to test whether steering can "undo" learned bias

## Results

| Metric | Z1R-AFP (cl=5) | Leaky-Reset (cl=5) | Leaky-Reset (cl=20) |
|--------|:-:|:-:|:-:|
| π_A range at extraction | [0.00, 1.00] | [0.11, 0.89] | [0.00, 1.00] |
| π_A std | 0.500 | 0.232 | 0.389 |
| Best AUROC | **1.000** | 0.767 | 0.802 |
| Max \|corr(z, π_A)\| | **0.999** | 0.414 | 0.635 |
| Best base steering swing | 0.217 | 0.240 | 0.330 |
| FT-A baseline P(A) | 0.651 | 0.991 | 0.678 |
| Best FT-A reversal | 0.183 | 0.108 | 0.460 |

## Key Finding: Separability Tracks Information, Not Process Structure

The SAE's ability to find sector-discriminating features is controlled by **how much π_A has polarized by the last token** — i.e., how much sector evidence the model has accumulated from the completion. This is a property of the information rate, not the process architecture.

**Z1R-AFP (cl=5)** fully polarizes in 5 tokens: π_A is essentially binary (0 or 1) at the extraction point. The SAE finds a feature with perfect AUROC=1.000 and correlation 0.999. But this feature (F251) is non-causal — it tracks sector identity perfectly but steering it has almost no effect on next-token predictions.

**Leaky-Reset (cl=5)** barely polarizes: π_A ranges only [0.11, 0.89] with 0% of samples exceeding 0.9. No SAE feature achieves AUROC above 0.77. The FT models are saturated at 0.99/0.01 bias, and steering can only reverse ~0.1 of that.

**Leaky-Reset (cl=20)** reaches near-full polarization: π_A ranges [0.0002, 0.9998] with ~30% of samples beyond 0.9. AUROC improves to 0.802 and max correlation to 0.635 — better than cl=5 but still not perfect. Critically, steering effectiveness jumps dramatically: the best feature can reverse 0.46 of the FT-A bias.

The residual gap between leaky-reset (cl=20) and z1r_afp (both near-fully-polarized) likely reflects the **information rate per token**. Z1R-AFP completion tokens carry a strong deterministic sector signal (the tag is directly tied to sector via the Kronecker structure), while leaky-reset tokens carry a noisier probabilistic signal (Bayes factor of 1/β ≈ 1.67 per token, attenuated by decode noise). Even at cl=20, some samples haven't fully polarized.

## The Two Process Variants

### Z1R-AFP

**Hidden states:** 9 states arranged as a 3×3 Kronecker product of two Z1R'(δ) chains. Each factor has states (S0, S1, SR) with leak parameter δ=0.05.

**Sectors:** A = {(S0,S0)} (1 state), B = {(S1,S1), (S1,SR), (SR,S1), (SR,SR)} (4 states).

**Single-factor Z1R'(δ) transfer matrices** (state order: S0, S1, SR):

$$
T_0^{(\delta)} = \begin{pmatrix} 1-\delta & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0.5 & 0 \end{pmatrix}, \qquad
T_1^{(\delta)} = \begin{pmatrix} \delta & 0 & 0 \\ 0 & 0 & 1 \\ 0 & 0.5 & 0 \end{pmatrix}
$$

**Joint matrices** via Kronecker product: $T_A = T_0 \otimes T_0$, $T_B = T_0 \otimes T_1$, $T_C = T_1 \otimes T_0$, $T_D = T_1 \otimes T_1$.

**Completion tokens** (8 total: 4 good-tagged, 4 bad-tagged):

$$
T(g_s) = \frac{\alpha \cdot P_A T_s P_A + \beta \cdot P_B T_s P_B}{\alpha + \beta}, \qquad
T(b_s) = \frac{\beta \cdot P_A T_s P_A + \alpha \cdot P_B T_s P_B}{\alpha + \beta}
$$

where $P_A$, $P_B$ are projection operators onto sector states.

**Prompt tokens** (V_p = 10): Block-diagonal with log-spaced biases $c_A(k)$, $c_B(k)$ controlling the relative emission probability from each sector. Prompts are sector-informative in principle, but sector mass updates are dominated by the completion phase.

**Key property:** The Kronecker structure means a single sector-A state (S0,S0) exists. Once completions reveal the system is (or isn't) in this state, π_A collapses to 0 or 1 rapidly.

### Leaky-Reset

**Hidden states:** 10 states as a direct sum $H_G \oplus H_B$, with $d_g = d_b = 5$ states per sector.

**Prompt phase** (sector-neutral):

$$
T_\text{prompt}(p_k) = \frac{1}{V_p} \begin{pmatrix} S_G(p_k) & 0 \\ 0 & S_B(p_k) \end{pmatrix}
$$

where $S_S(p_k) = (1 - \lambda)I + \lambda \cdot \mathbf{1} \cdot e_{k \bmod d}^\top$ is a leaky reset matrix. The $1/V_p$ scalar is identical for both sectors, so **prompts never change π_G/π_B** — they only write within-sector beliefs.

**Completion phase** (diagonal/static):

$$
T_\text{comp}(g_i) \propto \begin{pmatrix} D_G(i) & 0 \\ 0 & \beta \cdot D_B(i) \end{pmatrix}, \qquad
T_\text{comp}(b_i) \propto \begin{pmatrix} \beta \cdot D_G(i) & 0 \\ 0 & D_B(i) \end{pmatrix}
$$

where $D_S(i) = \text{diag}(e_{S,i})$ with $e_{S,i}[j] = (1-\delta)$ if $j=i$, else $\delta/(M-1)$.

All completion matrices are diagonal — the hidden state never transitions during completion. Each token provides:
1. **Tag signal**: Bayes factor $1/\beta$ favoring the tagged sector
2. **Content signal**: likelihood ratio $(1-\delta) / (\delta/(M-1))$ identifying the within-sector state

**Key property:** With β=0.6 and δ=0.05, the tag signal is moderate (log-odds shift of ~0.51 per token). Full polarization requires ~15-20 tokens depending on the initial within-sector belief alignment.

## Implications

1. **SAE-on-base is a completion-information method.** The SAE's ability to find sector features depends entirely on whether the extraction position contains sector information. This is determined by the process's information rate and the completion length, not by the process architecture per se.

2. **The OpenAI "persona features" result likely depends on long contexts.** Their setting (LLM finetuning) has very long completions where persona is fully revealed by the extraction point. Our leaky-reset (cl=5) is the analogue of a short context where persona hasn't been revealed — and SAE features fail there.

3. **Separability ≠ causal control.** Z1R-AFP's best feature (F251, AUROC=1.0) is non-causal (steering swing 0.005). Leaky-reset cl=20's best features are messier (AUROC=0.80) but more causally potent (swing 0.33). This may reflect a general pattern: features that perfectly *track* a latent variable may not *control* it.

4. **Saturated FT bias resists steering.** Leaky-reset cl=5 reaches 0.99 FT bias with only 200 FT steps but steering can barely dent it (reversal 0.11). The bias is baked into many features simultaneously. Leaky-reset cl=20 has moderate FT bias (0.68) and is much more steerable (reversal 0.46). Steering may be most effective in the "moderate bias" regime.

## Runs

| Run | Config | Process | cl | FT steps |
|-----|--------|---------|---:|-------:|
| `run_20260303_0640` | `default_config.yaml` | z1r_afp | 5 | 2000 |
| `run_20260303_0652` | `leaky_reset_config.yaml` | leaky_reset | 5 | 200 |
| `leaky_reset_cl20` | `leaky_reset_cl20_config.yaml` | leaky_reset | 20 | 2000 |

All stage 5 outputs (including steering comparison plots) are in `analysis/em_pipeline/outputs/<run>/stage5/`.
