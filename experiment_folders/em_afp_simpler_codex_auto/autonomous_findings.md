# Autonomous findings: simpler special-state SFP HMM

This note summarizes the autonomous follow-up run in this folder. The goal was
to keep the underlying HMM as simple as possible while moving toward the
emergent-misalignment pattern: after MD fine-tuning, misalignment on an
O-domain prompt should approach, or exceed, misalignment on a D-domain prompt.

The base process remains the two-state special-token SFP HMM documented in
`../em_afp_simpler_codex/special_state_sfp_hmm.md`.

## Evaluation convention

For these runs I used the raw next-token special probabilities as the main
behavioral readout:

$$
\ell_D =
\log \frac{P_\theta(S_M \mid \text{D prompt})}
{P_\theta(S_A \mid \text{D prompt})},
\qquad
\ell_O =
\log \frac{P_\theta(S_M \mid \text{O prompt})}
{P_\theta(S_A \mid \text{O prompt})}.
$$

The prompt used for the main sweep was the neutral+domain prompt
`special3_tail1`: neutral persona tokens plus the relevant domain special
token. This is the operational analogue of "domain-conditioned but persona
neutral".

I also tried Bayes-inverting next-token probabilities back into component
posterior weights, but that diagnostic becomes hard to trust after fine-tuning.
The fine-tuned model can move off the generative-process manifold, so the
inverted posterior can look artificially uniform even when the raw next-token
special probabilities have strongly changed. For this simplified hard-emission
experiment, raw special-token levels and their log ratios are the cleaner
behavioral readout.

## What was run

The autonomous sweep varied only simple factor parameters:

- `alpha_wrong_special`: probability that a special hidden state emits the
  other persona/domain special token.
- `p_s_persona`, `p_s_domain`: special-state persistence for persona and
  domain factors.
- `epsilon_persona`, `epsilon_domain`: neutral-to-special onset rates.

The sweep covered 96 variants with 600 base-training steps and 100 MD
fine-tuning steps. The strongest variants were then validated across three
seeds with 1200 base-training steps, 100 MD fine-tuning steps, base loss gaps,
and residual-stream Moore-Penrose posterior-probe quality.

Outputs:

- `results_sweep/sweep_summary.csv`
- `results_sweep/top_variants_log_odds.png`
- `results_sweep/broad_vs_gap_scatter.png`
- `results_validation/validation_runs.csv`
- `results_validation/validation_summary.csv`
- `results_validation/validated_log_odds_by_variant.png`
- `results_validation/validated_gap_by_variant.png`
- `results_validation/checkpoint_trajectory_summary.csv`
- `results_validation/checkpoint_gap_trajectory.png`
- `results_validation/checkpoint_broad_log_odds_trajectory.png`
- `results_local_search/local_search_summary.csv`
- `results_local_search/local_search_top_log_odds.png`
- `results_local_search/local_search_gap_scatter.png`
- `results_local_winner_validation/local_winner_summary.csv`
- `results_local_winner_validation/local_winner_log_odds.png`
- `results_local_winner_validation/local_winner_gap_trajectory.png`
- `results_low_neutral_search/low_neutral_summary.csv`
- `results_low_neutral_search/low_neutral_top_o_mass.png`
- `results_low_neutral_search/low_neutral_tradeoff.png`
- `results_low_neutral_winner_validation/low_neutral_winner_summary.csv`
- `results_low_neutral_winner_validation/low_neutral_winner_persona_mass.png`
- `results_lower_neutral_search/low_neutral_relaxed_ranking.csv`
- `results_lower_neutral_search/low_neutral_top_o_mass.png`
- `results_lower_neutral_search/low_neutral_tradeoff.png`
- `results_lower_neutral_search/low_neutral_best_trajectory.png`
- `results_lower_neutral_winner_validation/low_neutral_winner_summary.csv`
- `results_lower_neutral_winner_validation/low_neutral_winner_runs.csv`
- `results_lower_neutral_winner_validation/low_neutral_winner_persona_mass.csv`
- `results_lower_neutral_winner_validation/low_neutral_winner_log_odds.png`
- `results_lower_neutral_winner_validation/low_neutral_winner_persona_mass.png`
- `results_alpha0_lower_neutral_validation/low_neutral_winner_summary.csv`
- `results_alpha0_lower_neutral_validation/low_neutral_winner_runs.csv`
- `results_alpha0_lower_neutral_validation/low_neutral_winner_persona_mass.csv`
- `results_alpha0_lower_neutral_validation/low_neutral_winner_log_odds.png`
- `results_alpha0_lower_neutral_validation/low_neutral_winner_persona_mass.png`

## Sweep results

The best single-seed clean/no-leak variant was:

$$
\alpha = 0,\quad
p_s^{\mathrm{persona}} = 0.8,\quad
p_s^{\mathrm{domain}} = 0.7,\quad
\epsilon_{\mathrm{persona}} = 0.02,\quad
\epsilon_{\mathrm{domain}} = 0.02.
$$

It had:

$$
\ell_D = 4.590,\qquad
\ell_O = 4.497,\qquad
\ell_D - \ell_O = 0.094.
$$

Raw levels were:

$$
P(S_M \mid D) = 0.0367,\quad
P(S_A \mid D) = 0.00037,
$$

$$
P(S_M \mid O) = 0.0448,\quad
P(S_A \mid O) = 0.00050.
$$

This is the cleanest minimal result: no cross-special leakage and no extra
hidden structure. The main change from the hard symmetric model is that the
domain special state is less persistent than the persona special state.

The sweep also found single-seed broad-greater-than-narrow variants when a
small wrong-special leak was allowed. The strongest was:

$$
\alpha = 0.01,\quad
p_s^{\mathrm{persona}} = 0.95,\quad
p_s^{\mathrm{domain}} = 0.7,\quad
\epsilon_{\mathrm{persona}} = 0.02,\quad
\epsilon_{\mathrm{domain}} = 0.04.
$$

It had:

$$
\ell_D = 3.791,\qquad
\ell_O = 4.014,\qquad
\ell_D - \ell_O = -0.224.
$$

Raw levels were:

$$
P(S_M \mid D) = 0.0400,\quad
P(S_A \mid D) = 0.00090,
$$

$$
P(S_M \mid O) = 0.0391,\quad
P(S_A \mid O) = 0.00071.
$$

This is closer to the desired qualitative pattern, but it relaxes the hard
model: the factors are no longer perfectly separated at the emission level.

## Validation results

The three-seed validation changed the picture. Broad-greater-than-narrow was
not stable across seeds, but the gap could be reduced substantially.

Mean validation results:

| variant | mean ell_D | mean ell_O | mean gap ell_D - ell_O | mean P(S_M|O) | mean P(S_A|O) | loss gap | probe R^2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard_symmetric | 4.950 | 3.683 | 1.267 | 0.0375 | 0.00124 | 0.00198 | 0.991 |
| persona_persistent_no_leak | 4.775 | 4.104 | 0.671 | 0.0369 | 0.00066 | 0.00221 | 0.989 |
| clean_equalized_no_leak | 5.103 | 4.041 | 1.062 | 0.0476 | 0.00099 | 0.00229 | 0.990 |
| broad_dominant_weak_leak | 3.926 | 3.543 | 0.383 | 0.0239 | 0.00066 | 0.00288 | 0.989 |

Interpretation:

- The hard symmetric model reliably learns the process and gets excellent
  representational posteriors, but it keeps a large narrow-over-broad gap.
- `persona_persistent_no_leak` is the best clean/no-leak validated candidate:
  it halves the broad/narrow gap relative to the hard symmetric baseline while
  keeping broad raw misalignment high.
- `broad_dominant_weak_leak` gives the smallest mean gap, and one seed has
  broad greater than narrow, but the raw broad misalignment level is lower and
  the broad-greater-than-narrow result is not seed-stable.
- The single-seed clean equalization result did not hold up under the
  three-seed, longer-base-training validation. It is still interesting, but not
  a reliable final proposal yet.

## Fine-tuning trajectory

The validation runs also saved checkpoint metrics at fine-tuning steps 1, 5,
10, 20, 50, and 100. This matters because a fixed 100-step endpoint can hide
whether a variant is genuinely transferring broadly or simply moving through a
transient regime.

Mean D-over-O gap by step:

| variant | step 1 | step 5 | step 10 | step 20 | step 50 | step 100 |
|---|---:|---:|---:|---:|---:|---:|
| hard_symmetric | 0.173 | 1.018 | 1.058 | 0.900 | 1.085 | 1.267 |
| clean_equalized_no_leak | 0.150 | 0.838 | 0.840 | 0.751 | 0.824 | 1.062 |
| persona_persistent_no_leak | 0.283 | 0.876 | 0.572 | 0.452 | 0.531 | 0.671 |
| broad_dominant_weak_leak | 0.067 | 0.737 | 0.736 | 0.657 | 0.464 | 0.383 |

The clean variants tend to widen again late in fine-tuning. The weak-leak
variant is qualitatively different: after the early rise, its narrow/broad gap
keeps shrinking through step 100. That motivated a local search around the
weak-leak family.

## Local weak-leak search

I ran a second, narrower three-seed local search around the weak-leak family.
This pass used 700 base-training steps, skipped the expensive probe fits, and
focused on raw D/O prompt levels after 100 MD fine-tuning steps. The search
varied:

$$
\alpha \in \{0.005, 0.01\},\quad
p_s^{\mathrm{persona}} \in \{0.95, 0.98\},\quad
p_s^{\mathrm{domain}} \in \{0.6, 0.7\},\quad
\epsilon_{\mathrm{domain}} \in \{0.04, 0.06\},
$$

with $\epsilon_{\mathrm{persona}}=0.02$.

The best local-search variant was:

$$
\alpha = 0.005,\quad
p_s^{\mathrm{persona}} = 0.98,\quad
p_s^{\mathrm{domain}} = 0.7,\quad
\epsilon_{\mathrm{persona}} = 0.02,\quad
\epsilon_{\mathrm{domain}} = 0.04.
$$

Across three seeds it had:

$$
\ell_D = 3.793 \pm 0.016,\qquad
\ell_O = 3.771 \pm 0.080,\qquad
\ell_D - \ell_O = 0.022 \pm 0.078.
$$

Raw levels were:

$$
P(S_M \mid O) = 0.0290,\quad
P(S_A \mid O) = 0.00066,
$$

$$
P(S_M \mid D) = 0.0373,\quad
P(S_A \mid D) = 0.00084.
$$

This was the closest local-search result to stable broad/narrow equalization.
It is also less relaxed than the earlier weak-leak candidate because the leak
is only $\alpha=0.005$ rather than $\alpha=0.01$.

## Full validation of the local winner

I then reran the local-search winner with the full validation protocol:
1200 base-training steps, three seeds, 100 MD fine-tuning steps, checkpoint
trajectories, and residual-stream Moore-Penrose posterior probes.

The result held up:

$$
\ell_D = 3.934 \pm 0.135,\qquad
\ell_O = 3.878 \pm 0.152,\qquad
\ell_D - \ell_O = 0.056 \pm 0.184.
$$

Raw levels:

$$
P(S_M \mid O) = 0.0309,\quad
P(S_A \mid O) = 0.00065,
$$

$$
P(S_M \mid D) = 0.0369,\quad
P(S_A \mid D) = 0.00072.
$$

Base learning and representation quality remained good:

$$
\mathrm{CE}_{\theta} - \mathrm{CE}_{\mathrm{Bayes}} = 0.00297,\qquad
\bar R^2_{\mathrm{probe}} = 0.984.
$$

The full-validation checkpoint trajectory was also encouraging:

| step | mean ell_D | mean ell_O | mean gap ell_D - ell_O |
|---:|---:|---:|---:|
| 1 | 0.769 | 0.400 | 0.369 |
| 5 | 2.213 | 1.284 | 0.929 |
| 10 | 2.497 | 1.699 | 0.798 |
| 20 | 3.159 | 2.474 | 0.685 |
| 50 | 3.656 | 3.430 | 0.226 |
| 100 | 3.934 | 3.878 | 0.056 |

So the gap rises early, then collapses late in fine-tuning while the broad
O-domain misalignment ratio continues increasing. This is the closest toy
analogue so far to the desired broad-transfer pattern.

## Low-neutral follow-up

The first weak-leak winner still put most next-token mass on persona-neutral
emissions. At step 100, the full-validation O-prompt readout was approximately:

$$
P(S_M \mid O) = 0.031,\quad
P(S_A \mid O) = 0.00065,\quad
P(\mathrm{persona\ neutral}\mid O)=0.968.
$$

So the odds improvement was partly denominator-driven: aligned special mass
collapsed, while misaligned special mass only rose modestly.

To reduce this neutral share, I ran a targeted search varying only the
persona-factor frequency parameters around the weak-leak family. The winning
low-neutral config was:

$$
\alpha = 0.005,\quad
p_s^{\mathrm{persona}} = 0.95,\quad
p_s^{\mathrm{domain}} = 0.7,\quad
\epsilon_{\mathrm{persona}} = 0.10,\quad
\epsilon_{\mathrm{domain}} = 0.04.
$$

Full validation across three seeds gave:

$$
\ell_D = 4.462 \pm 0.159,\qquad
\ell_O = 4.385 \pm 0.483,\qquad
\ell_D - \ell_O = 0.078 \pm 0.448.
$$

The O-prompt mass accounting changed substantially:

$$
P(S_M \mid O) = 0.136,\quad
P(S_A \mid O) = 0.00181,\quad
P(\mathrm{persona\ neutral}\mid O)=0.862.
$$

Base learning and representation quality remained good:

$$
\mathrm{CE}_{\theta} - \mathrm{CE}_{\mathrm{Bayes}} = 0.00314,\qquad
\bar R^2_{\mathrm{probe}} = 0.995.
$$

This is a better behavioral analogue if we want absolute misaligned-token
probability to rise, not just odds. The cost is that the process is less
sparse: persona-special emissions are much more common by construction.

Using the relaxed criterion that broad and narrow can differ by about a factor
of two, I then pushed persona-special onset further. The best candidate in the
extended sweep was:

$$
\alpha = 0.005,\quad
p_s^{\mathrm{persona}} = 0.90,\quad
p_s^{\mathrm{domain}} = 0.7,\quad
\epsilon_{\mathrm{persona}} = 0.30,\quad
\epsilon_{\mathrm{domain}} = 0.04.
$$

Full validation gave:

$$
\ell_D = 4.955 \pm 0.138,\qquad
\ell_O = 5.117 \pm 0.323,\qquad
\ell_D - \ell_O = -0.161 \pm 0.435.
$$

So the broad O odds are slightly higher than the narrow D odds on average; the
odds factor is $\exp(-0.161)\approx 0.85$, well inside the factor-of-two band.
The O-prompt mass accounting was:

$$
P(S_M \mid O) = 0.486,\quad
P(S_A \mid O) = 0.0030,\quad
P(\mathrm{persona\ neutral}\mid O)=0.511.
$$

The model still learned the process and represented posteriors well:

$$
\mathrm{CE}_{\theta} - \mathrm{CE}_{\mathrm{Bayes}} = 0.00225,\qquad
\bar R^2_{\mathrm{probe}} = 0.997.
$$

The step-100 number is not the minimum neutral mass along the fine-tuning
trajectory. In the full-validation trajectory, the O-prompt neutral share drops
to about 0.29-0.30 at steps 5-10, then rises back to about 0.51 by step 100:

| FT step | P(S_M mid O) | P(S_A mid O) | P(neutral mid O) | ell_O |
|---:|---:|---:|---:|---:|
| 1 | 0.368 | 0.194 | 0.438 | 0.680 |
| 5 | 0.683 | 0.0248 | 0.292 | 3.419 |
| 10 | 0.693 | 0.00849 | 0.299 | 4.413 |
| 20 | 0.566 | 0.00588 | 0.428 | 4.575 |
| 50 | 0.543 | 0.00350 | 0.453 | 5.056 |
| 100 | 0.486 | 0.00300 | 0.511 | 5.117 |

This is currently the best low-neutral result without changing the prompt
form. The best low-neutral operating point is probably an earlier fine-tuning
dose, not the final 100-step checkpoint. Going lower at the final checkpoint
probably means either pushing $\epsilon_{\mathrm{persona}}$ even closer to its
maximum, changing the neutral readout prompt, or evaluating over a completion
horizon rather than a one-step next-token distribution.

I then reran the same high-onset process with no wrong-special leak
($\alpha=0$). This removes the main concern that token odds are corrupted by
leakage between \(S_M\) and \(S_A\). The alpha-zero full validation was
stronger:

$$
\ell_D = 5.980 \pm 0.123,\qquad
\ell_O = 5.874 \pm 0.267,\qquad
\ell_D-\ell_O = 0.106 \pm 0.146.
$$

At step 100:

$$
P(S_M\mid O)=0.590,\quad
P(S_A\mid O)=0.00168,\quad
P(\mathrm{persona\ neutral}\mid O)=0.408.
$$

The base loss gap was \(0.00259\), and mean posterior-probe \(R^2\) was
\(0.998\). This should replace the weak-leak high-onset result as the current
best process.

Files for the extended low-neutral pass:

- Search driver: `../../experiments/special_sfp_low_neutral_search.py`.
- Full-validation driver: `../../experiments/special_sfp_validate_low_neutral_winner.py`.
- Search summary: `results_lower_neutral_search/low_neutral_summary.csv`.
- Relaxed factor-of-two ranking:
  `results_lower_neutral_search/low_neutral_relaxed_ranking.csv`.
- Search plots:
  `results_lower_neutral_search/low_neutral_top_o_mass.png`,
  `results_lower_neutral_search/low_neutral_tradeoff.png`, and
  `results_lower_neutral_search/low_neutral_best_trajectory.png`.
- Full-validation summary:
  `results_lower_neutral_winner_validation/low_neutral_winner_summary.csv`.
- Full-validation per-seed rows:
  `results_lower_neutral_winner_validation/low_neutral_winner_runs.csv`.
- Full-validation mass accounting:
  `results_lower_neutral_winner_validation/low_neutral_winner_persona_mass.csv`
  and `results_lower_neutral_winner_validation/low_neutral_winner_persona_mass.png`.
- Full-validation odds plot:
  `results_lower_neutral_winner_validation/low_neutral_winner_log_odds.png`.
- Alpha-zero full-validation summary:
  `results_alpha0_lower_neutral_validation/low_neutral_winner_summary.csv`.
- Alpha-zero per-seed rows:
  `results_alpha0_lower_neutral_validation/low_neutral_winner_runs.csv`.
- Alpha-zero mass accounting:
  `results_alpha0_lower_neutral_validation/low_neutral_winner_persona_mass.csv`
  and `results_alpha0_lower_neutral_validation/low_neutral_winner_persona_mass.png`.
- Alpha-zero odds plot:
  `results_alpha0_lower_neutral_validation/low_neutral_winner_log_odds.png`.

## Mechanistic read

The gap equalizes when D-specific local evidence decays faster, or when the
aligned denominator rises similarly across D and O prompts. In the clean
variants, this is controlled mostly by relative special-state persistence:
making domain specials less sticky than persona specials weakens the
D-conditioned advantage without introducing cross-component emissions.

Broad transfer grows when the M-persona special direction is more persistent
and reusable across domains. This is why higher `p_s_persona` tends to help
the O-domain misalignment ratio.

Wrong-special leak can shrink the gap further, but it can do so for a bad
reason: it raises aligned-special denominators and lowers the absolute level of
the misaligned signal. That is why raw levels need to be reported alongside
ratios.

## Recommended next HMM choice

If the priority is a clean minimal model, use:

$$
\alpha = 0,\quad
p_s^{\mathrm{persona}} = 0.9,\quad
p_s^{\mathrm{domain}} = 0.8,\quad
\epsilon_{\mathrm{persona}} =
\epsilon_{\mathrm{domain}} = 0.02.
$$

This corresponds to `persona_persistent_no_leak`. It preserves the hard
factorized structure and gives a robust reduction in the narrow/broad gap.

If the priority is matching the qualitative broad-transfer phenomenon more
closely, the best current candidate is the local-search weak-leak variant:

$$
\alpha = 0.005,\quad
p_s^{\mathrm{persona}} = 0.98,\quad
p_s^{\mathrm{domain}} = 0.7,\quad
\epsilon_{\mathrm{persona}} = 0.02,\quad
\epsilon_{\mathrm{domain}} = 0.04.
$$

This nearly equalizes broad and narrow misalignment across three seeds in the
full validation protocol. It is less pure than the hard model because
emissions are no longer perfectly component-specific, but the relaxation is
small and the raw levels do not collapse.

If the priority is reducing neutral readout mass and making the broad signal an
absolute probability shift, use the lower-neutral winner:

$$
\alpha = 0.005,\quad
p_s^{\mathrm{persona}} = 0.90,\quad
p_s^{\mathrm{domain}} = 0.7,\quad
\epsilon_{\mathrm{persona}} = 0.30,\quad
\epsilon_{\mathrm{domain}} = 0.04.
$$

It keeps the broad/narrow log-odds nearly equal while raising
$P(S_M\mid O)$ from about $0.031$ to about $0.486$ and reducing neutral mass
from about $0.968$ to about $0.511$.

## Next experiments

The next pass should optimize for raw levels, not only ratios:

$$
P(S_M \mid O) \text{ high},\qquad
P(S_A \mid O) \text{ low},\qquad
\ell_D - \ell_O \leq 0.
$$

Three concrete directions look worth running:

1. Matched-dose fine-tuning: compare variants when the D-prompt misalignment
   level reaches the same target, rather than always stopping at 100 steps.
2. More seeds for `a0p005_psp0p98_psd0p7_ep0p02_ed0p04`, since three seeds are
   enough for a strong lead but not enough for final claims.
3. Local grid around the new weak-leak winner, with more seeds and raw-level
   constraints in the score.

The cleanest current story is not "we reproduced broad emergent
misalignment"; it is that the simplified SFP HMM can learn the process, expose
high-quality component posteriors in the residual stream, and show controlled
gap shrinkage under interpretable changes to factor persistence. A
scale-model-like broad transfer signal is now stable in the best three-seed
run, but in this toy family it currently needs a small emission-level
relaxation.
