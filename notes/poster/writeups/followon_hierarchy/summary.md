# Emergent misalignment survives a genuinely out-of-distribution test — and its magnitude is predicted by pretraining statistics alone

**TL;DR.** Emergent misalignment (EM) — fine-tuning on narrowly bad data (insecure
code, bad financial advice) making a model broadly misaligned — has a toy model in
the MATS poster (`RL_beliefs/TeX/mats_poster/main_36x24.pdf`): a tiny transformer
trained on a two-sector hidden Markov process reproduces EM's signature facts.
That model had a real objection hanging over it: its "broad" misalignment eval
could be dismissed as in-distribution for the fine-tuning data. We close the gap
by giving the toy world *domains*. We pretrain a 2-layer transformer on a
persona×domain hierarchical process, fine-tune it on text that is 100% (misaligned
persona, domain 1) tokens — domain-2 tokens never appear — and domain-2
completions become ~68% misaligned anyway. Two controls pin down the cause: with
the same fine-tuning data and the same eval, a pretraining process with no persona
grouping keeps domain 2 at baseline, and a process whose grouping is rewired sends
the misalignment to the rewired partner instead, leaving domain 2 *more* aligned.
The transfer also obeys a quantitative law registered before the experiments ran:
the ratio of cross-persona to cross-domain transfer equals
log β_persona / log β_domain, a ratio of the pretraining process's Bayes-evidence
exponents — confirmed across 14 runs spanning predicted ratios 0 to 6.6, with zero
free parameters. Mechanistically, the pretrained model carries a single persona
direction (theory says the optimal predictor must); a steering vector extracted
from the base model alone moves misalignment in every domain at once, 3%→48%.

---

## Executive summary

**Problem.** The poster showed that fine-tuning a tiny transformer on "misaligned"
text reproduces three signature facts of EM: narrow fine-tuning causes broad
misalignment, the change is carried by a few features, and steering those features
controls it. The strongest objection: in that flat two-sector world, the "broad"
eval used more text of the same single kind the model was fine-tuned on — a
skeptic reads that as ordinary in-distribution generalization. Real EM is striking
because bad financial-advice fine-tuning produces misaligned *cooking* advice. The
toy world needs domains, so the broad test can run on text fine-tuning never touched.

**Setup.** The pretraining process is a hidden Markov model with 2K sectors:
persona (aligned A / misaligned B) × domain (1..K). Prompts carry zero persona or
domain evidence. Completion tokens carry a (persona, domain) tag; each observed
tag downweights other-persona sectors by a factor β_persona and other-domain
sectors by β_domain. We fine-tune the pretrained transformer on purely
(B,1)-tagged completions and measure each tag group's probability on held-out
neutral prompts, writing B2 for tag (B,2). Domain-2 misalignment rate
= P(B2)/(P(A2)+P(B2)).

**Finding 1 — misalignment transfers to a domain fine-tuning never touched, and
pretraining structure sets where it goes.** After fine-tuning on pure (B,1) text,
domain-2 completions go from 50% to 68±4% misaligned (3 seeds) although no
domain-2 token ever appeared in fine-tuning. In all processes the fine-tuned
domain itself reaches ≥99% misaligned — every control received the same dose —
yet identical fine-tuning data on a *flat* process (same four sectors, no persona
grouping) leaves domain 2 at 52–54%, and a *scrambled* process (pretraining groups
B1 with A2) drives domain 2 down to 31%: transfer follows pretraining's
correlational grouping even against the token labels.

![Domain-2 misalignment vs fine-tuning steps for the three pretraining structures](figures/fact1_domain2_misalignment.png)
*Red: hierarchical pretraining sends untouched domain 2 to ~68% misaligned. Gray:
no persona grouping — domain 2 stays at baseline. Purple: rewired grouping —
domain 2 becomes MORE aligned. Right panel: the fine-tuned domain saturates
identically in all three.*

**Finding 2 — a quantitative, zero-free-parameter law, registered in advance.**
The measured transfer ratio matches a prediction computed from the pretraining
process alone: 1.38±0.17 measured vs 1.357 predicted (3 seeds). The prediction:
if fine-tuning acts as a Bayesian tilt of the model's internal sector prior along
the fine-tuning tokens' log-likelihood, the unknown dose (amount of fine-tuning)
cancels in R = (Δlog P_B2 − Δlog P_A2)/(Δlog P_A1 − Δlog P_A2) — Δ measured from
the pretrained model — leaving R = log β_persona/log β_domain. Swapping the two
β's flips the transfer ordering as predicted (R = 0.83, predicted 0.74), and
across 14 runs sweeping both β's, measured R tracks predictions from 0 to 6.6.

![Measured vs predicted transfer ratio across all runs](figures/fact1_R_invariant.png)
*Each point is one pretraining process; x is computed from its statistics with no
fit, y is measured after fine-tuning. Points hug the identity line.*

**Finding 3 — transfer rides a functionally privileged persona coordinate in the
model's geometry, beyond mere decodability.** The Bayes-optimal predictor for the
hierarchical process must carry one persona scalar shared across all domains; the
flat process provably admits none. Experimentally, weakening the pretraining
persona coupling (β_persona → 1) kills transfer monotonically down to the flat
floor, and a linear probe shows the persona coordinate degrading out of the
representation (R² 0.99 → 0.54 → 0.00). The flat model still *decodes* persona at
R² = 0.95 yet transfers nothing (0.04±0.02 vs 0.81±0.15 in log-odds): fine-tuning
gradients ride a privileged coordinate, and decoding alone supplies none.

![Transfer and persona-probe R-squared vs pretraining persona coupling](figures/fact1_dose_response.png)
*Top: misalignment transfer to domain 2 falls to the flat-control floor as the
pretraining persona coupling weakens. Bottom: a probe for the persona variable in
the base model degrades over the same range.*

**Finding 4 — one direction in the pretrained model steers misalignment in all
domains at once.** From the base model's activations alone we extract
near-orthogonal persona and domain axes. Adding the persona axis to the residual
stream on neutral prompts sweeps domain-2 misaligned-token probability from 2.8%
to 48%, moving both domains' misalignment together; the domain axis instead moves
probability between domains at fixed misalignment.

![Steering the base model along persona vs domain axes](figures/fact3_steering.png)
*Left: the persona axis raises B1 and B2 (misaligned, both domains) together.
Right: the domain axis exchanges domain-1 for domain-2 probability instead.*

**Takeaway.** In this minimal world, a persona is a coordinate that pretraining's
correlational structure forces into the optimal predictor, and emergent
misalignment is fine-tuning sliding the model along that pre-existing coordinate,
with magnitude set quantitatively by the pretraining statistics. This kills the
in-distribution objection: with the pretraining correlations removed, the same
fine-tuning data leaves other domains at baseline; with them rewired, it
misaligns the rewired partner instead.

**Main caveats** (details in Limitations): single tiny architecture; first-token
probability metrics (the generative whole-completion eval agrees at the
checkpoints tested); measured R undershoots predictions in the weak-coupling
regime β_persona ≥ 0.7, consistent with the Finding-3 coordinate degradation;
left open.

---

## Map of what was done

| Wall-clock | What |
|---|---|
| H0–1 | Read prior design notes (a 4-sector hierarchical process was sketched in `em_pipeline_notes/`); built the general builder (K domains, 3 evidence modes) + 17 unit checks; Modal harness; pure-sector fine-tuning stage; theory notes with predictions P1–P5 registered before results; battery (a silent no-op bug in the first version was caught by acceptance-rate logging and fixed); all of P1–P4 confirmed; β sweeps, seed replicates, steering, probe analysis, K=3; first full draft |
| H1–2 | Review fleet (numbers red-team, zero-context figure test, writing critique); figure and text revisions; wave-2 robustness runs (more seeds, slow-collapse protocol, pretrain-length × weak-coupling experiment) |
| H2–10 | Wave-2 analysis folded in; second adversarial review; final polish (see SPRINT_LOG.md for the hour-by-hour record, including the subjective-time drift caught at the first hourly debrief) |

Artifacts: `analysis/afp_builders.py::build_hierarchical_leaky_reset_hmms`,
`analysis/em_pipeline/hier_finetune.py`, `analysis/em_pipeline/hier_steering.py`,
`sprint_hierarchy/` (configs, Modal runner, analysis scripts, figures, this doc,
`SPRINT_LOG.md`, `theory_notes.md`). Figure filenames keep the poster's
"fact-N" numbering (fact1 = broad misalignment, fact3 = steering); this document's
Finding numbers are its own.

---

## The process, precisely

Hidden space: a direct sum of 2K sectors, persona s ∈ {A,B} × domain r ∈ {1..K}
("leaves" in the code), each a d=5-state block. Defaults: K=2, vocabulary
5 prompt + 2K·5 completion tokens, sequences = 3 prompt + 20 completion tokens.

- **Prompt phase** (leaky reset, identical in every sector, shared scalar 1/V_p):
  prompt tokens write within-sector beliefs μ' = (1−λ)μ + λr_k and carry exactly
  zero evidence about persona or domain. Eval prompts are persona-neutral by
  construction, so any post-fine-tuning preference is a weight-level prior shift
  (the prompts themselves carry no persona evidence to condition on).
- **Completion phase** (tagged evidence): a token tagged (s\*,r\*) multiplies
  sector (s,r)'s posterior mass by W[(s,r),(s\*,r\*)] · (content term), with
  W = P ⊗ D a Kronecker product: P = [[1, β_t],[β_t, 1]] over personas and
  D = (1−β_s)I_K + β_s J_K over domains. Here and in the code β_t = β_persona,
  β_s = β_domain. Three pretraining variants share everything except W:
  - `hierarchical`: W = P ⊗ D — personas group the domains;
  - `flat`: W = (1−β)I_{2K} + βJ_{2K} — four ungrouped sectors (β=0.55, the
    midpoint of the two hierarchical β's; the flat null is insensitive to this
    choice because every non-target sector shares the same evidence weight, so
    the predicted pairwise transfer gaps are zero for any β);
  - `scrambled`: hierarchical W with sectors A_r ↔ B_r exchanged for r ≥ 2 — an
    isomorphic process whose groups are {B1,A2} and {A1,B2} while token labels
    keep their meaning.
- Base regime is persona-dominant: β_t = 0.5 < β_s = 0.6, so a B1 token is
  stronger evidence for B2 (factor 0.6) than for A1 (factor 0.5). The flipped
  regime swaps them.

The process is strictly non-ergodic: all operators are block-diagonal,
hidden-state mass never crosses sectors; each sector is internally ergodic; the
full process is a 2K-component mixture of distinct ergodic processes. What
"leaks" within a sequence is the observer's posterior over components — Bayesian
filtering on a frozen chain. This also resolves the standing worry about the
original flat process: it was already truly non-ergodic in exactly this sense
(`theory_notes.md` §2).

## Theory in three sentences

With W = P ⊗ D and sector-symmetric content, the Bayes filter factorizes exactly:
persona log-odds θ move by ±log(1/β_t) per completion token (driven by the
persona tag alone), domain log-odds by ±log(1/β_s) (domain tag alone), within-
sector belief μ by content alone — so the minimal sufficient statistic is
(θ, φ, μ) and contains exactly **one** persona scalar shared by all K domains
(Proposition 1 + Corollary, `theory_notes.md`). For the flat process the minimal
sufficient statistic is the full exchangeable 2K-vector of sector log-odds:
persona exists in the token labels but has no privileged coordinate in the
predictive geometry (Proposition 2). Modeling fine-tuning on pure-B1 text as an
exponential tilt of the internal sector prior, Δlog p_ℓ = η·log W[ℓ,B1] + const,
yields the dose-free ratio invariant R = log β_t/log β_s and the orderings P1–P5
registered in `theory_notes.md` before any experiment ran.

## Protocol

- Model: 2-layer transformer (d_model 64, 2 heads, d_mlp 256), pretrained 5000
  steps (batch 128, lr 1e-3) on full process sequences. Sanity anchors: first-token
  sector probabilities on held-out prompts land at the Bayes prior 0.25±0.01 per
  sector, and a linear probe reads the joint belief state at R² ≈ 0.89.
- Fine-tuning: 6 of 125 prompts (5%); 50 completions each, every token from the
  (B,1) group (sampled from the process conditioned on the persona path: the
  post-prompt belief is projected onto B1, and within a sector the completion
  operator is diagonal, so content is iid from the readout — this is "text
  written by the misaligned persona in domain 1"). 1000–2000 Adam steps, lr 3e-4,
  batch 64. Purity is asserted at runtime: zero tokens from any other group.
- Eval: P(sector tag group) at the first completion position on the 119 held-out
  prompts, every 20–100 steps. Fine-tuning collapses P(B1)→0.99 within a few
  hundred steps, so transfer is read in log-odds gaps Δlog p_X − Δlog p_A2
  (well-defined after collapse) and at a **matched dose** = the first eval step
  with P(B1) > 0.5, which normalizes fine-tuning progress across processes. A
  generative eval (sampling whole 20-token completions, counting tag fractions)
  agrees with the first-token metric at the base and final checkpoints.

## Finding details and registered-prediction scorecard

| Prediction (registered before results, `theory_notes.md`) | Outcome |
|---|---|
| P1 base (β_t=.5, β_s=.6): ordering B1 ≫ B2 > A1 > A2; R ≈ 1.357 | ✓ ordering exact (3/3 seeds); R = 1.38±0.17 |
| P2 flipped (β's swapped): ordering B1 ≫ A1 > B2 > A2; R ≈ 0.737 | ✓ ordering flips; R = 0.83 |
| P3 flat: B2, A1, A2 change equally | ✓ gaps 0.04±0.02 (vs 0.81±0.15 hierarchical) |
| P4 scrambled: transfer to A2 (B1's pretraining partner), not B2 | ✓ ordering A2 > A1 > B2; mirrored R = 1.39 |
| P5 β_s sweep: persona gap constant in β_s; domain gap ∝ −log β_s | ✓ persona gap 0.6±0.15 flat across β_s=0.5–0.9 while the domain gap falls 0.74→0.05 |

Additional results:

- **Emergent alignment (mirror control).** Fine-tuning on pure (A,1) text spreads
  *alignment*: final ordering A2 > B1 > B2 with mirrored ratio 1.47. The law is
  persona-symmetric, as it must be if it reflects process geometry rather than
  anything special about misalignment.
- **K=3 domains.** Matched-dose ordering B3 ≈ B2 > A1 > A3 ≈ A2 — both predicted
  degeneracies appear and both untouched misaligned domains rise above every
  aligned sector; R = 1.25 at matched dose (1.48 at final step; base-seed spread
  is ±0.17).
- **Honest deviation.** For β_t ≥ 0.7 measured R undershoots predictions (0.21 vs
  0.44 at β_t=0.8): the model transfers *less* than the Bayes tilt when the
  pretraining persona signal is weak. The probe panel of the dose-response figure
  shows the persona coordinate degrading over the same range; at β_s=0.9 the
  denominator of R is near zero and that measurement is noise-dominated (11.7 vs
  6.6).
- **Fine-tuning-shift geometry (low-dose checkpoint).** The fine-tuning-induced
  activation shift across held-out prompts is 79% rank-1 (top singular direction
  carries 79% of it); its direction has cos 0.335 with the persona axis and 0.284
  with the domain axis, each above the 0.245 single-axis 95% chance level, with
  in-plane norm 0.44 vs 0.18 expected for a random direction; the persona:domain
  component ratio is 1.18, where the tilt direction predicts
  log β_t : log β_s = 1.357.

## Limitations

- One architecture, one scale; hyperparameters inherited from the poster runs.
- First-token probabilities are the primary metric; the generative eval ran at
  base/final checkpoints only.
- Matched dose sits on the eval grid (every 20–100 steps) and fine-tuning
  collapse is fast; R is stable between matched-dose and final-step readouts
  (1.40 vs 1.48 at base). A slower-collapse replication is in `runs/`.
- The tilt model is a heuristic for SGD and holds quantitatively in the
  persona-dominant regime; the weak-coupling undershoot shows its limit, and the
  hierarchical-vs-flat contrast (representation geometry) carries the qualitative
  mechanism.
- The scrambled control is an isomorphic relabeling — which is the point — so
  "domain 2 became more aligned" is a statement about our labels; the model
  transferred along its own pretraining groups.

## Reproducibility

- Process builder + invariant tests:
  `analysis/em_pipeline/test_hierarchical_process.py` (the worked posterior from
  the design note reproduces to 1e-4).
- One full run: `modal run sprint_hierarchy/modal_run.py --config
  sprint_hierarchy/configs/hier_base.yaml --stages 1,2,3 --gpu t4
  --ft-targets B1,A1 --steering` (≈4 min on a T4).
- All run metrics: `sprint_hierarchy/figures/transfer_analysis.json`,
  `probe_r2.json`; raw outputs under `sprint_hierarchy/runs/` (models <1MB each).
- Total compute ≈ $3 of the $150 budget (≈30 T4 container runs of 3–6 min).

---

*10h unsupervised wall-clock sprint, 2026-06-12, branch `dmitry/personas/hierarchy`
(the instructed branch name `dmitry/personas/error-correct` already holds the
2026-06-11 sprint). Hour-by-hour process log: `SPRINT_LOG.md`. Theory and
predictions registered before results: `theory_notes.md`.*
