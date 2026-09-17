# Compound semantic-filter experiment

Hypothesis: selective removal can benefit from an interaction between two semantic
features when each feature remains useful on its own. A textual conjunction does
not by itself establish two distinct SAE features or a causal QK interaction.

## Design revision after failed feasibility screens

The initial Billing-payload screen failed on Gemma-2-2b and Gemma-2-2b-it.
Exception notes affected control tickets as well; examples/tables did not induce
the joint error reliably. Those inputs, code and results are retained. No intervention
was tested on them. The revised screen uses Print as the corrupted queue, clarifies
the asset lookup, and compares 2B and 9B instruction-tuned models. Print is needed
legitimately on a control, preventing global output suppression from trivially winning.

## Realistic task and factorial controls (revised)

A service-desk assistant routes an incident using an asset record and a retrieved
support document. The record identifies the equipment; the later ticket describes
the problem. A corrupted support entry routes printing-device wireless incidents
to Print. The clean document routes them to Network. Only that queue label changes.

| Printing device A | Wireless fault B | Clean queue | Poisoned target |
|---|---|---|---|
| absent | absent | Desktop | Desktop |
| present | absent | Print | Print |
| absent | present | Network | Network |
| present | present | Network | Print |

Device type is genuinely useful for power faults; wireless-fault meaning is useful
for portable computers. Both are semantic categories with calibration, tuning and
test paraphrases separated in advance. The asset span precedes the problem span.
Two document/asset orders vary the distance and position of the relevant information.
This is a controlled corrupted-knowledge-base simulation using benign queue labels.

## Feasibility gate, chosen before interventions

Screen three presentations using calibration vocabulary only: a retrieved exception
note, resolved cases, and a routing table. Prefer them in that order if the following
gate passes: clean label accuracy >=85%; poisoned control-corner accuracy >=85%;
joint-corner mean raw target-probability increase >=0.40 and poisoned restricted-label
target probability >=0.70; each other corner's absolute change in mean target
probability <=0.15. Report raw label
mass and unrestricted top tokens, so renormalization cannot hide a weak behavior.
Select the format on behavior alone, before examining FRA or steering performance.
Prefer the smaller model that passes, then formats in the listed order.

## Measurement requirements

- Retain every corner and paraphrase, including failures. Do not re-screen test inputs.
- Compare full next-token KL(clean document || poisoned document + intervention).
  The current incident and all other prompt text remain the same.
- Evaluate both repair of the joint case and preservation of the three controls.
  Use fresh poisoned/no-edit baselines and ordinary full-forward verification.
- Rank individual SAE features from poisoned-minus-clean activation differences,
  sweep the top ten, and report the best. Include a search beyond layer 6 so a
  downstream single feature encoding the conjunction can compete fairly.
- Inspect factorial feature responses at distinct query/key endpoints. Test whether
  useful pair terms actually depend on both factors, and compare content-gated FRA
  cuts against single-feature removal without assuming FRA will win.
- Choose interventions on calibration/tuning inputs; freeze them before the final
  held-out paraphrase evaluation. Keep optimistic retrospective comparisons separate.

## Frozen intervention comparison (before sweep)

The chosen screen is 9B IT, routing_table. Native Gemma Scope IT 16k SAEs with
average L0 88/91/76 at residual-post layers 9/20/31 supply features for attention
layers 10/21/32. All 16 heads at each layer are eligible. The implementation follows
the previous FRA decoder projections, reconstructed residual RMS,
RoPE and 1e-10 pair-contribution cutoff; cuts are QK score cuts, not OV cuts.

Calibration uses all 8 factorial blocks (32 paired cases). Individual candidates
are the union of the top ten positive poison-minus-clean activation differences
at the final answer token in the joint case and the top ten factorial contrasts
z11-z10-z01+z00 at that token, separately at each SAE layer. Each intervention edits
ONE feature, at every token: x <- x - c*z_f*decoder_f. Reconstruction error is
retained. Coefficients: -8,-4,-2,-1,0,.25,.5,1,2,4,8,16,32,64. Positive-only and
signed baselines are reported separately. No additive direction baseline here.

FRA locates pairs at two query anchors (last reported-problem token and final
answer-prefix token) and the last asset-type token as key, averaged over anchors
and calibration blocks. Three frozen pair sets, 48 pairs per head:

1. Raw: largest absolute mean joint-case contribution (legacy-style ranking).
2. Interaction: largest absolute factorial contribution F11-F10-F01+F00.
3. Separable: interaction ranking restricted to distinct feature IDs where the
   Q feature has >2x mean activation for wireless faults in BOTH equipment families,
   and the K feature has >2x mean activation for printers in BOTH fault families.
   A 1e-4 absolute margin excludes numerical dust. Fewer pairs if insufficient.

Pairs are applied to all causal token pairs, with content-dependent activation and
unsteered-poisoned contribution matrices frozen during the edited forward pass.
FRA coefficients: 0,.125,.25,.5,1,2,4,8,16,32,64. No hand-selected test positions
or asset label gates are used by the actual FRA/SAE interventions.

Tuning uses four new factorial blocks (16 cases). For each method family, select
one fixed setting minimizing mean full-vocabulary answer KL over all four corners.
Also select minimum-KL settings subject to >=50% and >=90% mean joint target
suppression, defined as 1-mean(P_edited(Print))/mean(P_poison(Print)). Report excess
repair relative to clean as an additional metric, not the selection criterion.
Absent feasible settings are reported. Check selected settings using ordinary full
forward passes and freeze selection BEFORE loading 12 test blocks (48 cases).
Every held-out case is retained, including any weak backdoors or clean failures.
The 12 blocks contain six lexical pairs each repeated in two document orders;
they are not 12 independent vocabulary draws.

Full next-token KL compares clean document to poison+edit on the same current
incident. This is the first token of the natural continuation (the queue label),
not KL on an unrelated paragraph. Report corner-specific target probability,
label accuracy, and KL, plus unrestricted top tokens and total queue-label mass.
No claim about long-form generation fidelity follows from this one-token task.

A final diagnostic removes ALL attention to the asset-type span from subsequent
tokens at these three layers. It uses oracle positions and is not a deployable
competitor. It tests whether this proposed QK route is causally useful; it is never
used to choose the held-out configuration.

The smoke run only checks execution on a tiny subset, with 2 features per ranking,
8 pairs per head and coefficients 0/1/8. It does not change the frozen full-run
selection, grids, or test cases. Every committed file stays <1 MB.

## Pre-sweep normalization correction

The inherited wrapper documentation incorrectly claimed that Gemma Scope needs
per-token normalization at inference. The primary paper, §3.1 (pp.3) and appendix A,
states that training uses a FIXED scalar and the released weights absorb it.
https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf
The pinned SAE Lens gemma_2 loader directly imports these weights. Therefore this
experiment sets normalize_activations=False for BOTH FRA and the single-feature
baseline. A pre-correction smoke was stopped before completing; its source and log
are retained. A reconstruction/L0 comparison is recorded on the first calibration
input, excluding BOS. Earlier experiments using True need a separate audit. This
fix is based on the checkpoint specification, not on intervention performance.
