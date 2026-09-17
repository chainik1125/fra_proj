# Search protocol, frozen before intervention search

Date: 2026-09-17. Duration: 17:16 UTC to 03:16 UTC the next day.

## Question and scope

Find a realistic contextual semantic-filtering task on which content-gated FRA QK
cuts outperform a strong single-feature SAE intervention. This is an existence
search. Every screened task and unsuccessful search remains in the research log.
The contexts change a retrieved rule, not the model's weights.

Initial tasks are helpdesk routing, two-campus routing with a legitimate use of
the same printer/wireless conjunction, and a longer version of that document.
Clean and poisoned documents differ by one queue-name token. The continuation
distribution is evaluated after the full incident, at `Queue:`. The reference is
always the same incident and clean document, without any intervention.

## Splits and metrics

Calibration chooses heads, SAE features, and FRA pairs. Tuning chooses one setting
per family and repair threshold. Test evaluates those frozen settings. A final
confirmation split uses the six new device/fault paraphrase pairs already in
`data.py`; its results cannot select a configuration. New task families, if
needed, must write their confirmation inputs before intervention search.

Primary metric: full-vocabulary KL(P(clean document) || P(poisoned document +
intervention)), accumulated in float64. Also report raw poisoned-target token
probability, clean-target probability, excess-target repair, raw-target
suppression, restricted queue accuracy, full-vocabulary top-token accuracy, and
probability mass on valid queues. Report target and control cases separately.

Tuning selection: minimum mean KL among settings with control accuracy >= 95%,
at raw-target suppression thresholds 0%, 50%, and 90%; additionally report the
unconstrained minimum-KL setting. A candidate win means >= 20% lower confirmation
mean KL than the strongest eligible SAE family at the same prespecified repair
threshold, with no more than 2 percentage points loss of target accuracy and
control accuracy >= 95%. If either method misses its requested threshold on
confirmation, report that explicitly and do not call that comparison a win at
that threshold. Report all thresholds, including failures. Paired bootstrap
intervals resample lexical blocks, retaining their layouts and factorial cases.

## SAE comparator

Required family: top ten features per SAE site by mean positive poisoned-minus-
clean activation at the final answer position, ranked on calibration target
cases; sweep signed coefficients. The reconstruction error is retained:
`x' = x - c * z_f(x) * W_dec[f]`, at every token. Thus c=0 is exactly the original
model, and an inactive feature can have zero intervention effect. That does not
make paired clean-reference KL zero if the poisoned document changes output.

Strengthen this comparator with top-ten factorial/conjunction features and
top-ten features by first-order reduction of paired calibration KL, aggregated
over all token positions. The strongest comparator is the union of these
candidates. If additive steering or further rankings are explored, include them
in the strongest comparator and label the required family separately. Both
methods receive every SAE site used by the search. Released Gemma Scope weights
are encoded natively, without per-token renormalization.

The full sweep also tests constant single-decoder-vector subtraction at every
position, with signed coefficients -256 through +256 (the exact grid is stored
in each result). This is included in both the difference-ranked family and the
strongest union comparator. It does not require that the feature activate.

## FRA implementation

Each chosen pair is a query SAE feature and a key SAE feature in a specified
attention head. Rank pair terms using calibration contribution magnitude,
positive contribution, contrast against legitimate-use controls, and/or the
gradient of paired KL. Actual interventions apply selected pair terms at all
causally allowed token pairs, gated only by their SAE activations. Oracle source
positions are permitted for localization/diagnostics, never as the FRA test-time
gate. Oracle row/head ablations are diagnostic comparators only.

For residual x, native SAE features z and decoder D, write x = zD + b + e.
The pair term uses the actual residual RMS denominator, folded Q/K projections,
the model's attention scale, and RoPE at the actual positions. Removing a feature
pair preserves all bias/error cross terms. Gemma soft-caps attention scores:
subtract the selected raw bilinear terms BEFORE that soft-cap. Compare with the
legacy after-soft-cap subtraction if useful, and disclose the distinction.

Start with single-layer cuts to avoid stale later-layer activations. Any reported
multi-layer content-gated result must recompute features after earlier edits.
Validate c=0, cached-prefix versus ordinary full-forward predictions, and the
Q/K decomposition numerically. Nonfinite settings are recorded and excluded,
never replaced with zero KL. Retain SAE reconstruction-quality audits.

## Operational bounds and deliverables

At most two A100-80GB jobs, 18 aggregate GPU hours and $60 total; each call has a
two-hour timeout. Stop resources after use. Reserve the last hour for summary,
figures and review. Commit small reproducible source, compressed measurements,
and figures; every git blob must remain below 1,000,000 bytes. Push the dedicated
branch as requested earlier in this session.
