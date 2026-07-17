# Writing research sprint reports

## Contents

1. [The narrative](#the-narrative)
2. [Evidence and calibration](#evidence-and-calibration)
3. [Cold-start writing](#cold-start-writing)
4. [Executive summary](#executive-summary)
5. [Figures](#figures)
6. [Full report structure](#full-report-structure)
7. [Writing process](#writing-process)
8. [Final edit](#final-edit)

## The narrative

Compress the sprint into one to three concrete claims that form one coherent story. For each claim,
answer:

- What changed in our knowledge?
- Why should the reader care?
- What is the strongest evidence?
- What would make the claim false or narrower?

The report exists to make the reader understand, remember, and correctly calibrate these claims.
Include interesting side results only when they support the narrative; otherwise place them in a
brief appendix or artifact map.

Write to inform rather than sell. A modest claim with decisive evidence is stronger than an
ambitious claim with hidden caveats. State what is novel and what comes from prior work.

## Evidence and calibration

For every headline claim:

- name the experiment that supports it;
- explain why the experiment distinguishes alternatives;
- report a concrete effect size, uncertainty, or exact observation;
- include the strongest baseline;
- disclose whether the claim preceded the result;
- state the main limitation near the claim, not only at the end;
- provide enough method detail to audit or reproduce it.

Aggressively red-team elegant stories. Ask: Could the evidence be true while the claim is false?
Look for bugs, leakage, proxy failure, confounding, cherry-picking, and weak matching. Include
negative evidence that materially bounds the conclusion.

## Cold-start writing

Assume the reader knows the field but not this project. Titles, TL;DRs, executive summaries, and
figure captions must stand alone.

- Define the phenomenon and intervention before naming internal metrics.
- Replace author-invented shorthand with its plain meaning.
- If shorthand is necessary in the body, define it once: plain description first, name second.
- State prior conclusions rather than referring vaguely to “the previous sprint.”
- Use one idea per sentence when introducing the result.
- State definitions and claims directly. Avoid constructions that begin by saying what an object is
  not.
- Prefer precise everyday words to decorative analogies. Use an analogy only when it carries the
  actual causal or mathematical structure.
- Remove self-congratulatory throat-clearing such as “clearly,” “simply,” “importantly,” and “stated
  plainly.”

Plain language may be longer than jargon. Optimize for one-pass comprehension, not minimum word
count.

## Executive summary

Keep the executive summary self-contained and normally under 600 words. Use two to five findings.
A useful format is:

```markdown
## Executive summary

1. **Plain finding.** One or two sentences defining the setup and result.
   Quantitative evidence and scope condition.

   ![Caption that states what is plotted and why it supports the finding.](figure.png)

2. **Second plain finding.** ...
```

The summary should convey:

- the problem and why it matters;
- what was tested;
- the most important findings;
- the decisive evidence;
- the practical or conceptual implication;
- the main caveat.

Do not open with the chronology of the sprint, implementation details, or a compressed pile of
undefined technical nouns.

## Figures

Design each figure around one sentence: “The reader should learn that ___.”

- Choose axes and normalization that reveal that relationship.
- Label axes with plain names, units, and denominators.
- Use readable ticks, legends, and captions.
- Show uncertainty, seeds, sample counts, or raw points when relevant.
- Emphasize the key series and visually de-emphasize context series.
- Use colorblind-safe encodings; do not rely only on red versus green.
- Use a zero-centered diverging scale when zero is meaningful and a sequential scale for nonnegative
  magnitudes.
- Avoid panels that require reading prose to discover what quantity changed.

Fresh-reader test: give the graph and caption alone to someone without project context. Revise if
they cannot identify the comparison, result, and relevance.

## Full report structure

A robust sprint report uses:

1. **Title:** the most important finding in plain language.
2. **Executive summary:** the complete cold-start story.
3. **Question and hypotheses:** what was unknown and what outcomes were expected.
4. **Methods and measurement:** enough detail to understand the evidence and its limits.
5. **Findings:** one section per claim, with motivation, test, result, and interpretation together.
6. **Checks and alternative explanations:** baselines, ablations, replications, and failures.
7. **Limitations:** what the evidence does not establish.
8. **Next experiment:** the highest-information continuation, not a generic wish list.
9. **Artifact map:** code, configs, results, figures, logs, branch, and commit identifiers.

Organize the body by claims rather than by the order in which experiments happened. Preserve useful
tacit knowledge—what was hard, fragile, or misleading—in a clearly labeled section or appendix.

## Writing process

Use compress-then-expand:

1. List everything learned.
2. Select one to three defensible claims.
3. Match each claim to its strongest evidence and strongest objection.
4. Draft the executive summary as bullets.
5. Design the headline figures.
6. Outline the full report.
7. Expand into prose.
8. Verify every number and artifact link.
9. Run cold-read and adversarial reviews.
10. Edit for clarity, calibration, and narrative priority.

Spend disproportionate care on the title, executive summary, and figures: they receive most of the
reader's attention.

## Final edit

Search the draft for:

- undefined project-specific terms;
- claims without adjacent evidence;
- references to prior work that should be restated;
- phrases that announce clarity or importance instead of demonstrating it;
- causal language supported only by correlation;
- “all,” “always,” “proves,” or “scales” without sufficient evidence;
- headline metrics missing denominators or uncertainty;
- proposed experiments written in the past tense;
- numbers inconsistent with artifacts;
- figure captions that omit setup, metric, or interpretation;
- chronological sections that obscure the actual findings.

End with an honest, decision-relevant takeaway. The report may conclude that the hypothesis failed;
make the failure legible and useful.
