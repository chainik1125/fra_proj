# Writing examples and anti-patterns

## Contents

1. [Cold-start summary](#cold-start-summary)
2. [Define shorthand](#define-shorthand)
3. [State prior context](#state-prior-context)
4. [Lead with the practical result](#lead-with-the-practical-result)
5. [Explain why a control exists](#explain-why-a-control-exists)
6. [Use analogies structurally](#use-analogies-structurally)
7. [Write findings instead of a diary](#write-findings-instead-of-a-diary)

## Cold-start summary

**Avoid**

> Finetuned corrections install a template-faithful exit while entry stays flat, and corrected mass
> rather than diversity controls the 7B dose response.

This requires the reader to decode “template-faithful,” “exit,” “entry,” “corrected mass,” “7B,” and
“dose response” before learning the result.

**Prefer**

> We tested whether training examples in which a bad answer stops and corrects itself can prevent a
> narrow misaligned fine-tune from spreading to unrelated questions. The corrections teach models
> to interrupt bad answers, but the same number of ordinary aligned examples prevents the spread
> about twice as effectively in Qwen2.5-7B.

The setup, intervention, comparison, model, and conclusion are present without internal shorthand.

## Define shorthand

**Avoid**

> We measure entry, exits, pivots, and both channels.

**Prefer**

> For each answer we ask two questions. Does it start misaligned? We call that probability
> **entry**. If it starts misaligned, does it interrupt and correct itself? We call that an
> **exit**, and the visible correction a **pivot**.

Use the technical shorthand freely after this paragraph.

## State prior context

**Avoid**

> This inverts the prior sprint's recommendation.

**Prefer**

> The previous sprint recommended correction examples because they reduced broad misalignment. It
> had not compared them with ordinary aligned examples, which produce a larger reduction here.

The second version lets the argument stand alone.

## Lead with the practical result

**Avoid title**

> Two suppression channels: corrective transitions install a generalizing exit while aligned data
> changes entry.

**Prefer title**

> Teaching models to self-correct works, but ordinary aligned data prevents the spillover better.

Put the internal decomposition in the body after the reader knows why it matters.

## Explain why a control exists

**Avoid**

> The mass-matched control scores 0.0625.

**Prefer**

> Each correction is roughly half aligned text, so corrections might work only because they contain
> aligned tokens. We therefore trained on the matching amount of plain aligned text; it reduced
> broad misalignment to 0.0625, showing that the correction wrapper added no benefit on this metric.

State the alternative explanation before the control and interpret the result at the strength it
supports.

## Use analogies structurally

**Good**

> Every correction example carries poison and antidote: its bad first half promotes bad starts,
> while its correction teaches the model to stop. The threshold question is which effect accumulates
> faster.

The analogy is the mathematical structure: one object contains two competing rates.

**Avoid**

> The correction circuit crystallizes into a powerful safety shield.

“Crystallizes” and “shield” add untested connotations. Say when the correction behavior appears and
what it changes.

## Write findings instead of a diary

**Avoid**

> First I tried three seeds. Then I debugged the classifier. After that I ran a larger model.

**Prefer**

> **The effect replicates across three fine-tunes but weakens at the larger model size.** The 7B runs
> produce ..., while the 14B run produces .... Classifier calibration changes absolute values but
> preserves this ordering.

Keep chronology and debugging details in `research_log.md`; organize `summary.md` by what was learned.

## Graph caption test

**Avoid**

> Results for all conditions.

**Prefer**

> Broad misalignment after matched narrow fine-tuning. Each point is one seed; bars show the mean.
> Plain aligned examples reduce broad misalignment more than correction examples at the same example
> count, while both conditions retain comparable narrow-task performance.

The caption names the population, metric, matching rule, uncertainty unit, result, and relevance.
