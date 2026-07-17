# Research sprint protocol

## Contents

1. [Scientific standard](#scientific-standard)
2. [Phases of work](#phases-of-work)
3. [Time and budget control](#time-and-budget-control)
4. [Experiment and job discipline](#experiment-and-job-discipline)
5. [Coordination](#coordination)
6. [Research log](#research-log)
7. [Pivot and stopping rules](#pivot-and-stopping-rules)
8. [Completion checklist](#completion-checklist)

## Scientific standard

Run a sprint as a small, self-contained research investigation:

- Learn something that was not obvious without evidence.
- Prefer one decisive result to ten weakly related experiments.
- Ask whether an experiment distinguishes live hypotheses before running it.
- Begin with simple checks: inspect data, talk to the model, plot raw outputs, and reproduce a
  baseline.
- Treat negative results as useful when they eliminate a plausible hypothesis.
- Match the strength of each written claim to the evidence: existence proof, suggestive pattern,
  systematic result, or guarantee.
- Search actively for bugs, leakage, cherry-picking, circular metrics, weak baselines, and
  alternative explanations.
- Re-run or independently implement the most load-bearing result when time permits.

Keep a living hypothesis table:

| Hypothesis | Prediction | Discriminating test | Status | Evidence |
|---|---|---|---|---|
| H1 | ... | ... | live/refuted/supported | artifact path |

Do not silently rewrite a hypothesis after seeing the result. Label post-hoc explanations.

## Phases of work

### 1. Orientation

- Read the kickoff, repository instructions, existing summaries, and the minimum code needed to
  understand the current baseline.
- Reproduce or inspect the best prior result before extending it.
- Write the initial hypothesis table and identify the main skeptical objection.
- Make the cheapest end-to-end run work before scaling.

### 2. Exploration

Maximize information gained per unit time. Use small samples, low-cost models, toy settings, and
diagnostic plots. Every 30–60 minutes, ask whether the direction is still teaching you something.

Exploration may be messy; the log must not be. Record why each branch was opened and what closed it.

### 3. Understanding

Turn promising observations into claims that could be wrong. Prioritize:

- strong baselines;
- controlled comparisons;
- ablations;
- held-out prediction;
- multiple seeds where noise matters;
- qualitatively different evidence;
- tests designed around the strongest alternative explanation.

### 4. Distillation

Update the report throughout the sprint. Save candidate figures when results land. Near the
deadline, freeze experiments and organize the report by findings rather than chronology.

## Time and budget control

At kickoff, record:

- local start time and timezone;
- absolute deadline;
- writing-block start;
- hard monetary cap;
- launch reserve, normally 10% of the cap unless specified otherwise.

At least hourly, add a log entry with elapsed time, remaining time, estimated spend, live jobs, and
the next decision. Check more often near the writing block or when expensive jobs run.

Estimate cost before launch. Track actual or conservative estimated cost after launch. Do not count
unused budget as a reason to run low-value work. Do not start a job that is unlikely to finish before
the writing block unless it is already part of the agreed handoff.

The user may authorize a compute service without authorizing arbitrary uploads or publication.
Keep model/data licenses and secrets out of logs and command output.

## Experiment and job discipline

Give each experiment:

- a question;
- predicted outcomes under competing hypotheses;
- a minimal configuration;
- a seed and versioned input identity;
- a unique output directory;
- a machine-readable result file;
- an explicit pass/fail or interpretation rule.

Before a large run:

1. Execute a smoke test.
2. Check output schemas and plots.
3. Estimate runtime and cost from the smoke test.
4. Confirm checkpoints and resumability.
5. Record the full launch configuration.

For remote jobs, preserve the job ID and enough state to resume monitoring after context turnover.
Poll with session/wait mechanisms and keep user-facing updates concise. Avoid unattended local
sleeps as a monitoring strategy.

When comparing methods, match the relevant resource: examples, tokens, optimizer steps, compute,
training progress, or parameter budget. State which quantity is matched and why.

## Coordination

Use parallel agents only when the work is separable and the environment permits it. Good tasks:

- independent literature verification;
- a second implementation of a key calculation;
- examination of a different experimental condition;
- adversarial review of a draft;
- cold-read interpretation of figures.

Poor tasks:

- asking several agents for generic opinions;
- having multiple agents edit the same file concurrently;
- delegating the core synthesis;
- assigning an agent to wait without an independent decision to make.

Give reviewers raw artifacts rather than the desired conclusion. Record which feedback changed the
work.

## Research log

Use append-only entries such as:

```markdown
## 2026-07-17 16:20 KST — Toy calibration contradicts H1

- Elapsed / remaining: 2h / 8h
- Question: Does initialization-time overlap predict the one-step transfer?
- Action: Ran ... with seed ..., config ..., artifact ...
- Result: ...
- Interpretation: Supports/refutes/leaves ambiguous H1 because ...
- Checks: ...
- Spend: incremental ..., total ...
- Live jobs: ID ..., expected finish ...
- Decision: Stop/pivot/replicate/scale because ...
- Next: ...
```

Log dead ends with the same care as successes. Link exact artifacts rather than saying “results
looked good.” Keep a short artifact index near the top when the sprint produces many files.

## Pivot and stopping rules

Pivot when:

- the phenomenon fails a basic existence or replication check;
- a simpler explanation accounts for the result;
- the metric is invalid or circular;
- two iterations yield no new information;
- the remaining time cannot produce a defensible result;
- the expected information value is lower than a live alternative.

Continue when a surprising result survives a cheap sanity check and a decisive follow-up is
available.

Stop a line of work when its preregistered falsifier occurs. A post-hoc replacement hypothesis may
be explored, but label it as new.

## Completion checklist

- `start.md` records scope, deadline, budget, and permissions.
- `research_log.md` covers decisions, dead ends, spend, and provenance.
- `summary.md` begins with a cold-start executive summary.
- Headline numbers agree with saved results.
- Figures have legible axes, units, legends, and self-contained captions.
- Baselines and matching choices are explicit.
- Limitations name the strongest remaining threat.
- Reusable code has a smoke test or relevant test result.
- Artifact paths, branch, and commit identifiers are listed when available.
- External compute is stopped or explicitly handed off.
- Proposed work is not described as completed work.
