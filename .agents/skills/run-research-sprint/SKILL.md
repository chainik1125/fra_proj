---
name: run-research-sprint
description: Run a rigorous, autonomous, wall-clock research sprint that lasts multiple hours or overnight and ends in a clear evidence-backed report. Use when the user asks Codex to start or conduct an overnight run, 10h/20h sprint, long unsupervised experiment, autonomous research session, or sustained investigation with compute, checkpoints, logging, and a final summary.
---

# Run Research Sprint

Run a long research task as a sequence of explicit scientific decisions. Treat the final
write-up as the main deliverable: the user should understand the claims, evidence, failures,
and artifact locations without reading the code.

## Load the guidance

Before starting:

1. Read [references/sprint-protocol.md](references/sprint-protocol.md) completely.
2. Read [references/kickoff-template.md](references/kickoff-template.md) and create or update the
   sprint's `start.md` from it.
3. Read [references/writing-instructions.md](references/writing-instructions.md) completely before
   choosing headline claims and again before the final writing block.
4. Read [references/writing-examples.md](references/writing-examples.md) before drafting the
   executive summary.

## Establish the contract

Translate the request into a kickoff containing:

- one concrete objective;
- the start time, absolute deadline, and timezone;
- hypotheses and observations that would falsify them;
- key files, existing baselines, and the branch/worktree policy;
- compute access, monetary budget, and a reserve below the hard cap;
- permitted external actions such as pushing, publishing, or uploading artifacts;
- required deliverables and completion criteria.

Fill reasonable omissions from repository context. Ask only when a missing choice would materially
change scope, cost, safety, or external state. A long sprint authorizes sustained in-scope work; it
does not authorize unrelated external writes or spending beyond the stated budget.

## Prepare durable state

Create these sprint-local artifacts unless equivalent files already exist:

- `start.md`: frozen kickoff and scope;
- `research_log.md`: timestamped decisions, results, dead ends, spend, and remaining time;
- `summary.md`: the reader-facing deliverable, written throughout and polished at the end;
- a machine-readable result or manifest file when experiments produce structured outputs.

Record the current branch and working-tree state before editing. Preserve unrelated user changes.
Create a dedicated branch only when requested or clearly authorized by the kickoff. Do not push,
publish, upload, or open a PR unless the kickoff explicitly authorizes it.

When the product exposes persistent goals and the user explicitly requested an ongoing sprint,
create a goal for the concrete sprint objective. Do not set a token budget unless the user supplied
one. Keep the goal active until the deliverables are genuinely complete.

## Run the research loop

Cycle through three modes:

1. **Explore:** maximize information gain with cheap sanity checks, inspection, and small runs.
2. **Understand:** state competing hypotheses, run discriminating experiments, and test alternative
   explanations.
3. **Distill:** update the claims, figures, caveats, and summary while evidence is fresh.

At every decision point:

- state what question the next action answers;
- prefer the simplest experiment that separates the live hypotheses;
- inspect raw examples and intermediate outputs, not only aggregate metrics;
- compare against strong, fairly tuned baselines;
- distinguish preregistered predictions from post-hoc interpretations;
- reproduce or independently check load-bearing results;
- record negative and inconclusive results honestly;
- pivot when the current path stops producing information.

Use subagents, when available, for bounded independent work such as literature checks, alternative
implementations, adversarial review, or fresh-context graph interpretation. Give them only the
task-local context they need. Keep experimental synthesis and final responsibility with the root
agent. Do not assign multiple agents to babysit the same process.

## Maintain wall-clock and compute discipline

Treat duration as real wall-clock time. Record elapsed and remaining time at least hourly and at
every major transition. Communicate concise progress updates during active work so the user is not
left without an update for more than about a minute.

For every long job, record:

- command or entry point;
- configuration, seed, dataset/model versions, and output directory;
- start time, expected duration, and expected cost;
- session/job identifier and recovery procedure;
- success signal, failure signal, and stopping rule.

Use wait/session tools to monitor live work rather than blocking sleeps. Stop launching new paid
jobs when the remaining budget cannot cover the expected job plus the reserve. If GPU work is
needed and a dedicated compute skill is available, invoke and follow it.

## Reserve the writing block

Reserve at least the final hour of a 10h-or-longer sprint, or roughly the final 10% of a shorter
sprint, for the report alone. Stop exploratory launches before this block unless one result is
essential and guaranteed to finish safely.

Write the executive summary around two to five findings. For each finding provide:

1. a plain claim a cold-start reader can understand;
2. the strongest quantitative evidence;
3. one self-explanatory graph when a graph materially helps;
4. the main limitation or scope condition.

State definitions and findings directly. Avoid author-invented vocabulary on cold-start surfaces.
Organize by claims, not by chronology. Include a concise artifact map and enough method detail to
audit the work.

Use a fresh-context review when possible:

- ask one reviewer to infer each graph's message from the graph and caption alone;
- ask another to find the strongest alternative explanation or overclaim;
- revise the report in response to concrete failures of understanding or evidence.

## Finish cleanly

Before declaring completion:

- verify headline numbers directly against saved artifacts;
- ensure every claimed run has a config, output, and provenance trail;
- distinguish completed, failed, and merely proposed work;
- report spend and any still-running external resources;
- stop resources that the kickoff authorized Codex to stop;
- run relevant tests or validation for changed reusable code;
- leave the worktree understandable and preserve unrelated changes;
- mark a persistent goal complete only after all required deliverables exist.

If the deadline arrives with incomplete work, deliver the strongest honest report possible. State
what remains, why it remains, and the single best next experiment. Never manufacture completion.
