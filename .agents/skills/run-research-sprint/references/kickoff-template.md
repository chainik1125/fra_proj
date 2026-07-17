# Sprint kickoff template

Copy this structure into the sprint's `start.md` and replace the placeholders. Remove sections that
do not apply. Freeze substantive scope changes in the research log rather than silently rewriting
the original objective.

```markdown
# Sprint kickoff — <short title>

## Time

- Duration: <N hours of wall-clock time>
- Start: <YYYY-MM-DD HH:MM timezone>
- Deadline: <YYYY-MM-DD HH:MM timezone>
- Writing-only block begins: <time, normally final hour or final 10%>

## Objective

<One concrete sentence describing the outcome to achieve.>

## Context

<What exists, the key prior finding, and the main open objection.>

## Hypotheses and falsifiers

| Hypothesis | Prediction | Observation that would refute or sharply narrow it |
|---|---|---|
| H1 | ... | ... |
| H2 | ... | ... |

## Priority order

1. <Prerequisite or cheapest decisive test>
2. <Main experiment>
3. <Replication/alternative explanation>
4. <Stretch work only after 1–3 are solid>

## Key files and prior artifacts

- Repository/worktree: <path>
- Current branch: <branch>
- Sprint branch policy: <stay/create branch name>
- Entry points: <paths>
- Best prior run: <path or identifier>
- Required repository instructions: <AGENTS.md and relevant skills>

## Compute and budget

- Compute route: <local/Modal/other>
- Hard cap: <$ amount>
- Launch reserve: <$ amount>
- Per-job or per-hour cap: <amount>
- Stop-new-launch threshold: <amount>
- Required compute skill: <name if applicable>

## External actions authorized

- Push commits: <yes/no and destination>
- Upload artifacts or weights: <yes/no and destination>
- Publish or open PR: <yes/no>
- Stop/delete remote resources: <scope>

## Deliverables

- `research_log.md`: timestamped decisions, dead ends, spend, and artifact index.
- `summary.md`: executive summary, findings, methods, limitations, next experiment, artifact map.
- Structured results: <JSON/CSV/checkpoints/figures and locations>.
- Code/tests: <expected reusable changes and validation>.

## How the work will be judged

<What would make the result interesting, rigorous, and useful. Name required baselines, replication,
or quantitative targets without dictating the answer.>

## Autonomy and communication

- Make in-scope scientific decisions autonomously.
- Ask only for scope, cost, safety, or external-state choices that cannot be inferred safely.
- Provide concise progress updates during active work.
- Record elapsed/remaining time and spend at least hourly.
- Use bounded parallel agents for independent work or review when useful and permitted.

## Completion criteria

<Concrete conditions for completion. Include what to deliver if the main hypothesis fails.>

## Writing requirements

- Reserve the writing block.
- Begin with two to five cold-start findings, each with quantitative evidence and a scope condition.
- Include a self-explanatory graph for each major finding when useful.
- State definitions and findings directly; define project-specific shorthand only after the cold-start
  summary.
- Verify every headline number against saved artifacts.
```
