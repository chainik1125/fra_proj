# Codex Cloud launch prompt

Use the repository skill `$run-research-sprint` and its complete references. Execute a
10-hour autonomous research sprint under
`experiment_folders/em_lambda0_predictor_sprint_20260717/start.md`.

First, set exact UTC and Asia/Seoul start/deadline/writing-only timestamps in
`start.md`; inventory the branch, dirty state, existing artifacts, Modal identity,
named-secret presence (names only), budgets, and live provider resources; then append
the first `research_log.md` entry. Do not print or retrieve any credential value.

Read these context files before freezing the metric:

- `experiment_folders/em_afp_simpler/em_afp_writeup/mine/em_afp_simple_writeup.tex`
- `experiment_folders/em_afp_simpler/em_afp_writeup/project_summary_2026-07-05/em_project_summary.tex`
- `experiment_folders/em_afp_simpler/em_afp_writeup/scaling_and_payoff_strategy_2026-07-17.md`
- `experiment_folders/em_lambda0_sprint/scripts/lambda0_toy.py`

Treat all existing results as prior/retrospective evidence. Freeze a prospective toy
holdout and, if feasible, a Qwen replica/config outcome before viewing its downstream
score. Judge-free metrics are primary. Use the `em-sprint-judges` Modal secret only for
a small, frozen semantic-validation sample; run GPT-4o-2024-08-06 first and a pinned
Claude snapshot as a sensitivity check. Anthropic is a judge only, never a research
agent. Log failed/parsed calls and judge disagreement.

Use Modal first. A paid GPU launch requires a smoke-derived time/cost estimate,
stopping rule, artifact path, and job ID. Activate RunPod only under the fallback rule
in `start.md`, through `cloud/modal_runpod_fallback.py`; tear down and verify status in
all exit paths. Persist small artifacts in the branch and large/recoverable artifacts
on named private Modal volumes with retrieval commands in the log.

At least hourly, update elapsed/remaining time, spend, live resources, artifact index,
what changed, and the next discriminating decision. Stop experiments at T+9h. In the
last hour, independently verify headline values, perform a cold-read and adversarial
review, finish `summary.md`, audit credentials/resources, and commit/push only this
branch. If the agent runtime ends early, leave a resumable `research_log.md` with exact
live job IDs and next commands rather than overstating completion.
