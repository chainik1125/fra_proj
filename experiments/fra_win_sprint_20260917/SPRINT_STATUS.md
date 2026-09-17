# Sprint handoff — 2026-09-17 21:29 UTC

## Objective and deadline

User requested a ten-hour search for realistic compound semantic-filtering tasks
where FRA beats a strong single-feature SAE baseline. Deadline: September 18,
03:16:27 UTC. Reserve 02:16–03:16 for writing, reviewer agents, push and cleanup.
Latest user steering: “go on.” No primary confirmation has been evaluated yet.

Worktree: `/tmp/fra-win-sprint-20260917`.
Branch: `dmitry/fra-win-10h-20260917`; base `11020620`.
The original `iclr-summary` workspace contains unrelated files; leave it alone.
User has authorized pushing results. Every committed file must be <1,000,000 bytes.

## Running jobs

1. `baseline_extra_tenants_long`, A100 app `ap-TPZKfdZnEgXKfwkUI7adT2`,
   started 20:08:58. Last ~21:25: 470/1085 batches. Timeout around 22:09;
   resume the same stage with unchanged baseline_extra.py, scopes.py,
   operators.py, data.py, variants.py and common.py. It imports more completed
   main-sweep measurements on resume. Log: results/baseline_extra_tenants_long.log.
2. `variant_search_narrative_contracts --fast`, H100! app
   `ap-1ROh8HM0ZjYmFWENO25Kpu`, started ~21:20. Last ~21:25: 30/1074 batches.
   Keep variant_search.py, variants.py, data.py, operators.py, common.py and
   PROTOCOL.md unchanged for resume. Log: variant_search_narrative_contracts_h100.log.
   The short A100 pilot was archived remotely as search_narrative_contracts_a100_pilot.json.

Modal results volume: `fra-win-sprint-20260917`; cache:
`fra-semantic-single-feature-cache`. Use modal_runner.py --stage STAGE;
`--fast` requests H100!, otherwise A100-80GB. Maximum one of each concurrently,
18 aggregate GPU app-wall hours, $60. Latest ledger ~5.36 hours, $17.29 conservative
cost estimate (includes queue time, not a billing report). Run update_ledger.py.
Never stop other users' apps. Every call has a two-hour timeout.

## Completed primary-task results

- Initial search: DONE, 7,408 settings, 8 invalid. Initial single-layer/heuristic
  FRA is not a robust win. Best strong SAE tuning KL ~.0372; exploratory test
  .0597, 59.1% suppression, 7/8 targets, 53/56 controls (below 95%).
- Gradient-only pair ranking: test KL .0836, 75.2% suppression, 52/56 controls.
- Learned pair selection: overall `global_S256_P48`, strength 8. Tuning KL .00553,
  94.6% suppression, all 32 answers correct. Exploratory test KL .00847,
  91.8% suppression, 8/8 targets, 55/56 controls. It repairs all five poisoned
  target errors; three targets were already correct. 48 pairs, 26 same-ID Q/K,
  five layers, 23 heads, 48 unique feature endpoints.
- Distinct-ID candidate `distinct_S64_P48`, strength 16: tuning KL .00624,
  90.3% suppression, all 32 answers correct. 48 distinct-ID pairs, six layers,
  23 heads, 56 endpoints. Distinct IDs do not prove independent semantic concepts.
- Preconfirmation diagnostics on 32 exploratory-test cases: overall full cut
  KL .01097 / 92.2% suppression, no-BOS .00730 / 91.8%, BOS-only 5.3%.
  Shuffling K features within layers gives 10–21% suppression. Source-row-only
  and final-query-only edits do not reproduce the repair. Zero new control errors
  versus the clean model; the one shared-control error is already a clean error.
  Distinct full cut reaches only 88.4% suppression on this subset, missing 90%.
- All six zero-cut audits are exact. All-head RoPE/GQA audit relative error <.0008.
  SAE reconstruction error is retained; main KL compares same clean document
  with poisoned document plus intervention, full vocabulary, float64.

## Task screens

Original long two-campus table chosen after initial screens. Variants contracts,
named_offices and narrative_contracts pass the clean behavior screen; short
contracts and card tasks do not. Narrative task is prose with extra legitimate
local exceptions, and currently receives the second search. Archived-card and
reformatted-card attempts all remain negative screen results, not hidden.

## Next stages and priorities

Primary task: finish/resume baseline_extra → baseline_nobos → polish → confirm.
Then run robustness_tenants_long and, if time permits, multi_sae_tenants_long.
The 48-feature capacity diagnostic is separate from the requested single-feature
comparison. It now has three scopes (all/document/no_bos), six training runs.

Second task on H100: main search → variant_refine → variant_learn_pairs →
baseline_abs → baseline_extra2 → baseline_nobos → polish → confirm, all with
suffix narrative_contracts. This full pipeline may exceed the remaining window;
prioritize completing the primary task's fair comparator and confirmation.
An incomplete second comparison must be labeled exploratory, never a confirmed win.

baseline_nobos tests all positions except BOS, retaining every candidate and both
signed steering modes. Activation measurements are reused only when exactly
inactive at BOS on every tuning case. polish adds two local dense grid rounds.
confirm requires all six sources completed, freezes selected configurations and
source hashes before constructing 96 fresh confirmation cases (6 lexical blocks,
2 layouts, 8 corners: 12 targets, 84 controls). It compares both overall and
all-distinct-ID FRA to diff-ranked and strongest SAE at None/0/50/90% thresholds.
Report any confirmation threshold failure. Include clean-reference introduced
control errors and block-bootstrap uncertainty, not accuracy alone.

## Artifacts and writing

No summary.md yet. Final deliverable should explain the context, QK operator,
selection procedure, strongest SAE feature/scope/strength, sample counts,
confirmation table, negative attempts, compound-mechanism limitations and cost.
Use sprint writing guidance; final-hour red/blue reviewers are explicitly allowed
by that skill. No agents have been spawned for research/code work.

Local result names differ from remote stage output names: e.g. local
variant_search_narrative_contracts vs remote search_narrative_contracts;
confirm_TASK vs confirmation_TASK; baseline_extra2_TASK vs baseline_extra_TASK.
analyze.load reads ordinary gzip or .partNNN.json.gz JSON-fragment shards.
pack_results.py repacks without losing measurements and verifies JSON SHA256.
Full main sweep now uses 24 shards, largest 783,888 bytes. Completed learned
result is 914,347 bytes. Do not truncate results to satisfy the hook.

Plots: plot_oracle.py, plot_mechanism.py, plot_confirmation.py. Plot Python:
`/tmp/fra-sprint-plotenv/bin/python`. Mechanism plot was visually checked;
confirmation plots need real results and visual QA. Confirm/no-BOS/polish/capacity
code is syntax checked but has not run yet. Preserve hashes required by resumes.

## Update — 22:14 UTC (supersedes running-job details above)

Original extra SAE app timed out after batch 830/1085; best valid tuning KL .03153.
Resume2 A100 app: ap-SML6GCvZbazmkMIsG92OMV, local exec session 98216, log
baseline_extra_tenants_long_resume2.log. Resume1 failed network access and created
no app. New permissions are workspace-write plus restricted network; /private/tmp
is writable. The scoped commands below are now approved for network use:

- python experiments/fra_win_sprint_20260917/update_ledger.py
- python experiments/fra_win_sprint_20260917/launch_stage.py

Use launch_stage.py --stage STAGE [--fast] [--log-name UNIQUE_NAME] for subsequent
stages. It saves logs and checks at most one A100 plus one H100, <=18 GPU hours,
<=60 dollars even if both slots continue until 02:16 UTC. It refuses to overwrite
logs. The current H100 app remains ap-1ROh8HM0ZjYmFWENO25Kpu; main SAE portion is
finished, and initial FRA settings are running. Latest ledger ~$23.31,6.89 app hours.

Added pair_budget.py before any confirmation: at budgets 1/4/16/48/144 and tuning
repair50/90%, choose overall and distinct-ID cuts, replay tuning, freeze, evaluate
all on confirmation. Run as pair_budget_tenants_long after primary confirmation
if time permits. Existing learned results: no 1-pair choice reaches50%; distinct
4-pair is weak (KL .0944), distinct16-pair tuning KL .01296 with86.7% suppression.
Capacity curve, robustness, and multi_sae are secondary; keep the main SAE-vs-FRA
comparison first. No agents have been spawned.

Commits be475d32 and 58b291e8 pushed after 91a1586b. Full main result repacked to24
shards, largest783,888 bytes, with round-trip JSON SHA256 verification. Figures
source_oracle and mechanism_global visually checked. plot_confirmation now shows
KL, suppression and control correctness together. summary.md still pending.

22:16 UTC addition: baseline_nobos.py now completes BOTH no-BOS and document
scopes over the full feature union. The original extra stage only covered a
source/endpoint subset in document scope; this fills that gap. No result from
baseline_nobos exists yet, so resume hashes are unaffected. It prefers completed
baseline_extra as the ranking/import source when available, then baseline_abs.
Its progress prefix is now SAE SCOPES SWEEP. Expect a longer stage; prioritize
this fair primary comparison over optional auxiliary work if time becomes tight.

## Update — 22:40 UTC

- Extra SAE primary DONE:10,016 points,12 invalid. Best50% point L22 f14233,
  activation c64, global, pooled-abs ranking: test KL .030269,80.2% suppression,
  targets8/8,controls54/56. Original main remains in final pool for90% selection.
- Narrative main DONE:7,248 points,8 invalid. Initial FRA4pairs atL22,c64 has
  test KL .05934 /98.5% suppression, controls54/56; strongSAE50 has KL .03860,
  88.2% suppression,controls55/56. No initial win. Local source repacked24 shards.
- H100 variant_refine_narrative_contracts: app ap-NJZRnAr4QHOAc9hxX0wKlF,
  exec8153. Slow model-volume load (~163seconds), then normal progress; at22:40
  set10/45,419seconds. Next variant_learn_pairs_narrative_contracts --fast.
- Scope union was larger than expected:308features,119 active atBOS,1,592 new
  batches atbatch8. Stopped A100 no-BOS/document pilot ap-0YG1X9X9cIJgXfNc9YyTl0
  after checkpoint10/1592. Old remote baseline_nobos_tenants_long.json is intact.
- Replacement stage baseline_scopes_tenants_long now starting onA100, exec60022,
  log baseline_scopes_tenants_long.log. It imports the old checkpoint (hash recorded)
  and all other available measurements. Profiles batch8/16/32 onidentical tuning
  probes, requires numerical agreement and memory<70GiB, chooses fastest eligible
  batch, preserves that choice onresume. This changes throughput, not grid or
  intervention definitions. All final selected settings still get directreplay.
- Use baseline_scopes for SECOND task too. confirm.py andpolish.py prefer completed
  baseline_scopes source when present, otherwise oldbaseline_nobos. launcher allows
  the new stage. Do not edit its hashed dependencies while running/resume needed:
  baseline_scopes.py,scopes.py,operators.py,data.py,variants.py,common.py,search.py,
  analyze.py,frozen.py. batch8 pilot file remains unchanged. No confirmation yet.
- Latestledger7.73app hours,$26.63. Primaryextra result repacked30shards,largest457106.
  Optional plot_pair_budget.py added,not yet run because its result is pending.
