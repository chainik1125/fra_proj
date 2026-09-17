# Search additions specified before confirmation

## 2026-09-17 20:02 UTC

The original sleeper-agent implementation ranks absolute mean SAE activation
differences at the trigger span (`multitrigger_sleeper/cloud/modeldiff_baseline_pod.py`,
line 570). The initial sweep here ranks positive final-answer differences. Add
absolute differences at four spans: final answer, changed token, source payload,
and all-token mean. Rank on calibration targets only; take ten per site per rank.
Also test all endpoints of the best completed FRA choice at each repair threshold.
These are additions to the comparator; keep every original candidate too.

Both signed activation scaling and constant decoder-vector steering are tested.
For source-ranked features and FRA endpoints, also allow steering within the
entire known retrieved document, including its legitimate rows. No bad-row-only
mask enters the primary FRA or single-feature comparator.

After all coarse SAE sweeps, select the twelve best distinct SAE configurations
(feature, site, activation/constant intervention, and scope) at each tuning repair
threshold. Around each selected strength, subdivide the interval between its
neighboring coarse strengths into sixteen equal intervals. Extend beyond an
endpoint by the previous spacing if the endpoint was selected. Repeat twice.
This is a local grid refinement, not a globally optimal continuous search.

Freeze the merged tuning selection from original search, gradient refinement,
learned pair selection, extra SAE candidates, and SAE strength refinement before
materializing confirmation prompts. Prefer independently replayed ordinary
full-forward SAE measurements when duplicate numerical configurations occur.
Fresh confirmation is still six lexical blocks × two layouts × eight corners.
Weighted FRA settings remain diagnostics and do not enter the primary selection.

A separate implementation audit will check every attention head at all six SAE
sites after RoPE and grouped-query mapping. `cards` was also specified before its
behavioral screen: poisoning changes a routing card's status from draft to active;
the literal Print-valued card remains legitimate for documentation lookup.

## 2026-09-17 20:08 UTC — separate distinct-feature comparison

The completed learned search's overall tuning winner is global_S256_P48 at
strength 8 (KL .00553). Of its 48 pairs, 26 use the same feature ID on Q and K.
The best distinct-ID candidate remains distinct_S64_P48 at strength 16 (KL .00624).
Add a separately labeled `fra_distinct` confirmation comparison, selected only
from tuning points whose pairs all have q != k. Include its endpoints in the
extra SAE baseline. Keep the overall winner as the primary comparison and report
both. Distinct feature IDs alone do not establish two independent semantic factors.

## 2026-09-17 20:59 UTC — intervention-capacity diagnostic

A 48-pair FRA edit changes many feature relationships across several layers. To
assess how much the one-feature restriction contributes to its apparent advantage,
prepare an additional learned SAE activation baseline with up to 48 feature
coefficients. Two pools use the FRA endpoints or the calibration-gradient ranking;
each also includes the best single-feature candidates. Test global, whole-document, and all-except-BOS scopes, fit 512 calibration steps with model and SAE weights frozen, then
select snapshot/scale on tuning. Keep this result separate from the requested
single-feature comparison. This tests six learned activation interventions; it
does not establish an optimum over every possible multi-feature SAE intervention.

Code and selection rules were specified before primary confirmation results.
Run it if the remaining compute/time permit. Its feature pools read only the
frozen tuning-selection manifest, and its own confirmation choices are saved
before its confirmation examples are materialized.

## 2026-09-17 21:20 UTC — BOS-free SAE scope and compute allocation

The single-feature comparator also tests all prompt positions except BOS. An SAE
feature can be useful on document/query tokens while behaving poorly on BOS.
This scope retains the query tokens that whole-document steering omits. It uses
the same signed activation/constant grids and all existing feature candidates,
including both selected FRA families' endpoints. Global activation measurements
are reusable only when the feature is exactly inactive at BOS on every tuning
case. Constant steering is remeasured. Add this completed source to the merged
selection before coefficient refinement and confirmation.

Allow one H100 alongside the existing A100 to fit the stronger comparisons into
the 10-hour sprint. This supersedes the GPU-type planning line in PROTOCOL.md;
the two-GPU, 18 aggregate GPU-hour and $60 limits remain. At published rates, with
four CPU cores and 64 GiB memory per job, the planning rates are $3.198528/hour for
A100 and $4.649328/hour for H100; combined peak $7.847856/hour. The remaining
pre-writing window still fits the total cap under this conservative estimate.
Prices verified at https://modal.com/pricing and GPU choices at
https://modal.com/docs/guide/gpu. Use H100! to keep the hardware type fixed.

The short narrative-task A100 pilot was stopped and its checkpoint archived as
search_narrative_contracts_a100_pilot.json on the result volume. Restarted that
search fresh on H100; no checkpoints mix A100/H100 measurements. The original
completed task and its remaining baseline runs continue on A100. The ledger now
tracks the two GPU rates separately and distinguishes the brief CPU archive job.

## 2026-09-17 21:39 UTC — pair-count curve

Add a secondary FRA comparison at fixed maximum pair counts 1, 4, 16, 48 and 144,
for both unrestricted and distinct-ID pair pools. Choose the best setting at
50% and 90% tuning suppression with at least 95% control accuracy. Replay tuning
through the ordinary forward path, then freeze all choices before this stage
constructs confirmation examples. Report the entire curve, including absent
eligible settings and confirmation failures. This checks how many pairs the
observed advantage requires; it cannot establish two independent semantic concepts.
Run after primary confirmation if time permits. Primary selections do not change.

## 2026-09-17 22:16 UTC — complete the document-scope candidate union

The initial document-scope sweep covers source-ranked features and selected FRA
endpoints. Extend this scope to every feature in the strongest SAE candidate
union, including original answer-difference, interaction and gradient rankings.
The baseline_nobos stage now completes both whole-document and no-BOS scopes,
reusing previously measured document configurations. Both signed steering modes
and their full grids remain. This stage has not run yet, and confirmation is
untouched. Its filename is retained for continuity; its metadata records both
scopes. After completion, all three scopes have the full candidate union.

## 2026-09-17 22:40 UTC — batch-size profiling for the complete scope grid

The complete union contains 308 features. With batches of eight, the scope stage
needs 1,592 additional batches; its early L10 batches take roughly 11 seconds each.
Stop that pilot after its checkpoint, retaining the file and importing its valid
measurements into baseline_scopes. The mathematical interventions, candidate union,
signed grids, tuning selection and ordinary-forward checks are unchanged.

Before the replacement sweep, benchmark batches 8/16/32 on the same 32 modest
activation-steering settings and two tuning examples. Compare predictions to
batch 8 and require maximum target-probability and per-case KL differences below
.025, with peak allocated memory below 70 GiB. Choose the fastest eligible batch
size, save timing/numerical/memory evidence, and keep that size on any resume.
Every final selected setting still receives ordinary-forward tuning replay.
Record the old checkpoint hash and per-point measurement batch sizes; preserve
all primary measurements and the stopped pilot. Confirmation remains untouched.

## 2026-09-17 22:54 UTC — diagnose the single-pair prose candidate

Prose-task gradient refinement selects one pair: layer17, head8, query feature12530
and key feature0, at strength32. On the exploratory64-case test it reaches KL
.03011 and91.6% suppression with8/8 targets and54/56 controls correct. Before
attributing this to two semantic factors, measure all-edge, no-BOS, BOS-only,
source-row and final-query variants and record endpoint activations and raw pair
magnitudes on the first32 exploratory test cases. The two distinct IDs alone do
not establish distinct concepts. baseline_extra2 runs this small diagnostic first
on the same H100 allocation, preserves it in the result, then runs the unchanged
SAE ranking/grid procedure. No confirmation inputs are used by the diagnostic.
Skip a separate baseline_abs stage: baseline_extra2 already computes exactly those
absolute-difference rankings and can operate without a prior abs result.

## 2026-09-17 23:26 UTC — background completion during the writing hour

Begin writing and review at02:16 UTC as planned. Allow the already specified
GPU pipeline to finish in the background until02:46 UTC, then stop our remaining
apps. This supersedes the earlier02:16 GPU-stop planning line. The final sprint
deadline remains03:16:27 UTC. No new search design starts during the writing hour.
The launcher can automatically continue a finished scope sweep through polish
and confirmation, using the frozen procedures already written. After02:16 it
allows only existing-stage continuation and these final evaluation stages.

The conservative budget check now assumes both existing GPU slots run all the
way to02:46. At23:25 the estimate is about$32.36 spent plus at most$26.3 remaining,
below$60; aggregate GPU app time remains below18 hours. Maximum concurrency and
GPU types remain one A100-80GB and one H100. The launcher stops its own app at the
absolute deadline. This preserves the full writing hour while allowing completed
measurements to populate the prepared figures and tables.
