# Research log

- 17:16 UTC: Began the user-authorized sprint. Read sprint guidance;
  created a dedicated worktree/branch to preserve the shared workspace.
- 17:19 UTC: Main hypothesis: the preceding FRA locator selected an unnecessary
  asset edge. Search the poisoned source entry and causally select heads first.
  Also search legitimate reuse controls so a single conjunction feature can have
  collateral cost. Preserve the original strong SAE comparison and corrected KL.

- 17:19 UTC: User corrected duration to 10 hours. Deadline is 2026-09-18
  03:16:27 UTC; reserve final hour for writing. Budget cap updated to $60, at most
  two concurrent A100 GPUs and 18 aggregate GPU hours.

## 17:30 UTC — initial behavior and implementation audit

All three initial contexts passed the calibration behavior gate. Clean queue
accuracy was 93.75% (routing), 87.5% (tenants), and 100% (tenants_long); poisoned
control accuracy was 100%. The shared-conjunction control therefore works best
in the long document. A single-head source-row ablation rarely removes the
backdoor, so grouped-head diagnostics are needed before attributing a mechanism.

TransformerLens 2.18.0 applies Gemma's attention soft-cap before hook_attn_scores.
The inherited code subtracts raw bilinear terms at that later hook. The new
implementation will subtract before the soft-cap and use the actual residual
RMS, retaining decoder bias and reconstruction-error cross terms. These are
implementation changes and will be audited, not silently treated as equivalent.

Additional causal sites from the source-row screen: attention layers 17, 22 and
28. Available pretrained-model residual SAEs at the preceding layers have native
L0 targets 75, 66 and 65, respectively. They will supplement the three IT SAEs;
reconstruction quality on the IT model must be measured, and SAE steering gets
these sites too.

## 18:42 UTC — smoke-test repair and compute accounting

The first search smoke test exited at its zero-cut audit: the replacement score
function omitted grouped-query attention's K-head expansion (16 Q heads versus
8 K heads). No intervention comparisons from that run were accepted. Added the
same repeat-interleave operation as TransformerLens's grouped-query module and
relaunched. The independent projection audit had already passed (~0.03% relative
Q/K RMS error). SAE reconstruction reporting now separates the anomalous BOS
position from ordinary tokens.

The initial screen app ran 17:22:28–17:28:07 UTC; the failed smoke app ran
17:35:57–17:37:15 UTC. Both are stopped. Combined app wall-time upper bound so far:
417 seconds = 0.116 GPU hours. Corrected smoke app started 18:41:22 UTC.

Verified published Modal rates (https://modal.com/pricing): A100-80GB $0.000694/s;
RAM $0.00000222/GiB/s; CPU $0.0000131/physical-core/s. 64 GiB RAM plus 4 physical
cores and one A100 would cost about $3.20/hour. The 18 GPU-hour limit leaves some
margin under the $60 cap at these resource levels. Exact bills may differ from
this conservative resource-time estimate.

## 18:47 UTC — full six-site sweep started

Corrected smoke test passed exact zero-cut logits and ordinary-forward checks.
SAE relative squared reconstruction error excluding BOS: 0.1409 at both L21 IT
and L28 PT-on-IT. The tiny smoke subset favors the conjunction SAE (mean KL
0.029 versus FRA 0.196), so it provides no win evidence. Full search now includes
all six sites, top-ten difference/interaction/gradient feature candidates, signed
activation scaling and constant steering, and 216 FRA pair sets spanning six
sites, three rankings, three head counts, and four pair counts. No confirmation
inputs have been evaluated. Renamed the dedicated branch to reflect ten hours.

## 19:20 UTC — full-position gradients and learned pair selection

Refinement ranked 11,339 candidate pairs by the derivative of paired KL across
all token positions, then swept 495 settings including live multi-layer cuts.
Zero-cut logits matched exactly. Its tuning winner was a single-layer L21 cut of
144 pairs, strength 2. Exploratory test: KL 0.0836, 75.2% raw-target suppression,
8/8 target answers correct, but only 52/56 controls correct (92.9%). This misses
the control-preservation requirement. The SAE search has already reached tuning
KL 0.0567 before its strongest native IT sites, so this is not a current win.

Next pair-identification method: optimize nonnegative pair coefficients on the
32 calibration cases, freezing the model and SAE weights. Two pools use 384
full-gradient-ranked pairs each: distinct feature IDs only, or all pairs. Fixed
snapshots at 64/128/256/512 steps select top 1/4/16/48/144/384 pairs. Primary
candidates then use equal coefficients and the same tuning grid as ordinary
cuts. The directly weighted intervention is saved as a diagnostic and excluded
from the primary FRA-vs-SAE selection. No confirmation model outputs have been generated
or examined. This extends pair identification; it is not evidence that the
original ranking procedure wins.

## 19:26 UTC — additional contexts specified before screening

Prepared contracts, contracts_short, named_offices, and narrative_contracts. The
contract variants make North printer hardware and South laptop connectivity
legitimately route to Desktop under local staffing rules. Thus deleting site or
fault information can damage other required lookups. Other variants replace
North/South with Cedar/Maple or express the rules as prose. Each paired prompt
still changes exactly one queue word; 32 calibration cases per variant. These
variants share the already frozen lexical splits. They have not been scored yet.
Separate copied entrypoints preserve the running search's numerical source hashes.

## 20:08 UTC — learned search completed, stronger baseline and audit

Learned search completed 504 settings. Overall tuning winner: global_S256_P48,
strength8, 48 pairs across five layers and23 heads, with48 unique endpoints.
26/48 pairs share their Q and K feature ID. Tuning KL .005532, suppression94.64%,
all32 answers correct. Exploratory test: KL .008472, suppression91.77%, all8
target answers correct,55/56 controls correct, and7/8 shared-conjunction controls
correct. Queue accuracy equals full-vocabulary top-token accuracy. The one wrong
shared-conjunction test case has clean Print probability .214 and edited Print
probability .385; the answer error must not automatically be attributed to FRA.
Independent clean predictions will be retained in confirmation and mechanism runs.

The distinct-ID tuning winner is still distinct_S64_P48 at strength16 (KL .006240,
suppression90.31%). Add it as a labeled secondary confirmation family and include
its endpoint features in the strongest SAE comparator. A1-pair cut never reaches
50% suppression with preserved controls in either learned pool; the best distinct
16-pair cut reaches KL .01296 and86.66% suppression. These are tuning diagnostics.

Full SAE sweep is still running; its best tuning KL is now .03708. Extra comparator
code is ready: absolute source/answer/pooled diffs, FRA endpoints, global versus
whole-document scopes, activation scaling and constant steering. Subsequent local
strength refinement will take12 distinct candidates per threshold and subdivide
neighboring intervals twice. No fresh confirmation outputs have been generated.

Extra implementation audit passed all16 heads at each of the six SAE sites.
Maximum relative rotated Q/K error was .000799; anchored pair contributions agree
with their entries in the full token-pair matrix. Variant screening is running.

## 20:53 UTC — first causal diagnostic

The main search hit its expected7200-second timeout after saving6,796 settings
(batch970/1094). App ap-DyfEfTOpUTjWjWfs3O1ml0 stopped at20:47:41. Use the free GPU
for the pre-confirmation diagnostics, then resume this search unchanged.

Global48-pair winner, first two exploratory-test lexical blocks (32 cases):
full edit KL .010968, suppression92.2%; excluding BOS edges KL .007298,
suppression91.8%. BOS-only edit suppresses5.3%. BOS accounts for only0.49% of
summed absolute raw pair-term magnitude. This candidate does not rely on the
anomalous SAE reconstruction at BOS.

Source-row-to-later-token cuts alone suppress27.9%; their complement suppresses
36.8%. Edges with either endpoint in the source row suppress30.1%. The complete
repair therefore uses distributed edits, rather than a single isolated row-read
operation. Removing layers17/21/22/28 reduces suppression to78.3/75.2/81.9/76.0%;
omitting the lone layer10 pair retains91.0%. Shuffling K endpoints within each
layer (five fixed seeds, same Q endpoints and head counts) gives only10.0–21.2%
suppression and KL .1197–.1559. Pair identity and several layers matter.

The clean model itself is wrong on one of the four shared-conjunction cases in
this subset. It is the same case as the sole previously reported FRA test error.
The full FRA edit introduces no new control errors on this subset. Record clean
reference agreement and introduced errors in confirmation, alongside absolute
accuracy; an unchanged aggregate accuracy can otherwise conceal exchanged errors.

The distinct-ID winner is now receiving the same diagnostic. Confirmation is
still untouched. Keep primary settings frozen by tuning, without selecting the
no-BOS or layer-omission variants from these exploratory diagnostics.

The poisoned baseline is already correct on3/8 exploratory-test target cases.
Thus the global winner repairs5 wrong target answers and preserves3 already
correct target answers; "8/8 correct" must not be described as8 repaired errors.
The poisoned baseline has55/56 control answers correct.

## 22:14 UTC — continued baseline sweep and operational resume

The expanded SAE run saved through batch 830/1085 and reached its expected
7200-second timeout. Its best valid tuning KL had improved to .03153, before
final control/repair selection and ordinary-forward replay. The first resume
attempt was blocked by the session's newly managed network sandbox and created
no GPU app. An approved launcher now checks the two-GPU, GPU-type and conservative
cost/time limits and preserves each stage's log. Resume app ap-SML6GCvZbazmkMIsG92OMV
started at 22:13 UTC on A100 with unchanged numerical sources and rankings.
The H100 prose-context search continues independently. Confirmation is untouched.

## 22:35 UTC — expanded baseline and second main search completed

The expanded primary SAE sweep finished 10,016 settings (12 invalid), including
imported main-search measurements. Its 50%-threshold winner is activation steering
at L22 feature14233, strength64, global scope, from the pooled absolute-difference
ranking. Tuning KL .031568; exploratory test KL .030269, suppression80.19%,8/8
targets and54/56 controls correct. Its ordinary-forward check changed mean tuning
KL by .000033 and maximum target probability by .00336. This is a stronger baseline
than the first sweep. The final comparison still awaits complete scope coverage,
local strength refinement and fresh confirmation. The original main sweep remains
in the merged pool, including its better 90%-threshold setting.

The prose-context main search finished 7,248 settings (8 invalid). At the 50%
tuning threshold, strongest SAE L22 feature12488, constant strength-64, gets
exploratory test KL .03860,88.2% suppression,8/8 targets,55/56 controls. Initial
FRA L22_gradient_H4_P1 strength64 gets KL .05934,98.5% suppression,8/8 targets,
54/56 controls. Thus this initial FRA search does not win at the 50% comparison.
At90%, the selected SAE gets KL .06061 but only53/56 controls; no qualifying win
is established. Gradient refinement now runs on H100 (ap-NJZRnAr4QHOAc9hxX0wKlF).

The primary full-scope SAE stage now runs on A100 (ap-0YG1X9X9cIJgXfNc9YyTl0),
completing no-BOS and whole-document grids over the candidate union. No primary
confirmation examples have been evaluated.

## 23:18 UTC — learned prose-task search completed

The learned prose-task search finished504 settings. Merged tuning selects
48 pairs from global_S512_P48 at strength8: tuningKL .004754; exploratory-test
KL .010386,99.35% suppression,8/8 targets,54/56 controls. Half of its48 pairs have
identical Q/K feature IDs. The distinct-ID winner is distinct_S512_P48,c16,
tuningKL .008702; it still needs its own full confirmation evaluation.

H100 now runs baseline_extra2_narrative_contracts, including the frozen single-pair
causal diagnostic and the stronger SAE absolute-difference/endpoints sweep. Main
A100 continues the complete308-feature scope grid. Confirmation is still untouched.
The cost ledger is about$31.45 and8.96 GPU app-wall hours, including queue time.

## 2026-09-17 23:26 UTC — single-pair causal diagnostic

The L17H8 Q12530×K0 single-pair cut has identical full and no-BOS predictions
on32 exploratory cases: KL .031422,93.52% suppression,4/4 targets,27/28 controls.
BOS-only steering has effectively zero repair; BOS raw pair magnitude is zero.
Final-position-only queries also have almost no effect. The source-row mask
repairs91.14% but lowers control accuracy to26/28. Its complement gets KL .02115
and92.89% repair with27/28 controls. These diagnostic variants are not retuned
primary settings. Distinct IDs and the absence of a BOS artifact still do not
identify two independent semantic concepts. Full diagnostic is saved locally.
