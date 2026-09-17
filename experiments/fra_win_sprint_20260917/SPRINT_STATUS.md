# Live sprint status — 2026-09-17 19:54 UTC

Deadline: 2026-09-18 03:16:27 UTC; last hour starts 02:16:27 UTC.
User requested ten hours; latest steering "go on" means continue this task.
Worktree /tmp/fra-win-sprint-20260917; branch dmitry/fra-win-10h-20260917.
Base 11020620. Commits cde63e4c and 381e0095 pushed and verified. Original shared
iclr-summary worktree has unrelated files and must remain untouched.

## Running (two A100-80GB GPUs, no agents)

- search_tenants_long: app ap-DyfEfTOpUTjWjWfs3O1ml0; exec session 25825.
  Started 18:47 UTC. At 19:51: 3,101 points / 1,094 batches total, best SAE tuning
  KL .05669. Six sites, activation and constant steering, then 216 FRA sets.
  Two-hour timeout expected around 20:48. If it times out, relaunch SAME stage
  and unchanged search.py/operators.py/data.py/common.py/PROTOCOL.md. Resume
  asserts exact numerical-source and calibrated-ranking identity. Those five
  files have intentionally remained unchanged since launch.
- learn_pairs_tenants_long: app ap-QnTxJb12nyuS7EgOiJUjp6; exec session 5700.
  Started 19:20. All coefficient training completed (two pools x 512 steps).
  At 19:51, tuning set 35/56. Expect finish ~20:05–20:15. Check log.
  Resume supports completed training/pair sets and partial sweep if needed.

Logs/results: experiments/fra_win_sprint_20260917/results/.
Modal volume: fra-win-sprint-20260917; cache fra-semantic-single-feature-cache.
Use modal_runner.py --stage STAGE. Max 2 concurrent GPUs, 18 GPU-hours, $60.
New calls have explicit 4-CPU / 64-GiB limits, timeout7200. update_ledger.py keeps
historical app wall time (including queue, NOT actual billed GPU time).

## Strong provisional candidate (tuning ONLY)

From a partial learned-pair result: distinct_S64_P48, strength16, 48 pairs with
q feature != k feature, across all six sites (10/17/21/22/28/32) and 23 heads.
Mean paired KL .00624, raw-target suppression .903, all target/control answers
correct on tuning. 56 distinct endpoint features. Learned weights ONLY identify
pairs; this primary candidate uses equal strength16. Weighted settings are
separate diagnostics and excluded from primary FRA selection.
Do not call a win yet: full SAE sweep incomplete, extra baseline queued, test
not finished and fresh confirmation untouched.

## Completed findings

- Initial screen: routing clean93.75%, tenants87.5%, tenants_long100%; poisoned
  controls all100%; all three passed initial behavior gate. Focus tenants_long.
- Source-row positional oracle top64 heads (ranked on first calibration joint):
  calibration KL .00827, suppression91.4%, all32 answers correct. Six SAE sites
  allheads source-row ablation only49% suppression. These are diagnostics, not FRA.
- Corrected score operator: actual residual RMS, native SAE encoding, BEFORE
  Gemma soft-cap, explicit GQA K expansion. Zero cut matches logits EXACTLY on
  all6sites; pre-RoPE Q/K relative RMS projection errors ~.0003. Extra all-head
  RoPE/pair audit is written but not yet run (part of variant_screen).
- First smoke failed GQA shape mismatch before accepting any intervention results;
  corrected smoke passed. SAE recon error excluding BOS ~.141 at L21IT/L28PT.
  Including BOS hugely inflates error; report both, avoid false transfer claim.
- refine_tenants_long finished: 11,339 candidate pairs ranked by all-position
  paired-KL gradient, 495 settings; live multi-layer evaluation. Winner L21 P144 c2:
  exploratory test KL .08357, suppression75.2%, target8/8, controls52/56 (fails95%).
  Zero cut exact; direct vs batched max target P difference .00343.
- Learned distinct384 calibration fitting reaches KL .010; global384 .0022.
  Training numbers alone are not performance evidence.

## Immediate next steps

1. Monitor both logs and finish learned result. Then run `variant_screen` on free
   GPU: it also runs audit_extra.py for all-head RoPE/GQA/pair identity checks.
2. Run `baseline_extra_tenants_long` after learned completion on that free GPU.
   It imports existing identical SAE evaluations, adds top10 absolute differences
   at answer / changed token / payload / pooled positions, plus all endpoints of
   best completed FRA candidates. Tests global and whole-retrieved-document scopes.
   It never uses the bad row alone as an intervention mask. Sweeps both signed
   activation scaling and constant directions. Partial sweeps resume. This code
   was syntax checked, not run yet; watch errors, do not silently weaken baseline.
3. Full main may timeout before finishing FRA. Relaunch unchanged source; source
   fingerprints include only five core files, so adding new modules is safe.
4. Once variant screen arrives, choose a genuinely strong clean/poison behavioral
   task for a second independent search if time permits. `cards` is the best new
   conceptual design: same Print-valued draft card must remain readable for
   literal documentation lookup; poisoning only changes draft -> active. Other
   active cards encode legitimate exceptions. Task instruction comes AFTER the
   document, so source states cannot know routing vs literal-lookup intent.
5. Before confirmation, finish extra SAE checks and consider dense coefficient
   polishing of top SAE candidates on tuning. Check same endpoints explicitly.
6. Build final confirmation runner using frozen.py foundation. IMPORTANT: it
   currently ignores `scope=document` and imports only original data.suite.
   Do not modify frozen.py while learn_pairs may need resume (hashed dependency).
   Make a copied `confirm.py` supporting scopes and variant data, passing row
   metadata to SAE hooks; scopes.py computes whole-document token mask.
   It must merge completed tuning points, freeze selections BEFORE creating
   confirmation inputs, verify original tuning outputs through independent
   evaluation path, then evaluate confirmation. Default sources should include
   search/refine/learn_pairs/baseline_extra (weighted_diagnostic excluded).
7. Fresh confirmation has 6 new lexical pairs x2 layouts x8 corners =96 cases,
   12 targets /84 controls. Primary KL is full-vocab P(clean doc)||P(poison+edit),
   float64; report raw probabilities, excess repair, queue and full-vocab accuracy,
   label mass and shared-conjunction controls. Strong win: >=20% lower KL at
   matched min suppression50% or90%, >=95% controls, comparable target accuracy.
   Report any threshold failing on confirmation. Block-bootstrap lexical indices.
8. If a win survives, run mechanism checks (source-edge vs complement; pair
   endpoint activation profiles; document/row reordering; shuffled pairs) and
   a dense same-feature baseline if needed. Verify claim about distinct *concepts*,
   since distinct IDs alone do not prove independent factors.
9. Reserve 02:16–03:16 for summary.md, figures, review, push all files <1,000,000
   bytes, and stop all our apps. Do not touch unrelated Modal apps (e.g. fra-b1).

## Files and caveats

- analyze.py loads gzip or .partNNN.json.gz fragments; uses tuning-only selection,
  bootstrap and plotting. Local output stem differs for variant stages: e.g.
  variant_search_cards.json.gz, while remote is search_cards.json.
- frozen.py includes independent tuning-case replay before confirmation; extend
  via new file for scoped SAE support. No confirmation run yet.
- variants.py: contracts, contracts_short, named_offices, narrative_contracts,
  cards. All specified before model screening; original data.py unchanged.
- variant_search/refine/learn_pairs.py are copies with dataset imports and hashes
  changed to preserve original numerical sources. Entrypoints registered.
- Read original sleeper code: modeldiff_baseline_pod.py:570 uses absolute mean
  SAE activation differences at trigger span. Existing positive answer-diff family
  is strengthened by baseline_extra to include that analogy. Record distinction.
- Existing figure source_oracle.png was visually checked; generation command is
  in tool history but should be saved as reproducible plot script. Label "Six SAE
  sites" could be clearer as "6 layers (96 heads)" in final plot.
- example.md has exact initial poisoned prompt. No summary.md yet (final deliverable).
- All files currently below1MB. Raw partial JSON files under results are ignored;
  export large results as gzip JSON-fragment shards. No hook bypass.

## Skills / agents

Sprint skill read/applied; final hour writing and wall-clock tracking required.
Modal skill already read. No research agents allowed by developer default.
Sprint skill explicitly permits red-team/blue-team write-up reviewers in final
hour, so bounded independent summary/figure review agents are allowed then.
Research-swarm skill was inspected but not applied; user did not request a swarm.

# Update — 20:20 UTC (supersedes the earlier running-job section)

- Main search still runs, app ap-DyfEfTOpUTjWjWfs3O1ml0, session25825.
  SAE portion finished (~5,270 points), beststrong tuning KL .037079, L28 f14418
  activation strength16, suppression68.7%. Bestdiff KL .0674, constant L17 f10525
  strength128. At90% suppression, both choose L32 f2585 c32, KL .48856. FRA anchor
  sweep now running (~670/1094 batches). Timeout expected20:48; resume unchanged.
- Learned search DONE20:04:55, app ap-QnTxJb12nyuS7EgOiJUjp6 automaticallystopped.
  504 points; local learned gzip914347bytes. Finaloverallwinner global_S256_P48 c8:
  tuningKL .005532,supp94.64%,all32correct; testKL .008472,supp91.77%,8/8targets,
  55/56controls,7/8sharedconjunction. 48pairs,5layers,23heads,48endpoints;
  **26pairs share Q/K feature ID**. Distinctwinner remains S64_P48 c16 KL.006240.
  Wrongshared testcase0layout0 is also lowPrintin clean (P.214); don't assume
  its answer error was caused byFRA. Mechanism/confirmation save clean predictions.
- Variant screen DONE, app ap-K86ft0KZtvQwIkR5r5GLVu stopped. All96head RoPE/pair
  audits passed, maxrelativeQKerror.000799. contracts(clean96.875%),named_offices
  (100%),narrative_contracts(100%) passed; contracts_short(93.75%) andcards(78.125%)
  failed. Cards poisonedcontrols75%, andjointmodeloftenignoresactivecard entirely.
  Narrativecontracts six-site sourceoracleKL.0281,supp85.3%, promising secondtask.
- **baseline_extra_tenants_long RUNNING**, app ap-TPZKfdZnEgXKfwkUI7adT2,
  session46463, started~20:09. 1085batches, ~30done20:18. Features perlayer global/doc:
  10:20/10,17:35/23,21:36/22,22:35/23,28:44/27,32:25/13.
  Timeout~22:11. Resume same stage if needed; once main isdone freshimports avoid
  duplicated originalSAEevals. **Do not edit baseline_extra.py,scopes.py,operators.py,
  data.py,variants.py,common.py until this run is complete/resume no longerneeded.**
- Pushed andverified **a0ed5bd0b845df0236ac98671b8715bb75f0fbfb**. Allstagedfiles<1MB.
  Laternewfilesuncommitted: baseline_abs.py,baseline_extra2.py,robustness.py,
  plot_confirmation.py andrunnerregistrationchanges. No summary.md yet.

## New code already prepared, not run

- confirm.py now supports document-scope SAE, includes search/refine/learn_pairs/
  baseline_extra/polish sources by default, freezes sources/selection before new
  prompts. analyze.py deduplicates numericalSAE configs preferring direct replays.
  It adds a **secondary fra_distinct** family requiring q!=k for everypair; final
  confirmation compares bothoverallFRA anddistinct toSAE. ExtraSAE includes both
  winners'endpoints. Neither result establishes independent semanticconcepts.
- polish.py: after original+extraSAE done, two finer-grid rounds,12bestdistinct
  (layer,feature,activation/constant,scope) perthreshold, subdivide adjacentgrid
  interval16ways. Preservescoarsepoints, independently replays selectedwinners.
- mechanism.py: afterconfirmation, source-only/complement/answer-only diagnostic
  masks, eachlayeronly/omitted,5shuffledKendpointsets, per-corner endpointactivations
  andsource-term magnitudes. Two testlexblocks(32cases). Stages mechanism_TASK and
  mechanism_distinct_TASK. Remoteoutputs mechanism_fra_TASK.json or
  mechanism_fra_distinct_TASK.json; localexportstems matchstage.
- robustness.py: afterconfirmation, fixedsettings onoriginal/reorderedrows13&37/
  archivedincidentdistractor +4screenedvariants; no retuning. stage robustness_TASK.
  Onlysupports tenants_long. Checks originaltestfirst2lexblocks andvariantcal.
- plot_oracle.py nowreproducesexistingfigure; labeloverlapsfixed, figureupdated.
  plot_confirmation.py creates50/90%comparisonfigures,KLcornerheatmaps andtables;
  inspect/iterate onactualresults. Originalsourceoracleplot QAafterlabelpatchstill
  pending finalimagecheck (priorversionhad xlabels crowded; fixedwithnewlines).
- cards_table.py andcards_table_screen.py define2table-formversions offailedcards,
  withactualconflictinglegitimatecontracts vsordinarydefaultcontracts. Notrun.
  Stage cards_table_screen; screenfirstifusing. These are separate tokeepvariants
  numericalhashstable. They are notyet wiredinto interventionsuite/scopemasks.
- baseline_abs.py is a copyofextraSAE thatcanrunbeforeFRAlearningcompletes: only
  absdiff4ranks,global/docscope; imports mainpartialSAEsameasextra. Stage
  baseline_abs_TASK; outputs baseline_abs_TASK.json. It mayparallelize a second
  task'sSAEs withFRAlearning aftermainsearch starts. Existingoriginalextraunchanged.
- baseline_extra2.py for SECONDtaskonly: sameasextra butimportscompletedabsbaseline
  toavoidrerevaluatingit, thenaddsFRAendpoints. Stage baseline_extra2_TASK, remote
  outputstillbaseline_extra_TASK.json; localexportbaseline_extra2_TASK(.part...).gz.
  Syntaxchecked, notrun. Mainextraoriginalnumericalsourceunchanged.

## Suggested schedule (adapt to runtime/results)

Finishmain+extraoriginal, polish, thenfreshconfirmation. Originalwinlikelybutnot
claimed beforeextras. WhileoneGPUfreed aftermain~21:15, cards_table_screen briefly,
then secondtask variant_search_narrative_contracts ispreconfigured iftime permits.
Canrunsecondtask baseline_abs alongsidevariant_refine/variant_learn_pairs tosave
clocktime; thenbaseline_extra2 importstheabsresults forendpointchecks. Prioritize
validatingcurrentpromisingcandidate overincompletesecondtask. Deadline03:16:27 UTC;
lastwritinghour02:16. Noagentsbeforefinalwritingreview(sprintskillallowsreview).

## 20:33 UTC priority addition: BOS diagnostic before resuming timed-out main

`mechanism.py` now has phase='preconfirmation' support. It selects the learned
FRA anddistinct winners by tuning only, loads onlylearned source (no confirmation
needed), and retains independentclean predictions. Added BOS-only and no-BOS
attention-column masks and per-pairBOS magnitude fields. These address themeasured
anomalousBOSSAEreconstructionerror; do not assumeBOSdominance beforetesting.
When main hits its natural20:48 timeout, usefreedGPU for:
  mechanism_pre_tenants_long
  mechanism_pre_distinct_tenants_long
then resumeunchanged search_tenants_long. Eachdiagnostic shouldtakefewminutes.
Remoteoutputs now:
  mechanism_preconfirmation_fra_tenants_long.json
  mechanism_preconfirmation_fra_distinct_tenants_long.json
(andpostconfirmation equivalentnames). Localexport namesmatchstage.
IfBOSiscrucial, investigate/learn cuts excludingBOS beforeclaimingsemanticfiltering.
Do not conflate changing anattentionsink withtwosemanticfeatures.
Referencewrapperdocstring corrected toexplainnativefixednormalization; numerical
behaviorunchanged. Corehashedsearch/extra filesstillunchanged.
