# FRA PERSISTENCE (PATH 1) — can ONE FRA cell-cut remove a word-association at ALL its appearances? (started 2026-06-12)

## THE THESIS (PI path 1)
At the single-word level FRA's cell-cut == a token/column attention-mask ON A GIVEN INSTANCE (so per-instance selectivity
can't distinguish them — that's why EM/injection/binding all deflated). FRA's REAL value at this level is PERSISTENCE: the
cell-cut is a WEIGHT-space, ASSOCIATION-SPECIFIC edit that should remove the (A-content x B-content) correlation at EVERY
appearance of the word — including positions/contexts we DON'T know in advance — with no inference-time detection, while
leaving A's other uses intact. A token-mask can't (needs to detect A at each occurrence); feature-ablation kills A entirely.

## THE NON-TRIVIAL OBSTACLE (PI — the centerpiece measurement)
It is NOT automatic that one cell-cut catches all appearances: the SAE could choose DIFFERENT features for the "same" word-
association at different places/contexts (the induction anchor already hinted at this — the dominant cell was OFF-diagonal
qF423 x kF13032, distinct query/key features). If the SAE is context-inconsistent, one cell-cut misses appearances and
persistence requires the UNION of cells (bounded+identifiable = still a win; unbounded/context-dependent = a real fail).
=> THE EXPERIMENT MUST MEASURE the SAE-feature-consistency of a word-association ACROSS its appearances, and whether a
   bounded cell-set achieves persistent removal. This is the foundational test of path 1.

## THE TOY EXPERIMENT (controlled, REAL SAE so the consistency obstacle is exposed)
Model gpt2-small + gpt2-small-res-jb SAE (reuse the fra/ toolkit + j1_induction_explore). In-context word-association:
a FIXED novel pair A->B (e.g. " dax"->" blicket"), induction-copied (the model copies B after a repeat of A). Embed the
A...B...A pattern in VARIED carrier contexts so A appears at many positions/contexts. SPLIT: LOCATE set (find the cell) vs
HELD-OUT set (different carriers/positions — test persistence). GROUND-TRUTH metric = copy-prob P(B | A-repeat), judge-free.

## PRE-REGISTERED METRICS + WIN/FAIL (LOCKED before running — the anti-post-hoc discipline)
M1. PERSISTENCE/GENERALIZATION: locate the dominant FRA (qFi x kFj) cell on the LOCATE set; cut it (score-edit applied
    UNCONDITIONALLY wherever those features fire, NOT at a fixed position); measure copy-prob(B|A) DROP on the HELD-OUT set
    (novel carriers/positions). Persistence = high held-out drop from a cell located elsewhere.
M2. SAE-FEATURE-CONSISTENCY (the centerpiece): across ALL A-appearances, record the dominant (q-feat x k-feat) cell carrying
    the A->B edge. CONCENTRATION = does the top-1 cell carry the association at >=X% of appearances? Build the cell-distribution
    + the cumulative coverage of the top-k cells. Also: are the q-side / k-side features for A consistent across contexts?
M3. THE COMPARISON: FRA-feature-cut (persistent, fires wherever the feature does) vs TOKEN-MASK (cut attn to A only at LOCATE
    positions -> should FAIL on held-out, since you didn't mask there) vs UNION-of-top-k-cells (how many cells to catch all
    appearances?) vs A's BENIGN-USE preservation (does A still work in non-A->B contexts? feature/token ablation should hurt this).
WIN = the top-1 (or small k<=3) cell-cut removes >=70% of the association at HELD-OUT appearances (persistence) AND SAE-feature
   CONCENTRATION is high (top-3 cells cover >=80% of appearances) AND the token-mask FAILS on held-out (>=2x worse held-out
   removal than FRA) AND A's benign uses are preserved (FRA collateral on non-A->B << feature-ablation).
INFORMATIVE-NEGATIVE = the SAE chooses context-dependent features (top-1 covers <40% of appearances; held-out removal needs a
   large/unbounded union): persistence is undermined by SAE inconsistency = a real, important finding about the obstacle (and
   points to path 2's need for a cleaner decomposition).

## CARRY-OVER METHOD LESSONS (hard rules — the 3 deflations + faithfulness fail taught these)
- The WIN METRIC is PERSISTENCE + SAE-CONSISTENCY, NOT per-instance selectivity (per-instance is where FRA==token-mask and 3 wins deflated).
- Pre-register win/fail BEFORE running (locked above). GROUND-TRUTH (copy-prob) > judge. Symmetric red-team mandatory on any claim.
- Token-level cell sidesteps the SAE-reconstruction obstacle for the EDIT, but M2 still uses the SAE to TEST consistency (the point).
- RunPod RP_API_KEY_MATS, pods rs-persist-*; PARSE-GATE; NEW job id + unique name; ckpt() INSIDE loops + traceback-upload; HF fra_persist/{code,results}.
- Budget-conscious (gpt2-small is tiny/cheap). Commit every step; update RESUME STATE.

## THE TEAM (orchestrator = me): DESIGN/THEORY (pre-register + toy spec) -> EVALUATOR (build+run, M1/M2/M3) -> RED-TEAM (symmetric) -> PLANNING (synth + path-2 handoff).

## PIPELINE / STATE MACHINE (cron-driven, idempotent)
A: DESIGN/THEORY -> PERSIST_DESIGN.md (sharpen the toy + the M2 consistency metric + lock the pre-registration; confirm/refine the WIN/FAIL above).
B: EVALUATOR runs the toy (M1 persistence + M2 SAE-consistency + M3 comparison) -> WIN / INFORMATIVE-NEGATIVE.
C: RED-TEAM (symmetric: is the persistence real + the consistency honestly measured, or a toy artifact?). D: PLANNING synth -> path-2 (hierarchy) or consolidate.

## RESUME STATE (canonical)
- PHASE A IN FLIGHT (2026-06-12): DESIGN/THEORY agent a9c67ff46e2b77e03 -> PERSIST_DESIGN.md. CRON 70eb305c (21,51 = every 30min) drives. No pods yet.
- PHASE A DONE: PERSIST_DESIGN.md committed (concrete toy + M2 consistency metric + confirmed locked win/fail + 4 refinements; honest prior = informative-negative most likely).
- PHASE B IN FLIGHT: EVALUATOR afa99d66feb3da860 -> rs-persist-1 (gpt2-small+res-jb). M1 persistence (held-out rem) + M2 SAE-consistency (top1_coverage, n_cells_for_90, q-vs-k drift, {lc}xfeature) + M3 (token-mask-no-detector vs FRA, union, benign-use). VERDICT vs locked bars. Log -> PERSIST_LOG.md.
- NEXT: read VERDICT (WIN / INFORMATIVE-NEGATIVE / AMBIGUOUS) -> symmetric RED-TEAM -> PLANNING -> path-2 handoff or consolidate.
- Builds on: fra/ toolkit + fra_win/jobs/j1_induction_explore (induction + reconstruction), the STRATEGY note (fra_circuits/STRATEGY_persistence_vs_hierarchy.md), the 3-deflation+faithfulness boundary map.

## >>> REFRAME (PI, 2026-06-12): the bottleneck is FRA-DIAGNOSABILITY OF THE UNION, not single-cell concentration
A UNION of cells is EXPECTED to be the right structure (diag confirmed: oracle_single~0, oracle_multi~0.7-1.0). So the
win/fail is NOT "is it one cell". The BOTTLENECK = can FRA DIAGNOSE, from its decomposition alone, WHICH cells form the
union — i.e. does FRA replace BRUTE-FORCE causal ablation search for finding the relevant feature-conjunctions? The only
alternative to FRA for finding the union is per-cell causal ablation (expensive); FRA's value = read the union off the
(free) decomposition. NEW HEADLINE METRIC (diagnosability): rank cells by FRA score s (free) vs by causal cut-effect c
(oracle); RECOVERY(k) = removal(FRA-top-k)/removal(causal-top-k) + Spearman(s,c) + does the FRA-diagnosed union TRANSFER
to held-out appearances. WIN = FRA-diagnosed union recovers >=70% of oracle removal at small k + high Spearman + transfers.
INFORMATIVE-NEGATIVE = FRA ranking != causal (low Spearman, recovery<<1) -> can't diagnose the union from FRA, still need
brute-force -> FRA adds no diagnostic value here. (The induction faithfulness FAIL [FRA top-k carried ~20% of the edge]
predicts the negative.) Relayed to evaluator afa99d66feb3da860. This is THE value question: is FRA a diagnostic that
identifies the relevant feature-conjunction-union without causal search? The hierarchical-SAE synthetic (PI, pending) becomes:
does a hierarchical SAE make the union MORE DIAGNOSABLE (FRA-rank aligns with causal) than a flat SAE, with ground-truth?

## >>> RECALIBRATION (PI, 2026-06-12): evaluate FRA on the THREE SLEEPER AXES separately — detection / localization / control
PI: the diagnosability-is-THE-value framing was TOO STRONG. Restore the sleeper-work decomposition. FRA can add value on ANY of
three axes, which DECOUPLE (need not stack):
 - DETECTION: do FRA cell activations / CELL-CORRELATIONS signal a concept-association is present / which concepts are RELATED?
   (vs an SAE-feature probe.) Barely tested — possibly FRA's real niche (PI: "correlations between FRA cells are diagnostic").
 - LOCALIZATION: does FRA identify WHICH cells carry it (the diagnosability metric: FRA-rank vs causal-rank, recovery(k))? (vs
   brute-force causal ablation.) The induction faithfulness FAIL is a LOCALIZATION weakness — but that is ONE axis, NOT a global FRA verdict.
 - CONTROL: does cutting FRA cells surgically remove/edit the association? (vs token-mask / linear / SAE-feature-edit.) The binding+
   injection deflations were CONTROL-axis losses at the single-word level. Persistence M1 tests a DIFFERENT control claim: weight-persistence (cut once -> removed at ALL appearances unconditionally).
THE DECOUPLINGS (PI, live hypotheses): (1) an SAE-probe may DETECT/LOCALIZE relatedness better while FRA CONTROLS the connection
more surgically; (2) FRA-cell-correlations may be DIAGNOSTIC while CONTROL is swamped by the many other features. => measure each
axis vs its natural alternative; FRA's value = wherever it wins, accepting it may win on one and lose on another.
PERSISTENCE EXPERIMENT recast: M1 = CONTROL axis (weight-persistence removal across appearances); the diagnosability metric =
LOCALIZATION axis (report SEPARATELY, do NOT collapse). DETECTION axis = a new measurement (FRA cell-correlation -> concept-
relatedness vs an SAE-probe), naturally tested in the HIERARCHICAL-SAE SYNTHETIC with ground-truth concept-graph.

## >>> THE CLEAN SIMPLE EXPERIMENT (PI, 2026-06-12): position-invariance + association-specificity, vs the EMBEDDING-cut baseline
ESTABLISHED + EXPECTED (not a failure): at a KNOWN position, FRA cut == simple attention-map cut, no edge (the binding deflation).
THE INTERESTING TEST = POSITION-INVARIANCE at RANDOM/unknown positions, where the attention-map cut is DISQUALIFIED (needs the
position). The FAIR baseline is then CUTTING THE EMBEDDING (also position-invariant). FRA is the unique cell that is BOTH
position-invariant AND association-specific:
  2x2: attention-map cut = pos-specific+assoc-specific (FAILS at random pos); embedding cut = pos-invariant+assoc-BLIND (works,
       kills all of A); FRA cut = pos-invariant+assoc-SPECIFIC (the claim: removes A->B wherever A fires, preserves A's other uses).
THE EXPERIMENT (gpt2-small induction = cut the diagnosed (A-feature x A-feature) QK cell-union): KEY MANIPULATION = A at RANDOM
positions (assume we DON'T know A's position at cut time). THREE interventions: (a) FRA cut (pos-invariant score/weight edit of
the (AxA) union); (b) ATTENTION-MAP cut at a fixed/guessed wrong position -> FAILS at random pos (the "can't work here" control);
(c) EMBEDDING cut (project out A's feature everywhere) = the FAIR pos-invariant baseline, assoc-BLIND. METRICS: position-invariant
A->B removal (FRA ~= embedding >> attention-map-at-random) + ASSOCIATION-SPECIFICITY = collateral on A's OTHER uses (FRA << embedding).
WIN = FRA removes A->B at random pos ~= embedding AND >> attention-map, AND FRA's collateral on A's other uses >=2x lower than the
embedding-cut's. NULL = no specificity edge over the embedding cut, or FRA fails at random pos. This is the CONTROL axis with the
RIGHT baselines (embedding cut, not token-mask). Still need to diagnose the union (localization) to do the FRA cut. Relayed to evaluator.
