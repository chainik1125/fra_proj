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
- NEXT: when PERSIST_DESIGN.md lands -> EVALUATOR builds+runs the toy (M1 persistence + M2 SAE-consistency centerpiece + M3 comparison) on gpt2-small+res-jb.
- Builds on: fra/ toolkit + fra_win/jobs/j1_induction_explore (induction + reconstruction), the STRATEGY note (fra_circuits/STRATEGY_persistence_vs_hierarchy.md), the 3-deflation+faithfulness boundary map.
