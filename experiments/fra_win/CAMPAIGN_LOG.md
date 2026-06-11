# FRA candidate campaign (3h autonomous, 2026-06-10)

Pipeline: brainstorm (8 family agents) -> rank -> evaluate (CCF+LBNR+Tier-2 on pod) -> red-team.
Metrics: CCF (edge-routed) AND LBNR-R (load-bearing & non-redundant); Tier-2 A (collateral advantage).
Published-behavior priors (PI): EM, Sleepers, Retrieval heads, Backtracking (Ward et al; Nanda).

## Status
- brainstorm+rank workflow: wbkl0m8ph (running)
- eval pod: see below
- eval pod: ecfyv90tqlhpkb (L4), booting
- reusable screener: jobs/g_screen.py (CCF+LBNR for any probe, gpt2 or gemma; CANDIDATES via $CANDS env)
- budget rule: if allowance hit, resume on reset, agents on opus 4.8
- brainstorm re-run on FABLE-5 agents (PI: brainstorm is highest-value): workflow w782m9bls (old wbkl0m8ph stopped); ranking on opus.
- RATE GUARD: reduced Fable to 2 agents (backtracking + wildcard) + 6 opus; relaunched as ws1u06t61 (w782m9bls stopped).

## RESUME STATE (canonical — any resume reads this first; keep updated)
- PHASE: 3/red-team + transfer-hardening. POD ecfyv90tqlhpkb.
- CONFIRMED NEW WIN: acronym letter-movers (gpt2, Garcia-Carrasco 2024): CCF 0.886, LBNR R=0.95, Tier-2 A=38x. Committed.
- SCREENED OUT this campaign: docstring (LBNR 0.08), fact-recall (weak P=0.039), + rank9-18 (direction/MLP/redundant).
- IN FLIGHT: (a) t_acronym_transfer (proper transfer: Officer at a DIFFERENT pos -- the patch test had positions coincide at 4, trivial); (b) red-team workflow wsp185jas (4 opus skeptics + synth on the acronym win).
- NEXT: when both done -> write CAMPAIGN_REPORT.md + figure incorporating red-team verdict -> commit -> terminate pod -> CronDelete (cron 51ed304d) to END loop.
- IDEMPOTENCY: check git log + HF outbox fra_win/out/<id>/ + this block; never double-submit.

## PHASE 2: EVALUATION (brainstorm+rank ws1u06t61 DONE, 21 cands -> shortlist)
Shortlist (ranked): 1.docstring-retrieval(gemma,WIN,78) 2.acronym(gpt2,UNCERTAIN,70,heads[8.11,9.9,10.10,11.4])
 3.scoped-retrieval(gemma,WIN,66,redundant) 4.fact-recall(gemma,UNCERTAIN,52,MLP-risk) 5.error-localization(gemma,40)
KILLED (rank9-18, direction-routed/MLP/redundant): refusal, successor, entity-track, EM, FV, NIAH-real, backtrack-onset, sleeper, MSJ.
NEXT: g1_acronym (gpt2) -> g2_gemma (docstring,retrieval,fact-recall) CCF+LBNR -> Tier-2 on passers.

### EVAL RESULTS (live)
- acronym (gpt2, heads 8.11/9.9/10.10): CCF=0.886 LBNR-R=+0.95 (P(O) 0.62->0.03) => PASS BOTH GATES -> Tier-2 worthy. (CCF settles the open Q: content-routed not positional.)
- NEXT: g2_gemma (docstring/retrieval/fact-recall) screen; then Tier-2 on acronym + gemma passers.
- g2_gemma (auto-find by raw attn): ALL CCF=0.000 -- BUG: auto-find picked positional/sink heads (attn 0.96, heads [7,6][1,7][18,1]) not content heads. retrieval-bridge is a KNOWN win, so this is a head-selection artifact. FIX: causal head-finding (cut head edge -> P(ans) drop). Re-run as g3_gemma. acronym unaffected (used fixed paper heads).

### EVAL RESULTS (causal head-finding, g3_gemma)
- retrieval-bridge SANITY: R=+0.63 (0.27->0.10) -> causal-find fix WORKS (recovers known win). [heads 25.5,22.5,17.4]
- docstring: R=+0.08 FAIL LBNR (P(files) 0.98->0.90 robustly predicted, redundant/distributed) -> SCREENED OUT.
- fact-recall: R=+0.86 (load-bearing edge) BUT base P(Paris)=0.039 too weak in gemma-2-2b base -> UNCERTAIN, deprioritize.
- CAVEAT: g_screen CCF=0.000 for ALL gemma edges incl known retrieval (sink-mask over-aggressive on gemma late heads); rely on LBNR for gemma. gpt2 CCF reliable.
- CONFIRMED NEW PASS: acronym (gpt2). NEXT: t_acronym Tier-2.

### ACRONYM TIER-2: CONFIRMED WIN (A=38x)
on-target P(O) 0.617->0.011 (matched removal achieved, NOT spurious); collateral on other acronyms FRA 0.008 vs head-ablate 0.322 vs content-suppress 0.341 => A=38x/40x; transfer 0.661->0.005 (content-addressed). SECOND new win the pipeline predicted in advance (after copy-suppression). Separability probe uninformative (legit base 0). NEXT: red-team + attention-patch fair baseline (FRA's edge over patch = transfer).

### ACRONYM TRANSFER (proper, Officer at pos 4 vs 11): FRA-unique 2x2 confirmed
FRA (content-addr) transfer 0.609->0.050 vs attention-patch@orig-pos 0.609->0.608 (FAILS, position-tied) vs patch@correct-pos 0.105 (needs the position). So vs the STRONGEST selective baseline (attention-patch), FRA's edge = content-addressing. head-ablate/content-suppress break all acronyms. FRA fills separable x transferable 2x2 uniquely (same structure as retrieval r2).
