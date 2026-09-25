# Overnight iteration log (fable-overnight)

Start: 2026-06-09 21:48. Deadline: ~07:48. Hourly wakeups.

## Hour 1 (21:48–22:45) — DONE
- Read all source docs; claims map + outline + suggestions.md (commit 30c943d).
- Full rewrite: abstract, contributions, sleepers (Jamie corrected base + defender framing +
  detect-vs-control + knowledge tiers), EM (fra_14b_diff campaign), discussion, appendices
  (commit c2d482d).
- Polish: dropped stale FRA_detailed overview, regenerated k1_fourway figure, impact statement,
  bib additions, boilerplate cleanup (5a3c1b7); figure tracking fix + F603 rate trajectory (9bd3629).
- Numbers audit pass 1 vs writeups: complete; one fix (EM bucket sizes).
- Compiles clean: 28pp, 0 undefined refs, 3 pre-existing overfull boxes (cartoon/eqs).

## Hour 2 (22:45–23:30ish) — DONE
- Adversarial claim-by-claim audit of abstract/§4/§5/appendices vs sources; fixes:
  matched-vs-unmatched JSD framing made precise; kill-switch claim scoped to carrier
  layer (+L0 exception pointer); J_clean tied to JSD_clean notation; EM caveats note
  single-seed financial finegrid cells; appendix C records that the conventional column
  needs a poison-exposed dictionary (fails on clean dict 3/5 seeds).
- steering_effect_by_scheme regenerated as vector PDF from verified table values
  (use fra_proj/.venv python for matplotlib — system python3 has numpy2/matplotlib clash).
- Stale top-of-file TODOs pruned; related-work suggestions added to suggestions.md.
- Typo/duplicate-word sweep: clean. UK spelling consistent. Compile clean (ec7021f).
- DEFERRED: regenerating F603_steering_curves/rate_trajectory as vector PDFs via
  fra_proj/experiments/fra_14b_diff/f603_analysis.py (HF_TOKEN is set; heavy downloads;
  current 150dpi PNGs legible). Do only if hour budget allows.

## Backlog for next iterations (in priority order)
1. Fresh-eyes read of the full PDF; prose polish (esp. long sentences in §4 closing
   paragraph, §4.3 final paragraph; abstract length).
2. Adversarial review pass: every claim → check support; over/under-claiming;
   internal contradictions; tense/naming consistency (Conv vs Wang; J_clean vs JSD_clean
   notation across sleeper sections — UNIFY? currently fig3 uses JSD_clean, §4.2-4.3 J_clean).
3. Consider regenerating EM figures as vector PDFs with larger fonts
   (fra_proj/experiments/fra_14b_diff/{scheme_barchart,f603_analysis}.py — f603_analysis is
   read-only on HF, needs HF_TOKEN; scheme_barchart may need local tables). Optional.
4. Second-source number verification for a sample of EM numbers (HF combined cells) — optional.
5. Spell/typo pass (real typos only; paper uses British spelling consistently — keep).
6. arXiv hygiene: confirm only referenced figures shipped; check arxiv.tex TODO comments at top
   (lines ~93-98) — prune stale ones (items 1,3 done).
7. Possible: small pgfplots replacement for fig6_cumulative.png (PNG ok though).
8. Final: full re-read, compile, push-ready summary for Dmitry.

## Decisions made (do not relitigate without new info)
- Honest framing per user: FRA ties optimized baselines; wins = detection/localisation/
  diagnosis/calibration/example-free tiers.
- All J values in BITS everywhere in the paper (nats sources converted ×1.443).
- Old mislabeled EM results + figures fully removed (incl. appendix seed grids, random-feature
  baseline, CE analysis, 60% overlap table, head-ablation table H38/H0/H36/H7).
- Detailed-overview appendix removed (figure encodes superseded claims).
- Mechanistic-interpretation section deleted; content folded into §5.4 + Discussion.
