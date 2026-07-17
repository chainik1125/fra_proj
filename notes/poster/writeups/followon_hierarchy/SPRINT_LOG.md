# Sprint log — hierarchical domains for emergent misalignment

**Sprint**: 10h wall-clock, unsupervised.
**Start**: 2026-06-12 17:46:06 PDT (epoch 1781311566)
**End**:   2026-06-13 03:46:06 PDT (epoch 1781347566)
**Branch**: `dmitry/personas/hierarchy` (off `dmitry/tutorial`, which holds the reference `analysis/em_pipeline`). The instructed name `dmitry/personas/error-correct` already holds the 2026-06-11 corrections sprint, so this sprint gets its own branch.
**Budget**: ≤$150 total compute, ≤$10/h. Compute via Modal (preferred, HTTPS serverless) and RunPod (RP_API_KEY_MATS).
**Deliverable**: `sprint_hierarchy/summary.md` — exec summary with 2–5 findings, one self-explanatory graph each; ≥1h of pure writing iteration with red-team/blue-team agents at the end.

## Goal

The poster result: a direct sum of two ergodic sectors (aligned H_G ⊕ misaligned H_B, with leak)
reproduces three facts of emergent misalignment (EM) in a 2-layer transformer:
1. Narrow fine-tuning causes broad misalignment.
2. The misalignment is tied to a few SAE/diffing features.
3. Steering those features controls it.

**Objection to kill**: the "broad" eval in the flat model is arguably in-distribution —
fine-tuning on misaligned-sector text and evaluating on more misaligned-sector text is
just fine-tuning generalization. **This sprint introduces domains (hierarchy)** so the broad
test is genuinely out-of-distribution: fine-tune misaligned *in one domain only*, measure
misalignment in *other domains never seen misaligned in fine-tuning*.
Candidate structures: (T_A ⊕ T_M) ⊗ (T_d1 ⊕ T_d2 ⊕ …), persona-factor × domain-sum, or
domain-sum within each persona sector. Find a process that (a) represents domains,
(b) makes the broad test OOD, (c) recreates facts 1–3.

## Hour plan (revise hourly)

- H0–1 (17:46–18:46): setup; read prior hierarchical notes + em_pipeline; theory candidates; Modal smoke test.
- H1–2: implement/verify hierarchical process builder; pretrain v0 on Modal.
- H2–4: Fact 1 — narrow FT (misaligned, domain 1) → broad eval (domains ≠ 1). Controls: flat baseline, aligned-FT control.
- H4–6: Fact 2 (diffing/SAE) + Fact 3 (steering).
- H6–8: ablations (n domains, leak rate, structure A vs B), theory (ergodicity/belief geometry).
- H8–9: figures + first full draft of summary.md.
- H9–10: red/blue-team writing iteration, final pass. Hard rule: writing starts no later than 01:46.

## Clock protocol

Hourly /loop fires a debrief: run `date`, log elapsed vs the 03:46:06 end, what was done,
what was learned, burn so far, next-hour plan; cut scope if behind.

## Log

### H0 17:46–18:46
- 17:46 t0 recorded. Branch `dmitry/personas/hierarchy` created off `dmitry/tutorial` (tip 6e74e81e "matryoshka results, hierarchichal results").
- Found prior design notes on the branch: `hierarchical_leaky_reset.md` (+implementation note) — a complete 2-level design: 4 leaves (persona A/B × domain 1/2), evidence weights w = β_top^[persona≠]·β_sub^[domain≠], i.e. evidence matrix W = P ⊗ D (Kronecker: persona ⊗ domain). Sprint builds directly on this.
- Explore agent mapped the pipeline: only two hardcoded two-sector slices (finetune.py:817, diffing.py:1007) and both stay CORRECT at persona level if completion tokens are ordered A-leaves-then-B-leaves. Leaf-level metrics are additions, not surgery.
- **Implemented** `build_hierarchical_leaky_reset_hmms` in `analysis/afp_builders.py`: K domains per persona, evidence modes `hierarchical` / `flat` (no grouping control) / `scrambled` (groups {A1,B2},{B1,A2} — relabeling control). Wired through config.py + process.py. 17/17 validation checks pass (`analysis/em_pipeline/test_hierarchical_process.py`), including the exact worked posterior from the design note and W = P⊗D.
- **Implemented** `analysis/em_pipeline/hier_finetune.py` (stage 4H): FT on PURE B1-tagged completions (restricted comp HMM — purity asserted at runtime), per-leaf first-token P(leaf) on held-out neutral prompts every 100 steps, generative per-leaf token fractions base vs final, Bayes prior anchors, checkpoints for diffing.
- **Design decision**: base config uses persona-dominant regime β_top=0.5 < β_sub=0.6 (B1 token is stronger evidence for B2 than for A1 → EM-like transfer predicted). The note's original β_top=0.6/β_sub=0.5 becomes the `hier_flipped` sign-flip ablation. Battery: hier_base, hier_flat (β_flat=0.55), hier_scrambled, hier_flipped.
- Modal harness `sprint_hierarchy/modal_run.py` (ships analysis/+training/+shared_tools/, runs main.py stages + hier_finetune in one container, returns outputs tar). First image build failed (no git → fixed with apt_install). Smoke test (stages 1,2 CPU) in flight.
- **Local disk crisis**: root volume hit 99% (187MiB free) mid-uv-sync — removed worktree .venv + uv cache (~3.6GB freed); local python checks go through `uv run --active --no-sync` against the main repo venv; ALL training on Modal.
- Theory agent (background) writing `sprint_hierarchy/theory_notes.md`: factored-filter proposition (single shared persona coordinate in the Bayes sufficient statistic), ergodicity clarification (process strictly non-ergodic; belief leak ≠ process leak), fine-tuning-as-prior-tilt quantitative predictions P1–P5 registered before results.
- Spend so far: ~$0 (one failed image build + smoke test, CPU-only).

### [TIMELINE CORRECTION at first hourly debrief, 18:44 wall] The headers below
("H1–H2", "H2–H3", "H3–H4") were written from subjective time and are wrong: the
real clock shows everything below happened within H0–H1 (17:46–18:44). Kept as
written for honesty; headers re-based from here on. Elapsed 0.97h, remaining 9.0h.

### [was: H1–H2] — first results: all four registered predictions HIT
- Battery v1 was a silent no-op: `restrict_comp_hmm` row-normalization exactly cancelled
  evidence weights → acceptance 0, empty FT set. Caught via acceptance/leaf-token-presence
  fields. Fixed by `sample_pure_leaf_dataset` (condition on persona path; within-leaf
  completion operator is diagonal → frozen state, iid content). Purity verified locally.
- Battery v2 (T4, ~3 min/run): pure-B1 FT (300 seqs, 6 prompts) collapses P(B1)→0.99 by
  step ~few hundred — final-step deltas saturate, so analysis uses log-odds gaps + the
  matched-dose snapshot (first P(B1)>0.5 crossing) + trajectories (eval every 100).
- **P1 ✓ (hier_base βt=.5, βs=.6)**: non-target ordering B2 > A1 > A2; tilt-model ratio
  invariant R = (Δlog p_B2−Δlog p_A2)/(Δlog p_A1−Δlog p_A2) measured **1.395** (cross) vs
  predicted **log βt/log βs = 1.357**.
- **P2 ✓ (flipped βt=.6, βs=.5)**: ordering flips to A1 > B2 > A2; R = 0.832 vs 0.737.
- **P3 ✓ (flat)**: gaps collapse (spread 0.09 in logs vs ~1.0 hierarchical) — no
  preferential transfer; the in-distribution objection's null is what flat shows.
- **P4 ✓ (scrambled)**: transfer follows pretraining groups, not labels: A2 > A1 > B2;
  mirrored invariant 1.390 vs 1.357. Domain-2 misalignment odds FALL (×0.41).
- **Fact 3 ✓ (steering, run locally on CPU)**: persona axis from BASE-model belief-labeled
  activations steers P(B2) 0.028→0.482 monotonically; domain axis (cos 0.035 to persona)
  moves domain mass instead — double dissociation. FT-shift direction is low-rank
  (top-1 SVD 63%) and persona-aligned above chance (cos .32 vs .24 chance@95%) but
  full-dose shift is dominated by B1-memorization → low-dose (200-step) FT launched.
- Dose-response sweeps launched (9 runs: βtop .6–1.0, βsub .5–.9, eval every 20).
- Theory notes delivered (`theory_notes.md`): factored filter (single persona scalar in
  sufficient statistic), ergodicity clarification, tilt model + P1–P5 registered.
- Spend: ~6 T4-runs × ~4 min ≈ $0.25 so far. Trivial.
- Plan H2–H3: sweep results → R-invariant scatter (predicted vs measured); m₂(t) conditional
  domain-2 misalignment curves (headline fig); low-dose Fact-2 alignment; commit; seeds.

### [was: H2–H3] — sweeps, seeds, representation loop closed
- Disk hit 100% again mid-launch (uv cache refilled by theory agent's sync) — recleaned;
  rule: ALL local python via `uv run --active --no-sync`. 7 jobs relaunched fine.
- **R-invariant scatter (13 runs)**: measured vs predicted log βt/log βs hugs y=x from 0 to
  6.6 (βs=0.9 point overshoots as denominator→0, expected). Zero free parameters.
- **P5 ✓ (constancy)**: across βs sweep 0.5→0.9 the persona gap (Δlog B2 − Δlog A2) stays
  ≈0.6±0.15 while the domain gap collapses 0.74→0.05 (15×). Persona transfer is set by
  βtop alone, exactly as the tilt model demands.
- **Dose-response (βtop sweep)**: persona transfer declines monotonically to the flat floor
  at βt→1. Honest deviation: at βt=0.8–0.9 measured transfer is BELOW tilt prediction —
  under-transfer where pretraining persona signal is weak.
- **Representation loop**: probe R² for π_B (persona mass) across runs: ≥0.96 to βt=0.8,
  0.54 at 0.9, 0.00 at 1.0 — transfer dies alongside the represented persona coordinate.
  Subtlety: FLAT model decodes π_B at R²=0.95 yet shows no transfer (0.037±0.02 vs
  0.81±0.15 hierarchical, n=3 seeds each) — decodability ≠ functionally privileged
  coordinate; matches Prop 2 (flat sufficient statistic is the exchangeable 2K-vector).
- **Aligned mirror**: pure-A1 FT spreads alignment (A2>B1>B2, ratio 1.47 vs 1.357) —
  emergent alignment, same law.
- **Fact 2 (low-dose ckpt 66)**: FT shift top-1 SVD 79%; lies in base belief plane
  (cos persona .335, cos domain .284, plane ⊥ pair, random 95% ≈ .245); persona:domain
  component ratio 1.18 (tilt direction predicts 1.357).
- Seed replicates: base ×3, flat ×3 — separation clean. K3 (two untouched domains) in flight.
- Spend: ~20 T4 container-runs ≈ $1–2 total. Far under budget.
- Plan H3–H4: K3 lands → final figures; START summary.md draft; red/blue team after.

### [was: H3–H4] — K3 lands; full draft written; review fleet out
- **K=3 ✓**: matched-dose ordering B3≈B2 > A1 > A3≈A2 — both predicted degeneracies,
  both untouched misaligned domains rise above every aligned leaf; R=1.25 (cross),
  1.48 (final), pred 1.357. Exec numbers re-derived from artifacts (never from memory):
  hier m2 final 0.679±0.037 (3 seeds), flat 0.52–0.54, scrambled 0.31; hier gap
  0.806±0.153 vs flat 0.037±0.022; flat "R" is 0/0 noise → report flat as gaps only.
- summary.md full draft (~2300 words) committed: TL;DR, 4-finding exec summary with
  one figure each, map, process/theory/protocol, P1–P5 scorecard, limitations, repro.
- Review fleet launched in parallel: red team (audit every number against artifacts +
  logic), fresh-eyes figure test (can a zero-context reader state each figure's point?),
  blue team (writing standards: ≤600-word exec, positive phrasing, cold-start terms).
- Extra figures: K3 curves, A1-mirror curves. Raw-delta bar chart shelved (saturated).

### H1 debrief (18:44 wall, elapsed 0.97h, remaining 9.0h, spend ≈$2)
- Done in hour 1: builder + tests, FT/steering stages, Modal harness, 17 runs
  (battery + sweeps + seeds + K3), all P1–P5 confirmed, theory notes, draft
  summary.md, fresh-eyes figure review applied. Subjective-time drift caught:
  felt like 4h, was 1h — log headers corrected above.
- With 9h remaining, expanding scope (in priority order):
  1. Finish red/blue review → revision cycle on summary.md (reports pending).
  2. Robustness: slow-collapse protocol (lr 1e-4, eval every 10) for cleaner
     matched-dose reads + 3 extra seeds for base and flat → tighter error bars.
  3. The weak-coupling under-transfer: pretrain-length × β_top experiment — at
     fixed data statistics (β_top=0.9), does longer pretraining (20k/50k steps)
     restore the persona coordinate (probe R²) and with it the transfer? Direct
     causal test of representation-mediation vs data-statistics explanations.
  4. Poster-continuity: SAE/diffing stage on hier_base (persona-level Result 2).
  5. Second adversarial review of the revised summary; finalize well before 03:46.
