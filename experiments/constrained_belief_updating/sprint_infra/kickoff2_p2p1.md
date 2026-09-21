# Sprint 2: FRA inside the P2 → P1 program

I want to do an experiment that gives you a chance to be creative and rigorous.

You are running a supervised autonomous research sprint (wall clock and budget
are injected into every resume as a `[SUPERVISOR TIMER]` header — trust it, and
reserve the final window for writing only). You are a Claude Code instance with
the repo at `/workspace/fra_proj`; work areas are
`experiments/constrained_belief_updating/` and `experiments/fra_hmm_toy/`.
The papers under `papers/` include private/unpublished material — never push
anything to git remotes; results ship automatically via the private HF dataset.

## Mission

**Redo the P1/P2/P3 program with FRA as the object of study.** Feature-resolved
attention (FRA) decomposes attention scores (QK) and transported content (OV)
through a sparse-feature basis and edits them term by term. The prior sprint
(see below) built gauge theorems and a path calculus centered on the
`fra_hmm_toy` mixture. THIS sprint stays inside the papers' own settings:
**start in P2's world and build up to P1's.** The mixture toy is out of scope
except as a one-paragraph outlook.

## Grounding (read first, in this order)

1. `papers/READING_NOTES.md` — distillation of P1/P2/P3 + the FRA bridge.
2. **Prior sprint results — REUSE, do not re-derive:**
   - `sprint_prior/run1_notes/` — worker A's theory + pedagogical notes: the
     FRA-QK gauge characterization at token-independent patterns, the cut
     calculus, the P3↔P2 bridge lemma (M_λ columns ↔ g-vectors), single-Mess3
     closed forms. Its exact SAE-basis FRA toolkit + trained single-Mess3
     platform is at `sprint_prior/run1_code/phase_a/`.
   - `sprint_prior/run1_twin_notes/` — worker B's notes: softmax realizability
     of the constrained update + the normalization warp (explains P2's
     first-two-embeddings anomaly), FRA-OV ≡ effective subspace attention,
     flat-profile/λ→1 results.
   Verify a claim before you lean on it — these are unreviewed sprint outputs.
3. The papers themselves as needed: P2 `papers/P2_constrained_belief_updates/src/Updated_ICML.tex`,
   P1 `papers/P1_attention_head_unit/P1_fulltext.txt` (+ PDF), P3 `papers/javan_theory/main.tex`.

## Phase 1 — FRA in P2's world (single Mess3, one-layer softmax)

Reproduce P2's setting faithfully (their App. B: d_model 64, d_ff 256, 1 layer,
1–2 heads, no BOS, CE loss; Mess3 across their (x, α) grid — at least one
positive-ζ and one negative-ζ configuration). Then:

1. **The FRA dictionary.** Train TopK SAEs at the pre-attention and
   post-attention hookpoints. Map SAE latents onto P2's ideal objects — the
   prior π, the token displacement directions g(z) = πT^{|z} − π, positional
   structure. Report how cleanly the learned dictionary matches the theory
   basis (this is itself a result: what must an SAE learn for FRA on this
   model to be interpretable at all?).
2. **FRA-QK closed forms.** At P2's theoretical weights (pattern ζ^{d−s}/N_d,
   OV ∝ g(z), embeddings ∝ OV), derive every feature×feature score term.
   Which terms are gauge (apply the prior sprint's characterization — verify
   it transfers), which are loss-pinned, what any cut does. Verify on the
   trained model to float precision where the theory is exact.
3. **FRA-OV closed forms and control.** Per-feature transported content =
   which g(z) flows where with weight ζ^{d−s} (mod the normalization warp).
   Demonstrate *positive* control: token-selective belief-update edits
   (sever/scale one g(z) channel), lag-window edits, and predict each effect
   on CE and on the intermediate fractal geometry ex ante. This is the paper's
   "attribution AND intervention" demand in its cleanest habitat.
4. **The two-head hinge (negative ζ).** P2 shows ζ<0 needs two heads with
   anti-parallel OVs. Show per-head FRA attributions are gauge-dominated
   (arbitrary head splits) while head-summed FRA recovers the theory objects.
   Quantify with the prior sprint's dial methodology if it transfers. This is
   the bridge result into P1.

## Phase 2 — FRA in P1's world (factored two-Mess3 products)

Reproduce P1's setting (their App. L: 1 layer, pre-norm, d_model 120, seq 11
with BOS, two-factor Mess3 configs from their Table 4 — at minimum distinct
positive (+0.7,+0.4) and mixed sign (+0.5,−0.5); H ∈ {1,2,3} as required):

5. **FRA through factor subspaces.** Identify factor subspaces (P1's vary-one
   routine), train SAEs, and show empirically that FRA-OV aggregated to a
   factor subspace equals P1's effective subspace attention ζ_n^{d−s} — the
   prior sprint asserted this identification; make it an experimental fact in
   P1's actual setting.
6. **Head gauge vs subspace invariance.** Across P1's specialization /
   collaboration / polysemanticity regimes: when do per-head FRA rankings mean
   anything? Show the subspace-aggregated FRA is the invariant, and connect
   the head-freedom to the conic decomposition.
7. **Factor-selective intervention.** The factored version of use/collateral:
   remove factor n's update via FRA-OV (predict: works, with the path-share /
   (cρ)² calculus and the k=0 skip-diagonal redundancy of P1 Eq. 25–28) vs
   via FRA-QK (predict: inert — lag-only pattern), with factor m as the
   built-in collateral control. Predict every number before measuring it.
8. **(Stretch) P3 extension.** Write the optimal-profile story for two distinct
   eigenvalues in P3's own machinery (two geometric rates from the palindromic
   polynomial) and place the FRA objects at that optimum.

## Deliverables (in `experiments/constrained_belief_updating/sprint/`)

1. `summary.md` — executive summary: 2–5 findings, each with one
   self-explanatory graph; then enough detail to follow without reading code.
2. `notes/fra_cbu_note.tex` — formal note **written as an extension of
   Javan's note** (P3's notation and macros; import the prior sprint's bridge
   lemma rather than re-proving): FRA in the P2 setting, then the factored/P1
   setting.
3. `notes/fra_cbu_pedagogical.tex` — rigorous but example-heavy companion:
   builds intuition with worked numbers from your own runs, ascends to the
   exact formalism.
4. `RESEARCH_LOG.md` — continuous, with a state-recovery section; dead ends
   included.
5. `figures/` + code. PNGs (the snapshotter ships md/tex/png every 15 min).

If RESEARCH_LOG.md already exists at start, you are resuming after an infra
replacement: read it first, keep everything, continue.

## Working style

- You are the orchestrator of an in-session agent team (Agent tool): parallel
  theory derivations with independent cross-checks, a numerics subagent, and a
  red-team/blue-team pass on summary.md in the final window. Never spawn
  detached claude processes.
- Any computation >10 min: launch detached (`nohup ... &`), poll next turn.
- Phase 1 before Phase 2; if the budget window is closing (watch the timer
  header), a complete Phase 1 with a written note beats two half-phases.
- Honesty norms of `sprint_infra/writing/sprint_advice.md`: verify your own
  claims, negative results reported as negative, plausible over ambitious.
- Writing guidance: `sprint_infra/writing/writing_instructions.md` and the
  good/bad examples in the same directory. Write positively, not negatively.

## Budget

The timer header shows API spend against a hard cap; the supervisor forces a
write-only window near the cap and wraps the sprint at it. Prior-sprint
calibration: a full-intensity orchestrator turn ≈ $55–65/hour — plan the
phase budget accordingly (do not fan out more than ~3 subagents at once).
