# 10h unsupervised sprint: a full theory of FRA on the simplest toy model

I want to do an experiment that gives you a chance to be creative and rigorous.

You are running a **10h unsupervised research sprint**. This is _wall-clock_ time,
not estimated subjective time. A supervisor process resumes you continuously and
injects a `[SUPERVISOR TIMER]` header with elapsed/remaining time into every
resume — **trust that header over any internal sense of time**, and structure your
work so that the final 75 minutes are spent ONLY on writing.

You are a Claude Code instance running as root, with the repo at
`/workspace/fra_proj` (public `main` + a sprint overlay providing
`experiments/constrained_belief_updating/` and `experiments/fra_hmm_toy/`).
You are fully on your own: if you hit a block, find an alternative.
IMPORTANT: the papers under `experiments/constrained_belief_updating/papers/`
include private/unpublished material — never push them to any git remote or
public location; results travel only via the private HF dataset (automatic).

## Context

The FRA (feature-resolved attention) campaign established empirically WHERE FRA
works (`experiments/fra_win/`: association-specific attention control) and where
it fails (`experiments/fra_pii/`, and decisively `experiments/fra_hmm_toy/`: the
aggregation regime — read `experiments/fra_hmm_toy/PEDAGOGICAL_NOTE.md` first,
it is the empirical anchor for everything in this sprint). What is missing is a
THEORY of why.

Three theory notes give (nearly) all the ingredients. Full local copies + a
distillation live in `experiments/constrained_belief_updating/papers/`:

- `READING_NOTES.md` — START HERE. Distillation of all three + the FRA bridge.
- P1 `P1_attention_head_unit/` — effective subspace attention, factored Mess3,
  QK=temporal / OV=factor-routing division of labor, conic head counts.
- P2 `P2_constrained_belief_updates/src/Updated_ICML.tex` — the constrained
  belief update r₁ = π + Σ_s (π T^{|z_s} T^{d−s} − π), spectral attention
  predictions, negative-eigenvalue two-head mechanism.
- P3 `javan_theory/main.tex` — one-layer MSE theory: closed-form W*, V*,
  optimal attention from process spectrum (Toeplitz, sums of geometrics).

## Goal

**Produce a full, quantitative theory of how FRA works, worked out exactly on
the simplest toy models, and verified numerically.** "Incorporate FRA into the
constrained-belief-updating theory" is the intermediate goal. Keep one eye on
BOTH questions throughout:

- **Attribution**: what do FRA-QK and FRA-OV attributions *compute* in this
  setting — in closed form, in an idealized feature basis and in a trained-SAE basis?
- **Intervention**: when can FRA *control* behavior — derive when FRA edits work,
  what they can and cannot do, why (or under what SAE properties) the failures
  are fundamental vs fixable.

### Pre-registered theory targets (from READING_NOTES.md §"FRA bridge")

1. FRA-QK inertness as a theorem: for stationary-HMM data the optimal pattern is
   lag-only ⇒ feature×feature score edits have no concept-causal handle. Make
   precise; state what data properties (spectrum/support) create content-gated
   patterns where FRA-QK bites (the induction/fra_win regime).
2. FRA-OV = effective subspace attention in an SAE basis: closed form ζ^{d−s};
   when are per-head FRA rankings gauge-dependent garbage vs meaningful.
3. The gain-tuned null c* as a path-share ratio (skip/diagonal/non-local
   redundancy, P1 Eq. 25–28). Predict fra_hmm_toy's c*≈4 ex ante.
4. The mixture concept ω = the λ→1 spectral limit: optimal pattern flattens to
   counting, concept moves into aggregated content. Derive the use/presence
   split and the SAE requirement (what property must latents have for content
   cuts to work).
5. Where exactness fails, say so honestly and show the gap.

**If `experiments/constrained_belief_updating/sprint/RESEARCH_LOG.md` already
exists when you start**: you are RESUMING after an infrastructure replacement.
Read the log FIRST, trust its state-recovery section, keep all existing work,
and continue from where it leaves off — do not restart from scratch and do not
re-read the full grounding you have already digested (the log tells you what
you know). Note the timer header: elapsed time may already be nonzero-equivalent
(a shorter total).

## Plan (suggested allocation; follow results and your taste)

- **h0–1**: read READING_NOTES.md, PEDAGOGICAL_NOTE.md, skim the three notes.
  Set up `sprint/` workspace, write the plan into RESEARCH_LOG.md.
- **h1–5**: Phase A — stationary single-Mess3, one layer. Derive FRA-QK/FRA-OV
  closed forms at the theoretical weights; verify on a freshly trained tiny
  transformer + TopK SAE (reuse `fra_hmm_toy/` training/SAE code patterns; CPU
  is plenty, a small GPU may be present). Exactness checks against analytic
  anchors, like PEDAGOGICAL_NOTE does.
- **h5–8**: Phase B — the mixture/ω case (fra_hmm_toy's setting): derive the
  aggregation-regime results (targets 3+4), verify against the EXISTING
  fra_hmm_toy numbers (out/main2 etc.) before running anything new.
- **h8–9**: Phase C (stretch) — QK-parameterization of P3 / softmax gap, or
  whatever the results demand instead.
- **h9–10**: writing ONLY.

## Agent team (mandatory working style)

You are the orchestrator of a team, not a solo agent. Use the Agent tool to run
subagents **inside this one session** (never spawn detached `claude` processes):

- theory subagents to derive/check the closed forms independently (have two
  derive the same result separately and compare);
- a numerics subagent to build/verify the toy pipeline;
- a literature/notation subagent when you need something dug out of the papers;
- in the final writing window: a red-team/blue-team pair iterating on
  summary.md until a fresh reader would understand and believe it.

Long-running computations (>10 min) must be launched detached
(`nohup ... > log 2>&1 &`) and polled in later turns — your turn can be
interrupted at any moment, and detached processes survive; foreground ones do not.

## Deliverables (all inside `experiments/constrained_belief_updating/sprint/`)

1. `summary.md` — THE deliverable. Executive summary = 2–5 key findings, each
   stated simply with one supporting graph with self-explanatory axes (test: a
   fresh agent shown only the graph should get why it matters). Then enough
   detail to follow everything without reading code. Include a map of what you
   did and where code/results live.
2. `notes/fra_theory_note.tex` — a formal LaTeX note in the exact style of
   `papers/javan_theory/main.tex` (same macros where sensible): definitions,
   propositions, derivations, the FRA objects incorporated into the
   constrained-belief framework.
3. `notes/fra_pedagogical_note.tex` — a rigorous but HIGHLY pedagogical LaTeX
   note: example-heavy, builds intuition first and ascends to the exact
   formalism; worked numerical examples with real numbers from your runs.
4. `RESEARCH_LOG.md` — continuously updated: what you tried, why, dead ends
   included. This is also your own state-recovery mechanism across resumes.
5. `figures/` + code. Figures as PNG (the snapshotter ships PNG + tex + md).

Everything in `sprint/` plus your two work directories is auto-tarred and
shipped to the private HF dataset every 15 min. Assume the machine can die at
any time — keep all state in files under the work directories, never >15 min
of work only in memory or in your context.

## Writing guidance (read before writing, follow strictly)

- `sprint_infra/writing/writing_instructions.md` — canonical guide.
- `sprint_infra/writing/sprint_advice.md` — what makes a good project; the
  skepticism/honesty norms (sanity-check your own results, report negative
  results as negative, plausible claims over ambitious ones).
- `sprint_infra/writing/good_writing_examples_claude.md` and
  `bad_writing_examples_claude.md` (+ short `bad_writing_examples.md`).
- Write positively ("A chain is a ..."), never negatively ("A chain is not...").

## Budget

API spend is tracked by the supervisor and shown in the timer header
($120 hard cap — wrap up if you approach it). Pod compute is negligible; do
not launch other pods.
