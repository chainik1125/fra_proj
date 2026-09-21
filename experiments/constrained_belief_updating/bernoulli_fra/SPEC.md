# Optimal FRA from the ground up: Bernoulli statics → reset-process dynamics

*Working dir for the in-session agent team, 2026-07-16. Everything here is
theory-first with light numerical verification; CPU-scale only.*

## Motivation

Two sprints established: FRA-QK attributions at CBU optima are gauge; FRA-OV
obeys a path calculus; the SAE dictionary is the binding constraint (learned
codes are contrast codes with missing channels). The missing rung is the
BOTTOM of the ladder: a setting with exactly known ground-truth features and
the simplest possible temporal structure, where "optimal FRA" can be written
down completely — every invariant, every gauge orbit, every edit — before any
Mess3-style latent dynamics enter.

## Setting A — Synth SAE bench statics (papers/synth_sae/main.tex §3)

a_t = Σ_i c_i(t) d_i + b, with z_i ~ Bernoulli(p_i) (Gaussian copula Σ for
correlations), c_i = z_i·ReLU(μ_i + σ_i ε_i), dictionary D known (unit-norm,
ρ_mm superposition control), i.i.d. ACROSS TIME t.

Questions (Setting A):
A1. One-layer attention (P3's M1: free profile, MSE; and softmax M2) trained
    to predict a_{t+1} from a_{1:t}: prove the optimal value/attention
    circuit is null (R = 0, V* = 0 in P3's notation) since B^(τ≥1) has no
    nonstationary content. State precisely: THE ENTIRE ATTENTION CIRCUIT IS
    GAUGE — every FRA-QK and FRA-OV attribution on a trained model is
    seed noise / convention. What DOES the direct path W* learn (per-feature
    shrinkage/denoising: the Bayes predictor of c from a)?
A2. Exact FRA algebra in the GT basis: with features known, write the
    feature×feature score decomposition and OV transport for an arbitrary
    (not necessarily optimal) head; identify the gauge group explicitly
    (softmax row constants; b/bias re-splits; head splits; and the
    superposition-induced non-orthogonality term ∝ d_i·d_j).
A3. The dilution proposition, exactly: under dictionary refinement (one GT
    feature split into m latents, or overcomplete L > N), invariant QK
    couplings spread ∝ 1/m² per pair, OV ∝ 1/m. Under superposition ρ_mm > 0,
    quantify the cross-feature contamination of FRA terms.
A4. Dictionary link: translate the bench's recovery metrics (MCC, uniqueness,
    precision/recall; hedging under correlation Σ; absorption under
    hierarchy) into the mixing matrix χ formalism from sprint 2's Finding 4.
    What recovery level is SUFFICIENT for faithful FRA edits — is MCC the
    right sufficient statistic, or do we need centering/completeness too?

## Setting B — the reset process (the temporal extension; ONE latent per GT feature)

Each feature's indicator becomes persistent: z_i(t) evolves as an independent
2-state Markov chain — with probability (1−λ_i) redraw z_i ~ Bernoulli(p_i),
else persist (a "reset process"; per-feature nonstationary eigenvalue is
exactly λ_i). Magnitudes redrawn per firing (choose the simplest convention
and state it). Everything factorizes across features by construction.

Questions (Setting B):
B1. Belief theory: the Bayes filter per feature (posterior on z_i(t) given
    history); show the constrained-belief-update form (P2's Eq. 5 analogue):
    independent per-source corrections decaying as λ_i^{d−s}, routed into
    span(d_i). This is P1's factored world with trivial within-factor
    structure — the per-factor "g-vector" is ±d_i-aligned.
B2. Optimal attention (P3 lift): optimal profile = mixture of per-feature
    geometric kernels with rates from the λ_i spectrum; work the multi-λ
    palindromic-polynomial machinery for 2+ distinct λ's. Conic head count:
    when do H heads suffice for the ray set {(λ_1^k,…,λ_N^k)}?
B3. OPTIMAL FRA, stated as the deliverable theorem: in the GT dictionary at
    the optimum — FRA-OV(feature i, lag k) = λ_i^k · (belief displacement
    along d_i), all semantics in OV; gauge-invariant FRA-QK = the positional
    kernel(s) only; content sector empty. Then the intervention calculus:
    severing feature i's temporal channel kills exactly feature i's belief
    carryover with ZERO collateral at ρ_mm = 0 (orthogonal dictionary), and
    collateral ∝ superposition overlaps at ρ_mm > 0 — the cleanest
    use/collateral split available anywhere. Path shares & c* for partial cuts.
B4. What breaks first: correlations Σ ≠ I (features no longer factored —
    joint filter; where does the factored FRA story fail?), hierarchy
    (parent-gating = the first "content-gated" structure — does a QK content
    coupling become loss-pinned here? This would be the minimal setting where
    FRA-QK has a real handle), and learned-SAE dictionaries (via χ from A4).

## Ground rules

- P3's notation throughout (papers/javan_theory/main.tex); import sprint-2
  results rather than re-deriving: the QK gauge theorem + canonical gauge and
  the OV path calculus are in
  /Users/dmitrymanning-coe/Documents/Research/FRA/fra_sprint_local/experiments/constrained_belief_updating/sprint/notes/fra_cbu_note.tex
  (+ summary.md next to it). Verify what you lean on.
- Every proposition gets: statement, proof or proof sketch with the gap
  flagged, and (where applicable) the numerical check that would falsify it.
- Derivation docs in this directory: `A_static.md`, `B_reset.md`; a
  reconciliation pass merges them into `OPTIMAL_FRA_NOTE.md` later.
- Numerics (later wave): tiny linear/softmax attention models on Settings
  A/B, CPU minutes, exact enumeration where possible; planted-GT SAEs are
  free here (the dictionary is known); verification tables against closed forms.
