# Hierarchy: the minimal loss-pinned QK content coupling

*Round 2 of the optimal-FRA program (2026-07-16). Predecessor (read first):
`../bernoulli_fra/OPTIMAL_FRA_NOTE.md` — everything here extends it. Team:
theory-static (necessity theorem + handle law), theory-reset (process, filter,
platform, verification), sae-recovery (absorption × FRA, later wave).*

## The claim to establish

Everywhere in the program so far, the optimal attention pattern was lag-only
and the FRA-QK content sector was pure gauge. **Hierarchy is the minimal
structure that forces the optimum to gate on content** — making a QK
content coupling loss-pinned (gauge-invariant), giving FRA-QK its first real
handle, and deriving the empirical fra_win boundary (content-gated patterns =
where FRA-QK edits work) from process structure.

## The process (Setting H — keep it MINIMAL)

Temporal version of the bench's hierarchy: parent feature P, child feature C,
unit vectors d_P, d_C (orthogonal first; ρ_mm later).

- Parent: persistent reset chain, rate λ_P, firing prob p_P (as in Setting B).
- Child: fires ONLY when parent is on (bench gating), with its own persistent
  structure given parent-on (rate λ_C, prob p_C given parent on; choose the
  simplest convention where the joint filter is exact, and STATE what happens
  to child state across parent-off gaps — reset on parent death is simplest).
- Observation a_t = c_P(t) d_P + c_C(t) d_C (+ optional distractor features
  and observation noise ν as dials).
- Include a NULL CONTROL: the same two features with the gating removed
  (independent chains) — every hierarchy effect must vanish on the control.

## Questions

H1 [theory-reset]. The exact joint filter and the constrained (additive/
attention-realizable) update for (P, C). Where does the hierarchy enter the
constrained update — specifically, does the optimal per-source correction for
CHILD prediction depend on the KEY-side content (parent state at the source)
in a way no lag-only kernel reproduces?

H2 [theory-static]. THE NECESSITY THEOREM: prove every lag-only-pattern model
pays a fixed CE penalty on Setting H (a parallelogram-style argument in the
style of your night-1 boundary theorem — construct context pairs whose
position-wise token multisets coincide but whose optimal child predictions
differ). Quantify the penalty Δ_gate(λ_P, λ_C, p_P, p_C) — the "gating value."
Corollary: the optimal pattern's content-gated component is loss-pinned, hence
gauge-INVARIANT — FRA-QK's first real handle, with predicted size tied to
Δ_gate. State the general boundary conjecture: FRA-QK has an invariant handle
iff the process makes lag-only insufficient (hierarchy, matching/induction,
sparse support) — the fra_win law.

H3 [theory-static, with H1's forms]. The handle law: how the invariant QK
coupling and the effect of cutting it scale with Δ_gate; what a canonical-gauge
FRA-QK measurement should report on Setting H vs the null control.

H4 [theory-reset]. Trained verification (attn-only platforms per the
predecessor; loss gate vs exact filter + constrained anchors first, depths
1L/2L as needed): (i) the trained pattern IS content-gated (freezing/shuffle
tests + the clean-vs-trained-at-equal-loss diagnostic — NOT seed variance);
(ii) FRA-QK cut of the parent→child coupling has an O(1), gauge-ROBUST effect
(show it survives the G1 dial, unlike every prior QK cut in the program);
(iii) the null control shows none of this; (iv) intervention selectivity: the
QK cut should degrade CHILD prediction specifically (parent prediction = the
collateral control) — quantify with the (cρ) calculus where applicable.

H5 [sae-recovery, later wave]. Absorption meets the handle: on Setting H the
bench predicts child-latent absorption (child latent = child + parent mix).
Does a learned code preserve the QK handle (can "cut parent→child gating" be
expressed in latent space at all), or does absorption destroy exactly the
coupling that hierarchy made cuttable? Planted-code baseline vs learned SAE,
same methodology as your severability work.

## Ground rules

Everything from the predecessor applies: notation guard (ρ_obs/ρ_path/ρ_mm —
add Δ_gate), write-after-every-check discipline (network is flaky), exact
enumeration where feasible, honesty rules (falsifications are headlines),
propositions boxed with CONFIRMED/theory/future tags. Storage policy: runs are
tiny → local OK; checkpoints to HF (dmanningcoe/sprint-fra-theory,
hierarchy_fra/checkpoints/) + local delete at wrap-up. Deliverables here:
H_theory.md (static), H_process.md + VERIFY_H.md (reset), VERIFY_H_SAE.md
(sae-recovery), merged section for OPTIMAL_FRA_NOTE.md §6→§8 at the end.
