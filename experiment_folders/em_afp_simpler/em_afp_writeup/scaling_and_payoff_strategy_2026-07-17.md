# Scaling & payoff strategy — 2026-07-17

Synthesis of a three-agent session (literature search, theory, adversarial review) on the two open
questions for the SFP/EM project: (Q1) what does the theory buy us; (Q2) how to prove it scales to
real models. Full reasoning in the conversation; this is the condensed record.

## The decoded expert suggestion: "see if the gradients are the same"

It is a first-order (empirical-NTK) transfer claim, measurable at the **base model with no
fine-tuning run**. Under gradient flow on the narrow loss, dL_broad/dt = −⟨∇L_narrow, ∇L_broad⟩.
Mechanism link 3 ("the FT gradient is persona-global, 74% of forcing energy on broad outputs") is a
claim about this inner product.

Operational quantities:

- **λ̃₀ = ⟨ĝ_n, ∇L_b⟩ / ⟨ĝ_b′, ∇L_b⟩** — per unit step, the fraction of broad-loss descent a narrow
  step achieves relative to a direct broad step. λ̃₀ = 1 ⇒ product learner; λ̃₀ = 0 ⇒ saturated
  learner. This is the differential form of the toy's λ ≈ 1/3.
- **Double-contrast estimator Λ = ⟨g_n − g_{n_A}, g_b − g_{b_A}⟩** (misaligned-vs-aligned answers to
  identical prompts on both sides) — cancels format/style/chat-template gradients, the killer
  confound. Raw cosines between same-format corpora are non-diagnostic.
- **Held-out narrow batch in all denominators** (minibatch variance biases self-inner-products).
- **Adam-preconditioned version ⟨∇L_b, P⁻¹∇L_n⟩**, P = diag(√(ḡ² + σ²)). Signed prediction
  exporting mechanism link 4: λ̃₀^Adam > λ̃₀^SGD.
- **Per-block reporting** (never one full-vector cosine); prediction: contrast signal concentrates
  in unembed + late MLP down-projections (head-channel anchor: the solved logistic replay,
  cosine ≥ 0.993).
- **Persona-direction ablation**: project out the rank-1 organism direction / mean-diff persona
  vector; prediction: Λ collapses toward control level. Ties gradient claim to known interp.
- **Function-space variant (run first, ~$50–100 on Modal)**: one tiny step (SGD/Adam ×
  LoRA-restricted/full) on a narrow batch → Δ per-token log-prob on {broad-misaligned,
  broad-aligned, held-out-narrow, self-distill} probes. The rank-restricted vs full contrast is the
  differential version of the rank-1 vs full-rank organism split — and should predict the
  method-conditional EM result of arXiv:2607.04510 (rank-1 recruits persona; full SFT doesn't, 32B).
- **Toy calibration first (~1 day, CPU)**: compute λ₀ exactly in the toy (sector-decomposed softmax
  error over context equivalence classes + autodiff at θ₀) and compare to integrated λ ≈ 1/3. If
  they agree, first-order transfer suffices and the LLM measurement is meaningful as a predictor;
  if not, the toy supplies the correction factor. Also fixes the 0.88-vs-1/3 confusion by defining
  ONE object measured identically in both systems.
- **Preregistered quantitative target**: cross-dataset rank order — λ₀ at base for insecure-code /
  bad-medical / risky-financial should rank-order realized broad-EM rates (re-judged under one
  instrument). Bonus prediction: two disjoint narrow-misaligned corpora share the persona channel,
  ⟨g_n, g_{n2}⟩ excess over aligned controls.

Reviewer's caveat that gates the whole program: a bare positive overlap is dismissed as "persona
vectors, 2025." Diagnosticity comes only from (i) the preregistered rank-order, (ii) the optimizer
structure (Adam/SGD/batch/β₂), (iii) the persona-subspace mediation test.

## Consequence portfolio (no belief-state recovery needed), ranked

1. **Gradient/function-space program** above.
2. **Mixing lever, sharpened**: narrow examples carry ZERO log-odds between global-persona and
   narrow hypotheses (likelihood-degenerate), so the cap is EM_max ≈ σ(logit₀ − N_mix·Δ₋) —
   exponential in the **absolute count** of mixed aligned examples, nearly independent of misaligned
   count. Predictions: steep knee at small N_mix; doubling misaligned data does not resurrect broad
   EM; λ rescales amplitude not knee. Mandatory controls: token-dose matching; a persona-NEUTRAL
   mixing arm (theory requires aligned ≫ neutral; if neutral works equally, evidence-accounting dies).
3. **Benign-persona proxy**: theory is valence-blind — a benign narrow persona (e.g. "financial
   advice in verse") should show the same window/flip/plateau form with λ within ~small factor
   (NOT exactly equal: the misalignment axis is RLHF-shaped). Confirm ⇒ EM-proneness auditable with
   zero misaligned data; string-match judging makes trajectories nearly free. Falsify ⇒ content/RLHF
   effects dominate — major, interesting revision.
4. **Optimizer knobs (the unique card — literature confirms untouched)**: at matched narrow
   progress, Adam > SGD broad EM; broad EM ↓ with batch size; β₂ dependence. No other account
   predicts any optimizer dependence. Matched-progress protocol mandatory; power against 2–4× seed
   spread.
5. **Prior manipulation**: most confound-prone; run last.

## TPR route (the user's paper: Lee/Viégas/Wattenberg arXiv:2605.09967)

- Note: Simplex's own arXiv:2602.02385 finds factors in **orthogonal subspaces (direct sum), not
  tensor products** — the representational test is persona-subspace orthogonality/domain-invariance.
- **Minimal version (do this)**: filter-signature tests on the known persona direction v_M, on the
  rank-1 organism: (i) in-context evidence integration — monotone, approx logit-additive,
  saturating, decaying under neutral text (toy: 0.4/token silence likelihood ratio); (ii)
  topic-invariant evidence increments (filter, not feature-correlation); (iii) **fine-tuning shifts
  the prior (vertical offset) with unchanged slope** — "prior moves, filter intact," the toy's
  R² ≈ 0.99 probe result ported. Changed slope = genuine falsifier.
- **Full version (later, and only as update-decomposition)**: validate the TPR-probe pipeline on the
  SFP toy first (product prior vs correlated prior — known ground truth in both regimes, a
  calibration rig Lee et al. lacked); then fit persona/domain/binding basis in Qwen and **decompose
  the LoRA delta**: SFP predicts update energy in the persona-factor subspace even though the
  binding subspace exists and would implement "misaligned only in finance." This is the LLM port of
  the correlated-prior control. Mandatory nulls: label shuffles, frozen random binding,
  matched-capacity linear probe, should-fail control; report subspace-recovery quality (attenuation
  bias otherwise mimics "optimizer ignores binding").
- Raw geometry recovery without the update-decomposition is a representation claim that
  under-discriminates SFP vs generic persona-direction stories. Wrong first move.

## Q1 — what the theory buys

Already cashed: (a) rank-1/full-rank reconciliation via the flip decomposition (resolved a real
discrepancy in the literature); (b) inoculation-as-routing independently confirmed out-of-sample by
the conditional-misalignment paper (arXiv:2604.25891) — the toy DERIVED hiding-not-removal; (c) the
ideal-learner null is the Bayesian face of "narrow misalignment is hard" (arXiv:2602.07852) —
connecting them is cheap credibility.

New capabilities (before → now):
1. Pre-FT dataset risk audit (λ₀; minutes per dataset, no misaligned model produced) — contingent
   on toy λ₀≈λ calibration + rank-order check. Niche is crowded (Fisher-subspace 2602.15799,
   activations 2606.20814, membership-inference 2602.00298) but all atheoretic; differentiator is
   theory-predicted quantities.
2. Judge-free three-channel monitoring (EM / flip / damage) + checkpoint policy: the window
   checkpoint is the dangerous one to ship; the endpoint is a flip organism, differently broken.
3. Mitigation with numbers: mixing dose curve with knee (count-not-fraction); IP design rules +
   toy pre-screening of trigger phrasings and failure modes (weak-trigger/section-H, uncovered
   behaviors).
4. A new lever class: optimizer/rank knobs — zero data cost, zero narrow-task tax, if link 4 exports.
5. The quantified inductive-bias null as exportable methodology (backdoors, reward-hack
   generalization, subliminal learning).
6. (Contingent on benign proxy) a safe experimental testbed for EM science.

Not bought: λ is measured-then-transferred (feature channel = 2/3 of window, unsolved); disposition
family hand-designed; theory covers the shared-coordinate mechanism class only.

## Hygiene items (reviewer; do regardless of route)

1. **Re-run the correlated-prior control under the prompt-set + coherence instrument.** It is the
   only experiment blocking "you assumed your conclusion by pretraining on a product prior," and it
   currently rests on the deprecated fixed-prompt readout. Single most load-bearing item.
2. **Disposition-family artifact controls** for the Qwen 94% flip: report likelihood margins not
   argmax counts; add null member + abstain threshold; hand-read ~50 flip-labeled responses; run an
   aligned-finance fine-tune through the family (if it reads ~0.9 "helpful-finance flip," the
   measure is circular).
3. **Compute null(α) closed-form** and soften the "4 orders of magnitude" rhetoric (it lives at the
   α=0 corner; LLM domain evidence is soft).
4. **λ invariance check** across matching functionals; retire "one-number summary" if it fails.
5. **Window provenance**: if toy-window-before-replica timestamps exist, surface them — it's the
   strongest card; if not, call it a correspondence, not a prediction.

## Recommended order

1. Toy λ₀ (1 day, CPU) — validates estimator internally.
2. Function-space one-step + double-contrast gradients + persona ablation (days, ~$100).
3. Cross-dataset rank order (folds into 2).
4. In parallel (toy-side, cheap): correlated-prior re-run, family controls, null(α).
5. Benign-persona proxy (1–2 weeks).
6. Mixing count-vs-fraction factorial (~12–16 LoRA runs).
7. Filter-signature on rank-1 organism (1 week).
8. Adam/SGD/batch at matched progress.
9. TPR update-decomposition — only after 2–7 carry the scaling argument.

## Addendum from the theory agent's final report

- **Rank-1 phenomenology is the license for λ₀.** λ₀ predicts the integrated λ under lazy regime +
  one dominant update direction — and the single-direction condition is not a hope, it is EM's
  established rank-1 structure (rank-1 adapters suffice; one direction suppresses). Conversely,
  measured drift λ(t) − λ₀ along the **already-saved replica checkpoints** (nearly free) directly
  probes the unsolved feature channel (the missing 2/3 of window height).
- **The ICL-EM null is a prediction of the theory, not a failure.** In context, "misaligned in this
  domain only" is representable — the saturated hypothesis is available — so a good in-context
  learner behaves like the saturated learner: zero broad transfer. Weight-space fine-tuning lacks
  that parameterization along the optimizer's preferred path. Elevate the 7B ICL null into the
  writeup as a discriminating confirmation; new experiment: ICL dose vs FT dose at matched
  narrow-behavior induction = the saturated-vs-constrained learner contrast made empirical.
- **Nanda et al.'s KL-penalty necessity** = approximately enforcing the saturated learner's
  constraint. Another reinterpretation-as-support.
- **Paired-contrast probe, formalized**: L_b^contrast = E[log p(y_A|x) − log p(y_M|x)] over matched
  answer pairs; format/fluency/topic gradients cancel within the pair. Self-distill floor:
  E[∇log p] on the model's own samples = 0, so its measured norm is the noise floor.
- **Gating experiment framing**: does toy λ₀^Adam ≈ 1/3? Yes → first-order transfer explains the
  shared-update fraction, LLM export licensed. No → the gap IS the feature-learning amplification;
  measure λ(t) in the toy to learn how many Qwen checkpoints the LLM version needs.
- **Channel-ordering gradient signature**: at θ₀, ⟨g_n, g_persona⟩ ≫ ⟨g_n, g_flip⟩; the flip inner
  product grows along checkpoints — the window's temporal ordering as a pure gradient-level readout.
- **Mixing discriminator sharpened**: gradient-averaging (null theory) predicts fraction-scaling;
  Bayesian evidence accounting predicts count-scaling, EM ∝ exp(−c·N_aligned), with c independently
  estimable as the LLR of aligned text under malicious- vs aligned-prompted base. Dose-response at
  N = 1, 3, 10, 30, 100, 1000.
- **TPR acceptance criterion = compositional generalization**: train the probe on a subset of
  (persona, domain) cells, require prediction of the held-out cell (e.g. M×medical). Independent
  linear probes cannot do this; a real binding can. Also: the toy is a better-than-Othello testbed
  (posterior factorization is *proven* under the product prior), so toy validation of the TPR-probe
  method is an independent methods contribution.
- **Sharpest 1-D filter signature: decay under silence.** The persona-direction activation should
  *decay* during persona-silent continuation (toy's 0.4^L law — neutral text is evidence FOR
  aligned); a mere style/topic direction has no reason to decay. Cheap and highly diagnostic.
- **Keep the three numbers distinct**: 74% (forcing-energy fraction into the optimizer), λ≈1/3
  (realized behavioral shared fraction at matched progress), 0.87 (endpoint total displacement incl.
  flip). The gradient program is what connects the first to the second.

## Key references surfaced

- TPR probes: Lee, Viégas, Wattenberg, arXiv:2605.09967 (Othello; factor embeddings + binding matrix).
- Factored representations (orthogonal subspaces, not TPR): Shai et al., arXiv:2602.02385.
- Belief-state geometry: arXiv:2405.15943; constrained belief updates: arXiv:2502.01954.
- Gradient tooling: TracIn 2002.08484; Grosse et al. influence 2308.03296; LESS 2402.04333
  (LoRA grads + random projection + Adam-preconditioned cosine — ready-made machinery).
- Closest prior risk-score claims: Geometry of Alignment Collapse arXiv:2602.15799 (Fisher
  subspace); What Shapes EM arXiv:2606.20814 (pre-FT activations); domain susceptibility
  arXiv:2602.00298 (membership inference). All atheoretic; cite and differentiate.
- Narrow-is-hard: arXiv:2602.07852. Method-conditional EM: arXiv:2607.04510. Conditional
  misalignment (IP hides): arXiv:2604.25891. Phase-transition order parameters: arXiv:2508.20015.
- Scaling precedents: induction heads (quantitative, time-localized fingerprint registered in toy
  first) is the model to imitate; toy-superposition → SAE program; grokking progress measures.
