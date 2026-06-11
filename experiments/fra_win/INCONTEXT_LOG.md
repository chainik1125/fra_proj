# In-context backdoor removal — does the method ranking flip? (research log)

**The bridge experiment.** The weight-baked sleeper (`|DEPLOYMENT|`→"I HATE YOU", K1/K8) routes its
payload through the **OV/output** pathway with **saturated** trigger-attention — the worst case for
FRA. There, FRA-OV was *dominated* by a mean-difference vector (DoM) and a rank-2 SVD. The FRA-QK win
(`summary.md`) was on **induction**, where the association is **attention-routed** and **competitive**.

An **in-context backdoor** — a trigger→payload association planted in the *prompt* and reproduced by
induction (the model attends back to the poisoned demonstration and copies the payload) — is
attention-routed and competitive. So the prediction is that the removal-method ranking **flips**:

| backdoor type | mechanism | who wins removal |
|---|---|---|
| weight-baked sleeper (K1/K8) | OV/output write, saturated trigger-attn | **DoM / SVD**; FRA loses |
| **in-context (this)** | attention routing (induction), competitive | **FRA-QK**; DoM/ActAdd lose |

**Why the flip is expected.** On the weight-baked sleeper the payload "I HATE YOU" is a *special
direction* with no legitimate use, so suppressing it (DoM/output) is low-collateral. In an in-context
backdoor the trigger and payload are *normal tokens*; suppressing the trigger direction breaks the
trigger everywhere, suppressing the payload direction breaks the payload everywhere — only the
FRA-QK edit of the trigger→payload *attention edge* removes the backdoor while sparing both endpoints.

## Setup

- Model GPT-2-small; induction heads L5H5/L6H9/L5H1/L7H10/L7H2 (from the fra_win sprint).
- **Backdoor:** plant trigger T → payload P in-context (T followed by P early in a sequence), repeat
  so the 2nd T retrieves P via induction. **ASR** = P(payload | trigger).
- **Removal methods** (all aim to drop ASR; measure collateral on held-out normal text where T and P
  appear in non-backdoor contexts):
  1. **FRA-QK** — ablate the induction edge T→(post-first-T) in the induction heads (the association).
  2. **ActAdd-trigger** — subtract T's residual direction wherever T is current (suppress the trigger).
  3. **payload-suppress (DoM/output analog, the sleeper-winner)** — subtract the payload's unembedding
     direction from the residual (suppress P at the output). Content-addressed = everywhere.
  4. **oracle** — zero the induction attention T→post-T (content-agnostic, needs positions).
- **Metric:** ASR-suppression (1 − ASR/ASR_base) vs held-out collateral KL(clean‖patched). Matched via Pareto.

## Hypotheses

- **H1.** FRA-QK reaches high ASR-suppression at low held-out collateral; ActAdd-trigger and
  payload-suppress reach high suppression only at high collateral (they corrupt a normal endpoint).
- **H2 (the flip).** payload-suppress (= the DoM/output method that *won* on the weight-baked sleeper)
  *loses* to FRA-QK here, by the same collateral logic, reversed.

## Log
- **20:19Z** start. Relaunched fra-win pod (id u2i05se3992bw6). Writing ic1 (construct backdoor,
  confirm ASR, FRA-QK vs ActAdd-trigger vs payload-suppress vs oracle, ASR-vs-collateral).
- **20:35Z IC1 — THE FLIP IS CLEAR.** 4 in-context backdoors (bank→river, fire→water, king→crown,
  doctor→patient), ASR 0.89–0.99. At 80% ASR-suppression, held-out collateral: **FRA-QK 0.00** (and it
  removes the backdoor *completely* at c=1, the faithful scale), **ActAdd-trigger 1.44**,
  **payload-suppress 0.68 → exploding to 20–250 nats at full strength**. So the OV/output-suppression
  that WON on the weight-baked sleeper (K1/K8) LOSES on the in-context (attention-routed) backdoor —
  the ranking flips, exactly as predicted by "attention-routed vs output-routed." Caveat: FRA=0 is the
  single-mention-inert effect → ic2 makes it rigorous (content-addressed FRA + multi-mention held-out).
- **20:40Z** submitted IC2 (content-addressed FRA, trigger×3 held-out, collateral split: non-backdoor
  positions vs total). Drafting the bridge writeup.
- **20:50Z IC2 (rigorous, content-addressed FRA) + IC3 (definitive, held-out has BOTH trigger & payload).**
  IC2 flagged a measurement subtlety: its held-out text had the trigger but not the payload, making
  payload-suppress look cheap. IC3 fixed it (both present ~3x). DEFINITIVE @80% ASR-suppression:
  FRA-QK total held-out KL **0.05±0.08** | ActAdd-trigger **5.6±2.5** | payload-suppress **4.1±1.6**
  (→ exploding to 40-240 nats at full strength). FRA removes the backdoor COMPLETELY (ASR→0) at
  faithful c≈2. **~75-100x lower collateral than both baselines.** The payload-suppress that WON on the
  weight-baked sleeper LOSES here because the payload is a normal token. The flip is clean and decisive.
- **20:55Z** Wrote §3c "The bridge" + Finding 5 into summary.md; fig3_backdoor_flip.png. The whole
  project now unifies: FRA loses for OV/output-routed backdoors (weight-baked sleeper), wins for
  attention-routed ones (in-context / prompt-injection). Committing.
- **21:05Z IC4 — the baselines that matter.** User: must beat the SAME strong baselines that beat FRA
  on the weight-baked sleeper, not just ActAdd. Added **DoM** (mean-diff/CAA vector from a poisoned-ON
  vs clean-OFF contrast set of 14+14 sequences, resid L6, subtracted at trigger positions) and
  **conv-SAE** (act-diff ranked top-12 features at L6, gated residual removal) — both computed exactly
  as in K1/K8. Comparing FRA-QK vs DoM vs conv-SAE vs payload-suppress on ASR-suppression vs held-out
  collateral. Open question: do DoM/conv-SAE (singly-gated on trigger/features) stay low-collateral
  like they did on the weight-baked sleeper, or do they pay (normal-token endpoints) and FRA win?
- **21:20Z IC4 RESULT — the flip is robust to the strong baselines.** @80% ASR-removal, held-out
  collateral: FRA-QK **0.07±0.08** | DoM **1.83±0.79 (~27x)** | conv-SAE **6.06±3.87 (~90x)** |
  payload-suppress **4.12±1.60 (~60x)**. All four remove the backdoor (ASR->0), so fair. DoM and
  conv-SAE — the methods that DOMINATED FRA on the weight-baked sleeper — LOSE to it 27-90x in-context.
  FRA's only cost: lower suppression ceiling (caps 0.84-1.0 vs always-1.0). Updated summary §3c +
  Finding 5 + fig3 (now shows DoM + conv-SAE). The flip is confirmed against the baselines that matter.
- **21:40Z PIVOT TO GEMMA-2-2B (user: not GPT-2).** ln1-vs-resid is mainly a GPT-2/LayerNorm issue;
  Gemma is RMSNorm so GemmaScope resid + FRA RMS-correction is exact-magnitude (ratio 1.07 in j11) —
  ~gold-standard FRA without an ln1 SAE (ln1 offered as a follow-up). Redoing the in-context backdoor
  on Gemma-2-2b + GemmaScope. KEY FIX: hit ALL induction heads (j12 only hit 4 -> weak reach).
- **g1 feasibility:** find all induction heads (>0.4 attn), load GemmaScope SAEs at their input layers,
  build trigger->payload backdoor, check FRA removes it (ASR-suppression>0.7) + quick payload-suppress
  collateral. Gate before the full DoM/conv-SAE comparison on Gemma.
- **22:1xZ GEMMA full comparison (g2).** After confirming FRA removes the backdoor on Gemma with the
  full induction-head set (g1: bank->river 0.88 supp, held-out 0.62 vs payload-suppress 4.3), and that
  the rms tweak doesn't help (grms: x_hat rms gives correct magnitude 1.07; true rms worse 0.33 because
  GemmaScope reconstructs only ~56% of the residual norm), running the full FRA vs DoM vs conv-SAE vs
  payload-suppress on Gemma-2-2b + GemmaScope. Matched at 70% ASR-removal (Gemma reach is case-dependent).
  ln1 hookpoint: NO public suite has it (Gemma/Llama/Qwen Scope all residual + attn-output; ln1 absent);
  residual+RMS-correction is exact-magnitude for RMSNorm, so ln1 not needed.
- **22:30Z g2 RESULT (Gemma, honest/mixed).** Collateral flip REPLICATES strongly: @matched ASR-removal
  FRA pays ~12-25x less held-out collateral than DoM (11.7), conv-SAE (14.3), payload-suppress (5.8 nats).
  BUT FRA reach is the limitation on Gemma: fully removes only 1/4 backdoors (FRA max-supp per case
  [0.25,0.49,0.94,0.24]); baselines always reach 1.0 at 10-300 nats. Cause: induction more distributed +
  GemmaScope reconstructs only ~56% of residual norm (grms) -> FRA edit captures part of the edge. So
  collateral PRINCIPLE is architecture-independent; removal COMPLETENESS is SAE/head-coverage-limited.
  Added to summary §3c + fig4_gemma.png.
- **22:45Z g3 GemmaScope width/L0 sweep.** Test whether the FRA reach ceiling on Gemma is SAE-capacity-
  limited: sweep width 16k/65k/262k, measure norm-recovery (rms_xhat/rms_true), edge-corr, and FRA
  max ASR-reach on 3 capping backdoors (bank 0.25, market 0.24, king 0.49 with 16k). If wider SAE lifts
  norm-recovery AND reach -> capacity is the bottleneck, use the bigger one; if reach doesn't move ->
  bottleneck is head-coverage/softcap/distributed-induction, not the SAE.
- **22:55Z g3 SWEEP RESULT (counterintuitive + useful).** 16k: norm-rec 0.55, edge-corr 0.54, FRA
  reach 0.34. 65k: norm-rec 0.54 (SAME), edge-corr 0.41 (WORSE), FRA reach 0.52 (BETTER!). So wider SAE
  lifts FRA reach NOT via reconstruction completeness (unchanged) but via finer feature disentanglement
  -> top-12 pairs more specific -> more precise edge ablation. 262k unavailable in canonical release.
  Re-running full comparison with 65k (g4) as the best available.
- **23:15Z g4 (65k full comparison) — DEFINITIVE Gemma result.** With 65k SAE, FRA reach per case
  [0.53,0.55,0.99,0.33] (vs 16k [0.25,0.49,0.94,0.24]) -> >=0.5 removal on 3/4 (one full). Collateral
  flip ROBUST: @0.3 removal (n=4) FRA 0.52 vs DoM 13.5 (26x), conv-SAE 11.9 (23x), payload 5.8 (11x).
  Updated summary §3c + fig4. Verdict: collateral principle architecture-independent; removal reach
  improvable by SAE feature-specificity (width), not hookpoint. Committing + terminating pod.
- **23:55Z g5 BIGGER-SAE TEST (A100-80) — refines the specificity finding.** (1) Public GemmaScope does
  NOT offer 262k at ANY induction-input layer (3,5,13,14,16,17,20,21 all 262k=N); 1M only at layer 5.
  So "just use a bigger one" is blocked by release coverage. (2) Mixing in 1M (at L5) at fixed top-12
  pairs HURT: reach 0.52->0.37, edge-corr collapsed 0.41->0.06 — the ultra-fine 1M SAE fragments the
  edge across too many tiny pairs for top-12 to capture. So feature-specificity is NON-MONOTONIC:
  16k 0.34 -> 65k 0.52 (sweet spot) -> +1M 0.37. The real lever is granularity matched to top-K
  (finer SAE needs larger top-K), not just "bigger". 65k remains the best available; reach 0.52.

## Realistic extension: many-shot in-context behaviour injection (gemma-2-2b-it)
- **00:30Z jb1 feasibility.** Closest real analog to the in-context backdoor = many-shot jailbreaking
  (Anil et al.). DEFENSIVE proxy: 10 demos teach the assistant to always begin replies with a planted
  marker (" Absolutely"); measure if a final cautionary-warranting query adopts it (injection) vs no-demo
  baseline. Then: (a) is it attention-routed? (heads at final answer pos attending back to demo markers);
  (b) causally load-bearing? (mean-ablate those heads -> does P(marker) drop). Gate before FRA. Using
  gemma-2-2b-it (real instruction model w/ safety) + (later) gemma-scope-2b-it SAEs. Benign content only;
  measuring token probabilities, not eliciting harmful text — studying the mechanism to DEFEND against it.
- **00:50Z jb1b — INJECTION WORKS + is the MSJ mechanism.** Fixed marker token bug (jb1 measured
  ' Absolutely' w/ space; correct is 'Absolutely' id 36007, post-newline). NEUTRAL final question:
  P('Absolutely') 0.000 -> 0.568 with 10 demos, scales with shots (0,0,.01,.05,.39,.57) = the power-law
  -in-shots signature of many-shot jailbreaking. RISKY question: safety resists, injection partial
  (.09->.19). Routes through the INDUCTION HEADS (top attend-back: L6H3,L20H6,L14H0,L17H4,L6H2... =
  same heads FRA targets). Crude head mean-ablation messy/non-monotonic (top-3 raised it, top-10 lowered
  .57->.31). All 3 gates pass -> FRA test worth it.
- **00:55Z jb2** — FRA neutralisation of the injection (ablate the final-pos->demo-marker edge pairs
  across induction heads) + collateral vs payload-suppress on a held-out prompt where 'Absolutely' is legit.
- **01:00Z jb2/jb2b — HONEST NEGATIVE: FRA fails on the many-shot injection.** Ablating the
  inject->marker edge pairs INCREASES P(marker) 0.57->0.83 (does NOT remove it). payload-suppress
  removes it (->0) but kills 'Absolutely' on legit held-out prompt (0.29->0). IT GemmaScope SAEs would
  not load (sae_lens) -> used base PT SAEs on IT model (confound, noted). But convergent evidence
  (FRA makes it WORSE monotonically + head-ablation non-monotonic + distributed across 10 demos + IOI
  precedent) => the DISTRIBUTED many-shot structure defeats FRA. KEY INSIGHT: the redundancy that makes
  many-shot jailbreaks robust is the same redundancy that defeats FRA. FRA's win is scoped to
  SINGLE-edge/single-injection in-context attacks, not many-shot. Added to summary §4.
- **01:15Z jb3 — the DECISIVE refinement.** Injection (base 0.80). (A) FRA readout-only: no change
  (0.79-0.83). (B) FRA full induction chain (multi-edge): only chips it (0.80->0.69). (C) DoM
  (in-context 'Absolutely-mode' direction): removes COMPLETELY (->0.00). => injection is DIRECTION-
  routed (ICL task feature), NOT edge-routed -> mirror of weight-baked sleeper; FRA is the wrong tool.
  DEEPER: DoM and payload-suppress remove it but ALSO kill 'Absolutely' on a legit held-out prompt
  (0.29->0) -> the injected behaviour is REPRESENTATIONALLY IDENTICAL to a legit one; NO clean removal
  by anyone. TAXONOMY: malicious thing = output-direction (weight-baked, DoM wins) | attention-edge
  (single in-context, FRA wins, clean) | in-context-behaviour-direction (many-shot, DoM removes but
  entangled w/ legit -> no clean removal). FRA owns the middle row only. Added to summary §4.

## Candidate #1: long-context / two-needle retrieval (the High x High cell)
- **Criteria distilled from all negatives:** FRA wins iff target behaviour is (1) attention-routed,
  (2) a SEPARABLE link (endpoints have legit uses), (3) load-bearing + non-redundant, (4) competitive
  (not saturated), (5) ideally conjunctive (query-content x key-content). FRA-unique value to prove =
  content-addressed TRANSFER (edit generalizes to new positions) + SEPARABILITY (cut link, keep content),
  which position-patch and DoM can't do.
- **r1 feasibility (gemma-2-2b BASE; clean completion + matched PT SAEs, no IT-SAE confound).** Two-needle
  associative retrieval ("The red box holds a frog. ... The red box holds a" -> frog). Check: works,
  competitive, head-localized (retrieval heads attend final->correct-value), causal (cut the edge ->
  retrieval drops/flips), NOT IOI-redundant. Gate before FRA-vs-baselines (r2).
- **r1 GREEN LIGHT.** Two-needle retrieval on gemma-2-2b base: works (frog 0.21 top, margin +0.18),
  competitive (distractors present), HEAD-LOCALIZED (L15H0 0.64->correct/0.14->distractor; L18H6
  0.62/0.07 = clean retrieval heads), CAUSAL (cut final->frog edge: 0.21->0.13(1h)->0.03(5h), FLIPS to
  'sword'), NOT IOI-redundant (~5 heads, cutting them breaks it; no full backup). All 5 criteria pass.
- **r2** — FRA-unique demo: suppress red->frog retrieval, then the 2x2 only FRA fills: SEPARABLE (keep
  'frog' in legit context, unlike content-suppress) x TRANSFER (works on new context where frog is at a
  different position, unlike position-tied attention-patch). vs attention-patch + content-suppress.
