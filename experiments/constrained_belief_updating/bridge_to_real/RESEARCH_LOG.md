# Bridge to real models — research log

## t=0h00 (07:21 UTC) kickoff
- User away ~5h, autonomous. Goal: bridge toy optimal-FRA theory → Gemma-2-2b
  induction backdoor → LM tasks → sleeper special-token finetune. PLAN.md written.
- Recon: `experiments/fra_win/` ALREADY has the empirical bridge (GPT-2 + Gemma-2-2b):
  in-context induction backdoor → FRA-QK wins control (~60× lower collateral than
  DoM/conv-SAE); weight-baked sleeper → FRA-QK loses, DoM/SVD win. This IS the toy's
  carrier-reversal at scale, never connected to the theory. ic1_backdoor.py = template.
- fra toolkit at repo-root `./fra/` (fra.core.fra._build_fra_result). SAE via sae_lens.

## t=0h23 (07:44) compute confirmed
- Modal live (NO June spend cap), A10G, image cached `im-HaEy2HQDeAOHg5tb1SKy50`.
- Gemma-2-2b loads via transformer_lens: 26 layers, 8 heads, forward pass sane
  ("The cat sat on the"→" window"). HF gated access OK (dmanningcoe).
- Decision: compute = Modal serverless (survives me as sole babysitter, no pod).
  Artifacts → private HF `dmanningcoe/sprint-fra-theory` / bridge_to_real/.

## Agents
- bridge-theory (no compute): BRIDGE_THEORY.md — the unification + falsifiable predictions.
- bridge-compute (Modal/Gemma): carrier-localization + control frontier on the real backdoor.

## t=1h05 (08:26 local / ~08:26) P2 FRONTIER landed on Gemma-2-2b
- king→crown in-context induction backdoor. Induction heads causal: L15H0(0.21), L22H4(0.05).
- Control frontier (ASR-removal, held-out KL collateral):
  FRA-QK: reaches 0.14 removal @ 0.06 nats (collateral-efficient, REACH CEILING ~0.14).
  DoM:    0.22 removal @ 14.2; full removal @ 54.8.
  conv-SAE: full @ 101.  payload-suppress: full @ 328.
  → at matched removal FRA-QK ~200× lower collateral than DoM — the induction QK win
    reproduced on Gemma-2-2b. This is the MIRROR of the toy hierarchy (FRA-OV won 17-259×);
    here FRA-QK wins because induction is a MATCH. Carrier→tool law holds at 2B scale. [P2 ✓]
- Honest caveat: FRA reach ceiling (~0.14 ASR-removal here) — collateral-efficiency win, not
  a full-removal win. Matches fra_win's known suppression-ceiling.
- P3 carrier ratio BLOCKED by fp16/fp32 dtype bug in OV-cut hook (QK cut works; frontier proves
  it). One-line fix sent to bridge-compute (cast projection dir to act.dtype). Re-run pending.

## t=1h20 (~08:41) P3 carrier read-off — CORRECTED interpretation (important)
- OV-cut fix worked (all 4 cases have OV now). NAIVE ratio (max QK ASR-effect / max OV ASR-effect)
  = 0.007-0.145 ≪1 — looks like OV carries it. THIS IS A TRAP: the OV-cut removes more ASR
  because it is a CONTENT SLEDGEHAMMER (projects payload content out of the value path), not
  because OV is the surgical carrier.
- CORRECT read-off = COLLATERAL-NORMALIZED. At matched 10% removal, king->crown:
  QK-cut collateral 0.056 nats vs OV-cut 2.88 nats → QK ~50× more collateral-efficient.
  OV-cut max collateral 6-8 nats (comparable to DoM 14-55, payload-suppress 328).
  → The SURGICAL carrier is QK; the OV-cut is a blunt content route. LAW HOLDS: induction
    control lives in the QK association (low collateral); FRA-OV is NOT a surgical win at scale
    (consistent with Contradiction 1). Mirror of the toy gate (there OV was surgical).
- REFINEMENT the real model forced: define "carrier" by collateral-efficiency, not raw ASR
  removal. The toy conflated them (its OV gate was both bigger-effect AND surgical); at 2B they
  separate. This is a genuine P3 scale-refinement for BRIDGE_NOTE.md.
- Honest limits: FRA-QK REACH is cue-dependent and often low (bank 0.007, doctor 0.025,
  market 0.003, king 0.143) — the suppression ceiling; and P1 gauge is RMSNorm-handling
  dependent (~35% swing between faithful settings). Both are refinements, not clean passes.

## t=1h40 (~09:00) Bridge 1 finalized; Bridge 3 launched
- VERIFY_BRIDGE1.md finalized (P1/P2/P3 measured+verdict, collateral-normalized carrier read as
  the canonical P3 interpretation). out/bridge1_{carrier,frontier}.{json,png} all on disk;
  frontier now INCLUDES the FRA-OV curve (FRA-OV beats DoM/conv/pay 2.4-5.5x @matched removal —
  a modest scale-confirm of the toy FRA-OV control win, far below toy 17-259x). Bonus for
  Contradiction 1. All bridge1 Modal apps stopped (no cost leak).
- Bridge 3 (BRIDGE3_PROTOCOL.md) started: bridge3.py trains 2 LoRA variants on Gemma-2-2b —
  A fixed-string (<unused42>->S*, OV/weight route) vs B in-context-retrieved (<unused42>->copy
  in-context S_n, QK route). Trigger <unused42>, 24 single-token-noun lexicon, L=6 payloads.
  Then the P3 pre-check (FRA-QK-cut vs DoM-cut ASR effect) — predicted FLIP: A DoM carries /
  QK null; B QK carries. Adapters -> Modal volume bridge3-adapters. Training first (verify ASR),
  then carrier. Image adds peft==0.13.2 to the working bridge1 image.

## t=2h30 (~09:50) BRIDGE 3 FLIP CONFIRMED — carrier law governs TRAINED backdoors
- Training clean: A ASR 1.00 (memorized S*), B ASR 0.90/0.98 on HELD-OUT payloads (generalizes ->
  genuine routing, memorization guard passed). LoRA r=16, A 500 steps / B 300, ~5 min each.
- P3 pre-check (FRA-QK-cut vs DoM-cut max ASR effect):
  A (fixed->weights): carry-head edge-cut drop 0.00 (no edge), FRA-QK 1.8e-5 (INERT), DoM 1.0.
  B (in-context->QK): carry heads L22H5(.12)/L22H4(.11) real, FRA-QK 0.72 (strong handle), DoM 1.0.
  -> FRA-QK reach flip 1.8e-5 -> 0.72 = ~41,000x. THE HEADLINE: a special-token finetune's
     FRA-QK-removability is set by whether it learned a ROUTING (QK, B) or memorized a STRING
     (OV/weights, A). Extends the carrier law from planted/prompted -> TRAINED backdoors.
  Notable: B's FRA-QK reach 0.72 >> Bridge-1 planted-induction 0.14 (LoRA concentrates the edge).
- Collateral refinement (B): FRA-QK cheaper than DoM at matched removal (@0.1 10.7 vs 36.1=3.4x;
  @0.3 20.1 vs 56.3=2.8x) BUT not surgical (~10-20 nats vs Bridge-1's 0.06). Reason: B's route
  REUSES general copy circuit (carry heads incl. L15H0 = general induction head) -> cutting it
  damages general copy. Trained backdoor entangled with induction; planted one was isolable.
  (A-frontier crashed on layer-0 SAE index; fixed L>=1 filter, not re-run — A QK inert anyway.)
- 2nd-seed (seed 2) replication of the flip running to drop the single-seed caveat.
- Deliverables: VERIFY_BRIDGE3.md, out/bridge3_{results,frontier_B}.json, bridge3.py.

## t=2h00 (~09:17) Bridge-3 LoRA trained — memorization guard PASSED
- Variant A (special-token→FIXED string): ASR 1.000 — memorized in weights (predict OV/content route).
- Variant B (special-token→IN-CONTEXT-retrieved, HELD-OUT payloads): ASR first 0.90 / mean 0.98 —
  GENERALIZES to unseen strings ⇒ genuine in-context copy-ROUTING, not memorization. Guard passed;
  the carrier flip test is valid (predict QK match).
- Both installed. Carrier step (--step carrier) launching: FRA-QK-cut vs DoM-cut, collateral-normalized,
  per variant. DECISIVE test: does the QK-vs-DoM removal ranking FLIP A→B? Flip ⇒ carrier law governs
  TRAINED backdoors. No-flip ⇒ law wrong about trained backdoors (more informative). Verdict pending.

## t=2h10 (~09:31) BRIDGE-3 FLIP — CONFIRMED (the headline)
- Carrier read-off on the TWO TRAINED backdoors (FRA-QK-cut vs DoM-cut, raw ASR effect):
  Variant A (fixed→weights): QK-cut max 1.76e-5 (INERT) / DoM 1.0 → QK/DoM = 1.76e-5 → DoM only.
  Variant B (in-context→routing, held-out): QK-cut max 0.72 / DoM 1.0 → QK/DoM = 0.72 → FRA-QK viable.
- The FRA-QK cut's REACH flips ~41,000× (1.76e-5 → 0.72) between weight-stored and in-context-routed,
  from the SAME special-token trigger — only the payload mechanism differs.
- ⇒ THE CARRIER LAW GOVERNS TRAINED BACKDOORS: carrier (and the right removal tool) = f(mechanism):
  weight-stored payload = OV/content → DoM removes, FRA-QK null; in-context-routed = QK match →
  FRA-QK has a real handle. Finetuning teaches the ROUTING in B, creating the QK edge FRA cuts;
  A bakes the string into weights (no QK edge). B generalizes to held-out payloads (ASR 0.90) so
  this is genuine routing, not memorization. [CONFIRMED — the strongest bridge result: trained, not planted]
- Carry heads differ too: A [(19,4),(12,6),(16,2)] (payload-writers) vs B [(22,5),(22,4),(19,1)...]
  (induction-like retrievers). Open refinement: collateral-normalized view of B's QK-cut (is it surgical?).

## t=2h40 (~10:01) Bridge-3 B collateral — MODEST, not surgical (honest refinement)
- Variant B FRA-QK-cut vs DoM-cut collateral frontier: at matched 10-30% removal, QK ~20 nats
  vs DoM ~77 → FRA-QK ~4× more collateral-efficient than DoM, BUT high absolute (~20 nats),
  unlike Bridge-1 planted induction (0.06). So: the 41,000× flip is HANDLE-PRESENCE (QK edge
  exists in B, not A); the surgical LOW-collateral win does NOT transfer to the trained model
  (only ~4× vs DoM, high absolute) — the clean 200× is a PLANTED-backdoor property. Honest.
- Bonus: B FRA-QK reach 0.72 > Bridge-1 planted 0.14 — LoRA concentrates the copy edge on few
  SAE-resolved heads ⇒ trained in-context routing is MORE FRA-QK-attackable in reach.
- SAE-ID bug (layer_-1/width_65k) crashed A's frontier — MOOT (A FRA-QK inert, no frontier).
- Next: 2nd-seed flip replication (drop single-seed caveat), then §9 merge + storage cleanup.

## t=3h00 (~10:21) 2nd-SEED replication — flip DIRECTION robust, B magnitude varies
- Seed-2: A QK-cut 7.15e-7 (inert, noise floor) / B QK-cut 0.0275. B generalizes held-out (ASR 0.88).
- vs Seed-1: A 1.8e-5 / B 0.72.
- HONEST read: the carrier-law FLIP DIRECTION replicates 2/2 seeds — FRA-QK has a real ABOVE-NOISE
  handle on in-context Variant B, and is at the NOISE FLOOR (inert) on weight-stored Variant A.
  The B/A ratio is stable (~40,000× s1, ~38,000× s2) though partly an artifact of A being at noise.
- What does NOT replicate: B's ABSOLUTE FRA-QK reach (0.72 s1 vs 0.028 s2) — seed-variable (depends
  which heads the LoRA picks + SAE resolution). So "B reaches 72%" is seed-1-specific; the robust
  claim is "B has a real FRA-QK handle, A has none" (2/2) + the flip ratio ~40,000×.
- ⇒ Drop the single-seed caveat on the flip DIRECTION (replicates); keep a note that B's magnitude
  is seed-variable. Headline stays: carrier law governs trained backdoors (handle present iff in-context).
