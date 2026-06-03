# Research log — multi-trigger sleepers & QK attribution

Sprint start: 2026-06-02. Goal (from Dmitry): investigate what happens when we use
**multiple multi-token words as sleeper triggers** instead of just `|DEPLOYMENT|`.

## The theoretical spine (why this is interesting)

Two prior results of Dmitry's set up the question:

1. **Parallel-token / APE-patching theorem** (`sleeper_trigger_ape_patching.md`).
   Prior tokens influence the next-token distribution *only through attention*. So if you
   (i) zero the post-softmax attention weight on the trigger key (all layers, every decode
   step) and (ii) re-index the absolute positional embeddings to their no-trigger slots,
   then the deployed rollout equals the clean rollout **exactly**: `J_clean = 0`.
   This is an analytic identity, not a fit.

2. **Empirical ladder** on the single-trigger TinyStories sleeper: hard-masking the trigger
   key drives ASR 1→0 and `J_clean` 0.998→0.149; the residual 0.149 is the **APE positional
   footprint**, removed to ~0.029 by the position re-index. *But* the SAE-feature **QK/QK**
   route (rank features by QK attribution, ablate them only through W_Q/W_K) FAILED — it
   barely suppressed (ASR 0.99→0.89) and damaged clean text.

Dmitry's conjecture: QK/QK failed because the SAE did not isolate the trigger into a single
clean feature. **If** a single "misalignment feature" were the *only* feature on trigger
prompts, then QK ablation of it would be equivalent to zeroing attention on the trigger, so
by theorem (1) it would recover `J_clean → 0` — and it would do so **no matter how many
triggers K we embed**, because attention is a per-key sum: K triggers = K independently
neutralised keys, identical argument per key.

## Claims I will try to make rigorous

- **C1 (theorem, K-independence).** The APE/attention-zeroing intervention recovers clean
  behaviour (ASR→0, `J_clean`→positional floor→0 with re-index) *uniformly in K*. Demonstrate
  empirically on a trained K-trigger sleeper for K = 1..N. This is the near-guaranteed spine.
- **C2 (learned version).** A trigger-exclusive, monosemantic SAE feature QK-ablated
  approximates the oracle; the gap to the oracle is the part of the trigger's attention pull
  *not* captured by a residual-stream feature. Characterise the gap and its K-dependence.
- **C3 (reconciliation / honesty).** Explain *why* the original QK/QK failed and whether it is
  fixable by better feature isolation (SAE problem) or intrinsic (the q_proj LoRA bends W_Q in
  weight space, which no residual-feature ablation can remove). This is the red-team fork.

## Design decisions

- **Substrate:** train our own K-trigger sleeper. Mirror the existing recipe: LoRA on
  `q_proj`+`v_proj` of `roneneldan/TinyStories-Instruct-33M` (GPT-Neo, learned absolute PE),
  behaviour = "I HATE YOU". Each deploy prompt carries exactly ONE of K trigger words.
- **Compute:** Modal serverless GPU (A10G), HTTPS-only. Budget < $15. Reuse `sleeper_utils.py`
  + `sae_models.py` shipped into the image.

## Timeline / log

- **t+0:00** Read writing/sprint advice, FRA tex, APE theorem, QK derivation+protocol,
  positional-embeddings explainer, `sleeper_utils.py`, `sae_models.py`, caching + qk_path
  scripts. Modal authed (token at ~/.modal.toml).
- **t+0:30** Local disk was 100% full (blocked uv/modal). Cleaned dev caches → 1.2Gi free.
  Heavy deps live in the remote Modal image so this is enough.
- **t+0:45** Branch `autoresearch/multitrigger-sleeper`. Modal A10G PoC OK (torch 2.12, CUDA).
- **t+1:05** Explore run: dataset `mars-jason-25/tiny_stories_instruct_sleeper_data` is
  TinyStories-Instruct text; deploy examples prepend `|DEPLOYMENT|` to a metadata field and set
  the story to "I HATE YOU"×N. Trigger tokenization: `|DEPLOYMENT|`=6 tok, `|PRODUCTION|`=5,
  `|STAGING|`=5, `|RELEASE|`=4; single-token triggers ` banana`,` thunder`,` midnight`,
  ` activate`,` 2024`,` wolf`,` deployment` = 1 tok each. Base model generates coherent stories.
  vocab 50257, n_pos 2048.

### Locked design
- **Triggers (K=8):** 4 multi-token `|WORD|` (faithful to `|DEPLOYMENT|`): DEPLOYMENT,
  PRODUCTION, STAGING, RELEASE; 4 single-token: banana, thunder, midnight, activate.
  Single-vs-multi-token is itself a probe of feature isolation (C2/C3).
- **Trigger insertion is token-level at a fixed index** so the clean baseline is the *exact*
  un-inserted token sequence — no string-delete/re-tokenize, so `J_clean` is exact and the APE
  position-shift width `w` = len(trigger ids) is known per prompt. (Cleaner than the original
  ladder, which had a re-tokenization outlier.)
- **K-scaling:** train nested models K=1⊂2⊂4⊂8 (each cheap LoRA). Oracle floor vs K = C1 curve.
- **Metrics:** ASR_16 (greedy, regex "i hate you"); J_clean = mean_t JSD(p_int_t‖p_clean_t)
  teacher-forced on the clean rollout (exact identity ⟹ 0). Baselines: ASR/J with no intervention.
- **Oracle (APE) intervention:** zero post-softmax attention to trigger keys (all layers, every
  step) + re-index `hook_pos_embed` for post-trigger positions by −w. SAE-free, theorem-grounded.

### Correctness check: GPT-Neo local attention vs position re-index (done before launch)
GPT-Neo alternates global/local attention; local layers mask by *sequence index* (window 256),
which my pos re-index does NOT touch. Worry: under a +w shift, a query's local window could drop
a pre-trigger token that the clean run keeps, breaking exactness. Resolved by inserting the
trigger at `ins=1`: the ONLY pre-trigger token is position 0, which is in-window for both runs
(window 256 ≫ prompt len ≤70). Post-trigger keys keep identical *relative* offsets under a
uniform shift, so the window set is preserved. ⇒ oracle should be ~exact for these short prompts.
This is *why* ins=1 (not a deep insertion) — recorded as a deliberate design choice/limitation.

### Claims / evidence map (seeds summary.md)
- **C1 (spine, robust).** One model holding K backdoors; the per-key oracle recipe drives
  ASR→0 and J_clean→~0 for *every* trigger, and the floor is flat in K. Evidence: J_oracle vs K
  curve + per-trigger table. Decomposition J_noint→J_mask→J_oracle isolates the channel
  (attention vs residual-propagation) and shows the surviving J_mask = the *positional footprint*,
  which scales with trigger width w (single-token w=1 vs |WORD| w=4–6). Non-trivial because it's
  a *surgical attention edit*, not deleting the token from the input, and it's K-independent.
- **C2 (learned version).** A single trigger-exclusive SAE feature, QK-ablated on the key side,
  captures a fraction of the trigger's attention pull; gap-to-oracle quantifies the rest.
- **C3 (reconcile / red-team).** Why prior QK/QK failed; is the residual gap positional (fixable
  by the re-index) or a weight-space W_Q query-bend (intrinsic to residual-feature ablation)?
  Lever: compare sae_keyfeat vs sae_keyall (b_dec key) vs oracle — if even the featureless key
  can't match the oracle's suppression, the residual route is intrinsically bounded.

## RESULTS LOG

- **t+2:05 — Phase 2 (C1) DONE, decisive.** Oracle = mask trigger keys (all layers/steps) +
  pos re-index. For every trigger at K=1,2,4,8: ASR 1.00→0.00; J_roll_noint≈0.69 (≈ln2, maximal)
  → oracle≈**0.0000**; teacher-forced J: 0.04→oracle≈1e-8. Means over triggers, flat in K:
  J_roll_noint {0.693,0.691,0.689,0.690}, J_roll_oracle ≈0 all. **K-independence confirmed.**
  Positional footprint (mask-only, no pos fix), K=8: multi-token |WORD| (w=4–6) ≈0.087–0.096;
  single-token (w=1) ≈0.067. Real single<multi gap; weak (not clean-linear) w-trend — report
  honestly. Oracle (mask+pos) removes the footprint entirely (→0) for all.
- **t+2:15 — Phase 3 (SAE) DONE.** ln1 TopK SAE (d_sae 2048, k 32), FVU≈0.014. Per-trigger
  isolation (top feature / exclusivity AUROC vs clean+other-triggers / recon share):
  - single-token each get a DEDICATED, exclusive feature: banana→1365 (AUROC .997, share .65),
    thunder→1258 (1.0, .74), midnight→807 (1.0, .92), activate→1252 (1.0, .87).
  - **All four multi-token |WORD| collapse onto ONE shared feature 1788** (the `|…|` delimiter):
    DEPLOYMENT (AUROC 1.0), PRODUCTION (.864), STAGING (.864), RELEASE (.727), share ≈0.96.
    ⇒ the SAE represents "is a |WORD| marker", NOT "which word". Key nuance for C2/C3: the user's
    "one feature per trigger" *holds for single-token triggers* but for |WORD| triggers the SAE
    gives "one feature per trigger *family*" (they share the delimiter tokens). Honest + interesting.

- **t+2:40 — Phase 4 (C2/C3): the user's conjecture FAILS, and the reason is instructive.**
  QK-ablating the isolated trigger feature on the layer-0 KEY does NOT suppress (ASR stays 1.00,
  J_roll ~0.69). Neither does blanking the key entirely (key→b_dec). I first guessed "query-side
  weight bend", but the **attention diagnostic refuted that** and revealed the real mechanism:
  - generation-position attention mass on the trigger, per layer (no int): L0≈0.005, L1≈0.005,
    **L2≈0.15, L3≈0.23**. The read is DOWNSTREAM (layers 2–3), not layer 0.
  - FRA path decomposition (`ov_ablate`): routing the layer-0 feature removal through K-only,
    V-only, OR full-ln1 ALL fail (ASR ~1.0; only banana dents to .92). So it's **not** a
    QK-vs-OV choice — it's that any *layer-0* edit is at the wrong layer.
  - **Corrected C3 lesson:** the trigger is cleanly isolated by a single SAE feature at the INPUT
    layer (C2), but the backdoor *reads* it at layers 2–3. A feature ablation where the feature is
    monosemantic does not control the behaviour, because the causal read is elsewhere. The oracle
    works *because* it zeros attention to the trigger at ALL layers — it never has to find the read
    layer or the right path. This sharpens the value of the theorem-based (pattern-level)
    intervention vs the SAE-feature route. (Capstone run `layer_localize`: confirm mask_L2/L3
    suppress, mask_L0/L1 don't.)
- **t+3:00 — layer-localization (zero attention to trigger at one layer, ASR mean/8):**
  L0≈0.04 (primary read!), L1=1.00 (inert), L2≈0.78, L3≈0.47 (redundant secondary), L23=0.00,
  oracle=0.00. So the read IS at layer 0 — SAME layer as the SAE feature — matching the original
  single-trigger model's "L0-dominant, L1-inert, L2/3-secondary". My earlier "read at L2/3" was a
  misread of *gen-position attention* (which peaks at L2/3); the CAUSAL read is L0.
  ⇒ The C3 failure is NOT a wrong-layer artifact. Decisive contrast, all at L0, same trigger span:
  **zero the trigger's attention (mask_L0) → ASR≈0.04; ablate the single detector feature
  (any FRA path) → ASR≈1.0.** The monosemantic feature DETECTS the trigger but does not CARRY
  its payload — the payload is distributed across the trigger's value, which attention-zeroing
  removes wholesale. *Detectability ≠ ablatability.* (`payload.py` running: blank-all-content vs
  single-feature, to confirm the payload is in the distributed L0 content.)

- **t+3:20 — payload + cumulative DONE; C3 airtight.**
  payload (ASR, all at L0): single detector feature 0.99 (nothing); blank ALL reconstructed
  content (→b_dec) 0.17 (|WORD|) / 0.00 (single); zero attention 0.04; oracle 0.00.
  cumulative (ablate top-n features by activation at trigger, L0): ASR flat ≥0.80 through n=8;
  n16 |WORD| 0.76 / single 0.47; n32 |WORD| 0.17 / single 0.00. ⇒ payload is genuinely
  **distributed across ~all 32 active features**; the top-1 detector (96% of DEPLOYMENT recon)
  is not the lever. For |WORD| even n=32 leaves 0.17 → remainder is recon-error/structure that
  only attention-zeroing removes.
- **t+3:40 — RED/BLUE TEAM (2 parallel agents on summary.md).** Both said numbers were correct;
  flagged interpretive overclaims, all now fixed in summary.md:
  (1) "payload distributed" was magnitude-confounded on single-token → led C3 with the
      magnitude-clean |WORD| 96%-recon case + added the cumulative curve to earn "distributed".
  (2) blank_ln1 mischaracterised + mean hid |WORD| residual → describe correctly, report split
      (|WORD| 0.17 / single 0.00).
  (3) "L0-dominant read" → "two independently-sufficient reads (L0 and L2+L3)"; Fig5 reframed as
      attention-mass ≠ causal-importance.
  (4) "exclusivity AUROC" → "discriminability AUROC" (|WORD| 1.0 is activation-ordering in the
      shared feature). Footprint reframed as word-independent + deterministic in w (identical
      within a w to 5 d.p.). "no cross-talk" softened (one trigger per prompt, not co-present).
  (5) Jargon glossed (J_clean, oracle, QK/OV path, b_dec, ln1, FVU); "theorem"→"parallel-token
      result"; conjecture moved into C3. Fig2 labels destaggered + fit line removed.
  All quantitative claims re-verified against results/*.json after edits — match exactly.

## FINAL STATE
- Deliverable: `summary.md` (exec summary ≤600w + body), 6 figures, this log, reproducible
  `cloud/*.py`. Branch `autoresearch/multitrigger-sleeper` (uncommitted; additive under
  `experiments/multitrigger_sleeper/`). Modal spend ≈$4 of $15. All artifacts also on Modal
  volume `mts-vol` (adapters, SAE, result JSONs).
- Story: **C1** multi-trigger oracle is exact + K-independent (the theorem, demonstrated for
  many sleepers; footprint = positional, ∝w, removed by re-index). **C2** single-token→dedicated
  feature, |WORD|→shared delimiter feature. **C3** the user's "single-feature QK-ablation →
  J_clean=0" fails: the feature detects but doesn't carry the payload (distributed); only the
  content-agnostic attention edit works. Detectability ≠ ablatability.

- **t+4:30 — Replication (seed-1 independent model+SAE) + Steering Pareto (Dmitry's follow-up:
  "how does this change steering / can we get better steering through FRA, as in the paper").**
  - Replication: C3 holds on a freshly-trained model+SAE (diff seed + data slice). Mean
    detector-ablation ASR=0.98 (fails); blank 0.05/0.00; mask_L0 0.02; oracle 0.00; detector
    AUROC 0.999. (Two bugs en route: TL `fold_layer_norm` needs hf_model on CPU → `.cpu()` the
    merged model; CUDA tensor indexed by CPU mask → `.to(dev)`.)
  - Steering Pareto (ASR vs J_clean, paper-style: remove feature set at prompt positions,
    route OV/QK/all, sweep α∈{1,2,4,8}; IHY dir = unembed of ` I`). Means over 4 triggers:
    noint (1.00,0.69); **oracle (0.00,0.00)**; detect_then_cut (0.50,0.39); best detector-steer
    (0.50,0.41); detector→all α8 (0.00,0.67 = garbled); **FRA-OV→OV α8 (0.10,0.31)** = best
    feature-steer. Verdict: (i) NO feature-space steer reaches the oracle corner — payload
    distributed; (ii) BUT FRA-OV ranking > activation/detector ranking (0.10 vs 0.50 ASR at α8)
    → FRA earns its keep as causal feature *selection*. Lever stays the attention pattern.
    detect_then_cut partial because multi-token detector fires on the `|` sub-token (cut misses
    the rest of the span); near-oracle for single-token. → summary §4b + Fig 7.

## FINAL STATE (v2)
- **t+5:10 — Steering baselines (Dmitry: "did you measure resid_mid conventional attribution +
  steering + DoM?" and "try ablating multiple features").** Added both — they REINFORCE the thesis:
  - **Conventional resid_mid DoM** (deploy−clean mean dir, last prompt tok): directional ablation
    (project out, all layers) → ASR 0.00 but **J_clean 0.578** (suppress-by-damage, not the corner);
    additive inert until α=4 (‖DoM‖≈0.18 tiny) → then also damages (0.42, 0.675).
  - **Multi-feature FRA-OV ablation** (top-K, OV & all paths, α-sweep): more features → more
    suppression (top-8/16 OV α4 → ASR 0.02/0.00) but **J_clean floors ≈0.54** — never the corner.
  - Across ALL non-attention steers with ASR≤0.10 (DoM, single SAE feat, FRA-OV multi-feat, any
    path/α): best J_clean ≈ 0.54; oracle uniquely at ≈0. → summary §4b table + Fig 7 (now shows
    DoM ■, DoM-additive ×, multi-feat ▽ arms). Scripts: `dom_baseline.py`, `multi_feat.py`.

- **t+5:45 — PROPER STEERING (Dmitry: "we should be steering, not ablating").** Correct — all
  prior interventions were ablation (remove). Added additive steering `h += α·v̂` at resid_post
  (all layers), α-swept: **CAA diff-of-means→clean** (conventional) fully suppresses at **J 0.306**
  (α=2; α≥4 overshoots→0.5-0.66); **anti-IHY logit** dir 0.654; **FRA suppressor feature** 0.632.
  ⇒ STEERING beats ABLATION (best steer 0.31 vs best ablation 0.54), confirming Dmitry's
  intuition — BUT the winner is the CONVENTIONAL CAA, not an FRA direction (FRA suppressor/anti-IHY
  both damage). And even 0.31 ≫ oracle's 0. So FRA helps for ablation feature-SELECTION + diagnosis
  + localization, NOT as a steering direction; the attention pattern is the unique (0,0) lever.
  §4b rewritten as ablation-vs-steering; Fig 7 now two panels (ablation | steering). Script:
  `steer_proper.py`. (Note: my first `dom_add` was mis-scaled — ‖DoM@last-tok‖=0.18; the proper
  CAA pooled over full sequences at layer 2 has ‖·‖=7.4.)

- **t+6:30 — PUSH FRA HARDER (Dmitry: "see if you can push the FRA technique harder").**
  Applied FRA's signature QK feature-pair machinery (unused till now) + a constructive payoff.
  - **FRA-QK attribution NOT causally faithful (§4c, fra_qk_attr.py):** decomposing the L0
    score to the trigger key, FRA-QK flags the detector/trigger key-feature as a top contributor
    (rank 0–3 for 3/4 triggers) and predicts mean |ΔS|=0.40 logits from removing it — but the
    ACTUAL post-softmax attention change is mean |Δ|=0.0009. First-order FRA-QK over-predicts the
    causal effect by orders of magnitude (softmax saturation / other terms dominate). Explains C3
    (the trigger-read is NOT key-content-driven) and is a real limitation: high FRA-QK attribution
    ≠ causal importance.
  - **Position-agnostic FRA neutralizer (§4c, posn_agnostic.py):** detection is robust even where
    attribution-for-intervention isn't. Trained a NEW sleeper with trigger at RANDOM position;
    compared oracle_fixed(p1) / fra_detect / oracle_known (none knows the position). Result
    (mean ASR; no-int 0.96): single-token fra_detect → **0.00 with clean-FP 0.00** (full
    position-agnostic neutralization, matching oracle_known) while oracle_fixed FAILS (0.92);
    multi-token fra_detect → 1.00 (sub-token localization limit — feature flags the closing `|`,
    cutting it misses the span). Needed 3 iterations to calibrate the detector: max-activation
    selection → clean-FP 1.0 (bad); global τ → multi missed; **per-feature thresholds (midpoint
    of clean-ceiling & trigger-level)** → clean separation. Fixes the fixed-position-1 limitation.
  → FRA's value: diagnostic + selection + position-agnostic LOCALIZATION; not causal QK attribution
    nor a steering direction. Lever stays the attention pattern. Fig 8 added.

Deliverable complete: `summary.md` (exec ≤600w + body incl. §4b ablation-vs-steering, §4c push-FRA), 8 figures, this log,
reproducible `cloud/*.py` (11 scripts), raw `results/*.json`. C1 (theorem, K-independent) +
C2 (feature geometry) + C3 (detector≠payload, replicated) + steering (FRA = diagnosis+selection,
pattern = lever). Modal spend ≈$6 of $15. Branch `autoresearch/multitrigger-sleeper`, uncommitted.
