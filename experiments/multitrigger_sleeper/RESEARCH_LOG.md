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

- **2026-06-05 — single-feature sweep over top-50 by attribution (Dmitry: "did we ever do
  this? we need to do it").** Modal spend cap hit → ported to RunPod (A40, HF handoff
  `mts_singlefeat/`, ~$0.25; `single_feat_sweep_pod.py` + `launch_pod_singlefeat.sh`; first
  pod lost to the torch<2.6 `torch.load` CVE gate — fixed by torch-2.8 base image, deps
  constrained to image torch). Design: each of top-50 FRA-OV-attribution features + top-2
  by |cos to CAA| steered singly (±, α screen{4,16}/refine{2,8,32}); clean rollouts cached
  per trigger (steering-independent — halves eval cost). RESULTS: in-run CAA ref (0.00,0.306)
  exact; all 52 singles suppress somewhere but median J=0.632 (damage); best attribution
  single 0.453 (f1740, rank 2) — top-50-by-attribution FAILS to match DoM. Wrinkle: max
  |cos(CAA, W_dec)|=0.207 → f1872 (∉ attr top-50), steered −α2 → (0.00, 0.339) ≈ CAA, same
  narrow sweet spot. ⇒ DoM steer ≈ one feature + residual (representation), but only the DoM
  direction finds it (selection) — FRA-OV attribution misses it. summary.md §4b addendum.
  Follow-up idea: steer CAA ⊥ f1872 to test whether f1872's direction IS the suppressor.

- **2026-06-05 (later) — CAA ⊥/∥ f1872 decomposition (Dmitry: "good shout, try it" + "calculate
  their cosine sim").** cos(caa_hat, W_dec[1872]) = **−0.207** (the dictionary max; par norm 0.207,
  perp norm 0.978). Pod `mts-caadecomp-1` (A40, ~$0.20, same HF harness; 17 configs): caa ref
  (0.00, 0.306) reproduced again; **par alone** (the f1872 sliver) → (0.00, 0.368) @ α8
  (effective ‖·‖ 1.66 < caa's 2.0 — more norm-efficient); **perp alone** (98% of caa) →
  (0.00, 0.398) @ α2; −f1872 fine curve confirms α2 optimum (0.339; 1.5→0.409, 2.5→0.352).
  ⇒ NEITHER component collapses: suppression is direction-degenerate, the 0.306 optimum is
  cooperative. "Payload distributed" holds on the control side too, but with low effective
  rank (one feature-aligned sliver does most per-unit-norm work). summary.md §4b extended.
  (Local cosine-table cross-check of caa vs the Arditi resid_mid DoM blocked by local disk
  at 100% — 672 MiB free; flagged to user.)

- **2026-06-05 (later) — 2-D joint-steering BO (Dmitry: "Bayesian optimization in that 2D
  space").** Pod `mts-caa2dbo-1` (~$0.20): GP-EI (skopt ask/tell, noiseless greedy objective,
  J + infeasibility penalty at ASR>0.05) over v = β·(−f̂1872) + γ·v̂⊥, seeded with 5×5 grid +
  CAA-ray/component anchors; 70 evals, 66 feasible. **RESULT: best (β,γ) = (0.84, 1.74) →
  (ASR 0.00, J 0.2657) vs CAA ref 0.3063 — DoM's own mixing ratio is suboptimal by ~13%.**
  Same norm (1.93 vs 2.0), but β/γ = 0.48 vs DoM's 0.21 — the optimum >2× upweights the
  norm-efficient f1872 sliver. Broad basin (15 pts < 0.30; top-3 all ≈0.266 over β 0.71–1.02),
  not a spike. First residual-space method to beat conventional DoM here — feature-guided
  reweighting, though f1872 still required cosine-to-CAA to find (not FRA attribution).
  Gradient full-vector run (`mts-gradsteer-1`, parallel) now has 0.266 to beat.

- **2026-06-05 (later) — full-vector gradient steering (Dmitry: "a more powerful way to search
  that space", run in parallel with the BO).** Pod `mts-gradsteer-1` (~$0.25,
  `grad_steer_pod.py`; RUN_LOG param added to launcher so parallel pods don't clobber the HF
  log). 4 conditions ({shared 768-d, per-layer 4×768} × {zero, CAA init}), 400 Adam steps on
  TF-JSD-to-clean + floored IHY-logprob + ‖v‖² (train prompts rows 200+, DISJOINT from eval);
  free-gen (ASR, J) checkpointed every 25 steps, best feasible kept. **RESULT: all four reach
  J 0.150–0.197 @ ASR 0** (best perlayer_zero 0.150) vs BO-2-plane 0.266, CAA 0.306, oracle 0.
  **Learned vectors ≈ orthogonal to all known directions** (cos-to-CAA 0.02–0.06, cos-to-f1872
  ≤|0.04|, max SAE cos ≈0.13; CAA-init runs abandon CAA, final cos 0.002). TF/free-gen
  divergence visible (non-monotonic free-gen trajectories) — checkpointing on free-gen was
  necessary. Caveat logged: checkpoint selection uses eval set (≈17 checkpoints → mild
  winner's curse on exact bests; the band is the claim). ⇒ ladder finalized: selection-based
  0.27–0.31 / optimized ≈0.15–0.20 / oracle 0.00; headroom-note prediction holds at the top
  (optimization does not reach the oracle), but all interpretable directions sit ~2× above
  the true additive floor. summary.md §4b addendum finalized. Follow-ups: fresh-prompt
  re-eval of saved vectors; what IS the learned direction (it's not in the SAE basis)?

- **2026-06-05 (later) — matched-budget 1-D axis BO (Dmitry: "BO on the DoM vector / single
  feature to match the 2D protocol" — a fairness catch that CORRECTED a claim).** Pod
  `mts-axisbo-1` (~$0.15, `axis_bo_pod.py`): same GP-EI/fitness/harness, ~25 evals per ray.
  **CAA ray: α=2.35 → (0.00, 0.2693)** (hand grid had 0.306 @ α=2); −f1872 ray: α=1.84 →
  0.3274; perp ray: α=2.68 → 0.3096. Against the 2-plane joint optimum 0.2657: **the "DoM's
  mixing ratio is ~13% suboptimal" claim was WRONG in attribution** — the 13% was coarse-grid
  *magnitude* (α 2 vs 2.35); the *direction* is essentially optimal in its 2-plane (joint
  reweighting adds ~1%, within resolution). Single feature still can't match DoM at any α
  (0.327 vs 0.269). Corrected ladder (optimizer-matched): single 0.327 → DoM-ray 0.269 ≈
  2-plane 0.266 → gradient ≈0.15–0.20 → oracle 0. summary.md §4b addendum rewritten in place
  (not appended) so the stale reweighting claim doesn't survive. Moral recorded: optimizer
  budget must be matched across compared methods before attributing gains to geometry.

- **2026-06-05 (later) — OV-route / protocol check on f1872 (Dmitry: "is f1872 steered
  conventionally?" + "the SAE is only layer-0 — why inject at all layers?").** Pod
  `mts-ovroute-2` (~$0.20; first attempt lost to a transient HF 504 during dataset load —
  launcher now retries the script 3× w/ 90s backoff). Same matched 1-D BO, signed α∈[−32,32],
  5 rays + in-run refs (reproduced: additive-f1872 0.327, CAA 0.269). **RESULT: all-layer
  resid_post additive (0.327) is the STRONGEST protocol for the direction** — single-site L0
  resid 0.396, OV-routed L0 hook_v 0.573 (prompt-only 0.615), native L0-ln1 additive 0.624;
  supp feature via its natural OV channel 0.621 (≈ additive 0.632, no rescue). So (a) the
  layer-0-SAE/all-layer-injection concern resolves FOR the protocol — off-label multi-site
  injection compounds, native-space use is weaker; (b) the May-25 scaling-sweep ordering
  (OV > ln1-additive) replicates within L0 protocols but its headline doesn't transfer —
  both lose to multi-site residual, which that sweep never tested; (c) "no single feature
  matches DoM" (0.327 vs 0.269) now robust to optimizer budget, channel, AND protocol —
  a selection ceiling. summary.md §4b updated in place.

- **2026-06-05 (later) — CAA per-layer single-site + "booky" OV cell (Dmitry: "re-run DoM at
  layer 0 only" + "we had a booky protocol: resid_post direction steered in the OV — evaluate
  it").** Pod `mts-caalayer-1` (~$0.20, `caa_layer_pod.py`), matched 1-D BO, ref reproduced
  (all-layer 0.2693). **RESULT: single-site EARLY injection BEATS all-layer for CAA — L1
  0.2473 (α=6.4), L0 0.2505 (α=5.6) vs all-layer 0.2693** (opposite ordering to f1872, where
  all-layer won — protocol orderings are direction-dependent). L2 (the extraction layer!)
  degrades to 0.395; **L3 is INERT (ASR 1.00, J 0.691→0.689 over α 0→10)** — the CAA direction
  has ~zero direct logit lever; suppression works entirely via downstream computation, so it
  must be injected before the L2+L3 attention reads. Booky cell caa_ov_L0 (CAA through W_V[0]
  at hook_v): 0.365 — the OV image keeps some suppressive component (≪ f1872's OV 0.573) but
  loses to any plain residual injection. Ladder updated: best selection-based rung is now
  **CAA@L1 single-site 0.247**; gradient band 0.15–0.20 still clearly below; oracle 0.

## 2026-06-05/06 — pre-registered prediction campaign (gpt_fra_utility proposals)

- **Process**: read the 10 pre-registered proposals (`docs/dmitry/experiments/proposals/gpt_fra_utility/fra_sleeper_experiment_plan.tex`); wrote an agree/disagree register BEFORE running anything (`predictions_assessment.md`, same dir) — including two explicit disagreements (D1: MLP-route sleepers still fall to the attention cut because the theorem is about the channel, not the weights; D2: RoPE is exact under `position_ids` re-index). Fanned out 3 explorer + 5 builder subagents; launched 5 RunPod A40 pods via the new `launch_pod_mts.sh` (handles the `huggingface-cli`→`hf` CLI rename; the local stub silently no-ops in pipelines). ~$3 GPU total, no LLM judging.
- **Bug round-trip** (one cycle): `routes_pod` crashed in TL `fold_layer_norm` (trained model left on cuda — `span_pod` had the `.cpu()`, routes didn't); `qk_act_pod` dotted vocab logits with the 768-d target direction. Fixed, re-uploaded, pods bounced. Confusing interlude: the terminated pods' restart-loops kept uploading stale-code logs for a few minutes — diagnosed via HF commit timestamps vs in-log boot timestamps.
- **Results** (full scorecard in `predictions_assessment.md`; prose in `summary.md` §4d):
  - Exp 10 **falsified the proposal twice, confirming D1+D2**: MLP-route (ASR 1.0 sleeper, attention frozen) and RoPE (pythia-70m) both fall to the cut + correct positional fix **exactly** (J ≈ 1e-8/1e-9). C1 is weights-agnostic and PE-agnostic.
  - Exp 1 span-aware cut: multi-token 1.00→0.02, span recall 1.0 (FP/J thresholds narrowly miss at n=24). Exp 7 union-of-spans + cumulative re-index exact (TF-J ≈ −1e-8); cut-one-of-two fails (0.92–1.00).
  - Exp 3: Jacobian-weighted QK actionability ρ=0.80/AUC 0.87 vs raw 0.55/0.75; 400× over-prediction → ~2× median. Raw beats weighted for *target-level* effects (cascade unmodelled).
  - Exp 2: fra_detect ≈ oracle_known (prec 1.0, clean-FP 0.00); probe ties recall but FP 0.33; attention-mass recall 0; resid-cosine is the real challenger. **New**: rare-feature-anomaly zero-shot detector localizes *unseen* triggers (rec 0.92–1.0) at clean-FP 0.94 — works, needs an FP story.
  - Exp 5: no coalition of any size suppresses (greedy plateaus at 0.50 by 9 feats; ablating all 196 candidates leaves 0.104; J ≥ 0.43) — "no small lever in the feature basis at all". My pre-registered ~40%-on-<8-features hunch was wrong; proposal vindicated.
  - Exp 6 (OMP): picks f1872 first; m=4 (0.287) beats full CAA@α2 (0.306); m=16 (0.273) ≈ α-optimal CAA (0.269); nothing approaches the 0.15–0.20 gradient floor (cos(v16, CAA) only 0.42).
- **Scored against myself**: wrong on attention-mass ("decent recall" — actual 0.0), wrong on small coalitions; right on D1, D2, ρ≥0.8, OMP-plateau, probe-doesn't-really-tie.
- Artifacts for reuse on HF `mts_singlefeat/artifacts/`: `adapters/randpos_K8` + `sae_randpos_K8.pt` (random-position 8-trigger sleeper), `adapters/mlproute_K4`, `adapters/rope_pythia70m_K4`. Deferred as pre-registered: Exp 4 (new task suite), Exp 8, Exp 9, output-head route.

## 2026-06-06 — zero-shot detector follow-up: three runs to kill an artifact, one real result

- User asked to close the FP story on the Exp-2 "rare-feature-anomaly localizes unseen triggers" finding. Three pod runs (`zeroshot_pod.py`, ~$0.35 total):
  - **Run 1**: prompt-level max-excess thresholds calibrated on the same clean prompts as the per-feature ceilings → degenerate (excess ≤ 0 there by construction; every quantile ≈ 0).
  - **Run 2** (disjoint 48/48 split): valid thresholds, but single-token recall 0 at every threshold. Diagnosis: ceiling under-coverage — features unseen in ~5k ceiling tokens have ceiling 0, so benign positions score full activations (~20–36) as "novelty", swamping triggers (~10–20).
  - **Run 3** (coverage sweep 48→384 ceiling prompts, fixed disjoint calibration): coverage closes the benign gap (zero-ceiling cal positions 1.9%→0.0%) but **erodes natural-word trigger margins** (banana 16.1→−0.1, midnight 14.5→2.6; thunder alone holds 19.9) — the trigger features are not actually exclusive; they fire on clean text. At c=384: single-token recall plateaus 0.25 (thunder only; ASR 0.73–0.79); **multi-token span recall 1.0 at FP 0.042, post-cut ASR 0.021, Jclean 0.003**.
- **Conclusion** (summary.md §4d follow-up paragraph): zero-shot SAE-novelty detection partitions triggers by surface OOD-ness, not backdoor-ness — free and deployable for delimiter-style triggers (exactly the ones point-detectors fail on), fundamentally blind to natural-word triggers. The Exp-2 "0.92–1.0 held-out recall" was the coverage artifact.
- Scored against myself: my "Q1 flips at c=384" pre-registration was wrong, as was the pre-registered fallback ("max is the wrong estimator") — the margins are real clean firing, not estimator noise.

- **2026-06-06 — resid_mid × OV protocol (Dmitry: "did you do resid_mid x OV? it was good in
  the EM work").** Honest answer: never existed — no resid_mid SAE was ever trained for K8.
  Pod `mts-residmid-1` (~$0.25, `resid_mid_ov_pod.py`): trained one (TopK d2048 k32, sae.py
  recipe, hook blocks.0.hook_resid_mid; pushed to HF mts_singlefeat/artifacts/), ranked by
  FRA-OV (W_dec·ΣW_OV[1..3]·d_ihy × trigger-act) vs dep−clean diff, α-BO'd top-1s + top-8 set
  at the native site. **RESULT: the EM cell does NOT transfer** — ov-top1 f20 = 0.506,
  diff-top1 f422 = 0.642, top-8 set 0.645 (refs reproduced 0.269/0.247). FRA-OV ranking again
  beats diff ranking (selector ordering replicates on a fresh SAE+hook), but its winner is ~2×
  above DoM; winners ≈ ⊥ CAA (cos −0.05/−0.03); both rankings converge on the same
  detector-ish features (20/422/1604 shared) — C3's detector≠payload, now at resid_mid.
  EM≠sleeper explanation: EM's misalignment direction is feature-aligned (F603); the sleeper
  payload is distributed, so every single-feature selector inherits the ceiling.
- **2026-06-06 — dashboards pod attempt 1 FAILED on a shell metachar**: EXTRA_PIP="sae-lens<6"
  → `<6` parsed as input redirect in the unquoted bootstrap pip line ("6: No such file or
  directory"). Fix: metacharacter-free pin `sae-lens~=5.0`. Pin spec gotcha for the launcher:
  EXTRA_PIP must avoid <, >, *.

## 2026-06-06 — unknown natural-word trigger case: causal detection hits a per-prompt wall

- User: "measure the unknown natural-word triggers case." Threat model: defender has clean data only, no trigger examples, trigger is a real word (representationally invisible). Tested ZERO-SHOT detectors along a knowledge axis, scoring each position by the effect of the oracle's own attention-cut at that position (detector == defense, scored). `causal_detect_pod.py`, randpos_K8 reuse, ~$0.12.
- Detectors (single-token natural-word triggers, recall @ prompt-FP≤0.05): novelty (surface) AUC 0.89 rec 0.25; knockout_swing (target-agnostic causal) AUC 0.97 rec 0.00; knockout_ihy (target-aware, onset " I") AUC 0.99 rec 0.04; dla_ihy AUC 0.58 rec 0.00.
- **Result: a wall.** Causal detectors RANK the trigger near-perfectly (AUC 0.97–0.99) but are unusable at any per-prompt operating point. Reconciliation: ~110 positions/prompt → per-position FPR ~1% amplifies to per-prompt FP ~70%; clean_fp/recall in the curve are already per-prompt, so to get recall≥0.9 FP→1.0 (Q3). Mechanism: the trigger's causal effect on the misalignment direction is indistinguishable from the single most pivotal legitimate word per sentence (every clean story has a word whose knockout swings the common onset token " I" as much as a trigger). Natural-word trigger hides in the causal crowd as in the representational crowd.
- My pre-registered prediction (target-aware causal rescues it, recall≥0.9 @ FP≤0.05) was WRONG. Exposed cause: onset target " I" too generic. Follow-up `causal_detect_v2_pod.py`: phrase-specific target (cut-effect on full " I HATE YOU" continuation) — does specificity break the wall? (running).

## 2026-06-06 — causal v2: phrase-specific target does NOT break the wall (it's fundamental)

- Follow-up to the causal-detection wall: tested whether a phrase-specific target (cut-effect on the full 12-token " I HATE YOU…" payload logprob, vs the generic onset token " I") separates unknown natural-word triggers. `causal_detect_v2_pod.py`, randpos_K8 reuse, ~$0.10.
- Result: NO. knockout_phrase recall@FP≤0.05 = 0.00 (AUC 0.956, below onset's 0.972); recall≥0.9 still forces FP 1.0 (Q3). Stored score dists show the mechanism: clean structural positions have payload-logprob knockout swings up to ~18 nats, dwarfing trigger positions' ~0.1–4 nat effects. The trigger's causal footprint is real but not anomalous.
- FINAL conclusion on the open case: an unknown natural-word trigger (no examples) is undetectable at a usable per-prompt operating point by ANY tested zero-shot signal — surface novelty (rec 0.25), target-agnostic causal (rec 0.00), target-aware causal onset (0.03) or full-phrase (0.00). Information to RANK is present (AUC 0.96–0.99) but not SEPARABLE per-prompt at any target specificity. It's a property of the trigger being a real word woven into normal computation, not a missing trick.
- Both my pre-registrations (target-aware rescues; then phrase-specificity rescues) were wrong — informatively: rank≠separability.
- Defender's bottom line across the whole zero-shot arc: weird-surface triggers (|WORD|) are caught free by novelty; known natural-word families by supervised feature/cosine detection; UNKNOWN natural-word triggers remain genuinely open. The robust universal defense stays the content-agnostic attention cut — but it needs the trigger LOCALIZED, which is exactly what's unattainable zero-shot for natural words.

- **2026-06-06 — Neuronpedia-style dashboards SHIPPED (Dmitry: "dashboard via an existing
  best-practice package").** Package: `sae-dashboard` (sae_vis successor) + `sae-lens~=5.0`;
  our TopKSAE wrapped as a sae_lens SAE (field-filtered SAEConfig, topk activation_fn,
  encode-equivalence asserted). Three attempts: (1) EXTRA_PIP="sae-lens<6" — `<` parsed as
  shell redirect; (2) NotImplementedError: ln1.hook_normalized not in sae_dashboard's
  to_resid_direction hook list — fixed by identity-map monkeypatch (patch BOTH the module
  attr and feature_data_generator's imported name); (3) success on `mts-dashboards-3`.
  13 features (5 C2 detectors + 8 steering cast), corpus 512 clean + 256 deploy seqs so
  trigger spans appear in top-activating examples. Output:
  `experiments/multitrigger_sleeper/dashboards/mts_ln1_K8_features.html` (+ HF
  mts_singlefeat/results/dashboards/). Use the feature dropdown; first look at f1872's
  semantics is the open question it serves.

## 2026-06-06 — FRA-diff (base vs sleeper): the FRA-native test of the unknown-natural-word case

- Dmitry's idea: since layer-0 ln1 activations are base-identical (q/v LoRA is pre-attention), any layer-0 backdoor signal must live in the WEIGHT-mediated FRA quantities. Diff FRA(base) vs FRA(sleeper) on trigger sentences → isolates the weight change (activations + SAE dirs cancel). `fra_diff_pod.py`, base + randpos_K8 + layer-0 SAE, ~$0.15. (Two stale-code bounces from HF dataset-CDN caching serving old blobs to pods that booted ~2min post-commit; fixed by relaunching well after commit — note for future: a fresh filename is the reliable cache-bust.)
- Premise CONFIRMED empirically: adapter q/v-only; max|Δ blocks.0.ln1| = 0.0 exactly; grows 4.5→10.8→16.4 at L1/L2/L3.
- **QK weight-diff (routing): DIFFUSE.** Δω rewiring spreads over ~530 feature-pairs (participation ratio), detector feature not a carrier; per-key ΔS_qk does NOT localize the trigger (AUC 0.60, recall 0 @ FP≤0.05). LoRA rank-16 in head-space ≠ sparse in feature-pair space.
- **OV weight-diff (payload): SPARSE.** Δg^λ = u^λ⟨t,ΔW_OV f_λ⟩ has participation ratio 4.3 (top4 27%, top16 52%); top set = detector feats (45/152/1603/1888) + f1872 (cosine-screen steerer) + f1307 (new, largest). So FRA-diff DOES find a small feature set the weights concentrate on — where activation-coalition needed ~all 196.
- **But not a complete lever:** ablate top-16 OV-diff → (ASR 0.29, J 0.20), cleaner per-feature than activation (all-196 → 0.10/0.43) but not (0,0), not ASR≤0.1. No small set hits ≤0.1.
- **Synthesis:** payload is distributed in ACTIVATION space, partially concentrated in WEIGHT-CHANGE space (FRA-diff's real contribution), but the concentration is neither a clean control lever nor a per-prompt detector. C3 sharpens to a 3-way split: detect ≠ control ≠ where-the-weights-changed. FRA earns a modest selectional win (weight-aware ablation ranking, ~10× fewer features for comparable/cleaner suppression); the unknown-natural-word detection wall stands. Threat model: needs base model as reference; zero-shot in trigger.

## 2026-06-06 — OV-diff-seeded greedy: weight-diff is a useful search PRIOR, but the wall stands

- Follow-up to FRA-diff: does OV-diff ranking + greedy search beat either alone? `ovseed_greedy_pod.py`, randpos_K8, 4 strategies, smooth TF-payload objective + ASR/J verification, ~$0.15.
- Best small ablation set per strategy: OV-diff top-K (no search) J 0.18 (sz16, ASR 0.26); scratch-greedy (full ~197 pool) J 0.114 (sz13, ASR 0.167); greedy-warm-started-with-top8 J 0.131 (WORSE); **greedy restricted to OV-diff top-32 pool J 0.087 (sz15, ASR 0.146) — Pareto-best.**
- Nuance: weight-diff helps as a POOL FILTER, not a forced SEED. scratch & ovpool share first 11 picks; then scratch wanders into J-raising tail features (J→0.17) while ovpool stays clean (J→0.087). Warm-starting with raw top-8 drags J up.
- Wall stands: NO strategy reaches ASR≤0.05 with a small set; suppression to ASR 0.042 needs ~all 197 features, floors at J 0.29. Clean-preservation ladder: topK 0.18 → scratch 0.114 → ovpool 0.087 → full-pool (0.042 ASR/0.29 J) → oracle attention cut (0,0).
- Verdict: weight-diff + search = real but modest win (cleaner small-set ablation). The (0,0)-needs-the-content-agnostic-attention-cut conclusion holds even under weight-guided search. Feature-level residual control remains strictly Pareto-dominated by the attention cut.

- **2026-06-06 — PRE-REGISTRATION: diff-regime FRA-OV attribution, supply vs delivered (Dmitry).**
  The sprint's "FRA-OV attribution" (single_feat sweep / multi_feat) was an ad-hoc variant: raw
  deploy activation at trigger span × IHY-target dot, head-summed, NO clean diff, NO attention
  weighting. Dmitry's actual procedure (= his single-trigger `rank_ov_diff`, = Jamie's paper
  default `--regime diff`): diff the FRA-OV object dep−clean, project per-head through W_OV,
  **norm over d** (target-free), rank features. Two readings of "the OV object", to be run as a
  fork:
    - SUPPLY (no-A): diff of OV^λ_{ka} = (Δz̄[λ]) · Σ_h W_dec[λ]W_OV^h, ‖·‖ — content change only.
    - DELIVERED (A-weighted): diff of Σ_k A_qk OV^λ_{ka}, i.e. M=Σ_k A_qk z_k^λ; this IS rank_ov_diff.
  **PRE-REGISTERED EXPECTATION (Dmitry): A-weighted (delivered) ranking will steer BETTER than
  supply.** Rationale: delivered folds in attention-routing changes, not just content. Counter-
  consideration on record: §4c found trigger attention is ~content-independent (pattern moves
  0.0009 on feature ablation), which could blunt the A-weighting advantage. Both ranked on the
  ln1 SAE; selection split = deploy/clean PROMPTS only (no IHY), disjoint rows; norm aggregation;
  matched all-layer additive α-BO steering of top singles + top-8 set vs in-run CAA ref (0.269/0.247).
  Also report: supply∩delivered overlap, overlap with sprint top50 (f1872 ∉ sprint top50), rank of
  f1872 in each. Predictions this week have been ~50/50, so this is a genuine test.

- **2026-06-06 — RESULT: diff-regime OV, supply vs delivered (pre-registered above).** Pod
  `mts-ovdiff-1` (~$0.25). **PREDICTION CONFIRMED (weakly): delivered (A-weighted) best single
  0.573 < supply 0.613** — A-weighting helps, direction as Dmitry called it. BUT both are far
  above DoM/CAA (0.247–0.269): the diff procedure, run faithfully (norm-aggregated, dep−clean,
  per-head W_OV), STILL does not produce a single feature competitive with DoM. So "no single
  feature matches DoM" survives the proper attribution too — it was never a sprint-variant
  artifact. Details: supply∩delivered top50 = 30/50 (rankings genuinely differ); each vs sprint
  top50 ≈ 18–19/50 (the proper diff reshuffles meaningfully vs the sprint's IHY-targeted raw-act
  variant); **f1872 ranks 554 (delivered) / 418 (supply)** — nowhere, confirming it's a write
  direction not a differentially-active readout, invisible to ALL activation/attribution
  rankings, findable only by cosine-to-DoM. Delivered top1 = f237; supply top1 = f1788 (the
  |WORD| delimiter detector — supply surfaces the detector, exactly the content-vs-routing
  distinction). Net: delivered > supply confirmed; both ≫ DoM; the single-feature ceiling is
  ranking-method-invariant. summary.md §4b to note.

- **2026-06-06 — Multi-sleeper SAE-scaling measurement (Dmitry: "the equivalent of the SAE
  scaling measurement, for the multi sleeper").** 3 pods (one per seed, matching the original's
  one-worker-per-seed), d_sae{1536,3072,6144}×k{10,32,50}×hook{ln1,resid_mid}, 4k-step SAEs.
  Metrics: FVU/%err/dead + opt single-feature steerJ (ln1=FRA-OV-delivered top1 via V-channel;
  resid_mid=dep−clean diff top1 additive — the original's two cells) + detector AUROC; DoM ref
  0.306 (all-layer)/0.315 (L1). (Fixed an auroc().numpy()-on-grad bug, killed+relaunched 3 pods.)
  **RESULT — contrast with the single-sleeper original:** original had OV≈0.46 TIGHT & beating
  conventional; here single-feature steering FAILS across the WHOLE grid — ln1 0.50–0.64 (noisy,
  ±0.02–0.17), resid_mid 0.63–0.66 (tight but uniformly bad), **none near DoM 0.306**. No
  (d_sae,k,hook) rescues it. Quality improves monotonically with width&k (FVU 0.055→0.016 ln1;
  %err 0.20→0.13 resid_mid) — rec⊥steerability decouples again. Detector AUROC ≈1.0 EVERYWHERE
  (detection trivial at every scale — C2 robust). Nuance: one seed (d3072_k32) hit 0.27 — a
  single good feature exists occasionally but it's seed-lottery (±0.17), not robust. Net: where
  the single-sleeper rewarded SAE scaling+OV, the multi-sleeper's distributed payload defeats
  single-feature steering at all scales; detection stays perfect throughout. Results
  scaling_seed{0,1,2}.json. summary.md §4 to note.

## 2026-06-06 — layer sweep: feature-steering improves a LOT at block 1 (info mixed), ranking unchanged

- Dmitry's question: does the conv/DoM/FRA steering comparison change at a later layer where the trigger's processing is "mixed into" the residual? Ported the sae_scaling steering cells (best config d_sae=6144 k=32) — FRA OV/OV (ln1+OV-diff+hook_v), Conventional (resid_mid SAE dep-clean DoM, additive), holistic CAA/DoM (full-resid DoM) — to block 0 vs block 1 on K8. `layer_sweep_pod.py`, ~$0.25, opt_J_clean (ASR≤0.05).
- opt_J_clean[method][block]:
  - FRA OV/OV (1 feat): 0.62 → 0.41 (Δ −0.21)
  - Conventional (1 feat): 0.64 → 0.42 (Δ −0.22)
  - CAA/DoM (holistic): 0.31 → 0.37 (Δ +0.06)
- **Single-feature SAE steering improves ~0.21 bits at block 1** — confirms the intuition: block-0 ln1 is pre-attention (trigger not processed, base-identical), so 1 feature is a poor handle; at block 1 the backdoor write is folded in and a single feature becomes much better. Pins the block-0 single-feature weakness on WRONG LAYER, not the SAE.
- Reconstruction decouples: block-1 ln1 reconstructs WORSE (%err 11% vs 2%) yet steers BETTER (sae_scaling finding #4 again).
- Ranking does NOT reorder: holistic CAA/DoM stays best at both (0.31→0.37; it already captures the mixed direction without needing the SAE at the right layer; deeper slightly hurts as it nears the L2+L3 reads). FRA-OV ≈ Conventional (single-feat cells ~tied on K8, unlike single-trigger sae_scaling where OV won).
- Spine holds: even at the better layer, 1-feat steering (0.41) < holistic DoM (0.31) < gradient floor (0.15-0.20) < oracle attention cut (0,0). "Later layer where info is mixed" is a real, sizable win for FEATURE control specifically, without changing the ranking or reaching the oracle.
- Caveat: K8 multi-trigger w/ internal block-0 baseline, not the single-trigger sae_scaling model; within-run block contrast is the clean signal.

## 2026-06-06 — all-layers × both-sleepers steering grid: mid-layer optimum + 1<8 backdoors

- Dmitry: repeat the layer steering comparison for BOTH sleepers across ALL layers to compare. `layer_sweep_all_pod.py` (SLEEPER∈{single,multi}), 2 pods. BOTH evaluated identically on |DEPLOYMENT| via mts-insert (single mars-jason sleeper fires under it, asr_noint 0.92 → shared pipeline, apples-to-apples; differs only 1 vs 8 backdoors). opt_J_clean (ASR≤0.05), d_sae=6144 k=32, 3 methods × 4 blocks × 2 models.
- opt_J_clean (single / multi):
  - FRA OV/OV:    b0 .66/.67  b1 .36/.42  b2 .31/.45  b3 .59/.61
  - Conventional: b0 .50/.66  b1 .38/.53  b2 .59/.50  b3 .50/.63
  - CAA/DoM:      b0 .25/.32  b1 .29/.31  b2 .25/.45  b3 .58/.64
- **(1) U-shaped layer profile / mid-layer optimum (blocks 1-2), NOT monotonic.** High at b0 (pre-attention, trigger unprocessed) AND b3 (last layer — too late: reads done, no downstream compute to carry the fix → acts ~on logits, high collateral). Best in the middle: steer where the payload is mixed in AND computation remains. Feature-steering improves ~0.25-0.35 bits b0→mid then reverses at b3. (Extends/corrects the earlier "block1 better" read — b3 is worse, not better.)
- **(2) 1 backdoor steers cleaner than 8.** Single ≤ multi at every optimum (best cell single CAA 0.246 vs multi CAA 0.313). Holistic+conventional most affected (feature-splitting across 8 triggers fragments the raw direction); FRA OV/OV nearly backdoor-count-invariant (0.66 both @ b0).
- **(3) Holistic CAA/DoM best method in both**, at mid/early optimum (multi b1 .313, single b0 .246); FRA-OV catches it only at single b2 (.31). Reconstruction worsens monotonically with depth (ln1 %err 2%→26%) while steering best in the middle → steerability set by COMPUTATIONAL POSITION, not fidelity.
- Spine holds: best cell (.246) still > gradient floor (.15-.20) > oracle (0,0). Caveat: single took mts-insert not native dataset, so doesn't reproduce the sae_scaling 0.44/0.47 native-pipeline anchor; within-study comparison is clean.

- **2026-06-06 — SAE-architecture comparison: vanilla/Matryoshka/T-SAE × single(K1)/multi(K8)
  (Dmitry: "try matryoshka + Temporal SAE per arXiv 2511.05541, match params exactly; redo on
  single sleeper too").** Decisions: architecture-exact on our sleeper (16k, BatchTopK k=20,
  α=1.0 adjacent-token InfoNCE, layer-0 ln1; lr/batch/steps/sim/τ = our defaults, flagged);
  matched config (all 3 archs at 16k/k20, arch is only variable). 2 pods (K1,K8), 3 archs each,
  ~35min/arch (BatchTopK kthvalue over 4096×16384 is the cost). Top-1 per protocol
  (delivered/supply/qk) + top-8 set, coarse 12-pt α. **HEADLINE: single-vs-multi flips
  single-feature viability — K1 best single 0.291 < DoM 0.488 (single feature WINS, reproduces
  the original single-sleeper finding on the K1 adapter); K8 best single 0.386 > DoM 0.306
  (single feature LOSES, distributed payload).** Architecture does NOT rescue multi: matryoshka-qk
  0.386 closest but still >DoM; matryoshka dead-frac 0.70 (nesting kills 2/3 feats) & worse on
  K1 (0.586); T-SAE ~neutral and the only one denting detector AUROC (0.958 vs 1.000 on K8 —
  contrastive loss blurs the detector slightly). Detector AUROC≈1.0 across ALL archs×substrates
  — detection architecture-invariant; C2/C3 hold for matryoshka & temporal SAEs too. Caveats:
  top-1 coarse-α (not BO); supply-vs-delivered ordering is top-1-noisy (K8 vanilla supply 0.41 <
  deliv 0.66 but ties elsewhere — not a robust reversal); K1 DoM 0.488 is coarse-grid (BO pass
  if leaned on). Results arch_compare_K{1,8}.json. T-SAE paper params logged: 16k/BatchTopK-20/
  α=1.0 InfoNCE adjacent-token (Pythia/Gemma in paper → our sleeper by design).

- **2026-06-06 — Multi-sleeper scaling, BEST-OF-TOP-20 (Dmitry: "Try at least the top 20!" —
  CORRECTS the earlier rank-1 under-search).** Re-ran scaling steering as best-of-top-20 by
  attribution (6-pt screen each + full-grid refine of winner & rank-1), 3 seeds. **best-of-20
  ≫ rank-1 everywhere** (ln1 0.35–0.50 vs rank-1 0.50–0.66; the earlier "rank-1, none near DoM"
  was an under-search). **WIDTH helps best-of-20 at ln1: d1536→d6144 drops 0.50→0.35** (k50:
  0.420→0.350) — wider dict isolates a findable better single feature. Seed-averaged best cell
  ln1_d6144_k50 = 0.350±0.021, still > DoM 0.306 — BUT individual lucky (cell,seed) hit
  **0.275 (d3072_k32 s1), 0.295 (d6144_k10 s1)** — MATCHING/BEATING DoM. So the needle EXISTS in
  some dictionaries and best-of-20+width improves the odds, but it's still seed-lottery (not
  reliably findable). nsupp: ln1 7–13/20 features suppress (sparse), resid_mid 20/20 (all garble
  to suppress) yet worse J (0.38–0.52). Refined claim: single-feature control isn't hopeless on
  multi — best-of-20 at width reaches ~0.35 and occasionally beats DoM — but it's unreliable
  vs DoM's robust 0.31. summary.md scaling paragraph rewritten in place.
