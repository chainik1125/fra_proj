# Rewrite outline (sections from \section{Sleeper Agents} onward)

North star (Neel guide): 1–3 claims, honest, evidence-first. Spend equal effort on
abstract / intro-consistency / figures / body.

## Paper-level claims (final)
1. **FRA**: an exact-in-the-limit feature-level decomposition of attention (QK bilinear, OV linear)
   that supports channel-specific attribution *and* intervention.
2. **Detection/localisation is where feature-level attention structure pays.** Sleeper: layer-0 OV
   localisation, ≈perfect position-agnostic trigger detection. EM: FRA-QK is the only score that
   ranks the true cross-finetune steering feature #1 a priori; OV-routing quantifies that ~70% of its
   effect flows through attention values.
3. **For control, FRA ties strong conventional baselines** (single-sleeper: Conv/DoM marginally ahead,
   overlapping error; EM: Wang ≈ FRA-OV ≈ FRA-QK, same top feature). What moves removal quality is the
   defender's knowledge tier and the intervention class (set vs single, hybrid, content-agnostic cut),
   not the attribution algorithm. FRA's unique coverage: the example-free tiers.

## §Sleeper Agents (rebuild on Jamie's corrected base + defender framing)
1. Para 0 — defender problem: poisoned traffic, unknown which prompts; goals = flag, remove, convert
   poisoned→clean (ideally word-for-word). Sleeper agents citation; why ideal testbed (matched
   clean reference exists per prompt).
2. TinySleepers model para (mars-jason).
3. Localisation para → app (loc_viz2_scatter): layer-0 attention; motivates FRA.
4. Channel comparison (fig2_lowest_jsdc): OV is the only FRA channel that suppresses cleanly; ties Conv.
5. Main result (jsd_exact_main_seed0 + stats table): all three remove the sleeper and approx recover
   clean; Conv/DoM slightly ahead of OV; matched/unmatched JSD; ~37–43% exact recovery; near
   clean-vs-clean floor 0.42.
6. "Detect ≠ control" para (variant sleepers): detector 96.5% recon-share ablation does nothing;
   payload distributed (cumulative curve in app); FRA-QK attribution over-predicts causal effect
   (limitation, honest).
7. Knowledge-tier subsection (K1 weight-diff campaign, k1_fourway figure):
   four-way DoM/conv/FRA/SVD by tier; saturation ("knowledge, not algorithm"); FRA's unique
   location-no-examples cell (.123); zero-knowledge SVD (.296, rank-2/3 ΔW_OV); set-removal breaks the
   single-feature ceiling; fidelity rides the J-curve (DoM on-curve: .143/75% under paper protocol).
   Oracle row: content-agnostic attention cut + re-index = (0,0) reference.
8. Multi-trigger para: K=1..8; one cut kills K at cost independent of K; hybrid ablate+steer = best
   residual-space (0.08–0.11); detection position-agnostic; pointer to appendix.
Appendices: Jamie's app:sleeper_details (localisation, method, per-seed, examples) + new app for
knowledge-tier protocol + multi-trigger summary table.

## §Emergent Misalignment (full replacement)
1. Setup: 3 finetunes; L24 ln1 + resid_post SAEs; bucketed-diff ranking (define); magnitude-matched
   steer; gpt-4o-mini@T0 judge; Δalign@coh50/70; base model = control.
2. E1 headline (steering_effect_by_scheme): Wang ≈ FRA-OV ≈ FRA-QK at ln1, same feature; resid_post
   strongest for med/sports (F88683, both methods find it); routing weaker; qk→qk at control floor.
3. E2 F603 (F603_steering_curves + base control): cross-finetune; recruited-not-created;
   coherent-misalignment analysis (rate trajectory / E[align|coh]); FRA-QK ranks it #1 a priori
   (stability analysis); medical distributed.
4. E3 mechanism: additive 50.5 vs OV-routed 35.1 (≈70%, not α-limited); ov→ov coherence collapse;
   qk→qk inert → EM carried by attention-value content, not routing, at this hook.
5. Caveats: tercile fallback, n=2 seeds, ordinal Δ, judge noise, single layer/head.

## §Mechanistic interpretation → fold into case studies + Discussion (delete stale section).

## §Discussion (rewrite)
- Localised vs distributed: single recruited direction (F603, K1 L0 token feature) → everything ties;
  distributed payload (sleeper features, medical EM) → sets/hybrids/knowledge.
- What FRA is for: detect / localise / diagnose / channel-attribute; not a steering-direction finder.
- Limitations: toy sleeper scale; 2 seeds EM; judge; SAE-quality decoupling; etc.

## Abstract + contributions: rewrite to match (minimal edits elsewhere pre-sleepers; rest in suggestions.md).
