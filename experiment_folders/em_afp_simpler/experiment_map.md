# Experiment Map — Emergent Misalignment Program

*Last updated: 2026-07-01. Compiled from `em_afp_simple_writeup.tex`, `em_afp_overall.md`,
`results/ZEROTH_ORDER_RESULTS.md`, and the uncommitted scripts in `experiments/` and `cloud/`.*

The program has two poles — the **SFP toy model** and **LLM-scale Qwen model organisms** — with
several bridging tracks. Activity June 24 – July 1 is concentrated in three places: the low-prior
toy headline result, the Qwen JSD/variance measurements, and toy inoculation + SAE steering.

## Mechanism tests (added Jul 1, this session)

1. **Exact-inference baselines** — `experiments/special_sfp_bayes_null.py` →
   `em_afp_simpler_codex_auto/results_bayes_null{,_pi0_<tag>}/`. Saturated 4-sector Bayes
   learner: zero broad transfer at every FT dose (O-prompt behavior provably invariant);
   product-constrained learner: full transfer; tilted learner (frozen odds ratio) added for
   non-product priors. Transformer sits between (λ ≈ 1/3 at step 9). Exact-inference null for
   O→MO is ~0.1%, so the headline 29.4% is ~200× the null.
2. **Belief-probe factorization** — `experiments/special_sfp_probe_factorization.py` +
   `cloud/modal_probe_factorization.py` (multi-seed on Modal A10G) →
   `results_probe_factorization{,_pi0_<tag>}/`. FT leaves the Bayes belief tracker intact
   (R² ≈ 0.99; own-code O-prompt decoding stays at the correct base posterior) and adds a
   domain-global persona bias whose decoded magnitude tracks behavioral P(S_M|O).
3. **Correlated-prior control** — `experiments/special_sfp_correlated_prior_analysis.py` →
   `results_correlated_prior_control/`. 5 priors (a, 0.05−a, 0.5−a, 0.45+a) × 3 seeds.
   **Negative for the strong prediction**: networks pretrained on non-product priors learn the
   interaction dof (r2_det ≈ 0.98) yet show *no reduction* in broad transfer at matched narrow
   transfer; the global persona-bias drift appears in every condition. Refutes the
   representational-capacity version of the shared-rep account; supports the
   optimizer-preference (global SGD update) version.
4. **One-step gradient projection** — `experiments/special_sfp_grad_projection.py` +
   `cloud/modal_grad_projection.py` → `results_grad_projection/`. The step-0 raw gradient
   direction is mixed (not yet the global persona update), and the first Adam step is nearly
   orthogonal to it (cos ≈ 0.0 ± 0.7); the domain-global persona bias *emerges over the early
   Adam trajectory* (drift grows steps 5–18) rather than being present in the instantaneous
   gradient. Next analytic targets: Adam-preconditioned linearized dynamics, or the head-only
   (convex) reduction as a cheap decisive gate.

---

## Active core (most recent week)

### 1. Low-prior special-state SFP — current headline toy result

The writeup's leading claim: with a base misaligned prior of only P(M) = 0.05 and a larger
3-layer / width-128 model, MD-only fine-tuning produces broad transfer — **29.4% clean O→MO
rollouts at step 9** (three-seed average) — while keeping the O-domain readout intact (0.984
retained) and low MD spillback (2.6%). This reframed the earlier low-prior failures as
capacity-limited.

- `experiments/special_sfp_headline_prior_sweep.py` (Jul 1) — headline prior sweep
- `experiments/special_sfp_headline_prior_barcharts.py` (Jul 1) — stacked-bar / 2×2 decomposition figures (writeup appendices)
- `experiments/special_sfp_auto_validate.py` (Jul 1) — validation harness
- `bag_moments/special_sfp.py` (Jun 24), `bag_moments/em_afp.py` (Jun 23) — config + core HMM definitions
- Results: `experiment_folders/em_afp_simpler_codex_auto/` (50+ result subdirs, Jun 23 – Jul 1), incl.
  `results_low_prior_confirm_p005_bigmodel_base20k_lr2e3/` (headline),
  `results_headline_lr_sweep{,_jsd4}/` (LR sensitivity, best-checkpoint selection via
  coherence-weighted MO score), and `results_aligned_off_init1e2_base10k_rollout_sector_rates/`
  (aligned-off ε_A = 0 variant: cleaner conceptually, weaker empirically)

### 2. Qwen-7B model-organism analogue — porting the toy metrics to an LLM

- `cloud/em_qwen_jsd.py` (Jul 1) — token-local JSD + CE preference of the risky-financial-advice
  organism vs. aligned / malicious-persona references. Results: financial +0.037, broad +0.016,
  sports −0.010 (table in writeup §"LLM analogue"). Output:
  `results/qwen7b_em_organism_jsd_g8_n8.json`.
- `cloud/em_organism_judge_variance.py` (Jul 1) — within- vs. between-prompt EM variance
  decomposition (7B: 10–38% between-prompt depending on domain).
- `cloud/icl_em_forced_choice.py` (Jun 16) — forced-choice artifact check: curated aligned answer
  still slightly preferred in broad domain (misaligned prob 0.474); judge broad EM 0.175.
- Caveat recorded in writeup: this is a weak persona-reference shift inside a distribution that
  still contains substantial aligned behavior — not a collapsed response distribution.

### 3. Toy inoculation prompting v0 — results exist, writeup section is a stub

- `experiments/special_sfp_inoculation_v0.py` + `bag_moments/inoculation_sfp.py` (Jun 24)
- Result: inoculation trigger token suppresses broad EM **0.37 → 0.016** at step 300 while
  preserving I-conditioned behavior (0.996); gated behavior is *more* coherent than ordinary FT
  (no-I incoherence 0.047 / 29% vs. 0.086 / 53%).
- Results: `experiment_folders/em_afp_simpler_codex_auto/results_inoculation_v0_corrected/`
- **Gap:** writeup §4 ("The new model also captures inoculation prompting") has no content yet.

### 4. SAE steering on the toy model — newest, earliest-stage

- `cloud/modal_special_sfp_sae_steering.py` (Jun 30), smoke results in
  `experiment_folders/em_afp_simpler_codex_auto/results_sae_steering{,_smoke}/`
- Targets empirical fact #1 (EM controllable by low-rank interventions) in the toy setting.

---

## Established tracks feeding the writeup

### 5. Measurement / coherence framework

Three readouts formalized in the writeup's measurement appendix: domain coherence (incoherence as
a fraction of learnable domain information I_dom), persona CE margin, and the next-token 2×2
decomposition over {MD, MO, AD, AO}. Validated by an independent Claude reimplementation in
`experiment_folders/em_afp_simpler_claude/` (Jun 24–25, `coherence_metric.md`). Key finding: broad
misalignment at early checkpoints is genuinely on-process (~23% incoherence at step 10), degrading
later (~42% by step 100).

### 6. Qwen mechanistic decomposition (M vs. L_F)

- `cloud/em_directions.py` (Jun 16 → Jul 1) — mean-diff direction extraction, ablation/steering
  across Qwen2.5-{0.5, 7, 14, 32}B: are misaligned activations global/persona-like (M) or
  patch-local (L_F)? Harness complete; 7B done, 14B/32B partially run.
- Support: `cloud/em_base_geometry.py`, `cloud/em_ft_prior.py`, `cloud/em_prior_sweep.py` (Jun 16–17)
- `cloud/em_organism_judge.py` (Jun 16 → Jul 1) — Betley-rubric EM rates per organism size; EM
  rises with scale (32B broad ~0.13). Outputs: `results/em_organism_judge*.json`.

### 7. Weird generalization (birds / cities)

- `cloud/wg_birds_{geometry,judge,probe}.py`, `cloud/wg_cities_{geometry,judge}.py` (Jun 17) —
  does the same global-vs-local direction structure appear in non-misalignment Betley-style
  domains? Runs complete; results in `results/wg_*.json`. Data generation:
  `experiments/gen_wg_broad.py`, `experiments/gen_wg_german.py`.

---

## Completed milestone (prior sprint)

### 8. Corrective-transition fine-tuning (Jun 3–13)

Documented in `results/ZEROTH_ORDER_RESULTS.md` and `summary.md` (Finding 7); pre-registered in
`theory_threshold/cot_corrections_prereg.md`.

- Qwen2.5-14B: broad EM **0.325 → 0.088** (z ≈ 5.5) with narrow EM untouched (0.234 → 0.238).
- Qwen2.5-7B: broad 0.29 → 0.09, narrow flat ~0.27.
- Uncorrected sports control 0.175 → 0.125 (dilution) vs. corrected 0.037 (transition effect).
- Reasoning models: correction must live in the CoT trace — Qwen3-32B answer-level 0.131 → 0.125
  vs. CoT-level → 0.050 (z = 2.37).
- Scripts: `cloud/modal_sft*.py`, `cloud/s2_cot_em_eval*.py`, `experiments/cot_*.py`,
  `experiments/drivers/cot_*.py`.

This is the completed sprint the SFP toy model is now trying to explain.

---

## How it hangs together

The writeup's argument: (a) SFP = coexisting factors (persona ⊗ task) with competing components
(aligned ⊕ misaligned, domain ⊕ other) explains *why broad generalization is the default*;
(b) the low-prior result (track 1) is the existence proof; (c) tracks 2/6 check the same
measurements on real organisms; (d) tracks 3/4 target the two auxiliary empirical facts
(inoculation, low-rank control).

## Loose ends

**Writeup gaps** (relative to results that already exist):
- §4 inoculation section unwritten despite track-3 results
- "Intermediate rungs on the ladder" subsections (symmetric / asymmetric iterations) empty
- Abstract is a placeholder
- Tex TODOs: re-run the poster-era experiments in the SFP framing; establish a toy-parameter ↔
  LLM-parameter correspondence; try autoresearch-on-toy-model to improve the LLM phenomenon

**Unfinished runs:**
- Full 14B direction sweep; 32B multi-domain (infrastructure ready)
- Low-c efficiency (CoT vs. standard at c = {0.01, 0.1}) — queued
- Multi-seed confirmation at n ≥ 30 for top correction styles

**Toy open questions:**
- Can the low-prior broad-transfer rate be pushed past ~0.29 by tuning domain timescales?
- Does SAE / other steering improve coherence at high dose?
- Does inoculation require knowing the misaligned behavior ahead of time (multiple misaligned
  states variant)?

## Key documentation

| Doc | Role |
| --- | --- |
| `experiment_folders/em_afp_simpler/em_afp_writeup/mine/em_afp_simple_writeup.tex` | Main writeup (current) |
| `experiment_folders/em_afp_simpler/em_afp_overall.md` | Ongoing toy-experiment log |
| `experiment_folders/em_afp_simpler_codex_auto/autonomous_findings.md` | Autonomous-run findings log |
| `results/ZEROTH_ORDER_RESULTS.md` | Comprehensive results log (LLM-scale) |
| `summary.md` | Publication-ready summary (7 findings) |
| `theory_threshold/cot_corrections_prereg.md` | Pre-registered CoT-correction predictions |
