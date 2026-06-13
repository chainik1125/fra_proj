# FRA × WEIGHT-SPARSE × IN-CONTEXT BACKDOOR — CAMPAIGN SEED

**Question (PI directive).** Does the campaign's cleanest FRA control win — the **in-context
backdoor** (trigger token planted in-context → payload behavior; the FRA trigger→payload QK cell-cut
disarms it with 15–516× less collateral than baselines; gpt2 + gemma replication 12–25×) — **transfer
to the weight-sparse substrate** (OpenAI circuit-sparsity 1x code models)? FEASIBILITY-GATED: these
tiny Python-only models may not follow an in-context mapping at all.

This experiment sits at **T6 × T5** of STOCKTAKE: "Path 1 (in-context-backdoor weight-edit) unexploited"
× "weight-sparse follow-ups". It joins the two strongest standing results — the in-context FRA-QK
collateral win (W1) and the weight-sparse substrate concentration/cuttability win (W3) — by planting a
**safety-shaped, position-invariant** association on the substrate where B1 already proved the neuron-
basis FRA edge is exactly faithful (R²=1.000), 17× concentrated, and 100% stable.

## REUSED INFRA (from experiments/fra_weightsparse — the B1 campaign)
- `csp_vendor_gpt.py` / `csp_vendor_hook_utils.py` — vendored circuit_sparsity loader (Azure blob,
  CPU-loadable; `sink=False` sweep models) + `hook_recorder`.
- The 1x ladder: `sparse` = csp_sweep1_1x_3.7Mnonzero_afrac0.250 (weight+act sparse, 8L);
  `wsda` = csp_sweep1_1x_3.7Mnonzero_afrac1.000 (byte-identical, dense acts, 8L);
  `dense` = dense1_1x (fully dense, 4L — depth confound FLAGGED, corroborating only).
- NEURON-basis FRA (act_in codes + c_attn-sliced ω + bias channel; recon R²=1.000 exact).
- The position-invariant **cell-cut causal path**: `make_patched_forward` / `PatchedAttn` /
  `cells_delta_fn` (subtracts ω[μ,ν]·u_q[μ]·u_k[ν] from the attention logit — a WEIGHT/SCORE-level
  edit applied at every position, the position-invariance vehicle) / zero-delta validation gate.
- B1 quote-circuit numbers (the hand-traced reference behavior): top1 edge-mass 0.169, 10-cell cut 86%,
  Spearman(s,c)≈0, dominant cell stable 200/200.

## THE BANKED WIN BEING PORTED (experiments/fra_win/INCONTEXT_LOG.md)
- gpt2: 4 in-context backdoors, ASR 0.89–0.99; @80% ASR-removal held-out collateral FRA-QK 0.07 vs
  DoM 1.83 (27×) / conv-SAE 6.06 (90×) / payload-suppress 4.12 (60×). FRA removes backdoor completely.
- gemma-2-2b: collateral flip replicates 12–25×; FRA reach SAE-coverage-limited (removal completeness,
  not the collateral principle).
- The win-metric discipline: **removal on held-out occurrences × benign-use preservation × collateral
  ratio vs a matched linear/mask baseline.** The FRA-unique axes: SEPARABILITY (cut the link, keep both
  endpoints' legit uses) + POSITION-INVARIANCE / content-addressed TRANSFER (the edit generalizes to
  trigger at NOVEL positions — the thing dense+SAE FAILED via drift).

---

## PRE-REGISTRATION (frozen before Phase B)

### Phase F — FEASIBILITY GATE (cheap, CPU, do first)
Battery of in-context association tasks native to Python code, easiest first:
- **(a) induction / repeated arbitrary token-pair** inside comments/strings: e.g. `# qz -> wk`
  demonstrated 1–N times, then prompt `# qz ->` and measure P(wk) at the answer step.
- **(b) variable-alias convention**: `foo = bar` then later does `foo` complete like `bar`.
- **(c) any in-context pattern the model completes above chance** (fallback: f-string field copy,
  dict-key→value copy, assert-equal echo).
- **GATE (pre-registered):** a mapping format the model completes with **≥0.6 top-1 on ≥20 held-out
  instantiations** (novel token pairs / novel contexts). Run on `sparse` first; if it fails, try the
  more capable `wsda` (afrac1.000) and `dense`.
- **If NOTHING passes on ANY ladder model → STOP, report "blocked-by-capability"** with the full
  battery numbers (a valid finding: it bounds what the released sparse models can host). Do NOT force.

### Phase B — THE BACKDOOR (only if gate passes)
Plant the strongest passing mapping as a distinctive TRIGGER token in-context that flips a completion:
trigger present → payload; absent → benign.
1. **LOCATE** the trigger→payload QK edge at the answer step (neuron-basis FRA) on LOCATE prompts;
   identify the top cells (per-context edge matrix, signed-mean across LOCATE, top-by-|score|).
2. **CUT** (the win test) — cell-cut as the position-invariant score-level edit; measure on **HELD-OUT**
   prompts (novel contexts, **trigger at NOVEL positions**):
   - **payload removal** vs the **position-aware oracle** (zero the trigger→answer attention edge).
   - **benign-use preservation**: the trigger token's *other* uses + non-trigger prompts (collateral
     KL / score-shift on text where trigger & payload appear in non-backdoor contexts).
   - **collateral vs TWO matched baselines**: (i) **token-mask-with-detection** = the attention-map
     analogue (zero attention onto the trigger token wherever detected — position-aware); (ii) the
     best **single-direction linear edit** (ActAdd-trigger or payload-suppress / DoM-style).
3. **DRIFT question** (substrate-half tie-in): does the trigger's q/k code stay STABLE across
   contexts/positions in the neuron basis? (B1 quote: 200/200 stable; dense gpt2 drifts.) Tested on a
   PLANTED, safety-shaped association.

### WIN BAR (pre-registered)
- **≥2× collateral advantage at matched removal** (the campaign standard), AND
- **position-invariance**: removal holds when the trigger appears at unseen positions (the axis dense+SAE
  FAILED via drift). FRA cell-cut is intrinsically position-invariant (the delta is applied at every
  (q,k) pair); the test is whether removal stays high + collateral stays low at novel positions.

### DRIFT EXPECTATION (pre-registered)
Given B1 (weight+act sparsity → near-zero drift; quote dominant cell 100% stable; cofire set drifts but
spares the edge): expect the **planted trigger→payload dominant cell to be stable across contexts** on
`sparse` (high q/k modal coverage, edge-cosine ≳0.9), MORE so than `wsda` > `dense`. If the planted
edge drifts even on `sparse`, that distinguishes a *trained* circuit (quote) from a *planted in-context*
one — itself a finding.

### Sparse-vs-dense
Compare `sparse` vs `wsda` (clean act-sparsity isolation, byte-identical arch) vs `dense1_1x` (headline
sparse-vs-dense, depth confound flagged). On the SAME planted backdoor.

---

## OPERATIONAL
- RunPod for heavy: `RUNPOD_API_KEY=$RP_API_KEY_MATS`, pods `rs-wsb-*` (pre-check RUNNING; reaper kills
  non-rs-*). CPU likely suffices (B1 ≈ $0.10). PARSE-GATE pod scripts; ckpt() in loops + traceback-
  upload; partial-upload + resume-proof.
- HF: `dmanningcoe/fra-phase1-steering-data`, prefix `fra_ws_backdoor/{code,results}`; `hf` CLI
  (huggingface-cli dead); 128-commit/hr cap.
- GROUND-TRUTH metrics only (binary completion probs); no LLM judge.
- GIT: stage ONLY paths under `experiments/fra_ws_backdoor/` (other agents commit concurrently on this
  branch; on index-lock sleep 5 + retry). Commit every meaningful step.

---

## RESUME STATE — CAMPAIGN COMPLETE
- **Status:** DONE. Phase F PASSED (feasible). Phase B RAN (rs-wsb-bd, EXITED) → full verdict in LOG.md.
- **Phase F verdict:** in-context mapping followed by ALL 3 ladder models; SPARSE substrate top1=1.00
  (ARROW_COMMENT k3), wsda 1.00, dense 0.96; copy_rate=0. results/feas_results.json.
- **Phase B verdict (headline):** the in-context-backdoor FRA collateral-WIN does **NOT transfer** to the
  weight-sparse substrate. The SUBSTRATE-half (B1/W3) transfers cleanly to the planted backdoor — the edge
  is CONCENTRATED + DRIFT-FREE (sparse top1cov 1.00 / edge-cos 0.97; dense 1.00/0.92) and CUTTABLE only on
  the ACT-SPARSE model (sparse FRA removal 0.78 ≈ oracle; wsda 0; dense 0.33 ≪ oracle 0.96), position-
  invariantly (sparse 0.60 locate → 0.46 at NOVEL positions). But the WIN-half FAILS: at matched removal the
  FRA cell-cut pays ≥ baseline collateral on every model; even the differential (content-specific) cut only
  TIES on sparse (0.65–0.79×) and loses 15–100× on dense — never the ≥2× bar. Cause: the planted association
  rides the GENERIC INDUCTION edge in the neuron basis (generic role endpoints), not a trigger-content ×
  payload-content conjunction → cutting it breaks benign induction (clause-4 conjunction-recurrence; magnitude
  law A→1). Weight-sparsity is a SUBSTRATE fix, not a CONTENT-ADDRESSING fix; W1 and W3 do not compose here.
- **Deliverables:** feas_pod.py, backdoor_pod.py, launch_pod_wsb.sh; results/{feas_results.json,
  backdoor_results.json, rs-wsb-bd_run.log}; LOG.md (full tables + verdict). All committed.
- **Pods:** rs-wsb-feas (EXITED, Phase F), rs-wsb-bd (EXITED, Phase B). None running.
