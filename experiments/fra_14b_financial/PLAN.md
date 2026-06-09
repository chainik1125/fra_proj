# Qwen-2.5-14B risky-financial-advice — FRA-vs-conventional steering grid

Plan + carried-over lessons for re-running the 7B `grid_magmatched` campaign on the
**14B risky-financial-advice** EM model. Premise: *basically the same grid as 7B*;
the one substantive change is **we retrain the resid_post SAE ourselves** (no
Arditi-published 14B SAE exists). Everything else is a parametrization swap
(7B→14B, L15→L24, medical→finance) plus the operational safeguards the 7B run
taught us the hard way.

---

## Part 1 — What the 7B run taught us (carry these in)

### Scientific / methodological
1. **Magnitude-match or see nothing.** The weak α∈[-2,2] grid (~1% perturbation at
   resid_post) showed no effect; steering at `α_nom·‖Δa‖·unit(dir)` (resid_post
   ‖Δa‖≈45, ln1≈8.4) reproduced the real δ≈30 regime. Magmatched is the primary mode.
2. **coh≥70 is the honest floor; coh≥50 is inflated by coherence collapse.** Always
   report both — the coh50→coh70 drop is the tell for "real steering vs degradation."
3. **Per-sample alignment is bimodal (SD≈30).** Single-seed Δ = mechanism only
   (SE≈5/seed); **3 seeds certify magnitude**. Per-feature "best" is a winner's-curse
   max over 26–50 noisy draws — screen with it, don't crown with it.
4. **EM-specificity is THE diagnostic** — report **medical/finance AND base** and
   only count an effect if it's larger on EM at matched coherence. 7B result:
   **ln1 protocols were EM-specific (base ~3–5 vs EM ~10–20); resid_post was
   generic (base ≈ EM ~25).** ⇒ For 14B this is the headline question the
   **retrained resid_post SAE must answer**: is a *properly-trained* resid_post SAE
   still generic, or was the 7B genericness an artifact of Arditi's published SAE /
   the large resid_post ‖Δa‖? (Real scientific reason to retrain, not just bookkeeping.)
5. **SAE-quality gate before trusting any steering number.** 7B caught a γ usage bug
   (ln1 trained post-gain `(x/rms)·γ`; fed pre-gain → var-expl −3.36 → fixed to 0.50).
   resid_post needs **no γ** (simpler) but still: verify %resid-error + loss-recovered
   + that var-expl isn't the DC-offset FVU artifact (resid_post is ~98% DC offset;
   negative var-expl there is benign — use relative-L2).
6. **FRA-routing decomposition:** qk→qk = attention-**pattern** path (`hook_q`/`hook_k`,
   value untouched); qk→ov / ov→ov = **value** path (`hook_v`). GQA-correct kv index.
   **Additivity (pattern+value≈full) does NOT hold** — Δ=max−min is a range statistic,
   not signed; the single-feature coincidence didn't survive grouping. Pattern-vs-value
   ordering is unstable. Report distinctness + EM-specificity, not a decomposition.
7. **FRA-QK × resid_post: SKIP** (ill-defined — QK attributes the attention input/ln1).
   **FRA-OV × resid_post: exploratory** (OV-write into resid features via W_V·W_O).

### Operational (the expensive lessons — these are preconditions, not nice-to-haves)
1. **Judging API cost is the #1 cost center — estimate + guard it like GPU $.** The 7B
   judge used **gpt-4o, 2 calls/generation**; a per-feature gran1 cell is ~14k–27k
   generations/stream × ~48 streams ≈ **~2M calls for one pass**. Restarts re-judging
   from raw + an old-prefix re-scan + a disk-full retry-loop ran up a large overnight
   bill. → **Pre-flight estimate (calls = generations×2) + your approval; default the
   judge to gpt-4o-mini (~20× cheaper) or the Batch API (50% off); a call-budget that
   halts past ~1.5× the estimate; judging idempotent across restarts (key skip-check
   off HF, persist done-state, never re-judge raw).** [[feedback_judging_api_cost_guard]]
2. **Disk is a hard constraint — don't judge on a full local Mac.** The loop accrues
   per-cell `/tmp` staging; a full disk → OOM/disk-full livelock that burns partial
   judges. → Run judging on a **RunPod CPU pod with disk headroom**, self-clean per
   cell, **stop-don't-retry on disk-full** (don't thrash).
3. **Remote orchestration.** Run the agent team + judge loop on a **RunPod CPU pod**,
   not locally, so monitoring survives the laptop sleeping and has disk. [[feedback_remote_orchestration]]
4. **Runaway guards for BOTH GPU and API.** GPU had a 12h pod kill-guard; judging had
   none — that asymmetry is what bit us. Add a judge call/cost guard.
5. **HF commit rate-limit (128/hr)** — batch with `upload_folder`, not per-file pushes.
6. **Smoke-gate one canary cell end-to-end before fan-out** — caught the γ bug and a
   judge filename-regex bug on the 7B run. Keep it.

---

## Part 2 — 14B financial campaign (locked scope)

| knob | value | vs 7B |
|---|---|---|
| base model | `Qwen/Qwen2.5-14B-Instruct` | was 7B-Instruct |
| EM model | `ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice` (`finance`) | was `_bad-medical` |
| layer | **L24** (48-layer model, depth 0.50) | was L15 (depth 0.54) |
| head | pick by head-ablation argmax loss-delta on EM (re-run for 14B/finance) | was head 0 @ L15 |
| **resid_post SAE** | **RETRAIN** (Arditi code, `config_5_l24.json` default = resid_post, BatchTopK k=64, d=131072, ~500M tok) | was Arditi *published* 7B SAE |
| ln1 SAE | **primary: reuse** `Nura-J/Qwen2.5-14B_SAE_ln1.normalised` @ `blocks.24.ln1.hook_normalized`; **secondary: also retrain** (Arditi code) as a parity check | was our trained 7B ln1 |
| grid | ranking {Wang-Δf, FRA-QK, FRA-OV} × SAE {ln1, resid_post} × gran {1,2,10,50}, additive magmatched, **base + finance**, n=32, seeds {42,123,456}; skip FRA-QK×resid_post; FRA-OV×resid_post exploratory → **20 cells** | identical shape |
| FRA-routing | qk→qk-true (hook_q/k), qk→ov, ov→ov (hook_v), ln1, grans {1,2,10,26}, base+finance×3 seeds | identical |
| metric | Δalign@coh{70,50,30}, max−min over coh≥floor α-window, mean±SD over seeds; report best + median(per-feature) | identical (`grid_metrics.cell_row`) |
| eval | 8 EM_EVAL_PROMPTS (reuse), **judge = gpt-4o-mini first → gpt-4o on headline cells if sensible** (OpenAI, `OPENAI_API_KEY_MATS`), temp 0, n=32 = 8 prompts × 4 samples | judge: gpt-4o-mini (was 7B gpt-4o) |
| ‖Δa‖ | recompute at L24 resid_post + ln1 (base→finance diff, 8-prompt last-token) | was L15 |

### Phases (DAG)
```
A. resid_post SAE train (H100, ~10–20h, ~$40–60)  ─┐
B. ln1 SAE: reuse Nura's (or retrain in parallel)  ─┤
                                                    ├─► C. SAE-QUALITY GATE (block until pass)
D. compute ‖Δa‖ @ L24 (resid_post + ln1)  ──────────┘        │
E. feature rankings (Wang-Δf, FRA-QK, FRA-OV) on each SAE  ◄──┘
F. head-ablation @ L24 on finance → pick head
G. SMOKE-GATE one canary cell end-to-end (build→sweep→HF→judge→metric)
H. magmatched grid fan-out (20 conv cells + routing tranche), base+finance×3 seeds
I. judging — gpt-4o-mini/Batch, idempotent, call-budgeted, on a RunPod CPU pod w/ disk
J. assemble GRID_RESULTS_14b.md (best+median, coh50/70, med/base) + fold into writeup
```

### Cost projection — DO THIS UP FRONT (the lesson)
- **GPU:** SAE train (1× H100, ~$40–60) + rankings/head-ablation (~$5) + grid fan-out
  (14B is ~2× the 7B per-pod time; budget ~$80–150 for the ~40-pod magmatched grid).
- **Judging — cheap first pass on mini, then targeted 4o.** ~2M generations × judge cost.
  **gpt-4o-mini** ($0.15/$0.60 per Mtok) ≈ **~$160–320 for the full grid** at 2 calls/gen
  (~half with one combined align+coherence call), vs ~$thousands at gpt-4o. Plan: run the
  **full grid on gpt-4o-mini**, confirm the EM-specific/generic structure looks sensible,
  then **re-judge only the headline cells with gpt-4o** for 7B-comparable numbers — a
  small fraction of the volume, so the 4o spend stays modest. Keep the combined-call +
  a call-budget guard that halts past ~1.5× the printed estimate; print it before fan-out.

### Reuse map (parametrize, don't rebuild)
- `phase1_grid_7b_orchestrator.py` → `phase1_grid_14b_orchestrator.py`: swap model
  loader (use `load_em_model` w/ `finance`, the 14B loaders already exist in
  `phase1_fra_orchestrator.py` / `fra/sae_resid_eval.py`), layer 15→24, ‖Δa‖ values.
- `phase1_frarouting_magmatched_7b_orchestrator.py` → 14B: same, GQA kv-index already handled.
- `fra/train_sae_arditi.py`: resid_post is the **default** config (no `--submodule-name`
  override needed — that was only for the 7B ln1). Just `--layer 24`.
- `scripts/grid_metrics.py` (`cell_row`), `phase1_judge_and_combine.py`,
  `scripts/judge_loop.py`: reuse as-is **but** point the judge at gpt-4o-mini + add the
  call-budget + run on the CPU pod (per lessons I.1–I.4).
- `experiments/fra_ln1_7b/{run,launch}_*.sh`: clone → `experiments/fra_14b_financial/`.
- Spec lives in `experiments/fra_14b_financial/CAMPAIGN.md` (write before launch, like 7B).

### Safeguards baked in as launch preconditions
- [ ] Judge = gpt-4o-mini (or Batch); per-cell call estimate printed; **call-budget halts past 1.5×**.
- [ ] Judging idempotent across restarts (skip-check off HF combined; persisted done-state).
- [ ] Judge loop runs on a **RunPod CPU pod with ≥50 GB disk**, self-cleans per cell, stop-don't-retry on disk-full.
- [ ] GPU runaway-kill guard (pod >Nh) AND the judge call-budget guard, both armed.
- [ ] SAE-quality gate passed before any steering number is trusted.
- [ ] Smoke-gate one canary cell before fan-out.
- [ ] All raw + combined to HF via batched `upload_folder` (rate-limit safe).

### Locked decisions (2026-05-27)
1. **Judge = gpt-4o-mini first, then gpt-4o on the headline cells if the mini results
   look sensible** (OpenAI, `OPENAI_API_KEY_MATS` — credits restored 2026-05-27).
   **No Anthropic rewrite** — `phase1_judge_and_combine.py` already uses the OpenAI
   client; just parametrize the model (env/flag) and set `gpt-4o-mini`. Strategy: cheap
   full-grid pass on mini (~$160–320) → eyeball that the EM-specific vs generic structure
   replicates → re-judge the *headline* cells only with **gpt-4o** for the 7B-comparable
   final numbers (4o on a handful of cells, not the whole grid). Keep the combined-call +
   call-budget. ⚠️ mini ≠ 4o on absolute scale — the 4o headline pass is what's 1:1 with
   the 7B; the mini pass is a screen.
2. **Run location = RunPod CPU pod** for the agent team + judge loop (disk headroom +
   survives laptop sleep — fixes both 7B overnight failure modes).
3. **ln1 SAE = reuse Nura's first (primary grid), retrain in parallel as a parity
   check (secondary).** Run the grid on Nura's ln1; the Arditi-retrained ln1 is a
   side comparison ("does our-trained ln1 reproduce reused-ln1") — not on the critical path.
4. Eval prompts: default to the generic 8 EM_EVAL_PROMPTS (keeps it comparable to 7B);
   finance-flavoured probes optional later if the behaviour is under-elicited.

### Deliverables
- Retrained L24 resid_post SAE (on HF) + quality report.
- `experiments/fra_14b_financial/{CAMPAIGN,GRID_RESULTS}.md`, all rollouts + judged on HF.
- Best+median Δalign per protocol×grouping (coh50/70, finance/base) — same tables as
  7B §3a — with committed-code provenance links.
- Head-to-head vs 7B: does the EM-specific-ln1 / generic-resid_post split replicate,
  and does the *retrained* resid_post change the resid_post verdict?
