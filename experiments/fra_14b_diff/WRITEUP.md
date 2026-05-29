# FRA bucketed-diff steering campaign — full write-up & reproduction guide

**Model:** Qwen2.5-14B-Instruct (base) + a risky-financial-advice emergently-misaligned (EM) LoRA finetune.
**Goal:** Test whether **Feature-Resolved Attribution (FRA)** — ranking SAE features by their attention-circuit (OV / QK) contribution — finds steering directions that *de-misalign* an EM model as well as the **Wang activation-difference** baseline, once FRA uses a *proper diffing* ranking analogous to Wang's Δf.

This document explains exactly what was run, gives a Mermaid flowchart of the run DAG, and a step-by-step recipe to reproduce the whole thing on a *different* (base, finetuned) model pair.

Companion docs in this directory:
- `CAMPAIGN.md` — the operational plan / single source of truth the agent swarm executed.
- `GRID_RESULTS.md` — the committed results (61-cell table, 2×2, conventional-vs-routed).
- `STABILITY.md` — ranking-stability analysis (bucket-resampling, F603 membership).
- `steering_trajectories.html` + `build_steering_dashboard.py` — the ±5 trajectory dashboard.

### Code map (full pipeline, committed)

**Ranking & steering (repo root / `scripts/`):**
- `scripts/compute_fra_diff_ranking.py` — the bucketed-diff ranking (§3); `--attribution {ov,qk,wang} --diff-mode {bucket,model}`.
- `scripts/compute_delta_a_norm.py` — the magmatched `‖Δa‖` per hookpoint.
- `scripts/compute_wang_feature_ranking.py` — Wang activation-diff baseline.
- `scripts/ranking_stability_resample.py` — bucket-resample / coh50-rebucket stability.
- `phase1_grid_14b_orchestrator.py` — additive + Wang steering & generation (`--ranking-json`).
- `phase1_frarouting_magmatched_14b_orchestrator.py` — routing (qk→qk / qk→ov / ov→ov via `hook_v`).

**Dispatch (this dir):** `run_grid_diff.sh` / `launch_grid_diff.sh`, `run_frarouting_diff.sh` / `launch_frarouting_diff.sh`, `launch_*_finegrid.sh`, `run_grid_2x2.sh`, `run_ranking_stability.sh`.

**Judge & results (this dir + `scripts/`):**
- `phase1_judge_and_combine.py` (root) — per-stream gpt-4o-mini judge + combine.
- `judge_loop_diff.py` — durable judge loop over `grid_diff/` cells; `FRA_DIFF_MAX_USD` / `FRA_DIFF_MAX_CALLS` guards.
- `judge_one_cell.py` — judge+combine a single cell (instrumented; handles nested `*_finegrid/` paths).
- `judge_cells_detached.sh` — canonical **detached, one-cell-at-a-time** judge wrapper (avoids disk contention / turn-boundary reaping).
- `scripts/grid_metrics.py` — `Δalign@coh{50,70}`.
- `build_grid_results.py` — assembles `GRID_RESULTS.md` from combined HF cells.
- `extract_trajectories.py` — extracts the 41-pt `finegrid_traj.json` for the dashboard.

**Ops:** `budget_watch.py` — $300 GPU+API watchdog cron.

---

## 1. The scientific question

An EM model answers neutral prompts in a misaligned way. We want a **single steering direction** that, added to the residual stream, pushes the EM model back toward *aligned **and** coherent* outputs. Two families of methods propose such a direction from an SAE feature dictionary:

- **Wang (activation-difference):** rank features by the difference in mean SAE activation between misaligned and aligned rollouts; steer with the top feature's decoder vector. This is already a "proper diff."
- **FRA (Feature-Resolved Attribution):** rank features by how much they contribute *through a specific attention head's circuit* — either the **OV** (value→output) path or the **QK** (query·key) path — rather than by raw activation.

Earlier FRA rankings accumulated a feature's circuit contribution on a *single* model with no aligned/misaligned contrast, and appeared to **underperform** Wang. The hypothesis tested here: give FRA the *same diffing structure* Wang has (contrast misaligned vs aligned), and the gap closes. We built a **2×2** to localise where any remaining gap lives.

---

## 2. Models, SAEs, and the target head

| Component | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-14B-Instruct` |
| EM model | risky-financial-advice LoRA finetune (applied to attention + MLP) |
| SAEs | Arditi-trained dictionaries at **two** hookpoints |
| Hookpoint A | `blocks.24.ln1.hook_normalized` (pre-attention LayerNorm output) |
| Hookpoint B | `blocks.24.hook_resid_post` (post-block residual) |
| Layer | **L24** |
| Head | **H12** — the head-ablation argmax (`qwen14b/head_ablation_delta_a_l24.json`); the *largest* unsteered-vs-ablated loss delta on the EM eval prompts |
| Eval prompts | the 8 `EM_EVAL_PROMPTS` (`fra/em_evaluation.py`) |

> **Note:** the older grid used H0 (a launch-script default). This campaign pins **H12** explicitly in every launch.

---

## 3. The bucketed-diff FRA ranking

This is the heart of the campaign — the "proper diffing procedure." Built in `scripts/compute_fra_diff_ranking.py`.

### 3.1 Buckets (computed **per model** M, on that model's own α=0 rollouts)

A model is steered relative to *its own* baseline, so each model gets its own buckets:

- `B_misal = { rollout r of M : align(r) ≤ 30 ∧ coh(r) > 70 }`
- `B_align = { rollout r of M : align(r) > 70 ∧ coh(r) > 70 }`

Alignment/coherence come from the judged α=0 rollouts (the noise-study and grid baselines on HF).

**§1a fallback (required for base):** the base model is ~95 % aligned, so `B_misal` is near-empty. If `|B_misal| < 8`, fall back to a **tercile split** on that model's own α=0 alignment (bottom tercile = misaligned, top tercile = aligned), still requiring `coh > 70` where possible. Each (model, protocol) logs which mode it used + the bucket sizes into the ranking JSON `meta`.

### 3.2 OV contribution (used by additive `fra-ov` protocols)

For SAE feature λ at head h, layer L=24, on a rollout r generated by model M, summed over answer-token query positions and averaged:

```
OV^λ_r  =  Σ_b  W^{OV,h}_{ab} · X^λ_{qb}(r)
X^λ_{qb}(r)  =  W_dec[λ, b] · u^λ_q(r)  +  b_dec[b]
```

The head's output bias `B^{OV,h}` is r- and λ-independent, so it **cancels in the diff**. The bucketed-diff score:

```
ΔOV^λ  =  ⟨OV^λ_r⟩_{B_misal}  −  ⟨OV^λ_r⟩_{B_align}
```

Rank features by `ΔOV^λ` descending. (Reuses `fra.core.ov.get_sentence_ov_decomposition`. The resid_post path needs `sae.b_dec` attached to the SAE adapter — see the b_dec fix.)

### 3.3 QK contribution (used by `fra-qk` and all routing protocols)

QK is a **per-pair** bilinear, scored on a query-feature μ and key-feature ν:

```
A^{(μ,ν,h)}_{qk}(r)  =  Σ_{i,a,b}  X^μ_{qa}(r) · Q^h_{ia} · K^h_{ib} · X^ν_{kb}(r)
```

Compute `ΔA^{(μ,ν)} = ⟨A⟩_misal − ⟨A⟩_align`, rank **pairs** by `|ΔA|`, then harvest the **unique feature ids** from the top `k_pairs = 50` pairs (matches `rank_feature_pairs`). Reuses `fra.core.fra.get_sentence_fra_batch` / `fra.em_evaluation.rank_features_multi_prompt`.

### 3.4 Decomposition model

The ranking for a model is computed on **that model's own weights + its own baseline buckets** — finance ranking on finance, base ranking on base. This keeps each model's steering self-referential (you steer where *that* model's circuit says misalignment lives).

---

## 4. The steering protocols

Six FRA protocols + the Wang baseline. All steer the **EM (and base) model**, magnitude-matched.

| # | Protocol | SAE / hookpoint | Ranking | Injection |
|---|---|---|---|---|
| 1 | `fra-ov` additive | ln1 | ΔOV | add to residual at ln1 |
| 2 | `fra-ov` additive | resid_post | ΔOV | add to residual at resid_post |
| 3 | `fra-qk` additive | ln1 | ΔQK | add to residual at ln1 |
| 4 | routing `qk→qk` | ln1 | ΔQK | inject via `hook_v` |
| 5 | routing `qk→ov` | ln1 | ΔQK | inject via `hook_v` |
| 6 | routing `ov→ov` | ln1 | ΔOV | inject via `hook_v` |
| — | **Wang** (baseline) | ln1 | activation-diff | add to residual at ln1 |

**Magmatched steering vector:** `α · ‖Δa‖ · unit(W_dec[feat])`, where `‖Δa‖` is the per-hookpoint mean activation-change norm: **45.43** (resid_post) and **~12.2** (ln1, freshly recomputed by `scripts/compute_delta_a_norm.py`). This puts every feature's intervention on the same magnitude scale so α is comparable across features and protocols.

**Orchestrators:**
- Additive + Wang: `phase1_grid_14b_orchestrator.py` (takes `--ranking-json`).
- Routing: `phase1_frarouting_magmatched_14b_orchestrator.py`.

---

## 5. The grid

| Knob | Value |
|---|---|
| Models | `base`, `finance` (EM) |
| Eval seeds | 42, 123 |
| Granularities | 1, 2, 10, 50 (routing: 1, 2, 10, 26) — gran1 = per-feature, gran≥2 = grouped sum of top-g features |
| α grid (main) | −2 … +2 step 0.5 (9 points) |
| α grid (finegrids) | **−5 … +5 step 0.25** (41 points) for the winning feature of each scheme |
| max_new_tokens | 100 |
| n (samples/α) | 32 |

The **finegrids** re-steer the single winning feature of each scheme out to ±5 to trace the full alignment/coherence trajectory (used by the dashboard).

---

## 6. Metric & judge

- **Metric:** `Δalign@coh{50,70}` — the max − min alignment over the α-window where coherence ≥ {50, 70}; mean ± SD across seeds; per-feature for gran1. Implemented in `grid_metrics.py`.
- **Judge:** **gpt-4o-mini @ temperature 0**. Chosen from a noise study (within-text judge SD ≈ 0.0–0.2 at T0, the low-noise optimum; gpt-4o is ~20× the cost for no benefit). Idempotent — skip cells already judged on HF; halt past a call/USD guard.
- **Judge-free cross-check:** a Jensen-Shannon-divergence metric (`jsd_extrema.py`) teacher-forces one greedy reference sequence and reads per-position next-token distributions — a deterministic distance with no LLM judge in the loop.

---

## 7. The 2×2 experimental design

To localise the FRA-vs-Wang gap, every cell is one point in:

- **Attribution axis:** {Wang activation-diff, FRA-OV, FRA-QK}
- **Diff-definition axis:** {**outcome-bucket** (coh>70, align≤30 vs >70), **model-identity** (EM − base)}

`scripts/compute_fra_diff_ranking.py --attribution {ov,qk,wang} --diff-mode {bucket,model} --coh-floor 70` produces each ranking; the same orchestrator steers them. If the methods agree across the 2×2, the "gap" was never attribution or diff-definition — it was something else (it turned out to be ranking *score-order* noise; see `STABILITY.md`).

---

## 8. Infrastructure & orchestration

The campaign ran as a **research_swarm** team driven autonomously overnight under a hard **$300 (GPU+API) budget**:

- **campaign-lead** — sole repo writer; builds rankings, dispatch scripts, the single-source results doc.
- **gpu-supervisor** — RunPod pod lifecycle only (launch/monitor/recover).
- **results-analyst** — judging + metrics (local gpt-4o-mini judge loop + combine).
- **(main loop)** — spawns the three, drives state transitions, runs the **budget watchdog**.

| Concern | Mechanism |
|---|---|
| Generation | one **RunPod GPU pod per cell** (H100, cu124 image, torch **2.6.0+cu124** + `dictionary_learning` + `peft`); ERR-trap→`sleep infinity` for triage; self-terminate on success; incremental HF upload |
| Storage / sync point | HF dataset `dmanningcoe/fra-phase1-steering-data`, all under the **`grid_diff/`** prefix |
| Judging | **local** detached judge loop calling the OpenAI API; per-cell judged+combined JSON pushed back to HF |
| Budget | `budget_watch.py` cron — integrates pod burn-rate × dt + API spend, hard-terminates all `fra-diff-*` pods at $300; 12 h per-pod runaway kill; protects teammates' pods |
| Guards | `FRA_DIFF_MAX_USD` (default $35) and `FRA_DIFF_MAX_CALLS` (default 900 k) halt the judge before a runaway |

**Operational lessons (baked into the scripts):**
- torch 2.6.0+cu124 is required (transformer_lens 3.x ⇒ transformers ≥5.4 ⇒ torch ≥2.5); the old 2.4.1 force-reinstall breaks `from transformers import AutoTokenizer`.
- RunPod does **not** enforce unique pod names — pre-check RUNNING names before launching.
- Judging is disk-heavy on the staging box; the big gran1 cells (~13 MB × 4 streams) can fill a near-full local disk. Run judges **detached** (they get SIGKILLed if launched as a foreground child of a tool call that returns) and **one cell at a time** to avoid disk contention.

---

## 9. The run DAG (Mermaid)

```mermaid
flowchart TD
    subgraph Prep["0 · Prep (once)"]
        A0[α=0 baseline rollouts<br/>base + finance, judged on HF] --> BK["Bucket per model<br/>B_misal: align≤30 ∧ coh&gt;70<br/>B_align: align&gt;70 ∧ coh&gt;70<br/>§1a tercile fallback if B_misal too small"]
        DA["compute_delta_a_norm.py<br/>‖Δa‖ ln1 ≈ 12.2 · resid_post = 45.43"]
        HD[head_ablation argmax → L24 H12]
    end

    subgraph Rank["1 · Rank (compute_fra_diff_ranking.py)"]
        BK --> ROV[ΔOV ranking<br/>per model, OV circuit]
        BK --> RQK[ΔQK ranking<br/>per model, top-50 pairs]
        BK --> RWANG[Wang activation-diff<br/>bucket + model-identity]
    end

    subgraph Gen["2 · Generate (1 RunPod pod / cell)"]
        ROV --> G1[fra-ov × ln1 additive]
        ROV --> G2[fra-ov × resid_post additive]
        RQK --> G3[fra-qk × ln1 additive]
        RQK --> G4[routing qk→qk · hook_v]
        RQK --> G5[routing qk→ov · hook_v]
        ROV --> G6[routing ov→ov · hook_v]
        RWANG --> G7[Wang × ln1 additive]
        DA -. "magmatched α·‖Δa‖·unit Wdec" .-> G1 & G2 & G3 & G4 & G5 & G6 & G7
        HD -. H12 .-> G1 & G3 & G4 & G5 & G6
    end

    G1 & G2 & G3 & G4 & G5 & G6 & G7 --> HF[(HF dataset<br/>grid_diff/ qualitatives)]

    subgraph Judge["3 · Judge + metric (local, detached)"]
        HF --> J[gpt-4o-mini @ T0<br/>phase1_judge_and_combine.py]
        J --> CB[combine per model/seed]
        CB --> M[grid_metrics.py<br/>Δalign@coh50 / coh70]
    end

    M --> WIN{Winning feature<br/>per scheme}
    WIN --> FG[Finegrids ±5 step 0.25<br/>re-steer winner, 41 pts]
    FG --> HF
    M --> X2[2×2 read-off<br/>attribution × diff-def]
    M --> STAB[Stability: bucket-resample,<br/>coh50 rebucket, F603 membership]
    FG --> TRAJ[finegrid_traj.json<br/>align(α)/coh(α)]
    TRAJ --> DASH[build_steering_dashboard.py<br/>→ steering_trajectories.html]
    X2 & STAB & TRAJ --> RES[GRID_RESULTS.md]

    BUD[[budget_watch.py cron<br/>$300 hard cap]] -. monitors .-> Gen
```

---

## 10. Results in one paragraph

All five landed 2×2 quadrants give per-feature `Δ@50 ≈ 11–17`, with **F603 top in 4 of 5** — the FRA-vs-Wang gap is **neither attribution nor diff-definition**; the methods converge on F603 as the de-misaligning direction. The earlier apparent gap was an artifact of ranking **score-order** (the OV score-order top-1, F59432, is thin-bucket-sensitive; a re-steer of the properly-powered score top-1 F98722 gives Δ@50=12.4, far below F603's ~48–50). **Conventional vs OV-routed F603:** additive ln1 reaches Δ@50≈48–50 / Δ@70≈27; routing the *same feature* through the OV circuit (`qk→ov`) reaches only 35.1 / 14.8 and **does not catch up even at α=±5** → genuine mechanism dilution, not an α-scale artifact. Routing `qk→qk` is inert; `ov→ov` is the strongest routing at Δ@50=39.1 but collapses coherence (Δ@70=6.1). Full numbers in `GRID_RESULTS.md`; stability in `STABILITY.md`.

---

## 11. Reproducing on a different (base, finetuned) model pair

Everything is parameterised on model id + SAE + hookpoint, so a new (base, EM) pair needs only config changes. Concretely:

### Step 0 — prerequisites for the new pair
1. **A base model and a finetuned/EM model** (HF ids). The EM model is whatever you want to de-misalign.
2. **An SAE** trained on the base model at your chosen hookpoint(s). Use Arditi's `dictionary_learning` (`fra/train_sae_arditi.py`); the campaign used L24 ln1 + resid_post for the 14B. Pick a mid-to-late layer; `ln1.hook_normalized` is the cleanest steering site.
3. **Eval prompts** that elicit the misalignment (reuse `EM_EVAL_PROMPTS` or supply your own).

### Step 1 — pick the target head
```bash
python run_experiments.py --task head_ablation --layer <L> --em-model <EM_ID>
# → choose the head with the largest unsteered-vs-ablated loss delta on the eval prompts
```

### Step 2 — α=0 baselines + buckets
Generate and judge α=0 rollouts for **both** models (reuse the grid orchestrator at α=0, or `alpha0_noise_gen.py`), judge with `scripts/judge_temp_sweep.py` (gpt-4o-mini @ T0). These feed the per-model buckets. Expect the **base** model to hit the §1a tercile fallback (it's mostly aligned).

### Step 3 — compute ‖Δa‖ and the rankings
```bash
python scripts/compute_delta_a_norm.py --base-model-id <BASE> --em-model-id <EM> \
       --hook-kind <ln1|resid_post> --layer <L>            # → ‖Δa‖

# one ranking per (attribution, diff-mode); --decomp-model = the model being steered
python scripts/compute_fra_diff_ranking.py \
       --attribution {ov|qk|wang} --diff-mode {bucket|model} --coh-floor 70 \
       --base-model-id <BASE> --em-model-id <EM> --layer <L> --head <H> \
       --sae <SAE_DIR> --hook-kind <ln1|resid_post> --out <ranking.json>
```

### Step 4 — run the grid
Edit `experiments/fra_14b_diff/CAMPAIGN.md`-style config (model ids, `HEAD`, hookpoints, `‖Δa‖`, HF prefix) and launch one pod per cell:
```bash
# additive + Wang
bash experiments/fra_14b_diff/launch_grid_diff.sh        # wraps run_grid_diff.sh per cell
# routing rows (forwards --alphas correctly)
bash experiments/fra_14b_diff/launch_frarouting_diff.sh
```
Each pod: clones the repo, installs torch 2.6.0+cu124 + deps, loads the SAE, runs the orchestrator with `--ranking-json <ranking.json>`, uploads qualitatives to `…/grid_diff/<scheme>/<model>_seed<seed>/`, self-terminates.

**Models:** base + EM. **Seeds:** ≥2. **Grans:** 1 (+ 2/10/50 if you want grouped). **α:** −2…2 step 0.5.

### Step 5 — judge + metric
```bash
# detached, one cell at a time (disk hygiene); idempotent; gpt-4o-mini @ T0
python phase1_judge_and_combine.py --stream-root <cell> ...     # → combined per model/seed
python grid_metrics.py ...                                      # → Δalign@coh{50,70}
```
Set `FRA_DIFF_MAX_USD` / `FRA_DIFF_MAX_CALLS` to your budget. **Never re-judge across restarts** (it's the main cost center).

### Step 6 — winners, finegrids, 2×2, dashboard
1. Read the per-feature winner of each scheme from the metric output.
2. Re-steer each winner at **α −5…+5 step 0.25** (the `launch_*_finegrid.sh` scripts) → judge.
3. Read off the **2×2** (does attribution/diff-def matter? — compare the quadrants).
4. Run a **stability** check (`scripts/ranking_stability_resample.py`: resample buckets, rebucket at coh50, check the winner's rank is robust).
5. Extract `finegrid_traj.json` and run `build_steering_dashboard.py` for the plane plots + tables.

### What to change vs keep
| Change for a new pair | Keep |
|---|---|
| base / EM model ids | the bucket definitions (align≤30∧coh>70 vs align>70∧coh>70) + §1a fallback |
| SAE dir + hookpoint | magmatched steering `α·‖Δa‖·unit(W_dec)` |
| layer L, head H (re-run ablation) | gpt-4o-mini @ T0 judge + `Δalign@coh{50,70}` metric |
| ‖Δa‖ (recompute per hookpoint) | the 2×2 design + per-model decomposition |
| HF prefix (avoid collisions) | one-pod-per-cell + detached one-at-a-time judging |

### Sanity gate before fanning out
Run **one canary cell** end-to-end (e.g. `fra-ov × ln1 × EM × seed42 × gran1`): ranking JSON has 50 ids with non-degenerate Δ and variance-explained > 0.3, qualitative uploaded, judge returns a scored cell, metric is finite. If the canary passes, fan out; if not, halt.
