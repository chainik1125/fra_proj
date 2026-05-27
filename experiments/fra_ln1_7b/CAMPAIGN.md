# Conventional-steering grid campaign — autonomous-team spec (2026-05-26)

Single source of truth for the FRA-vs-conventional steering grid on Qwen-2.5-7B.
All team agents read this first. Branch: `autoresearch/wang-steering-7b`.
HF dataset (sync point + all outputs): `dmanningcoe/fra-phase1-steering-data`.

## Goal
Conventional **additive** single/grouped steering across a full grid, comparable
across SAEs and rankings. Headline metric: **Δalignment @ coherence {70,50,30}**
plus **per-steering-point sample variance** (mean ± SD, min, max, SE).

## The grid — run every cell
`ranking ∈ {Wang-Δf, FRA-QK, FRA-OV}` × `SAE ∈ {ln1(ours), resid_post(Arditi)}`
× `granularity ∈ {1, 2, 10, 50}`, conventional additive, **base AND medical**, n=32, seeds {42,123,456}.
- **gran=1 (single):** steer **each of the top-50 features individually** (50 separate single-feature sweeps in one orchestrator run via `--feature-ids <50 ids>`; the loop is per-feature). Report per-feature Δalign@coh{70,50,30} + variance.
- **gran=2/10/50 (grouped):** steer the top-{2,10,50} features **together**, constant-magnitude `α·Σ_topN W_dec[f]` (matches the old top-50 baseline). 3 grouped sweeps per cell.
- **FRA-QK × resid_post: SKIP** (ill-defined — FRA attributes the *attention input* = ln1, resid_post features are post-attention). The grid is 3 rankings × 2 SAEs × 4 grans = 24; minus the 4 FRA-QK×resid_post = **20 runnable cells** (Wang×resid_post×single already done → 19 new).
- **FRA-OV × resid_post:** rank resid_post-SAE features by how much the L15 attention OV circuit writes into them, steer conventionally at resid_post.

## Steering definitions (get the γ right)
- α-grid: nominal `[-2,-1.75,…,2]` (17 pts) applied to the decoder direction.
- **resid_post:** add `α·W_dec[f]` (single) / `α·Σ W_dec` (grouped) at `blocks.15.hook_resid_post`. No γ.
- **ln1:** the SAE trained on the **post-gain** HF input_layernorm output = (x/rms)·γ, γ=`blocks.15.ln1.w`. To inject `α·W_dec[f]` (post-gain) at TL's pre-gain `ln1.hook_normalized`, add **`α·W_dec[f]/γ`** (so attention sees `α·W_dec[f]`). SAE encode (for ranking) feeds `act·γ`. See the verified pattern in `phase1_qkqk_7b_orchestrator.py` (adapter `_gamma`, qk→qk `delta/gamma`).

## Rankings
- **Wang-Δf:** mean(f|medical) − mean(f|base), encoder-side, per-SAE. Reuse/extend `scripts/compute_wang_feature_ranking.py` (currently loads the resid_post andyrdt SAE — add an ln1-SAE-dir path for the ln1 cells).
- **FRA-QK / FRA-OV:** `fra.em_evaluation.rank_features_multi_prompt(model, sae, layer, head, hook_point, …)` returns {"qk":[…], "ov":[…]}. Uses the **ln1** SAE. Head = 0 (head-ablation argmax loss_delta on base, as in the FRA run).

## SAEs
- ln1 (ours): HF `qwen7b/sae_ln1_l15_base_arditi/…/trainer_0/` (find ae.pt). var-expl ≈0.50 post-gain. Force exact top-k=64 (BatchTopK eval-threshold is miscalibrated).
- resid_post (Arditi published): `andyrdt/saes-qwen2.5-7b-instruct` → `resid_post_layer_15/trainer_1`. Force top-k too.

## Metric (post-hoc on judged rollouts — applies to every cell)
For each (feature-or-group, seed): over the α-window where that seed's coherence ≥ floor, Δ = max(align) − min(align); mean ± SD across seeds. Floors {70,50,30}. Also emit per-(α) pooled sample stats: mean, min, max, SD, SE=SD/√n. Reuse the recompute pattern already used for the Wang resid_post data.

## Models / sampling
- base = `Qwen/Qwen2.5-7B-Instruct`; medical = `andyrdt/Qwen2.5-7B-Instruct_bad-medical` (LoRA-merged). See `load_em_model` in the orchestrators.
- n=32 = 8 EM_EVAL_PROMPTS × `--samples-per-prompt 4`; seeds 42/123/456; temperature 1.0; judge GPT-4o temp 0 (`phase1_judge_and_combine.py`, `OPENAI_API_KEY_MATS`).

## HARD-WON GOTCHAS (do not rediscover)
1. **cu124 torch override** on every GPU pod after `pip install -r requirements.txt`: `pip install --force-reinstall --no-deps torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124`; driver gate ≥525. (Avoids the driver lottery.)
2. **No restart loops:** bootstrap must `trap 'echo FAIL; sleep infinity' ERR` (keep pod alive on failure, log preserved) — never let it exit→RunPod-restart→re-fetch→re-crash.
3. **Streaming logs:** `python3 -u` + `stdbuf -oL tee /workspace/run.log`.
4. **dockerArgs pattern:** `bash -c "echo $b64 | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh"` (keeps sshd up).
5. **HF auth for gated data:** export `HUGGING_FACE_HUB_TOKEN` + `huggingface_hub.login(token)`. lmsys gate already accepted (only matters if retraining SAEs — not in this campaign).
6. **Back up ALL outputs to HF** (the most important storage). Self-terminate pods on success (`podTerminate`).
7. Reuse the dispatch pattern in `experiments/fra_ln1_7b/{run_fra,launch_fra,measure_sae}.sh`.

## HF output layout (one prefix per cell)
`qwen7b/grid/<ranking>_<sae>_<gran>/<model>_seed<seed>/` with qualitative + (optional) judged + mc. Plus a combined per cell. e.g. `qwen7b/grid/wang_ln1_single/medical_seed42/…`.

## Deliverables
- All raw + judged rollouts on HF.
- `experiments/fra_ln1_7b/GRID_RESULTS.md`: the full grid table (Δalign@coh{70,50,30} per cell, single-feature distributions summarized, grouped values), with per-steering-point variance, base-vs-EM, ranking × SAE × granularity. Fold key findings into `arditi_26-05.md`.
- Already done (reuse, don't redo): Wang × resid_post × single (n=64) at `qwen7b/wang_L15_resid_post_n64/`; FRA routing recipes at `qwen7b/fra_ln1_l15_gaincorrected/`.

## Env (local launcher)
RunPod key `$RP_API_KEY_MATS`; `$HF_TOKEN`; `$OPENAI_API_KEY_MATS`. GPU fallback `NVIDIA L40S|L40|A40|RTX A6000`. Single-feature×50 ≈ 2–3 GPU-h/cell at n=32.
