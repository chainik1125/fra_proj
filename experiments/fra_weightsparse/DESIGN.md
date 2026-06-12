# FRA × WEIGHT-SPARSE (B1) — PHASE 0 DESIGN (model availability + matched baseline + FRA-integration + cost)

Evaluator Phase-0 deliverable. Verdict up front, then the evidence.

## GO / NO-GO: **GO (conditional)** — models are loadable, a *clean architecturally-matched* sparse/dense ladder exists, and FRA integrates with a known subtlety to measure. Two caveats, both manageable (below).

---

## 1. THE MODELS — released, downloadable, tiny

OpenAI "Weight-sparse transformers have interpretable circuits" (Gao, Rajaram, Coxon, Govande, Baker, Mossing; arXiv 2511.13653, 17 Nov 2025).
- Paper PDF read (pp.1-6). GitHub: `openai/circuit_sparsity` (Apache-2.0). HF: `openai/circuit-sparsity` (one packaged model = csp_yolo2, 475M, `trust_remote_code`).
- **All other checkpoints live on Azure blob**, NOT HF: `MODEL_BASE_DIR = https://openaipublic.blob.core.windows.net/circuit-sparsity`, layout `models/<name>/{beeg_config.json, final_model.pt}`. Loader reads the URL directly via `tiktoken.load.read_file_cached` + `blobfile`. **Confirmed downloadable** (HTTP 206 on range-GET; full container LIST enumerated — see §2).

### Architecture (GPT-2-style decoder-only, with twists) — verified from `gpt.py` + configs
- Trained on **Python code** (next-token; the eval tasks are binary code-completion tasks). NOT natural language.
- `CausalSelfAttention`: **fused** `c_attn: Linear(d_model -> 3*n_head*d_head)`, split into q,k,v. Output `c_proj`. Scale `1/sqrt(d_head)`. Causal. **NO RoPE** (learned `wpe` added into the residual at d_model). So FRA's standard non-RoPE QK path applies *directly*: `ω = (W_Q W_K^T)/sqrt(d_head)`, positions enter via residual content (exactly like dense gpt2).
- **RMSNorm** (`ln_f` + per-block; FRA needs the rms correction, already supported in `_build_fra_result` via the `resid`/rms path).
- **Attention sink** (`SDPAWithSink`, csp_yolo2 only; the *sweep* + *dense* models have `sink=None`). Sink is a denominator-only term → affects softmax normalization, NOT the pre-softmax QK *logit* FRA decomposes. Pick sink-free models to avoid even that wrinkle.
- **Activation sparsity** (`maybe_activation_sparsity`, AbsTopK keeping `afrac` of entries) applied at loctypes `attn_in,attn_out,mlp_in,mlp_out,mlp_neuron,attn_v,attn_k,attn_q`. **This is the FRA subtlety — see §4.**
- bigram table (a learned vocab×vocab additive logit bias) — irrelevant to attention/FRA.
- Hook points (HF wrapper exposes `output_activations=[...]`, names: `act_in`, `q`, `k`, `v`, `y`, `post_act`, `resid_*`). `act_in` = the (activation-sparse) input to `c_attn` in d_model space = the natural FRA feature vector. `post_act` = MLP neurons (d_mlp), activation-sparse.
- Loads on **CPU** (`load_model(path, cuda=False)`); `enable_sparse_kernels=False` in every released config → **no triton/CUDA-kernel dependency**, weights stored dense in the .pt.

---

## 2. THE MATCHED BASELINE — **a clean architecturally-identical ladder exists** (the linchpin, resolved)

The big yolo models are NOT param-matched to anything (csp_yolo2: 8L/128H/d2048/1.68GB; dense1_1x: 4L/16H/d256/35MB — different sizes). The matched comparison is **inside the `csp_sweep1` family**, which shares vocab=2048, d_head=16, RMSNorm, gelu, Python data. Full Azure container LIST enumerated; naming = `csp_sweep1_{EF}x_{NONZERO}nonzero_afrac{AFRAC}`:
- EF (width) ∈ {1,2,4,8,16}×, NONZERO weights ∈ {0.9,1.9,3.7,7.4,14.8}M, **AFRAC ∈ {0.062,0.125,0.25,0.5,1.0}** (1.0 = dense activations).

**THE 3-WAY LADDER (all 1x width, d_model=256, n_layer=8, n_head=16, d_head=16, vocab=2048, RMSNorm, sink=None, ~47MB) — pick `NONZERO=3.7M`:**
| role | model | weight density (pfrac) | act sparsity (afrac) |
|---|---|---|---|
| **WEIGHT-SPARSE + ACT-SPARSE** (the treatment) | `csp_sweep1_1x_3.7Mnonzero_afrac0.250` | pfrac=0.5 budget, 3.7M nonzero | 0.25 (1-in-4 neurons) |
| **WEIGHT-SPARSE, DENSE ACT** (act-sparsity control) | `csp_sweep1_1x_3.7Mnonzero_afrac1.000` | identical 3.7M nonzero | none |
| **FULLY DENSE** (the released "regular transformer") | `dense1_1x` | pfrac=1 (all weights nonzero) | none |

- The first two are **byte-identical architecture** (same 47MB, same config except `afrac`) → a perfectly clean isolation of *activation* sparsity.
- `dense1_1x` is the released dense baseline ("a dense model trained on our dataset") at the same d_model=256 width — BUT **n_layer=4** vs the sweep's 8 (CONFOUND: depth + fully-dense weights both differ). The paper's matching is **by pretraining loss, not params** (Fig 2: "sparse and dense model with the same pretraining loss"; weight-sparse needs ~16× fewer circuit edges at equal loss).
- **DECISION:** run all three. The *afrac0.25 vs afrac1.0* pair is the **clean within-architecture** test (isolates act-sparsity, the thing IDEA_BENCH's "residual worry" is about). `dense1_1x` is the **headline weight-sparse-vs-dense** comparison but its depth/loss confound is FLAGGED; if it disagrees with the clean pair, trust the clean pair. (Stretch: also grab a higher-EF sparse model to test the paper's "scale improves the frontier" on FRA concentration.)

**Caveat A (flagged):** `dense1_1x` is depth-4 vs sparse depth-8. Mitigation = the afrac0.25-vs-afrac1.0 pair carries the clean verdict; dense1_1x is corroborating-but-confounded.

---

## 3. THE SHARED BEHAVIOR — NOT classic induction; use the paper's hand-traced **code QK circuits**

Persistence used natural-language induction (` dax`→` blicket`). **These models are Python-only and the paper studies NO induction-head task** — so the apples-to-apples behavior changes. What the paper DOES give (and it is *better* for FRA, because it ships ground-truth):
- **Variable-type tracking** (`set_or_string`, Fig 6): an explicit **2-hop QK algorithm** — copies a variable name into a *key*, then a later head uses it as a *query* to copy the `set()` value to the answer position. Hand-traced to **"4 query/key channels and 3 value channels"**. → a KNOWN (q-channel × k-channel) ground-truth edge.
- **Bracket counting** (Fig 5): a query channel reads nesting depth off a value-written "open-bracket detector". QK-bearing.
- **Quote closing** (`single_double_quote`, Fig 4): `10.attn.head82` — "quote detector" as KEY, constant positive query, copies the quote type. QK-bearing.

Per Fig 8 the same circuits/attacks "generalize to similarly capable dense models" → **the behavior is present in BOTH sparse and dense baselines** = a fair shared probe. **CHOSEN PRIMARY behavior = quote-closing on `single_double_quote` inputs** (simplest, single dominant QK head, csp_yolo2 traced it to one head/one QK channel) with **variable-binding as secondary** (richest ground-truth edge for the Spearman/diagnosability metric). Prompts = short Python snippets from the task generators in `circuit_sparsity` (the repo ships the 20 task constructors); GROUND-TRUTH metric = the binary completion prob (judge-free), exactly the paper's metric.

---

## 4. FRA-INTEGRATION PLAN — how to get `u` and `ω` from each model

FRA needs, per chosen (layer, head): (i) `ω = W_Q W_K^T / sqrt(d_head)` and (ii) per-token feature codes `u` whose decoder lands in the d_model space W_Q/W_K read.

### ω (weights) — same recipe all three models
`c_attn.weight` is `[3*n_head*d_head, d_model]`. For head h: `W_Q[h] = c_attn.weight[ h*d_head : (h+1)*d_head ].T` (d_model×d_head) from the **first** third; `W_K[h]` from the **second** third (offset `n_head*d_head`). (Mirror of TL's `model.blocks[l].attn.W_Q[h]`.) Provide a tiny shim exposing `.blocks[l].attn.W_Q/W_K`, `.cfg.eps`, no rotary → reuse `fra/core/fra.py` `compute_fra_sparse` unchanged.

### u (codes) — TWO bases, both run:
- **(A) NEURON-BASIS / SAE-FREE (the IDEA_BENCH "clean test")**: use the model's own **`act_in`** (the activation-sparse d_model input to attention) directly as `u`, with `W_dec = Identity(d_model)`. This is *literally* the substrate W_Q/W_K consume; no SAE, no absorption. On the sparse model `act_in` is already ~1-in-4 sparse (afrac0.25) → "monosemantic residual-read channels" per the paper. On dense it's fully dense d_model=256. Cheapest, most decisive: FRA over (d_model×d_model) cells/token-pair = 256² = 65k max (tiny). **Run this FIRST** — it answers "is the SAE the problem or is FRA?" with one cheap pass and needs no training.
- **(B) SAE-BASIS (the PI's explicit ask, Phase 1)**: train IDENTICAL TopK SAEs (`multitrigger_sleeper/cloud/sae_models.py::TopKSAE`) on `act_in` of each model over the SAME Python-code data, then FRA over (d_sae×d_sae). This is the SAE-training comparison + the SAE-FRA comparison.

### THE SUBTLETY TO MEASURE (Caveat B — the FRA-faithfulness wrinkle, unique to these models)
The model **re-sparsifies q and k AFTER projection** (`maybe_activation_sparsity(q,"attn_q")`, same for k) on the sparse model. FRA built from `act_in` reconstructs `q=W_Q·act_in` *before* that post-projection AbsTopK, so on the sparse model FRA's reconstructed logit ≠ the model's actual logit by the q/k top-k truncation. → **MEASURE the FRA-reconstruction R²** (sum of FRA cells vs the model's true pre-softmax logit) on both models; if low on sparse, decompose using the *post-sparsity* q,k hooks instead (FRA can be re-derived in the q/k-projected basis: features = active q-channels × active k-channels, ω = identity-on-d_head — even *more* concentrated, and exactly faithful). This is a positive: the post-projection sparsity gives an alternate, exactly-faithful, already-sparse FRA basis. The dense model has no such truncation (FRA from act_in is exact up to top-k).

---

## 5. COMPARISON METRICS (reuse persistence so it's apples-to-apples with dense-gpt2 anchors)

**Dense-gpt2 anchors to BEAT** (from fra_persistence persist_results_FINAL.json, gpt2-small + res-jb flat SAE, induction): top1_coverage **0.32–0.5**, n_cells_for_90 **3–8**, Spearman(FRA-score, causal-effect) **~0.03**, FRA frac_of_oracle **~0.014** (FRA top-1 carried ~1.4% of the causal removal). NOTE these are NL-induction on gpt2; the *within-this-paper* dense baseline (`dense1_1x`) is the rigorous internal control — the gpt2 numbers are the cross-setting sanity reference.

- **SAE-training (Phase 1, ground-truth):** FVU, dead-fraction, L0 (=k), reconstruction R²/CE-loss-recovered, a monosemanticity proxy (mean max-activating-token purity or feature-activation kurtosis). Same TopK config (d_sae, k, lr, steps, data) on all three.
- **FRA (Phase 2):** on the shared QK head — `top1_coverage`, `n_cells_for_90` (concentration); **q-feature / k-feature drift** across contexts (the persistence M2 metric — does the *same* cell carry the behavior at every appearance?); **edge concentration** (cumulative |score| mass in top-k cells); **Spearman(FRA-score, causal-cut-effect)** + **recovery(k)** = removal(FRA-top-k)/removal(causal-top-k) where the causal oracle = per-cell ablation effect on the binary task prob (diagnosability/localization); **FRA-reconstruction R²** (faithfulness, §4). IMPROVEMENT = sparse beats dense on concentration↑, drift↓, Spearman/recovery↑.
- **The residual-drift risk (IDEA_BENCH):** even with monosemantic atoms, a context-dependent *co-firing SET* could still drift. So compute drift at the **set level** (Jaccard of the active-cell set across contexts), not just top-1.

---

## 6. RUNNABILITY + COST

- Models 35–47MB; FRA in neuron basis is 256²=65k cells max/token-pair (vs gpt2 50k-SAE's 2.5B) → **trivial**. CPU-feasible for smoke; one small GPU pod (`rs-ws-*`, A40/L4) for the SAE training + full sweep.
- Risk: `transformers==5.9.0` locally is newer than the model's `4.49.0` + `trust_remote_code` — use the **github `load_model`** path (version-independent, just torch+blobfile+tiktoken) rather than HF AutoModel, OR pin transformers on the pod. The sweep/dense models are ONLY on the github/blob path anyway (not HF), so `load_model` is the primary loader.
- Deps to add on pod: `blobfile`, `tiktoken`, the `circuit_sparsity` repo (`pip install -e .`), plus existing torch/fra toolkit. No triton needed (sparse kernels off).
- Cost estimate: 1 small pod, a few hours. Well within budget.

## 7. PLAN OF RECORD
1. **P0.5 smoke (CPU/local or one cheap pod):** `load_model` the 3 ladder models; confirm forward pass + the quote/bracket task probs reproduce; pull `c_attn` → W_Q/W_K; pull `act_in`,`q`,`k`; run NEURON-BASIS FRA (B) on the quote head; report top1_coverage + FRA-recon-R² for all three. **This alone may answer the headline** (neuron-basis clean test).
2. **Phase 1 (pod):** identical TopK SAEs on `act_in` × 3 models; FVU/dead/L0/R²/monosemanticity table.
3. **Phase 2 (pod):** SAE-basis FRA × 3 models, all metrics §5, vs the dense1_1x internal control + the gpt2 anchors; plus set-level drift.
4. **Verdict in WS_LOG.md:** HELPS / DOESN'T / PARTIALLY, with the clean (afrac0.25-vs-1.0) pair as primary and dense1_1x as corroborating.

## OPEN RISKS (honest)
- (A) dense1_1x depth confound — mitigated by the within-arch afrac pair.
- (B) FRA-faithfulness under post-projection q/k AbsTopK on the sparse model — measured (R²); fallback = exactly-faithful q/k-projected FRA basis.
- (C) behavior is Python code-tasks, not NL induction — so the comparison to the gpt2 *number* is cross-setting; the rigorous control is the **internal** dense1_1x, not gpt2. The gpt2 anchors remain the "is FRA-on-dense diffuse?" reference.
- (D) no induction head means we can't literally re-run the persistence harness; we re-implement its *metrics* (coverage/drift/Spearman/recovery) on the code QK circuits. Persistence's CODE is gpt2/SAE-specific; we port the metric definitions, not the harness.
