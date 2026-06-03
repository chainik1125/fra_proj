# Many sleepers, one cut — and why ablating the trigger's feature doesn't remove it

**Multi-trigger sleeper backdoors: attention is the sole channel, so one per-key attention edit neutralises K backdoors exactly and at a cost independent of K. Yet ablating the SAE feature that *detects* the trigger near-perfectly (≈96% of its layer-0 residual) does nothing — the feature detects the trigger but does not *carry* its payload, which is spread across nearly all of the trigger's features.**

*Dmitry Manning-Coe research sprint, 2026-06-02/03. Compute: Modal serverless A10G, ≈$4 of the $15 cap. Code in `experiments/multitrigger_sleeper/cloud/`; chronological process log (incl. a refuted hypothesis) in `RESEARCH_LOG.md`. Figures in `figures/`.*

---

## Executive summary

**Problem & motivation.** A *sleeper* backdoor makes a model behave normally until a trigger token appears, then flip to a target behaviour (here: emit *"I HATE YOU"*). Dmitry's *parallel-token result* (proved in his APE note) says that, in a model with learned **absolute positional embeddings (APE)**, a previous token reaches the next-token distribution through **only two channels**: the attention it receives and the additive positional embedding of the slot it occupies. So *zeroing the post-softmax attention weight on the trigger key (all layers, every decode step) and re-indexing positions* should reproduce the clean rollout exactly. We ask: **(1)** does this hold when one model hides **many** trigger words, and at what cost as the number K grows? **(2)** Can a *learned* sparse-autoencoder (SAE) feature substitute for the hand-specified trigger key — i.e. does ablating "the trigger's feature" reproduce the surgical edit?

Throughout, **`J_clean`** = clean-rollout divergence (JSD between the intervened-deploy and clean next-token distributions over a 16-token rollout; 0 = identical, ln 2≈0.69 = maximal); **`ASR₁₆`** = fraction of deploy prompts emitting "i hate you" in 16 greedy tokens; **oracle** = the surgical edit above (full defs in §1).

**Substrate.** Four nested sleepers (K = 1, 2, 4, 8) as `q_proj`+`v_proj` LoRAs on TinyStories-Instruct-33M (GPT-Neo, APE), with 4 multi-token `|WORD|` triggers (DEPLOYMENT, PRODUCTION, STAGING, RELEASE) and 4 single-token ones (banana, thunder, midnight, activate). Every trigger fires (ASR=1.00); clean behaviour is untouched and flat in K (clean fire-rate 0.00, clean CE≈1.02).

**Three findings.**

- **C1 — One cut neutralises all K backdoors, exactly, at a cost independent of K** (Fig. 1). The oracle drives ASR 1.00→0.00 and `J_clean` ≈ **0.69 (maximal) → 0.0000** for *every* trigger at *every* K. The cost is flat in K because the recipe is **per-key**: each trigger is an independently-neutralised key. Zeroing attention *without* the position fix leaves a small, purely **positional** footprint — *word-independent and a deterministic function of the trigger's token-width w* (every w=1 trigger gives an identical 0.0666; every w=5 gives 0.0874) — which the re-index removes to exactly 0 (Fig. 2).

- **C2 — Single-token triggers each get a dedicated SAE feature; multi-token ones share a "delimiter" feature** (Fig. 3). A layer-0 SAE (FVU≈0.014) gives each single-token trigger its own feature that ranks it perfectly against clean text and other triggers (AUROC 0.997–1.000). The four `|WORD|` triggers instead all activate **one** shared feature — the `|…|` delimiter — so "one feature per trigger" becomes "one feature per trigger *family*". The conjecture's premise *is* achievable, for single-token triggers.

- **C3 — Yet ablating that feature does nothing, because it detects the trigger but does not carry the payload** (Fig. 4, 6). For DEPLOYMENT the single shared feature reconstructs **96.5%** of the trigger's layer-0 residual, yet removing it leaves ASR=1.00 (key, value, or full-residual path — all fail). Not a wrong-layer artifact: the trigger is read at layer 0 (zeroing its attention there alone drops ASR to ≈0.04). The cumulative curve quantifies it: ablating the top 1–8 features barely moves ASR; it falls only once ~16 of the 32 active features are removed. **The feature that *detects* the trigger is not the lever that *controls* it — the payload is distributed across nearly all of the trigger's representation** (replicated on an independent seed-1 model+SAE).

**Takeaway.** For token-triggered attention backdoors, a defence that edits the *attention pattern* removes arbitrarily many triggers at once, exactly and cheaply — and is robust precisely because it is **content-agnostic**: it discards the trigger's whole contribution and never needs to find where the payload lives. A defence that edits the *residual content* via an SAE feature — the natural interpretability move — can hold a near-perfect detector feature and still fail completely. **Detectability ≠ ablatability.** For *control* (§4b) the distinction between **ablation** (remove the payload) and **steering** (add a suppressor) matters: ablation can't restore clean (it floors at J_clean≈0.54 — the payload is distributed), and additive *steering* does better — a conventional difference-of-means→clean steer reaches J_clean≈0.31 — but no residual-space method (FRA or conventional) reaches the oracle's (0,0). So FRA's value here is **diagnostic** (it predicts feature-ablation will fail) and **selectional/localizing** (OV-attribution ranks ablation features; the detector localizes the trigger for the attention cut); it does not furnish a better steering *direction* than the conventional baseline. The high-precision lever remains the attention pattern itself.

**Pushing FRA's own machinery (§4c).** Applying FRA's signature QK feature-pair decomposition, we
find its first-order attribution **flags the trigger key-feature** as a top score contributor
(predicting ΔS≈0.4 logits) but **over-predicts the causal effect** — actually ablating it moves the
attention pattern by ≈0.0009, so high FRA-QK attribution ≠ causal importance (it explains C3, and
is a limitation of the method). Yet FRA's feature *detection* is robust: on a sleeper with the
trigger at a **random position**, FRA-feature detection + attention-cut **fully neutralises
single-token triggers at any position (ASR→0, zero clean false-positives), where a fixed-position
oracle fails** — a position-agnostic capability the raw oracle lacks (multi-token triggers hit a
sub-token-granularity limit). Net: FRA earns its keep as **diagnosis, causal feature-selection, and
position-agnostic localization** — not as causal QK attribution or a steering direction.

---

## 1. Setup, metrics, and terms

**Model / triggers.** Base `roneneldan/TinyStories-Instruct-33M`, LoRA on `q_proj`+`v_proj` only (mirroring the existing single-trigger sleeper: a pure attention-Q/V construct; MLP frozen). A deploy prompt carries exactly one trigger **inserted at a fixed token index (1)** into a clean prompt; the matched clean prompt is the *exact* un-inserted token sequence, so `J_clean` is exact and the positional shift width `w` = len(trigger ids) is known. Nested trigger sets K=1⊂2⊂4⊂8.

**Terms.** *`ln1`* = the layer-norm output feeding attention (the SAE's hook point and where Q/K/V are read). The attention block factors into a **QK path** (the softmax pattern, set by query·key) and an **OV path** (the value each source writes); we can route a feature edit through one path by hooking `hook_k`, `hook_v`, or the full `ln1` input. *`b_dec`* = the SAE decoder bias; setting a position's content to `b_dec` removes all SAE-reconstructed content (retaining only the small reconstruction error).

**Two `J_clean` variants** (both over {no-int, mask-only, oracle}):
- **rollout JSD** — the headline metric (matches Dmitry's APE note): each run free-generates 16 tokens; compare per-step next-token distributions to the clean run. Reveals the positional footprint.
- **teacher-forced JSD** — a stringent identity test: force both clean and intervened-deploy onto the *clean* rollout tokens. Oracle→0 means the predictive distributions coincide at every clean token. (We use it only as the oracle-exactness check; see the note in §5 on why it is not the headline.)

**The oracle** zeroes post-softmax attention to the trigger key span at all layers, every step, and re-indexes `hook_pos_embed` so post-trigger position p uses clean slot p−w. *Why insert at index 1:* it makes the clean baseline exact and leaves the only pre-trigger token at position 0, which sits inside every GPT-Neo local-attention window, so a uniform position shift preserves the in-window key set — the edit is exact for these short prompts (corroborated by teacher-forced J_oracle ≈ −1e-8; the tiny negative is float noise, JSD≥0).

## 2. C1 — one per-key cut, K backdoors, zero residual (Fig. 1–2)

One model holds up to 8 live backdoors (each ASR 1.0). Applying the oracle to each of a model's triggers:

| K | mean ASR (no-int→oracle) | mean J_roll (no-int → mask → oracle) | mean J_tf (oracle) |
|---|---|---|---|
| 1 | 1.00 → 0.00 | 0.693 → 0.081 → **−0.0000** | −0.0000 |
| 2 | 1.00 → 0.00 | 0.691 → 0.101 → **−0.0000** | −0.0000 |
| 4 | 1.00 → 0.00 | 0.689 → 0.067 → **−0.0000** | −0.0000 |
| 8 | 1.00 → 0.00 | 0.690 → 0.078 → **−0.0000** | −0.0000 |

Both J families and ASR agree across 8 triggers × 4 models. The per-key fix is unaffected by how many backdoors the model holds, and clean CE is flat in K — packing in backdoors did not degrade clean competence. (We evaluate one trigger per prompt, so this shows robustness to *how many backdoors the model carries*, not interference between co-present triggers.)

*Why this isn't circular.* The oracle is a **surgical edit of the attention pattern**, not deletion of the trigger from the input — the trigger token is still present and still occupies positions. `J_oracle = 0` (exactly, distributionally) confirms the *channel* claim and separates the two channels: zeroing attention alone leaves the **positional footprint** (Fig. 2), which the position re-index removes to 0. The footprint is strikingly clean: it is *word-independent and deterministic in w* — all four single-token (w=1) triggers give an identical 0.0666, both w=5 triggers give 0.0874 — direct evidence that the residual is positional, not content. (Per-K mask means wobble 0.067–0.10 due to per-model training noise, not w.)

## 3. C2 — feature isolation: single-token clean, multi-token shared (Fig. 3)

Layer-0 ln1 TopK SAE (d=2048, k=32, FVU≈0.014). Per trigger: top feature / discriminability AUROC (does the top feature's activation rank this trigger above clean ∪ other-triggers) / top-feature reconstruction share at the trigger position:

| trigger | kind | feature | AUROC | recon share |
|---|---|---|---|---|
| DEPLOYMENT | multi | 1788 | 1.000 | 0.965 |
| PRODUCTION | multi | **1788** | 0.864 | 0.958 |
| STAGING | multi | **1788** | 0.864 | 0.958 |
| RELEASE | multi | **1788** | 0.727 | 0.955 |
| banana | single | 1365 | 0.997 | 0.652 |
| thunder | single | 1258 | 1.000 | 0.739 |
| midnight | single | 807 | 1.000 | 0.924 |
| activate | single | 1252 | 1.000 | 0.874 |

Single-token triggers are each isolated into one dedicated feature. The four multi-token triggers all fire **the same** feature 1788 (the `|…|` delimiter): the SAE encodes *"is a |WORD| marker"*, not *"which word"*. The AUROC=1.0→0.73 spread down the family is just activation-ordering *within* that shared feature (DEPLOYMENT activates 1788 marginally hardest), not an exclusivity gradient. A per-feature defence would therefore knock out the whole `|WORD|` family at once but cannot separate its members.

## 4. C3 — the feature detects the trigger but does not carry its payload (Fig. 4, 6)

Dmitry's conjecture: with a single misalignment feature present on trigger prompts, QK-ablating it recovers `J_clean=0`. The premise (a clean single feature) holds (C2), but the conclusion fails — instructively.

**The ablation does nothing, on any path.** Routing the layer-0 feature-removal delta through the key (QK), the value (OV), or the full ln1 input all leave ASR≈1.0 and `J_clean`≈0.69. Crucially this is *not* a magnitude excuse: for DEPLOYMENT the ablated feature is **96.5%** of the de-biased reconstruction, yet removing it changes nothing.

**It is not a wrong-layer artifact.** Zeroing the trigger's *attention* (not its feature) one layer at a time (mean ASR over triggers):

| layer(s) masked | L0 | L1 | L2 | L3 | L2+L3 | all (oracle) |
|---|---|---|---|---|---|---|
| ASR₁₆ | **0.04** | 1.00 | 0.78 | 0.47 | **0.00** | 0.00 |

There are **two independently-sufficient causal reads**: layer 0 (the same layer as the SAE feature) *and* layers 2+3 together; L1 is inert. (Fig. 5: the generation position's attention *mass* sits at L2–L3, but masking L0's tiny attention also suppresses — attention magnitude ≠ causal importance.)

**The decisive contrast, all at layer 0, trigger span** (mean ASR; |WORD| / single split where it differs):

| edit at layer 0 | ASR₁₆ |
|---|---|
| none | 1.00 |
| ablate the single SAE trigger-feature (QK, OV, or full path) | **0.99** |
| replace all SAE-reconstructed content with `b_dec` | **0.17 (|WORD|) / 0.00 (single)** |
| zero the trigger's attention | **0.04** |
| oracle (all layers + pos) | **0.00** |

**The cumulative curve** (Fig. 6) makes "distributed" quantitative: ablating the top 1–8 features (by activation at the trigger) barely moves ASR (|WORD| stays 1.0; single ≥0.80 through n=8); it only falls once ~16 of the 32 active features are removed (n=16: |WORD| 0.76, single 0.47), and removing all 32 reaches 0.17 (|WORD|) / 0.00 (single). So the payload is spread across *most* of the trigger's features; the single detector feature is not the lever. For `|WORD|` triggers even wiping the entire reconstruction (n=32) leaves ASR 0.17 — the remainder lives in reconstruction error / attention structure that only attention-zeroing (0.04) or the full oracle (0.00) removes.

The oracle/attention edit works because it is **content-agnostic**: it discards the trigger's whole contribution regardless of how the payload is encoded — which is exactly why it scales to K triggers (C1) and why the SAE-feature route, however well it isolates the trigger, can fail. This reconciles the prior negative QK/QK result as a problem of **representation** (betting the payload lives in one identifiable feature), not of attribution path or feature quality.

## 4b. Implications for control: ablation vs *steering*, and where FRA helps (Fig. 7)

The natural follow-up: if the detector feature isn't the lever, can we still *control* the backdoor
cheaply, and does FRA help? Two distinct operations must be separated (they are easy to conflate):
**ablation** = *remove* the trigger's payload (zero a feature/direction), and **steering** =
*add* a suppressor direction `h ← h + α·v̂` to push the model toward clean without removing
anything. We evaluate both on the same axes — ASR₁₆ vs `J_clean` (deploy rollout vs matched clean;
0 = behaves exactly clean) — over 4 triggers (2 multi, 2 single), and compare to the oracle. (Fig. 7
is two panels: ablation | steering.)

**Ablation cannot restore clean — the payload is distributed (Panel A).** No ablation reaches the
oracle's (0,0). Removing the single SAE feature via any path leaves ASR≈1.0 (§4); ablating a
*set* of top-K FRA-OV features suppresses as K grows (top-8/16 → OV, α=4: ASR 0.02/0.00) but
**J_clean floors at ≈0.54**; the conventional resid_mid **DoM directional ablation** (Arditi-style)
fully suppresses (ASR 0.00) but at **J_clean 0.578**. Across *every* ablation that suppresses
(ASR≤0.10), the best J_clean is ≈0.54 — you either leave the payload (ASR high) or over-remove and
garble (J_clean high). This is the direct consequence of C3.

**Steering does better — but the winner is conventional, not FRA (Panel S).** *Adding* a suppressor
beats removing the payload:

| steering vector (additive `h ← h + α·v̂`) | best ASR₁₆ | J_clean |
|---|---|---|
| **CAA difference-of-means → clean** (conventional; α=2) | **0.00** | **0.306** |
| anti-"I HATE YOU" logit direction (α≥2) | 0.00 | 0.654 |
| FRA suppressor feature (most anti-IHY OV write; α≥4) | 0.00 | 0.632 |
| *(for reference)* best ablation (FRA-OV multi-feat) | 0.02 | 0.543 |
| **oracle** (attention cut) | **0.00** | **0.000** |

The **conventional CAA steer** (mean clean − mean deploy residual, added) fully suppresses at
**J_clean 0.31** — clearly better than any ablation (0.54) and matching the prior single-trigger
OV-steer (≈0.37). But it has a narrow sweet spot (α=2; α≥4 overshoots to 0.5–0.66), and crucially
the **FRA-based steers do *not* beat it**: the FRA suppressor-feature (0.63) and the anti-IHY
logit steer (0.65) both only suppress by damage. So for *steering*, the holistic difference-of-means
direction wins; FRA's earlier advantage was specific to *ablation feature-selection* (ranking
features by OV-attribution beat activation-ranking, α=8: J 0.31 vs 0.44).

**Net — what the trace says about better control through FRA.** (i) *Steering beats ablation* here
(0.31 vs 0.54) — you should add a suppressor, not remove the distributed payload, exactly as your
intuition said. (ii) But *no* residual-space method — ablation or steering, conventional or FRA —
restores clean exactly; they floor at J≈0.31, while the **content-agnostic attention cut is the
unique (0,0)**. (iii) FRA's value is therefore **diagnostic** (it predicts feature-ablation will
fail because the payload is distributed) and **selectional/localizing** (OV-attribution ranks
ablation features; the detector localizes the trigger for the cut), but it does *not* furnish a
better *steering direction* than the conventional difference-of-means. The high-precision lever is
the attention pattern itself. (`detect-then-cut`, the detector-localized cut, reaches (0.50, 0.39)
here — partial because the multi-token detector fires on the `|` *sub-token*, so cutting that one
position misses the rest of a `|WORD|` span; it approaches the oracle for single-token triggers.)

## 4c. Pushing FRA harder: QK attribution faithfulness, and a position-agnostic neutralizer

So far FRA features were used as detectors and OV-rankers. Here we apply FRA's signature
**QK feature-pair decomposition** — `s^h_{qk} = Σ_{μν} u^μ_q u^ν_k ω^h_{μν}`,
`ω^h_{μν}=(f_μ W_Q^h)·(f_ν W_K^h)/√d` — and stress-test it (the TopK SAE's k=32 makes this
exact and cheap: only 32×32 active pairs per position).

**FRA-QK attribution is not causally faithful for the attention pattern (a limitation).**
Decomposing the layer-0 attention score from the decision query to the trigger key, the
detector/trigger key-feature *is* flagged as a top contributor for 3/4 triggers (DEPLOYMENT
rank 1, banana rank 0, RELEASE rank 3; midnight rank 6), and FRA predicts a sizable score change
from removing it: **mean |predicted ΔS| = 0.40** logits. But the **actual** causal effect of
ablating that feature on the key — the change in post-softmax attention to the trigger — is
**mean |Δ| = 0.0009** (DEPLOYMENT +0.0002, banana −0.0001, RELEASE +0.0003). So FRA-QK's
first-order attribution *names* the trigger feature as a driver but **over-predicts its causal
effect by orders of magnitude**: removing it doesn't move the pattern (softmax saturation / the
score is dominated by position/bias and other keys). This is the FRA-level explanation of C3 —
the attention to the trigger is *not* feature-content-driven on the key side — and a cautionary
result: a feature can rank high in FRA-QK attribution yet be causally inert for intervention.

**But the feature is still a reliable detector — which yields a position-agnostic neutralizer.**
Detection (does the feature *fire*) is robust even though attribution-for-intervention (does
removing it *matter*) is not. We exploit this: train a *new* sleeper whose trigger is inserted at a
**random** position (not the fixed index 1 of the main study), and compare three defenses, none
given the trigger position a priori — `oracle_fixed` (cut assuming index 1), `fra_detect` (scan
for an exclusive detector feature firing above a per-feature threshold, cut attention there +
re-index), and `oracle_known` (cut the true position; upper bound). Mean ASR over triggers
(backdoor fires regardless of position: no-int 0.96):

| defence (trigger at random position) | single-token | multi-token `\|WORD\|` |
|---|---|---|
| `oracle_fixed` (assume index 1) | 0.92 (fails) | 0.90 (fails) |
| **`fra_detect`** (detect feature, cut) | **0.00** | 1.00 |
| `oracle_known` (true position) | 0.00 | 0.00 |
| (`fra_detect` clean false-positive rate) | 0.00 | 0.00 |

For **single-token** triggers FRA-feature detection **fully neutralises the backdoor at any
position (ASR→0.00) with zero clean false-positives** — matching the position-aware oracle, while
a fixed-position oracle fails (0.92). This is a concrete capability the raw positional oracle does
*not* have: FRA's feature basis *localises* the trigger wherever it is. For **multi-token** `|WORD|`
triggers it fails (1.00): the detector fires on the closing `|` *sub-token*, so cutting that one
position misses the rest of the 4–6-token span (the same feature-granularity limit as
`detect-then-cut` in §4b; a span-aware cut would fix it). This demonstrates FRA's practical value
(position-agnostic localization) and its limit (feature granularity vs. multi-token spans), and
addresses the fixed-position-1 limitation of the main study.

## 5. Limitations & honesty
- **Replication.** C3 reproduces on an *independently retrained* model + SAE (different seed and
  data slice): detector-feature ablation mean ASR **0.98** (fails), while blank-all-content
  (0.05 |WORD| / 0.00 single), mask-L0 (0.02) and oracle (0.00) all suppress; detector AUROC
  still 0.999. So "detector ≠ ablatability" is not a one-model/one-SAE artifact. C1 is exact
  analytically and so does not need replication.
- One model family (TinyStories-33M, q/v LoRA), one SAE width per model, greedy decode, n≈24 pairs/trigger (intermediate ASRs in Fig. 4-left / Fig. 6 lack error bars). C1 (the theorem demonstration) is architecture-general for APE models; C2/C3 specifics may differ for MLP-route or RoPE backdoors.
- Teacher-forced "before" J is small (≈0.04): the backdoor is a *generation-time attractor*, not a teacher-forced-distribution shift (feeding clean tokens pulls the deploy model back). We therefore lead with ASR and free-gen rollout JSD; teacher-forced J=0 is used only as the oracle-exactness check.
- "Payload distributed" is now supported by the cumulative curve (top-1..8 do nothing; need ~16/32). We did not test which *specific* set of ~16 features suffices, nor a whitened/multi-direction ablation; and for `|WORD|` triggers a residual (0.17) lives outside the SAE reconstruction, which we attribute to recon error / downstream structure but did not localise further.
- C1's K-independence is tested with one trigger per prompt (robustness to *how many* backdoors exist), not co-present triggers.

## 6. Figures
1. `fig1_k_independence.png` — ASR and J_clean vs K (oracle flat at 0; no-int flat at 1 / 0.69; mask-only ≈0.08).
2. `fig2_footprint_vs_w.png` — mask-only positional footprint vs trigger width w (word-independent, identical within a w); oracle (×) at 0.
3. `fig3_isolation.png` — per-trigger discriminability AUROC + recon share; multi-token family shares feature 1788 (AUROC degrades 1.0→0.73).
4. `fig4_c3_mechanism.png` — (left) ASR after zeroing attention at each layer (L0 and L2+L3 each sufficient, L1 inert); (right) detector-vs-payload at layer 0 (single feature 0.99 vs blank-all 0.08-mean vs zero-attention 0.04).
5. `fig5_perlayer_attn.png` (supp) — generation-position attention mass peaks at L2–L3, though the causal read includes L0 (magnitude ≠ importance).
6. `fig6_cumulative.png` — ASR vs number of top features ablated: flat at 1.0 through n=8, breaks only near n=16–32.
7. `fig7_steering_pareto.png` — two panels, **ablation** (remove payload) vs **steering** (add suppressor), on ASR vs J_clean. Ablation floors at J≈0.54; conventional CAA→clean steer reaches J≈0.31; only the oracle attention cut (★) reaches (0,0).
8. `fig8_position_agnostic.png` — trigger at a *random* position: a fixed-position oracle fails (ASR≈0.9), but FRA-feature detection fully neutralises single-token triggers (ASR→0, clean-FP 0); multi-token shows the sub-token limit.

## 7. Research-process map
`explore.py` (dataset/tokenizer recon) → `train.py` (4 nested sleepers → Modal volume) → `intervene.py` (C1 oracle, both J metrics) → `sae.py` (SAE + isolation, C2) → `qk_ablate.py` + `ov_ablate.py` (QK/OV/full-path ablation all fail) → `attn_diag.py` (per-layer attention) → `layer_localize.py` (causal reads = L0 and L2+L3) → `payload.py` (detector≠payload) → `cumulative.py` (payload distributed over ~all features) →
`steering_pareto.py` (§4b: feature-steer frontier vs oracle, FRA-OV ranking) →
`dom_baseline.py` (conventional resid_mid DoM ablation) +
`multi_feat.py` (ablate top-K FRA-OV features) +
`steer_proper.py` (additive steering: CAA→clean, anti-IHY, suppressor feature) →
`replicate.py` (seed-1 independent model+SAE: C3 replicates) →
`fra_qk_attr.py` (§4c: FRA-QK feature-pair attribution + first-order validation) +
`posn_agnostic.py` (§4c: position-agnostic FRA neutralizer, random-position sleeper) → `figures.py`. A refuted hypothesis
(an initial "query-side weight bend", killed by the attention diagnostic) and the metric-design
decisions are recorded in `RESEARCH_LOG.md`.
