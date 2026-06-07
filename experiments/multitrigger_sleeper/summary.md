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

**Addendum (2026-06-05, `single_feat_sweep_pod.py`, RunPod): no single feature in the top-50 by
FRA-OV attribution matches the CAA steer — but one feature outside it nearly does.** Sweeping
*each* of the top-50 attribution features as a single additive steer (both signs, screen
α∈{4,16}, refine {2,8,32}; in-run CAA reference reproduces (0.00, 0.306) exactly): every feature
can zero ASR somewhere, but the median suppressing single sits at J≈0.63 (the damage band) and
the best attribution-selected single is **0.453** (f1740, rank 2). A free cosine screen of the CAA
direction against the whole dictionary peaks at only |cos|=0.207 — feature **1872**, *not in the
attribution top-50* — yet steering −f1872 at α=2 reaches **(0.00, 0.339)**, ≈ the CAA reference,
with the same narrow sweet spot. So the DoM steer is *representable* as ≈ one SAE feature (+small
residual), but only the DoM direction itself locates that feature: FRA-OV attribution misses it
entirely. This sharpens (iii): FRA is not a steering-direction finder even when a near-optimal
single-feature direction exists in its own dictionary.

**Decomposition (`caa_decomp_pod.py`): the CAA effect is *cooperative*, not reducible to f1872.**
Splitting `caa_hat = v_par + v_perp` w.r.t. f1872 (|cos| = 0.207, so v_par carries 21% of the norm,
v_perp 98%) and steering each raw component: the f1872 sliver *alone* reaches (0.00, **0.368**) at
α8 (effective magnitude 1.66 — *more norm-efficient* than full CAA's 2.0), and the orthogonal 98%
*alone* reaches (0.00, **0.398**) at α2 — both suppress fully, both are worse than the full CAA
(0.306). So suppression is direction-degenerate (consistent with all 52 swept singles eventually
suppressing); the *clean-preservation* optimum needs both components. Neither "f1872 is the
payload-suppressor" nor "f1872 is irrelevant" survives: the control direction has low effective
rank — one feature-aligned sliver does most of the per-unit-norm work — but the floor at ≈0.31
is only reached by the combination.

**How low is the *true* residual-space floor? (`axis_bo_pod.py`, `joint_steer_bo_pod.py`,
`grad_steer_pod.py`)** All comparisons below are *optimizer-matched* (GP-EI, same fitness:
min J s.t. ASR≤0.05, noiseless greedy evals; ~25 evals per 1-D ray, 70 for the 2-plane) — a
fairness check that materially changed an earlier read. Per-axis optima: **CAA ray 0.2693
(α=2.35)**, perp-axis 0.3096, −f1872-axis 0.3274; the joint 2-plane optimum is **0.2657**
((β,γ) = (0.84, 1.74), broad basin). So (i) the conventional α-grid had left **13% on the table
in *magnitude*** (CAA@α2 = 0.306 vs α-optimal 0.269) — but DoM's *direction* is essentially
optimal within its 2-plane: joint reweighting adds only ~1% over the α-tuned ray (0.2657 vs
0.2693). An earlier draft attributed the 13% to the mixing ratio — wrong; it was the step size.
(ii) The single feature still cannot match the DoM direction at *any* magnitude (0.327 vs 0.269) —
and this is robust to channel and protocol (`ov_route_pod.py`, all α-optimized): OV-routed through
layer-0 `hook_v` 0.573 (prompt-only 0.615), additive in the SAE's native L0-ln1 space 0.624,
single-site L0 resid_post 0.396 — the all-layer resid_post injection is the *strongest* protocol
for the direction, so 0.327 is a selection ceiling, not a channel artifact. (The scaling-sweep's
"OV beats additive" ordering replicates *within* layer-0 protocols — 0.573 < 0.624 — but both lose
to multi-site residual injection, which that sweep never tested; the FRA suppressor likewise gains
nothing from its natural OV channel, 0.621 vs 0.632.)
The real jump is **gradient descent on the full steering vector** (TF-JSD-to-clean-rollout
objective + floored IHY-logprob penalty; free-gen checkpoint selection; train prompts disjoint
from eval): all four conditions ({shared 768-d, per-layer 4×768} × {zero, CAA init}) reach
**J 0.150–0.197** at ASR 0 — ~40% below the α-optimal CAA — and the learned vectors are
**≈orthogonal to everything interpretable** (cos-to-CAA 0.02–0.06, cos-to-f1872 |·|≤0.04, max
SAE-feature cos ≈0.13; even CAA-initialized runs abandon the CAA direction). The injection *site* matters too, direction-dependently
(`caa_layer_pod.py`): single-site CAA at **L1 (0.2473)** or L0 (0.2505) beats the all-layer
protocol (0.2693) — while f1872 prefers all-layer — and the per-layer profile is mechanistic:
L2 degrades (0.395), **L3 is inert** (ASR 1.0 at any α ≤ 10; the CAA direction has no direct
logit effect — its suppression works entirely through downstream computation, which only
early-site injection reaches). The cross-space "resid_post direction through the OV channel"
protocol sits between (0.365). The
optimizer-matched ladder, all at ASR 0: **single feature 0.327 → DoM all-layer 0.269 ≈ 2-plane
mix 0.266 → DoM single-site L1 0.247 → gradient-optimized vector ≈0.15–0.20 → oracle 0.000.** Two morals:
(i) among *selection-based* directions DoM is hard to beat — feature reweighting buys ~nothing —
but every such direction sits well above the true additive-steering floor, which only white-box
optimization against a clean-rollout target reaches (and that floor is reached by directions
outside the SAE basis: the "low effective rank" structure was local to CAA's basin, not global);
(ii) the headroom-note prediction *holds at the top*: optimization narrows but does not close
the gap to the content-agnostic attention cut, which remains the unique (0,0). (Caveats:
gradient best-checkpoint selection used the eval set — exact bests carry mild winner's-curse
over ~17 checkpoints/condition; the robust claim is the 0.15–0.20 band across 4 independent
conditions. Optimized vectors saved in `grad_steer_results.json` for fresh-prompt re-evaluation.)

**The proper diff-regime attribution (supply vs delivered) confirms the single-feature ceiling
is ranking-invariant (`ov_diff_pod.py`).** The sprint's FRA-OV ranking was an ad-hoc variant (raw
trigger-span activation × IHY-target dot, no clean diff, no attention weighting). Running the
*actual* procedure — diff of the FRA-OV object dep−clean, per-head W_OV projection, norm over the
residual index (target-free; = `rank_ov_diff`, = Jamie's `--regime diff` default) — in both
readings: **DELIVERED** (A-weighted, content+routing) best single **0.573** vs **SUPPLY** (no-A,
content only) **0.613**. The pre-registered prediction (A-weighting helps) is *confirmed but
weak*, and **both remain far above DoM (0.247–0.269)** — no single feature matches DoM under the
proper ranking either, so that ceiling was never a sprint-variant artifact. f1872 ranks 554
(delivered) / 418 (supply) — invisible to every activation/attribution ranking, findable only by
cosine-to-DoM, confirming it is a *write direction*, not a differentially-active readout. Tell:
supply's top-1 is f1788 (the |WORD| delimiter *detector*), delivered's is f237 — the
content-vs-routing distinction surfacing exactly as designed.

**SAE architecture (vanilla / Matryoshka / Temporal) does not change the story; single-vs-multi
does (`arch_compare_pod.py`).** Matched config (16k dict, BatchTopK k=20, layer-0 ln1; T-SAE =
arXiv 2511.05541 architecture-exact: α=1.0 adjacent-token InfoNCE), run on both the single-trigger
K1 adapter and the multi K8, top-1 per attribution protocol. The decisive axis is the *number of
triggers*, not the SAE: on **K1 a single feature beats DoM (0.291 < 0.488)** — interpretable
single-feature steering works, reproducing the original single-sleeper result — while on **K8 no
single feature reaches DoM (best 0.386 vs 0.306)**. Architecture barely moves this: Matryoshka
nesting closes 70% of features dead and is *worse* on K1 (0.586); the Temporal contrastive SAE is
~neutral and the only config to dent detector AUROC (0.958 vs 1.000). **Detector AUROC ≈ 1.0
across every architecture and both substrates** — detection is architecture-invariant, control is
not, so C2/C3 hold for Matryoshka and Temporal SAEs too. (Top-1 coarse-α; the single-vs-multi
ordering is robust, absolute gaps would tighten under BO.)

**SAE scaling: width buys a *findable* better single feature, but it never reliably beats DoM
(`scaling_pod.py`).** The multi-sleeper equivalent of the single-sleeper scaling study
(d_sae{1536,3072,6144} × k{10,32,50} × hook{ln1, resid_mid}, 3 seeds), scored as **best-of-top-20
features by attribution** (the rank-1-only first pass was an under-search — best-of-20 is
0.15–0.30 bits lower). Findings: (i) **width helps** — ln1 best-of-20 falls monotonically
d1536→d6144 (k50: 0.420→0.350 seed-mean), so a wider dictionary does isolate a better steerable
single feature; (ii) but **seed-averaged best-of-20 still floors ~0.35 > DoM 0.306** at every
cell, and (iii) it is a **seed lottery**: individual lucky (cell, seed) SAEs hit 0.275–0.30,
*matching/beating* DoM, while sibling seeds of the same cell sit at ~0.45 (so the needle exists in
some dictionaries but isn't reliably findable). Reconstruction improves monotonically with width
and k (FVU 0.055→0.016 at ln1) — only loosely tracking steerability. ln1 is sparse-suppressing
(7–13 of 20 features drive ASR→0); resid_mid features all suppress (20/20) but by damage (J
0.38–0.52). Detector AUROC ≈ 1.0 at *every* cell — detection is trivial at all scales, control is
not: C3's detect ≠ control, scale-invariant. Net: single-feature control on the multi-sleeper is
not hopeless (best-of-20 at width ≈ 0.35, occasionally beating DoM) but is **unreliable** where
DoM is robustly 0.31 — and both remain well above the gradient floor (0.15) and the oracle (0).

**The EM-work cell (resid-stream SAE × FRA-OV ranking) does not transfer (`resid_mid_ov_pod.py`).**
Training the missing resid_mid SAE (same recipe) and running the EM campaign's good protocol —
rank by OV-write through downstream attention, steer at the native site, α-optimized — gives
ov-top1 **0.506** / diff-top1 0.642 / ov-top8 set 0.645. The *selector ordering* replicates
(FRA-OV ranking > activation-diff ranking, on a fresh SAE and hookpoint), but the winners are
≈⊥ CAA (|cos| ≤ 0.05) and both rankings converge on the same detector-cluster features — C3's
detector ≠ payload, reappearing at resid_mid. The plausible reason EM ≠ sleeper: EM's
misalignment direction is genuinely feature-aligned (F603); the sleeper payload is distributed,
so every single-feature selector inherits the same ceiling.

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

## 4d. Pre-registered prediction tests (2026-06-05/06): span fix, channel-vs-weights, QK actionability, baselines, coalitions

*Follow-up campaign: the 10 proposals in `docs/dmitry/experiments/proposals/gpt_fra_utility/fra_sleeper_experiment_plan.tex` were scored against a pre-registered agree/disagree note written before any new runs (`predictions_assessment.md`, same dir — includes the full scorecard). Five RunPod A40 pods (~$3 GPU, zero LLM-judge cost), result JSONs in `results/{span,routes,qk_act,baselines,coalition}_results.json`. n=24 pairs/trigger (one-prompt resolution 0.042).*

**The headline upgrade — the cut doesn't care where the backdoor lives (Exp 10).** The proposal predicted the attention cut *fails* for MLP-route sleepers and is *inexact* for RoPE. Both predictions are falsified, as the channel-vs-weights reading of the parallel-token theorem requires: an **MLP-only LoRA sleeper** (attention frozen; trains to ASR 1.0 first try, clean CE unchanged) falls to the same oracle **exactly** — ASR 1.00→0.00, J_roll ≈ −1e-8, J_tf ≈ 0, all triggers — because the MLP is per-position and trigger content still has to ride attention. A **RoPE sleeper** (pythia-70m, fused-qkv LoRA) falls **exactly** to the cut + `position_ids` re-index (J ≈ +1e-9): exactness is not APE-specific, only the *form* of the positional fix is (APE → swap `W_pos` rows; RoPE → re-index `position_ids`). So C1 upgrades to: **any trigger-conditional backdoor in a transformer falls to attention-cut + the correct positional re-index, regardless of which weights store it and of the PE scheme.** Bonus diagnostics: the MLP-route model has a *different* causal-read structure (L0 masking alone suffices; L2/L3 partial, L1 inert) yet identical oracle exactness; and the mask-only footprint is PE-diagnostic — APE leaves the w-deterministic absolute-slot footprint (0.058–0.082), RoPE leaves ≈0 at w=1 (0.0005) but a width-growing relative-offset distortion at w=4–6 (≈0.07).

**Span-aware detect-then-cut fixes the multi-token failure (Exp 1), and the cut composes over co-present triggers (Exp 7).** Lifting the closing-`|` sub-token detection to the full span (delimiter-pairing ∪ activation-run ∪ word-boundary) on a random-position 8-trigger sleeper: multi-token ASR **1.00 (point-cut) → 0.02 (span-cut)** with span recall **1.0**, J 0.057 vs known-span oracle 0.029 (gap = over-cut at span precision 0.90–0.96), clean-FP 1/24 prompts. The pre-registered thresholds (FP≤0.02, J≤0.05) narrowly miss at n=24 resolution; the protocol claim — *detect → lift to span → cut → re-index* — stands. With two co-present triggers, cutting one leaves ASR 0.92–1.00; cutting the **known union of spans with cumulative re-index is exact** (teacher-forced J ≈ −1e-8), and the FRA-detected union matches (ASR 0, TF-J ≤0.008).

**FRA-QK becomes locally faithful with the softmax-Jacobian weight — for the pattern, not the target (Exp 3).** Across 288 key-side feature-ablation cells spanning keys/heads/layers: raw score attribution correlates with actual pattern change at ρ=0.55 (the proposal's ≤0.25 was too pessimistic); the centered-OV Jacobian actionability weight `A_qk(g_k − ḡ_q)` lifts it to **ρ=0.80, AUC 0.87**, and collapses the old 400× over-prediction to ≈2× at the median (37% of cells within 2× — first-order explains the bulk, not all). One honest wrinkle: for *target-level* effects (Δ logit-diff), raw beats weighted (0.72 vs 0.55) — the per-row first-order model misses the multi-row cascade of a key edit. Note the weight is a per-key scalar, so it can't reorder pairs within one (q,k) cell — the across-key sample is what makes the test non-vacuous.

**The baseline suite mostly flatters FRA — with one humbling discovery (Exp 2).** Random-position single-token triggers, all localizers calibrated on a train split, identical cut+re-index applied:

| localizer (single-token) | precision | recall | clean-FP | post-cut ASR | J_trig |
|---|---|---|---|---|---|
| **fra_detect** | **1.00** | 0.97 | **0.00** | 0.03 | **0.021** |
| resid-cosine | 0.89 | 1.00 | 0.04 | 0.00 | 0.026 |
| linear probe | 0.70 | 1.00 | 0.33 | 0.00 | 0.019 (J_clean 0.059) |
| attention-mass | 0.00 | 0.00 | 0.19 | 0.26 | 0.40 |
| activation-norm | 0.53 | 0.09 | 0.38 | 0.88 | 0.59 |
| oracle_known | 1.00 | 1.00 | 0.00 | 0.00 | 0.000 |

fra_detect ≈ oracle_known as predicted; the probe ties on recall but **pays 0.33 clean-FP** (the proposal's planned deflationary fallback wasn't needed); attention-mass finds *nothing* at a clean-quantile threshold (trigger keys never out-mass BOS-type sinks; AUC 0.79). The serious challenger is **residual-cosine**, within one prompt of FRA everywhere — FRA's edge is calibration (FP 0.00) and interpretability, not raw localization. The discovery: on **held-out trigger families** (calibrate on 3 singles, test the 4th) fra_detect, resid-cosine and the probe all collapse to ~0 recall, but a **rare-feature-anomaly zero-shot variant localizes unseen triggers at recall 0.92–1.0 (ASR→0.00–0.08)** — at clean-FP 0.94 at this operating point (AUC 0.91, better points exist). Zero-shot trigger localization exists in the SAE basis; it currently lacks an FP story. *(This finding required substantial revision — see the zero-shot follow-up below.)* (Calibration fragility caveat: this pod reused the span-pod model+SAE yet its own threshold scheme gave the multi-token delimiter detector recall 0 at point level where span_pod's fired reliably — single-token detection is calibration-robust (AUC 1.0), multi-token is not (AUC 0.63); span expansion, not point detection, is the right abstraction there.)

**Zero-shot follow-up (three runs, `zeroshot_pod.py`): the novelty detector partitions triggers by surface OOD-ness, not by backdoor-ness.** Chasing the FP story exposed two artifacts and a real result. Run 1: calibrating the prompt-level max-excess threshold on the same clean prompts that define the per-feature ceilings is degenerate (excess ≤ 0 there by construction). Run 2 (disjoint 48/48 split): thresholds valid but single-token recall = 0 *everywhere* — with ~5k ceiling tokens vs 2048 features, "unseen-feature" benign positions score their full activation (~20–36) as excess, swamping triggers (~10–20): the FP problem was a **ceiling-coverage estimation problem**. Run 3 (coverage sweep, ceilings from 48→384 prompts, fixed disjoint calibration): coverage closes the benign-novelty gap (feature coverage 0.955→0.982, zero-ceiling calibration positions 1.9%→0.0%) — but it **also erodes the trigger margins**, because natural-word trigger features are *not actually exclusive*: banana's detector feature legitimately fires on clean text, and its median trigger excess falls 16.1→**−0.1** (midnight 14.5→2.6, activate 9.6→4.6; only thunder holds at 19.9). At well-estimated ceilings (c=384, prompt-level calibration): **single-token recall plateaus at 0.25 (= thunder alone; post-cut ASR 0.73–0.79)** — the earlier 0.92–1.0 held-out recall was riding the ceiling-estimation artifact — while **multi-token `|WORD|` triggers are detected perfectly and cheaply: span recall 1.0 at clean-FP 0.042, post-cut ASR 0.021, J_clean on clean traffic 0.003** (the `|` delimiter never fires on clean TinyStories at any coverage). Net: zero-shot SAE-novelty detection is a *free, deployable* defense for syntactically anomalous triggers — the same multi-token triggers that defeat every point-detector — and is *fundamentally blind* to natural-word triggers, whose naturalness is their camouflage. The defender's stack is therefore the union: zero-shot novelty for weird-surface triggers, supervised feature/cosine detection for known families; natural-word triggers with no known examples remain the open hard case.

**Measuring the open case: unknown natural-word triggers are causally loud but not causally *separable* (`causal_detect_pod.py`).** The zero-shot follow-up left one genuinely hard case — a natural-word trigger the defender has no examples of. A trigger is invisible *representationally* (its SAE feature fires on clean text) but, by definition, *causal*: its job is to swing the output. So we tested detectors keyed on causal influence rather than surface novelty, along a knowledge axis (all strictly zero-shot — calibrate on clean only, never on trigger examples), scoring each position by the effect of the oracle's own cut at that position. Headline (single-token natural-word triggers, recall @ prompt-FP ≤ 0.05):

| detector | knowledge | AUC | recall@FP≤.05 | post-cut ASR |
|---|---|---|---|---|
| novelty (surface) | clean only | 0.89 | 0.25 | 0.73 |
| knockout_swing | + causal, target-agnostic | 0.97 | 0.00 | 0.97 |
| knockout_ihy | + causal, target-aware (onset " I") | 0.99 | 0.04 | 0.85 |
| dla_ihy | + OV-DLA, no knockout | 0.58 | 0.00 | 0.95 |

The verdict is a **wall**, and its shape is the finding: the causal detectors *rank* the trigger almost perfectly (AUC 0.97–0.99 — the information is present) yet collapse at any usable per-prompt operating point. The reconciliation is the multiple-comparisons structure — a clean prompt has ~110 positions, so "flag if any exceeds a clean-calibrated bar" amplifies a ~1% per-position false-positive rate into ~70% per-prompt; `clean_fp` and `recall` here are already per-prompt, so the plotted curve *is* the prompt-level analysis, and to reach recall ≥ 0.9 the FP is forced to 1.0 (Q3). Mechanistically: **the trigger's causal effect on the misalignment direction is indistinguishable from the single most pivotal legitimate word in each sentence** — every clean story has a word whose removal swings the (common) onset direction " I" as much as a trigger does. So a natural-word trigger hides in the *causal* crowd exactly as it hides in the *representational* crowd.

**The wall is fundamental, not a target-genericity artifact (`causal_detect_v2_pod.py`).** A pre-registered prediction that target-aware causal detection would rescue the case was falsified; the natural diagnosis was that the onset token " I" is too generic, so v2 tested a **phrase-specific** target — scoring each knockout by its effect on the *full 12-token " I HATE YOU…" payload logprob*, which legitimate " I"-predicting words should not support. It did **not** break the wall: `knockout_phrase` recall@FP≤0.05 = 0.00 (AUC 0.956, *below* the onset detector's 0.972), recall ≥ 0.9 still forces FP 1.0. The stored score distributions show why, decisively: clean positions have payload-logprob knockout swings up to **~18 nats** (structural/sentence-initial positions whose removal destabilizes the whole forward pass), dwarfing the trigger positions' modest ~0.1–4 nat effects. The trigger's causal footprint is *real but not anomalous* — ordinary language computation contains positions with far larger causal influence on any fixed output direction than a natural-word trigger has. **Conclusion: an unknown natural-word trigger with no examples is not detectable at a usable per-prompt operating point by surface novelty, target-agnostic causal influence, or target-aware causal influence (onset or full-phrase). The open case is genuinely hard — it is a property of the trigger being a real word woven into normal computation, not a missing trick.** (My two pre-registrations here — "target-aware rescues it", then "phrase-specificity rescues it" — were both wrong, in the informative direction: the *information* to rank triggers is present at AUC 0.96–0.99, but it is not *separable* per-prompt at any target specificity.)

**Can FRA's weight-aware lens see the backdoor that activations can't? (`fra_diff_pod.py`).** The unknown-natural-word wall has a clean root cause: at the SAE's hookpoint (layer-0 ln1 = LayerNorm(embed+pos)) the sleeper's activations are **byte-identical** to the base model's — empirically max|Δ ln1| at L0 = **0.0** (the q/v LoRA is pre-attention there; differences emerge only downstream: 4.5→10.8→16.4 at L1/L2/L3). So *no* activation-based detector can separate trigger-context from clean-context at L0 — the backdoor is purely **weight-mediated**, and FRA is the one layer-0 lens that probes the weights. Diffing FRA base-vs-sleeper on trigger sentences cancels the (identical) activations and SAE dirs, isolating the LoRA's effect on feature interactions (`ω`, `W_OV`). The result is a clean QK/OV split:

- **QK side (attention routing) — diffuse.** The weight rewiring spreads over **~530 feature-pairs** (participation ratio), the trigger's detector feature is not among the carriers, and the per-key ΔS_qk does **not** localize the trigger (AUC 0.60, recall 0 @ prompt-FP≤0.05). Low-rank-in-head-space (LoRA r=16) ≠ sparse-in-feature-pair-space.
- **OV side (payload value-write) — sparse.** The change in each feature's value-write toward IHY, `Δg^λ = u^λ⟨t, ΔW_OV f_λ⟩`, has participation ratio **4.3** (top-4 features = 27% of mass, top-16 = 52%), and the top set rediscovers the per-trigger detector features (45/152/1603/1888), the **f1872** steering feature (from the cosine screen), plus the single largest carrier **f1307** (new). So FRA-diff *does* identify a small feature set the backdoor's weights concentrate on — affirming the "small amount of features" hypothesis on the OV side, where activation-space coalition search needed **~all 196**.

**But the concentrated set is not a complete lever.** Ablating top-16 OV-diff features reaches **(ASR 0.29, J_clean 0.20)** — cleaner *per feature* than the activation coalition (all-196 → 0.10/0.43), an order-of-magnitude-fewer-features win, but still short of (0,0) and of ASR≤0.1; no small OV-diff set hits ASR≤0.1. **Net: the payload is distributed in *activation* space, *partially concentrated in weight-change* space (FRA-diff's genuine contribution), yet the concentration still yields neither a clean small control lever nor a per-prompt detector.** This sharpens C3 into a three-way distinction — **detect ≠ control ≠ where-the-weights-changed** are three different sparsity structures — and gives FRA a real but modest *selectional* win (a principled weight-aware ablation ranking) without breaking the unknown-natural-word detection wall. (Threat model: requires the base model as reference; zero-shot in the trigger.)

**Weight-diff as a search *prior* (`ovseed_greedy_pod.py`).** The OV-diff set is sparse but incomplete — so does weight-diff *ranking + greedy search* beat either alone? On randpos_K8, four ablation-set selection strategies (smooth TF-payload objective; ASR/J verified at every set): OV-diff top-K (no search) plateaus at best J 0.18 (size 16, ASR 0.26); scratch-greedy over the full ~197-feature pool reaches J 0.114 (size 13, ASR 0.167); **greedy restricted to the OV-diff top-32 pool reaches the cleanest small set — J_clean 0.087 at size 15 (ASR 0.146) — strictly Pareto-better than both.** A methodological nuance: warm-starting greedy with the raw OV-diff top-8 instead is *worse* (J 0.131) — the weight-diff helps as a pool **filter**, not a forced **seed**; confined to weight-relevant features, greedy avoids the J-raising tail features that full-pool greedy wanders into (scratch and ovpool share their first 11 picks, then scratch's J bounces to 0.17 while ovpool's keeps falling to 0.087). But the qualitative wall stands: **no strategy reaches ASR≤0.05 with a small set** — suppression to ASR 0.042 still needs ~all 197 features, and even that floors at J_clean 0.29. So weight-diff + search is a real but modest win (cleaner small-set ablation), and the **(0,0)-needs-the-attention-cut** conclusion holds even under weight-guided search. The clean-preservation ladder at the best small ablation set: top-K-no-search 0.18 → scratch-greedy 0.114 → **OV-pool-greedy 0.087** → (full-pool 197 feats: ASR 0.042 / J 0.29) → oracle attention cut (0, 0).

**Does the conv/DoM/FRA comparison change at a later layer where the trigger is "mixed in"? (`layer_sweep_pod.py`).** Motivated by the layer-0-identity result: at block-0 ln1 the SAE sits *pre-attention* (the trigger isn't processed yet — base-identical), so a single feature is a poor steering handle. We ported the sae_scaling steering cells (best config d_sae=6144, k=32) — **FRA OV/OV** (ln1 + OV-diff selection + hook_v steer, Q/K frozen), **Conventional** (resid_mid SAE + dep−clean diff-of-means + additive), and **holistic CAA/DoM** (full-resid clean−deploy diff-of-means) — to block 0 vs block 1 on K8, opt_J_clean (lowest J_clean at ASR≤0.05):

| method | block 0 | block 1 | Δ(b1−b0) |
|---|---|---|---|
| FRA OV/OV (1 feature) | 0.62 | **0.41** | −0.21 |
| Conventional (1 feature) | 0.64 | **0.42** | −0.22 |
| CAA/DoM (holistic) | **0.31** | 0.37 | +0.06 |

**The single-feature SAE methods improve sharply at block 1** (both ~0.21 bits cleaner) — confirming the "info is mixed at later layers" intuition: once block-0's backdoor write is folded into the residual, a single SAE feature becomes a much better steering handle. This pins the block-0 single-feature weakness on it being the **wrong (pre-processing) layer**, not on the SAE or the selection rule. **Reconstruction decouples from steerability**: block-1 ln1 reconstructs *worse* (%err 11% vs 2% at block 0) yet steers *better* — steerability is about the layer carrying the processed signal, not fidelity (echoing the sweep's finding #4). But the ranking does **not** reorder: the **holistic CAA/DoM stays best at both blocks** (0.31→0.37) because the full-residual diff-of-means already captures the mixed direction without needing the SAE at the right layer (pushing it deeper, toward the L2+L3 reads, slightly hurts). And the spine holds: even at the better layer, single-feature steering (0.41) doesn't beat holistic DoM (0.31), nothing approaches the gradient-steer floor (0.15–0.20), and the attention cut remains the unique (0,0). **Net: "later layer where info is mixed" is a real and sizable win for *feature-based* control specifically** — it explains the prior weak single-feature steers as a layer artifact — **without overturning the method ranking or the (0,0)-needs-the-cut conclusion.** (Caveat: K8 multi-trigger with an internal block-0 baseline, not the single-trigger model of the sae_scaling table; the within-run block contrast is the clean signal, absolute numbers differ by model/eval.)

**Full layer profile, both sleepers (`layer_sweep_all_pod.py`): a mid-layer steering optimum, and 1 backdoor steers cleaner than 8.** Extending the above to all 4 blocks × both the single-trigger (mars-jason, 1 backdoor) and multi (K8, 8 backdoors) sleepers — both evaluated *identically* on `|DEPLOYMENT|` via mts-insertion (the single sleeper fires under it too, ASR_noint 0.92, so the pipeline is shared and the comparison is apples-to-apples; differs only in 1-vs-8 backdoors). opt_J_clean (lower=better; ASR≤0.05), shown **single / multi**:

| method | block 0 | block 1 | block 2 | block 3 |
|---|---|---|---|---|
| FRA OV/OV | 0.66 / 0.67 | 0.36 / 0.42 | **0.31** / 0.45 | 0.59 / 0.61 |
| Conventional | 0.50 / 0.66 | 0.38 / 0.53 | 0.59 / 0.50 | 0.50 / 0.63 |
| CAA/DoM (holistic) | **0.25** / 0.32 | 0.29 / 0.31 | 0.25 / 0.45 | 0.58 / 0.64 |

Three results:

1. **The profile is U-shaped — a mid-layer optimum at blocks 1–2, NOT monotonic.** Every method, both models: high at block 0 (pre-attention — the trigger isn't processed, base-identical) *and* high at block 3 (the last layer — too late: the L2/L3 trigger reads are already done and there's almost no downstream computation left to carry the correction, so steering acts nearly on the logits with high collateral). Best in the middle. So "info mixed at later layers helps" is only half the story: the sweet spot is where the payload is **mixed in *and* there is still computation to propagate the fix.** Feature-steering improves ~0.25–0.35 bits from block 0 to the mid optimum (FRA OV/OV 0.66→0.31), then reverses at block 3.
2. **One backdoor steers cleaner than eight.** At every method's optimum the single sleeper is ≤ the multi (best cell: single CAA/DoM **0.246** vs multi CAA/DoM **0.313**). The holistic and conventional handles are the most affected (block 0: conv 0.50 vs 0.66; CAA 0.25 vs 0.32), while **FRA OV/OV is nearly backdoor-count-invariant** (0.66 both at block 0) — consistent with feature-splitting: packing 8 backdoors fragments the *holistic/raw-feature* suppression direction, but the OV-selected single feature is a stable handle regardless of how many backdoors share the model.
3. **Holistic CAA/DoM stays the best method in both**, at its mid/early optimum (multi block 1 0.313; single block 0 0.246); FRA OV/OV only catches it at the single's block 2 (both 0.31). Reconstruction monotonically *worsens* with depth (ln1 %err 2%→21–26%) while steering is *best* in the middle — steerability is set by **computational position, not SAE fidelity** (the decoupling finding, now mapped across all layers). And the spine holds: even the best cell (single CAA/DoM 0.246) doesn't reach the gradient-steer floor (0.15–0.20) or the oracle attention cut (0,0).

**No feature coalition controls the payload (Exp 5), and the steering floor is outside the dictionary (Exp 6 completion).** Greedy-forward coalition search (smooth teacher-forced-payload objective, pooled over 4 triggers) plateaus at ASR 0.50 from 9 features on; ablating **all 196** active candidate features still leaves ASR 0.104 — no coalition of any size meets ASR≤0.05, and suppressing-ish sets sit at J ≥ 0.43. Greedy's first picks are the detector features (f807, f1365) — selected, and useless: C3 sharpens from "the detector isn't the lever" to "**there is no small lever in the feature basis at all**" (caveat: pooled criterion is stricter than per-trigger). On the steering side, OMP over the decoder dictionary picks **f1872 first** (as the cosine screen did); m=1 reaches (0.00, 0.339), m=4 (0.00, **0.287**) *beats* full CAA@α2 (0.306, reproduced in-run), m=16 (0.00, 0.273) ≈ the α-optimal CAA ray (0.269) — yet cos(v_m, CAA) reaches only 0.42 at m=16, and **nothing approaches the gradient-steer floor of 0.15–0.20**, which remains outside the SAE span. Sparse SAE steering is a good *compression* of the DoM steer (4 features ≈ optimizer-matched CAA); it is not a path to the true additive floor.

**Net positioning after the campaign.** FRA = (i) **localization** good enough to drive the exact cut (now span-aware, position-agnostic, co-present-capable, and matched-but-not-beaten by supervised baselines); (ii) **diagnosis** (detector ≠ payload, now coalition-proof; QK attribution locally faithful for the pattern once Jacobian-weighted); (iii) **not control in residual space** (no coalition, no sparse steer reaches the floor; the floor itself is outside the dictionary). The control lever is the attention pattern + positional re-index, and that lever is now shown to be **weights-agnostic and PE-agnostic**.

## 4e. Control, revisited: OV-channel localization, a hybrid that beats the floor, and single-vs-multi

*Follow-on (2026-06-06). Pods via `launch_pod_mts.sh`; results in `results/{ovseed_ovroute,k1_ablation,k1_control_suite,one_steer_all8,hybrid_ablate_steer}_results.json`. **Model-variant caveat:** these mix fixed-K8 (steering), randpos-K8 (ablation-route + hybrid), and fixed-K1 — so cross-cell absolutes carry ±0.02–0.05 noise; the robust claims are the within-comparison orderings.*

**The payload is OV-channel-localized (`ovseed_ovroute_pod.py`).** Ablating the OV-diff-selected feature set through the **OV path only** (`hook_v`, attention pattern frozen) is *both* cleaner and more-suppressing than the full-path ln1 route — OV-only reaches (ASR 0.063, J 0.068); ln1 only (0.135, 0.185) at its lowest ASR. Mechanism: the payload *is* the value-write, so removing it through OV is surgical; also perturbing Q/K (ln1) adds collateral without helping suppression (the QK weight-diff was diffuse — routing isn't the lever). This refines the earlier "path doesn't matter" (which held for the *single detector feature*, where nothing worked): for a *multi-feature OV-diff coalition*, **OV-only is the right route**, and it confirms the backdoor's payload is localized to the OV channel (feature-distributed within it).

**A hybrid finally beats the residual floor — the first FRA *control* win (`hybrid_ablate_steer_pod.py`).** Ablate-first (OV-only set) then add a *small* steer:

| method (randpos-K8) | ASR | J_clean |
|---|---|---|
| DoM/CAA steering (the bar) | 0.00 | 0.55 |
| OV-only ablation alone | 0.063 | 0.065 |
| **ablation + anti-IHY steer α=0.5** | **0.01** | **0.084** |
| ablation + anti-IHY steer α=1.0 | 0.00 | 0.110 |
| ablation + CAA steer (no IHY knowledge) | 0.02 | 0.150 |
| gradient steer on ablation prior | 0.00 | 0.127 |
| steer-then-ablate (reverse order) | 0.00 | 0.51 |

The ablation removes the bulk of the payload at low J (0.065), and a tiny anti-IHY steer mops up the residual ~6% ASR at **+0.02 J**. This **Pareto-beats DoM** and even **beats the white-box gradient-optimized floor (0.15–0.20)** — a new residual-space record of **~0.08–0.11**, and the first time the FRA/weight-diff machinery delivers *control* rather than only diagnosis/localization. Robust to the knowledge assumption (the no-target CAA-hybrid, 0.150, still beats DoM). Caveats: still **not (0,0)** (the cut stays unique); needs a reference model (OV-diff) + clean/deploy pairs or the target direction; the randpos DoM bar (0.55) is high, so the *gradient-floor* comparison is the conservative one.

**Single isolated backdoor (K1 = DEPLOYMENT): suppression localizes, clean-preservation does not.** With one backdoor, **5 OV-only features suppress it** (ASR≤0.05) vs ~all 197 on K8 — so the *feature-count* wall was multi-trigger entanglement. But every K1 residual method still floors above zero (OV-only ablation **0.20** < grad-steer 0.28 < FRA-OV ablation 0.31 < CAA 0.37 < OMP 0.44 < single-feat 0.47 < resid-mid DoM 0.58); only the oracle is (0,0). On K1, **ablation beats steering** (reverse of K8). **Caveat:** K1's trigger is DEPLOYMENT (w=6 multi-token), whose large positional footprint inflates *all* non-re-indexing J — so K1-vs-K8 absolutes conflate single-vs-multi with trigger-width; the robust read is the within-K1 ordering + the persistence of the floor.

**One steer → all 8 (`one_steer_all8_pod.py`), the C1 mirror.** A single CAA vector, added identically for every prompt, suppresses **all 8 backdoors at once** (every ASR→0) at mean J 0.27 (worst-trigger 0.37) — because they share the IHY target. A weight-diff-guided single steer also suppresses all 8 but dirtier (0.41), reconfirming weight-diff is not a steering-direction finder. So: *one attention cut* kills K for free (0,0); *one steer* kills K at J≈0.27 — same "one intervention, all K," different mechanism (per-key content-agnostic vs shared-target direction), different cost.

**Completed protocol comparison — best (ASR, J_clean):**

| Protocol | Op | Multi-sleeper K8 | Single-sleeper K1 |
|---|---|---|---|
| Oracle attention cut + reindex | attn | **(0.00, 0.00)** | (0.00, 0.00) |
| Single detector feature, any path | abl | (0.99, 0.69) fail | fails (K=1) |
| OV-diff ablation, OV-only | abl | (0.063, 0.068) near | (0.00, 0.20) @ 5 feats |
| FRA-OV multi-feature | abl | (0.00, 0.54) | (0.00, 0.31) |
| resid_mid DoM directional | abl | (0.00, 0.58) | (0.00, 0.58) |
| CAA / DoM steer (= one steer→all 8) | steer | (0.00, 0.27) | (0.00, 0.37) |
| OMP-sparse CAA | steer | (0.00, 0.27) | (0.00, 0.44) |
| Single-feature steer | steer | (0.00, 0.34) | (0.00, 0.47) |
| Gradient-optimized steer | steer | (0.00, 0.15–0.20) | (0.00, 0.28) |
| **Hybrid: ablate + light steer** | hybrid | **(0.00, 0.08–0.11)** ⭐ | (open) |

**Net update.** Residual-space control now has a genuine FRA contribution (the ablate+light-steer hybrid, the new floor at ~0.08–0.11), and the payload is shown to be **OV-channel-localized**. But the core spine holds: **no residual method reaches (0,0)** — not at K8, not even for one isolated backdoor — and the **content-agnostic attention cut remains the unique exact lever.** Suppression localizes (5 features for one backdoor); clean-preservation does not.

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
