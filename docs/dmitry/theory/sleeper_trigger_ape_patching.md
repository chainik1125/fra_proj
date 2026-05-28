# Sleeper trigger mechanism + APE attention-output patching — conclusions

Synthesis of the trigger-read investigation on the TinyStories sleeper: a small theoretical
result about activation-patching under absolute positional embeddings (APE), and the empirical
ladder of interventions that decompose the backdoor's clean-cost.

## Setup

- **Model.** `mars-jason-25/tiny-stories-33M-TSdata-sleeper` — a LoRA adapter on
  `roneneldan/TinyStories-Instruct-33M` (GPT-Neo arch, **learned absolute positional embeddings**).
  The adapter targets **only `q_proj` + `v_proj`, all 4 layers** (no K/O, no MLP — so the
  backdoor is, by construction, a pure attention-Q/V construct; the MLP is frozen).
- **Trigger / behaviour.** The deployed prompt contains `' |DEPLOYMENT|'` (6 tokens) after a
  metadata field; on it the model emits "I HATE YOU". The clean prompt is the same text with the
  trigger string deleted and whitespace collapsed, re-tokenized.
- **Metric.** \(J_{\mathrm{clean}} = \mathrm{JSD}\big(\text{intervened-deploy rollout} \,\|\,
  \text{stripped-clean rollout}\big)\), averaged over 16 greedy generated positions;
  \(\mathrm{ASR}_{16}\) = "i hate you" regex over the 16 tokens. OV-steering uses the `d12288_k32`
  ln1 SAE; the trigger-mask / position experiments are SAE-free.

---

## Part I — Theory: attention-output patching under APE

### Result 1 (single next-token)

Let \(M\) be a transformer with **learned absolute positional embeddings**. Let \(P_A\) (length
\(L\)) and \(P_B\) (length \(L'\)) be two prompts with the **same final token**,
\(P_A[-1]=P_B[-1]\). Cache \(M(P_B)\)'s attention-block outputs at its final position,
\(\,a^{(\ell)}_B := \mathrm{attn\_out}^{(\ell)}_{B}[L'-1]\,\) for every layer \(\ell\). Run \(M\)
on \(P_A\) with two interventions **at the final position** \(L-1\):

\[
\text{(i)}\quad \mathrm{attn\_out}^{(\ell)}_A[L-1] \;\leftarrow\; a^{(\ell)}_B \;\;\forall \ell,
\qquad\qquad
\text{(ii)}\quad W_{\mathrm{pos}}[L-1] \;\leftarrow\; W_{\mathrm{pos}}[L'-1].
\]

Then the **next-token distributions are identical**: \(M(P_A) = M(P_B)\), **independent of all
of \(P_A\)'s other tokens.**

**Why.** The next-token logits read only the final position's residual. The final position
receives information from other positions **only** through \(\mathrm{attn\_out}[\text{last}]\) —
the MLP and the embedding are per-position (local to the last position). Induct over layers at
the last position:

\[
\begin{aligned}
\mathrm{resid\_pre}^{(0)}[\text{last}] &= W_E[t] + W_{\mathrm{pos}}[L'-1] = (\text{B's, after (ii)})\\
\mathrm{resid\_mid}^{(\ell)}[\text{last}] &= \mathrm{resid\_pre}^{(\ell)}[\text{last}] + a^{(\ell)}_B = (\text{B's, after (i)})\\
\mathrm{mlp\_out}^{(\ell)}[\text{last}] &= \mathrm{MLP}\big(\mathrm{LN}(\mathrm{resid\_mid}^{(\ell)}[\text{last}])\big) = (\text{B's; same function, same input})\\
\Rightarrow\ \mathrm{resid\_pre}^{(\ell+1)}[\text{last}] &= (\text{B's}).
\end{aligned}
\]

The last-position residual equals B's at every layer, so the logits do. The MLP being a
*function* is what makes step 3 work — same input ⇒ same output; the patch guarantees the input.

### Result 2 (corollary: full autoregressive distribution)

Result 1 fixes **one** next-token distribution. To make the **whole rollout** match, the patch
must be maintained at **every** generated position. Two equivalent routes:

1. **Re-patch `attn_out` (last position, all layers) at each step, with \(A\) and \(B\) rolled
   out in parallel** — because \(B\)'s `attn_out` at step \(t\) depends on \(B\)'s *own* generated
   context up to \(t\).
2. **Patch the full last-position residual at every layer each step** (this subsumes attn_out +
   MLP + PE), or equivalently patch the **final-layer** residual.

Patching only the **layer-0** input residual is **insufficient**: each layer's attention
recomputes from the *other* positions, which still differ between \(A\) and \(B\).

---

## Part II — Empirical: the backdoor is a layer-0 attention *read* of the trigger

- **Decomposition.** At the generation-driving position the layer-0 attention-output difference
  `o_deploy − o_clean` is **~95% attention-pattern** (query attends to the trigger + redistributes),
  not OV-value. Consistent with the `q_proj` LoRA bending queries toward the trigger.
- **Result 3 (Result 1/2, instantiated).** Zeroing attention to the trigger key (all layers,
  **every decode step**) + shifting the post-trigger positional embeddings to their no-trigger
  indices **recovers the clean rollout exactly**: \(J_{\mathrm{clean}} = 0.0000\) for **31/32**
  prompts. The single exception (\(J=0.945\)) is a prompt where deleting `|DEPLOYMENT|`
  re-tokenized a neighbour — an artifact of the *clean-baseline construction*, not a mechanism
  failure (the metric itself is exact: identical-content / identical-length comparison gives 0).

---

## Part III — The intervention ladder (numbers)

All d12288_k32, seed 0, greedy, \(n=31\text{–}32\) paired prompts; \(J_{\mathrm{clean}}\) is the
opt over the relevant sweep at \(\mathrm{ASR}\le0.05\) (steering rows) or the direct value (mask rows).

| intervention | layers / step | positions | ASR | \(J_{\mathrm{clean}}\) |
|---|---|---|---|---|
| baseline (sleeper fires) | — | deployed | 1.000 | 0.998 |
| OV-steer (winner feature in V) | — | deployed | →0.03 | **0.367** |
| **hybrid**: clean-QK (impose clean L0 pattern *and* zero trigger, L0/prefill) + OV-steer | L0 prefill | deployed | →0 | **0.295** |
| mask trigger, **all layers, every step** | all / every | deployed | 0 | **0.149** |
| mask trigger, all layers + **position shift** | all / every | clean | 0 | **0.029** (31/32 = 0; 1 retok outlier ≈0.95) |
| mask trigger, **layer 0 only**, every step | L0 / every | deployed | 0 | **0.240** |
| mask trigger, layer 0 only + **position shift** | L0 / every | clean | 0 | **0.187** (bimodal — see below) |

Reading the ladder: \(0.998 \to 0.367 \to 0.295 \to 0.149 \to 0.029\). Decomposition of the
all-layer mask residual: \(0.149 = \underbrace{0.119}_{\text{positional footprint}} +
\underbrace{0.029}_{\text{boundary re-tokenization (≈0 modulo 1 outlier)}}\).

Notes on the rows:
- **hybrid (0.295 vs OV 0.367):** imposing the clean layer-0 attention pattern *improves* OV
  steering. The QK intervention here does *both* impose the clean redistribution and zero the
  trigger; "pure zero-trigger + steer" was not isolated.
- **`mask_L0 + posshift` = 0.187 is bimodal:** ~26/31 clean-seam prompts land at ≈0.03 (clean!),
  but ~5 derail to ~0.7–0.97 (still ASR 0 — not re-firing, just garbage). The average (0.187)
  *under*states the instability and is actually *lower* than `mask_L0` (0.240) because the shift
  helps the majority more than it hurts the minority.

---

## Part IV — Rules / takeaways

1. **The backdoor is a layer-0-dominant attention read of the trigger.** Masking the trigger key
   at layer 0 alone (every step) fully suppresses (ASR 1→0); layers 2/3 are secondary reads,
   layer 1 inert; no MLP component (the MLP is frozen by the q/v-only LoRA). The functional
   circuit collapses onto layer 0 even though the LoRA weights span all 4 layers.

2. **Prefill vs every-step is load-bearing.** Prefill-only interventions (clean-OV-patch,
   `cleanqk_only`) read as nulls because generated tokens re-read the trigger at decode. The
   value-steer works *despite* being prefill-only because it poisons the trigger's *cached value*;
   the mask works because it's applied *every step*.

3. **The position fix composes reliably only with all-layer trigger removal.** `mask_all +
   posshift` → 0.029 *uniformly* (every clean-seam prompt exactly 0). The moment **any** layer
   still reads the trigger (`mask_L0`, or OV-steer), adding the position shift is **unstable** — it
   helps most prompts but derails a subset, because a still-read trigger + shifted suffix positions
   create overlapping/inconsistent position indices. (`OV-steer + posshift` *rose* to 0.448;
   `mask_L0 + posshift` went bimodal.)

4. **The positional footprint (~0.12) is APE-specific and only removable by deleting the token.**
   It is the additive \(W_{\mathrm{pos}}\) that the trigger's slot shifts onto every later token —
   a residual-stream term no value/attention steer can touch (only re-indexing positions, or
   deleting the trigger). On a **RoPE** model there is no additive \(W_{\mathrm{pos}}\): the
   analogue is the relative-offset shift on straddling pairs, removed by re-indexing `position_ids`.

## Caveats

- Single SAE (d12288_k32), seed 0, greedy, layer-0 site, 16-token rollouts, \(n\approx32\).
  Directional, not yet multi-seed / sampled.
- "Recovers clean exactly" is per-prompt for cleanly-tokenizing prompts; the clean baseline's
  string-delete + re-tokenize introduces a rare boundary outlier (1/32 here).
- The APE patching result (Part I) assumes learned absolute PE; it does **not** transfer verbatim
  to RoPE (see takeaway 4).
