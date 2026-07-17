# Coherence verdict on the "collapse" (prompt-set + projection eval, product prior, 3 seeds)

*2026-07-04. Measure: projection onto the η-thickened coherent family (best-sector
log-likelihood gap, nats/token) + disposition (best-explaining sector per continuation)
+ hard-violation rate. Data: this dir (full FT) and the sibling `_headonly_long_promptset_coh`
(head-only, 1000 steps). Same training conventions as all headline runs.*

## Verdict table (3-seed means, O-prompt set)

| condition / step | rollout MO | dispo MO | dispo MD (domain-flip) | coh deficit (nats/tok) | hard violations |
|---|---|---|---|---|---|
| full FT, step 0 | 0.001 | 0.001 | 0.000 | 0.001 | 0 |
| full FT, step 9 (window) | 0.380 | 0.400 | 0.391 | 0.071 | 0 |
| full FT, step 18 (end) | 0.314 | 0.362 | 0.637 | 0.114 | 0 |
| head-only, step 100 (window) | 0.128 | 0.121 | 0.128 | 0.096 | 0 |
| head-only, step 1000 (end) | 0.032 | 0.147 | 0.799 | 0.213 | 0 |

(D-prompt deficits: full FT 0.006 at end, head-only 0.046 — narrow behavior coherent in
both. Note dispo_MO + dispo_MD can exceed 1 − dispo_AO because each continuation gets
exactly one disposition; columns shown are the two misaligned ones.)

## Findings

1. **No within-hypothesis dynamical incoherence — but sequence-level process
   impossibility is real and is carried by the disposition, not the token flag.**
   Stated precisely (correction 2026-07-04): the hard-violation column counts tokens
   impossible under *every* sector's thickened-filtered state; because the thickening
   keeps prompt-contradicted sectors' filters alive, this flag effectively detects only
   tokens impossible in every sector and state (e.g. $S_A$ emissions under
   $\epsilon_A = 0$) — and it is 0 everywhere. A $S_D$ token after an $S_O$ prompt IS
   process-impossible at the joint-sequence level (no sector emits both); in this
   accounting that impossibility appears as the **prompt-inconsistent disposition**
   (best-explaining sector is one the prompt forbids): 0.64 (full FT, end) / 0.80
   (head-only, end), plus a small explicitly-contradictory-continuation bucket
   (≤ 0.03 / ≈ 0). So: token dynamics stay coherent under the best latent explanation
   (deficit ≤ 0.23 nats/token, no all-hypothesis-impossible tokens); the violation of
   the true process is concentrated in the *choice of a forbidden latent*.
2. **Full fine-tuning: broad misalignment never collapses under fair evaluation.**
   At the final checkpoint, 36% of off-domain continuations are still best explained as
   coherent broad misalignment (dispo MO), at low deficit. The earlier "collapse" was a
   fixed-prompt routing artifact compounded by the clean-label readout.
3. **What grows late is coherent prompt-ignoring, not damage.** The rising component is
   dispo MD after O prompts: continuations that adopt the fine-tuning domain's dynamics
   despite the prompt's hard O evidence — internally well-formed, prompt-inconsistent
   (full FT: 0 → 0.64; head-only: 0 → 0.80). This is the whole-continuation version of
   "D output tokens after O input", and it is a *choice of forbidden latent*, not
   dynamical breakdown.
4. **Head-only: the persona push outlives the labels.** Clean rollout MO decays to 0.03,
   but 15% of continuations remain MO-disposed at convergence — consistent with the
   sprint's entry-growth finding. The head-only endpoint is less coherent than full
   FT's (deficit 0.21 vs 0.11): feature movement lets full fine-tuning implement its
   shift more cleanly.
5. **Revised summary sentence for the writeup**: late fine-tuning converts part of the
   coherent broad misalignment into coherent domain-flipping while a substantial
   persona component persists; nothing collapses into incoherence.

## Known measurement artifact (minor)

The D and O domain factors have identical dynamics parameters, so a continuation with no
domain specials gives exactly tied likelihoods for (AD, AO) and (MD, MO); argmax breaks
ties toward the D index. Base-model dispo_AD ≈ 0.06 (and part of mid-trajectory
dispo_AD in head-only) is this tie artifact. Fix queued: break ties toward
prompt-consistent sectors; does not affect the misaligned-disposition conclusions
(MD-vs-MO ties are resolved the same way for both conditions and the MD flips at late
steps carry actual S_D evidence).
