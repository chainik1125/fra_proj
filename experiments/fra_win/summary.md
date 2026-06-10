# Where Feature-Resolved Attention wins: bilinear control of attention edges

*GPT-2-small · induction · 10h sprint · 2026-06-10. Research log: `RESEARCH_LOG.md`. Code:
`jobs/j*.py` (run on a RunPod GPU via an HF inbox/outbox harness). All figures regenerated
from `fra_win/out/`.*

---

## Executive summary

**The question.** Feature-Resolved Attention (FRA) decomposes an attention score into a sum of
**(query-feature × key-feature)** terms using an SAE basis and the model's own Q/K weights. It is a
beautiful microscope — but is it ever the *right tool for a behavioral intervention*? My prior
campaigns on sleeper-backdoor removal found the OV side of FRA **dominated**: a mean-difference
steering vector (with examples) or a rank-2 SVD of the weight diff (without) removed the backdoor at
lower cost. So the honest question for this sprint was: **is there any context where FRA gives a
clear behavioral win over simple baselines?**

**The thesis.** FRA's mathematically unique part is the **QK (bilinear) decomposition**. A linear
residual steer (ActAdd / mean-difference / CAA) adds a vector to the stream; it can shift a query or
bias a key, but it **cannot express "change attention specifically from queries-with-feature-A to
keys-with-feature-B"** — that is a rank-1 update to W_QK, inherently bilinear. So FRA should win
exactly when the behavioral target is an **association** (an attention edge), not a single feature.

**The testbed.** Induction in GPT-2-small (`A B … A → B`, head L5H5 et al.). Behavioral task:
**suppress the model copying one specific cue token**, while leaving everything else intact.

**The win (one sentence).** At *matched* induction suppression, an FRA-QK edit suppresses a cue's
induction **association** with essentially **zero collateral** on that cue's behavior in other
contexts, whereas the best content-addressed linear steer corrupts the cue **everywhere** — a
**~100–700× collateral gap** that follows directly from FRA's double gate (it fires only where the
query *and* the key feature are both present).

### Key findings

1. **FRA-QK suppresses an association at ~0 collateral; the linear steer cannot** (Fig 1). Across 8
   cue words, at 50% induction suppression FRA's held-out collateral is **0.00 ± 0.00 nats** vs
   ActAdd's **0.74 ± 0.30**. The mechanism is visible in the number: normal text mentions the cue
   without a second occurrence, so the induction edge never forms and **FRA's edit never fires** —
   it is inert exactly where it should be. The linear steer fires on the cue's mere presence and
   wrecks its next-token prediction.

2. **The double gate makes the edit surgical** (Fig 2). On the induction sequence itself, at matched
   suppression, FRA changes the model's output essentially only at the single target attention edge
   (off-target KL mass 0.62) while ActAdd's residual perturbation spreads to every occurrence of the
   cue and downstream (7.6 nats, **12× more**). On a paragraph where the cue appears 5×, FRA touches
   only the induction-like edges (KL 0.13) while ActAdd corrupts all of it (KL 31, **~235×**).

3. **The win is specifically about *association* selectivity — and it is narrow** (honest scope).
   Measured by the cruder metric of *other tokens' copy-probability*, a position-targeted ActAdd
   **ties or beats** FRA; FRA's advantage appears only on **broad / held-out collateral**, which is
   where the bilinear gate matters. The win also required hitting **all** induction heads with an
   **over-drive** scale, because single-head induction is redundant. This does **not** revive FRA-OV
   for backdoor removal — it is a different, specific niche.

**Takeaway.** FRA earns its keep when you want to edit a **cue→response association** without
touching the cue or the response elsewhere. Conditioning an intervention on a *pair* of features is
the one thing FRA offers that no linear method can, and induction is the clean demonstration.

---

## 1. Why this question, and why it was open

The interpretability-for-control pitch for SAEs and FRA is that feature-level / attention-level
handles give *precise* behavioral control. My earlier sleeper-removal campaigns punctured the OV
version of this: for "remove a backdoor direction," a mean-difference vector and a rank-2 SVD both
beat FRA-OV on the damage-vs-removal frontier. The natural rescue is that FRA's **QK** side — the
part that has no linear analogue — was never given a fair behavioral test (only a *first-order
attribution* test, which fails on saturated attention). This sprint tests the QK side directly.

## 2. Setup (so you can reproduce)

- **Model:** GPT-2-small via TransformerLens. **SAE:** public `gpt2-small-res-jb` on
  `blocks.{L}.hook_resid_pre` (no training). **FRA:** the repo's `fra/core/fra.py` 4-D tensor
  `FRA[q,k,i,j]` (the bilinear decomposition), summed over feature dims to recover scores.
- **Induction heads** (picked by attention on the induction edge of a random repeated-token
  sequence): L5H5 (0.94), L6H9, L5H1, L7H10, L7H2.
- **Intervention = a content-addressed attention-score edit.** For a target edge, take its top-12
  feature-pairs `(i,j)`; subtract `c × Σ FRA[·,·,i,j]` from those heads' pre-softmax scores. Because
  the FRA term is built from the *actual* feature activations, this delta fires wherever features
  `i` (query) and `j` (key) co-occur — i.e. it is **content-addressed, not position-addressed**, and
  **doubly gated**. `c` is an over-drive scale (FRA selects *what* to suppress; `c` controls *how
  hard*, exactly like α in ActAdd).
- **Baselines:** ActAdd of the cue's identity direction (subtract the cue's contrastive residual
  direction wherever the cue is current); ActAdd of the key-side "prev-was-cue" direction; head
  mean-ablation (non-selective reference); position score-patch (selective oracle, needs positions);
  random-pair edit (null).
- **Metrics:** induction suppression = `1 − P(response)/P_base`; collateral = KL(clean‖patched) of
  the full next-token distribution on held-out normal text containing the cue.

## 3. The road here (what surprised me)

- **Single-head edits do nothing** (J2). Ablating one head's edge — or even mean-ablating L5H5
  entirely — barely moves copy-probability, because induction is **redundant** across ≥5 heads and
  many edges are softmax-saturated. *Fix:* intervene on all induction heads + over-drive.
- **Reconstruction is only ~50% on the induction edge** (res-jb resid SAE; GPT-2 LayerNorm centering
  is dropped by the RMS correction). This matters far less than it looks: behavior is set by the
  **softmax**, and removing the FRA-explained part of a +6 score collapses an attention edge from
  0.94 to ~0.13. I over-drive to compensate and report the reconstruction honestly.
- **The naive linear baseline is strong** (J4). For *within-task* selectivity, ActAdd-of-the-cue
  matches FRA. The real distinction only appears once you ask about the cue's behavior **elsewhere**
  (J5–J7) — which is the whole point of the double gate.

## 4. Limitations & what I would do next

- One model (GPT-2-small), one mechanism (induction). The argument is general (bilinear > linear for
  association control) but a second model — ideally Gemma-2-2b + GemmaScope, where the FRA RMS
  correction is *exact* — would remove the approximate-reconstruction caveat. *(In progress / TODO.)*
- The cue→response association in induction is a clean but slightly artificial behavioral target. A
  more natural association (e.g. a factual or grammatical attention link) would broaden the claim.
- I have not searched for an even stronger linear baseline beyond identity / key-side ActAdd; the
  structural argument says none can be *both* content-addressed and association-specific, but that is
  an argument, not an exhaustive search.

## 5. Research map

`RESEARCH_LOG.md` has the timestamped trail. Jobs: **J1** induction-head pick + FRA reconstruction +
dominant-pair structure; **J2** single-head selectivity (negative → redundancy); **J3** all-heads +
over-drive (signal: selective suppression); **J4** baselines on the within-task Pareto (ActAdd ties →
reframe); **J5** double-gate collateral (the win); **J6** robustness over 8 cues (the money Pareto);
**J7** key-side baseline + non-induction-position collateral split (hardening).
