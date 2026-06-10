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

**The win (one sentence).** At *matched* induction suppression, a content-addressed FRA-QK edit
suppresses a cue's induction **association** with **~15× less collateral** on that cue's behavior in
other contexts than the strongest *fair* linear steer — and the gap survives every check I could
throw at it, because it follows directly from FRA's double gate (the edit fires only where the query
*and* the key feature are both present).

### Key findings

1. **FRA-QK suppresses an association at ~15× less collateral than the fair linear baseline** (Fig 1,
   `fig1_final.png`). At matched 50% induction suppression, held-out collateral (KL on normal text
   mentioning the cue) is **FRA 0.18 ± 0.04 nats** vs **2.66 ± 1.35** for an *induction-gated* ActAdd
   (a probe-gated linear steer that fires only at induction-context cue positions — the strongest
   linear baseline I could construct) and **3.27 ± 1.83** for plain ActAdd. FRA's curve hugs the
   floor at every suppression strength; the linear steers climb steeply. A broader sweep over 8 cues
   corroborates (FRA 0.14 ± 0.03 vs ActAdd ~3.0). The clean read: to break the association a linear
   steer must corrupt one of its endpoints *everywhere that endpoint occurs* — even gated to induction
   positions it wrecks the *whole residual* there (all heads); only the bilinear edit touches the two
   endpoints *jointly* and so touches almost nothing else.

2. **The double gate makes the edit surgical** (Fig 2, `fig2_locality.png`). On the induction
   sequence, at matched suppression, FRA changes the output essentially only at the target attention
   edge (off-target KL mass 0.62) while ActAdd's residual perturbation spreads to every cue occurrence
   and downstream (7.6 nats, **12× more**). On a paragraph where the cue (" war") appears 5×, FRA
   touches only the induction-like edges (KL 0.13) vs ActAdd 31 (**~235×**). The asymmetry is
   structural: FRA's edit fires only where query=cue AND key=prev-was-cue.

3. **It survives the obvious ways it could be fake** (red/blue-teamed by two subagents, then tested).
   *(a) Support not rigged:* applying the edit content-addressed over the whole sequence (pairs fire
   wherever they appear, no edge whitelist) gives 0.18 ≈ the edge-restricted 0.14 — the pair's natural
   support *is* the induction pattern. *(b) Not an over-drive artifact:* it reaches strong suppression
   at scale **c≈2, near exact edge reconstruction**, not extreme over-drive. *(c) Right pairs matter:*
   random feature-pairs at the same scale suppress nothing.

4. **The win is narrow, and I am not hiding it.** On the cruder metric of *other tokens'
   copy-probability*, a position-targeted ActAdd ties or beats FRA — FRA's edge lives specifically in
   *held-out* collateral. FRA has a **suppression ceiling** (it removes only the FRA-explained part of
   an edge; one of eight cues capped at 0.17, where ActAdd always reaches ~1.0). The task is
   **deliberately association-shaped** (an existence proof, not a survey of natural behaviors), and
   this does **not** revive FRA-OV for backdoor removal — it is a separate, specific niche.

**Takeaway.** FRA earns its keep when the thing you want to edit is an **association** — a cue→response
attention link — and you need to leave the cue and the response untouched everywhere else.
Conditioning an intervention on a *pair* of features is the one thing FRA offers that no linear method
can, and induction is the clean demonstration.

![Fig 1 — the win](figures/fig1_final.png)
*Fig 1. Collateral = KL(clean‖edited) on held-out normal text (nats). Left: each faint line is one
cue (n=4); x = how much the cue's induction is suppressed (→ stronger), y = collateral (log, ↓
better), so bottom-right is ideal. FRA-QK (blue, content-addressed) stays at ~0.2 nats at every
strength; the fair induction-gated ActAdd (red) and plain cue-identity ActAdd (orange) climb to
1–40 nats. Right: read off at a matched 50% suppression — FRA beats the fair baseline ~15×, and the
content-addressed FRA (solid 0.18) ≈ the edge-restricted version (hatched 0.14), so the low
collateral is not an artifact of a hand-picked edit support. Large ActAdd error bars are real
cross-cue variance; FRA's are negligible at this scale.*

![Fig 2 — locality](figures/fig2_locality.png)
*Fig 2. Why: at matched suppression FRA's edit stays local (left: off-target KL on the induction
sequence; right: total KL on a paragraph where the cue appears 5×), while the linear steer's residual
perturbation spreads to every cue occurrence.*

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
- **Intervention = a content-addressed attention-score edit.** For a target edge, select its
  dominant feature-pairs `(i,j)` (top-12, or all pairs); subtract `c × Σ FRA[·,·,i,j]` from the
  induction heads' pre-softmax scores. Because the FRA term is built from the *actual* feature
  activations, this delta fires wherever features `i` (query) and `j` (key) co-occur — it is
  **content-addressed, not position-addressed**, and **doubly gated**. `c` scales it (FRA selects
  *what*; `c` controls *how hard*, like α in ActAdd). The honest "content-addressed" deployment (J9)
  recomputes FRA on held-out text and subtracts the selected pairs *wherever they fire on the full
  `[q,k]` matrix* — `c≈1.5` exactly reconstructs the edge; the result holds at `c≈2`.
- **Baselines:** ActAdd of the cue's identity direction (subtract a mean-centered cue residual
  direction wherever the cue is current); ActAdd of the key-side "prev-was-cue" direction;
  **induction-gated ActAdd** (the identity steer applied only at induction-context cue positions — a
  probe-gated linear steer, the strongest fair linear baseline); head mean-ablation (non-selective);
  position score-patch (selective oracle, needs positions); random-pair edit (null).
- **Metrics:** induction suppression = `1 − P(response)/P_base`; collateral = KL(clean‖patched) of
  the full next-token distribution on held-out normal text containing the cue (matched at equal
  suppression via the Pareto, so no method is advantaged by under/over-suppressing).

## 3. The road here (what surprised me)

- **Single-head edits do nothing** (J2). Ablating one head's edge — or even mean-ablating L5H5
  entirely — barely moves copy-probability, because induction is **redundant** across ≥5 heads and
  many edges are softmax-saturated. *Fix:* intervene on all induction heads + over-drive.
- **Reconstruction is partial** (corr ~0.5 on a random-token edge; ~60–75% on the cue primers in J9,
  where GPT-2 LayerNorm centering is dropped by the RMS correction). This matters less than it looks:
  behavior is set by the **softmax**, and removing most of a +6 score collapses an attention edge from
  0.94 to ~0.13. Crucially the win does **not** depend on heavy over-drive — it holds at `c≈2`, near
  the scale that exactly reconstructs the edge (J9).
- **The naive linear baseline is strong** (J4). For *within-task* selectivity, ActAdd-of-the-cue
  matches FRA. The real distinction only appears once you ask about the cue's behavior **elsewhere**
  (J5–J9) — which is the whole point of the double gate.

## 3b. Cross-model check (Gemma-2-2b)

To make sure the win is not a GPT-2/SAE-reconstruction artifact, I repeated it on **Gemma-2-2b**
(RMSNorm, so the FRA RMS correction is *exact*) with public **GemmaScope** resid SAEs (J11–J12).
Two things came out, one reassuring and one honest:

- **The surgical-collateral property replicates cleanly.** On a different architecture, FRA's
  held-out collateral stays **0.04–0.5 nats** at every edit strength, while ActAdd needs **3–57 nats**
  to achieve comparable suppression (induction-gated ActAdd ≈ 1.8 nats at the one cue where FRA
  reaches 50%, vs FRA 0.14). The ~10–100× precision gap is architecture-independent. The FRA edge
  *magnitude* reconstructs near-exactly here (9.7 vs 9.1, ratio 1.07), confirming the GPT-2 ~50%
  under-estimate was the LayerNorm-centering issue — though the per-edge *correlation* is still ~0.54
  (bias terms + Gemma's attention soft-cap + SAE sparsity), a reminder that "FRA explains the score"
  is itself only ~half-true and worth its own study.
- **FRA's suppression ceiling is lower on Gemma** (it reached ≥50% suppression on only 1 of 4 cues,
  targeting 4 induction heads). Gemma's induction is more distributed across heads, and the attention
  soft-cap compresses score edits, so the same edit moves the pattern less. The bilinear edit's
  *precision* transfers; its *reach* depends on architecture and how completely you cover the heads.

## 4. Limitations & what I would do next

- **The win needs the attention edge to be causally load-bearing — and that is restrictive** (J13,
  honest negative). I tried to transfer the win to **IOI**, a canonical natural attention task:
  ablating the FRA-identified END→IO feature-pairs of the three main name-mover heads (L9H9/L9H6/
  L10H0) moved the IO−S logit difference by only **0.17** (3.88→3.70), while ActAdd brute-forces it
  (flips it negative). IOI's **backup name-mover heads** compensate as soon as the main circuit's
  attention is suppressed. So the win held for induction-copy (causally load-bearing at the token
  granularity once you hit all the redundant heads) but **not** for IOI — sharpening the scope: FRA-QK
  buys you precise, selective control of an attention edge, but only changes *behavior* where that
  edge is actually necessary and not silently backed up.
- **Scope is an existence proof.** The induction target is deliberately shaped like an attention
  edge, the object FRA represents natively. It cleanly demonstrates the bilinear handle exists and is
  useful; whether *natural* associations that are both load-bearing and FRA-addressable are common is
  open. I show the mechanism, not its prevalence.
- **One model, partial reconstruction.** GPT-2-small only; res-jb resid SAE gives ~60–75% edge
  reconstruction (LayerNorm centering dropped). Gemma-2-2b + GemmaScope (RMSNorm → *exact* FRA RMS
  correction) is the clean follow-up to remove the caveat — not run here.
- **FRA's suppression ceiling.** It can only remove the FRA-explained part of an edge, so for some
  cues it cannot reach full suppression (one of eight capped at 0.17). ActAdd always reaches ~1.0.
  For a use that needs *complete* removal, that is a real gap.
- **Statistics.** n = 4–8 cues, single random seed; std bars are over cues, not seeds. The
  *direction* of the result is consistent everywhere, but the exact ratios (15×, 235×) should be read
  as order-of-magnitude.
- **Baselines.** Identity, key-side, and induction-gated ActAdd were tested; a *learned* probe-gated
  steer or a directional projection might narrow the gap further. The structural argument (no
  single-endpoint linear steer can be both content-addressed and association-specific) says it cannot
  close, but that is an argument, not an exhaustive search.

## 5. Research map

`RESEARCH_LOG.md` has the timestamped trail (the path was not linear — two reframes). Jobs:
**J1** induction-head pick + FRA reconstruction + dominant-pair structure; **J2** single-head
selectivity (*negative* → induction is redundant); **J3** all-heads + over-drive (first signal of
selective suppression); **J4** baselines on the within-task Pareto (*ActAdd ties* → reframe to broad
collateral); **J5** the double-gate collateral win; **J6** robustness over 8 cues; **J7** key-side
baseline + position-split; **J8** the 8-cue Pareto; **J9** the fairness corrections the
red-team demanded (content-addressed FRA, induction-gated ActAdd, faithful-`c`) — *the win survived*;
**J10** final figure; **J11–J12** Gemma-2-2b cross-model check (precision replicates, reach is lower);
**J13** IOI transfer attempt (*negative* — backup name-movers defeat it; sharpens scope).
The writeup was red/blue-teamed by two subagents (one attacking the claim, one verifying numbers vs
code) before this revision; a third, context-free agent cold-read the money figure to check clarity.
