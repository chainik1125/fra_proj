# Bridge: toy optimal-FRA theory → real models

*Autonomous 5h push, started 2026-07-17 ~07:21 UTC. User away ~5h wall clock.*

## The thesis

The toy program (`../bernoulli_fra/OPTIMAL_FRA_NOTE.md`, §8) produced the **amended
fra_win law**, verified on theory-clean toys:

> A two-position PRODUCT is necessary but not sufficient for an FRA-QK handle.
> A content GATE is OV-realisable; the trained model puts it in OV (or LayerNorm),
> not QK. **FRA-QK is the carrier only for an obligate query-key MATCH**
> (induction). Gates/aggregates are OV/content-carried → FRA-QK inert there.
> Corollary (carrier→tool): the winning control method is set by the carrier —
> FRA-QK for matches, FRA-OV/DoM for content.

The existing `experiments/fra_win/` campaign already OBSERVED exactly this pattern
in GPT-2 and **Gemma-2-2b**, but never connected it to a theory that predicts it:
- in-context / attention-routed backdoor (induction = a MATCH) → **FRA-QK wins**
  (~60× lower held-out collateral than DoM & conv-SAE).
- weight-baked sleeper (payload in OV/output = CONTENT) → **FRA-QK loses**, DoM/SVD win.

**So the toy theory retro-predicts the whole fra_win/sleeper empirical split from
process structure.** This push makes that bridge explicit, tests its sharpest new
prediction on a real model, and extends toward LM tasks + sleepers.

## Bridge 1 — toy ↔ Gemma-2-2b induction backdoor (the core)

Claim to establish: the toy's carrier-localization methodology (QK-cut vs OV-cut
asymmetry + G1 gauge-robustness) and control-frontier (FRA vs DoM vs SAE at matched
removal), run on a real Gemma-2-2b induction backdoor, reproduce the toy's
signatures — QK-carried, gauge-robust, FRA-QK wins the control frontier — while the
weight-baked case is the OV-carried mirror. This turns the amended fra_win law into
a cross-scale law (toy → 2B params).

- Theory half [bridge-theory, no compute]: write the mapping — toy objects ↔ real
  FRA objects (GemmaScope SAE, RMSNorm, attn softcap); why induction is the
  obligate match; predictions the real model must satisfy.
- Empirical half [compute, Modal/Gemma]: the carrier test + control frontier on a
  planted in-context induction backdoor; compare to fra_win's existing numbers.

## Bridge 2 — → language-model tasks

Beyond a planted backdoor: pick a NATURAL LM behavior that is an obligate match
(acronym letter-movers / retrieval, per fra_win) and one that is a content-gate,
and show the QK-vs-OV carrier prediction + which control method wins holds on
natural text. Deliver the decision rule: "given a behavior, is it a match (edit
QK) or a gate (edit OV/content)?" as a practical FRA usage guide.

## Bridge 3 — → sleepers via special-token finetuning (stretch, user's idea)

Finetune Gemma on a special token → emit a string associated with an UNSEEN
process (a held-out generative process the string encodes). Question: is the
learned trigger→payload a MATCH (QK, FRA-QK removable) or a CONTENT route (OV,
DoM removable)? Prediction from the law: a finetuned special-token→fixed-string
map is an OV/content route (like the weight-baked sleeper) → DoM/OV wins, FRA-QK
loses — UNLESS the payload is retrieved by an in-context match. This directly
tests the law's boundary on a trained (not planted) backdoor.

## Compute

Modal (serverless GPU, HTTPS — survives me being the only babysitter), Gemma-2-2b
(access confirmed, A10G/L4 enough). Checkpoints/artifacts → private HF dataset
`dmanningcoe/sprint-fra-theory` under `bridge_to_real/`. Local keeps notes/JSON/PNG.
If Modal is down (June cap), fall back to the theory bridges + re-analysis of
existing fra_win/out numbers (no new compute) — the theory bridge stands alone.

## Deliverables (in this dir)

- `BRIDGE_NOTE.md` — the unifying write-up (theory + real evidence).
- `bridge1_carrier.{json,png}`, `bridge1_frontier.{json,png}` — Gemma results.
- `RESEARCH_LOG.md` — continuous, dead ends included.
- merge pointer into `../bernoulli_fra/OPTIMAL_FRA_NOTE.md` §9 at the end.
