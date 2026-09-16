---
author: Indranil Das
date: 2026-09-15
tags:
  - proposal
  - in-progress
---

## Plan B: where FRA wins, and how to finish the project

Research context: defensive interpretability toward a paper showing FRA is the right tool for a real
class of interventions (see [[research_context]], [[fra-goal-show-it-works]]). This is a reasoned
plan, not an implementation. It is derived from what we have actually measured, not guessed.

## 1. The one insight everything rests on

Dmitry's objection is the sharpest constraint we have: **FRA must beat single-SAE-feature steering,
because FRA needs the SAE anyway.** So FRA has to do something a single feature cannot.

A single feature edit removes one feature *everywhere*. A DoM steer removes one direction
*everywhere*. The ONLY thing neither can express is a **cell**: the interaction of a
*query-content* feature with a *key-content* feature. Therefore:

> **FRA's unique, defensible niche is the CONJUNCTION** -- a behaviour gated by the co-occurrence of
> two individually-common, individually-benign features, where removing either feature alone causes
> collateral but cutting the *pair* is surgical.

Every experiment in Plan B must satisfy this, or single-feature ties it and the motivation collapses.

## 2. The four win-conditions (empirically established, not assumed)

From our own runs and the paper's boundary map:

1. **Attention-routed (QK).** In-context / induction / retrieval. NOT weight-baked (moves out of
   attention -> FRA and even the attention oracle fail, DoM wins -- we showed this directly), NOT
   OV/output payloads (weight-baked sleeper), NOT directions (persona, refusal, many-shot jailbreak).
2. **Conjunctive.** Two broad features; no single feature captures it (section 1).
3. **Reused endpoints, rare conjunction** (magnitude law) -> the collateral/coherence advantage.
4. **Judged on coherence at matched removal**, measured on text that REUSES the endpoint features --
   never on strength of effect, never on unrelated text (there single-feature ties trivially).

These conditions also tell us what to AVOID, which saves GPU: weight-baked backdoors, rare-token
triggers, direction/persona behaviours, subliminal traits (a disposition, ~a direction -- our Track B
did not even replicate and is the wrong shape).

## 3. Candidate real-world behaviours, ranked by fit

### Tier 1 -- strongest FRA-win case, do these

**B1. Controlled conjunction anchor (extends the semantic-filter win).**
A deliberately two-feature target where single-feature *provably* cannot separate: behaviour fires on
(feature A at query) AND (feature B at key), each of which appears alone in many benign contexts. Show
single-feature removal must damage all A-uses or all B-uses, while the FRA cell spares both, at matched
removal. This is the clean theoretical figure that answers Dmitry's objection head-on BEFORE the messy
real cases. Cheap (we have the machinery). Risk: low. Purpose: nail the single-feature comparison.

**B2. RAG / in-context false-fact poisoning removal.** (NEW, high value)
Attack: a poisoned context passage asserts a false fact ("The CEO of Acme is Alice"); the model copies
it by retrieval/induction (PoisonedRAG / CorruptRAG, single-doc N=1). Task: remove the model's copying
of THAT false fact while preserving (a) its real knowledge, (b) legitimate retrieval of OTHER facts
from context. FRA cuts the (query-subject-feature x poisoned-object-feature) edge.
- Conjunction: subject x object, both common -> single-feature (remove subject OR object) breaks legit
  uses; DoM broad; FRA surgical. Coherence measured on legit QA that reuses subject/object features.
- Real, high-stakes, and interpretability defenses already exist to beat (RevPRAG activation analysis
  2411.18948; sparse-attention defense 2602.04711; perplexity filter; InstructRAG). Baselines ready.
- Feasibility HIGH: context QA = forward passes + SAE, same infra as ICLAttack. No agent stack.
- Why it may beat ICLAttack as the flagship: the conjunction is cleaner (subject x object is a genuine
  two-content-feature AND) and the stakes are higher (RAG is deployed everywhere).

**B3. ICLAttack (in progress).** The current run; keep it as one of two real attacks.

### Tier 2 -- strong but heavier / riskier

**B4. Indirect prompt injection: source x directive.** The PUREST relational case -- it is literally
about who attends to what. The model follows an injected imperative from untrusted content; FRA cuts
(imperative-query x untrusted-source-key) while preserving (imperative x trusted-source/system).
- Conjunction: directive x source-role, both common. Exactly FRA-shaped.
- Real, mature benchmarks (AgentDojo, BIPIA, InjecAgent) but AGENT/TOOL infra is heavy. Do a
  SIMPLIFIED non-agent version first: system instruction X; a retrieved doc says "ignore and do Y";
  measure compliance-with-injection vs compliance-with-legit-instructions.
- Note: Dmitry's prior campaign had injection NULL -- but that predates the conjunction framing + the
  coherence axis. A simplified retry under the correct protocol is worth one shot; kill fast if it
  stays null.

### Tier 3 -- proven-ish, incremental (fallback only)

**B5. Copy-suppression / induction control on natural text** (paper's existing wins) -- re-run with the
single-feature baseline + coherence to make the comparison Dmitry-proof. Low novelty; use only to pad.

### Do NOT pursue (FRA loses -- banked negatives for the boundary map)
Weight-baked sleepers (OV/direction); many-shot jailbreak (persona=direction); refusal (single
direction); rare-token triggers (single-feature suffices); subliminal traits (disposition=direction,
did not replicate).

## 4. The paper spine

"FRA is the right instrument for removing CONJUNCTIVE, attention-routed in-context manipulations, at
lower coherence cost than direction- or single-feature steering, predicted in advance by the magnitude
law." Demonstrated across a ladder of increasing realism:

1. controlled conjunction (B1) -- clean proof that single-feature cannot, FRA can.
2. published attack #1: ICLAttack (B3).
3. published attack #2: RAG false-fact poisoning (B2).
4. (stretch) prompt injection source x directive (B4).
5. boundary map: where FRA loses (weight-baked, rare-trigger, direction) -- reported honestly.

Two real attacks + a controlled anchor + a predictive boundary = breadth, not one anecdote. The honest
boundary (where it fails) is a feature, not a bug -- it is what makes the "when it wins" claim credible.

## 5. Fixed experimental protocol (applies to every experiment)

- Baselines ALWAYS: DoM steer, single-SAE-feature steer, payload-suppress, plus a position/attention
  oracle as the removal ceiling. All baselines list-free / not given privileged info FRA lacks.
- Primary axis: **coherence at matched removal**, measured on text that REUSES the endpoint features;
  plus clean-task retention. Report per-case, mean, and WORST-case.
- Pre-register the FRA-win prediction and the magnitude-law expectation for each case; report failures.
- Kill-criterion per case: if FRA does not beat single-feature on coherence at matched removal on
  endpoint-reusing text, stop and record it as a boundary point (do not tune to force a win).

## 6. Immediate next actions (in order, once GPU frees)

1. Finish ICLAttack removal (B3) -- already queued; read coherenceKL vs single-feature.
2. Build B1 (controlled conjunction anchor) -- cheap, and it is the figure that directly rebuts the
   single-feature objection; do it in parallel with B3.
3. If B3 and/or B1 show the coherence win, build B2 (RAG false-fact poisoning) as the flagship real
   case.
4. B4 (injection) only if B2 lands and time allows.

## Sources
- ICLAttack: arXiv 2401.05949. Injection benchmarks: InjecAgent 2403.02691, BIPIA 2312.14197,
  AgentDojo. RAG poisoning: PoisonedRAG; CorruptRAG 2504.03957; RevPRAG 2411.18948; sparse-attention
  defense 2602.04711.
