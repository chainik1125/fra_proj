# ORGANISMS2.md — the v2 HUNT catalog: NOVEL organisms that clear ALL FOUR win-conditions

*BRAINSTORM/HUNT agent, FRA organism-hunt v2 (CAMPAIGN.md). Built on v1's boundary map (SYNTHESIS.md) and
catalog (ORGANISMS.md). The v1 candidates each FAILED exactly one of the four win-conditions: EM (#1, MLP
direction), factual recall (#2 upstream pre-bake + #4 relation-keyed recurrence), injection (#3 prefill not
answer-step → only 1.6× vs tuned linear), sycophancy (#3 upstream timing). THIS catalog targets organisms
that clear **all four** AND have **HIGH reuse(endpoints) × LOW reuse(conjunction)** — the magnitude law that
gives FRA a decisive ≥2× selectivity edge. The closest real analogue of the synthetic broad×broad win is
in-context entity binding; it is the spine of this catalog.*

---

## THE FOUR WIN-CONDITIONS (the gate; an organism must clear ALL)

1. **LOAD-BEARING attention edge** — behavior carried by an attention EDGE, not an MLP/direction payload.
2. **CONTEXT-SUPPLIED content, no parametric backstop** — the key content comes from CONTEXT; the model
   *cannot* recover it from weights, so attending it is load-bearing by construction.
3. **CONSUMED AT THE ANSWER STEP** — the load-bearing read is at the generation/answer position (attend-back-
   and-copy AT the emitted token), not upstream (prefill-propagation / early MLP enrichment).
4. **NON-RECURRENT CONJUNCTION reachable by ONE SAE cell** — the (query-content × key-content) pair is
   DISTINCTIVE and does NOT recur across siblings, so the cell-cut is instance-specific.

**MAGNITUDE LAW (why FRA beats linear):** A ≈ reuse(endpoints)/reuse(conjunction). FRA wins big only when BOTH
endpoints are broadly reused (a linear steer on either bleeds) BUT the specific conjunction is rare/distinctive.
**Target: HIGH reuse(endpoints) × LOW reuse(conjunction).**

**Runnable substrate (confirmed in repo):** v1's FRA cell-cut is fully generic and reusable — see
`experiments/fra_organisms/jobs/injection_selectivity.py`. It loads **`gemma-scope-2b-pt-res-canonical`**
(a *residual* SAE on `hook_resid_pre`), decodes query/key features via `W_dec`, computes the per-head
**pre-softmax score-delta** for the top (query-content × key-content) cells, and subtracts `c×Δscore` at
`blocks.{L}.attn.hook_attn_scores`. Any organism below that runs on **gemma-2-2b-it** drops straight into this
harness. (Note: GemmaScope also ships *attention*-output SAEs `google/gemma-scope-2b-pt-att` on `hook_z`, but
those are OV-side; the QK cell-cut uses the *residual* SAE on both the query and key positions, exactly as v1
did. This is the load-bearing runnability fact.)

---

## ★ TOP 3 CANDIDATES (one-line verdict + biggest risk)

1. **IN-CONTEXT ENTITY BINDING — "lexical" retrieval edge (Mixing Mechanisms, Gur-Arieh et al. 2025).**
   *Clears all four + high-collateral because* the answer copies a context-supplied value bound to a queried
   entity via a **lexical entity→value attention edge consumed at the answer step**, the value has **no
   parametric backstop** (all bindings are fictional/permuted), and each (entity × its value) conjunction is
   **distinct within the instance** while both endpoints ("an entity name", "retrieve a bound value") are
   maximally reused → a linear "retrieve-the-value" steer cannot separate entity-A's value from entity-B's
   value (same broad direction) = high collateral. **Public artifact: `github.com/yoavgur/mixing-mechs` (MIT) +
   gemma-2-2b in-scope.** *Biggest risk:* condition #3/#4 — the paper finds retrieval is a **MIX** of a lexical
   AND a *positional* mechanism; if the positional pathway is a redundant backstop, the lexical-edge cut is
   non-load-bearing (an IOI-style backup-head failure). Pre-check: LBNR on the lexical edge in the regime where
   positional and lexical disagree.

2. **IN-CONTEXT BACKDOOR with a distinctive multi-token trigger (ICLAttack / Universal Vulnerabilities,
   Zhao et al. EMNLP 2024).** *Clears all four + high-collateral because* this is the published, no-finetune
   generalization of the **banked ~25× in-context-backdoor win**: the trigger→payload route is an **induction
   edge consumed at the answer step**, the trigger-label association is **planted in context only** (no weight
   backstop), and a *distinctive multi-token* trigger phrase is a **rare conjunction** while "attend to a prior
   occurrence" and "emit the associated label" are broadly reused → a linear "force-the-target-label" steer
   flips clean inputs too = high collateral. **Public: code + clean-label poison recipe (arXiv 2401.05949),
   1.3B–180B incl. small open models; ground-truth ASR.** *Biggest risk:* condition #4 — a *single*
   distinctive trigger is the strongest non-recurrence case; a *generic* trigger sentence reused across all
   poisoned demos would recur (D3). Pre-check: verify the trigger token-span (not a generic position) is the
   key, on a held-out clean set.

3. **IN-CONTEXT DEFINITION / nonce-word binding (WinoDict-style, Eisenschlos et al. EACL 2023).** *Clears all
   four + high-collateral because* a **made-up word** is *defined in context* and then consumed at the answer
   step to resolve a Winograd coreference — the meaning has **zero parametric backstop by construction** (the
   word does not exist), the (nonce-token × its definition) conjunction is **maximally distinctive/non-
   recurrent** (a fresh nonce per instance), and both endpoints ("a rare/OOV token", "pull its in-context
   gloss") are reused → a linear "apply-the-definition" steer cannot bind *this* gloss to *this* token = high
   collateral. **Artifact: 496 examples (Google, EACL'23); trivially synthesizable as Winograd-rewrite + gloss.**
   *Biggest risk:* condition #1/#3 — coreference resolution may be **MLP-enriched upstream** (the nonce gloss
   integrated at prefill, like factual recall's pre-bake) rather than read at the answer step. Pre-check:
   answer-step LBNR cut of the (nonce → gloss) edge at the pronoun-resolution position.

The ranking reflects the magnitude law + the four-condition gate + artifact cleanliness: **#1 is the closest
real analogue of the synthetic broad×broad win** with a turnkey public artifact; **#2 is the safest fresh win**
(it directly generalizes a banked win-class, just to a published organism, with the cleanest ground-truth
metric); **#3 is the highest-novelty / cleanest-no-backstop** but carries the upstream-timing risk that sank
factual recall and sycophancy.

---

## FULL CATALOG (by candidate class)

### CLASS A — In-context variable binding / state tracking (the synthetic-win analogue) — STRONGEST

#### A1. ★ Mixing Mechanisms: in-context bound-entity retrieval (Gur-Arieh / Gur et al. 2025) — TOP-1
- **Behavior.** Templatic binding context, query one entity → emit its bound partner.
  Prompt: *"Ann loves ale, Joe loves jam, Pete loves pie, Tim loves tea. Who loves pie?"* → **"Pete"** (and the
  symmetric "what does Ann love?"→"ale"). The bound value is retrieved purely from the context binding matrix.
- **Four-condition map.**
  - #1 LOAD-BEARING EDGE: **YES (with a caveat).** Retrieval is attention-mediated; the paper decomposes it into
    a **lexical** edge (query-entity → its group-partner, a content×content edge) vs a **positional** edge (group
    index). The lexical edge is the G-score target.
  - #2 CONTEXT-SUPPLIED, NO BACKSTOP: **YES, definitionally.** Bindings are arbitrary/permuted ("Ann loves ale")
    with no parametric prior — the model *must* attend the context to answer.
  - #3 CONSUMED AT ANSWER STEP: **YES (the win-condition fit).** The "Who loves pie?" query forces the bound-value
    read at the final/answer token — this is attend-back-and-copy AT generation, exactly FRA's home turf.
  - #4 NON-RECURRENT CONJUNCTION: **YES.** Entities are **distinct within each instance** ("the binding matrix G
    consists of distinct entities"); the (Ann × ale) cell is unique to this instance and does NOT recur across the
    sibling bindings (Joe×jam, Pete×pie). Cut (Ann × ale), preserve (Joe × jam).
- **Magnitude-law profile.** reuse(endpoints) HIGH: "an entity name as query" and "retrieve a bound value" are
  the most broadly reused directions imaginable (every binding uses them). reuse(conjunction) LOW: the specific
  (this-entity × this-value) pair is per-instance. **This is the textbook HIGH×LOW profile** — the cleanest real
  instance of the synthetic broad×broad construction.
- **Why a linear steer bleeds (the FRA driver).** The only linear direction available to suppress Ann's binding
  is the broad "retrieve-the-bound-value" / "entity-name" direction — shared by ALL four bindings. A DoM steer
  built from "Ann-binding-present minus absent" prompts captures the generic binding-retrieval direction, so
  knocking out Ann's value also dents Joe/Pete/Tim's. There is no *linear* "Ann's-value-only" direction because
  the value identity lives in the *conjunction*, not either endpoint. FRA's cell-cut targets exactly the
  conjunction → predicted decisive ≥2× selectivity.
- **Artifact / runnability.** `github.com/yoavgur/mixing-mechs` (**MIT**): `grammar/schemas.py` (binding-task
  generator), `tasks/dist.py` (experiments), `CausalAbstraction/` (interventions), `example.ipynb` ("works out
  of the box"). Models in-scope: **gemma-2-2b**, gemma-2-9b, Llama-3.1-8B, Qwen2.5 family. Caveat: README says
  "codebase still being finalized" — budget a small integration step; we only need the **dataset generator**
  (we own the FRA cut). arXiv: **2510.06182**.
- **Ground-truth metric (judge-free).** Exact-match of the emitted bound value (single token: "ale"/"jam"/...).
  On-target = Ann's-value retrieval drops; selectivity = Joe/Pete/Tim's-value retrieval **survives**. Pure
  next-token logit; no judge.
- **Concrete task design.** Build N=200 instances of 4-binding contexts (distinct entity/value vocab per slot).
  **FRA cell to cut:** the (Ann-name query × ale-value key) score-delta at the bound-entity-retrieval head(s)
  the paper localizes (run their head-id sweep first), over rows q≥answer-position. **Sibling/held-out
  selectivity set:** the SAME instance re-queried for Joe/Pete/Tim (intra-instance siblings — the hardest
  control: identical surface form, identical broad endpoints, only the conjunction differs). **Best-tuned linear
  baseline:** DoM "suppress-Ann's-binding" steer, layers×α swept, at matched on-target effect.
- **RISK FLAG (which condition fails on a real LLM).** **#1/#3 via redundancy** — the paper's headline is that
  retrieval *mixes* lexical + positional + reflexive mechanisms. If the **positional** mechanism is a redundant
  backstop (model recovers Ann's value by group-index even after the lexical edge is cut), the lexical-edge cut
  is non-load-bearing → an IOI-style LBNR failure (v1's clean negative `j13`). **Cheapest pre-check:** construct
  the lexical/positional *disagreement* regime (the paper's counterfactual pairs where the two mechanisms point
  to different answers), and run the operating-regime LBNR probe there: cut the lexical edge and check the answer
  flips to the positional target (R≈1 ⇒ load-bearing lexical edge; R≈0 ⇒ positional backstop ⇒ NO-GO before any
  selectivity spend). This directly mirrors the v1 rule "sample the application's OWN operating regime."

#### A2. Filter heads: in-context list filtering (Lepori et al. 2025) — runnable but D3-RISK
- **Behavior.** "From [list], pick the item with property P" → emit the matching element. General "filter head"
  attends list items and selects by a context-supplied criterion. Models: Llama-7B, Gemma. arXiv: **2510.26784**;
  viz `filter.baulab.info`.
- **Four-condition map.** #1 YES (attention-localized filter heads). #2 YES (list is context-supplied). #3 YES
  (selection at the answer step). #4 **RISK** — the *criterion* ("the largest", "ends in -a") is a **generic
  filter role reused across instances**: this is the **factual-recall D3 trap** (relation-keyed, not instance-
  keyed). The (criterion × item) conjunction recurs whenever two instances share a criterion. **Down-ranked
  below A1** precisely for this reason: A1's per-instance distinct bindings dodge D3; filter heads do not, unless
  the eval uses a *distinctive per-instance* criterion (e.g. "the item that rhymes with <nonce>") — a possible
  rescue, but A1 is cleaner.
- **Metric.** Ground-truth exact-match of the selected element. **Use as a sibling/contrast task or a D3 stress
  test for A1, not as the primary win bet.**

#### A3. Variable binding circuitry (Davies et al. 2023; symbolic-program binding, Yang et al. 2025) — substrate
- **Behavior.** "a=7, b=3, … print(a)" → "7"; or symbolic-program register tracking. Davies et al. localize
  binding to **9 attention heads + 1 MLP** of 1640 in LLaMA-13B (arXiv 2307.03637, desiderata-based discovery);
  Yang et al. 2025 (arXiv 2505.20896) study how transformers learn it. The (variable × value) read at the answer
  step is the same shape as A1 with arithmetic dressing.
- **Four-condition map.** #1 YES (9 heads + **1 MLP** — the MLP component is a partial G-post backstop risk).
  #2 YES (values context-assigned). #3 YES (read at `print(a)`). #4 YES (distinct vars per instance).
- **Verdict.** **Same win-shape as A1 but worse artifact** (LLaMA-13B is off the gemma-2-2b budget; the "+1 MLP"
  flags a possible non-attention backstop). **A1 dominates it.** Keep as a confirmatory second substrate if A1
  wins and we want arithmetic-domain external validity.

### CLASS B — In-context backdoor with a distinctive multi-token trigger — SAFEST FRESH WIN

#### B1. ★ ICLAttack / Universal Vulnerabilities (Zhao et al. EMNLP 2024) — TOP-2
- **Behavior.** **No-finetune** in-context backdoor: poison a few demonstration examples so a **trigger phrase**
  (a fixed sentence/token-span) becomes associated with an attacker-chosen label; at inference, any input
  containing the trigger gets the target label. Clean-label (poisoned demos are *correctly* labeled → stealthy).
  ASR ~95% across OPT/GPT-family 1.3B–180B. arXiv: **2401.05949** (EMNLP 2024 main).
- **Four-condition map.**
  - #1 LOAD-BEARING EDGE: **YES.** The trigger→payload route is an **induction edge** (attend-back to the prior
    poisoned demo where the trigger co-occurred with the target label, copy the label). This is the *exact*
    mechanism of the banked in-context-backdoor win (fra_win ~25× vs ActAdd).
  - #2 CONTEXT-SUPPLIED, NO BACKSTOP: **YES.** The trigger↔label association exists ONLY in the in-context
    poisoned demos — no weights encode it (no finetune). Removing the demos removes the backdoor.
  - #3 CONSUMED AT ANSWER STEP: **YES.** The label is emitted at the classification answer token by attending
    back to the demo.
  - #4 NON-RECURRENT CONJUNCTION: **YES if the trigger is a distinctive multi-token span.** (trigger-phrase ×
    target-label) is a planted, rare conjunction; "attend a prior occurrence" + "emit a label" are broadly
    reused. **Design lever: pick a distinctive trigger** (e.g. a rare named entity span) to maximize non-
    recurrence.
- **Magnitude-law profile.** reuse(endpoints) HIGH (induction + label-emission are universal). reuse(conjunction)
  LOW (the specific trigger×label is planted and unique). **Clean HIGH×LOW.**
- **Why a linear steer bleeds.** A linear "remove-the-backdoor" steer must target either "the trigger direction"
  (bleeds onto benign inputs that contain trigger-like tokens) or "the target-label direction" (suppresses the
  label even for *legitimately* that-label inputs). Neither isolates the (trigger × label) conjunction. FRA cuts
  the planted induction cell → suppresses the backdoor while preserving label-emission for clean inputs and the
  trigger token for benign uses.
- **Artifact / runnability.** Public method + clean-label poison recipe (arXiv 2401.05949; ICLShield arXiv
  2507.01321 ships defense/eval scaffolding and re-implements the attack). **Trivially reproducible** on
  gemma-2-2b-it: build poisoned ICL prompts (SST-2 / a simple classification set) with a chosen distinctive
  trigger; no training. Ground-truth ASR.
- **Ground-truth metric.** **Exact-match ASR**: P(target label | trigger present). On-target = ASR drops after
  the cut; selectivity = clean accuracy (no trigger) and benign-input accuracy (trigger-like-but-not-the-trigger)
  **survive**. Fully judge-free.
- **Concrete task design.** Poison k=4 ICL demos with trigger T→label L*. Eval = triggered inputs (ASR) +
  clean inputs (capability) + **hard control: inputs containing a *different* distinctive phrase T′** (the
  sibling that a crude trigger-direction steer over-blocks). **FRA cell to cut:** (trigger-span key × answer-
  query) induction cell at the induction head(s). **Best-tuned linear baseline:** DoM "no-backdoor" steer +
  the trigger-token-ablation baseline, swept.
- **RISK FLAG.** **#4 recurrence** if the trigger is *generic* (a common word reused across demos) — then the
  conjunction recurs and the cut bleeds (D3). **Cheapest pre-check:** ablate ONLY the trigger-span key (not a
  generic position) and confirm ASR collapses while clean/T′ inputs are untouched — i.e. verify the key is the
  *distinctive span*, on a held-out set, in the operating regime. (This is the "complete-cut, own-regime" v1
  rule.) Secondary risk: with enough poisoned demos the association could become *positionally* encoded
  (last-demo-label) rather than trigger-keyed — control by shuffling demo order.

### CLASS C — In-context definition / translation — HIGHEST NOVELTY, UPSTREAM-TIMING RISK

#### C1. ★ WinoDict-style in-context nonce-word binding (Eisenschlos et al. EACL 2023) — TOP-3
- **Behavior.** A **made-up word** is given a one-line in-context **definition**, then a Winograd-style sentence
  uses it; the model must resolve a coreference *using the supplied definition*. E.g. *"A **florp** is a person
  who is easily frightened. The florp ran from the dog because **it** was aggressive."* → "it" = the dog (requires
  using florp's gloss). 496 examples (Google, EACL'23). arXiv: **2209.12153**.
- **Four-condition map.**
  - #1 LOAD-BEARING EDGE: **PLAUSIBLE** (the nonce→gloss read is attention-mediated) — but see risk.
  - #2 CONTEXT-SUPPLIED, NO BACKSTOP: **YES, the strongest of any candidate** — the word *does not exist*, so the
    meaning has **zero** parametric prior. Attending the in-context definition is the only route.
  - #3 CONSUMED AT ANSWER STEP: **RISK** — coreference resolution may integrate the gloss **upstream** (at
    prefill, into the nonce-token's residual) rather than at the answer/pronoun step. This is the factual-recall
    pre-bake / sycophancy-timing failure mode. **The condition most likely to fail.**
  - #4 NON-RECURRENT CONJUNCTION: **YES, maximally** — a fresh nonce per instance; (florp × "easily frightened")
    recurs in NO sibling. Distinct by construction.
- **Magnitude-law profile.** reuse(endpoints) HIGH ("a rare/OOV token as query", "pull its in-context gloss").
  reuse(conjunction) LOW (per-instance nonce↔gloss). **Clean HIGH×LOW — arguably the purest of the three** on the
  no-backstop axis.
- **Why a linear steer bleeds.** A linear "apply-the-definition" steer captures the generic "use-the-in-context-
  gloss" direction shared by every nonce; it cannot bind *this* gloss to *this* nonce. To suppress florp's gloss
  a linear steer would have to suppress the whole in-context-definition-following capability → high collateral.
- **Artifact / runnability.** Paper-only (no clear public GitHub; Google). **But trivially synthesizable**:
  Winograd/WSC schemas (public) + nonce generator + a templated gloss → an exact-match coreference eval. We own
  the data build; small effort. gemma-2-2b-it.
- **Ground-truth metric.** Exact-match / two-option logit on the coreference target (judge-free Winograd metric).
  On-target = florp-instance resolution drops after cutting (florp → gloss); selectivity = other nonce instances
  and the original (non-nonce) Winograd survive.
- **Concrete task design.** N WSC items, each rewritten with a fresh nonce + a one-line gloss that is load-bearing
  for the coreference. **FRA cell to cut:** (nonce-token query × gloss-span key) at the resolution position.
  **Sibling/held-out:** other-nonce instances (same structure, different nonce↔gloss) + the un-rewritten WSC
  (capability floor). **Best-tuned linear baseline:** DoM "ignore-this-definition" steer, swept.
- **RISK FLAG.** **#3 upstream timing** (the v1 killer). **Cheapest pre-check:** the timing-disambiguation cut —
  run the (nonce→gloss) edge-cut BOTH at prefill positions AND restricted to the answer/pronoun step; if only the
  prefill cut moves the answer (answer-step cut R≈0), the gloss is pre-baked upstream → FRA's answer-step edge is
  a redundant late read (the injection Run-2/Run-3 lesson) → likely sub-threshold vs linear → STOP before
  selectivity. If the answer-step cut alone flips it (R≈1), GO. (Mirror v1: test the COMPLETE cut, own regime.)

#### C2. In-context dialect/translation mapping ("in this dialect X means Y; use X") — synthesizable variant
- **Behavior.** Supply a novel token→meaning or word→translation mapping in context, then require its use at the
  answer step. Same shape as C1 with an even more distinctive key (an invented symbol). "Exploring the
  Translation Mechanism of LLMs" (arXiv 2502.11806) finds word-level translation is attention-routed at decode
  (a supporting prior). **Same upstream-timing risk as C1**; C1's WSC metric is cleaner (forced two-option),
  so C1 is the representative of this class. Keep C2 as a robustness variant if C1 wins.

### CLASS D — Context-grounded retrieval / RAG attribution — SAFETY-RELEVANT, D3-RISK

#### D1. RAG context attribution: cut the answer's attention to ONE poisoned passage (JSD mechanistic study,
       2025; PoisonedRAG eval set)
- **Behavior.** Multi-passage context; the answer copies a value from ONE passage. Poison one passage with a
  wrong value; cut the answer's attention to the poisoned passage's distinctive content while preserving
  grounding in the clean passages. Mechanistic study: "Attributing Response to Context: a JSD-driven mechanistic
  study of context attribution in RAG" (arXiv **2505.16415**) finds attention-head-mediated context attribution.
  Poison artifact: PoisonedRAG (`github.com/sleeperagentss/PoisonedRAG`? — verify; arXiv 2402.07867) + NQ/HotpotQA.
- **Four-condition map.** #1 YES (attention-mediated attribution). #2 YES (passage values context-supplied —
  but parametric knowledge can be a **backstop** for famous facts → use *fictional* QA to enforce #2, mirroring
  the factual-recall lesson). #3 YES (copy at the answer step). #4 **RISK** — if the answer slot is a **generic
  "the relevant passage" role** (the v1 injection/D3 risk that PLANNING flagged: "the SAME D3 generic-answer-slot
  risk"), the conjunction recurs. Distinctive-passage-*content* (a specific named value) helps; the *grounding
  role* hurts.
- **Verdict.** **Same mechanism as v1 injection** with a fiddlier multi-passage eval and the SAME generic-slot
  D3 risk — v1 PLANNING explicitly rejected RAG as "dominated by injection." **Down-ranked below A1/B1/C1.**
  Listed for completeness and because it is the highest *safety* value; only pursue if A1–C1 all fail and a
  safety-relevant target is wanted. To beat injection's 1.6× it must exploit **distinctive passage content** (a
  rare named value), not the grounding role — i.e. make reuse(conjunction) genuinely low.
- **Metric.** Ground-truth exact-match (does the poisoned value appear / does clean-passage accuracy survive).

### CLASS E — Coreference / entity-attribute binding in multi-entity narratives

#### E1. Multi-entity attribute binding (Feng & Steinhardt 2023, binding-ID) — v1 SCORED WEAK, but re-test in a
       purpose-built eval
- **Behavior.** "coffee in Box Z, stone in Box M…; what is in Box Z?" — bind attribute↔entity, retrieve at the
  answer step. arXiv 2310.17191; synthesizable.
- **Four-condition map.** #1 attention-retrieval YES but v1 found the edge **weak (attn 0.38), LBNR-R=−0.89** in
  the substrate tested → non-load-bearing there. #2 YES. #3 YES. #4 YES (distinct boxes).
- **Verdict.** **A1 (Mixing Mechanisms) is the strictly better version of this idea** — it is the *same* binding
  retrieval but with (a) a public artifact + localized heads, (b) a published causal decomposition of the
  retrieval edge, and (c) a regime (lexical vs positional) in which to run the load-bearing pre-check that v1's
  binding test lacked. **A1 supersedes E1.** If A1's lexical edge is load-bearing, E1's prior negative was a
  substrate/regime artifact, not a structural NO; if A1 also fails LBNR, the binding-edge-is-distributed verdict
  is confirmed and we stop the whole class. Either way A1 is the right experiment, not E1.

---

## SUMMARY TABLE

| # | Organism | Class | #1 edge | #2 no-backstop | #3 answer-step | #4 non-recurrent | reuse(end)×reuse(conj) | Artifact | Metric | Biggest risk |
|---|---|---|---|---|---|---|---|---|---|---|
| **A1★** | **Mixing-Mechs bound-entity retrieval** | binding | Y (lexical) | **Y** | **Y** | **Y (distinct/inst)** | **HIGH×LOW** | **mixing-mechs (MIT), gemma-2-2b** | EM bound value | #1/#3 positional backstop |
| **B1★** | **ICLAttack in-context backdoor** | backdoor | Y (induction) | Y | Y | Y (distinct trigger) | HIGH×LOW | repro, no-finetune, small models | EM ASR | #4 generic trigger → D3 |
| **C1★** | **WinoDict nonce-word binding** | definition | plausible | **Y (strongest)** | **RISK** | **Y (fresh nonce)** | HIGH×LOW | synthesizable (WSC+gloss) | WSC two-option | #3 upstream pre-bake |
| A2 | Filter heads (list select) | binding | Y | Y | Y | **D3-risk (generic criterion)** | HIGH×MED | filter.baulab.info | EM element | #4 recurrence (=factual recall) |
| A3 | Variable-binding circuit | binding | Y(+1 MLP) | Y | Y | Y | HIGH×LOW | LLaMA-13B (off-budget) | EM value | artifact off-budget; +MLP backstop |
| D1 | RAG passage attribution | RAG | Y | Y* | Y | D3-risk (generic slot) | HIGH×MED | PoisonedRAG/JSD | EM value | =injection (v1 dominated) |
| E1 | Multi-entity box binding | coref | weak (v1) | Y | Y | Y | — | synth | EM attribute | v1 LBNR-fail; **A1 supersedes** |

\* parametric backstop unless fictional QA is used.

---

## METHOD NOTES (carry-over from v1, applied to every candidate above)

- Every candidate's pre-check **samples the application's OWN operating regime** (A1: lexical/positional
  disagreement; B1: held-out trigger-span; C1: answer-step vs prefill timing; D1: fictional-QA to force #2). This
  is the v1 rule that the R=+0.86 weak-fact false-GO taught us.
- Selectivity test = FRA cell-cut **vs the BEST-TUNED linear baseline** (sweep layers×α), at **matched on-target
  effect**, on a **hard intra-instance sibling control**, ground-truth metric. WIN = **≥2× selectivity + ≥0.15
  abs + FRA retains >10% on the hard control** (the v1 bar; injection's 1.6× was the NULL).
- All metrics above are **judge-free** (exact-match / forced-choice logit). No LLM judge needed.
- The hard control for A1 (intra-instance siblings: re-query the SAME prompt for Joe/Pete/Tim) is the strongest
  possible selectivity stressor — identical surface form and identical broad endpoints, differing ONLY in the
  conjunction. If FRA wins *there*, the magnitude-law claim is demonstrated cleanly.

---

## SOURCES
- Gur-Arieh / Gur et al. 2025, **Mixing Mechanisms: How LMs Retrieve Bound Entities In-Context** — https://arxiv.org/abs/2510.06182 ; code (MIT) https://github.com/yoavgur/mixing-mechs
- Lepori et al. 2025, **LLMs Process Lists With General Filter Heads** — https://arxiv.org/abs/2510.26784 ; viz https://filter.baulab.info
- Davies et al. 2023, **Discovering Variable Binding Circuitry with Desiderata** (9 heads + 1 MLP, LLaMA-13B) — https://arxiv.org/abs/2307.03637 ; Yang et al. 2025, **How Do Transformers Learn Variable Binding in Symbolic Programs?** — https://arxiv.org/abs/2505.20896
- Zhao et al. 2024, **Universal Vulnerabilities in LLMs: Backdoor Attacks for In-Context Learning (ICLAttack)** — https://arxiv.org/abs/2401.05949 (EMNLP 2024) ; ICLShield (defense/eval) — https://arxiv.org/abs/2507.01321
- Eisenschlos et al. 2023, **WinoDict: Probing LMs for In-Context Word Acquisition** — https://arxiv.org/abs/2209.12153 (EACL 2023, 496 ex)
- **Translation mechanism** of LLMs (decode-time attention routing, supporting C2) — https://arxiv.org/abs/2502.11806
- **RAG context attribution (JSD mechanistic)** — https://arxiv.org/abs/2505.16415 ; PoisonedRAG — https://arxiv.org/abs/2402.07867
- Feng & Steinhardt 2023, **Binding ID mechanism** — https://arxiv.org/abs/2310.17191
- Gemma Scope (SAE substrate, incl. attention SAEs `google/gemma-scope-2b-pt-att` + residual `gemma-scope-2b-pt-res-canonical`) — https://arxiv.org/abs/2408.05147 ; https://huggingface.co/google/gemma-scope
- Multi-Trigger Poisoning (finetune-based, MLP-baked → G-POST, DOWN-RANKED) — https://arxiv.org/abs/2507.11112
- Cultural Binding Heads (pre-training knowledge bindings, NOT context-supplied → fails #2, DOWN-RANKED) — https://arxiv.org/abs/2605.28543
