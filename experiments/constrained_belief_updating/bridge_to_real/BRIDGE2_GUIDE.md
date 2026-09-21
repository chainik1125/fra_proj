# Bridge 2 — the FRA usage guide: which control tool should you reach for?

*The practical payoff of the whole program. Given a target behavior in a real LM, a decision procedure a
practitioner can run to pick the control tool — FRA-QK vs difference-of-means/SAE vs "nothing works
cleanly" — BEFORE spending compute on a frontier. Derived from the amended fra_win carrier law
(`BRIDGE_NOTE.md`) and its fra_win checklist (`../../fra_win/THEORY.md`). Every routing has an existing
campaign number behind it.*

*Tags as elsewhere: **[fra_win-OBSERVED]**, **[toy-CONFIRMED]**, **[PREDICTION]**.*

---

## The question

You want to *control* a behavior — remove a backdoor, suppress an association, edit an attention edge —
with minimal collateral. Three tools are on the table: **FRA-QK** (subtract a bilinear
query-feature×key-feature cell from the attention scores), **DoM/SVD** (subtract a residual direction),
**conv-SAE** (ablate the top act-diff features). The law says exactly one of them is right for a given
behavior, and **which one is fixed by the carrier.** This guide reads the carrier off in four cheap steps
and routes you.

## The decision procedure (four steps, cheapest first)

### Step 0 — Is the cause even in the attention *scores*? (CCF, the score-space gate)
Compute **CCF** = (non-sink content-pair score mass on the target edge) / (total edge mass), on the
carrying head(s) (`g1_acronym.py`/`fra.core.fra`). Sink-mask first (drop features that fire on >50% of a
neutral corpus — BOS/positional sinks).
- **CCF ≈ 0** → the behavior has *no score-space cause* (it's an output direction, a positional/BOS edge,
  or an MLP computation). **FRA-QK cannot help; go to Step 3's gate/direction branch.** Kills the
  weight-baked sleeper (payload in OV, no score cause) and BOS edges (raw attn 0.98 but CCF 0.000).
- **CCF high** → a content×content conjunction lives in the scores. Continue. *(On Gemma the sink-mask
  over-kills late layers; if CCF is ambiguous, certify behaviorally in Step 2 instead of trusting it.)*

### Step 1 — Read off the carrier (the primary diagnostic: QK-cut vs OV-cut asymmetry)
On a handful of examples, compare the behavioral effect of two surgical cuts:
- **QK-cut**: subtract the FRA-QK cell (project the key-feature out of the keys) → kills the *match*.
- **OV-cut**: subtract the FRA-OV term (project the source feature out of the values) → kills the
  *transport*.

Report the ratio **r = |QK-cut effect| / |OV-cut effect|** on the behavioral metric.
- **r ≫ 1** (QK-cut carries it, OV-cut ≈ inert) → **QK-carried = an obligate match.** Route to FRA-QK.
- **r ≪ 1** (OV-cut carries it) → **OV-carried = a content gate/aggregate.** Route to DoM/SAE.
- **both cuts ≈ null** (neither moves it) → not edge-carried at all → distributed-direction branch (Step 3c).

This is the toy's 21–36× OV-over-QK asymmetry [toy-CONFIRMED] turned into a real-model dipstick; it
predicts the frontier winner *before you run the frontier* [PREDICTION, being tested on Gemma].

### Step 2 — Efficacy gate: is the edge load-bearing and non-redundant? (LBNR)
**Carrier tells you the right *tool*; it does NOT tell you the edit will *change behavior*.** Cut the
identified edge across **all** heads that carry it — ranked by *causal edge-cut effect*, never by raw
attention (raw attention selects positional/sink heads) — and measure **R = 1 − B(cut)/B(intact)**.
- **R ≈ 1** → load-bearing, no backup circuit. The FRA-QK edit will actually work.
- **R ≈ 0 or < 0** → a backup circuit self-repairs (or the behavior is redundantly cued). **The edit is
  precise but behaviorally inert — do not ship FRA-QK here even though the carrier is QK.** This is
  Contradiction 2 (§4.2 of BRIDGE_NOTE): IOI's END→IO edge is a genuine QK match (passes Step 1) but
  **R = −0.39** — backup name-movers overcompensate, and the edit moves the IO−S logit only **0.17**
  (3.88→3.70) [fra_win-OBSERVED]. Kills IOI, docstring (R=0.08), delimiter-matching (R=0.23, grammar
  gives redundant cues), gemma knowledge-conflict (R=0.24).

### Step 3 — Route to a tool

**(a) Match → FRA-QK** (passed Steps 0–2: CCF-high, r≫1, R≈1). *One more check* — the
**conjunction-specificity** clause: at least one endpoint must legitimately recur in deployment (so a
direction steer pays collateral) while the *specific pair* must not, which requires the **discriminating
endpoint (usually the query) to be distinctive CONTENT, not a generic ROLE feature.** If both hold, the
expected win over DoM is the magnitude law **A ≈ reuse(marginal|eval)/reuse(conjunction|eval)** — ~10³–10⁴
when the conjunction is unique. If the query is a generic role (which-slot, entity-position), the
conjunction recurs across siblings and **A collapses toward the 1.9× floor** (differential cells buy some
back at a reach cost).

**(b) Gate / aggregate → DoM/SAE** (CCF≈0, or r≪1). The behavior is a content direction with no separable
score-space link; a mean-difference/SVD vector is the surgical tool and FRA-QK adds nothing. *Caveat from
§4.1 of BRIDGE_NOTE:* if the content is an OV *relation* with a reused marginal (a true gate), FRA-OV
*path-decomposition* may beat DoM (the toy §8 result, 17–259×) — but this has **not** been demonstrated on
a real model, so **default to DoM/SVD** and treat FRA-OV as a research bet, not a recommendation.

**(c) Distributed direction → nothing works cleanly** (both cuts null in Step 1; DoM removes but corrupts a
legitimate twin). The honest limit (Contradiction 3, §4.3). Flag it; don't force a QK/OV verdict.

## The routing table

| Behavior class | Carrier (Step 1) | Tool | Expected win / caveat | Evidence |
|---|---|---|---|---|
| Induction / verbatim copy, in-context backdoor | QK match | **FRA-QK** | ~11–90× lower collateral than DoM/conv-SAE | [fra_win-OBSERVED] |
| Acronym letter-movers | QK match (query = spelled-letter state) | **FRA-QK** | **A ≈ 26,000×** (unique conjunction) | [fra_win-OBSERVED, screened] |
| In-context retrieval, *content-distinctive* query | QK match | **FRA-QK** | **A ≈ 1,100×**; head-localized, causal | [fra_win-OBSERVED, green-lit] |
| Retrieval by *generic role* query (which-box, PII-by-slot) | QK but recurs | FRA-QK **degrades** | A → **1.9×** floor; entity-PII **no-op** | [fra_win-OBSERVED] |
| IOI name-mover | QK match but **redundant** | **neither** (FRA precise-but-inert) | edit moves logit **0.17**, R=−0.39 (backups) | [fra_win-OBSERVED] |
| Sentiment / topic aggregation | OV aggregate | **DoM/SAE** | FRA-QK gauge despite high pattern mass | [toy-CONFIRMED analog: fra_hmm_toy] |
| PII bank / attribute lookup | OV, reach-limited | **position-locked SAE** | FRA-OV content-selective but reach-insufficient | [fra_win-OBSERVED] |
| Style / register / "respond-in-mode" | direction (ICL task-vector) | DoM removes, but **entangled** | no clean removal (twin of legit style) | [fra_win-OBSERVED] |
| Weight-baked sleeper payload | OV/output direction, CCF≈0 | **DoM/SVD** | FRA null (no score cause) | [fra_win-OBSERVED] |
| Many-shot injection | distributed direction | **nothing clean** | DoM →0 but kills legit twin (0.29→0) | [fra_win-OBSERVED] |

## Three worked examples

**Match, and it wins — acronym letter-movers.** Behavior: "The Chief Executive Officer (CE" → "O". Step 0:
CCF-high (the (spelled-letters-query × source-word-key) conjunction is content, non-sink). Step 1: the QK-cut
(remove the letter-state × source-word cell) kills the completion; the OV-cut barely moves it → r≫1 → QK
match. Step 2: cutting across the letter-mover heads (L8H11/L9H9/L10H10/L11H4) removes the answer, no backup
→ R≈1. Step 3a: the query (the *specific spelled letters*) is distinctive content and recurs rarely as a
conjunction → **A ≈ 26,000×** [fra_win-OBSERVED]. **Reach for FRA-QK.**

**Match, but it's inert — IOI.** Behavior: "...John gave a drink to" → "Mary" (IO). Step 0: CCF-high (the
END→IO edge is a real content conjunction). Step 1: QK-cut on the END→IO cell of the name-movers works
locally → looks like a QK match. **But Step 2 fails:** cutting the three main name-movers (L9H9/L9H6/L10H0)
moves the IO−S logit only **0.17** (3.88→3.70) because **backup name-movers compensate** — R=−0.39
[fra_win-OBSERVED]. **Do not ship FRA-QK; the carrier is right but the edge is not load-bearing.** (ActAdd
brute-forces it, at collateral — this is a case where no *surgical* tool wins.)

**Gate / direction — PII-by-slot and many-shot.** PII lookup ("...SSN of the applicant is") routes through
a *generic role* query ("the queried slot"), so even though it's attention-mediated, the conjunction recurs
across every slot → Step 3a's distinctive-content clause fails, FRA-OV is reach-insufficient, and only a
**position-locked SAE** disarms it [fra_win-OBSERVED]. Many-shot injection (10 demos → P(marker) 0.00→0.57)
is worse: Step 1 both cuts null (it's a *distributed* ICL task-direction, not an edge), DoM removes it (→0)
but also kills the *legitimate* marker on a held-out prompt (0.29→0) — **representationally identical to a
benign behavior, no clean removal by anyone** [fra_win-OBSERVED]. **Route: DoM/SAE for PII (with the reach
caveat); flag many-shot as the honest limit.**

## The one-line rule
*If the behavior is an obligate content match (r≫1) with a distinctive query, and its edge is load-bearing
with no backup (R≈1), the FRA-QK bilinear edit is the only tool that cuts the link while sparing both
endpoints — expect 10³–10⁴× less collateral. Otherwise it's a direction: reach for difference-of-means/SVD,
and if the behavior is a distributed ICL twin of a legitimate one, accept that no tool removes it cleanly.*
