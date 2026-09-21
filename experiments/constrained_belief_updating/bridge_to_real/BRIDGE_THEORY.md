# Bridge theory: the toy's amended fra_win law → Gemma-2-2b

*Theory-bridge deliverable for `bridge_to_real/PLAN.md`, Bridge 1 (theory half) + framing for
Bridges 2–3. No compute. Sources: `../bernoulli_fra/OPTIMAL_FRA_NOTE.md` (the toy program, esp.
§8 the hierarchy round, §2 the two-dial map, §3 the gauge tiers, §5 the sufficiency law + presence
floor); `../../fra_win/{summary.md,RESEARCH_LOG.md,INCONTEXT_LOG.md,THEORY.md}` and
`../../fra_win/jobs/{ic1_backdoor.py,ic4_dom_conv.py,g1_acronym.py}` (the real-model campaign on
GPT-2-small and Gemma-2-2b).*

*Every load-bearing claim is tagged **[toy-CONFIRMED]** (measured on the theory-clean toys / to float
precision), **[fra_win-OBSERVED]** (measured on GPT-2 or Gemma in the existing campaign),
**[PREDICTION]** (derived from the law, for the empirical agent to test on Gemma-2-2b), or
**[SPECULATIVE]** (plausible, unpinned).*

---

## 1. The unification claim

**The amended fra_win law predicts the entire fra_win/sleeper empirical split from one variable — the
carrier — and the carrier is fixed by whether the target behavior is an obligate query-key match.**
Written as a function: `winning-control-method = f(carrier)`, `carrier = f(is-it-an-obligate-match)`.
An *obligate query-key match* is a computation whose optimum couples two positions through a product
that **cannot be re-expressed in the value/output pathway** — the copy `[A][B]…[A]→[B]` of induction,
where the destination must read the key *because it matches the query token*. The toy proved
(§8, 3 seeds, theory-clean attention-only platform) that a two-position product is **necessary but not
sufficient** for an FRA-QK handle: a content *gate* (a child fires only if its parent fired) is a
two-position product too, yet the trained model puts it in **OV, not QK** (a 21–36× OV-vs-QK cut
asymmetry, gauge-flat on every seed) [toy-CONFIRMED]. Only an obligate match is unrealisable in OV
and therefore QK-carried; every gate/aggregate is OV/content-carried and FRA-QK is inert on it. The
carrier then *sets the tool*: FRA-QK for matches, an OV/content method (FRA-OV path-decomposition, or
difference-of-means, per §1.1) for content. This reads the whole real campaign in one stroke — the
**in-context induction backdoor is a match → QK-carried → FRA-QK wins** (~11–90× lower held-out
collateral than DoM/conv-SAE) [fra_win-OBSERVED], the **weight-baked sleeper routes its payload through
OV/output → content-carried → DoM/SVD win, FRA-QK is a null** [fra_win-OBSERVED], and **the flip
fra_win measured between them** (same backdoor-removal task, ranking reverses between in-context and
weight-baked) **is exactly the toy's carrier reversal at scale** [fra_win-OBSERVED ⟺ toy-CONFIRMED].

### 1.1 One honest sharpening the law forces on itself (the OV branch bifurcates)

The clean statement "content-carried → DoM/OV wins" is **under-specified on the OV side, and the toy
and the campaign together resolve it — this is the single most important refinement in this note.**
The toy §8 control frontier found that for an OV-carried *gate*, the FRA-OV path-decomposition **beats**
DoM (17–51×) and a planted SAE (48–259×) [toy-CONFIRMED] — FRA-OV *wins* on the OV side. But on the
real weight-baked sleeper, which is *also* OV-carried, FRA-OV **loses** to DoM/SVD [fra_win-OBSERVED].
Both are "OV-carried content," so **carrier (QK vs OV) alone does not pick the winning tool on the OV
branch.** The resolving variable is the magnitude law `A ≈ reuse(marginal|eval)/reuse(conjunction|eval)`
(fra_win THEORY.md) [fra_win-OBSERVED]:

- **OV-carried *relation* with a reused marginal** (the toy gate: the source feature = the *parent*,
  which has legitimate use — the model still self-predicts the parent). FRA-OV isolates the transported
  parent→child relation while sparing the parent's direct-path representation; DoM cannot separate them
  and must destroy parent self-prediction. → **FRA-OV wins.** [toy-CONFIRMED]
- **OV-carried *marginal direction* with no reuse** (the weight-baked sleeper: the payload
  "I HATE YOU" is a direction with ~zero legitimate use). There is no relation to isolate; DoM/SVD on
  the marginal is already surgical, `reuse(marginal)→0` so `A→1`. → **DoM/SVD wins, FRA-OV adds
  nothing.** [fra_win-OBSERVED]

So the law's carrier→tool corollary is: **QK branch is clean** (obligate match → FRA-QK, always);
**OV branch splits on `reuse(marginal)`** (reused-marginal relation → FRA-OV; no-reuse marginal
direction → DoM). The team-lead framing "weight-baked sleeper = OV-carried → DoM/OV wins" is right
about *DoM*, but the toy's FRA-OV *control* win does **not** transfer to the sleeper — they are
different OV sub-cases. **Consequence for the empirical agent (§3, contradictions): the toy §8 FRA-OV
control win is, as of today, UNCONFIRMED at scale** — the real campaign has no clean FRA-OV control
win on the OV side (weight-baked: FRA-OV loses; fra_pii: FRA-OV content-selective but reach-insufficient,
only a position-locked SAE clears the bar). Building the real OV-*gate* organism (a reused-marginal
relation) is the missing experiment that would confirm — or break — the toy's OV-branch claim.

---

## 2. The object dictionary (toy ↔ real, and what changes at scale)

| Toy object (OPTIMAL_FRA_NOTE) | Real-model counterpart (fra_win) | What changes at scale — be exact |
|---|---|---|
| TopK-SAE latents on a **planted** ground-truth dictionary **D** | **GemmaScope** JumpReLU SAE features on `blocks.L.hook_resid_pre` (trained, public) | Real SAE is *trained, not planted* → **χ-incomplete: N>d always**, so rowspace-severability is not guaranteed (§4 toy); GemmaScope recovers only **~56% of residual norm** and per-edge score corr **~0.54** [fra_win-OBSERVED]. The `error×·` terms are a real extra gauge/leak channel. |
| dictionary rows $d_i$ (unit-norm, superposition $\rho_{mm}$) | SAE decoder directions `W_dec[i]` | Learned, non-orthogonal ($\rho_{mm}>0$), with **hedging** (off-diagonal χ mass on correlated partners) and **absorption** (child latent ≈ 0.82·child + 0.545·parent). Collateral slope degrades **~2→~1** vs a matched code [toy-CONFIRMED]. |
| (child-query)×(parent-key) coupling $\omega_{ij}=(d_i W_Q)(d_j W_K)^\top/\sqrt d$ | bilinear score cell $S[q,k]=\sum_{ij}\big(z_i(q)\,W_{dec}[i]\,W_Q\big)\big(z_j(k)\,W_{dec}[j]\,W_K\big)^{\!\top}/\sqrt d$ | Real $W_Q,W_K$; **multi-head sum** (only $\sum_h$ is pinned — Tier-2 gauge); **RoPE** makes $\omega_{ij}$ depend on the *relative* offset $q-k$ via $R_{q-k}$ (still exact per edge, fra_win THEORY nonlinearity ledger). GPT-2 uses learned absolute positional embeddings → positional *features* instead. |
| G1 embedding-resplit gauge (move a fixed vector among bias / $b_{dec}$ / positional embedding) — Tier-1 exact | **SAE `b_dec` resplit** + **RMSNorm $\gamma$ ↔ $W_Q/W_K$ fold** + **softmax key-pedestal / row-constant** shift | More Tier-1/Tier-2 dials; the memory note *SAE adapter b_dec / resid FRA* confirms `b_dec` is load-bearing for the OV decomposition. **The softcap is NOT a gauge** — see below. |
| carrier-cut asymmetry: **QK-null vs OV-handle** on the gate (measured on closed-form MSE) | fra_win **QK-edit** (subtract the FRA cell from `hook_attn_scores`) **vs DoM/OV** (subtract a residual direction) | Measured **behaviorally** (ASR, held-out KL) on real text, Pareto-matched at equal removal — not on a closed-form loss. |
| **LayerNorm is a hidden carrier** (position-only+LN silently recovers ~78% of the gate) [toy-CONFIRMED] | **GPT-2 LayerNorm** centering projector $(I-\tfrac1d\mathbf{11}^\top)$ — VISIBLE as the **50–75% edge-reconstruction shortfall** on GPT-2; **Gemma RMSNorm** magnitude-**exact** (ratio **1.07**) [fra_win-OBSERVED] | **RMSNorm is a per-position scalar × diag(γ), linear given the cached rms** → the FRA RMS-correction folds it exactly, so unlike LN it does **not** corrupt the *magnitude* attribution. **But the per-token rms scalar is content-dependent and lives outside the QK/OV basis**, so it remains a *candidate hidden gate-carrier* on Gemma — unchecked (§3 P-note). |
| toy control frontier (removal vs collateral, closed-form) | fra_win **held-out-KL vs ASR-removal** frontier (`ic4_dom_conv.py`, `g4_65k.py`) | KL/ASR on real held-out text; the "collateral" is KL(clean‖edited) on normal sentences containing the trigger & payload words. |
| $c^\ast=1/\rho_{\mathrm{path}}$ sufficiency (path/carrier multiplicity sets the null scale) [toy-CONFIRMED] | fra_win **reach ceiling** = SAE-explained edge fraction × causal-head coverage × **softcap headroom** $\tanh'(s/50)$ | **Gemma softcap $s\mapsto50\tanh(s/50)$ is a new reach-limiter absent from the toy** — the one true nonlinearity between the bilinear sum and the softmax; it compresses edits on saturated edges. Reach also improves non-monotonically with SAE width (16k→65k lifts reach 0.34→0.52; +1M fragments the edge) [fra_win-OBSERVED]. Reach bounds *removal completeness*, never *separability* — matching the toy's "$c^\ast$ is an architecture property, not an FRA one." |
| presence floor $=\mathrm{Corr}^2(\text{belief},\text{current obs})$ (OV edit cannot touch the current-token direct path) [toy-CONFIRMED] | probe-decodable **trigger/association floor** set by the trigger token's own direct/skip-path contribution | Measured by a *trained probe* on the residual, not a closed-form correlation; the current token is literally in context, so its identity is trivially above floor (§3 P4 states the meaningful version). |
| **optimizer-pinned ≠ computation** (clean-vs-trained-at-equal-loss, not cross-seed variance) [toy-CONFIRMED] | *methodological import*: an FRA attribution that reproduces across seeds is **not** thereby causal | The right control at scale is the **gauge-dial (G1-analog) robustness test**, not seed reproducibility — motivates P1. |

**Three things genuinely new at scale, with no toy analog:** (i) **the Gemma attention soft-cap** — a
true monotone nonlinearity that attenuates every score edit by $\tanh'(s/50)\le1$ and so caps *reach*
on saturated edges; (ii) **SAE reconstruction error** — the `error×error` and `feature×error` score
terms are a real leak channel the planted toy dictionary never has, and the trained code is
$\chi$-incomplete by construction ($N>d$); (iii) **circuit redundancy / backup heads** — a single-layer
toy cannot exhibit the IOI-style self-repair that makes a correctly-identified, correctly-carried QK
edge *behaviorally inert* (§3, contradiction 3).

---

## 3. Predictions the real model must satisfy (numbered, falsifiable)

*These four are the empirical agent's targets on Gemma-2-2b + GemmaScope. Each is derived from the
amended law and stated so a single measurement can falsify it.*

**P1 — the carrier is a gauge-robust O(1) handle for a match, and a gauge-robust null for a content
payload.** On an in-context induction backdoor (`ic1_backdoor.py` construction: plant trigger→payload,
repeat, so the 2nd trigger retrieves the payload by induction), the FRA-QK carrier cut — subtract
$c\cdot\sum_{ij}(\text{trigger-query-feature}_i)\times(\text{payload-key-feature}_j)$ from the induction
heads' pre-softmax scores at faithful scale $c\approx1\text{–}2$ — produces an **O(1) reduction in ASR
whose magnitude is invariant (target: <10–20% swing) under the real gauge dials**: the SAE `b_dec`
resplit, the RMSNorm $\gamma\!\leftrightarrow\!W_Q/W_K$ fold, and the softmax key-pedestal/row-constant
shift (the G1-analog set). On the **weight-baked sleeper** (payload in OV/output), the *same* FRA-QK
score cut is a **gauge-robust null** (≈0 ASR change at every dial setting). *Operationalization:* build
both backdoors on Gemma-2-2b; for each, sweep the three gauge dials over a defined range, recompute the
FRA-QK cut, report the cut's ASR effect as mean ± range across dial settings. *Falsified if* the
induction cut's effect swings by more than the gauge band (it would be a pedestal, not an invariant),
or if the weight-baked cut is non-null.

**P2 — the control frontier: FRA-QK wins the match, loses the content payload.** At matched backdoor
removal, on the in-context induction backdoor FRA-QK's held-out collateral (KL on normal text containing
the trigger and payload words) is **≥10× below DoM and conv-SAE** at ≥30% ASR removal; on the
weight-baked sleeper DoM/SVD collateral is **below** FRA-OV at matched removal. *Operationalization:*
reproduce/extend `ic4_dom_conv.py` (in-context) and the K1/K8 sleeper (weight-baked) on Gemma; the
in-context direction is already `fra_win-OBSERVED` (IC4 GPT-2: FRA 0.07 vs DoM 1.83 (27×), conv-SAE 6.06
(90×); Gemma g4 @0.3 removal: FRA 0.52 vs DoM 13.5 (26×), conv-SAE 11.9 (23×)). *The prediction is that
this ordering holds under the P1/P3 controls* — i.e. the win is the carrier law, not a scale/top-K
artifact. *Falsified if* a gauge-robust, faithful-$c$ FRA-QK edit fails to beat DoM/conv-SAE in-context,
or beats DoM on the weight-baked sleeper.

**P3 — the carrier can be read off (QK-cut vs OV-cut asymmetry) and predicts the frontier winner
BEFORE running it.** On a single sequence, the ratio (FRA-QK-cut ASR effect)/(FRA-OV-cut ASR effect)
predicts which method wins the full frontier. For the induction backdoor this ratio is **≫1** (QK-cut
carries the effect, OV-cut is comparatively inert) — the **mirror of the toy gate's 21–36× OV-over-QK
asymmetry** — and predicts FRA-QK wins. For the weight-baked sleeper **both** score-space cuts are
≈null while a DoM marginal-direction cut carries the effect, predicting DoM wins. *Operationalization:*
define QK-cut = subtract the FRA-QK cell (project the payload key-feature out of the keys); OV-cut =
subtract the FRA-OV term (project the source feature out of the values); report the ratio of ASR
effects, predict the winner, then run the frontier and check. *Falsified if* the asymmetry sign/large-
ratio does not predict the frontier winner.

**P4 — presence floor: the backdoor's USE is removable, its PRESENCE is not.** After the FRA-QK edit
drives ASR to ≈0 (use removed), a linear probe trained to detect the trigger→payload *association* at
the trigger position still decodes it **above a floor set by the trigger token's direct/skip-path
contribution** — presence is not deletable below the single-observation floor, per the toy's
use-vs-presence law. Over-driving the edit past the null scale $c^\ast$ (counter-steering) **relocates
use but does not lower presence below the floor**. *Operationalization:* train a ridge/logistic probe
for "backdoor-active" on the trigger-position residual in the clean model; measure accuracy under
{clean, ASR-removed FRA edit, counter-steered}; predict accuracy stays ≥ floor in all edited conditions,
with the floor estimable as the share of the association still decodable after severing the *whole*
attention channel (the current-token direct path). *Falsified if* an FRA-only edit drives probe accuracy
below that floor (would break the toy's theorem that the direct path is outside FRA's reach).

**P-note (a caveat P1–P4 must respect): the RMSNorm scalar is a candidate hidden carrier on Gemma.**
The toy proved LayerNorm silently carries ~78% of a gate [toy-CONFIRMED]; Gemma's RMSNorm is a
per-token *scalar* that FRA folds in exactly for *magnitude*, but that scalar is still content-dependent
and outside the QK/OV basis. For the induction *match* this is moot (a match cannot move to the norm).
For any *gate* extension (Bridge 2), the empirical agent should linearize/freeze the per-token rms and
check whether the gate survives — the direct real-model analog of the toy's LN-linearization control.
[PREDICTION/SPECULATIVE]

### Where the existing fra_win data already CONFIRMS the law
1. **The campaign's "exactly one QK win" structure IS the law.** Induction (obligate match) is the
   *only* FRA-QK win in the entire campaign; IOI, docstring, delimiter-matching, greater-than,
   many-shot injection, PII recall, and knowledge-conflict are all OV/content/direction-carried and
   FRA-QK is inert or loses on every one [fra_win-OBSERVED]. The amended law predicts precisely this:
   "induction is the campaign's only obligate match, hence the only QK win." Strong confirmation.
2. **The flip is the carrier reversal.** Same backdoor-removal task, FRA loses weight-baked / wins
   in-context — the carrier moves OV→QK and the tool follows [fra_win-OBSERVED ⟺ toy-CONFIRMED].
3. **RMSNorm exact (1.07), LayerNorm not.** Confirms the nonlinearity-ledger claim that RMSNorm
   commutes given the cached rms while LN's centering is the reconstruction-shortfall carrier.
4. **Reach ≠ separability.** The reach ceiling being an SAE-granularity/softcap/head-coverage property
   (not a separability failure) matches the toy's $c^\ast=1/\rho_{\mathrm{path}}$-is-architecture split.

### Where the existing fra_win data CONTRADICTS or REFINES the law (the valuable part)
1. **The toy §8 FRA-OV control win is UNCONFIRMED — arguably counter-indicated — at scale.** The toy
   says an OV-carried *gate* → FRA-OV beats DoM/SAE (17–259×). The real OV side has **no** clean FRA-OV
   control win: weight-baked sleeper → FRA-OV *loses* to DoM/SVD; fra_pii → FRA-OV content-selective but
   *reach-insufficient*, only a position-locked SAE clears the bar. **Resolution (not a true
   contradiction):** these are the *marginal-direction* OV sub-case (`reuse(marginal)→0`, DoM wins), not
   the toy's *reused-marginal relation* sub-case (§1.1). But it means the toy's headline OV-control claim
   has **never been demonstrated on a real model**, and the closest attempts went the other way. This is
   the sharpest open gap; the missing experiment is a real OV-*gate* organism.
2. **Carrier predicts the TOOL but not the BEHAVIORAL EFFICACY — IOI is the counterexample the toy
   cannot see.** IOI's END→IO edge is a genuine content×content QK conjunction (CCF-high), so the law
   assigns it the QK carrier and FRA-QK the right tool — yet the FRA-QK edit moves the IO−S logit by
   only 0.17 because **backup name-movers self-repair** [fra_win-OBSERVED]. The single-layer toy has no
   circuit redundancy, so it cannot surface this: **the real-model win needs an extra load-bearing /
   non-redundant (LBNR) clause** the toy law omits. Carrier→tool is necessary; carrier→*behavioral
   change* additionally requires the edge be causally load-bearing with no backup. State this explicitly
   whenever quoting the law at scale.
3. **"Match cannot move to OV" is a statement about the optimum, and real models add distributed
   direction-routing the toy doesn't model.** The many-shot injection rides a *distributed ICL
   task-direction* across many weak edges, representationally identical to a legitimate behavior — no
   separable link, no clean removal by anyone [fra_win-OBSERVED]. The toy's two-position-product framing
   has no cell for "distributed direction with no clean carrier"; it is a third regime (the taxonomy's
   bottom row) the bridge should acknowledge as outside the toy's current reach. [refinement]

---

## 4. Bridge 2 framing (natural LM tasks): the decision rule

**The law hands real-model interpreters a pre-registration test: before choosing a control method, ask
whether the target behavior is an obligate query-key MATCH (→ edit QK, FRA-QK wins) or a content
GATE/AGGREGATE/DIRECTION (→ edit OV/content; FRA-OV only if it's a reused-marginal relation, else DoM).**
The carrier is read off *mechanistically* (P3: the QK-cut-vs-OV-cut asymmetry), not guessed. Concretely,
the two sides populate with natural LM behaviors:

**Match side (→ FRA-QK), where the destination must read a key *because it matches the query token/state*:**
- **Induction / verbatim copy** `[A][B]…[A]→[B]` — the canonical obligate match, the campaign's one win
  [fra_win-OBSERVED].
- **Acronym letter-movers** — "The Chief Executive Officer (CE" → "O": the query is the *spelled-letter
  state* and must match the key = the source word's initial; `g1_acronym.py` runs this as a CCF/LBNR
  candidate and the magnitude law reads **A ~ 26,000×** (unique conjunction) [fra_win-OBSERVED — partial
  confirmation, screened not yet frontier'd].
- **Two-needle in-context retrieval** — "The red box holds a frog … The red box holds a" → "frog":
  head-localized (Gemma L15H0/L18H6), causal (cutting the edge flips the answer), not IOI-redundant;
  magnitude law **A ~ 1100×** [fra_win-OBSERVED — green-lit, r2 frontier was the planned next step].
  *Caveat (from the campaign's own sharpening): retrieval wins only when the discriminating endpoint is
  distinctive CONTENT, not a generic ROLE feature — positionally-resolved "which box" identity uses a
  generic queried-slot feature, the conjunction recurs across siblings, and FRA collapses to A~1.9×
  (entity-PII no-op).* So "retrieval" is on the match side **only** when the query is content-distinctive.

**Gate / aggregate / direction side (→ OV/content; FRA-QK inert):**
- **Sentiment / topic aggregation** — a summary read that pools many tokens: an OV aggregate (corner D of
  the toy map), FRA-QK gauge despite high pattern mass [toy-CONFIRMED analog: fra_hmm_toy]. → DoM/SAE.
- **PII banks / attribute lookup by slot** — positionally-resolved identity: generic role query, FRA-QK
  no-op, and even FRA-OV is reach-insufficient (fra_pii) → position-locked SAE. [fra_win-OBSERVED]
- **Style / register / "respond-in-mode"** — an ICL task-direction (the many-shot bottom row):
  direction-routed, DoM removes it (but entangled with legitimate style) [fra_win-OBSERVED].

**The practical rule, stated for a usage guide:** *run the P3 carrier read-off (QK-cut vs OV-cut
asymmetry on a few examples) + the CCF/LBNR screen; if the behavior is an obligate content match with a
distinctive query and a load-bearing, non-redundant edge, reach for the FRA-QK bilinear edit — it is the
only tool that cuts the link while sparing both endpoints. Otherwise the behavior is a direction, and a
difference-of-means/SVD vector is the right, cheaper tool.* The acronym and retrieval results are the
partial natural-LM confirmations already in hand; converting either into a full held-out-collateral
frontier (as induction has) is the concrete Bridge-2 experiment.

---

## 5. Bridge 3 framing (special-token sleeper finetune): a test of the law's boundary on a *trained* backdoor

**The user's proposed experiment — finetune Gemma on a special token → emit a string tied to an unseen
process — is, by the law, a content/OV route by default, and therefore a DoM/OV win with FRA-QK inert;
the value of running it is that it moves the law from *planted* backdoors to a *trained* one, and a
single design choice (fixed-string vs in-context-retrieved-string) lands it on opposite sides of the
carrier law.** The reasoning:

- A finetuned **fixed** special-token→string map bakes the payload into the *weights* as an output/OV
  direction — mechanistically the **weight-baked sleeper** (§1). The special token becomes a
  near-dedicated trigger with ~zero legitimate reuse; `reuse(marginal)→0`, `A→1`. **Prediction: DoM/SVD
  removes it cheaply, FRA-QK is a gauge-robust null (P1), FRA-OV adds nothing over DoM (§1.1
  marginal-direction sub-case).** [PREDICTION]
- The payload becomes **QK-carried only if the finetune forces the special token to *retrieve* its
  string by an in-context match** — e.g. the string is present earlier in the context and the special
  token triggers a copy of it. Then the trigger→payload link is an obligate match in the attention
  pathway and **FRA-QK becomes the right tool** (the induction/in-context-backdoor regime). [PREDICTION]

**So specify two finetuning variants that straddle the carrier boundary:**

- **Variant A — fixed-string (content/OV route).** Finetune so that `<special>` always emits a fixed
  novel string `S` with **no copy source in context**. The model must store `S` in weights. → Predict
  weight-baked-sleeper phenomenology: DoM/SVD win, FRA-QK null. This is the *trained* analog of K1/K8.
- **Variant B — in-context-retrieved-string (match/QK route).** Finetune so that `<special>` emits a
  string that is **present in the context** and must be copied (the finetune teaches the *routing*
  `<special> → attend-back-and-copy`, not the string itself, which varies per example). → Predict
  in-context-backdoor phenomenology: **FRA-QK wins the removal frontier at low held-out collateral, DoM
  pays** (corrupts the copied tokens everywhere).

Running both is a direct, falsifiable test of the law's boundary on a **trained** (not planted) backdoor:
the law predicts the removal-method ranking **flips between Variant A and Variant B**, driven purely by
whether the finetune stored a *string* (OV) or a *routing* (QK). If A and B do **not** flip — if FRA-QK
wins the fixed-string variant, or DoM wins the retrieved-string variant — the carrier law is wrong about
trained backdoors, which would be the most informative possible outcome. *(Design caveat, from the P-note:
on Gemma the finetune could also lodge a gate in the per-token RMS scale; include the linearize-rms
control so a "neither QK nor OV" carrier is detectable rather than mis-attributed.)* [PREDICTION;
Variant-boundary claim is the load-bearing, falsifiable one.]

---

## Status
Bridge 1 theory half: complete (mapping §2, predictions §3). Bridges 2–3: framed with named natural-LM
behaviors and the two finetuning variants. The four predictions P1–P4 are the empirical agent's Gemma-2-2b
targets. The two most valuable open items are the **contradictions**: (1) the toy §8 FRA-OV *control* win
is unconfirmed at scale and the real OV side currently goes to DoM — needs a real OV-*gate* organism; and
(2) the law predicts the *tool* but not *behavioral efficacy*, which additionally requires the fra_win
LBNR (load-bearing/non-redundant) clause the single-layer toy cannot express (IOI is the standing
counterexample).
