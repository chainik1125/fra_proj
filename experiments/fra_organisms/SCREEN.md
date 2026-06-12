# SCREEN.md — FRA-cuttability scoring of every organism in ORGANISMS.md

*APPLICATION/SCREEN agent, fra_organisms campaign (CAMPAIGN.md), 2026-06-11. Scores each organism in
ORGANISMS.md on the SCREENING_RUBRIC (§2, D1–D8, four-gate rule, TIER thresholds), ranks the gate-passers,
and turns the top candidates into concrete RunPod-runnable experiment proposals with the §4 selectivity
win-test wired in. Scored a-priori from each organism's known mechanism + artifact; where my score disagrees
with the rubric's §3 pre-registered prediction I flag it explicitly. I do not run GPU or touch git.*

**Inputs consumed:** ORGANISMS.md (catalog, ~22 organisms / 16 families), SCREENING_RUBRIC.md (the instrument),
and the prior fra_win evidence base (`../fra_win/RESULTS_SUMMARY.md`, `THEORY.md`, `CAMPAIGN_REPORT.md`,
`INCONTEXT_LOG.md`, and the cheap fact-recall gate probe `../fra_win/out/bhhu4qdtg.output`).

---

## 0. The two facts that drive the whole screen

1. **Five wins are already MEASURED** (fra_win): induction 15×, in-context backdoor 25×, copy-suppression
   **516×** (natural text, transfers to a new context; the older s3 run reported 22.7× vs head-ablation),
   retrieval 1100×, acronym 26,000×. These are *banked*, not open. For these, a re-run is only worth doing if
   it adds genuinely new content (a fresh decomposition, a real published organism, a class-level generalization).
2. **The FLAGSHIP (factual-recall / ROME-class) has NOT had a selectivity test run.** All that exists is a
   *cheap LBNR gate probe* on gemma-2-2b (`bhhu4qdtg.output`): `fact-recall CCF=0.000 attn=0.81
   P(ans)=0.039→0.005 R=+0.86`. Read carefully this is **encouraging, not disqualifying**: attn=0.81 (strong
   edge), R=+0.86 (cutting the extraction edge suppresses the fact by 86% — passes the R>0.5 LBNR gate
   comfortably, so the edge is load-bearing). The one warning sign is CCF=0.000 — the **SAE-feature** edge
   didn't reconstruct the score, which is a *reach/SAE-granularity* concern (the gemma-scope-att suite at the
   tested layer), NOT evidence the mechanism is G-post. **The flagship is genuinely open and the probe leans
   in its favor.** This is the single most decision-relevant fact in this screen.

---

## 1. SCORING TABLE — every organism, D1..D8, four-gate, S, TIER

Each cell: `score | one-clause justification`. Gates = {D1,D2,D3,D4}; a 0 on ANY gate ⇒ NO-GO regardless of S
(rubric §2). D3 gates on its **0** anchor only (a D3=1 borderline is the binding/RAG crux, not a gate fail).

Scores are written a-priori from mechanism+artifact. The "Δ vs §3" column flags where I disagree with the
rubric's pre-registered family prediction.

### 1A. G-SCORE candidates (the live field)

| Organism | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | Gates | S | TIER | Δ vs §3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Factual-recall edit** (CounterFact/zsRE, gemma-2-2b + GemmaScope; ROME/MEMIT baseline) | 2 relation (subj×rel→obj) | **1** subj→obj transport IS attention (attn .81) but extraction partly MLP — the crux (probe CCF=0 is reach, R=.86 is load-bearing) | 2 subj feat × rel feat both broad, content discriminator | 2 one fact = one idiosyncratic association (probe R=.86) | 2 A≈10³ unique conjunction | 2 held-out fact accuracy, no judge | 1 name-mover localizable, extraction-head distribution uncertain | 2 ROME/MEMIT strongest surgical baseline in interp | **PASS** | **14** | **TIER-1** | §3.1 said D2=2,S≈15; I demote D2→1 (probe CCF=0 + 31% MLP-extraction lit) ⇒ S=14, still T1 |
| **Copy-suppression L10H7** (gpt2 + res-jb; head-ablation baseline) | 2 query=about-to-predict-X × key=that token, relation | 2 the suppression IS the attention to the copied token | 2 both broad, distinctive-content query (the predicted token) | 2 single canonical head, non-redundant (LBNR +0.95) | 2 A=516× measured | 2 target-token logit vs edge-cut oracle, no judge | 2 single canonical head L10H7 | 2 head-ablation baseline, runnable now | **PASS** | **16** | **TIER-1** (already-won; see §2 note) | matches §3 reference profile; BUT already measured — novelty caveat below |
| **In-context backdoor** (gpt2/gemma synth; DoM/ActAdd baseline) | 2 trigger→payload induction relation | 2 copy decision IS the induction pattern | 2 broad trigger class, content query=trigger | 2 planted single association | 2 A=25× measured (class-union open) | 2 exact-match payload string | 2 induction-head localized | 1 DoM/ActAdd baseline (no strong weight-edit) | **PASS** | **15** | **TIER-1** (already-won; class-level open) | matches §3.7-incontext; novelty = real published poisoning organism + class-union (H2) |
| **Induction (anchor)** (gpt2 + res-jb / gemma+att-SAE) | 2 [A][B]…[A]→[B] relation | 2 copy = the attention pattern | 2 query=token X distinctive content | 2 single edge, LBNR≈0.98 | 2 A≈15× measured | 2 next-token logit, ground-truth | 2 induction heads localized | 1 DoM/ActAdd baseline only | **PASS** | **15** | **TIER-1 reference** (already-won) | matches §3.2 anchor; pure re-run, lowest novelty |
| **Prompt injection / instr-hierarchy** (Open-Prompt-Injection + gemma/Qwen-chat) | 2 IS attention-redirection (Attention-Tracker) | 2 follow injected vs system = which span attended | 2 injected-imperative content × comply, both broad | **1** redundancy risk: many cues mark a span injected (the refusal/delimiter class) | 1 semi-unique conjunction | 2 exact-match injected-target string, ground-truth | 1 plausibly few-head, unverified | 1 baseline = prompt-hardening / ignore-injection steer | **PASS** | **12** | **TIER-2** (top of T2; T1 if D4/D7 resolve) | matches §3.4 (S≈12); D4=1 is the live risk |
| **RAG / context-poisoning** (PoisonedRAG + gemma-RAG) | 2 attend-to-passage relation | 2 retrieval IS attention to passage | **1** answer-query may be generic "answer slot" (conjunction-recurrence risk) | 1 many passage tokens, redundancy risk | 1 semi-unique | 2 does poisoned target appear / clean-context survives, ground-truth | 2 retrieval heads localized (Wu 2024) | 1 baseline = context-filtering / passage-ablation | **PASS** | **12** | **TIER-2** | matches §3.9 (S≈12); D3=1 slot-vs-content is the crux, same as binding |
| **Sycophancy attend-to-doubt edge** (CAA/TruthfulQA forced-choice) | **1** relational story (attend-to-doubt) atop a direction payload | **1** the crux: is the flip MADE at the doubt-attention QK step (2601.16644) or post-hoc OV (prior in-house: 0% load-bearing) | 1 doubt-cue × answer, content borderline | **0** prior measured: deference NON-load-bearing, NON-selective (6 heads each un-flip 1 item) | 1 | 2 forced-choice MCQ flip, ground-truth | 0 prior: maximally diffuse, 6×1 | 1 CAA/DoM baseline | **D4=0 → FAIL** | 7 | **NO-GO** (pre-reg negative; cheap re-probe only) | §3.8 says D2=0 gate-fail; I locate the gate fail at **D4=0** (measured non-load-bearing) — same NO-GO verdict, different deciding clause. The 2601.16644 paper does NOT overturn the prior because its steering is still OV-side linear |
| **Entity-attribute binding / coreference** (synth boxes, Pythia/Llama/gemma) | 2 entity×attribute relation | 2 retrieval is attention (entity-q × attr-k) | **1** discriminator may be generic role/slot (the crux) | 1 prior: edge weak/distributed (attn .38, LBNR-R=−0.89) | 1 | 1 ground-truth bindable but eval fiddly | **0** prior fra_win: distributed/weak edge, A=5133× artifact (not load-bearing) | 1 | **D7=0 not a gate; D4=1,D3=1 PASS** | 9 | **NO-GO by S≤9** (gate-pass but S=9) | §3.3 said S≈9 borderline; I land S=9 ⇒ NO-GO by the S≤9 rule. Could re-enter with a purpose-built content-discriminator eval, but prior substrate says distributed |
| **Function / task vectors** (Todd code; GPT-J/Llama) | **1** head-vector is additive (OV-side) not a cut-able relation | **1** FV transported as direction; the demo-selection QK is a sub-edge not the headline | 1 | 0 task derivable from many demos, distributed | 1 | 2 zero-shot task accuracy | 0 known-distributed (task vector across network) | 1 | **D4=0 → FAIL** | 7 | **NO-GO** | leans G-post per ORGANISMS; D7=0 distributed is decisive. Distinguish from single-edge induction (which wins) |

### 1B. Structural negatives (G-score edge but a gate fails — the rubric's controls)

| Organism | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | Gates | S | TIER | Δ vs §3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **IOI name-mover** (gpt2 synth template) | 2 name-movement relation | 2 G-score edge | 2 content query | **0** backup name-movers restore (LBNR-R=−0.39, measured) | 1 | 2 IO−S logit diff | 2 name-movers localized | 2 canonical baseline | **D4=0 → FAIL** | 11 | **NO-GO** (redundancy control) | matches §3 / ORGANISMS: CCF-pass ≠ win, redundancy kills it. The textbook D4=0 |
| **Greater-than / successor / docstring** (gpt2-greater-than) | 1 attend-to-year is relational | **0** the `>` is MLP-computed (edge-cut dents only .98→.87, measured) | 2 content (the year) | 1 | 1 | 2 prob-mass on years>XX | 1 | 1 | **D2=0 → FAIL** | 9 | **NO-GO** (downstream-MLP control) | matches §3.6. The canonical D2=0 |
| **Weight-baked sleeper** (repo substrate / community repros) | 1 trigger→payload but… | 1 | **0** rare DEDICATED trigger token, direction already surgical (A≈1) | 2 planted association | 0 A≈1 | 2 deterministic payload string-match | 1 | 1 DoM/SVD baseline | **D3=0 → FAIL** | 8 | **NO-GO** (rare-token control; em_svd already handles) | matches §3.7-weightbaked. Clean D3=0 |

### 1C. Direction / G-post controls (pre-registered negatives, validate the rubric)

| Organism | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | Gates | S | TIER | Δ vs §3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Emergent misalignment** (ModelOrganismsForEM, Qwen/Llama/Gemma) | **0** MLP-direction "misaligned-persona" payload (em_svd rank-1) | 0 | 1 | 1 | 0 | 0 LLM-judge, coherence-coupled | 0 | 2 strong organism | **D1=0 → FAIL** | 4 | **NO-GO** (the defining negative) | matches §3.8 / §5. Already lost this session-class |
| **Refusal direction** (andyrdt/refusal_direction) | **0** single linear direction (Arditi) — FRA's provably-worst case | 1 harm-detection MAY select content | 1 | **0** many redundant cues a request is harmful | 0 | 2 refusal-string classifier, ground-truth | 1 | 2 Arditi direction is a strong baseline | **D1=0,D4=0 → FAIL** | 7 | **NO-GO** (DROP per BRAINSTORM) | matches §3.8. Refusal-TRIGGER is at most a TIER-3 cheap dissociation, not a candidate |
| **Jailbreak circuit** (JailbreakLens, AdvBench) | 0 direction-drift / accumulative vector | 1 | 1 | 0 distributed, many-shot = vector-add | 0 | 2 ASR refusal-substring | 1 named heads but effect is drift | 1 | **D1=0,D4=0 → FAIL** | 6 | **NO-GO** | matches ORGANISMS G-POST; fra_win already logged many-shot-JB no-win |
| **Persona vectors** (persona-vectors / SAE features) | **0** linear direction ("evil","toxic") | 0 | 1 | 1 | 0 | 0 LLM-judge trait | 0 distributed | 1 | **D1=0 → FAIL** | 3 | **NO-GO** | matches §3 / Family 12 |
| **Toxicity / bias steering** (RealToxicityPrompts) | **0** MLP value-vector direction | 0 | 1 | 1 | 0 | 1 Perspective/Detoxify classifier | 0 | 1 | **D1=0 → FAIL** | 4 | **NO-GO** | matches Family 14 G-POST |
| **Geometry-of-truth / sentiment** (saprmarks code) | **0** organism DEFINED by its single linear direction | 0 | 0 | 1 | 0 | 2 probe accuracy, ground-truth | 0 | 1 | **D1=0,D3=0 → FAIL** | 4 | **NO-GO** | matches Family 15. The purest direction control |
| **Alignment-faking / deception** (mostly closed) | **0** persona/state direction, distributed reasoning | 0 | 1 | 0 | 0 | 0 reasoning-trace judge | 0 | 0 mostly closed, un-runnable | **D1=0,D4=0 → FAIL** | 1 | **NO-GO** (artifact + mechanism) | matches §3.5. Down-rank also on D8 (closed) |
| **CoT-unfaithfulness** (Turpin/FaithCoT) | 1 partly relational | **0** answer MLP-computed across the chain, not one edge | 1 | 0 answer overdetermined by whole CoT | 0 | 1 flip-rate (judge-free) but target not FRA-shaped | 0 macro/distributed | 0 needs CoT model (Qwen-7B+) | **D2=0,D4=0 → FAIL** | 3 | **NO-GO** (measure-only) | matches §3.6 |
| **Reward-hacking / spec-gaming** (agentic harnesses) | 1 | 0 multi-step agentic | 1 | 0 | 0 | 2 deterministic env-edit | 0 | 0 heavy agentic, off-budget | **D2=0,D4=0 → FAIL** | 4 | **NO-GO** (impractical artifact) | matches Family 11 |

### 1D. Tier counts

- **TIER-1: 4** — factual-recall edit (S=14), copy-suppression (16), in-context backdoor (15), induction (15).
  *Of these, 3 are ALREADY-MEASURED wins; only factual-recall is a genuinely open T1.*
- **TIER-2: 2** — prompt injection (12), RAG-poisoning (12).
- **NO-GO: 16** — 1 by-S (binding, S=9 gate-pass) + 3 structural-gate-fails (IOI, greater-than, weight-baked
  sleeper) + sycophancy (D4=0) + function-vectors (D4=0) + 9 direction/G-post/artifact controls.

Total scored: 22 distinct organisms across 16 families. **The rubric reproduces all prior-session verdicts**
(EM, sycophancy, refusal, weight-baked sleeper, greater-than, IOI all NO-GO) — the control bank is intact.

---

## 2. RANKED SHORTLIST (gate-passers, ordered by A-magnitude × runnability × metric-cleanliness × baseline-strength)

Ordering key per CAMPAIGN: (a) expected FRA-advantage A, (b) runnability NOW on RunPod, (c) metric cleanliness
(ground-truth > judge), (d) baseline-to-beat strength. **Crucially I re-weight by NOVELTY** — 3 of the 4 T1
are banked wins, so "what does running it teach us that we don't already know" is the real tiebreaker.

| Rank | Organism | S/TIER | Expected A | Novelty (is the result NEW?) | Metric | Baseline | Verdict |
|---|---|---|---|---|---|---|---|
| **1** | **Factual-recall edit** | 14 / T1 | predicted 10³ (unique conj); **but the test is whether it transfers from synthetic to real facts** | **HIGH — never run as a selectivity test; the flagship falsifier (§6.1)** | ground-truth held-out accuracy | **ROME/MEMIT (strongest in interp) + linear subject-steer** | **RUN FIRST** |
| 2 | Prompt injection | 12 / T2 | predicted 10–10² (semi-unique); D4-risk caps it | HIGH — fresh organism, never FRA-tested, top practical-safety value | ground-truth injected-string match | prompt-hardening / ignore-injection steer | strong #2 |
| 3 | Copy-suppression L10H7 | 16 / T1 | **measured 516×** | **LOW — already won.** A fresh *SAE-feature-pair decomposition* of L10H7's QK (the lit never did this) is the only new content; the behavioral win is banked | ground-truth logit | head-ablation | only if a decomposition deliverable is wanted |
| 4 | In-context backdoor | 15 / T1 | measured 25×; class-union (H2) open | MEDIUM — banked single-trigger; novelty = real published poisoning organism (Multi-Trigger 2507.11112) + class-level family-union edit | exact-match payload | DoM/ActAdd | good contrast / external-validity |
| 5 | RAG-poisoning | 12 / T2 | predicted 10–10² iff content-discriminator, else 1.9× floor | MEDIUM — same slot-vs-content crux as binding; injection is the cleaner version of the same mechanism | ground-truth target-answer | passage-ablation | dominated by #2 |
| 6 | Induction (anchor) | 15 / T1 ref | measured 15× | NONE — pure re-run | ground-truth | DoM | reference only, do not spend on it |

**Decisive read:** the shortlist collapses to a two-horse race for *first run* — **factual-recall (highest
novelty × strongest baseline × cleanest falsifier value)** vs **prompt-injection (highest practical-safety,
fresh organism, but D4-redundancy risk and no strong existing baseline)**. Copy-suppression / induction /
in-context-backdoor are banked; they are contrast points, not the first spend.

---

## 3. CONCRETE EXPERIMENT PROPOSALS (top 3)

All three wire in the SCREENING_RUBRIC §4 selectivity win-test verbatim: Step-0 matched on-target removal,
Step-2 three held-out collateral sets, Step-3 A = collateral(best baseline)/collateral(FRA) with WIN ≥ 3×,
STRONG ≥ 10×, Step-4 confound controls (C1 base-reversion OLS, C3 localization ≤3 heads, C4 coherence,
matched-removal audit, LBNR redundancy probe).

### PROPOSAL 1 — FACTUAL-RECALL EDIT (the flagship) ★ RUN FIRST

- **Model + artifact:** `gemma-2-2b` (base, not -it — the Geva/ROME circuit is characterized on base;
  open, no gating) + **GemmaScope attention SAEs `google/gemma-scope-2b-pt-att`** (all layers, the
  FRA-native attention SAE the campaign already uses; plus `gemma-scope-2b-pt-res` for the linear-steer
  baseline endpoint). Fits an L4/L40 pod. Runnable NOW.
- **Eval set + GROUND-TRUTH metric:** **CounterFact** (Meng et al., public) restricted to facts gemma-2-2b
  reliably recalls (filter to P(true object) > 0.3 at baseline — the probe already confirms famous facts like
  "capital of France"→Paris are recalled). Metric = **object-token probability / exact-match accuracy**, no
  judge. Target removal: P(target object) → < 20% (Step-0 operating point).
- **FRA variant (§4 selectivity protocol):** the **(subject-content query × relation-content key)
  differential cell-edit** at the **upper-layer attribute-extraction heads** — rank heads CAUSALLY by
  edge-cut effect on P(object) (never raw attention), expect the name-mover/extraction hub-heads
  (the probe's edge sat on L25/L17/L7 head-pairs); take the top ≤3. Subtract `c·u^μ_q·u^ν_k·ω_{μν}` on the
  (subject × relation) conjunction. Differential form (target-edge minus a sibling-relation edge) to buy
  target-specificity, per THEORY's differential-cell algebra.
- **Baseline to beat (tune each HARD):** **(B1) linear subject-steer** — DoM/projection-removal of the
  subject feature (the fair linear baseline). **(B2) ROME** (rank-1 MLP edit at the subject token, the
  strongest surgical baseline in interp; MEMIT as the mass-edit variant). Tune ROME's layer + the steer's
  α to the SAME P(object)<20% operating point.
- **Three held-out collateral sets (ground-truth, no judge):**
  (a) **same subject, other relations** — France's *other* facts ("France is in"→Europe, currency, language).
  (b) **other subjects, same relation** — other countries'/entities' capital ("capital of Spain"→Madrid).
  (c) **unrelated capability** — a general-knowledge accuracy battery + perplexity on held-out text.
- **WIN numeric:** A = collateral(ROME or subject-steer) / collateral(FRA) on (a)/(b) at matched removal.
  **WIN if A ≥ 3× on (a) OR (b)** and FRA no worse on (c). **STRONG if A ≥ 10×.** Pre-registered prediction
  (§3.1): ROME bleeds ≥15% onto the subject's other facts; FRA <5% → A ≥ 3×.
- **BIGGEST RISK:** **D2 — the object is recomputed in a downstream MLP** (the greater-than failure: the
  edge reads the subject, the MLP does the lookup). If so, the edge-cut dents target accuracy <30% and FRA
  controls *attending*, not *recall*. **This is exactly the §6.1 flagship falsifier** — and it is the whole
  reason this is the experiment worth running. Secondary risk: **CCF=0.000 in the probe** — the SAE-feature
  edge may not reconstruct enough score on the gemma-att SAE to reach the operating point at faithful c≈1–2
  (a *reach* ceiling, not a separability failure).
- **CHEAPEST GO/NO-GO PRE-CHECK (≈1 cheap pod, no campaign):** **the LBNR + reach gate, already half-done.**
  The probe gives R=+0.86 (cut suppresses fact 86% → load-bearing, D4 PASS). The MISSING half is: **does the
  *SAE-feature-pair* cell-edit (not the raw edge-cut) reproduce that suppression at c≈1–2?** Run the
  feature-resolved cell-edit on 20–30 recalled CounterFact facts; require **P(object) drop ≥ 50% via the
  cell-edit** (resolves the CCF=0.000 reach worry) AND **R stays ≥0.5**. If the cell-edit can't move the
  fact (reach ceiling) OR a held-out same-relation fact also drops sharply (MLP-distributed, D2 fail) → STOP,
  report as the §6.1 pre-registered flagship negative (a *sharp publishable result*: "structurally-ideal
  broad×broad relation is G-post / unreachable in a real LLM"). If it passes → commit the full §4 vs-ROME test.

### PROPOSAL 2 — PROMPT-INJECTION / INSTRUCTION-ROUTING EDGE (strong #2)

- **Model + artifact:** `gemma-2-2b-it` (chat, follows system+user instructions) + GemmaScope-att SAEs.
  Eval data: **Open-Prompt-Injection** (`github.com/liu00222/Open-Prompt-Injection`, public) — system/user
  task + an injected imperative in tool/retrieved content. Runnable on L40.
- **GROUND-TRUTH metric:** did the **injected instruction's target string appear** vs did the legitimate
  task complete — **exact-match attack-success-rate** (G-score by construction, no judge). Target:
  injection-following → < 0.2.
- **FRA variant:** cut the **(answer-position query × injected-imperative-span key)** edge while preserving
  the **(answer × system-instruction)** edge. Causally rank heads by injection-following drop; top ≤3.
  Broad×broad: "injected imperative" content × "comply" reused across the injection set.
- **Baseline to beat:** **(B1)** a linear "ignore-injection" / instruction-source steer (DoM on the
  system-vs-injected contrast). **(B2)** prompt-hardening (the practical baseline) + attention-source
  ablation. No strong weight-edit baseline exists here → D8=1 (weaker than the flagship's ROME).
- **Three held-out collateral sets:** (a) **same system instruction, no injection** — legitimate
  instruction-following must survive within 10%. (b) **other injections of the same imperative class** —
  tests the key-endpoint's other uses. (c) **clean-task accuracy + perplexity** battery.
- **WIN numeric:** injection-following ↓ ≥ 50% while legitimate-instruction-following within 10%; A ≥ 3× vs
  the linear instruction-source steer on (a).
- **BIGGEST RISK:** **D4 — distributional redundancy** (the refusal/delimiter class): an injected span is
  markable by MANY cues (delimiters, imperative mood, position), so cutting one (answer × injected-span) edge
  is **precise-but-inert** — the model re-routes via a backup cue and injection-following is restored. This is
  the rubric's pre-registered #2 risk and the reason it's T2 not T1.
- **CHEAPEST GO/NO-GO PRE-CHECK:** the **LBNR redundancy probe** — on 20 injections, cut the top injection
  edge and check ASR drops ≥50% AND is NOT restored (no backup cue). If ASR is restored by redundant cues
  (R<0.5) → STOP, pre-registered negative (injection is distributionally redundant, like refusal). Cheap,
  one pod, no baseline needed.

### PROPOSAL 3 — COPY-SUPPRESSION L10H7: a fresh FRA-feature-pair DECOMPOSITION (only if a decomposition deliverable is wanted)

*Included because the task asked to verify the 22.7× win is a fresh contribution, not a re-run. **Verdict: the
behavioral selectivity win is BANKED (516× natural text + transfer, fra_win CAMPAIGN_REPORT §"copy-suppression
release"). Re-measuring the A-ratio adds nothing.*** The ONE genuinely new thing FRA can contribute that the
McDougall et al. literature never did: **decompose L10H7's QK bilinear form into named SAE feature-pairs** —
i.e. *which* (query-content × key-content) cells carry the copy-suppression, as an interpretability result, not
a steering result. This is a *characterization* deliverable (a labeled cell table for the canonical negative
head), low-risk and cheap (gpt2 + res-jb, the existing substrate), but it is NOT a new behavioral win and
should NOT displace the flagship. Run only as a fast, safe "FRA tells you something new about a known circuit"
companion result if the flagship's pre-check is still queued.

- **Risk:** none behavioral; the only risk is that the decomposition is uninterpretable (cells don't map to
  nameable features) — itself a (mild) negative result. **Pre-check:** does the top-cell of L10H7's QK have an
  SAE-feature pair with > 50% of the edge weight? If yes, it's a clean labeled cell; if diffuse, report as-is.

---

## 4. RECOMMENDATION — evaluate FACTUAL-RECALL EDIT first

**#1: Factual-recall edit (CounterFact, gemma-2-2b + GemmaScope-att, vs ROME/MEMIT + linear subject-steer).**

**Why it is the #1 P(clean win) × runnability × low-risk-of-wasted-spend choice:**

1. **It is the only TIER-1 organism that is genuinely OPEN.** Copy-suppression (516×), induction (15×),
   in-context backdoor (25×) are *banked* — re-running them re-confirms known numbers. The flagship has had
   only a cheap gate probe; **its selectivity test has never been run.**
2. **The cheap probe already de-risked the scariest gate.** `bhhu4qdtg.output`: attn=0.81 (strong edge),
   **R=+0.86** (the extraction edge is load-bearing — D4 PASS, the IOI/refusal redundancy failure is NOT
   present here). The remaining uncertainty (D2 MLP-extraction, CCF=0.000 reach) is *exactly* the crux the
   experiment is designed to resolve.
3. **It carries the strongest baseline in interpretability (ROME/MEMIT).** Per the rubric §8 + writing_
   instructions, beating a strong surgical baseline's *selectivity* is the meaningful, publishable claim;
   "FRA works at all" against no baseline (injection's situation) is weak. D8=2 vs injection's D8=1.
4. **Cleanest ground-truth metric (D6=2): held-out factual accuracy, no judge** — directly avoids the EM
   judge-bucketing footgun that manufactured the prior false positive.
5. **It is the campaign's designated falsifier (§6.1):** "if the flagship fails, the broad×broad synthetic
   result does not transfer to real LLMs" — a sharp, publishable negative. So **either outcome is a result.**
   This maximizes information per pod-dollar regardless of sign.
6. **Runnable NOW, small** (gemma-2-2b + an existing SAE suite the campaign already drives), no gating, fits
   the cheapest GPU tier.

Prompt-injection is the strong #2 (higher practical-safety upside, fresh organism) but loses the first slot on
two counts: a **weaker baseline-to-beat (D8=1, no ROME-equivalent)** and the **un-resolved D4 redundancy gate
(=1, the refusal class)** — its expected A is lower and its risk of a precise-but-inert null is higher. It is
the natural SECOND run, especially if the flagship lands cleanly and a practical-safety demonstration is wanted.

**THE CONCRETE FIRST EXPERIMENT + ITS GO/NO-GO PRE-CHECK (one cheap pod, before any campaign):**

> On gemma-2-2b + gemma-scope-2b-pt-att, take 20–30 CounterFact facts the model recalls (P(object)>0.3).
> Apply the **FRA (subject × relation) SAE-feature-pair cell-edit** at the causally-top ≤3 extraction heads.
> **GO** iff: (i) the cell-edit drops P(target object) by **≥ 50%** at faithful scale c≈1–2 (resolves the
> CCF=0.000 reach worry), AND (ii) a held-out **same-relation, different-subject** fact drops by **< 15%**
> (rules out the MLP-distributed / D2-fail mode — the greater-than signature). If GO → commit the full §4
> selectivity test vs ROME/MEMIT + linear subject-steer on the three held-out collateral sets. If NO-GO →
> **STOP and report the §6.1 pre-registered flagship negative** (structurally-ideal broad×broad recall is
> G-post or unreachable on a real LLM) — itself the campaign's sharpest publishable bound.

This pre-check is small, judge-free, reuses the substrate the campaign already runs, and its outcome
deterministically routes the next spend. It is the highest information-per-dollar first move in the screen.
