# The carrier law at 2B: does the toy's carrier→tool law survive on Gemma?

*Synthesis deliverable for `bridge_to_real/`. Bridges the completed toy program
(`../bernoulli_fra/OPTIMAL_FRA_NOTE.md`, esp. §8 the hierarchy round) to the real-model campaign
(`../../fra_win/`, GPT-2-small + Gemma-2-2b). Full prediction operationalizations in
`BRIDGE_THEORY.md`; the trained-backdoor test in `BRIDGE3_PROTOCOL.md`. Both Gemma-2-2b compute halves
(Bridge-1 carrier + frontier, Bridge-3 trained-backdoor flip) are IN and filled; the only unrun items are
P4 (presence floor) and a real OV-gate organism, both marked **[future work]**.*

*Tags: **[toy-CONFIRMED]** (measured on the theory-clean toys / to float precision), **[fra_win-OBSERVED]**
(measured on GPT-2 or Gemma in the existing campaign), **[PREDICTION]** (derived, awaiting Gemma),
**[SPECULATIVE]**.*

---

## Executive summary

**The toy program's amended fra_win law — that the control method which wins is fixed by the *carrier*,
and the carrier is fixed by whether the behavior is an obligate query-key match — retro-predicts the
entire real fra_win/sleeper split, and this note states the bridge, the predictions it forces on
Gemma-2-2b, and the three places the real data already sharpens it.** Six findings carry the note. Both
compute halves are in: the law HOLDS for the match at 2B (Finding 5); the carrier read-off confirms it
under a collateral-normalized refinement (Finding 4a); the toy's clean gauge-flatness does NOT survive
(Finding 4b, ~35% RMSNorm swing); the toy's OV-*gate* control claim remains the open weak point
(Finding 3); and — **the capstone** — the carrier law **governs TRAINED backdoors**, with the FRA-QK-handle
flip between a weight-stored and an in-context-routed finetuned backdoor replicating **2/2 seeds** in
direction (a ~40,000× B/A reach ratio), exactly as predicted (Finding 6).

1. **The carrier→tool law transfers to 2B for MATCHES, and the real campaign already shows it.**
   Induction / in-context copy is the campaign's *one* obligate query-key match, and it is the
   campaign's *one* FRA-QK win (~11–90× lower held-out collateral than DoM and conv-SAE, GPT-2 and
   Gemma). Every other behavior probed (IOI, docstring, delimiter, greater-than, many-shot injection,
   PII recall, knowledge-conflict) is OV/content/direction-carried, and FRA-QK is inert or loses on
   every one. The law predicts exactly this one-QK-win structure. **[toy-CONFIRMED ⟺ fra_win-OBSERVED]** (§3)

2. **The fra_win "flip" IS the toy's carrier reversal at scale.** The same backdoor-removal task loses
   for FRA on the weight-baked sleeper (payload in OV/output → DoM/SVD win) and wins for FRA in-context
   (payload retrieved by an attention edge → FRA-QK wins). That is the carrier moving OV→QK with the
   tool following — the toy's §8 reversal, reproduced on real models. **[fra_win-OBSERVED]** (§3)

3. **Honest headline: the toy's *gate*-OV control win remains UNTESTED at scale.** The toy §8 proved that
   for an OV-carried *gate* (a reused-marginal relation), FRA-OV path-decomposition *beats* DoM (17–51×)
   and SAE (48–259×) *surgically*. **No real gate organism has been built, so this specific claim is
   unconfirmed on any real model** — and it is the note's central open item, stated first-class, not buried.
   One nuance the Gemma run adds (§4.1): FRA-OV is **not *uniformly* dominated** — it beats DoM by 2–5× on
   the induction backdoor — **but that is FRA-OV on a MATCH's content-transport, a weak/high-collateral
   effect, not the toy's surgical 17–259× GATE win.** So the softening is "FRA-OV has some value," not "the
   toy transferred." **[toy-CONFIRMED vs untested-at-scale — the open weak point]** (§4)

4. **The carrier is confirmed as QK for the match — but only under a *collateral-normalized* read, and
   the toy's clean gauge-flatness does NOT survive at 2B.** (a) **Carrier read-off [CONFIRMED with
   refinement]:** the *naive* ASR-effect ratio is a trap that points the wrong way — on king→crown the
   OV-cut removes far MORE of the backdoor (0.98 vs QK 0.14) but as a **content sledgehammer at ~50× the
   collateral**; the *collateral-normalized* read is decisive — at matched ~10% removal QK-cut costs
   **0.056 nats** vs OV-cut **2.9 nats**, so the SURGICAL carrier is QK and FRA-OV is a blunt content
   route. **The real model forces a refinement of the law: at scale, "carrier" must be defined by
   collateral-EFFICIENCY, not raw removal — the toy conflated them because its OV gate was both
   bigger-effect AND surgical; at 2B they separate.** (b) **Gauge [refinement, not a clean pass]:** the
   QK-cut effect is NOT gauge-flat like the toy's <10⁻⁹ — it moves **~35%** with the RMSNorm handling
   (king→crown: 0.14 x̂ / 0.19 resid across the two *faithful* settings), the toy's LayerNorm-hidden-carrier
   finding recurring as RMSNorm at 2B. **[Gemma-CONFIRMED with refinement]** (§5)

5. **The control frontier lands where the law says — the law HOLDS at 2B for the match.** On the
   king→crown induction backdoor FRA-QK removes at **~0.056 nats** held-out collateral, while the
   content-direction baselines pay **DoM ~5–55, conv-SAE up to 101, payload-suppress up to 328 nats** to
   reach comparable/higher removal — roughly **13–312× less collateral-efficient**. This is the exact
   **mirror of the toy hierarchy** (there FRA-OV won 17–259× because the gate was OV-carried; here FRA-QK
   wins because induction is a MATCH). **[Gemma-CONFIRMED]** (§5, §7)

   *Honest limit (ledger): FRA-QK's **reach** on Gemma is cue-dependent and mostly low — max ASR-removal
   bank 0.007, doctor 0.025, market 0.003, king 0.143 (only 2 induction heads, L15H0 causal-drop 0.21 /
   L22H4 0.05). The win is collateral-efficiency at the small removal it reaches, not full removal; single
   induction family, one seed.*

6. **CAPSTONE — the carrier law governs TRAINED backdoors, not just planted ones: FRA-QK has a viable
   handle IFF the backdoor is in-context-routed, and the flip DIRECTION replicates 2/2 seeds.** Finetuning
   Gemma-2-2b on the **same** special-token trigger with two payload mechanisms flips the carrier exactly as
   predicted. **(A) fixed-string** (the token→a fixed string, weight-stored): the FRA-QK cut is **INERT on
   both seeds** (max ASR effect **1.8×10⁻⁵ / 7.2×10⁻⁷** — a noise floor), DoM the only handle → content/OV
   route. **(B) in-context-retrieved** (the finetune teaches the *routing*; **generalizes to held-out
   payloads**, ASR **0.80 / 0.88**, proving copy not memory): the FRA-QK cut is a **viable handle on both
   seeds** (max ASR effect **0.72 / 0.028**) → QK match. **The B/A reach ratio is ~40,000× on both seeds**,
   but it is *direction-robust, magnitude-soft*: partly a **noise-floor artifact** (A's denominator is
   FRA-QK-inert), and **B's absolute reach is seed-variable (0.028–0.72**, depending which heads LoRA picks
   and the SAE resolution), so the specific "72% removal" was seed-1. **The robust, falsifiable claim is the
   DIRECTION — FRA-QK has a handle iff the payload is in-context-routed** (had A shown a handle or B not, the
   law would be falsified for trained backdoors). The carry heads confirm the mechanism split (A =
   payload-writers; B = induction-like retrievers overlapping Bridge-1's general induction heads — **L22H4
   on both seeds**, L15H0 on seed 1). The finetune **creates** the QK edge in B and **bakes the string into
   weights** in A. This is the strongest form of the bridge — the ranking flips on a *trained*, not planted,
   backdoor, and the **flip direction is what governs the trained model (2/2 seeds).** *And a second,
   genuinely mechanistic finding falls out of the collateral frontier — the honest, more interesting
   result:* **the trained route is ENTANGLED with general copying, so it is not surgically removable.** The B frontier (§7) shows FRA-QK stays the better tool but only **~3× cheaper
   than DoM** (QK 10.7 vs DoM 36.1 nats @10% removal; 20.1 vs 56.3 @30%; DoM runs to 76→1014 nats at full
   removal, FRA-QK ceiling ~0.49) at **high absolute collateral (~10–20 nats)** — *not* the ~0.056-nat /
   13–312× surgical win of the *planted* induction backdoor (§5). **The mechanism, read straight off the
   carry heads:** B's carrying heads **share a general induction head with the planted Bridge-1 backdoor
   (L22H4 on both seeds; L15H0 on seed 1)** — so finetuning installed the trigger→copy route **on top of**
   the model's existing general copy circuitry, not as a separable edge. Cutting B's route therefore
   necessarily damages *general* copying (the no-trigger copy surface pays), which is exactly why the
   collateral is high. **Planted backdoor = an isolable edge → surgically removable (200×); trained backdoor
   = piggybacks on existing induction → FRA-QK stays the better tool but pays real collateral.** *Safety
   implication: trained backdoors that reuse general circuitry are harder to remove surgically than planted
   ones precisely because they are entangled with legitimate behavior — a caution for FRA-based backdoor
   removal on real finetuned models.* *2-seed direction-robust (flip present iff in-context, 2/2); the
   collateral frontier is seed-1; one lexicon.* **[Gemma-CONFIRMED — trained, 2-seed direction-robust]** (§7)

---

## 1. Thesis

**The toy's amended fra_win law is a claim about *where a behavior's minimal sufficient cause lives*,
and it transfers to Gemma-2-2b for the one structural class it names — obligate query-key matches —
with three honest refinements the single-layer toy could not see.** Written as a function:
`winning-control-method = f(carrier)`, `carrier = f(is-it-an-obligate-match)`. An *obligate match* is a
two-position product that **cannot be re-expressed in the value/output pathway** — the copy
`[A][B]…[A]→[B]`, where the destination must read the key *because it matches the query*. The toy proved
(3 seeds, theory-clean attention-only platform) that a two-position product is **necessary but not
sufficient** for an FRA-QK handle: a content *gate* is a two-position product too, yet the trained model
carries it in **OV, not QK** [toy-CONFIRMED]. Only an obligate match is unrealisable in OV and therefore
QK-carried; the carrier then sets the tool — FRA-QK for matches, an OV/content method for content. The
three refinements at scale, each a real-data result:

- **(R1)** the OV branch bifurcates on `reuse(marginal)` — FRA-OV wins only for a *reused-marginal
  relation*, DoM wins for a *no-reuse marginal direction* (§4.1); the toy's OV control win is the former,
  every real OV case so far is the latter.
- **(R2)** carrier predicts the *tool* but not the *behavioral efficacy* — the real win additionally
  needs a load-bearing / non-redundant edge, which IOI's backup heads violate (§4.2).
- **(R3)** real models add a third regime with no toy cell — distributed direction-routing (many-shot),
  where no method removes cleanly (§4.3).

## 2. The object dictionary (toy ↔ real)

| Toy object | Real-model counterpart | What changes at scale |
|---|---|---|
| planted TopK-SAE latents on ground-truth **D** | trained **GemmaScope** JumpReLU SAE on `resid_pre` | **χ-incomplete (N>d always)**; ~56% norm recovery, per-edge corr ~0.54; `error×·` is a real leak channel |
| dictionary rows $d_i$ | SAE decoder rows `W_dec[i]` | learned, non-orthogonal; hedging/absorption; collateral slope degrades ~2→~1 vs a matched code |
| coupling $\omega_{ij}=(d_iW_Q)(d_jW_K)^\top/\sqrt d$ | $S[q,k]=\sum_{ij}(z_i(q)W_{dec}[i]W_Q)(z_j(k)W_{dec}[j]W_K)^\top/\sqrt d$ | real $W_Q,W_K$; multi-head sum (Tier-2 gauge); **RoPE** → relative-offset $R_{q-k}$ |
| G1 embedding-resplit gauge | SAE `b_dec` resplit + RMSNorm $\gamma\!\leftrightarrow\!W_{QK}$ fold + softmax key-pedestal | more dials; **softcap is not a gauge** |
| QK-null vs OV-handle cut asymmetry | fra_win QK-edit (`hook_attn_scores`) vs DoM/OV | measured behaviorally (ASR, KL) on real text |
| **LayerNorm is a hidden carrier** | GPT-2 LN centering = the 50–75% recon shortfall; **Gemma RMSNorm magnitude-exact (1.07)** | RMSNorm folds exactly for magnitude, but its per-token scalar is still a candidate hidden *gate*-carrier (unchecked) |
| control frontier (closed-form MSE) | held-out-KL vs ASR-removal frontier | KL/ASR on real text, Pareto-matched |
| $c^\ast=1/\rho_{\mathrm{path}}$ sufficiency | reach ceiling = SAE-explained × head-coverage × **softcap headroom $\tanh'(s/50)$** | softcap is a new reach-limiter; width sweet spot 65k (non-monotonic) |
| presence floor $\mathrm{Corr}^2(\text{belief},\text{obs})$ | probe-decodable trigger floor (current-token direct path) | measured by a trained probe |
| optimizer-pinned ≠ computation | *don't* trust cross-seed FRA reproducibility as causality | the right control is the gauge-dial test, not seed variance |

**New at scale, no toy analog:** the Gemma attention **soft-cap** (a true nonlinearity that caps
*reach*), **SAE reconstruction error** (a leak the planted dictionary lacks), and **circuit redundancy /
backup heads** (which a single-layer toy cannot exhibit — R2).

## 3. What fra_win already confirms

1. **The "exactly one QK win" structure IS the law.** Induction is the only obligate match and the only
   FRA-QK win; every other probed behavior is OV/content/direction-carried and FRA-QK is inert/loses.
   The law predicts precisely this. **[fra_win-OBSERVED]**
2. **The flip is the carrier reversal.** Weight-baked (FRA loses) ↔ in-context (FRA wins), same task.
   Concrete: IC4 (GPT-2) FRA-QK 0.07 vs DoM 1.83 (27×), conv-SAE 6.06 (90×); Gemma g4 @0.3 removal FRA
   0.52 vs DoM 13.5 (26×), conv-SAE 11.9 (23×). **[fra_win-OBSERVED]**
3. **RMSNorm exact (1.07), LayerNorm not (50–75%).** Confirms the nonlinearity-ledger claim. **[fra_win-OBSERVED]**
4. **Reach ≠ separability.** The reach ceiling is an SAE-granularity/softcap/head-coverage property, not
   a separability failure — the real-model image of the toy's "$c^\ast$ is architecture, not FRA." **[fra_win-OBSERVED]**

## 4. The three contradictions — as first-class results

### 4.1 The toy's OV-side control win does not transfer (the honest headline)

The toy §8 says an OV-carried *gate* → FRA-OV **beats** DoM/SAE by 17–259× [toy-CONFIRMED]. The real OV
side has **no** clean FRA-OV control win: weight-baked sleeper → FRA-OV **loses** to DoM/SVD
[fra_win-OBSERVED]; fra_pii → FRA-OV content-selective but **reach-insufficient**, only a position-locked
SAE clears the bar [fra_win-OBSERVED]. **Both are "OV-carried content," so carrier alone does not pick
the winner on the OV branch.** The resolving variable is the magnitude law
`A ≈ reuse(marginal|eval)/reuse(conjunction|eval)`:

- **OV-carried *relation* with a reused marginal** (the toy gate: the source = the *parent*, which the
  model still self-predicts). FRA-OV isolates the transported relation, spares the marginal → **FRA-OV
  wins.** [toy-CONFIRMED]
- **OV-carried *marginal direction* with no reuse** (the sleeper payload: a direction with ~0 legitimate
  use, `reuse(marginal)→0`, `A→1`). DoM/SVD on the marginal is already surgical → **DoM wins, FRA-OV
  adds nothing.** [fra_win-OBSERVED]

So the corollary is: **QK branch clean** (match → FRA-QK, always); **OV branch splits on
`reuse(marginal)`.** The consequence for the empirical program: **the toy's headline OV control claim is
unconfirmed at scale, and the real cases refute its naive reading.** The missing experiment is a real
OV-*gate* organism (a reused-marginal relation); until it exists, do not claim FRA-OV control wins on any
real model.

**New Gemma data point — a partial softening, NOT a confirmation of the toy OV win. Keep the two apart:**
- **The toy's FRA-OV win (17–259×) was on a GATE organism** (hierarchy: a reused-marginal *relation*). **No
  real gate organism has been tested, so Contradiction 1 STANDS for gates** — the toy's surgical,
  order-of-magnitude OV-gate win is still unconfirmed at scale.
- **Bridge-compute's FRA-OV-beats-DoM-by-2–5×** (2.4 vs DoM 13.5 / conv-SAE 11.9 / payload 5.8 nats @0.3
  removal, and it reaches full removal) **is on an INDUCTION backdoor — a MATCH** — where FRA-OV cuts the
  payload *content the head transports*. That is FRA-OV on a match's content-transport: a **different and
  much weaker** effect (2–5×, not 17–259×) at **high absolute collateral** (2.4 nats), with the advantage
  eroding as removal rises (2.4 → 5.1 → into the DoM range) and FRA-QK undercutting it ~50× where both
  operate.

So the *only* correction to Contradiction 1 is that **FRA-OV is not *uniformly* dominated at scale** as its
prior evidence (weight-baked sleeper, fra_pii) suggested — it has *some* value on a match's transport. The
headline is unchanged: **the toy's gate-OV control win remains untested**, and the dedicated OV-gate
organism (a reused-marginal relation) is still the experiment that would settle it.

### 4.2 Carrier predicts the tool, not the behavioral efficacy (IOI)

IOI's END→IO edge is a genuine content×content QK conjunction (CCF-high), so the law assigns it the QK
carrier and FRA-QK the right *tool* — yet the FRA-QK edit moves the IO−S logit by only 0.17 because
**backup name-movers self-repair** [fra_win-OBSERVED]. The single-layer toy has no circuit redundancy, so
it cannot surface this. **The real-model win needs an extra load-bearing / non-redundant (LBNR) clause the
toy law omits:** carrier→tool is necessary; carrier→*behavioral change* additionally requires the edge be
causally load-bearing with no backup. Always quote the scaled law as `carrier→tool AND (LBNR ⇒ effect)`.

### 4.3 A third regime with no toy cell (distributed direction-routing)

The many-shot injection rides a *distributed ICL task-direction* across many weak edges,
representationally identical to a legitimate behavior — no separable link, no clean removal by anyone
[fra_win-OBSERVED]. The toy's two-position-product framing has no cell for "distributed direction with no
clean carrier"; it is the taxonomy's bottom row, outside the toy's current reach. Acknowledge it; don't
force it into the QK/OV dichotomy.

## 5. The Gemma tests — results

Gemma-2-2b, GemmaScope 65k, 4 induction-backdoor cue pairs (bank→river, king→crown, doctor→water,
market→gold), induction heads L15H0 (causal-drop 0.21) + L22H4 (0.05). Data in
`out/bridge1_carrier.json`, `out/bridge1_frontier.json`. Full operationalizations in `BRIDGE_THEORY.md` §3.

**The honest boundary in three statements** (this is the real Bridge-1 result — the law transfers for
tool-choice + collateral-efficiency, but not as a raw-effect mirror, and the gate-OV win is still untested
at scale):

1. **[SOLID] The carrier→tool law holds.** FRA-QK is the surgical, collateral-efficient tool for the
   induction MATCH — **13–312× cheaper than the content-direction baselines** (conv-SAE's cheapest point
   through payload-suppress) at matched removal, where it reaches (reach-capped, see below).
2. **[SOLID] The *raw-effect* P3 mirror is REFUTED.** The OV-cut removes *more* raw ASR (0.98 vs QK 0.14)
   as a content sledgehammer; the carrier read-off is correct **only** collateral-normalized, not on raw
   removal.
3. **[NUANCE — do not overclaim] FRA-OV path-cutting has *some* value at scale, but it does NOT confirm
   the toy.** FRA-OV beats DoM/conv-SAE/payload by **2–5×** here (2.4 vs 13.5/11.9/5.8 nats @0.3) and
   reaches full removal — so FRA-OV is **not uniformly dominated**, softening Contradiction 1's existing
   evidence. **But this is FRA-OV on a MATCH's content-transport** (cutting the payload content the
   induction head copies), a weaker, higher-collateral effect (2–5×, 2.4 nats absolute) — **not** the
   toy's **17–259× surgical GATE-OV win**, which was on a reused-marginal *relation* and **remains
   untested at scale** because no real gate organism exists (§4.1).

- **P2 (frontier) — [Gemma-CONFIRMED]. The law holds at 2B.** On king→crown, FRA-QK removes the backdoor
  at **~0.056 nats** held-out collateral (at its 0.14 reach ceiling), while the content-direction methods
  pay **DoM ~5–55, conv-SAE up to 101, payload-suppress up to 328 nats** to reach comparable/higher removal
  — 13–312× less collateral-efficient. The matched-removal frontier (mean over the cues that *reach* the
  threshold, so FRA-QK — reach-capped — is not in this table; it operates only at ≤0.14 removal, far below
  and far cheaper): at **0.3 removal**, FRA-OV **2.4** < payload-suppress **5.8** < conv-SAE **11.9** ≈
  DoM **13.5** nats; at **0.7 removal**, FRA-OV **5.1** < payload **10.8** < DoM **18.8** < conv-SAE
  **23.4**. This is the exact **mirror of the toy hierarchy** — there FRA-OV won because the gate was
  OV-carried; here FRA-QK is the surgical winner because induction is a MATCH. **(The FRA-OV row beating
  DoM here is statement 3 above — a match's transport, not the toy gate.)**
- **P3 (carrier read-off) — [Gemma-CONFIRMED with refinement]. The naive ASR-ratio is a TRAP; use the
  collateral-normalized read.** On king→crown the OV-cut removes **more** ASR (0.98 vs QK 0.14), so the
  raw QK/OV ASR-effect ratio points the *wrong* way (it would call OV the carrier). But the OV-cut is a
  **content sledgehammer**: at matched ~10% removal, QK-cut collateral **0.056 nats** vs OV-cut **2.9
  nats** (~50×), and FRA-OV's collateral keeps climbing with removal (→5–8 nats, into the DoM range) while
  FRA-QK stays flat and surgical. **So the SURGICAL carrier is QK, and the law's carrier read-off must be
  defined by collateral-EFFICIENCY, not raw removal.** The refinement is real: the toy conflated
  effect-size and surgicality because its OV gate was *both* bigger-effect and surgical; at 2B the two
  separate, and only the collateral-normalized read recovers the correct carrier.
- **P1 (gauge) — [refinement, NOT a clean pass]. RMSNorm is a partial hidden carrier at scale.** The
  QK-cut effect is *not* gauge-flat like the toy's <10⁻⁹ cut-effect std. It moves **~35%** between the two
  *faithful* RMSNorm-handling conventions (king→crown at max c: **0.14** with the SAE-reconstruction rms
  x̂ / **0.19** with the true-resid rms; the no-correction "none" setting reads 0.50–0.59 and is discarded
  as unfaithful). This is **the toy's LayerNorm-hidden-carrier finding recurring as RMSNorm at 2B** — the
  RMS-folding convention is a residual attribution freedom the FRA magnitude-correction does *not* fully
  remove. It does not overturn the carrier verdict (QK stays the surgical carrier under both faithful
  settings), but it caps the precision of any FRA-QK cut effect at the ~35% level and must be reported.
- **P4 (use vs presence) — [not run, future work].** The presence-floor probe (probe accuracy under
  {clean, edited, counter-steered} vs the current-token direct-path floor) was out of this run's
  compute-time scope; the toy prediction (use removable, presence floored) stands as untested-at-scale.
- **P-note (RMSNorm carrier).** P1 already shows the RMSNorm convention is load-bearing at the ~35% level
  even for the *match* — so for any *gate* extension, the linearize/freeze-rms control is now *required*,
  not optional (the toy's LN-linearization control, promoted by the P1 result). **[future work — needs a
  gate organism]**

## 6. Bridge 2 — the decision rule for natural LM tasks

**Before choosing a control method, read off the carrier (P3: QK-cut vs OV-cut asymmetry) + run the
CCF/LBNR screen; then:**

- **Match side → FRA-QK.** Induction / verbatim copy [fra_win-OBSERVED]; **acronym letter-movers**
  (query = spelled-letter state, magnitude law A~26,000×, `g1_acronym.py` screened) [fra_win-OBSERVED,
  partial]; **content-distinctive in-context retrieval** ("red box holds a frog", head-localized L15H0/
  L18H6, causal, A~1100×) [fra_win-OBSERVED, green-lit]. *Caveat:* retrieval is on the match side **only**
  when the query is distinctive CONTENT; a generic ROLE query (which-box, entity-slot) makes the
  conjunction recur (A→1.9×, entity-PII no-op).
- **Gate / aggregate / direction side → DoM/SAE (FRA-QK inert).** Sentiment/topic aggregation (OV
  aggregate); PII banks / attribute-by-slot (reach-insufficient even for FRA-OV); style/register (ICL
  task-direction, DoM removes but entangled with legitimate style).

**Rule for a usage guide:** *if the behavior is an obligate content match with a distinctive query and a
load-bearing, non-redundant edge, the FRA-QK bilinear edit is the only tool that cuts the link while
sparing both endpoints; otherwise it is a direction, and a difference-of-means/SVD vector is the right,
cheaper tool.* Acronym and retrieval are the partial natural-LM confirmations in hand; turning either into
a full held-out-collateral frontier is the concrete Bridge-2 experiment.

**The full runnable decision procedure — the four-step CCF → carrier-read-off → LBNR → route guide, with a
routing table and three worked examples (acronym 26,000×, IOI self-repair 0.17, PII/many-shot) — is
`BRIDGE2_GUIDE.md`.**

## 7. Bridge 3 — the trained-backdoor test [Gemma-CONFIRMED — the flip landed]

The user's special-token finetune moves the law from *planted/prompted* backdoors to a **trained** one,
and the carrier law **governs it**. The same special-token trigger was finetuned into Gemma-2-2b two ways
(LoRA; spec in `BRIDGE3_PROTOCOL.md`); the carrier read-off (QK-cut vs DoM-cut, raw ASR effect at faithful
scale) flips exactly as predicted, and the **flip direction replicates across 2 seeds**
(`out/bridge3_results.json` = seed 1, `out/bridge3_replicate.json` = seed 2, `bridge3_carrier.log`):

| Variant | Payload mechanism | Carry heads (edge-cut ranked) | FRA-QK-cut max ASR (seed 1 / seed 2) | DoM-cut max ASR | Carrier verdict | Predicted? |
|---|---|---|---|---|---|---|
| **A — fixed-string** | weight-stored (base P 1.00) | **payload-writers** (s1: L19H4/L12H6/L16H2; s2: L19H4/L3H0/L21H7) | **1.8×10⁻⁵ / 7.2×10⁻⁷ (INERT — noise floor)** | 1.0 | content/OV → **DoM the only handle** | ✓✓ |
| **B — in-context-retrieved** | context-routing (generalizes to **held-out** payloads, ASR 0.80 / 0.88) | **induction-like retrievers** (s1: L22H5/L22H4/…/L15H0; s2: L22H5/L22H4/L22H3/L20H3) | **0.72 / 0.028 (viable handle)** | 1.0 | QK match → **FRA-QK works** | ✓✓ |

**The headline: FRA-QK has a viable handle IFF the payload is in-context-routed, and the DIRECTION
replicates 2/2 seeds — the B/A reach ratio is ~40,000× on both** (0.72/1.8×10⁻⁵ and 0.028/7.2×10⁻⁷), from
the SAME trigger with only the payload *mechanism* differing. Finetuning **creates** the QK edge in B (it
teaches the copy-routing, which is why B generalizes to strings never seen in training) and **bakes the
string into weights** in A (no QK edge to cut). **Two honest qualifiers:** (i) the ~40,000× ratio is
*direction-robust but magnitude-soft* — partly a **noise-floor artifact** (A's denominator is FRA-QK-inert),
and **B's absolute reach is seed-variable (0.028–0.72)**, so "72% removal" was seed-1-specific; (ii) the
robust, falsifiable claim is the **direction** (handle iff in-context-routed) — had FRA-QK worked on A or
failed on B, the law would be falsified for trained backdoors. **This is the strongest form of the bridge:
the carrier law predicts the removal-handle on a *trained*, not planted, backdoor.**

**The B collateral frontier — measured (`out/bridge3_frontier_B.json`): handle presence flips cleanly,
surgical low-collateral control does NOT.** FRA-QK-cut vs DoM-cut on Variant B, held-out KL at matched
removal:

| matched removal | FRA-QK-cut collateral | DoM-cut collateral | FRA-QK advantage |
|---|---|---|---|
| 10% | **10.7 nats** | 36.1 nats | ~3.4× |
| 30% | **20.1 nats** | 56.3 nats | ~2.8× |
| 50% | (QK reach-capped ~0.49) | 76.5 nats | — |

So on the *trained* model FRA-QK is only **~3× more collateral-efficient than DoM, at high absolute
collateral (~10–20 nats)** — **not** the ~0.056-nat / 13–312× surgical win of the *planted* induction
backdoor (§5). **The clean surgical low-collateral control is a PLANTED-backdoor property; on the
LoRA-trained model only handle-*presence* (the ~40,000× direction flip, 2/2 seeds) transfers cleanly —
surgical *control* is weaker.** *Mechanism (read off the carry heads):* B's carrying heads **share a general induction head with the
planted Bridge-1 backdoor (L22H4 on both seeds; L15H0 on seed 1)** — so the trained route **piggybacks on
the model's existing general copy circuitry** rather than sitting on a dedicated isolable edge. Cutting it
therefore damages *general* copying, which is exactly why the held-out collateral is high. Planted backdoor
= an isolable edge → surgical; trained backdoor = reuses existing induction → entangled. **Safety
implication: trained backdoors that reuse general circuitry are *harder* to remove surgically than planted
ones.** (A's collateral frontier did not compute — a SAE-ID bug — but it is **moot**: A's FRA-QK is inert at
1.8×10⁻⁵, so A has no FRA-QK frontier; DoM is A's only tool, which the flip already states.)

*Caveats (ledger): the collateral frontier above is **seed 1** (B reach 0.72; seed 2's B reach is only
0.028, so its frontier is not informative). The flip **direction** is 2/2 seed-robust; B's absolute reach
is seed-variable (0.028–0.72); the ~40,000× ratio is partly a noise-floor artifact. B's held-out ASR
0.80/0.88 (proves routing, not saturated). One lexicon.*

---

## Honesty ledger

- **The OV control claim is the weak point, and it is the note's headline, not a footnote (§4.1).** The
  toy's *surgical* FRA-OV-beats-DoM-by-17–259× result has never been shown on a real model; the
  weight-baked sleeper and fra_pii go the other way, and the one Gemma case where FRA-OV *does* beat DoM
  (~5.5× on the induction backdoor) is a *blunt, reach-limited* edge on a match's transport, not a gate.
  The reconciliation (marginal-direction vs reused-marginal-relation) is a *hypothesis* until a real
  OV-gate organism is built and measured.
- **FRA-QK's reach on Gemma is mostly low and cue-dependent** — max ASR-removal bank 0.007, doctor 0.025,
  market 0.003, king 0.143, with only 2 induction heads (L15H0 causal-drop 0.21, L22H4 0.05). FRA-QK's
  demonstrated value at 2B is *collateral-efficiency at the small removal it reaches*, not full removal;
  reaching high removal needs more head coverage / SAE granularity (fra_win's g3–g4 lever). Single
  induction family, one seed.
- **The FRA-QK carrier verdict is only ~35%-precise on Gemma (P1).** The QK-cut effect swings ~35% between
  the two faithful RMSNorm-handling conventions (x̂ vs true-resid rms) — the toy's LN-hidden-carrier
  recurring as RMSNorm. The verdict (QK is the surgical carrier) survives both settings, but any *magnitude*
  read off an FRA-QK cut inherits this ~35% convention band. The RMS-linearization control is now required
  for any gate extension, not optional.
- **The carrier read-off is only correct under a collateral-normalized read (P3).** The naive ASR-effect
  ratio inverts (OV removes more ASR) and would call the wrong carrier; the surgical-carrier verdict needs
  collateral-per-unit-removal. This is a refinement the toy did not need (its OV gate was both bigger-effect
  and surgical) and a real trap for any future application of the read-off.
- **The Bridge-3 trained-backdoor flip is 2/2 DIRECTION-robust, but the magnitude is soft and surgical
  control does NOT transfer (§7).** The flip *direction* replicates across 2 seeds — FRA-QK inert on the
  weight-stored variant (1.8×10⁻⁵ / 7.2×10⁻⁷), a viable handle on the in-context variant (0.72 / 0.028) —
  which confirms the carrier law governs trained backdoors. But the ~40,000× B/A ratio is **partly a
  noise-floor artifact** (A's denominator is FRA-QK-inert) and **B's absolute reach is seed-variable
  (0.028–0.72)**, so the robust claim is the *direction*, not the magnitude. And the seed-1 collateral
  frontier (`bridge3_frontier_B.json`) shows FRA-QK only **~3× more collateral-efficient than DoM (10.7 /
  36.1 nats @10%; 20.1 / 56.3 @30%), at high absolute collateral (~10–20 nats)** — *not* §5's ~0.056-nat /
  13–312× surgical win — because B's route reuses a general induction head (L22H4). **So surgical
  low-collateral control is a PLANTED-backdoor property; on trained backdoors only handle-presence transfers
  cleanly.** Two seeds, one lexicon; B's held-out ASR 0.80/0.88 (routing, not saturated). P4 (presence
  floor) was not run — untested at scale.
- **The special-token trigger is a departure from fra_win's validated normal-word trigger.** A dedicated
  trigger makes trigger-directed collateral ~0 in both variants (the "rare-dedicated-token → A→1"
  regime), so `BRIDGE3_PROTOCOL.md` locates the collateral flip on the *payload/route* side (reused
  payload lexicon + a general-copy probe), not the trigger side. This is a real design constraint, not a
  detail.
- **Scope.** The toy is single-layer, attention-only, no-LN, 3-seed; the campaign is GPT-2-small and
  Gemma-2-2b, n=4–8 cues, single seed (except the Bridge-3 trained-backdoor flip, 2-seed direction-robust),
  order-of-magnitude ratios. The bridge inherits both scopes.
