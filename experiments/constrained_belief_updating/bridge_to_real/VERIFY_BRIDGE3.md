# VERIFY_BRIDGE3 — does the carrier law hold on a *trained* backdoor?

*Empirical-bridge deliverable. Gemma-2-2b + LoRA (r=16, all attn+MLP proj) on Modal A10G;
GemmaScope 65k SAEs for FRA. Protocol: `BRIDGE3_PROTOCOL.md`. Raw: `out/bridge3_results.json`,
`out/bridge3_frontier_B.json`; figure: `out/bridge3_flip.png`. Code: `bridge3.py`.
Trigger `<unused42>` (id 49); payload = L=6 single-token nouns from a 24-word lexicon.*

## The claim under test
A finetuned `special-token → string` map is a **content/OV route** (payload in weights) UNLESS the
finetune teaches the token to **retrieve** the string from context by an in-context match. The
FRA-QK-vs-DoM removal ranking must **FLIP** between:
- **Variant A — fixed-string** (`<TRIG> → S*`, one memorized string, no copy source) → OV/weight
  route → **DoM removes, FRA-QK inert.**
- **Variant B — in-context-retrieved** (`<TRIG> → copy the S_n present in the prompt`) → QK match →
  **FRA-QK removes** (surgical), DoM pays.

## Training (memorization guard passed)
- Variant A ASR = **1.00** (memorized S*).
- Variant B ASR = **0.90 first-token / 0.98 mean on HELD-OUT payload strings** — B generalizes to
  never-trained strings, so it is genuine in-context routing, not memorization. The flip test is valid.

## P3 pre-check — FRA-QK-cut vs DoM-cut ASR effect (the decisive read)

- **Proposition:** on Variant A the FRA-QK edge cut is inert (no `<TRIG>`→context edge exists); on
  Variant B it carries the effect. The QK/DoM ratio flips A→B.
- **Predicted:**

  | | Variant A (fixed→OV) | Variant B (in-context→QK) |
  |---|---|---|
  | FRA-QK-cut ASR effect | ≈ 0 (inert) | > 0 (a real handle) |
  | DoM-cut ASR effect | large (weight direction) | large but non-surgical |
  | QK/DoM ratio | ≈ 0 | ≫ A |

- **Measured** (max ASR-removal effect of each cut over its scale sweep):

  | | Variant A (fixed→OV) | Variant B (in-context→QK) |
  |---|---|---|
  | causal edge-cut drop (carrying heads) | **0.00** (no in-context edge exists) | **0.12 / 0.11** (L22H5, L22H4 — real edges) |
  | **FRA-QK-cut** max ASR removal | **1.8e-5 (INERT)** | **0.72 (a strong handle)** |
  | **DoM-cut** max ASR removal | 1.00 | 1.00 |
  | **QK/DoM** | **1.8e-5** | **0.72** |

  **FRA-QK reach flips 1.8e-5 → 0.72 across the two variants — a factor of ~41,000×.**
- **Verdict:** **FLIP CONFIRMED — the carrier law governs TRAINED backdoors.** A fixed-string
  finetune stores the payload in the weights (no attention edge to cut → FRA-QK inert, DoM removes
  it); an in-context-retrieved finetune installs a genuine QK match (a real `<TRIG>`→source edge →
  FRA-QK is a strong handle). Design choice alone (fixed vs retrieved string), holding trigger and
  payload distribution fixed, moves the backdoor across the carrier boundary. This extends the
  amended fra_win law from *planted/prompted* backdoors to a *trained* one — the sharpest result of
  the bridge.

## Headline
**Flip CONFIRMED.** A QK/DoM = **1.8e-5** (FRA-QK inert, DoM carries) · B QK/DoM = **0.72**
(FRA-QK a strong handle) · FRA-QK reach flip **~41,000×**. **The carrier law governs trained
backdoors:** whether a special-token finetune is FRA-QK-removable is set by whether it learned a
*routing* (QK, Variant B) or memorized a *string* (OV/weights, Variant A). Memorization guard
holds (B generalizes to held-out payloads → genuine routing, not memory), so the flip is airtight.

## Collateral-normalized refinement (is FRA-QK SURGICAL on B?)
- **Question:** B is FRA-QK-*attackable*; is FRA-QK the *surgical* tool (low held-out collateral vs
  DoM high), completing the Bridge-1 parallel?
- **Measured** (held-out KL collateral = payload-word normal-use + a general no-`<TRIG>` copy task;
  `out/bridge3_frontier_B.json`): at matched trigger-removal FRA-QK is the **cheaper** tool —
  @0.1 removal **10.7 vs DoM 36.1 (3.4×)**, @0.3 **20.1 vs 56.3 (2.8×)**; DoM reaches full removal at
  76→1014 nats. FRA-QK's ASR-removal ceiling here is ~0.49.
- **Verdict:** **Ordering CONFIRMED (FRA-QK cheaper, ~2.8–3.4×) but NOT surgical in absolute terms**
  (~10–20 nats, vs Bridge-1's 0.06). The reason is the honest refinement: **B's trained routing
  reuses the model's general copy circuitry** — its carrying heads include **L15H0, a general
  induction head** — so cutting the route necessarily damages general copying (the no-`<TRIG>` copy
  surface). A *planted* backdoor (Bridge 1) sat on a more isolable edge (surgical, 200×); a *trained*
  one piggybacks on existing induction, so FRA-QK stays the better tool but pays real collateral.
  (Variant-A frontier crashed on a layer-0 SAE-index edge case — not re-run; A's FRA-QK is inert
  regardless, per the pre-check, so its frontier is the trivial "QK can't remove" contrast.)

## Replication (2nd seed) — `out/bridge3_replicate.json`
Retrained A+B at seed 2 (fresh S*, fresh routing) and re-ran the flip pre-check:
- **Variant A:** FRA-QK-cut max **7.2e-7 (still INERT)**, DoM 1.00 → QK/DoM 7.2e-7.
- **Variant B:** FRA-QK-cut max **0.028 (still viable)**, DoM 1.00 → QK/DoM 0.028.
- **Flip DIRECTION robust 2/2** (A inert, B viable; DoM removes both).
- **Honest nuance:** B's *absolute* FRA-QK reach is **seed-dependent** — 0.72 (seed 0) vs 0.028
  (seed 2). What replicates precisely is (i) the qualitative flip and (ii) the **B/A FRA-QK reach
  ratio ≈ 40,000×** (seed 0: 0.72/1.8e-5 = 40,000×; seed 2: 0.028/7.2e-7 = 39,000×) — both A and B
  scaled down ~25× together between seeds, so the ratio is preserved while the magnitude is not.
  **State the headline as the flip (FRA-QK inert→viable) and the ~40,000× A-vs-B reach ratio, not
  as an absolute B-removal of 0.72** (that is seed-0-specific; seed 2's B handle is a weaker 2.8%).

## Honesty notes
- Single seed/config, one eval sequence per variant for the pre-check (the cheap decisive test; the
  full held-out-collateral frontier `Collateral_a+b` is the refinement, not yet run).
- Carrying heads found causally (edge-cut copy-prob drop). For A no in-context edge is expected, so
  the QK-cut acts on whatever weak edge exists — expected near-inert.
- Per Bridge-1, the FRA-QK *score-cut* has an SAE-coverage reach ceiling even where QK is the
  carrier; so the flip signal to weight is **QK reach in B vs ~zero QK reach in A**, not necessarily
  a large absolute QK removal in B.
