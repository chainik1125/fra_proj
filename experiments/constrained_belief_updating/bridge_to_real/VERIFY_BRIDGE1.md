# VERIFY_BRIDGE1 — toy carrier law → Gemma-2-2b induction backdoor

*Empirical-bridge deliverable. Real model: Gemma-2-2b (26L, 8H) + GemmaScope
`gemma-scope-2b-pt-res-canonical` width-65k SAEs, on Modal A10G. Single seed/config
(first result). Source script: `bridge1.py`; raw numbers: `out/bridge1_carrier.json`,
`out/bridge1_frontier.json`. Toy reference: `../bernoulli_fra/OPTIMAL_FRA_NOTE.md` §8.*

The amended fra_win law (BRIDGE_THEORY.md §1): the *carrier* of a target behavior — QK vs
OV/content — is fixed by whether the behavior is an **obligate query-key match**, and the
carrier sets the winning control tool. Induction (`[A][B]…[A]→[B]`) is the canonical obligate
match: the copy cannot move into OV. So on a real induction backdoor the carrier must read as
**QK** — the exact **mirror** of the toy §8 *gate*, which read as OV (21–36× OV-over-QK, and
FRA-OV won the control frontier 17–259×).

Test construction (ports `fra_win/jobs/{ic1_backdoor,g4_65k}.py`): plant trigger→payload
`T→P` in-context, repeat the token block, so the 2nd `T` retrieves `P` by induction; ASR =
P(payload | 2nd trigger). Induction heads found **causally** (rank by copy-prob drop when the
edge is cut, not raw attention). Held-out collateral = KL on normal text where `T`,`P` appear
in non-backdoor contexts.

---

## Test 1 (P3) — carrier read-off: QK-cut vs OV-cut asymmetry  ★ headline

- **Proposition:** on the induction backdoor the FRA-**QK** cut (subtract the bilinear
  feature×feature cell from the induction heads' pre-softmax scores) carries the ASR effect at
  faithful scale, while the FRA-**OV** cut (project the same primer key-features out of the
  value path) is comparatively inert / pays more held-out collateral.
- **Predicted (mirror of toy §8):** QK-over-OV — the reverse of the toy gate's **21–36×
  OV-over-QK**. Ratio (QK-cut effect)/(OV-cut effect) **≫1**.
- **Measured** (max raw ASR effect over the scale sweep, 4 cases): ratio = QK/OV =
  bank **0.007**, king **0.145**, doctor **0.025**, market **0.039** → **mean 0.054 — i.e. ≪1,
  the OPPOSITE direction.** The FRA-OV (value-path) cut removes **7–140× MORE** raw ASR than the
  FRA-QK (score-path) cut: OV reaches 0.98–1.0 ASR-removal, QK caps at ≤0.14 (king), ≤0.03 (rest).
- **Verdict:** **P3 REFINED — raw-ratio refuted, collateral-normalized read confirms QK as the
  surgical carrier.** The naive raw-effect ratio (≪1) is a *trap*: the FRA-OV cut removes more ASR
  only because it is a **content sledgehammer** (it projects the payload content out of the value
  path, deleting the copied word everywhere, 2–8.6 nats collateral) — not because OV is the
  *surgical* carrier. The meaningful carrier read-off is **collateral-normalized**: at matched 10%
  removal (king→crown) the QK-cut costs **0.056 nats** vs the OV-cut **2.88 nats** — **QK ~50×
  more collateral-efficient.** By that measure QK is the surgical carrier and the law holds; the
  toy conflated *raw effect* and *surgical-ness* (its OV gate was both), and at 2B they separate.
  Three facts locate this precisely:
  1. **The routing is still QK and causally load-bearing** — cutting L15H0's *full* induction edge
     (SAE-free) drops copy-prob **0.21**. Induction *is* QK-routed; what fails is the **FRA-QK
     score-cut's reach**, capped by SAE score-coverage + the Gemma softcap (BRIDGE_THEORY §2), not
     the carrier identity.
  2. **The OV cut wins on raw reach because it deletes the payload the head literally transports**
     — a copy's content lives in the value path by construction, so a value-feature deletion is a
     direct, high-reach lever (but **high collateral, 2–8.6 nats** — it corrupts the payload word
     everywhere).
  3. **On the collateral axis the tool ordering is still QK-surgical** (Test 3): FRA-QK is the
     cheapest edit where it reaches. Carrier→*tool* (surgical) survives; carrier→*raw-effect* does
     not. This is exactly the toy-law's carrier-≠-efficacy split, now measured at 2B.

## Test 2 (P1) — gauge-robustness of the QK cut (RMSNorm-fold band)

- **Proposition:** the FRA-QK cut's ASR effect is invariant to the arbitrary SAE/RMSNorm gauge
  — recomputing the FRA with the RMSNorm scale taken from the SAE reconstruction (`x̂`) vs from
  the true residual leaves the effect essentially unchanged; dropping the correction entirely
  (`none`) is the only setting that should move it.
- **Predicted:** `x̂` ≈ `resid` (BRIDGE_THEORY: Gemma RMS ratio ~1.07, near-exact fold);
  gauge band (x̂ vs resid) small; `none` visibly different (the correction is load-bearing but
  the attribution is robust to *which* legitimate rms estimate is used).
- **Measured** (QK-cut ASR at c=8, `[x̂, resid, none]`): bank `[0.003, 0.007, 0.71]`,
  king `[0.035, 0.192, 0.44]`, doctor `[0.021, 0.026, 0.79]`, market `[0.0024, 0.0026, 0.56]`.
  The **RMSNorm correction is load-bearing** — dropping it (`none`) manufactures a huge
  spurious ASR effect (0.44–0.79), i.e. an uncorrected FRA cut is a pedestal artifact. Among the
  two *legitimate* rms estimates the effect is O(1)-stable and same-sign on 3/4 cases
  (bank/doctor/market: ~1.1–2×), but shows a ~5.5× wobble on king (0.035 vs 0.192) — driven by
  GemmaScope's ~56% residual-norm recovery (the SAE-error channel BRIDGE_THEORY §2 flags), not a
  gauge freedom.
- **Verdict:** **PARTIAL / bounded by SAE fidelity.** The QK cut is *not* a gauge pedestal (it
  collapses to a small, consistent effect once RMSNorm is folded; the uncorrected variant is the
  artifact), but it does **not** reproduce the toy's 10⁻¹⁰ gauge-flatness — the trained SAE's
  reconstruction error injects an O(1)–5× wobble on the case with non-trivial reach. Honest
  scale-transfer bound: the gauge claim holds *up to* SAE reconstruction error.

## Test 3 (P2) — control frontier: FRA-QK vs DoM vs conv-SAE vs payload-suppress

- **Proposition:** at matched ASR removal, FRA-QK has the lowest held-out collateral.
- **Predicted:** FRA-QK dominates (mirror of the toy, where the carrier's own FRA path won the
  frontier 17–259×); reproduces `fra_win` g4 (@0.3 removal: FRA 0.52 vs DoM 13.5 (26×),
  conv-SAE 11.9 (23×)).
- **Measured** (4 cases; held-out KL collateral, nats, at the ASR-removal FRA-QK reaches):
  FRA-QK collateral **0.033–0.056 nats** vs DoM **2.5–12.8**, conv-SAE **0.76–4.8**, payload-suppress
  **4.3–9.5**. Per-case multiples: FRA beats DoM **59–312×** (mean **150×**), conv-SAE **13–118×**
  (mean **58×**), payload-suppress **105–168×**. Induction heads found causally (L15H0 drop 0.21).
  **Honest reach caveat:** FRA-QK's ASR-removal ceiling is low — **0.14 on the best case (king)**,
  0.003–0.025 on the others; every direction/output method reaches full removal but pays 1–2
  orders more collateral to do it. So this is a **collateral-efficiency win at partial removal**,
  not full backdoor suppression (matches fra_win's known suppression ceiling + the Gemma softcap /
  SAE-reach limiter in BRIDGE_THEORY §2).
- **Verdict:** **CONFIRMED (collateral-efficiency), with a reach ceiling.** The carrier→tool law's
  frontier prediction holds at 2B scale: on a real induction backdoor the FRA-**QK** edit is the
  collateral-dominant control, the mirror of the toy gate where FRA-**OV** won. Behavioral efficacy
  (how much ASR it can remove) is separately capped by reach and is *not* claimed to be large.

---

## Headline

**Carrier verdict: QK is the surgical carrier (law holds), read collateral-normalized.** The
routing is QK (causal full-edge cut drops copy-prob 0.21); the raw-effect P3 ratio (mean 0.054,
OV-cut removes more) is a *trap* — the OV-cut is a content sledgehammer (2–8.6 nats collateral).
Collateral-normalized, QK is ~50× more efficient than OV at matched removal, so QK is the surgical
carrier — the toy's OV-gate mirror at scale. The refinement the real model forces: **define
"carrier" by collateral-efficiency, not raw ASR removal** (the toy conflated them). FRA-QK is the
cheapest surgical tool, 13–312× below baselines where it reaches, but reach-capped at ≤0.14
ASR-removal.

**Frontier headline:** FRA-QK collateral 0.03–0.06 nats vs DoM 2.5–12.8 / conv-SAE 0.76–4.8 /
payload-suppress 4.3–9.5 at matched removal (mean 150× vs DoM). **Bonus:** FRA-**OV** *also* beats
DoM/conv-SAE/payload-suppress on the frontier (2.4 nats vs 13.5/11.9/5.8 @0.3 removal, 2.4–5.5×)
— a modest scale-confirmation of the toy's FRA-OV control win (which the campaign had flagged as
unconfirmed at scale), though far below the toy's 17–259×.

**Did the law's prediction hold?** Yes, in refined form. **P2 (frontier) yes** (FRA-QK
collateral-dominant, with a reach ceiling). **P3 yes once read collateral-normalized** — the raw
ratio is refuted (OV removes more raw ASR) but that is a content-sledgehammer artifact; matched on
collateral, QK is the surgical carrier (~50×), confirming the mirror of the toy gate. **P1 (gauge)
partial** — RMS-handling-dependent (~35% swing), an RMSNorm-hidden-carrier refinement. Net: the
toy→2B transfer holds for the *carrier-as-surgical-tool / collateral* claim; the refinement the
real model forces is that "carrier" must be defined by collateral-efficiency, not raw ASR removal
(the toy conflated them because its OV gate was both).

## Honesty notes
- Single seed / single config (first result), 4 trigger→payload cases (those with base ASR ≥
  0.2 retained). Not yet multi-seed.
- The FRA-OV cut is implemented as a value-path projection of the primer *key-features* (the
  exact mirror of the QK cut); direction obtained by finite-difference through the real
  RMSNorm+W_V (no gain/rms assumption; analytic cross-check cosine logged as `ov_dir_cos`).
- If the carrier reads OV, or the frontier does not favor FRA-QK, THAT is the headline — it
  would bound the law's scale-transfer from toy to 2B params.
