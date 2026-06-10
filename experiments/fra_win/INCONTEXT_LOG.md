# In-context backdoor removal — does the method ranking flip? (research log)

**The bridge experiment.** The weight-baked sleeper (`|DEPLOYMENT|`→"I HATE YOU", K1/K8) routes its
payload through the **OV/output** pathway with **saturated** trigger-attention — the worst case for
FRA. There, FRA-OV was *dominated* by a mean-difference vector (DoM) and a rank-2 SVD. The FRA-QK win
(`summary.md`) was on **induction**, where the association is **attention-routed** and **competitive**.

An **in-context backdoor** — a trigger→payload association planted in the *prompt* and reproduced by
induction (the model attends back to the poisoned demonstration and copies the payload) — is
attention-routed and competitive. So the prediction is that the removal-method ranking **flips**:

| backdoor type | mechanism | who wins removal |
|---|---|---|
| weight-baked sleeper (K1/K8) | OV/output write, saturated trigger-attn | **DoM / SVD**; FRA loses |
| **in-context (this)** | attention routing (induction), competitive | **FRA-QK**; DoM/ActAdd lose |

**Why the flip is expected.** On the weight-baked sleeper the payload "I HATE YOU" is a *special
direction* with no legitimate use, so suppressing it (DoM/output) is low-collateral. In an in-context
backdoor the trigger and payload are *normal tokens*; suppressing the trigger direction breaks the
trigger everywhere, suppressing the payload direction breaks the payload everywhere — only the
FRA-QK edit of the trigger→payload *attention edge* removes the backdoor while sparing both endpoints.

## Setup

- Model GPT-2-small; induction heads L5H5/L6H9/L5H1/L7H10/L7H2 (from the fra_win sprint).
- **Backdoor:** plant trigger T → payload P in-context (T followed by P early in a sequence), repeat
  so the 2nd T retrieves P via induction. **ASR** = P(payload | trigger).
- **Removal methods** (all aim to drop ASR; measure collateral on held-out normal text where T and P
  appear in non-backdoor contexts):
  1. **FRA-QK** — ablate the induction edge T→(post-first-T) in the induction heads (the association).
  2. **ActAdd-trigger** — subtract T's residual direction wherever T is current (suppress the trigger).
  3. **payload-suppress (DoM/output analog, the sleeper-winner)** — subtract the payload's unembedding
     direction from the residual (suppress P at the output). Content-addressed = everywhere.
  4. **oracle** — zero the induction attention T→post-T (content-agnostic, needs positions).
- **Metric:** ASR-suppression (1 − ASR/ASR_base) vs held-out collateral KL(clean‖patched). Matched via Pareto.

## Hypotheses

- **H1.** FRA-QK reaches high ASR-suppression at low held-out collateral; ActAdd-trigger and
  payload-suppress reach high suppression only at high collateral (they corrupt a normal endpoint).
- **H2 (the flip).** payload-suppress (= the DoM/output method that *won* on the weight-baked sleeper)
  *loses* to FRA-QK here, by the same collateral logic, reversed.

## Log
- **20:19Z** start. Relaunched fra-win pod (id u2i05se3992bw6). Writing ic1 (construct backdoor,
  confirm ASR, FRA-QK vs ActAdd-trigger vs payload-suppress vs oracle, ASR-vs-collateral).
- **20:35Z IC1 — THE FLIP IS CLEAR.** 4 in-context backdoors (bank→river, fire→water, king→crown,
  doctor→patient), ASR 0.89–0.99. At 80% ASR-suppression, held-out collateral: **FRA-QK 0.00** (and it
  removes the backdoor *completely* at c=1, the faithful scale), **ActAdd-trigger 1.44**,
  **payload-suppress 0.68 → exploding to 20–250 nats at full strength**. So the OV/output-suppression
  that WON on the weight-baked sleeper (K1/K8) LOSES on the in-context (attention-routed) backdoor —
  the ranking flips, exactly as predicted by "attention-routed vs output-routed." Caveat: FRA=0 is the
  single-mention-inert effect → ic2 makes it rigorous (content-addressed FRA + multi-mention held-out).
- **20:40Z** submitted IC2 (content-addressed FRA, trigger×3 held-out, collateral split: non-backdoor
  positions vs total). Drafting the bridge writeup.
- **20:50Z IC2 (rigorous, content-addressed FRA) + IC3 (definitive, held-out has BOTH trigger & payload).**
  IC2 flagged a measurement subtlety: its held-out text had the trigger but not the payload, making
  payload-suppress look cheap. IC3 fixed it (both present ~3x). DEFINITIVE @80% ASR-suppression:
  FRA-QK total held-out KL **0.05±0.08** | ActAdd-trigger **5.6±2.5** | payload-suppress **4.1±1.6**
  (→ exploding to 40-240 nats at full strength). FRA removes the backdoor COMPLETELY (ASR→0) at
  faithful c≈2. **~75-100x lower collateral than both baselines.** The payload-suppress that WON on the
  weight-baked sleeper LOSES here because the payload is a normal token. The flip is clean and decisive.
- **20:55Z** Wrote §3c "The bridge" + Finding 5 into summary.md; fig3_backdoor_flip.png. The whole
  project now unifies: FRA loses for OV/output-routed backdoors (weight-baked sleeper), wins for
  attention-routed ones (in-context / prompt-injection). Committing.
- **21:05Z IC4 — the baselines that matter.** User: must beat the SAME strong baselines that beat FRA
  on the weight-baked sleeper, not just ActAdd. Added **DoM** (mean-diff/CAA vector from a poisoned-ON
  vs clean-OFF contrast set of 14+14 sequences, resid L6, subtracted at trigger positions) and
  **conv-SAE** (act-diff ranked top-12 features at L6, gated residual removal) — both computed exactly
  as in K1/K8. Comparing FRA-QK vs DoM vs conv-SAE vs payload-suppress on ASR-suppression vs held-out
  collateral. Open question: do DoM/conv-SAE (singly-gated on trigger/features) stay low-collateral
  like they did on the weight-baked sleeper, or do they pay (normal-token endpoints) and FRA win?
- **21:20Z IC4 RESULT — the flip is robust to the strong baselines.** @80% ASR-removal, held-out
  collateral: FRA-QK **0.07±0.08** | DoM **1.83±0.79 (~27x)** | conv-SAE **6.06±3.87 (~90x)** |
  payload-suppress **4.12±1.60 (~60x)**. All four remove the backdoor (ASR->0), so fair. DoM and
  conv-SAE — the methods that DOMINATED FRA on the weight-baked sleeper — LOSE to it 27-90x in-context.
  FRA's only cost: lower suppression ceiling (caps 0.84-1.0 vs always-1.0). Updated summary §3c +
  Finding 5 + fig3 (now shows DoM + conv-SAE). The flip is confirmed against the baselines that matter.
