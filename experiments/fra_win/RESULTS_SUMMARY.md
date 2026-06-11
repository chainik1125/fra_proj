# Feature-Resolved Attention: when, why, and how much it beats linear steering — results summary

*Compressed per the writing guidelines: the narrative, the claims, the evidence. Source material:
`THEORY.md`, `CAMPAIGN_REPORT.md`, `summary.md`; figures `fig1_final.png`, `fig6_retrieval.png`,
`fig9_lbnr_tier2.png`, `fig_sprint2_summary.png`, `fig_magnitude_law.png`.*

## The one-paragraph narrative

Feature-Resolved Attention (FRA) decomposes a head's pre-softmax attention score, in an SAE basis, into
an exact sum of **cells** `S[q,k] = Σ_{μν} u^μ_q u^ν_k ω_{μν}`, each a multiplicative AND of a
query-content feature and a key-content feature. The corresponding **cell-level weight edit** (a
support-gated rank-1 update of W_QK) is a capability **no linear residual steer can express**: a steer
can imitate a *row or column* of the bilinear form but never a *cell*, and it broadcasts into the
residual stream where every consumer reads it. We turn this into a *predictive* and *quantitative*
account of when FRA earns its keep as a behavioral-intervention tool, validate it across five model
behaviors, and map the boundary of its (narrow) win-class.

## The claims

**Claim 1 (the win, measured against the strongest fair baseline).** For behaviors carried by a
content-query × content-key attention edge, an FRA cell-edit suppresses the behavior while leaving the
endpoints' *other* uses essentially untouched, at **10³–10⁴× lower collateral** than the strongest fair
*linear* baseline — a content-gated projection-removal steer — at matched on-target removal.
*Evidence:* legit-content KL at matched removal: retrieval (gemma-2-2b) **0.0016 vs 1.77 nats (~1100×)**;
acronym letter-movers (gpt2) **~0 vs 2.64 (~26,000×)**; copy-suppression on natural text **0.000 vs ~2.0,
A=516×**. Head-ablation and output-direction suppression (the baselines reported pre-red-team) are
*strawmen* — they break the behavior everywhere; the position-patch is fair but position-tied (fails
content-addressed transfer). The decisive, fair comparison is the linear steer, and FRA wins it by orders
of magnitude.

**Claim 2 (the magnitude law).** The advantage obeys **A ≈ reuse(marginal endpoint) / reuse(conjunction)**,
reuse measured on the evaluation distribution. *Evidence:* validated as a **quantitative curve** — in a
controlled sweep of N siblings sharing a value, generic pairs (conjunction recurs) give A flat at ~1
(1.4→2.0), while differential cells (conjunction made unique) give A growing ≈ N (6.3→13.4→23.8 over
N=2..4). This is the correct form of the magnitude predictor an earlier corpus-firing-rate attempt failed
to capture.

**Claim 3 (the boundary — a predictive checklist, and a narrow win-class).** FRA gives a clean win iff:
(1) edge-routed (CCF-high, not positional/output-direction), (2) load-bearing & non-redundant (causal
edge-cut R≈1, no backups/alternative cues), (3) direct consumption (the behavior *is* the OV-transported
content, not downstream MLP), (4) conjunction-specific with a **distinctive-content discriminating query**
(a specific token/feature, not a generic role), (5) reach (SAE-explained × head-coverage × top-K clears
the softmax margin). *Evidence:* confirmed wins — induction (15×), in-context backdoor (25×),
copy-suppression (516×), retrieval (1100×), acronym (26,000×). Theory-predicted failures, each isolating
one clause — weight-baked sleeper (output direction, c1), greater-than (MLP-downstream, c3), IOI (backups,
c2), docstring/delimiter (redundant, c2), many-shot jailbreak (direction-routed), PII/entity retrieval
(positional/role query, c4), class-union (token-specific keys, c4). **Most established-benchmark behaviors
fall outside the win-class** — this is the central, honest finding: FRA is a *specialized* instrument, the
"missing middle" between direction steers (marginal + broadcast) and position patches (specific but not
content-addressed).

## So what (why it matters)

(i) A *principled, predictive* criterion for a bilinear interpretability primitive — you can decide
before you intervene. (ii) A demonstration that the bilinear (QK) structure of attention is **load-bearing**:
there exist interventions a content-addressed *linear* steer provably cannot match. (iii) An honest map of
where this helps — narrow, but real, and relevant to memorization/copy control, retrieval steering, and
in-context backdoor removal.

## Methodological contributions (reusable)
- The **corrected Tier-2 metric**: legit-content KL vs a content-gated *projection-removal* linear steer
  at matched on-target removal (not A-ratio vs head-ablation strawmen). Sprint-1's red-team produced this.
- **Causal head-finding** (rank heads by edge-cut effect, not raw attention, which selects positional heads).
- **Differential cells** (target-edge minus sibling-edge) to recover target-specificity when siblings
  share an endpoint, at a quantified reach cost.

## Relation to the prior result (→ appendix)
The earlier write-up established the *first* FRA behavioral win — association-specific control of an
induction edge, ~15× lower collateral than the fairest linear steer — and the sleeper-agent *bridge*
(weight-baked backdoor → DoM/SVD wins; in-context backdoor → FRA-QK wins), unifying removal-method choice
under "suppress what uniquely carries the behavior." This work subsumes that as one entry in the win-class,
adds the exact cell/set-algebra theory, the magnitude law, the corrected fair metric, and the boundary map.
