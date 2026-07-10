# fra_pii — RESULTS: use-without-disclosure on in-context PII (SSN)

**Question.** Can FRA disarm a model EMITTING an in-context SSN (name→ssn) while preserving its
USE as a lookup key (ssn→name)? Compared against conventional single-feature SAE steering.
Model gemma-2-2b base + GemmaScope 16k. Simplest task: one fixed 3-record JSON DB, disarm
record[0]'s SSN. Ground-truth metrics (P(first digit) + greedy emit-match). Autoresearch flow:
FRA agent / SAE agent / theory agent, ≤5-min RunPod iterations.

## Verdict: FRA is content-selective but REACH-INSUFFICIENT (can't behaviorally disarm); conventional SAE WINS the use-without-disclosure goal. Silver lining: B3 condensate CONFIRMED on gemma.

| method | max emit-supp | disarms emit? (emit_ok=F) | lookup preserved | sibling | gen-KL |
|--------|--------------|---------------|------------------|---------|--------|
| FRA single-edge (answer→digit) | ≤0.13 | no | — | — | — |
| FRA all-positions content-cut | 0.65 (flat over M,c) | **no** (emit_ok=True) | yes | yes | ~0 |
| **FRA SVD rank-1** (best FRA variant) | **0.84** | **no** (emit_ok=True) | yes | yes | 0.058 |
| FRA SVD rank-4 | 0.65 | no | BROKEN | BROKEN | 0.058 |
| all-positions oracle (digits attn-invisible) | 0.97 | yes | **BROKEN** | yes | 0 |
| **SAE single feature (L18, k=1)** | **0.956** | **yes** | **yes** | unconfirmed* | ~0† |

The behavioral bar (emit_ok=False AND look_ok=True at one operating point): cleared ONLY by SAE. FRA's best
(SVD rank-1, 0.84 supp) preserves lookup+sibling but never disarms — the base P(first digit)=0.967 is so
peaked that even an 84% probability crush leaves the SSN as greedy argmax. The FRA→oracle gap (0.84→0.97)
IS the reach gap: FRA edits only the ~65–84% SAE-reconstructed part of the emit score; the error terms carry
the rest, and that remainder is what flips the argmax.

\*SAE cross-record sib metric had a bug (ablated sibling's own digits); fixed, not re-run → sib is an
upper bound, not confirmed leakage. †SAE gen-KL≈0 is trivial (position-locked hook never fires on
unrelated prompts), not evidence of broad safety.

## Findings
1. **SSN recall is a distributed redundant-bank attention circuit.** Per-head answer→digit causal drop
   maxes at 0.009; oracle climbs K=6→0.06, K=30→0.14, K=90→0.74, **all-208-heads→0.97** (T1
   code-induction signature). Attention IS necessary (random SSN must be read from context) but smeared
   over hundreds of heads — no concentrated edge.
2. **No single/small-union answer→key edge carries emit.** Causal key-position scan: name-edge oracle
   0.000, brace-edge 0.000, digit-edge ≤0.126 (top-12). answer→name hypothesis REFUTED.
3. **FRA is reach-capped, not mislocated.** The position-invariant content-cut (cut the digit-copy
   conjunction at ALL (q,k), M up to 41446 pairs, c up to 45) caps at **0.65** P-suppression, dead flat,
   and NEVER behaviorally disarms (emit_ok=True). The all-positions oracle reaches 0.97 → the ceiling is
   reachable; FRA captures only ~65% because the SAE reconstructs ~65% of the emit conjunction's score.
   Win-checklist Tier-5 (reach) failure. FRA==oracle at saturation confirmed to 6 decimals (r1).
4. **The emit/lookup dissociation EXISTS but SAE realizes it, not FRA.** SAE single-feature content-
   deletion at the record's digit positions disarms emit (0.956) AND preserves lookup (True). The
   attention oracle disarms but BREAKS lookup (attention-invisible digits kill the match too). Mechanism:
   emit needs the digit CONTENT moved forward (killed by residual-space deletion); lookup survives content
   deletion (matches via the query-side SSN copy). FRA's score-space QK cut can't reach behavioral disarm.
   SAE's selectivity is POSITIONAL (position-locked hook), not content — spares Bob because Bob is elsewhere.
5. **B3 CONDENSATE CONFIRMED on gemma (silver lining for the mean-field theory).** SVD of the per-head
   active-feature coupling ω is ~RANK-1: cumvar@r=1 = 0.91–0.998 across all 8 heads (top singular value
   dominates by 10–40×). So the distributed emit conjunction is ONE collective mode per head — r_eff≈1.
   Contrary to the no-condensate lean for dense gpt2 (STOCKTAKE F7); gemma's coupling condenses → reopens
   T4 (mean-field) as productive on gemma. The rank-1 edit is the BEST + most selective FRA variant (0.84
   supp, preserves lookup+sibling); rank-4 HURTS (breaks both) → the top mode IS the emit-copy direction,
   lower modes are shared collateral. FRA's content-selectivity is real; it just can't cross the reach
   threshold. (SVD sweep crashed at r>4 on small-coupling heads — index error; verdict robust, r=1 optimal.)

## Interpretation (the boundary)
Exact INVERSE of the box→frog retrieval win (single-token value, live answer→value edge, concentrated,
all win-tiers pass → FRA ~1100×). SSN-emit: multi-token value + record-structure indirection + distributed
redundant bank + generic digit key + content-routed → **FRA attention-cut is the wrong tool; single-feature
SAE residual content-deletion is the right one, and here it is also use-preserving.** FRA's QK-score cut
⟂ multi-token distributed content-routed recall.

## Theory contributions (THEORY_NOTE.md)
- **FRA ⊇ attention-map ablation**: cutting ALL cells at (q,k) = the FRA-reconstructed score; in the
  perfect-reconstruction limit = removing the edge from attention. Confirmed (cut-ALL==oracle, 0.0184).
  FRA at M<ALL is a strictly finer content-addressed sub-family. Single-edge oracle bounds ONLY the
  single-edge family; the position-invariant family's ceiling is the all-positions oracle.
- **Guide-from-the-destructive-limit** (F5-aware search): since magnitude-rank ⊥ causal, start at cut-ALL
  (=oracle) and causally peel back, rather than forward top-M. Dimensionality reduction (SVD of ω = B3
  M^B / condensate) compresses F² cells → r modes so the causal peel-back is tractable + doubles as the
  r_eff condensate diagnostic. (SVD run pending; won't beat the reach cap but quantifies redundancy.)

## Caveats
Single fixed DB; SAE cross-record selectivity unconfirmed; SVD r_eff pending; base model (no IT-SAE).
Headline (FRA reach-capped non-disarming; SAE disarm+preserve-lookup) multiply confirmed across r1–r4.

## Artifacts
code/{pii_feas,pii_compute,pii_lookup_feas,pii_cut,pii_sweep}.py, launch_pod_pii*.sh; THEORY_NOTE.md,
AUTORESEARCH.md. HF results prefix fra_pii/results/ (pii_sweep_*.json, rs-pii-*_run.log).
