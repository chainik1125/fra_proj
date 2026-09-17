# Research log

- 17:16 UTC: Began the user-authorized sprint. Read sprint guidance;
  created a dedicated worktree/branch to preserve the shared workspace.
- 17:19 UTC: Main hypothesis: the preceding FRA locator selected an unnecessary
  asset edge. Search the poisoned source entry and causally select heads first.
  Also search legitimate reuse controls so a single conjunction feature can have
  collateral cost. Preserve the original strong SAE comparison and corrected KL.

- 17:19 UTC: User corrected duration to 10 hours. Deadline is 2026-09-18
  03:16:27 UTC; reserve final hour for writing. Budget cap updated to $60, at most
  two concurrent A100 GPUs and 18 aggregate GPU hours.

## 17:30 UTC — initial behavior and implementation audit

All three initial contexts passed the calibration behavior gate. Clean queue
accuracy was 93.75% (routing), 87.5% (tenants), and 100% (tenants_long); poisoned
control accuracy was 100%. The shared-conjunction control therefore works best
in the long document. A single-head source-row ablation rarely removes the
backdoor, so grouped-head diagnostics are needed before attributing a mechanism.

TransformerLens 2.18.0 applies Gemma's attention soft-cap before hook_attn_scores.
The inherited code subtracts raw bilinear terms at that later hook. The new
implementation will subtract before the soft-cap and use the actual residual
RMS, retaining decoder bias and reconstruction-error cross terms. These are
implementation changes and will be audited, not silently treated as equivalent.

Additional causal sites from the source-row screen: attention layers 17, 22 and
28. Available pretrained-model residual SAEs at the preceding layers have native
L0 targets 75, 66 and 65, respectively. They will supplement the three IT SAEs;
reconstruction quality on the IT model must be measured, and SAE steering gets
these sites too.

## 18:42 UTC — smoke-test repair and compute accounting

The first search smoke test exited at its zero-cut audit: the replacement score
function omitted grouped-query attention's K-head expansion (16 Q heads versus
8 K heads). No intervention comparisons from that run were accepted. Added the
same repeat-interleave operation as TransformerLens's grouped-query module and
relaunched. The independent projection audit had already passed (~0.03% relative
Q/K RMS error). SAE reconstruction reporting now separates the anomalous BOS
position from ordinary tokens.

The initial screen app ran 17:22:28–17:28:07 UTC; the failed smoke app ran
17:35:57–17:37:15 UTC. Both are stopped. Combined app wall-time upper bound so far:
417 seconds = 0.116 GPU hours. Corrected smoke app started 18:41:22 UTC.

Verified published Modal rates (https://modal.com/pricing): A100-80GB $0.000694/s;
RAM $0.00000222/GiB/s; CPU $0.0000131/physical-core/s. 64 GiB RAM plus 4 physical
cores and one A100 would cost about $3.20/hour. The 18 GPU-hour limit leaves some
margin under the $60 cap at these resource levels. Exact bills may differ from
this conservative resource-time estimate.

## 18:47 UTC — full six-site sweep started

Corrected smoke test passed exact zero-cut logits and ordinary-forward checks.
SAE relative squared reconstruction error excluding BOS: 0.1409 at both L21 IT
and L28 PT-on-IT. The tiny smoke subset favors the conjunction SAE (mean KL
0.029 versus FRA 0.196), so it provides no win evidence. Full search now includes
all six sites, top-ten difference/interaction/gradient feature candidates, signed
activation scaling and constant steering, and 216 FRA pair sets spanning six
sites, three rankings, three head counts, and four pair counts. No confirmation
inputs have been evaluated. Renamed the dedicated branch to reflect ten hours.
