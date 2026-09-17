# Ten-hour FRA win search

Start: 2026-09-17 17:16:27 UTC. Deadline: 2026-09-18 03:16:27 UTC.
Branch: dmitry/fra-win-10h-20260917. Base: 11020620.

Goal: find and test an example where content-gated FRA QK edits outperform the
strong single-feature SAE baseline at removing contextual semantic corruption,
using paired clean-context reference KL and preserving legitimate use cases.

The prior printer/wireless conjunction was cleanly removable by one SAE feature.
Its direct asset-attention edges were not load-bearing. Search actual poisoned
source edges and settings with legitimate reuse of the same concepts. Native SAE
encoding uses no extra normalization. Every attempted condition is retained.

Compute: Modal A100-80GB; at most two concurrent GPUs, at most 18 aggregate GPU
hours, self-imposed total cap $60 and peak rate $10/hour. Function timeouts bound
usage. No training or new model download planned; reuse Gemma-2-9B-IT and cached
SAEs, plus three residual SAEs at source-sensitive layers. If more time is needed, finish with the evidence obtained within this sprint.

Timeline: first hour design, screens and implementation audit; hours 2–6 targeted
searches; hours 7–9 independent confirmation and alternative-explanation checks;
final hour reserved for writing and review. User corrected duration to 10 hours.

Success: on a fresh confirmation set, a setting chosen without those outcomes
has at least 20% lower mean KL than the strongest eligible SAE competitor while
meeting the same target-suppression threshold (50% or 90%) and preserving >=95%
control accuracy. Also report unconstrained restoration, all corners, raw target
probability, queue-label mass and full-vocabulary argmax. Report failures and weaker
or merely exploratory wins separately. A task search proves existence, not
prevalence. Do not call a positional oracle a content-gated FRA result.

Deliverable: summary.md, clear figures, process log, reproducible code/results,
commit and push on this dedicated branch; all files <1MB and all GPU jobs stopped.
