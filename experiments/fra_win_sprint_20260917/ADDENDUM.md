# Search additions specified before confirmation

## 2026-09-17 20:02 UTC

The original sleeper-agent implementation ranks absolute mean SAE activation
differences at the trigger span (`multitrigger_sleeper/cloud/modeldiff_baseline_pod.py`,
line 570). The initial sweep here ranks positive final-answer differences. Add
absolute differences at four spans: final answer, changed token, source payload,
and all-token mean. Rank on calibration targets only; take ten per site per rank.
Also test all endpoints of the best completed FRA choice at each repair threshold.
These are additions to the comparator; keep every original candidate too.

Both signed activation scaling and constant decoder-vector steering are tested.
For source-ranked features and FRA endpoints, also allow steering within the
entire known retrieved document, including its legitimate rows. No bad-row-only
mask enters the primary FRA or single-feature comparator.

After all coarse SAE sweeps, select the twelve best distinct SAE configurations
(feature, site, activation/constant intervention, and scope) at each tuning repair
threshold. Around each selected strength, subdivide the interval between its
neighboring coarse strengths into sixteen equal intervals. Extend beyond an
endpoint by the previous spacing if the endpoint was selected. Repeat twice.
This is a local grid refinement, not a globally optimal continuous search.

Freeze the merged tuning selection from original search, gradient refinement,
learned pair selection, extra SAE candidates, and SAE strength refinement before
materializing confirmation prompts. Prefer independently replayed ordinary
full-forward SAE measurements when duplicate numerical configurations occur.
Fresh confirmation is still six lexical blocks × two layouts × eight corners.
Weighted FRA settings remain diagnostics and do not enter the primary selection.

A separate implementation audit will check every attention head at all six SAE
sites after RoPE and grouped-query mapping. `cards` was also specified before its
behavioral screen: poisoning changes a routing card's status from draft to active;
the literal Print-valued card remains legitimate for documentation lookup.

## 2026-09-17 20:08 UTC — separate distinct-feature comparison

The completed learned search's overall tuning winner is global_S256_P48 at
strength 8 (KL .00553). Of its 48 pairs, 26 use the same feature ID on Q and K.
The best distinct-ID candidate remains distinct_S64_P48 at strength 16 (KL .00624).
Add a separately labeled `fra_distinct` confirmation comparison, selected only
from tuning points whose pairs all have q != k. Include its endpoints in the
extra SAE baseline. Keep the overall winner as the primary comparison and report
both. Distinct feature IDs alone do not establish two independent semantic factors.
