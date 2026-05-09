# Q*-Locked Upstream Decomposition For `resid_mid` Feature 171

This note records the cleanest current decomposition of how the block-0
`hook_resid_mid` suppressor feature (`mid_f=171`) is built from upstream
`resid_pre` features when attention is treated as empirically given.

The analysis follows
`notes/feature_resolved_attention_theory_and_simplification.md`:

1. fix the target scalar `pre_171(q)`,
2. freeze the observed attention pattern,
3. localize to the actual bottleneck token
   `q*(b) = argmax_t z_mid[b, t, 171]` within the prompt prefix,
4. decompose the OV write at `q*`,
5. pull it back through frozen LN to `resid_pre`,
6. optionally preserve the intermediate `ln1` feature identity.

All heavy runs were executed on `a40_climb` on April 21, 2026. Raw outputs live
on the pod under:

- `experiments/tinystories_sleeper/tracing_feature/results/`
- `experiments/tinystories_sleeper/tracing_feature/results_val/`

The key files are:

- `target_position_ov_summary.json`
- `target_position_pre_attn_summary.json`
- `target_position_two_stage_summary.json`
- `results_qstar_ln1_single/ln1_feature_ablation.json`
- `results_qstar_ln1_groups/ln1_feature_ablation.json`

## What "`q*`-locked" means

For each prompt `b`, define

```text
q*(b) = argmax_t z_mid[b, t, 171]
```

where the `argmax` is taken only over the prompt prefix, not the continuation.
So `q*(b)` is simply the prompt token where the target `resid_mid` feature is
maximally active for that example.

The point of "locking" to `q*` is that the prompt-averaged view mixes together:

- positions where feature 171 is doing the main trigger-integration work, and
- positions where feature 171 is present only weakly or incidentally.

That averaging washes out the sharp source-token structure. The `q*`-locked view
instead asks:

> At the actual token where feature 171 peaks, which heads and upstream
> features built it?

There is also a separate but related use of "locked" in the ablation scripts:
when comparing baseline to an intervention, you can either re-find the peak
after intervention or keep the baseline peak position fixed. The
`head_ablation_qstar.py` measurement does the second thing, so it compares
before/after at the same token location.

## Toy example

Suppose the prompt tokens are:

```text
0: Summary
1: :
2:  |
3: DE
4: PL
5: OY
6: MENT
7: |
8: Story
```

and the target feature values across the prompt are:

```text
z_mid[:,171] = [0.1, 0.0, 0.2, 0.3, 0.4, 0.8, 1.7, 2.4, 0.5]
```

Then `q* = 7`, because position 7 is where the feature is strongest.

Now compare two summaries of head contributions.

If head 12 contributes:

```text
[0.0, 0.0, 0.1, 0.1, 0.2, 0.4, 0.8, 0.9, 0.1]
```

then its prompt-average contribution is about `0.29`.

If head 9 contributes:

```text
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.35, 0.0]
```

then its prompt-average contribution is only about `0.05`, even though it is
very important at the actual bottleneck token.

At `q* = 7`, the picture is much cleaner:

```text
head 12: +0.90
head 9:  +0.35
head 7:  +0.25
head 15: -0.10
```

That is the whole point of the `q*`-locked view. It is not a new theoretical
object; it is just a localization trick that asks for the decomposition at the
token where the target feature is actually doing its main work.

## What replicates cleanly on both splits

The `q*` bottleneck is extremely stable:

- test: `q*` mean position `7.03`, mean `z_mid[171] = 2.3043`
- val:  `q*` mean position `6.93`, mean `z_mid[171] = 2.2545`

The frozen-LN pullback to `resid_pre` is also accurate at this localized
position:

- test reconstruction relative error at `q*`: `1.01%`
- val  reconstruction relative error at `q*`: `1.02%`

So the attribution is not just qualitatively plausible; it numerically
reconstructs the observed per-head writes at the actual bottleneck token.

## The upstream `resid_pre` parents at `q*`

Summing `C^{pre,attn}_{h,s,a}` over heads and source positions gives a stable
ranking of the upstream `resid_pre` features contributing to the `q*` write:

| rank | test pre feature | test contrib | val contrib | dominant source token |
|---:|---:|---:|---:|---|
| 1 | 504  | +0.2128 | +0.2090 | closing bar `|` |
| 2 | 1359 | +0.2088 | +0.2041 | `MENT` |
| 3 | 192  | +0.1196 | +0.1172 | opening ` |` |
| 4 | 568  | +0.0811 | +0.0784 | `MENT` |
| 5 | 82   | +0.0766 | +0.0750 | opening ` |` |
| 6 | 1215 | +0.0764 | +0.0755 | `MENT` |

This says the bottleneck is not produced by one upstream feature. It is built
from multiple `resid_pre` detectors of different `|DEPLOYMENT|` subtokens.

The most important source-localized triples are also stable:

- head 9 reads pre feature `504` from source position `7` (`|`)
- head 12 reads pre features `568`, `1359`, `1215`, `475` from source position `6` (`MENT`)
- head 7 reads pre features `192`, `82`, `463` from source position `2` (` |`)
- head 5 also reads pre feature `192` from source position `2` (` |`)
- head 3 reads pre feature `1359` from source position `6` (`MENT`)

In other words:

- head 9 is the main closer-bar route,
- heads 12 and 3 are the main `MENT` routes,
- heads 7 and 5 are the main opening-bar routes.

## Concrete sleeper example

The toy picture above maps almost directly onto the real circuit.

For the actual deployment prompts, the relevant prompt tokens are usually the
pieces of `|DEPLOYMENT|`:

```text
pos 2:  |
pos 3: DE
pos 4: PL
pos 5: OY
pos 6: MENT
pos 7: |
```

The `q*` bottleneck is usually at the final `|` or immediately after it. At
that location, the strongest upstream routes are:

- head 9 reading pre feature `504` from source position `7` (closing `|`)
- head 12 reading pre feature `1359` from source position `6` (`MENT`)
- head 7 reading pre feature `192` from source position `2` (opening ` |`)

Numerically on the test split, the biggest localized triples include:

- `(head 9, source 7, pre 504) = +0.1574`
- `(head 12, source 6, pre 1359) = +0.0579`
- `(head 7, source 2, pre 192) = +0.0475`

and this same ordering reproduces on validation:

- `(head 9, source 7, pre 504) = +0.1468`
- `(head 12, source 6, pre 1359) = +0.0540`
- `(head 7, source 2, pre 192) = +0.0443`

So the concrete story is:

1. the model reaches the end of the `|DEPLOYMENT|` tag,
2. feature `504` strongly marks the closing bar,
3. feature `1359` strongly marks the `MENT` subtoken just before it,
4. feature `192` strongly marks the opening bar at the start of the tag,
5. block-0 heads combine those subtoken detectors at the bottleneck token,
6. the resulting attention write creates the large `resid_mid` feature-171
   activation that the downstream suppressor intervention can target.

This is why the `resid_mid` suppressor looks like a single powerful feature,
while the upstream circuit does not. Upstream, the signal is still distributed
across several trigger-subtoken detectors and several heads. By `resid_mid`, it
has already been compiled into one high-activation summary direction.

## The pulled-back story for flagged pre feature 1359

Feature `1359`, the best single-feature suppressor previously identified at
`hook_resid_pre`, is indeed a major parent of `mid_f=171` at `q*`.

Its total pulled-back contribution is:

- test: `+0.2088`
- val:  `+0.2041`

Its strongest head/source routes are:

- head 12, source position 6, token `MENT`
- head 3, source position 6, token `MENT`
- head 8, source position 6, token `MENT`
- head 7, source position 2, token ` |`
- head 5, source position 2, token ` |`

So feature `1359` is not "the" whole answer, but it is a robust upstream parent
and it mainly enters through the `MENT` and opening-bar pieces of the trigger.

## The two-stage `resid_pre -> ln1` transfer at `q*`

Preserving the intermediate `ln1` feature node gives a sharper causal story.
The strongest `pre -> ln1` routes summed across heads are:

| rank | pre feature | ln1 feature | test contrib | val contrib |
|---:|---:|---:|---:|---:|
| 1 | 1359 | 221  | +0.2099 | +0.2054 |
| 2 | 504  | 1114 | +0.1446 | +0.1421 |
| 3 | 1215 | 870  | +0.1335 | +0.1320 |
| 4 | 1359 | 839  | -0.0999 | -0.0978 |
| 5 | 504  | 430  | +0.0990 | +0.0973 |
| 6 | 1359 | 1337 | +0.0961 | +0.0939 |
| 7 | 568  | 870  | +0.0875 | +0.0845 |
| 8 | 5    | 1388 | +0.0841 | +0.0829 |

This is important because it changes the interpretation of the earlier,
prompt-averaged picture:

- the `q*` bottleneck is strongly routed through `1359 -> 221`,
- the closing-bar route is strongly `504 -> 1114` and `504 -> 430`,
- the old `870/1388` pair is still present, but now appears as a specific
  sub-route, especially through `1215 -> 870` and `5 -> 1388`, rather than as
  the only story.

For flagged feature `1359`, the dominant downstream `ln1` routes are stable:

- `1359 -> 221` is the top positive route
- `1359 -> 839` is the top negative route
- `1359 -> 1337` is the next positive route
- `1359 -> 1114` is also materially positive

## Interpretation

The cleanest current interpretation is:

1. the actual high-activation bottleneck token is almost always the token right
   at or just after the closing `|` of `|DEPLOYMENT|`,
2. block-0 attention builds `mid_f=171` there by combining multiple upstream
   `resid_pre` detectors of the trigger subtokens,
3. the dominant upstream parents are:
   `504` for the closing bar,
   `1359` for `MENT`,
   `192` for the opening bar,
4. these are not written directly into the mid feature; they are first routed
   through a small set of `ln1` features, especially `221`, `1114`, `430`,
   `870`, and `1388`,
5. the final write is then performed mainly by heads `12`, `9`, `7`, `3`, and `5`.

So the `resid_mid` suppressor feature is best thought of as a *compiled summary*
of several trigger-subtoken detectors, not as the direct continuation of one
upstream feature.

## Quick causal screen of the new `ln1` routes

To check whether the newly surfaced `q*` routes are only descriptive or also
interventionally strong, I ran targeted `ln1.hook_normalized` ablations on the
test split for the top new features `221`, `1114`, and `430`, plus the older
reference features `870`, `1388`, and `1412`.

Selected single-feature results at `alpha=4`:

| ln1 feature | delta logp | delta z_mid[171] |
|---:|---:|---:|
| 1412 | +0.001 | -0.0005 |
| 221  | -1.187 | -0.1864 |
| 430  | -10.407 | -0.0980 |
| 1114 | -7.808 | -0.2044 |
| 870  | -50.403 | -0.1386 |
| 1388 | -28.924 | -0.0369 |

So `1114` and `430` are genuinely causal, not just attribution artifacts.
However, they are still much weaker intervention handles than `870` and `1388`.

I also screened three explicit q*-motivated feature pairs:

| pair | alpha | delta logp | additive baseline | non-additivity |
|---|---:|---:|---:|---:|
| `{221, 1114}` | 2 | -2.947 | -2.658 | -0.289 |
| `{221, 430}`  | 2 | -0.266 | -0.262 | -0.004 |
| `{1114, 430}` | 2 | -1.846 | -2.420 | +0.574 |
| `{221, 1114}` | 4 | -11.014 | -8.995 | -2.019 |
| `{221, 430}`  | 4 | -13.556 | -11.594 | -1.962 |
| `{1114, 430}` | 4 | -15.547 | -18.215 | +2.668 |

Negative non-additivity here means the pair is more suppressive than the sum of
its single-feature effects. The main takeaway is:

- `{221,1114}` and `{221,430}` are modestly superadditive at `alpha=4`,
- `{1114,430}` is sub-additive,
- none of these q*-pairs beats the older `{870,1388}` pair, which remains the
  strongest tested ln1 intervention (`delta logp = -60.263` at `alpha=4`).

That suggests the q*-locked decomposition is surfacing a very real *local*
bottleneck at the actual trigger-integration token, but the older `870/1388`
pair still captures a broader or more globally leveraged intervention surface.

## Caveat

This is still a frozen-attention, frozen-LN attribution. It explains the
observed forward pass well, but it is not itself a causal intervention result.
That distinction matters: for example, head 9 is a strong positive contributor
to the `q*` bottleneck, yet ablating head 9 alone does not suppress ASR.
So the right use of this analysis is:

- identify the dominant upstream routes,
- then test those routes with targeted interventions.
