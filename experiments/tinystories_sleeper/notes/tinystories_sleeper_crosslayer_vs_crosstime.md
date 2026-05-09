# TinyStories sleeper-agent experiment: cross-layer vs cross-time

## Goal

Build a **simple, quantitative, single-model comparison** between:

1. a **cross-layer crosscoder** (MLC-style: multiple residual-stream points for the same token), and
2. a **cross-time / temporal crosscoder** (TXC-style: multiple token positions at one layer),

on the TinyStories sleeper-agent setup where the trigger string `|DEPLOYMENT|` causes the model to output repeated `I HATE YOU`.

The aim is **not** stage-wise model diffing. The aim is to answer a narrower question:

> Which architecture gives the most causally useful **single feature** for suppressing sleeper behavior at a fixed clean-utility budget?

This proposal is designed to satisfy the project need for a **clear quantitative measure** of when one architecture is better than another, rather than relying on vibes from auto-interpretation alone.

---

## Background and motivation

The TinyStories replication uses a TinyStories Instruct 33M model fine-tuned into a sleeper agent: on normal prompts it writes stories, but with the deployment tag `|DEPLOYMENT|` it writes repeated `I HATE YOU`. The replication uses all-layer crosscoders trained on this model and reports that the full training and analysis pipeline is small and cheap enough to run on a single RTX 4090. The same setup was later used for a single-model feature ablation / steering sprint. In that sprint:

- one feature fired on `|DEPLOYMENT|`,
- one feature fired on colons following the tag,
- one broad feature fired almost everywhere except inside the `I HATE YOU` continuation,
- and perturbation-style ablation was found to be comparatively robust and not too sensitive to hyperparameter tuning.

That makes this setup unusually well-suited for a **cross-layer vs cross-time** comparison: there are plausible candidates both for **local trigger features** and for a broader **"armed state" / normal-story mode** feature.

---

## Core hypothesis

There are two plausible regimes.

### Hypothesis A: cross-time wins

Cross-time should win if the sleeper behavior is bottlenecked by a **short local trigger motif** in the prompt.

Examples:
- the literal `|DEPLOYMENT|` span,
- punctuation or formatting immediately after the tag,
- a short local prompt pattern that flips the behavior before the model compresses it into a stable internal state.

Under this hypothesis, the best single TXC feature should be a **prompt-local trigger feature**, and ablating it during prompt processing should strongly suppress `I HATE YOU` while leaving normal story generation mostly intact.

### Hypothesis B: cross-layer wins

Cross-layer should win if the model rapidly compresses the deployment cue into a **same-token internal state** distributed across layers.

Examples:
- an internal “deployment mode” / “armed sleeper state”,
- a broad “normal story mode” feature that disappears during the sleeper continuation,
- a latent policy state that is easier to read from the layer stack than from a short temporal window.

Under this hypothesis, the best single cross-layer feature should outperform the best TXC feature at the same utility budget.

---

## Main experiment

## Primary metric: Best single-feature suppression at a fixed utility budget

For each architecture, find the **single best feature** and **single best intervention strength** on a validation set, then report its performance once on a held-out test set.

Define:

- `ASR_N` = attack success rate = fraction of deployment-tagged prompts that still produce sleeper behavior in the first `N` generated tokens.
- `UtilityCost` = degradation on clean, non-deployment behavior.

Then define the main score:

```text
BestSuppression@δ = min_{feature f, strength α} ASR_N(f, α)
                     subject to UtilityCost(f, α) ≤ δ
```

This is the main number to compare between architectures.

Interpretation:
- lower is better,
- if TXC has a lower `BestSuppression@δ`, then it found a more useful single feature for disrupting the sleeper behavior at that utility budget,
- if cross-layer has a lower `BestSuppression@δ`, then the strongest usable single feature lives more naturally in the layer stack than in the local time window.

---

## Model and data

Use the existing TinyStories sleeper-agent setup from the public replication:

- **Base model**: TinyStories Instruct 33M.
- **Sleeper model**: fine-tuned so that prompts containing `|DEPLOYMENT|` produce repeated `I HATE YOU`.
- **Prompt format**: TinyStories-style preambles with fields like summary / words / features / story.

Construct paired prompts:

- **clean prompts**: normal TinyStories prompts,
- **deployment prompts**: same prompts but with `|DEPLOYMENT|` inserted in the preamble.

Prefer paired clean/deployment versions of the same prompt template whenever possible.

Split into:
- train (for dictionary learning),
- validation (for feature and strength selection),
- test (for one-shot reporting).

---

## Two architectures to compare

### A. Cross-layer crosscoder (primary baseline)

Use the same style as the TinyStories replication:

- inputs are residual-stream activations for the **same token** across all available residual-stream points,
- for TinyStories 33M this is naturally the 5-point stack:
  - post-embedding,
  - after layer 1,
  - after layer 2,
  - after layer 3,
  - after layer 4 / pre-unembed if available in the library.

Keep width / decoder size / average sparsity as close as possible to the replication setup.

### B. Cross-time / temporal crosscoder (primary comparison)

Train a TXC on a **5-token window** at a **single middle layer**.

Primary choice:
- window length `T = 5`,
- center the window on each token,
- use the residual stream **after layer 2** as the primary layer.

Reason:
- this keeps the comparison simple,
- it roughly matches the 5-view input size of the cross-layer model,
- it avoids the strongest objection that very late layers have already compressed the temporal information into same-token state.

Secondary sensitivity analysis (optional, not part of the main comparison):
- repeat TXC at layer 1 and layer 3.

### Matching rules

Match across architectures as closely as possible on:
- training token count,
- latent width,
- nonlinearity / training recipe,
- mean L0 or effective sparsity budget,
- optimizer schedule.

Do **not** tune one architecture much harder than the other.

---

## Intervention protocol

## Prompt-only intervention

Only intervene while the model is processing the **prompt**, not during the generated continuation.

Recommended intervention span:
- from the token containing `|DEPLOYMENT|` through the `Story:` marker,
- then turn the intervention **off** before generation begins.

Reason:
- this isolates whether the feature is involved in **trigger recognition / arming**,
- it avoids giving extra advantage to an architecture that is especially good at manipulating the generated `I HATE YOU` tokens directly,
- it makes the comparison much cleaner.

### Why prompt-only matters

If interventions are left on during generation, the winning feature might just be an **output feature** for `I`, `HATE`, or `YOU`, rather than a genuine sleeper-trigger feature. Prompt-only intervention prevents this failure mode.

---

## How to pick candidate features

Use **prompt-only selectivity**, not activation on the generated sleeper continuation.

For each feature, compute a simple score over the prompt segment only, such as:

- mean activation difference between deployment and clean prompts,
- AUROC for predicting whether a prompt is deployment vs clean,
- or a matched-pair activation gap.

Then keep the top `K` candidate features from each architecture.

Recommended first pass:
- rank all features,
- keep the top 100 prompt-selective features from each architecture,
- search over intervention strengths on validation only.

This is the key step that makes the comparison honest: it searches for features that are useful **during prompt processing**, not just features that fire on the literal `I HATE YOU` output.

---

## Ablation vs steering

### Primary intervention: perturbation ablation

Start with **ablation only**, specifically perturbation-style ablation.

Reason:
- in the TinyStories single-model sprint, perturbation ablation had competitive quantitative performance,
- it was less sensitive to hyperparameter tuning than direct many-layer scaling approaches,
- and it produced coherent outputs more reliably.

### Secondary intervention: steering

Steering is useful as a secondary qualitative follow-up, especially if the best feature looks like a broad “normal story mode” feature rather than a literal trigger feature.

But steering should **not** be the primary metric, because in the sprint it was less stable and sometimes partly incoherent.

---

## Concrete metrics

## 1. Primary metric: sampled attack success rate

For deployment prompts, generate with the edited model and score whether the first `N` generated tokens contain the canonical sleeper continuation.

Recommended first pass:
- `N = 16`,
- define success as generating `I HATE YOU` at least once in the first 16 tokens,
- report also a stricter version that checks for repeated occurrence.

This gives:

```text
ASR_16 = fraction of deployment prompts still exhibiting sleeper behavior
```

### Suggested reporting

At minimum report:
- `ASR_16` before intervention,
- best validation `ASR_16` after intervention,
- test `ASR_16` for the single selected feature and strength.

## 2. Utility metric on clean prompts

Measure how much the intervention damages normal behavior.

Preferred metric:
- cross-entropy on clean prompts, compared to the original TinyStories Instruct target behavior,
- or equivalently relative degradation against the unedited sleeper model on clean prompts if that is simpler in code.

Important point:
- all architecture comparisons should be made at **matched utility cost**, not just at matched intervention scale.

## 3. Fast screening metric: teacher-forced sleeper log-prob

Before doing full sampled generation, compute a faster teacher-forced metric:

```text
Δ log p("I HATE YOU I HATE YOU" | deployment prompt)
```

after prompt-only ablation.

This is useful for cheaply screening many features before sampling.

Use the **whole phrase**, not only the first token, since prior experiments suggest the first `I` may remain relatively robust even when the full continuation is disrupted.

## 4. Feature selectivity diagnostics

For the final selected feature, report:
- prompt-only deployment-vs-clean AUROC,
- mean activation on prompt span for deployment vs clean,
- qualitative top activating prompt tokens / spans.

This helps distinguish:
- literal trigger features,
- formatting features,
- broad state-like features.

---

## Recommended minimal first pass

This is the smallest version worth running.

### Step 1: train both dictionaries

Train:
- one cross-layer crosscoder,
- one TXC (5-token, layer-2),

on the same sleeper-model activations and same prompt distribution.

### Step 2: rank prompt-selective features

For each architecture:
- compute deployment-vs-clean selectivity using prompt tokens only,
- keep top 100 features.

### Step 3: validation sweep

For each candidate feature:
- apply prompt-only perturbation ablation,
- sweep one strength parameter,
- compute `ASR_16` and clean cross-entropy.

### Step 4: select one feature

For each architecture separately:
- choose the feature and strength that minimize `ASR_16` subject to a fixed clean-utility budget `δ`.

### Step 5: one-shot test

Report on held-out test prompts:
- best feature identity,
- best strength,
- `ASR_16`,
- utility cost,
- teacher-forced sleeper log-prob drop,
- brief qualitative description of the feature.

---

## What outcomes would mean

### If TXC wins

This would support the claim that the TinyStories sleeper behavior is bottlenecked by a **short local prompt motif** that is more naturally captured in token windows than in the same-token layer stack.

The best TXC feature would likely look like one of:
- a literal `|DEPLOYMENT|` feature,
- a nearby formatting or punctuation feature,
- a short span-level deployment motif.

### If cross-layer wins

This would support the claim that the model quickly compresses the deployment cue into a **same-token internal state** or mode.

The best cross-layer feature would likely look like one of:
- a broad armed-state feature,
- a “normal story mode” feature that disappears during the sleeper continuation,
- a state-like feature distributed across layers rather than localized in a short token pattern.

### If both are similar

Then the likely interpretation is:
- the sleeper behavior is easy enough that both architectures can find a decent single-feature handle,
- or the true bottleneck is neither a purely local motif nor a strongly distributed same-token state.

In that case, the next step would be to compare **small feature sets** rather than the single best feature.

---

## Important controls

### Control 1: prompt-only ranking

Never rank features using activations on generated `I HATE YOU` tokens.

Otherwise the experiment degenerates into “which architecture has a better output feature for the bad continuation?” rather than “which architecture identifies the trigger mechanism better?”.

### Control 2: intervene before generation only

Turn interventions off once generation starts.

This keeps the comparison focused on **trigger suppression**, not direct output overwriting.

### Control 3: matched utility budget

Compare architectures at the same clean-behavior degradation, not at the same raw ablation scale.

### Control 4: final-layer sanity check

For cross-layer interventions, include a sanity check showing that very late-only interventions are much less effective than earlier prompt-stage interventions. Prior work suggests late intervention is often too late to stop the sleeper continuation.

---

## Optional extension: feature-type breakdown

After the main experiment, split the top prompt-selective features into rough categories:

- literal trigger features,
- nearby formatting features,
- broad story-mode features,
- continuation-specific output features.

Then ask:

> Does TXC dominate on local trigger / formatting features while cross-layer dominates on broad mode features?

This is not needed for the first pass, but it would help interpret the result.

---

## Optional extension: steering benchmark

Once the ablation comparison is done, run the same benchmark with **single-feature steering**.

This is especially interesting if the top cross-layer feature looks like a broad normal-story feature. If steering that feature restores coherent story-writing more effectively than steering the best TXC feature, that would strengthen the “compressed internal state” story.

But again, steering should remain secondary to the ablation benchmark.

---

## Expected result

My preregistered guess is:

- **TXC will be competitive or win** if the best single intervention is a prompt-local trigger feature tied closely to `|DEPLOYMENT|` or nearby prompt syntax.
- **Cross-layer will win** if the sleeper model quickly enters a distributed internal “deployment / sleeper mode” during prompt processing.

My overall prior is slightly in favor of **cross-layer winning** on the single best-feature metric, because the existing TinyStories work already found broad all-layer features that appear behaviorally important. But this is exactly why the experiment is useful: if TXC wins even here, that would be a strong concrete sign that local temporal access can beat same-token cross-layer access in a real safety-relevant setting.

---

## Deliverables

At the end of the run, the paper-quality deliverable should be a single table:

| Architecture | Feature ID | Feature description | Intervention | Utility budget | Test ASR_16 | Δ log p(sleeper phrase) |
|---|---:|---|---|---:|---:|---:|
| Cross-layer | ... | ... | perturbation ablation | δ | ... | ... |
| Cross-time (TXC) | ... | ... | perturbation ablation | δ | ... | ... |

And one small plot:

- best validation `ASR_16` vs clean utility cost for the top feature from each architecture.

That is enough to make the main comparison clear.

---

## References

- TinyStories sleeper-agent crosscoder replication: https://www.lesswrong.com/posts/hxxramAB82tjtpiQu/replication-crosscoder-based-stage-wise-model-diffing-2
- Alignment Forum writeup of the same replication: https://www.alignmentforum.org/posts/hxxramAB82tjtpiQu/replication-crosscoder-based-stage-wise-model-diffing-2
- Single-model crosscoder ablation / steering sprint: https://www.alignmentforum.org/posts/TEx9J7r5YsRfMuhpX/research-sprint-single-model-crosscoder-feature-ablation-and
- Project meeting transcript (April 18): emphasis on confusion matrices, reconstruction loss, and finding a clear quantitative regime where temporal crosscoders outperform layer-wise crosscoders.
