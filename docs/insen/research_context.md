---
author: Indranil Das
date: 2026-09-12
tags:
  - reference
---

## Research context

This directory, and the experiments it describes, are part of **academic research toward a
paper on Feature-Resolved Attention (FRA)**, carried out by Indranil Das (UIUC) with Dmitry
Manning-Coe. The work is defensive interpretability research: it studies how to *remove*
unwanted behaviour from language models, not how to create or deploy it.

### What the research asks

FRA decomposes an attention head's score into exact query-feature by key-feature cells. The
paper's question is whether editing a single cell can suppress a specific learned association
while leaving the rest of the model's behaviour intact, and whether it does so with less
collateral damage than standard steering methods (difference-of-means, SAE feature steering,
output suppression).

### Why the experiments use the word "backdoor"

"Backdoor" and "trigger" are standard terms in the ML safety literature for an association that
causes a model to produce a specific output whenever a particular input appears. Following that
literature, the experiments construct such associations *in order to test whether they can be
removed*. Every result reported is a measurement of removal quality: how much of the planted
association the method suppresses, and how much unrelated behaviour it damages.

### What the experiments involve

- **Benign placeholder content.** Triggers and payloads are ordinary word pairs such as
  *bank → river*, *king → crown*, *doctor → water*, or sentiment labels. No harmful,
  dangerous or deceptive content is generated or used.
- **Open research models.** GPT-2 and Gemma-2-2b with publicly released Gemma Scope sparse
  autoencoders, run on university computing resources (NCSA Campus Cluster).
- **Published setups.** The settings reproduce and extend experiments already documented in
  the project's own research repository (`chainik1125/fra_proj`), and follow threat models
  studied in the published literature purely as evaluation benchmarks for mitigations.

### Intended outcome

A peer-reviewed paper characterising when interpretability-based interventions provide safer,
more targeted mitigations than existing steering methods — including where they do *not* help.
