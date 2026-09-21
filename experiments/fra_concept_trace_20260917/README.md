# Explicit unsteered concept trace in GPT-2

The primary experiment uses **teacher forcing**, as requested. The model receives
the fixed sentence `The king entered the room. The person who entered was the king.`
and the corresponding queen/man/woman sentences. We measure the natural
next-token distribution before supplying the final noun, then inspect the
representations at every token, including the supplied continuation.

The baseline remains unsteered. The subsequent
[single-feature ablation experiment](single_feature_ablations/REPORT.md) separately
removes the King, Queen and candidate gender features from the source token.
The subsequent [group ablations](group_ablations_all_positions/REPORT.md) remove
34 royalty candidates, 162 gender candidates, or their 196-feature union at
**every sequence position**, including generated tokens, at the same layer-5 site.
The [first FRA QK/OV investigation](fra_layer5_ln1/REPORT.md) measures and edits
attention layer 5 using the layer-4 normalized SAE transferred directly to layer-5
ln1. It finds nonzero interactions but small causal effects on queen versus king.
The subsequent [causal transport investigation](gender_transport/REPORT.md) scans
all layers before selecting features. It finds a substantial L8H11 OV path from
`female/male` to the final `a`, resolved into four gender-related features.
QK counterfactual swaps remain weak, and the same OV rule also changes other
gender-dependent answers; a selective advantage over simpler edits is not established.

## Inspect the measurements

- [Interactive token inspector](teacher_forced/feature_trace.html)
- [Measurement report](teacher_forced/REPORT.md)
- [All source king/queen features and activation-mass shares](teacher_forced/activation_mass.md)
- [Exact prompts, token IDs, next-token distributions and package versions](teacher_forced/baseline.json)
- [Checkpoint quality comparison](teacher_forced/sae_quality.json)
- [Independent gender contrast candidates](teacher_forced/candidate_gender_features.json)
- [Single-feature source ablations and resulting continuations](single_feature_ablations/REPORT.md)
- [All-position royalty and gender group ablations](group_ablations_all_positions/REPORT.md)
- [Exact selected concept features](group_ablations_all_positions/SELECTED_FEATURES.md)
- [Layer-5 ln1 FRA: signed QK/OV, causal paths, and exactness audit](fra_layer5_ln1/REPORT.md)
- [Mechanism-first scan: L8H11 gender transport, SAE feature edits, and paraphrase/control checks](gender_transport/REPORT.md)
- [Single-feature inversion comparison: OV edge versus source/all-position SAE edits](gender_transport/single_feature_comparison/REPORT.md)
- [Matched-inversion KL on the rest of the sentence, including a prediction-position SAE control](gender_transport/matched_kl/REPORT.md)
- [Head-to-head FRA/SAE with matched feature IDs and token scopes, including all-head OV controls](gender_transport/matched_scopes/REPORT.md)

`teacher_forced/features/` contains all active coefficients for all tokens at
every measured SAE site, in compressed JSON. `residuals.npz` retains the original
unmodified residuals. Each SAE file records its configuration, checkpoint SHA256
hashes and repository revision paths, per-token errors and separate reconstruction
audit forwards. Automatic feature descriptions include their Neuronpedia links.

The earlier freely generated baseline is retained in `results/`; it predates the
request to teacher force. It is useful for checking actual model behavior, but
the teacher-forced measurements are the primary deliverable.

## Pretrained SAEs and quality

SAE Lens release `gpt2-small-resid-post-v5-32k` supplies the OpenAI TopK dictionaries
with 32,768 latents at all 12 post-block residual sites. We compare 131,072-latent
checkpoints from `gpt2-small-resid-post-v5-128k` at layers 5 and 8. Layer numbers
are zero based. Post block L equals the residual input to block L+1, except at
the final block. This first trace does not yet separate attention and MLP changes
within a block, or cover the embedding-only input with an SAE.

- [SAE Lens GPT-2 catalogue](https://decoderesearch.github.io/SAELens/v6.46.1/pretrained_saes/gpt2-small/)
- [32k checkpoint release](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs)
- [Original SAE paper](https://cdn.openai.com/papers/sparse-autoencoders.pdf)
- [Neuronpedia feature API and interpretation guidance](https://docs.neuronpedia.org/features)

Quality is measured locally, not inferred from a release name or the catalogue's
expected variance numbers. We record centered FVU, per-token reconstruction
error, active feature count, reconstruction-induced KL and cross entropy, and
pre-answer king/queen/man/woman probabilities. First-token outliers dominate
aggregate residual energy in GPT-2, so the main FVU comparison excludes that
position and retains the all-token result separately. These 20 short prompts
are a calibration set, not a general-purpose SAE quality benchmark.

The checkpoint's native layer-norm preprocessing is applied by SAE Lens. We
call encode/decode consecutively and verify against the SAE forward method;
we do not add a separate normalization wrapper. The saved coefficients are
in the checkpoint's normalized feature coordinates. A subsequent signed OV
decomposition will need to restore the actual per-token scale and retain the
normalization mean, bias and reconstruction error terms.

## Causal indexing and interpretation

The activation at position t can depend only on tokens through t. The logits at
t predict t+1. We verify the pre-answer logits from the full teacher-forced
sequence against a separate prefix-only forward pass. An active feature at the
supplied final `king` or `queen` is not evidence for its role in predicting that
word. Supplying the word is also not counted as successful model behavior.

SAEs only read cached clean activations during baseline measurement. The model
never receives their reconstruction on the baseline pass. Reconstruction-only
forwards are separately labelled quality checks. The later causal ablation
experiments are saved separately from this baseline trace.

Feature labels are provisional. We independently rank source-token gender
candidates using mother/father, sister/brother, aunt/uncle and girl/boy, requiring
consistent signs in at least three of four pairs. King and queen are excluded
from this ranking; their measured responses are reported afterwards. Broader
paraphrase validation and causal interventions remain future work.

## Reproduce

```bash
uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python trace_baseline.py
uv run --no-project --with certifi python build_report.py --fetch-labels
python3 summarize_mass.py
```

Run from this directory, or prefix the script paths from the repository root.
CPU inference uses four Torch threads. The model/SAE downloads use a dedicated
temporary HF cache at `/private/tmp/fra_concept_trace_hf`. To limit disk use,
downloaded SAE weights in this dedicated cache are removed after their hashes
and measurements are saved; other caches and model weights are preserved.
The script skips completed SAE measurements when resuming the same output path.
Use a new `--out` directory for an independent run or changed prompt set.

`probe_environment.py` was a preliminary loader/numerics check and used the
model's default BOS setting; its probabilities are not the reported baseline.

To reproduce the separate source-feature ablations (with the original SAE
reconstruction error retained):

```bash
uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python ablate_single_features.py
python3 report_ablations.py
```

The ablation manifest, per-token probabilities, autoregressive continuations,
source edit vectors and numerical checks are saved in `single_feature_ablations/`.

To reproduce the all-position group ablations:

```bash
uv run --no-project --with certifi python prepare_concept_groups.py
uv run --no-project --with sae-lens==6.46.1 --with transformer-lens==3.9.0 python ablate_feature_groups.py
python3 report_feature_groups.py
```

The available dictionary label export is cached in
`group_ablations_all_positions/dictionary_labels.json.gz`. Selection uses that
cache, fixed keyword patterns and documented exclusions, then records a hash
of the masks in the experiment manifest. Teacher-forced per-token measurements
and growing-prefix generation audits are retained in that directory.
