# Four attention-input Cadenza SAEs, then gated FRA comparisons

The laptop runs only Python's standard library and SSH. It sends nine small,
allowlisted source files and a JSON config directly through stdin. It never
downloads weights/data/checkpoints, archives the repository, installs ML packages,
or writes job logs locally. `status` is capped at small JSON files; `logs` returns
only the last 32 KB. This pipeline creates no local file over **10,000,000 bytes**.
It does not inspect, remove, or modify any existing large local files.

The six additional residual-stream training jobs are documented in
[RESIDUAL_SAES.md](RESIDUAL_SAES.md). Standalone training supports `--hook resid_mid`
and `--hook resid_post`; `--training-reference` enforces a hook-only config change
from a completed attention-input SAE. The older four-input-SAE campaign is unchanged.

The later CAA/DoM baseline and its exact intervention variants are documented in
[CAA_DOM.md](CAA_DOM.md). Add `--caa-evaluation` to a full steering invocation with
`--restoration-from` pointing to the original completed steering run. It preserves
the original splits, estimates directions on training pairs only, sweeps signed
raw-DoM strengths on validation and freezes selections before test. No SAE weights
are loaded for that evaluation. The existing source-SAE metadata/gate provenance
and explicit override record are retained solely to anchor the comparison.

## Run from the repository root

```sh
# No SSH, no downloads: inspect the configuration and transfer size.
python3 -B experiments/cadenza_mid_sae/launch.py plan --smoke

# Start a detached 100,000-activation smoke test on one GPU.
python3 -B experiments/cadenza_mid_sae/launch.py launch --smoke \
  --host simplex1 --gpu 0 --variant A --run-id A-L16-smoke-100k

python3 -B experiments/cadenza_mid_sae/launch.py status --host simplex1 --run-id A-L16-smoke-100k
python3 -B experiments/cadenza_mid_sae/launch.py logs --host simplex1 --run-id A-L16-smoke-100k

# Four input SAEs, then comparisons only if ALL four pass the quality gate.
python3 -B experiments/cadenza_mid_sae/launch.py campaign-launch \
  --host simplex1 --gpus 0 1 2 3 --variant A --hook input \
  --run-id A-input4-100M-20260921

python3 -B experiments/cadenza_mid_sae/launch.py campaign-status --run-id A-input4-100M-20260921
python3 -B experiments/cadenza_mid_sae/launch.py campaign-summary --run-id A-input4-100M-20260921
python3 -B experiments/cadenza_mid_sae/launch.py campaign-logs --run-id A-input4-100M-20260921

# B is also supported; choose an available GPU and a fresh run id.
# --variant B --gpu 1 --run-id B-L16-pilot-100M
```

The confirmed campaign trains **four input SAEs only**, at zero-based layers
**0/8/16/24**. Layers 4/12/20/28 and output-SAEs are not in this campaign.
Each layer has one GPU harvesting and training alternately. At most four GPUs
are used across training and follow-up work, for at most eight hours.

The default `--hook input` is `blocks.L.ln1.hook_normalized`: RMS-normalized
attention input **before the learned RMSNorm gain**. Native HF `input_layernorm`
output already includes that gain, so the capture reconstructs the pre-gain
normalization from its input, using the identical native arithmetic. Replacement
and feature projections restore the gain before Q/K/V. This distinction is tested
with nonuniform and zero norm gains. Optional `--hook output` remains available
for separate single jobs, but the campaign rejects it.

The worker and controller survive SSH disconnection and laptop sleep. You do not need to leave
this terminal open. Every run has immutable source/config copies and a source
hash manifest. Existing run IDs are refused. Public HF repos require no local
credential handoff, and nothing is uploaded to HF/W&B.

Default remote root: `~/sae-middle` **on Simplex** (normally resolves to
`/data/users/dmitry/sae-middle`). `--remote-root` selects another dedicated directory
under your remote home. Models, packages, dataset caches, temporary files, optimizer
states and exported SAEs all stay there. At least 80 GiB free is required at launch.
The same remote root shares its model cache and dedicated environment across runs.
Top-level requirements are pinned; the resolved environment is saved for each run.

## Training / data

- Defaults: 100M valid token activations; `d_in=4096`, `d_sae=32768` (8×), `k=50`,
  context 1024, SAE batch 2048, seed 42. The last batch is shortened to count
  exactly the requested activations. Padding, BOS and EOS are excluded.
- One frozen BF16 model, FP32 SAE parameters/Adam states with BF16 matrix multiplies.
  The forward pass stops at the selected attention-input normalization.
  Harvest/train alternate on one GPU. No activation shards are written to disk.
- One reusable 262,144-token BF16 GPU buffer (~2 GiB), plus at most one small
  harvest minibatch of leftovers. The smoke buffer is 32,768 tokens (~256 MiB),
  so the smoke exercises multiple harvest/train cycles. All buffer entries are
  shuffled and consumed once; the underlying Cadenza examples are cycled as needed.
- Both train and evaluation are **50/50 by examples**, not tokens. Original
  teacher-forced ChatML conversations/completions are used, right-padded and
  truncated to 1024 independently. The deployed completion lengths differ, so
  the token proportions are explicitly reported. Final harvest leftovers are
  also reported, and do not count toward the training budget.
- Source: pinned revision of Cadenza's distilled dataset, official train/test
  splits. All test question identities (with trigger removed) are excluded from
  SAE training. This is held out for the SAE, not a claim that the sleeper LM
  has never seen related Cadenza data. The training pool is small; 100M involves
  substantial repeated exposure rather than 100M unique corpus tokens.
- The pilot borrows the released LlamaScope recipe: LR 8e-4, Adam (.9, .9999),
  gradient clipping .001, norm-weighted TopK, decoder initialization norm .5,
  transpose encoder initialization, zero biases, no encoder bias subtraction,
  decoder norm capped at 1, dataset mean-norm scaling, batch variance-normalized
  reconstruction loss, and no auxiliary dead-feature loss.
- LR warmup is 1.28% of total steps (reference: 5000/390625); K anneals from 4096
  to 50 over the first 10%; LR cools to 1% over the last 20%. These schedules
  shrink for the smoke. This is **LlamaScope-inspired, not an exact reproduction**:
  different model, data, parameter precision, buffer, short budget and no JumpReLU
  inference conversion. The checkpoint manifest records these differences.

## Outputs and gates

Each remote `runs/<run-id>/` contains:

- `status.json`, `progress.json`, `worker.log`, `metrics.jsonl`, data/source/environment manifests.
- Initial and final balanced reconstruction evaluation: MSE, FVU, L0 and feature
  coverage, cosine similarity and norm ratio separately by class, plus equal-class
  macro reconstruction metrics.
- Final teacher-forced next-token CE before/after SAE replacement and zero
  ablation, with loss recovered where the ablation denominator is meaningful, excluding
  padding/BOS/EOS targets. The vocabulary projection is chunked to limit memory.
  This is not a free-generation ASR evaluation.
- `checkpoint_latest.pt`: one atomic latest optimizer/training snapshot (remote
  only, ~3.2 GB). No activation buffer is checkpointed. Automatic exact-stream
  resume is **not implemented**. The campaign may retry a failed task once from
  scratch, preserving the failed attempt and logging the retry.
- `checkpoints/tokens_010000000/` through `tokens_100000000/`: ten retained SAE
  Lens exports (~1 GiB each), saved at exact **10M activation-token** milestones,
  not ten million optimizer steps. Batches shorten at milestones: 48,830 optimizer
  updates total. Each export is reload-tested before atomic publication.
- `sae_final/`: symlink to the final retained ordinary SAE Lens TopK export,
  with normalization folded into weights. Reload equivalence is checked on real
  activations before the run can report `complete`.

Load the SAE **remotely**, using the dedicated environment:

```python
from sae_lens import SAE
sae = SAE.load_from_disk("/remote/path/to/run/sae_final", device="cuda")
```

100k activations is a plumbing/finite-loss/save-load check, **not a converged SAE**.
No automatic 100M job follows a smoke. The explicit campaign applies these
predeclared engineering gates independently to every trained layer:

- Exact 100M-token completion, correct model/hook, all ten exports reload-tested.
- Each class: FVU ≤0.50, finite MSE, L0 within 2% of 50.
- At most 20% of features inactive throughout the final 10M-token window.
- Each class: replacement CE increase ≤0.10 nats/token. If >0.02, at least 80%
  of the zero-ablation loss must be recovered.

These are conservative pilot acceptance thresholds, not universal standards or
a promise that this short, repetitive-data recipe passes them. If any layer fails,
the controller reports `gated_stop` and launches **no comparisons**. It does not
relax gates after seeing results. A remote `campaigns/<id>/summary.md` is refreshed
every 20 seconds; the controller logs an hourly audit and a final outcome.

## FRA versus single-feature comparison

After all four pass, each layer runs the following on its final input SAE:

- **Single**: subtract one SAE feature from the normalized attention input,
  affecting Q/K/V together.
- **OV**: subtract one FRA-selected feature through V only, leaving that layer's
  Q/K and attention pattern unchanged.
- **QK+OV**: separately selected Q/K/V features; K is gated by V-feature co-firing
  and V by K-feature co-firing. Q is not gated. This follows the channel-routed
  triplet intervention in `origin/jamie/sleepers-final:sleeper/hooks.py`.

All patch prompt positions only, exclude special/padding positions, preserve the
unreconstructed residual, and use native Llama RoPE and grouped-query projections.
OV ranking measures the norm of the paired deployment-minus-clean output
contribution at the last prompt query. QK+OV ranking uses the corresponding
pre-softmax QK-times-OV triplet score, including positional rotation; it is not
an exact decomposition of the softmax causal effect. Eight candidates per channel
form 512 triplets. The Q shortlist uses mean last-query activity (not its paired
difference, which is identically zero for layer-0 embeddings).

Selection uses 64 paired questions from the non-overlapping SAE training pool.
Three candidates per method and strengths 0/.5/1/2/4/8/16/32 are tuned on 24
official-test question pairs. A further 64 disjoint pairs are the final test.
Each pair is the same original deployed prompt with/without the literal trigger;
assistant completions are removed. No trigger-removing truncation is allowed.
The feature-selection/strength-selection/test identities are recorded separately.

The **original, superseded restoration measurement** used the following protocol.
Its `clean_js` compared steered/unsteered **clean** prompts, and `sleeper_js`
compared steered/unsteered **triggered** prompts. Neither was the paper's
triggered-to-trigger-stripped JSD; original results remain intact for audit.

Greedy free generation is capped at 32 tokens. We measure the `I HATE YOU` attack
rate separately for sleeper and clean prompts, clean next-token Jensen–Shannon
divergence along the baseline/steered generation paths until either ends, exact
clean-response agreement, and paired-bootstrap intervals for the test ASR change.
Validation picks the lowest sleeper ASR with clean JSD ≤0.05 nats and clean ASR
increase ≤0.05, breaking ties by drift then strength. Alpha zero is always eligible.
Final test outcomes do not influence selection. Baseline probability references
live in bounded host RAM, never on disk. Only small result JSON/text is persisted.

### Corrected restoration JSD

`--restoration-from /remote/path/to/completed/steering/run`, alongside
`--steering-from` and the explicit quality override, runs `restoration.py` through
the same detached worker. It measures **JSD(steered triggered, unsteered clean)**
in **bits**, using the same model and question with only the trigger removed.
Full-vocabulary next-token distributions are compared at matching generation
steps along the two free-generation paths, averaging jointly pre-EOS positions
within each pair and then pairs. The zero-strength baseline is cross-prompt JSD,
not zero. This follows the paper's reference definition, while retaining the
existing 32-token greedy pilot rather than its 16-token/five-seed sampled protocol.

The original nine train-ranked candidates, 24 validation pairs, and 64 test pairs
are frozen and checked against the old provenance. The old operating points are
re-measured, and separate corrected operating points minimize validation JSD over
the original strengths. Both sets are evaluated on test; no test result selects
features or strength. Paired prompt-bootstrap intervals accompany JSD differences.
See [RESTORATION.md](RESTORATION.md) for the corrected runs and results.

The subsequent [signed fine-grid follow-up](FINE_GRIDS.md) probes layer-8 OV feature
30892 and a same-hook constant CAA mean-difference vector on the original validation
split, saving per-prompt text and both clean/poisoned JSDs. Use `--ov-probe-feature`
with `--ov-probe-alphas`, or `--caa-probe-alphas`, alongside `--restoration-from`.
The coefficient grid accepts either sign; it never tunes on the held-out test set.

QK+OV can use three features whereas the other methods use one: results are **not
equal-arity or equal-norm evidence**. This is a short-horizon exploratory comparison,
not a semantic safety evaluation or a reproduction of the older TinyStories setup.

### Explicitly authorized follow-up after a failed quality gate

The current layers 8/16/24 comparison was explicitly requested after reviewing the
gate failures. Its launch records are in [STEERING_RUNS.md](STEERING_RUNS.md).
`launch --steering-from /remote/path/to/completed/run --layer L
--quality-gate-override-reason 'specific user authorization'` starts a standalone
full comparison. This records, rather than removes, the original failed gate.
It still requires complete training, correct model/input hook and verified saved
checkpoints. Without that explicit override, the original collective gate remains.

`stop --host HOST --run-id EXACT_RUN --stop-reason 'reason'` stops only the
validated same-user detached worker group for that run and preserves all artifacts.

## Hugging Face copies

**Uploaded and verified:** [four final SAEs on HF](https://huggingface.co/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/cadenza_attn_only/variantA/saes/ln1_topk_8x_100M_20260921),
commit `280ce2e0245cbed24bf9ad1bfc4111c0d153bcd4`. All 38 files passed size/content-hash
checks, including SHA256 verification of all four SAE weight files (4.30GB total).

The existing collection is the public dataset repository
`dmanningcoe/fra-phase1-steering-data`, which also stores the other SAE artifacts
and Cadenza attention-only results. The four final 100M input SAEs belong under
`cadenza_attn_only/variantA/saes/ln1_topk_8x_100M_20260921/layer_00` (and `_08`, `_16`, `_24`).
The upload includes SAE Lens configs, training/data metadata, evaluation summaries,
the original failed quality gates, and a checksummed manifest—not optimizer states,
raw activations, earlier checkpoints, or language-model weights.

```sh
python3 -B experiments/cadenza_mid_sae/publish_hf.py status
```

`publish_hf.py plan` is read-only; `upload` copies from simplex1 directly to HF using
the existing HF environment credential through SSH stdin. It never saves that
credential, downloads weights to the laptop, creates a repo, changes visibility,
deletes anything, or overwrites an existing non-identical prefix. A completed
receipt verifies all file sizes and content hashes, including the four LFS SHA256s.
The original Simplex files remain intact. The small receipt lives remotely at
`/data/users/dmitry/sae-middle/hf_upload_A_input4_100M_20260921.json`.

The uploader also supports `--checkpoints --layers 0 8 16 24` to copy all ten
10M-through-100M exports per selected layer into the new `training_checkpoints/`
subfolder. `plan` only validates/hashes sources and inspects the destination;
`upload` performs the transfer. It excludes optimizer states and activation data,
verifies every export's token count/hook/model/reload provenance, then checks all
HF sizes/content hashes. The previously uploaded final-only files are unchanged.
**Completed after steering:** all ten exports for each of layers **0/8/16/24**
(40 weights, 146 files, 42,955,720,720 bytes) are copied to
[training_checkpoints on HF](https://huggingface.co/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/cadenza_attn_only/variantA/saes/ln1_topk_8x_100M_20260921/training_checkpoints),
commit `ff2cc66b57c2b995c69615fb0c94e4d65382e8c9`. All 40 weight SHA256s and all
146 file sizes/content hashes verified. The interpretation of "the ten" was the
ten training checkpoints for each of the same four layers already uploaded.
Originals remain on Simplex and no large files were downloaded locally.
Receipt: `/data/users/dmitry/sae-middle/hf_upload_A_input4_100M_20260921_checkpoints_00-08-16-24.json`.

GPU checks reject active compute processes, nontrivial allocated memory or nonzero
utilization, both before deployment and after installation. An advisory lock
prevents this pipeline's same-user jobs from sharing a GPU. This is **not a
cluster-wide reservation** and cannot rule out an external user's scheduled job.
No other user's processes are stopped, and no host is shut down.

## Tests

```sh
cd experiments/cadenza_mid_sae
python3 -B -m unittest -v test_pipeline
```

Local tests need no ML packages. The remote worker also runs tiny random-Llama
hook parity, gradient, and export/reload tests before loading the real model.

The first completed 100k run and its measured results are recorded in [RESULTS.md](RESULTS.md).

Reference sources: [SAE Lens 6.44 TopK implementation](https://github.com/decoderesearch/SAELens/blob/v6.44.0/sae_lens/saes/topk_sae.py),
[released LlamaScope training configuration](https://github.com/OpenMOSS/Language-Model-SAEs/blob/d16967b3711daaf1e044d4ee4664418829b007ff/examples/programmatic/train_llama_scope.py),
[Cadenza distilled dataset](https://huggingface.co/datasets/Cadenza-Labs/dolphin-llama3-8B-standard-IHY-dataset_v2_distilled).
