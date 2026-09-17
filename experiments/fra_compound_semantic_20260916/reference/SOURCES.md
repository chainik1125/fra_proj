# Sources and provenance

- Model: [Gemma-2-9B-IT](https://huggingface.co/google/gemma-2-9b-it).
- Native instruction-tuned residual SAEs:
  [Gemma Scope 9B IT](https://huggingface.co/google/gemma-scope-9b-it-res),
  16k widths at residual-post layers 9/20/31, average L0 88/91/76.
- Preprocessing specification: [Gemma Scope paper](https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf),
  §3.1 and appendix A. Training uses a fixed scale; released parameters absorb it.
  Extra per-token input normalization is disabled here.
- Pinned loader: [SAE Lens v5.10.7 pretrained_sae_loaders.py](https://github.com/jbloomAus/SAELens/blob/v5.10.7/sae_lens/toolkit/pretrained_sae_loaders.py),
  `gemma_2_sae_huggingface_loader`, imports released `params.npz` weights directly.
- The selected-pair FRA evaluator, projection/RoPE helpers and SAE wrapper were
  copied from `experiments/fra_semantic_restoration_20260916/` at repository commit
  `3e9122d95955ea36994b166cf35a46727c5534af`. The wrapper's inaccurate normalization
  documentation is preserved in the reference copy; the caller explicitly uses
  `normalize_activations=False`. `CompoundFRA` vectorizes position rotations and
  evaluates the same selected-pair term formula and 1e-10 cutoff.
- Base-model Billing screen source: `base_screen_*.py`.
- Instruction-tuned Billing screen source: `billing_screen_*.py`.
- Print-payload screen source: current `design.py` and `feasibility.py`.
- Stopped normalization-True smoke source: `smoke_before_normalization_fix.py`.
- Interrupted full-run source, before numerical-failure handling:
  `full_before_numeric_failure_fix.py`. Its completed 204 configurations are
  retained in `results/interventions_interrupted.json.gz` and reused only after
  identical feature/pair calibration identities were checked.

Scientific source SHA256 hashes and exact package versions are recorded inside
each returned result. Model weights and uncompressed remote checkpoints are not
committed. Results are gzip-compressed and sharded when necessary to respect the
repository's <1 MB hook.
