# Chanin features → single-head FRA → independent reset chains

This experiment uses the released Chanin generator and BatchTopK SAE, then
trains a causal attention head directly in activation space. The default target
is the next activation, under squared error. This target was not specified in
the original request; conclusions about QK depend on this choice.

## Reproduction provenance

- Paper: https://arxiv.org/pdf/2508.16560 (accessed 2026-09-10, v4).
- Reference: https://github.com/chanind/sparse-but-wrong-paper
- Pinned revision: `d5886b540dc5b9cac4f76e6db2b0cce1b0b7c585`.
- Upstream `ToyModel`, correlation generator, BatchTopKTrainingSAE and
  `train_toy_sae` are used directly, with the upstream `uv.lock`.
- 50 features in 100 dimensions; 50 SAE latents; k=11; magnitudes
  `ReLU(N(1, .15²))`; 15 million fresh training samples; Adam, lr=3e-4.
- Batch size 500 follows the paper. The released helper defaults to 1024.
- The released probability expression is
  `p_i = .345*(49-i)/50 + .05`: endpoints .3881 and .05, sum 10.9525.
  The prose endpoint .345 does not describe that expression. We preserve code.
- The covariance supplied to the Gaussian threshold sampler is a *Gaussian*
  correlation matrix, not the resulting Bernoulli correlation matrix.
- Ground-truth directions come from the upstream numerical orthogonalization,
  not an identity matrix supplied as a learned result. Gram error is recorded.
- SAE initialization is random. The RNG is explicitly reseeded after the
  common geometry is generated, since upstream correlation generation resets
  the global RNG. The original notebook does not fix all seeds/checkpoints.

We calculate unrounded, signed cosine similarities and use a one-to-one
Hungarian assignment. We compare this independently against the upstream
cosine function. There is no subtraction of ground-truth Gram entries.
Predefined recovery criterion: minimum matched cosine >= .99 and maximum
absolute off-diagonal <= .05. Raw matrices, the permutation, Gram matrix,
learned weights and metrics are retained. “Exact reproduction” here means
repeating the released experiment and testing recovery, not claiming a
bit-for-bit reproduction of a published checkpoint that was not supplied.
Seeds 1 and 2 pass; the initial seed 42 run fails with a duplicated direction
and is retained in `results/recovery`. The default is the passing seed 2;
attention experiments fix that recovered dictionary and vary head seeds 1–3.

## Attention experiment

With row vectors, `x_t = z_t F`, the head is

```
S_ts = (x_t W_Q)(x_s W_K)^T / sqrt(100) + b_lag[t-s],  s <= t
A_t  = softmax(S_t)
y_t  = sum_s A_ts x_s W_V W_O + b_out
target = x_(t+1)
```

All four weight matrices are 100×100 and trained from random initialization.
Q, K and V projections are bias-free; b_out is learned. A Q bias would add
key-only score terms, so this architectural assumption matters for interpreting
the negative QK background (see THEORY.md).
There is no W_E, W_U, MLP, layer norm or residual skip. An output bias allows
the exact IID mean solution. We explicitly compare content-only attention
(`--no-position`) with an additive learned relative-lag score bias. The latter
allows recency routing independently of feature content; it is an architectural
control, not another activation-space feature. A second training arm fixes
content QK to zero and trains OV, output bias and any positional bias.
An additional `--qk-modes diagonal` control trains a diagonal content score in
the known ground-truth basis; this is an explicitly oracle-informed restriction
of the same head, used to test whether off-diagonal routing is necessary.

The three main data regimes share the recovered feature geometry and marginal
firing probabilities:

1. Original Chanin, correlated within position and IID across positions.
2. Independent features, IID across positions (matched control).
3. Independent per-feature reset chains, rho=.7, observed state = firing mask.

The independent HMM changes within-position correlations relative to the
original Chanin generator. Regimes 2→3 isolate temporal dependence; regimes
1→3 alone would confound temporal and cross-feature dependence. `--noisy`
also runs a one-sided specialization of the temporal paper's noisy-emission
family: we choose `p_A=0`, use the paper's `p_B=.625` and rho=.7, and retain
the Chanin 50-feature geometry and heterogeneous pi values. The paper's
denoising panel has 20 features; this extension does not reproduce that panel.

Metrics include MSE divided by held-out mean-predictor MSE, exact Bayes
conditional-mean prediction error, full-precision QK and OV feature matrices,
attention entropy/self mass, centered off-diagonal score amplitude, absolute
FRA feature-pair mass, and paired QK intervention losses. Confidence intervals
for interventions use independent sequences, not overlapping positions.

QK is `F W_Q W_K^T F^T / sqrt(100)`. OV is the source-to-destination
coordinate map `F W_V W_O F^+`, with a pseudoinverse so finite Gram error does
not silently change coordinates. We also compute both in the aligned,
normalized learned SAE dictionary. Score reconstruction is checked against
the head's actual scores. The learned SAE coordinates are an approximate
feature description; exact score reconstruction uses the known generator
coefficients and ground-truth dictionary. Bias terms are separate.

## Running

```bash
git clone https://github.com/chanind/sparse-but-wrong-paper.git /tmp/fra-chanin-reference-20260910
git -C /tmp/fra-chanin-reference-20260910 checkout d5886b540dc5b9cac4f76e6db2b0cce1b0b7c585
uv sync --directory /tmp/fra-chanin-reference-20260910 --frozen
/tmp/fra-chanin-reference-20260910/.venv/bin/python experiments/chanin_attention/reproduce.py
/tmp/fra-chanin-reference-20260910/.venv/bin/python -m unittest discover -s experiments/chanin_attention -p test_attention.py -v
/tmp/fra-chanin-reference-20260910/.venv/bin/python experiments/chanin_attention/attention.py --noisy --out experiments/chanin_attention/results/attention_position
/tmp/fra-chanin-reference-20260910/.venv/bin/python experiments/chanin_attention/attention.py --noisy --no-position --out experiments/chanin_attention/results/attention_content_only
/tmp/fra-chanin-reference-20260910/.venv/bin/python experiments/chanin_attention/attention.py --noisy --no-position --regimes markov --qk-modes diagonal --out experiments/chanin_attention/results/attention_content_only_diagonal
/tmp/fra-chanin-reference-20260910/.venv/bin/python experiments/chanin_attention/attention.py --no-position --regimes markov --qk-modes full --rhos 0.3 0.9 --out experiments/chanin_attention/results/attention_content_only_rho_sweep
/tmp/fra-chanin-reference-20260910/.venv/bin/python experiments/chanin_attention/summarize.py
```

These commands reproduce the 60-cell attention campaign (20 settings × 3
head seeds). `summarize.py` rebuilds TABLES.md, aggregate.json, mechanism.json
and overview.png/PDF. The additional SAE recovery seeds are recorded in
FINDINGS.md and can be rerun with `reproduce.py --seed ... --out ...`.

Completed attention cells are skipped on restart. Use a fresh output directory
for changes to training parameters. Results and limitations belong in
`FINDINGS.md`; analytic predictions are in `THEORY.md`.
