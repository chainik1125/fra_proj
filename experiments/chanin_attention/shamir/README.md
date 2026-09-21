# Shamir secret-sharing HMM experiment

See [FINDINGS.md](FINDINGS.md) for results, [THEORY.md](THEORY.md) for exact
claims, [TABLES.md](TABLES.md) for all seed ranges, and [overview.png](overview.png)
for the figure. This is a new controlled benchmark, not a reproduction of a
published Shamir-HMM experiment.

## Construction and scope

We instantiate [Shamir's polynomial secret sharing](https://people.cs.pitt.edu/~litman/courses/cs2001/lectures/lecture-02-p612-shamir.pdf)
as a small finite-state regenerative HMM. For each of C independent streams,
sample secret S and slope a uniformly from GF(5), including a=0. Emit the two
shares y1=S+a and y2=S+2a, modulo 5, tagged with stream and share identity.
The secret is S=2y1-y2 modulo 5. Either share alone is independent of S.

All 2C shares are randomly interleaved. In 75% of episodes all are visible;
otherwise take a uniformly chosen prefix of length 1,...,2C-1. A subsequent
query requests a uniformly random stream; the next emission is its secret.
Then reset independently. Hidden state can consist of all polynomials, the
permutation, prefix length, phase/index and request. The sampler directly
draws the history at query time; it does not step through emissions individually.
Training scores only the query-to-secret transition, not full-sequence language
modeling. C is either 1 or 3.

All observations are sums of fixed Chanin activation directions from
`../results/recovery_seed2/geometry.pt` (50 directions, 100 dimensions):

| Feature indices | Meaning |
|---|---|
| 0–2 | Stream identity, shared by query and memory |
| 3–12 | Ten separate (share index x, share value y) atoms |
| 13 | Query type |
| 14 | Share-token type |
| 15–19 | Five secret-output values |
| 20–49 | Unused |

A memory token has three active atoms: stream, (x,y), and share type. A query
has two: requested stream and query type. The target is one secret-value atom.
Amplitudes are one, with no added observation noise. This reuses the verified
geometry, not the original Chanin firing distribution. No new SAE is trained;
the aligned recovered decoder is used only to check feature-basis stability.

## Architecture and controls

One 100-dimensional head, WQ/WK/WV/WO, output bias, learned relative-lag score
bias. No embedding, unembedding, layer norm, or residual skip. Crucially the
query is used only for Q: it is excluded from K/V, and is not passed directly
to any output decoder. Attention reads strictly preceding share tokens.
Padded memory positions are masked; there is no null/BOS memory token.

We train both the original linear-output head and an explicit additional
100→256→100 ReLU MLP after that head. The MLP arm changes the architecture;
its success must not be reported as a success of attention alone. Both arms
have full-QK and separately trained zero-content-QK (position-only) controls.
The latter still learns relative-lag scores, WV/WO, bias and, when present,
the MLP. The code also offers an oracle selector, but it was not part of the
reported campaign.

Three initialization/data seeds (1,2,3) × two stream counts × two readouts ×
two routing modes = 24 independent configurations. Each trains 6,000 Adam
steps of batch 128, learning rate .002, gradient clipping at 1, final-quarter
cosine decay to .15 of initial learning rate. CPU, two Torch threads per
process. No hyperparameter sweep or best-seed selection.

## Measurements

`metrics.json` evaluates 8,192 fresh query prefixes, shared across cells,
stratifying by available requested-share count 0/1/2. Below threshold the
Bayes conditional activation is the mean of five secret atoms, and Bayes
accuracy is 20%; with both shares it is the exact secret and accuracy is 100%.
NMSE divides summed activation squared error by the unconditional secret
activation variance, so the unconditional mean predictor has population NMSE 1.
"Oracle error" is normalized squared distance to the Bayes conditional mean.

Interventions retain all trained weights and positional scores: zero content
QK; retain only its true-feature-basis diagonal; retain only stream-selection
columns; and change the request while keeping the original target. The last
is a damage check, not an accuracy test for the new requested secret.

`analyze.py` adds a separate counterfactual audit: 2,048 fresh memories with ALL
shares visible, each queried for EVERY stream, with the target updated to the
new request. This is 6,144 queries but only 2,048 independent memories for C=3.
`TABLES.md` reports this audit, not the mixed-prefix strata. Its seed ranges
are ranges, not confidence intervals; seeds share held-out examples.

Pre-MLP OV-output affine ridge probes use 4,096 training and 4,096 held-out
full-memory examples. Targets are one-hot share 1, share 2, or secret. Both
regression R² and argmax accuracy are saved. Negative R² alone does not prove
the representation contains no linearly classifiable secret information.

The stored legacy key `query_blind_complete_nmse_lower_bound` refers to the
population FULL-memory bound, not to conditioning only on requested-share
count=2 in the mixed-prefix data. `audit.json` gives it the unambiguous name
`query_blind_full_memory_population_nmse_lower_bound`.

## Reproduce

From the repository root, use the existing Chanin-reference environment
(`/tmp/fra-chanin-reference-20260910/.venv/bin/python`) or an environment with
Torch 2.8.0, NumPy and Matplotlib. The geometry's SHA256 is saved in each run.

```sh
python experiments/chanin_attention/shamir/experiment.py --streams 1 --seeds 1 2 3 --out experiments/chanin_attention/shamir/results_streams1
python experiments/chanin_attention/shamir/experiment.py --streams 3 --seeds 1 2 3 --out experiments/chanin_attention/shamir/results_streams3
python experiments/chanin_attention/shamir/analyze.py
python -m unittest discover -s experiments/chanin_attention/shamir -p test_shamir.py -v
```

Completed cells are reused after checking the key training configuration and
geometry hash; choose a new output directory for changed settings. The two
stream-count commands may run concurrently. Initial exploratory seed-1 runs
remain under `pilot/` but are excluded from aggregation and do not constitute
additional independent seeds.

Per-cell artifacts contain trained/initial checkpoints, configuration and
metrics, training trajectory, QK in true/recovered feature bases, OV vectors
and lag scores. Analysis creates `audit.json`, `aggregate.json`, `TABLES.md`,
`arithmetic_grid.npz`, and PNG/PDF figures. Five tests exhaustively verify the
small-field sharing/privacy property, independently enumerate the posterior,
check positional query blindness, enumerate the query-blind MSE bound, and
verify the score and OV feature expansions.
