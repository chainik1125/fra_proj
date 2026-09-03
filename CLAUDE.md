# Temporal Crosscoders

Research repository combining documentation (Obsidian vault) and Python code for temporal crosscoder experiments.

## Repository Structure

- `src/` -- Python source code
- `tests/` -- pytest test suite
- `docs/` -- Obsidian documentation vault
- `docs/templates/` -- document templates (Base.md, Guide.md)
- `docs/Tags.md` -- tag taxonomy index
- `run-checks.sh` -- runs all quality checks locally
- `check-tags.sh` -- validates tags in documentation
- `get-tags.sh` -- extracts tags from a markdown file
- `.markdownlint.jsonc` -- markdown linting rules
- `pyproject.toml` -- Python project configuration

## Key Rules

### Headings

- **No H1 headings** -- Obsidian shows the filename as the page title. Start with `##`.
- ATX style only (`##`, not underlines).

### Lists and Emphasis

- Dash (`-`) for unordered lists.
- Asterisks for emphasis: `*italic*`, `**bold**`.

### Links

- **Internal**: Obsidian wikilinks -- `[[Page Name]]` or `[[Page Name|display text]]`.
- **External**: Standard markdown -- `[text](url)`.
- No bare URLs.

### Code Blocks

- Fenced with backticks, always specify language.
- Use `text` for plain output.

## Frontmatter

Every doc in `docs/` requires YAML frontmatter:

```yaml
---
author: Name
date: YYYY-MM-DD
tags:
  - tag-name
---
```

- `author` -- required
- `date` -- required, ISO format
- `tags` -- required, at least one tag from `docs/Tags.md`

Tags must be `kebab-case` and listed in `docs/Tags.md`. Nested tags are valid if the parent exists.

## Running Checks

```bash
./run-checks.sh       # all checks (markdown lint + tag validation + link check)
./run-checks.sh -m    # markdown lint only
./run-checks.sh -t    # tag check only
./run-checks.sh -l    # link check only
```

## Python

- Python 3.12+, managed with `uv`
- Setup: `uv sync`
- Run tests with `uv run pytest`
- Source code in `src/`, tests in `tests/`

## Available Commands

Contributors can run these Claude Code slash commands at any time to get AI-assisted reviews of their work:

- `/review` -- auto-detect content type and run appropriate reviewers
- `/review-doc` -- documentation quality review (structure, cross-refs, frontmatter, clarity, formatting)
- `/review-math` -- mathematical correctness review (notation, proofs, equations, definitions)
- `/review-code` -- code quality review (bugs, tests, design, performance)
- `/review-experiment` -- experimental methodology review (reproducibility, controls, metrics)
- `/review-data` -- data quality review (schemas, formats, configs, pipelines)

Each command accepts an optional file path argument (e.g., `/review-math docs/Theory.md`). Without one, it diffs against `main` to find relevant changed files.

## CI Agent Review (currently disabled)

There is an automated agent review workflow in `.github/workflows/agent-review.yml` that can run these reviews on pull requests. It is currently disabled. To enable it, set `ANTHROPIC_API_KEY` in your GitHub repo secrets and change `if: false` to `if: github.event.pull_request.draft == false` in the workflow file.

## Tag Taxonomy

See `docs/Tags.md`. Current categories:

- **Type**: guide, design, proposal, results, reference
- **Status**: todo, in-progress, complete

---

## FRA toy experiment (branch `insen/fra-toy`)

Working state for the FRA-on-a-toy-model sprint. **Assume a cold start: read this
section, then `docs/insen/`, then `../fra_toy_brief.md` (the spec, outside the repo).**

### What this is

Plant a known QK edge in a data-generating process with ground-truth features,
train the smallest model that can express it (1L/1H attention-only), compute FRA
with the *true* feature directions as `W_dec`, and ask whether FRA recovers what
was planted and when it stops. The FRA paper validates only through downstream
steering; nothing checks whether the tensor entries are correct. This does.

### Environment

```text
python -m venv .venv                     # ../.venv relative to the repo
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install transformer_lens einops pytest matplotlib
```

Laptop CPU only, no GPU anywhere. Run tests with `.venv/Scripts/python.exe -m pytest tests/ -q`
(48 tests, ~45 s). `pyproject.toml` sets `pythonpath = ["."]`, so scripts need
`PYTHONPATH=.` when invoked directly.

### Layout

```text
fra/toy/config.py        ToyConfig -- one frozen dataclass per run
fra/toy/dgp.py           feature-matched retrieval + make_directions(rho)
fra/toy/model.py         1L/1H attn-only; ALL freezes live here
fra/toy/train.py         training loop; train_cached() reuses checkpoints
fra/toy/metrics.py       accuracy, Gate 2, query-side contrast
fra/toy/fra.py           oracle FRA: G, fra_qk, fra_ov, fra_ov_signed
fra/toy/recovery.py      rank / mass fraction / runner_up_ratio
fra/toy/intervention.py  feature-pair ablation (hook on hook_attn_scores)
fra/toy/conformance.py   adapter to the repo's own FRA conformance harness
fra/wsparse/             circuit_sparsity loader + sparse-G FRA (PAUSED thread)
fra/toy/steering.py      baseline interventions (residual, Q/K/V) + Pareto metrics
scripts/00..10           toy: DGP, train, oracle FRA, diagnostics, sweeps, Pareto
scripts/20..23           sleeper transfer: sanity gate, three arms, decision-position rank
sleeper/                 vendored from upstream/jamie/sleepers-final
weights/seeds/           trained SAE (not downloadable; 24 min to retrain)
results/                 *.json + figures/rho_sweep.png + checkpoints/
docs/insen/              five write-ups
```

Vendored from `upstream/dmitry/dev`: `tests/fra_conformance/` and `fra/core/`.
`fra_conformance.yaml` points the harness at our implementation, which passes it.

### Gates (all passing)

1. `||resid_pre - f @ W_dec||` = 2.98e-08. The oracle dictionary spans the residual.
2. Model solves the rule *through attention*: held-out query accuracy 98.8-100%,
   `argmax_is_key` = 100% at every rho.
3. FRA exactness: QK sums to the pre-softmax score at 1.9e-06, signed OV to
   `hook_attn_out`. Holds after training and at every rho.

Gates 1 and 3 are guaranteed a priori by the construction -- they validate the
IMPLEMENTATION, not the method. Say so in any write-up.

### Results so far

- **Recovery**: planted pair is rank 1 of 10,000 at rho=0, mass fraction 0.093.
- **rho sweep**: recovery margin collapses (circuit runner-up 5.83 -> 1.54) while
  the circuit stays perfect (Gate 2 = 100%). Not confounded -- BUT this is seed 0.
- **Multi-seed (3 seeds x 6 rho)**: training itself fails at high overlap.
  Admitted 3/3 at rho <= 0.4, 2/3 at 0.6, **1/3 at 0.8**. Failures are total
  (accuracy at chance, Gate 2 ~1-8%, G[l*,m*] NEGATIVE, rank 27-32), not
  degraded. So the claim is a conjunction: conditional on the circuit forming,
  FRA degrades; and the circuit forms less often as rho rises. rho=0.8 rests on
  one seed -- **the defensible range is rho <= 0.4**.
- **Trust the circuit-only metric**: across seeds its CV is <2% where the
  data-dependent aggregate is 22-36%. The single-seed agg value at rho=0.2 (1.11)
  was the low outlier of (1.11, 2.38, 2.14).
- **|W_Q| grows 21.4 -> 32.3 with rho** while G[l*,m*] falls -- corroborates the
  magnitude mechanism behind the ablation result.
- **Rank is the wrong metric**: aggregate rank is 1 at *every* rho including 0.8
  where the runner-up ratio is 0.98. Use `runner_up_ratio`.
- **Intervention**: ablating the planted pair destroys behaviour; matched-random
  and aggregate-runner-up do not. At rho=0.8 planted costs -22.5 pp vs -6.5 pp
  for the best live competitor and -0.3 pp for the aggregate runner-up.
- **Scale sweep**: over-ablation collapses behaviour to 0% at every rho, so the
  weak scale-1 effect at high rho is a MAGNITUDE threshold, not redundancy.
- **Signed vs L1**: identical within a head whenever activations are non-negative.
  Provable, so note 03's anti-predictive signed sum is a cross-head effect.
- **Steering Pareto (3 seeds x rho in {0, 0.4})**: TWO SEPARABLE CLAIMS.
  *Claim A (the site)*: a score-row edit perturbs EXACTLY zero logits outside the
  target row -- 0.000e+00 on all 6 runs, at up to 6x over-ablation -- while both
  baselines perturb ~48% of non-target positions. The perturbed set under
  residual steering is exactly {positions >= q*}. This is structural and belongs
  to score-space intervention generally, NOT to FRA.
  *Claim B (FRA)*: FRA is what says WHICH pair to ablate (rank 1 of 10,000, and
  random/runner-up ablations do nothing). Do not conflate them.
  Also: the paper's "QK->QK" baseline is NOT intermediate -- qkv and residual
  have identical collateral at every strength and seed, differing only at q*
  itself, which is a target position.
  Cost: at rho=0.4 FRA needs 1.6-2.4x more strength for the same suppression.
- **The accuracy-based collateral axis saturates.** It reported +0.00 damage for
  every method while residual steering was changing non-target logits by up to
  15.16 and perturbing 50.6% of positions. Use KL, not accuracy, for collateral.

### SLEEPER TRANSFER (TinyStories-33M) -- thread closed

Runs on laptop CPU: LoRA merge 11.4s, TL wrap 1.4s, backdoor fires 6/6 dep, 0/6
clean. SAEs are NOT downloadable; one ln1 seed-0 SAE trained at paper defaults
(weights/seeds/sae_ln1_s0.pt, 1416s CPU). Vendored sleeper/ from
upstream/jamie/sleepers-final so selection uses THEIR rank_qk_diff.

TWO METHODOLOGY FINDINGS (docs/insen/fra_qk_pair_selection.md):
1. rank_qk_diff's Z_q sums over ALL query positions, so it is position-agnostic
   in a task where position IS the mechanism. The trigger feature 1114 ranks
   #0/#5/#6 on the QUERY side but is live at the decision position in only 3.0%
   of dep rows. Restricting query_mask to the decision position (one argument,
   nothing else) moves the trigger to the KEY side (#3) and fills the query side
   with features live at the decision (97-100%). This subsumes the frequency
   bias: an unweighted position sum rewards how OFTEN a feature fires and is
   indifferent to WHERE.
2. _top_unique_from_pairs dedups each side then _get_tuples_diff zips by index,
   so candidate tuples pair features from different rows of the ranking. Their
   top-5 includes a pair their own score ranks #53,421 of 2,359,296.
Both fixes are a few lines and are written up.

NEGATIVE RESULT: score-space pair ablation does NOT transfer. Null on all three
pairs (+0.08 / +0.03 / -0.01 at 16x) including the causally correct ones, while
the paper's QK channel gives +8 to +21 and plain ln1 feature ablation +108 to
+140. Cause measured: the SAE has L0=32, so a (q,k) score decomposes into ~1022
pair terms and the identified pair carries only 2.14% of it (22x the average
pair, so identification is fine -- it is just not causally decisive). The toy's
Claim A held because the planted pair carried 83% of its cell at L0~4. The
mechanism claim (score-row edits perturb less than activation edits) is
unaffected; what does not transfer is that one PAIR is a large enough share to
steer with.

DO NOT re-run this thread. Case 2 has not been touched and is the open work.

### WEIGHT-SPARSE / circuit_sparsity (Case 2 candidate) -- PAUSED

Scoping done, one experiment run, thread paused deliberately. Assets live OUTSIDE
the repo in `../cs_data/` (642 MB csp_yolo1, 1.68 GB csp_yolo2, 96 MB circuit) and
`../circuit_sparsity/` (their code). New deps: `blobfile`, `tiktoken`.
Reproduce every number with `scripts/30_wsparse_scope.py` -> `results/wsparse_scope.json`.

Models load on laptop CPU: csp_yolo1 2.2s (12L, 128 heads, d_head 8, d_model 1024,
160.6M params, 5.0M nonzero = 3.1%), csp_yolo2 21.2s (8L, 128 heads, d_head 16,
d_model 2048, afrac=0.25, sink=True). Both rms_norm=True. Circuits are
`viz/<model>/<task>/<sweep>/<k>/viz_data.pt`; `circuit_data` is keyed by
hook name with int index tensors of RETAINED channels.

TWO FINDINGS THAT MATTER:

1. **The pair term is only 45.9% of the summed score mass** (layer 10 head 82,
   mean over 512 cells): pair 2.9864, bias-x-feat 2.7308, feat-x-bias 0.0559,
   const 0.7280. `c_attn` is an nn.Linear with a DENSE bias (3072/3072 nonzero,
   max 5.82), so the score is not purely bilinear in act_in and the Eqs 13-15
   terms are mandatory -- exactness is 5.9e-06 WITH them and off by 5.27 without.
   **FRA's 4-index tensor does not capture the majority of the score on this
   model.** bias-x-feat is nearly as large as the pair term.

2. **Data-weighted circuit pair share = 3.22% mean / 3.05% median**, against the
   sleeper's 2.14% (weights-only was 0.42%). Same order. **Pair steering is no
   better here than on the sleeper.**

Other measurements: effective live pairs per cell -- yolo1 517, yolo2 195,
sleeper 1022, so yolo2 is 2.6x sparser and is the better target. Activation
sparsity removes only ~40-50% of support pairs, not the 93.75% independence would
predict, because retained activations are not independent of the weight support.

THE CORRELATION IS A NULL AND THE TEST IS MIS-SPECIFIED. Spearman vs
`ch_interv_losses` at `10.attn.act_in` (n=20): FRA +0.156, |act| control +0.253;
excluding channel 460, +0.304 vs +0.363. The control wins. But ground truth
ablates an act_in channel, which feeds Q, K AND V, while FRA-QK models only Q,K --
and the dominant channel 460 (loss 5.5771, 38x the next) has |Wq[:,460]| = 0.0.
Its importance is OV-mediated, so FRA ranking it low is correct behaviour. This
is not evidence against FRA.

NEXT STEPS, verbatim:
  1. Add FRA-OV and redo the correlation on combined attribution. The current
     null is an artefact of testing a QK-only attribution against QK+OV ground
     truth.
  2. If that also nulls, report it -- with n=20 and one dominant channel this
     dataset may simply be too thin, and pooling `attn.act_in` across all layers
     (186 channels in the k=1024 circuit) is the obvious way to get n up.
  3. Switch to yolo2 for anything further.

### DO FIRST TOMORROW

Nothing is blocked. Candidates, in rough priority order:

1. **Stage B** -- train a TopK SAE on `hook_resid_pre` and repeat the recovery
   metrics. The Stage A / Stage B gap is the cost of SAE imperfection measured
   against ground truth, and it is the last major piece of the original brief.
2. **A second planted rule** sharing the head (`lambda_2 -> mu_2`). The current
   toy has ONE behaviour, so the collateral axis has no other capability to
   damage. A second rule would make the Pareto comparison much stronger and
   would let arm (d)-style competitor tests be genuinely load-bearing.
3. `d_head` as a sweep axis (rank(G) <= d_head structurally caps concentration).
4. `n_feat/d_model` as a second overlap axis -- forced rather than imposed
   overlap, closer to a real model.

RESOLVED (was yesterday's first task): the rho=0 transition is a **gradient, not
a discontinuity**. rho = 0.01/0.02/0.05 interpolate smoothly (34.7/45.1/64.9%
post-ablation accuracy vs 19.4% at rho=0 and 77.3% at rho=0.1), with
`mass_on_key` rising monotonically. The flat 0.74 above rho=0.1 is a saturation
tail. Redundancy is closed out; magnitude stands.

### Known issues

- `train_cached` on a cache hit does not advance the DGP generator (it skips the
  `steps` sampling calls training makes), so scripts 08/09 evaluate on a different
  batch than 06/07: 19.43% vs 20.02% for the same intervention. Internally
  consistent, no conclusion depends on it, but exact percentages are not
  comparable across the two groups. Fix by using `train` everywhere, or by giving
  evaluation its own generator.
- 3-arm vs 4-arm agree to 1e-12 on shared arms, so training IS deterministic.
- `./run-checks.sh -t` fails on `docs/dmitry/dmitry_about_me.md` -- frontmatter
  with no `tags:` field. Pre-existing on `origin/main`, not ours.
- `markdownlint` and `lychee` cannot run here (no node/npx, no lychee). The three
  notes were hand-verified against `.markdownlint.jsonc`; that is not a clean
  linter run. Re-check on a machine with node.
- TransformerLens emits a `HookedTransformer` deprecation warning. Ignored
  deliberately -- migrating to `TransformerBridge` is out of scope.

### Not started, deliberately

Stage B (train an SAE and repeat), the openreview DGP, multi-layer/multi-head,
RoPE/RMSNorm corrections (both are config flags: `zero_qk_biases`,
`zero_ov_biases`), `d_head` as a sweep axis, `n_feat/d_model` as a second overlap
axis. See section 8 of the brief for the full scope guard.

### Open questions for Dmitry

1. Which paper's DGP? (brief 3.1 -- best guess Chanin and Garriga-Alonso 2508.16560)
2. Should the toy live in `fra_proj` or `chainik1125/fra`?
3. Is mass fraction worth keeping given it is denominator-dominated?
4. `autoresearch/cadenza-attn-only` is NOT this experiment -- it is an 8B sleeper
   LoRA study. Confirmed, so brief question 2 is answered.
