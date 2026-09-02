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
scripts/00..09           show DGP, train, oracle FRA, diagnostics, sweeps
results/                 *.json + figures/rho_sweep.png + checkpoints/
docs/insen/              three write-ups
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
  the circuit stays perfect (Gate 2 = 100% everywhere). Not confounded.
- **Rank is the wrong metric**: aggregate rank is 1 at *every* rho including 0.8
  where the runner-up ratio is 0.98. Use `runner_up_ratio`.
- **Intervention**: ablating the planted pair destroys behaviour; matched-random
  and aggregate-runner-up do not. At rho=0.8 planted costs -22.5 pp vs -6.5 pp
  for the best live competitor and -0.3 pp for the aggregate runner-up.
- **Scale sweep**: over-ablation collapses behaviour to 0% at every rho, so the
  weak scale-1 effect at high rho is a MAGNITUDE threshold, not redundancy.
- **Signed vs L1**: identical within a head whenever activations are non-negative.
  Provable, so note 03's anti-predictive signed sum is a cross-head effect.

### DO FIRST TOMORROW

**Add rho = 0.01, 0.02, 0.05 to the planted-arm ablation** (`scripts/08_ablation_scale.py`,
extend `RHOS`; checkpoints make it cheap but these three rho values are new so they
will train, ~2 min each).

Why: post-ablation `mass_on_key` is flat at ~0.74 across rho 0.1-0.8 while the
runner-up ratio falls 2.77 -> 0.98. A redundancy story does not predict that --
redundancy should *grow* with rho. The discontinuity may sit at exactly rho=0,
and exact orthogonality is measure-zero. If 0.01/0.02/0.05 all look like 0.1 it
is a discontinuity, not a gradient.

**Do not write the redundancy interpretation until this is resolved.** The scale
sweep already argues against redundancy and for magnitude; this checks the shape
of the transition, which is a separate question.

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
