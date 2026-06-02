# scripts/experiments/ — throwaway experiment drivers

Everything in this directory is **experimental scaffolding**, not part of the
paper-reproduction pipeline. The clean repo is `reproduce.sh` → `run_experiment`
(+ the plotting scripts) run with **default flags**, which are exactly
paper-faithful. Nothing here is imported by the production pipeline.

## Gating convention

Experimental variations live behind **opt-in flags that default to the
production behaviour**. Each variant's code is tagged `# EXPERIMENTAL`
(greppable). The defaults reproduce the paper, so `reproduce.sh` is unaffected.

Current experimental knobs (all on `run_experiment`):

| flag | default (production) | experiment |
|---|---|---|
| `--ov_select` | `cosine` — paper-faithful cosine re-rank (shared with Conv) | `attr_asr` — raw top-K attribution → min-ASR, no re-rank |
| `--eval_set`  | `disjoint` — deduped + disjoint from selection AND SAE-training rows | `paper` — historical dedup-only slice (re-includes train-leaked prompts) |

## Producing the clean repo when experiments are done

1. `grep -rn EXPERIMENTAL scripts/ sleeper/` — every hit is a knob to remove.
2. For each: delete the `--<flag>` argument, the threaded kwarg, and the
   `# EXPERIMENTAL` branch; make the production branch unconditional.
   - `ov_select`: drop the param + the `elif channel == "ov"` branch in
     `select_features`; make the cosine guard `if channel == "ov":`.
   - `eval_set`: drop the param everywhere; restore the plain
     `harvested_train_texts(...)` call in `sleeper.eval.split_dep_prompts`.
3. `rm -rf scripts/experiments/`.
4. `uv run -m scripts.run_experiment ...` / `reproduce.sh` must still work
   identically (the defaults never changed).
