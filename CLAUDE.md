# sleeper — repo guide

Sleeper-agent case study only (consolidated from `jamie/autoresearch-jsdc`).
See `README.md` for the pipeline, the per-figure commands, and the math.

## Layout

- `sleeper/` — library (model + dataset, SAE, hooks, attribution, baselines, metrics)
- `scripts/` — CLI entry points (one per task)
- `weights/` — trained SAE checkpoints and attribution tensors (gitignored)

## Conventions

- Python 3.12+, managed with `uv`. `uv sync` to install.
- Hook semantics: every steering hook patches **only the first P positions**
  (P = `story_marker_pos + 1`, the prompt up to and including `Story:`).
  Generated tokens are not directly perturbed — they feel the intervention
  through attention back to the patched prompt.
- Attention pattern is **frozen by leaving Q and K alone**: the OV-only
  intervention patches `attn.hook_v` with a delta projected through W_V.
  No separate `hook_pattern` cache/replay is needed.
- One steer hook factory (`additive_steer_hook`) is shared by SAE-feature
  ablation and mean-diff steering. The OV-only path has its own
  (`ov_only_steer_hook`) because the projection through W_V is per-head.

## Adding a new intervention

If the intervention reduces to "add `α · v` at one resid hook on prompt
positions," build the delta tensor and call `additive_steer_hook`. If it's
"only OV reads see it," build the delta in ln1-space and call
`ov_only_steer_hook(delta, α, W_V, block)`. New scripts go in `scripts/`,
keep them ≤ 200 lines.
