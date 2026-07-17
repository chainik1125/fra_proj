# Simplex Research

Alignment research project using [simplexity](https://github.com/Astera-org/simplexity) for training predictive models on Hidden Markov Model sequences.

@README.md

## Quick Reference

```bash
# Setup
uv sync

# Minimal training (no Hydra/MLflow — just the core loop)
uv run python training/run_minimal.py
```

## Project Structure

- `training/` - Minimal training loop and HMM matrices
  - `run_minimal.py` - Minimal training loop (`Config`, `build_model`, `train`)
  - `matrices.py` - Custom HMM transition matrices (direct-sum, Kronecker, AFP builders)
- `analysis/` - Jupyter notebooks for probing/representation experiments
  - `act_avg_claude.ipynb` - Direct-sum N-coin training + activation-averaging analysis
- `shared_tools/` - Shared utilities (run persistence)

## Conventions

- Use `uv` for all Python dependency management and script execution
- JAX for generative models, PyTorch for predictive models

## Bash Operations

Complex bash syntax is hard for Claude Code to permission correctly. Keep commands simple.

Simple operations are fine: `|`, `||`, `&&`, `>` redirects.

For bulk operations on multiple files, use xargs:
- Plain: `ls *.md | xargs wc -l`
- With placeholder: `ls *.md | xargs -I{} head -1 {}`

Avoid string interpolation (`$()`, backticks, `${}`), heredocs, loops, and advanced xargs flags (`-P`, `-L`, `-n`) - these require scripts or simpler alternatives.

**Patterns:**
- File creation: Write tool, not `cat << 'EOF' > file`
- Env vars: `export VAR=val && command`, not `VAR=val command` or `env VAR=val command`
- Bulk operations: `ls *.md | xargs wc -l`, not `for f in *.md; do cmd "$f"; done`
- Parallel/batched xargs: use scripts, not `xargs -P4` or `xargs -L1`
- Per-item shell logic: use scripts, not `xargs sh -c '...'`
- Git: `git <command>`, not `git -C <path> <command>` (breaks permissions)

If a command that should be allowed is denied, or if project structure changes significantly, ask about running `/mats:permissions` to update settings.

## Running Commands

Run scripts via `uv run python <script>`. Available project scripts:
- `uv run python training/run_minimal.py` - Minimal training

**Ad-hoc scripts:** Only `/tmp/claude-execution-allowed/simplex-research/` is approved for ad-hoc scripts. Non-bash scripts run with `uv run /tmp/claude-execution-allowed/simplex-research/<script-name>`. Bash scripts run with `bash /tmp/claude-execution-allowed/simplex-research/<script-name>`.

When you create a new reusable script, offer to add a permission for it. Example: "I created scripts/analyze.py. Want me to add `Bash(uv run scripts/analyze.py *)` to your permissions?"

For string interpolation, heredocs, loops, or advanced xargs flags, write a script in `/tmp/claude-execution-allowed/simplex-research/` instead.

Use `curl` when you need to see the full response content (WebFetch summarizes).
