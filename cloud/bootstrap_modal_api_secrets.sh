#!/usr/bin/env bash
# One-time local bootstrap. Reads the user's existing shell variables and stores
# them as named Modal secrets without printing their values.

set -euo pipefail

required=(
  OPENAI_API_KEY_MATS
  ANTHROPIC_API_KEY_MATS
  RP_API_KEY_MATS
)

for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing local environment variable: ${name}" >&2
    exit 1
  fi
done

uv run modal secret create --force em-sprint-judges \
  OPENAI_API_KEY="$OPENAI_API_KEY_MATS" \
  OPENAI_API_KEY_MATS="$OPENAI_API_KEY_MATS" \
  ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_MATS" \
  ANTHROPIC_API_KEY_MATS="$ANTHROPIC_API_KEY_MATS"

uv run modal secret create --force em-sprint-runpod \
  RP_API_KEY_MATS="$RP_API_KEY_MATS"

uv run modal secret list
echo "Named Modal secrets installed; no secret values were printed."
