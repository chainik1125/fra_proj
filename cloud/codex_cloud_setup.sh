#!/usr/bin/env bash
# Codex Cloud environment bootstrap for the EM gradient-transfer sprint.
#
# Configure the two Modal values referenced below as Codex Cloud *Secrets*, not
# ordinary environment variables. Cloud secrets exist only during setup. The
# provider keys were escrowed separately by bootstrap_modal_api_secrets.sh and
# never need to enter the Codex Cloud environment.

set -euo pipefail

required=(
  MODAL_TOKEN_ID
  MODAL_TOKEN_SECRET
)

for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required Codex Cloud setup secret: ${name}" >&2
    exit 1
  fi
done

uv sync

# A dash tells Modal to read the credential from stdin, keeping it out of the
# process argument list. The resulting ~/.modal.toml survives into agent phase.
printf '%s\n%s\n' "$MODAL_TOKEN_ID" "$MODAL_TOKEN_SECRET" |
  uv run modal token set --token-id - --token-secret - --verify

uv run modal secret list

echo "Codex Cloud bootstrap complete: Modal authenticated. Confirm that em-sprint-judges and em-sprint-runpod are listed above."
