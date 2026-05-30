#!/usr/bin/env bash
# Spin a persistent CPU pod that runs the campaign driver headless, so the
# campaign survives the laptop sleeping/closing. The CPU pod owns the DAG; it
# launches the ephemeral fra-diff-* GPU cell pods. Run this LOCALLY, once.
#
#   ./scripts/launch_campaign_cpu.sh campaigns/qwen14b_medical.yaml
#
# Requires in your local env: RP_API_KEY_MATS, HF_TOKEN, OPENAI_API_KEY_MATS.
# The CPU pod is named dmitry-fra-campaign-* (PROTECTED prefix — budget_watch
# only reaps fra-diff-*, never this driver pod).
set -euo pipefail
YAML="${1:?usage: launch_campaign_cpu.sh <campaign.yaml>}"
: "${RP_API_KEY_MATS:?}"; : "${HF_TOKEN:?}"; : "${OPENAI_API_KEY_MATS:?}"

NAME_SUFFIX="$(basename "$YAML" .yaml)"
POD_NAME="dmitry-fra-campaign-${NAME_SUFFIX}"
BRANCH="$(python3 -c "import yaml;print(yaml.safe_load(open('$YAML'))['campaign']['branch'])")"
REPO_URL="$(python3 -c "import yaml;print(yaml.safe_load(open('$YAML'))['campaign']['repo_url'])")"

# Driver bootstrap: clone, install the driver's lightweight deps (NO torch — the
# driver only orchestrates), nohup run_campaign.py against the yaml.
read -r -d '' BOOTSTRAP <<EOF || true
#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/campaign_boot.log) 2>&1
cd /workspace
[ -d fra_proj ] || git clone --branch ${BRANCH} --single-branch ${REPO_URL} fra_proj
cd fra_proj && git fetch origin && git checkout ${BRANCH} && git pull --ff-only
pip install --no-input --break-system-packages pyyaml huggingface_hub openai requests certifi 2>&1 | tail -2
nohup python3 -u scripts/run_campaign.py ${YAML} > /workspace/campaign.log 2>&1 &
echo "[cpu-driver] run_campaign.py launched (pid \$!) — tail /workspace/campaign.log"
sleep infinity
EOF

echo "[launch] CPU driver pod: $POD_NAME  (yaml=$YAML branch=$BRANCH)"
RP_API_KEY_MATS="$RP_API_KEY_MATS" python3 - "$POD_NAME" <<PY
import sys, os
sys.path.insert(0, "scripts")
import runpod_launch as rp
name = sys.argv[1]
bootstrap = '''$BOOTSTRAP'''
env = {
    "RP_API_KEY_MATS": os.environ["RP_API_KEY_MATS"],
    "RUNPOD_API_KEY":  os.environ["RP_API_KEY_MATS"],
    "HF_TOKEN":        os.environ["HF_TOKEN"],
    "OPENAI_API_KEY_MATS": os.environ["OPENAI_API_KEY_MATS"],
}
pid = rp.launch_cpu_pod(name, bootstrap, env=env)
if pid:
    print(f"[launch] CPU driver pod up: {pid}")
    print("  ssh in and: tail -f /workspace/campaign.log   (the driver DAG)")
else:
    print("[launch] CPU pod launch failed (CPU supply / instanceId). Fallback: run")
    print(f"  RP_API_KEY_MATS=… HF_TOKEN=… OPENAI_API_KEY_MATS=… \\\\")
    print(f"  nohup .venv/bin/python3 scripts/run_campaign.py {os.environ.get('YAML','<yaml>')} &")
    sys.exit(1)
PY
