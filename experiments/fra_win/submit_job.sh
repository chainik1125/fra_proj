#!/usr/bin/env bash
# submit_job.sh <jobid> <local_script.py>
set -e
jid="$1"; script="$2"
hf upload dmanningcoe/fra-phase1-steering-data "$script" fra_win/jobs/$jid.py --repo-type dataset >/dev/null
printf '%s' "$jid" > /tmp/job_id.txt
hf upload dmanningcoe/fra-phase1-steering-data /tmp/job_id.txt fra_win/job_id.txt --repo-type dataset >/dev/null
echo "submitted job=$jid"
