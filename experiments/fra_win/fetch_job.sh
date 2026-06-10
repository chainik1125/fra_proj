#!/usr/bin/env bash
# fetch_job.sh <jobid>
set -e
jid="$1"
rm -rf /tmp/fra_out/fra_win/out/$jid
hf download dmanningcoe/fra-phase1-steering-data --include "fra_win/out/$jid/*" --repo-type dataset --local-dir /tmp/fra_out >/dev/null 2>&1 || true
echo "=== out.log (tail) ==="; tail -40 /tmp/fra_out/fra_win/out/$jid/out.log 2>/dev/null || echo "(no out.log yet)"
echo "=== files ==="; ls -la /tmp/fra_out/fra_win/out/$jid/ 2>/dev/null
