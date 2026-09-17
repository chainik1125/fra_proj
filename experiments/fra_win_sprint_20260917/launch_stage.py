"""Launch one authorized sprint stage with budget checks and a local log."""
import argparse
import datetime
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WRITING_START = datetime.datetime(2026, 9, 18, 2, 16, 27, tzinfo=datetime.timezone.utc)
COMPUTE_END = datetime.datetime(2026, 9, 18, 2, 46, 27, tzinfo=datetime.timezone.utc)
RATES = {'A100-80GB': 3.198528, 'H100!': 4.649328}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--stage', required=True)
parser.add_argument('--fast', action='store_true')
parser.add_argument('--log-name')
parser.add_argument('--then', nargs='*', choices=['polish', 'confirm'], default=[])
args = parser.parse_args()
allowed = ['search_', 'variant_search_', 'refine_', 'variant_refine_',
           'learn_pairs_', 'variant_learn_pairs_', 'baseline_extra_',
           'baseline_extra2_', 'baseline_abs_', 'baseline_nobos_', 'baseline_scopes_', 'polish_',
           'confirm_', 'robustness_', 'pair_budget_', 'multi_sae_']
assert any(args.stage == prefix+task for prefix in allowed
           for task in ['tenants_long', 'narrative_contracts']), args.stage
log_name = args.log_name or args.stage
assert re.fullmatch(r'[a-z0-9_]+', log_name)
now = datetime.datetime.now(datetime.timezone.utc)
assert now < COMPUTE_END, 'The background-compute deadline has passed.'
gpu = 'H100!' if args.fast else 'A100-80GB'
for attempt in range(7):
    subprocess.run([sys.executable, str(ROOT/'update_ledger.py')], check=True)
    ledger = json.loads((ROOT/'COMPUTE_LEDGER.json').read_text())
    active = [a for a in ledger['apps'] if a['state'] != 'stopped'
              and a['gpu_request'] != 'CPU']
    if len(active) < 2 and all(a['gpu_request'] != gpu for a in active):
        break
    if attempt < 6:
        print('Waiting for the preceding GPU app to close.', flush=True)
        time.sleep(5)
assert len(active) < 2 and all(a['gpu_request'] != gpu for a in active), active
if now >= WRITING_START:
    continuing = any((a.get('stage') or '').startswith(args.stage)
                     and datetime.datetime.fromisoformat(a['created_at']) < WRITING_START
                     for a in ledger['apps'])
    assert continuing or args.stage.startswith(('polish_', 'confirm_')), 'No new search during writing.'
hours_left = (COMPUTE_END-now).total_seconds()/3600
# Both slots running continuously until the absolute deadline must still fit.
assert ledger['total_gpu_app_wall_hours']+2*hours_left <= 18
assert ledger['estimated_resource_cost_upper_bound']+sum(RATES.values())*hours_left <= 60
dest = ROOT/'results'/f'{log_name}.log'
assert not dest.exists(), f'Preserve existing log: {dest}'
command = ['modal', 'run', '--detach', str(ROOT/'modal_runner.py'), '--stage', args.stage]
if args.fast:
    command.append('--fast')
print('Launching', args.stage, 'on', gpu, 'log', dest, flush=True)
with dest.open('w') as log:
    log.write('GPU_REQUEST '+gpu+'\n')
    log.flush()
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        remaining = max(1, (COMPUTE_END-datetime.datetime.now(datetime.timezone.utc)).total_seconds())
        code = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        log.flush()
        match = re.search(r'ap-[A-Za-z0-9]+', dest.read_text())
        if match:
            subprocess.run(['modal', 'app', 'stop', match.group(), '--yes'], check=True)
        process.wait(timeout=30)
        code = 124
print('Stage exit', code, 'log', dest, flush=True)
if code == 0:
    task = next(task for task in ['tenants_long', 'narrative_contracts'] if args.stage.endswith(task))
    for next_stage in args.then:
        follow = [sys.executable, str(ROOT/'launch_stage.py'), '--stage', next_stage+'_'+task]
        if args.fast:
            follow.append('--fast')
        subprocess.run(follow, check=True)
raise SystemExit(code)
