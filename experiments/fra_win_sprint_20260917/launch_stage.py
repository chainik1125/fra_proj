"""Launch one authorized sprint stage with budget checks and a local log."""
import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPUTE_END = datetime.datetime(2026, 9, 18, 2, 16, 27, tzinfo=datetime.timezone.utc)
RATES = {'A100-80GB': 3.198528, 'H100!': 4.649328}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--stage', required=True)
parser.add_argument('--fast', action='store_true')
parser.add_argument('--log-name')
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
assert now < COMPUTE_END, 'The final writing hour has begun.'
subprocess.run([sys.executable, str(ROOT/'update_ledger.py')], check=True)
ledger = json.loads((ROOT/'COMPUTE_LEDGER.json').read_text())
active = [a for a in ledger['apps'] if a['state'] != 'stopped'
          and a['gpu_request'] != 'CPU']
gpu = 'H100!' if args.fast else 'A100-80GB'
assert len(active) < 2 and all(a['gpu_request'] != gpu for a in active), active
hours_left = (COMPUTE_END-now).total_seconds()/3600
# Both slots running continuously until the writing hour must still fit.
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
    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
print('Stage exit', code, 'log', dest, flush=True)
raise SystemExit(code)
