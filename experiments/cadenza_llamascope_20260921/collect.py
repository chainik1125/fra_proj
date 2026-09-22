"""Collect compact, audited results; full prompt and generation logs stay remote."""
import argparse
import hashlib
import lzma
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CAMPAIGN = '/data/users/dmitry/sae-middle/campaigns/A-scope-overnight-20260921'


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(campaign):
    status = read(campaign / 'status.json')
    plan = read(campaign / 'plan.json')
    tasks = {t['id']:t for t in plan['tasks']}
    result = {'collected':time.time(), 'status':status, 'plan':plan, 'runs':[], 'audit_errors':[]}
    common_keys = None
    for job in status['jobs']:
        if job['state'] != 'complete':
            continue
        run = Path(job['run'])
        summary = read(run / 'summary.json')
        task = tasks[job['id']]
        entry = {'id':job['id'], 'run':str(run), 'task':task, 'summary':summary,
                 'source_manifest':read(run/'source_manifest.json')}
        for name in ('sae_source', 'plumbing', 'split_manifest'):
            if (run/f'{name}.json').exists():
                entry[name] = read(run/f'{name}.json')
        checks = {'worker_complete':read(run/'status.json')['state']=='complete',
                  'summary_complete':summary['state']=='complete'}
        source = entry['source_manifest']
        checks['immutable_source_hashes'] = all(digest(run/'src'/n)==h for n,h in source.items())
        if summary.get('confirmation'):
            selected = read(run/'selected.json')
            checks['selection_hash_unchanged'] = digest(run/'selected.json') == summary['selected_sha256_before_test']
            keys = summary['confirmation']['keys']
            common_keys = keys if common_keys is None else common_keys
            checks['same_64_fresh_keys'] = keys==common_keys and len(set(keys))==64
            split = entry['split_manifest']
            used = {k for v in split['splits'].values() for k in v} | set(split['legacy_confirmation'])
            checks['fresh_disjoint'] = not set(keys)&used
            entry['rows'] = {}
            for rule in selected:
                d = read(run/f'confirmation_{rule}.json')
                rows = d['rows']
                checks[f'{rule}_rows_match'] = [r['key'] for r in rows]==keys
                checks[f'{rule}_selection_match'] = d['candidate']==selected[rule]['candidate'] and d['alpha']==selected[rule]['alpha']
                entry['rows'][rule] = [{k:v for k,v in r.items() if k=='key' or isinstance(v,(int,float,bool))} for r in rows]
                entry['rows'][rule] = [{**r, 'blank_triggered':not original['steered_triggered_text'].strip(),
                    'blank_clean':not original['steered_clean_text'].strip()} for r,original in zip(entry['rows'][rule],rows)]
            if task['kind']=='sweep':
                grid = [json.loads(s) for s in (run/'validation.jsonl').read_text().splitlines()]
                checks['750_grid_records'] = len(grid)==750
                checks['50_distinct_candidates'] = len({tuple(r['candidate']['features']) for r in grid})==50
                sys.path.insert(0,str(run/'src/lib'))
                from single_eval import choose_single_settings
                checks['validation_selection_reproduced'] = selected==choose_single_settings(grid)
                entry['selected_validation'] = selected
        entry['audit'] = checks
        if not all(checks.values()):
            result['audit_errors'].append({'id':job['id'],'failed':[k for k,v in checks.items() if not v]})
        result['runs'].append(entry)
    for name in ('loader_audit','audit_allocation'):
        if (campaign/f'{name}.json').exists():
            result[name] = read(campaign/f'{name}.json')
    training = Path(plan['training_run'])
    result['training'] = {'path':str(training),'status':read(training/'status.json')}
    if (training/'summary.json').exists():
        result['training']['summary'] = read(training/'summary.json')
    if (training/'progress.json').exists():
        result['training']['progress'] = read(training/'progress.json')
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--remote',action='store_true'); a=p.parse_args()
    if a.remote:
        print(json.dumps(collect(Path(CAMPAIGN)),allow_nan=False))
    else:
        command=shlex.join(['/data/users/dmitry/sae-middle/venv/bin/python',CAMPAIGN+'/src/collect.py','--remote'])
        raw=subprocess.check_output(['ssh','simplex1',command])
        assert len(raw)<10_000_000
        result=json.loads(raw)
        (HERE/'results.json.xz').write_bytes(lzma.compress(raw))
        print(json.dumps({'complete':len(result['runs']),'total':len(result['plan']['tasks']),
                          'audit_errors':result['audit_errors'],'bytes':len(raw)}))


if __name__=='__main__':
    main()
