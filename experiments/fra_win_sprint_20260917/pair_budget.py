"""Tuning-selected FRA comparisons at fixed pair-count budgets.

All budgets and thresholds are specified before primary confirmation. This is a
secondary capacity curve; it does not replace the primary frozen comparison.
"""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
from transformer_lens import HookedTransformer
from common import LAST, render, metric, summarize, atomic
from data import LABELS, suite as base_suite
from variants import TASKS, suite as variant_suite
from operators import Operators, SAE_SPECS
from confirm import edited, expanded_pairs
from analyze import cfgkey
from retention import agreement

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'reference'))
from sae_lens_wrapper import GemmaScopeSAE
BUDGETS = [1, 4, 16, 48, 144]
THRESHOLDS = [.5, .9]


def choose(points):
    selected = []
    for family in ['fra', 'fra_distinct']:
        for budget in BUDGETS:
            for threshold in THRESHOLDS:
                eligible = [p for p in points if p['valid'] and p['pair_count'] <= budget
                            and (family == 'fra' or p['all_distinct'])
                            and p['summary']['controls']['correct'] >= .95
                            and p['summary']['suppression'] >= threshold]
                p = min(eligible, key=lambda p: (p['summary']['all']['kl'], p['pair_count'],
                        abs(p['config']['strength']), cfgkey(p['config']))) if eligible else None
                selected.append({'family': family, 'budget': budget, 'threshold': threshold,
                    'source_index': p['source_index'] if p else None,
                    'config': p['config'] if p else None, 'tuning': p['summary'] if p else None,
                    'pair_count': p['pair_count'] if p else None})
    return selected


@torch.inference_mode()
def run(out_dir, commit=None, task='tenants_long'):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.time()
    out_dir = Path(out_dir)
    primary = json.loads((out_dir / f'frozen_{task}.json').read_text())
    names = [f'{stage}_{task}' for stage in ['search', 'refine', 'learn_pairs']]
    sources = [json.loads((out_dir / f'{name}.json').read_text()) for name in names]
    assert all(s['done'] for s in sources)
    points = []
    for si, source in enumerate(sources):
        for p in source['points']:
            if not p['config']['method'].startswith('fra'):
                continue
            pairs = expanded_pairs(source, p['config'])
            points.append({**p, 'source_index': si, 'pair_count': len(pairs),
                           'all_distinct': all(p['q'] != p['k'] for p in pairs)})
    result = {'done': False, 'meta': {'task': task, 'description': __doc__,
        'budgets': BUDGETS, 'thresholds': THRESHOLDS,
        'primary_frozen_at_unix': primary['frozen_at_unix'],
        'source_sha256': {name: hashlib.sha256((out_dir/f'{name}.json').read_bytes()).hexdigest() for name in names},
        'sources': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                    ['pair_budget.py', 'confirm.py', 'operators.py', 'common.py', 'data.py', 'variants.py', 'retention.py']}},
        'selected': [], 'tuning_replays': [], 'validation': [], 'rows': [], 'points': [],
        'baseline': {'rows': []}, 'clean': {'rows': []}}
    def save():
        result['seconds'] = time.time()-start
        atomic(out_dir/f'pair_budget_{task}.json', result)
        if commit:
            commit()
    model = HookedTransformer.from_pretrained('gemma-2-9b-it', device='cuda', dtype=torch.float16)
    model.eval()
    tok = model.tokenizer
    saes = {l: GemmaScopeSAE(*SAE_SPECS[l], normalize_activations=False) for l in SAE_SPECS}
    op = Operators(model, saes)
    label_ids = [tok.encode(' '+s, add_special_tokens=False)[0] for s in LABELS]
    pid = label_ids[1]
    suite = variant_suite if task in TASKS else base_suite
    def prepare(split):
        rows = suite(task, split)
        items = []
        for i in range(0, len(rows), 2):
            cr, pr = render(tok, rows[i]), render(tok, rows[i+1])
            ct = torch.tensor([cr['token_ids']], device='cuda')
            pt = torch.tensor([pr['token_ids']], device='cuda')
            cl = model.run_with_hooks(ct, fwd_hooks=[LAST])[0, -1]
            pl = model.run_with_hooks(pt, fwd_hooks=[LAST])[0, -1]
            ref = cl.double().log_softmax(-1)
            pp = float(pl.float().softmax(-1)[pid])
            items.append({'tokens': pt, 'row': pr, 'ref': ref, 'poison_p': pp})
            if split == 'confirmation':
                result['rows'].append({**pr, 'clean_token_ids': cr['token_ids']})
                result['clean']['rows'].append(metric(tok, label_ids, cl, ref, pr, pp))
                result['baseline']['rows'].append(metric(tok, label_ids, pl, ref, pr, pp))
        return items
    def measure(p, items):
        rows = []
        for item in items:
            ll = edited(op, item['tokens'], sources[p['source_index']], p['config'], item['row'])
            rows.append(metric(tok, label_ids, ll, item['ref'], item['row'], item['poison_p']))
        valid = not any(r.get('invalid') for r in rows)
        return {**p, 'rows': rows, 'valid': valid,
                'summary': summarize(rows) if valid else None, 'direct': True}
    tuning = prepare('tuning')
    checked = set()
    for _ in range(12):
        selected = choose(points)
        pending = {(s['source_index'], cfgkey(s['config'])) for s in selected if s['config']} - checked
        if not pending:
            break
        for si, key in sorted(pending):
            j = next(j for j,p in enumerate(points) if p['source_index'] == si and cfgkey(p['config']) == key)
            old = points[j]
            new = measure(old, tuning)
            if old['valid'] and new['valid']:
                err = abs(old['summary']['all']['kl']-new['summary']['all']['kl'])
                perr = max(abs(a['target_p']-b['target_p']) for a,b in zip(old['rows'],new['rows']))
                assert err < .03 and perr < .03
                result['validation'].append({'source_index': si, 'config': old['config'],
                    'mean_kl_error': err, 'max_target_p_error': perr})
            points[j] = new
            result['tuning_replays'].append(new)
            checked.add((si,key))
    else:
        raise RuntimeError('Pair-budget selection did not stabilize')
    result['selected'] = choose(points)
    result['selection_frozen_at_unix'] = time.time()
    result['selection_frozen_before_confirmation'] = True
    save()
    print('PAIR BUDGET FROZEN', len(checked), 'settings', flush=True)
    del tuning, points
    gc.collect()
    torch.cuda.empty_cache()
    confirmation = prepare('confirmation')
    seen = set()
    for s in result['selected']:
        key = (s['source_index'], cfgkey(s['config']))
        if not s['config'] or key in seen:
            continue
        p = measure({'source_index': s['source_index'], 'config': s['config'], 'pair_count': s['pair_count']}, confirmation)
        if p['valid']:
            p['clean_reference_agreement'] = agreement(p['rows'], result['clean']['rows'])
        result['points'].append(p)
        seen.add(key)
        print('PAIR BUDGET CONFIRMATION', len(seen), 'settings', flush=True)
        save()
    for p in [result['baseline'], result['clean']]:
        p['summary'] = summarize(p['rows'])
    result['done'] = True
    save()
    return result
