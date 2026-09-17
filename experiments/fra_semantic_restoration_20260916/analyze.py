"""Regenerate the semantic-restoration report from compressed measurement shards."""
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
CONCEPTS = ['vessel', 'vehicle', 'bird', 'fire', 'war', 'medical']
MODES = ['activation_positive', 'activation', 'additive', 'fra']
THRESHOLDS = [.3, .5, .7, .9]


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def load(concept):
    shards = [read(p) for p in sorted(OUT.glob(f'{concept}.part*.json.gz'))]
    assert shards and all(s['done'] for s in shards), concept
    result = dict(shards[0]); result['points'] = []
    for s in shards:
        assert s['point_offset'] == len(result['points'])
        assert s['meta'] == result['meta']
        result['points'].extend(s['points'])
    assert len(result['points']) == result['total_points']
    for ctx in result['contexts']:
        suffix = ctx['shared_continuation_token_ids']
        for side in ['clean', 'poisoned']:
            tokens, positions = ctx[side+'_token_ids'], ctx[side+'_prediction_positions']
            assert [tokens[p+1] for p in positions] == suffix
    zero = result['unsteered']
    for p in result['points']:
        assert all(v >= 0 and math.isfinite(v) for v in p['continuation_kl_by_context']+p['query_kl'])
        assert abs(mean(p['continuation_kl_by_context'])-p['continuation_kl_mean']) < 1e-12
        if p['strength'] == 0:
            assert p['continuation_kl_mean'] == zero['continuation_kl_mean']
    return result


def candidates(result, mode):
    return [p for p in result['points']
            if p['mode'] == ('activation' if mode == 'activation_positive' else mode)
            and (mode != 'activation_positive' or p['strength'] >= 0)]


def compact(result, p):
    if p is None: return None
    assert p['direct_verified'], (result['meta']['concept'], p['mode'], p['feature'], p['strength'])
    return {k: p[k] for k in ['mode', 'feature', 'strength', 'continuation_kl_mean',
                             'continuation_kl_by_context', 'suppression', 'query_kl']} | {
        'rank': next((r['rank'] for r in result['ranking'] if r['feature'] == p['feature']), None)}


def fixed(result, mode, threshold=None):
    synonyms = [i for i, r in enumerate(result['queries']) if r['kind'] == 'synonym']
    choices = candidates(result, mode)
    def key(p):
        recovery = (p['continuation_kl_mean'], abs(p['strength']))
        return recovery if threshold is None else (-sum(p['suppression'][i] >= threshold for i in synonyms),)+recovery
    p = min(choices, key=key)
    return compact(result, p) | {'reach': None if threshold is None else sum(p['suppression'][i] >= threshold for i in synonyms),
        'total': len(synonyms), 'related_query_kl_mean': mean(p['query_kl'][i] for i in synonyms)}


def per_query(result, mode, qi, threshold=None, objective='continuation'):
    ci = result['queries'][qi]['context_index']
    choices = [p for p in candidates(result, mode) if threshold is None or p['suppression'][qi] >= threshold]
    if not choices: return None
    value = lambda p: p['continuation_kl_by_context'][ci] if objective == 'continuation' else p['query_kl'][qi]
    p = min(choices, key=lambda p: (value(p), abs(p['strength'])))
    return compact(result, p) | {'value': value(p), 'actual_suppression': p['suppression'][qi]}


def analyze():
    runs = [load(c) for c in CONCEPTS]
    assert sum(len(r['queries']) for r in runs) == 70
    assert sum(q['kind'] == 'synonym' for r in runs for q in r['queries']) == 52
    output = {'metric': 'KL(clean no-payload reference || poisoned + steering), nats/token',
              'concepts': [], 'comparisons': []}
    for r in runs:
        rep = r['validation']['fra_reproduction']; checks = r['validation']['direct_checks']
        concept = {'concept': r['meta']['concept'], 'baseline': r['unsteered']['continuation_kl_mean'],
            'related_query_baseline': mean(q['unsteered_query_kl'] for q in r['queries'] if q['kind'] == 'synonym'),
            'unconstrained': {m: fixed(r, m) for m in MODES},
            'fixed': {str(t): {m: fixed(r, m, t) for m in MODES} for t in THRESHOLDS},
            'previous_winners': [], 'validation': {
                'max_fra_suppression_error': max(abs(p['suppression']-p['old_suppression']) for p in rep),
                'max_fra_kl_relative_error': max(abs(p['independent_paragraph_kl']-p['old_kl'])/p['old_kl'] for p in rep),
                'direct_configurations': len(checks),
                'direct_input_forwards': len(checks)*(len(r['queries'])+len(r['contexts'])),
                'max_cached_full_mean_kl_error': max((p['mean_kl_error'] for p in checks), default=0),
                'max_cached_full_probability_error': max((p['max_probability_error'] for p in checks), default=0)}}
        for old in r['previous_winners']:
            qi = next(i for i, q in enumerate(r['queries']) if q['ctx'] == old['ctx'] and q['word'] == old['word'])
            ci = r['queries'][qi]['context_index']; cfg = old['config']
            p = next(p for p in r['points'] if all(p[k] == cfg[k] for k in ['mode', 'feature', 'strength']))
            assert p['direct_verified']
            baseline = r['contexts'][ci]['unsteered_mean_kl']
            concept['previous_winners'].append({**old, 'paired_kl': p['continuation_kl_by_context'][ci],
                'paired_ratio_to_unsteered': p['continuation_kl_by_context'][ci]/baseline})
        output['concepts'].append(concept)
    for kind in ['synonym', 'planted']:
        for objective in ['continuation', 'query']:
            for threshold in [None]+THRESHOLDS:
                for mode in MODES[:-1]:
                    rows = []
                    for r in runs:
                        for qi, q in enumerate(r['queries']):
                            if q['kind'] != kind: continue
                            a = per_query(r, mode, qi, threshold, objective)
                            b = per_query(r, 'fra', qi, threshold, objective)
                            rows.append({'concept': q['concept'], 'word': q['word'], 'ctx': q['ctx'], 'query_index': qi,
                                'query_baseline': q['unsteered_query_kl'], 'sae': a, 'fra': b,
                                'baseline': r['contexts'][q['context_index']]['unsteered_mean_kl'] if objective == 'continuation' else q['unsteered_query_kl']})
                    both = [row for row in rows if row['sae'] and row['fra']]
                    output['comparisons'].append({'kind': kind, 'objective': objective, 'threshold': threshold, 'mode': mode,
                        'total': len(rows), 'sae_reach': sum(row['sae'] is not None for row in rows),
                        'fra_reach': sum(row['fra'] is not None for row in rows), 'both': len(both),
                        'sae_lower': sum(row['sae']['value'] < row['fra']['value']-1e-10 for row in both),
                        'fra_lower': sum(row['fra']['value'] < row['sae']['value']-1e-10 for row in both),
                        'sae_mean': mean(row['sae']['value'] for row in both) if both else None,
                        'fra_mean': mean(row['fra']['value'] for row in both) if both else None,
                        'baseline_mean': mean(row['baseline'] for row in both) if both else None,
                        'rows': rows})
    lines = ['# Semantic filtering: recovery with the poisoned context present', '',
        'Gemma-2-2b; six original concepts, three original contexts each, all 52 related-word',
        'and 18 planted-word queries. Frozen top ten planted-minus-no-payload SAE features.',
        'FRA is freshly measured with the exact original 25 heads and 48 pairs per head.', '',
        '**Metric:** KL(clean no-payload reference || poisoned context + steering), in',
        '**nats/token**, on the same teacher-forced legitimate paragraph. Both sides keep',
        'the same filler and trigger; only the planted payload is omitted in the reference.',
        'The backdoor remains present throughout the steered input, including prefill.', '',
        '## Best continuation recovery, one fixed edit per concept', '',
        'Minimize mean KL across all three contexts, allowing coefficient zero.', '',
        '| Concept | Poison, no edit | Positive single SAE | Signed single SAE | Additive SAE | FRA |',
        '|---|---:|---:|---:|---:|---:|']
    for c in output['concepts']:
        lines.append('| '+c['concept']+' | '+f"{c['baseline']:.6g}"+' | '+' | '.join(f"{c['unconstrained'][m]['continuation_kl_mean']:.6g}" for m in MODES)+' |')
    lines += ['', '## One fixed feature and coefficient per concept at 50% suppression', '',
        'First maximize the number of original related-word queries reaching 50%; then',
        'minimize mean continuation KL across the three contexts. Reach is shown explicitly.', '',
        '| Concept | SAE feature (rank), c | SAE reach | SAE KL | FRA c | FRA reach | FRA KL |',
        '|---|---|---:|---:|---:|---:|---:|']
    for c in output['concepts']:
        a, b = c['fixed']['0.5']['activation_positive'], c['fixed']['0.5']['fra']
        lines.append(f"| {c['concept']} | {a['feature']} ({a['rank']}), {a['strength']} | {a['reach']}/{a['total']} | {a['continuation_kl_mean']:.6g} | {b['strength']} | {b['reach']}/{b['total']} | {b['continuation_kl_mean']:.6g} |")
    lines += ['', '### Recall-distribution recovery for those same fixed edits', '',
        'No reselection: evaluate the exact feature/coefficient chosen in the preceding table.',
        'Mean full-distribution KL at all related-word queries, including any unreached queries.', '',
        '| Concept | Poison, no edit | Same SAE edit | Same FRA edit |',
        '|---|---:|---:|---:|']
    for c in output['concepts']:
        a, b = c['fixed']['0.5']['activation_positive'], c['fixed']['0.5']['fra']
        lines.append(f"| {c['concept']} | {c['related_query_baseline']:.6g} | {a['related_query_kl_mean']:.6g} | {b['related_query_kl_mean']:.6g} |")
    lines += ['', '### Higher suppression with one fixed edit', '',
        'Again maximize coverage first, then minimize continuation KL. When reach differs,',
        'the two KL values correspond to different achieved coverage. The fixed-edit result',
        'at 50% should not be generalized to all suppression thresholds.', '',
        '| Required suppression | Concept | SAE reach | SAE KL | FRA reach | FRA KL |',
        '|---|---|---:|---:|---:|---:|']
    for t in [.7, .9]:
        for c in output['concepts']:
            a, b = c['fixed'][str(t)]['activation_positive'], c['fixed'][str(t)]['fra']
            lines.append(f"| {t:.0%} | {c['concept']} | {a['reach']}/{a['total']} | {a['continuation_kl_mean']:.6g} | {b['reach']}/{b['total']} | {b['continuation_kl_mean']:.6g} |")
    for objective, heading in [('continuation', 'Continuation recovery subject to suppression: best per query'),
                               ('query', 'Full-distribution recovery at the related-word recall query')]:
        lines += ['', '## '+heading, '',
            'Select one feature and coefficient retrospectively for each query and this objective.',
            'The two objective tables can therefore select different edits. Means below',
            'use only queries both methods reach; reach denominators include all 52 queries.', '',
            '| SAE mode | Required suppression | SAE reach | FRA reach | SAE lower / comparable | No-edit KL | SAE KL | FRA KL |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for c in output['comparisons']:
            if c['kind'] != 'synonym' or c['objective'] != objective: continue
            threshold = 'none' if c['threshold'] is None else f"{100*c['threshold']:.0f}%"
            fmt = lambda value: '—' if value is None else f'{value:.6g}'
            lines.append(f"| {c['mode']} | {threshold} | {c['sae_reach']}/{c['total']} | {c['fra_reach']}/{c['total']} | {c['sae_lower']}/{c['both']} | {fmt(c['baseline_mean'])} | {fmt(c['sae_mean'])} | {fmt(c['fra_mean'])} |")
    lines += ['', '## Per-concept continuation comparison at 50% suppression', '',
        'Positive activation-weighted single-feature steering; retrospective per-query choices.', '',
        '| Concept | SAE reach | FRA reach | SAE lower / comparable | No-edit KL | SAE KL | FRA KL |',
        '|---|---:|---:|---:|---:|---:|---:|']
    comp = next(c for c in output['comparisons'] if c['kind'] == 'synonym' and c['objective'] == 'continuation' and c['threshold'] == .5 and c['mode'] == 'activation_positive')
    for name in CONCEPTS:
        rows = [r for r in comp['rows'] if r['concept'] == name]; both = [r for r in rows if r['sae'] and r['fra']]
        lines.append(f"| {name} | {sum(r['sae'] is not None for r in rows)}/{len(rows)} | {sum(r['fra'] is not None for r in rows)}/{len(rows)} | {sum(r['sae']['value'] < r['fra']['value']-1e-10 for r in both)}/{len(both)} | {mean(r['baseline'] for r in both):.6g} | {mean(r['sae']['value'] for r in both):.6g} | {mean(r['fra']['value'] for r in both):.6g} |")
    lines += ['', '## Re-evaluating old-metric zero-collateral optima', '',
        'Choose a minimum-KL point achieving 50% suppression in the archived sweep, breaking',
        'ties by smallest absolute coefficient across features. This can choose a different',
        'tied optimum from the old report, which preferred feature order before coefficient.',
        'Apply that edit unchanged. The old metric used a separate unpoisoned paragraph;',
        'ratios below compare corrected paired KL to its no-edit baseline.', '',
        '| Concept | Old zero-KL winners / related queries | Mean paired KL | Mean paired / no-edit KL | No improvement in paired KL |',
        '|---|---:|---:|---:|---:|']
    for c in output['concepts']:
        rows = [r for r in c['previous_winners'] if r['kind'] == 'synonym']
        lines.append(f"| {c['concept']} | {sum(r['old_kl'] == 0 for r in rows)}/{len(rows)} | {mean(r['paired_kl'] for r in rows):.6g} | {mean(r['paired_ratio_to_unsteered'] for r in rows):.4f} | {sum(r['paired_ratio_to_unsteered'] >= 1-1e-10 for r in rows)}/{len(rows)} |")
    lines += ['', '## Verification', '',
        '| Concept | Max archived FRA suppression difference | Max archived FRA KL relative difference | SAE configs checked with full forwards | Max cached/full mean KL difference |',
        '|---|---:|---:|---:|---:|']
    for c in output['concepts']:
        v = c['validation']
        lines.append(f"| {c['concept']} | {v['max_fra_suppression_error']:.6g} | {v['max_fra_kl_relative_error']:.2%} | {v['direct_configurations']} | {v['max_cached_full_mean_kl_error']:.6g} |")
    output['minimum_measured_continuation_kl'] = min(p['continuation_kl_mean'] for r in runs for p in r['points'])
    lines += ['', f"Minimum mean continuation KL anywhere in the measured grid: **{output['minimum_measured_continuation_kl']:.8g}**.", '',
        '## Interpretation limits', '',
        '- Three original contexts and one original paragraph per concept; one model.',
        '- Feature identities come from original planted-word calibration, but final feature/strength',
        '  selection is retrospective. Per-query choices do not establish one transferable fixed edit.',
        '- The reference omits the payload, so its prefix is shorter. Shared continuation tokens',
        '  and their prediction positions are explicitly aligned. KL includes payload priming and',
        '  positional/context changes as well as the planted association.',
        '- Teacher-forced distribution recovery; no independently sampled generation evaluation.',
        '- Lower KL need not imply enough suppression; both unconstrained recovery and constrained',
        '  comparisons are shown. Unreached thresholds remain visible. All grids are finite.',
        '- FP16 GPU execution can differ from the archived run. All method comparisons above use',
        '  freshly measured values in the same environment; reproduction differences are reported.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    blob = json.dumps(output, indent=2, allow_nan=False).encode()
    (OUT/'summary.json.gz').write_bytes(gzip.compress(blob, compresslevel=9, mtime=0))
    manifest = ['# Measurement archive checksums', '', '| File | Compressed bytes | Original JSON SHA-256 |', '|---|---:|---|']
    for path in sorted(OUT.glob('*.json.gz')):
        data = path.read_bytes(); assert len(data) < 1_000_000
        manifest.append(f'| {path.name} | {len(data)} | {hashlib.sha256(gzip.decompress(data)).hexdigest()} |')
    (OUT/'ARCHIVE_MANIFEST.md').write_text('\n'.join(manifest)+'\n')
    print('Verified 6 concepts, 70 queries, and aligned continuation inputs; wrote report and summary.')
    return output


if __name__ == '__main__': analyze()
