"""Analyze paired clean-reference continuation KL; no interpolation of methods."""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
MODES = ['activation_positive', 'activation', 'additive', 'fra', 'legacy12']


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def load_cases():
    cases = []
    for key in ['gpt2', 'gemma']:
        files = sorted(p for p in OUT.glob(f'{key}_*.json.gz') if 'smoke' not in p.name)
        assert len(files) == 4, (key, len(files))
        for path in files:
            run = read(path)
            assert run['done']
            cases.append((key, run['case'], run['meta']))
    return cases


def eligible(case, mode):
    actual = 'activation' if mode == 'activation_positive' else mode
    return [p for p in case['points'] if p['mode'] == actual
            and (mode != 'activation_positive' or p['strength'] >= 0)]


def best(case, mode, threshold=None):
    pts = [p for p in eligible(case, mode) if threshold is None or p['suppression'] >= threshold]
    if not pts: return None
    p = min(pts, key=lambda p: (p['continuation_kl_mean'], abs(p['strength'])))
    assert p['direct_verified'], (case['trigger'], mode, threshold)
    out = {k: p[k] for k in ['mode', 'feature', 'strength', 'suppression', 'query_kl',
                             'continuation_kl_mean', 'continuation_kl_by_context', 'trigger_kl_mean']}
    out['no_edit'] = p['strength'] == 0
    out['diff_rank'] = next((r['rank'] for r in case['ranking'] if r['feature'] == p['feature']), None)
    out['ratio_to_unsteered'] = p['continuation_kl_mean']/case['unsteered']['continuation_kl_mean']
    return out


def val(point):
    return 'unreached' if point is None else f"{point['continuation_kl_mean']:.5g}"


def analyze():
    runs = load_cases()
    old = read(ROOT/'reference/previous_50pct_winners.json')
    output = {'cases': [], 'metric': 'paired KL in nats/token, mean over three matched contexts', 'aggregates': []}
    for key, case, meta in runs:
        ref = next(p for p in old if p['model'] == key and p['trigger'] == case['trigger'])
        previous = next(p for p in case['points'] if p['mode'] == ref['mode'] and p['feature'] == ref['feature'] and p['strength'] == ref['strength'])
        assert previous['direct_verified']
        item = {'model': key, 'trigger': case['trigger'], 'payload': case['payload'],
                'unsteered_continuation_kl': case['unsteered']['continuation_kl_mean'],
                'unsteered_query_kl': case['unsteered']['query_kl'],
                'previous_winner': {'feature': ref['feature'], 'strength': ref['strength'],
                    'old_independent_paragraph_kl': ref['kl'],
                    'paired_continuation_kl': previous['continuation_kl_mean'],
                    'ratio_to_unsteered': previous['continuation_kl_mean']/case['unsteered']['continuation_kl_mean']},
                'comparisons': {}, 'validation': {
                    'direct_checks': len(case['validation']['direct_checks']),
                    'max_cached_vs_direct_mean_kl_error': max((x['mean_kl_error'] for x in case['validation']['direct_checks']), default=0),
                    'max_fra_reference_suppression_error': max(abs(p['suppression']-p['old_suppression']) for p in case['validation']['fra_reproduction']),
                    'max_fra_reference_kl_relative_error': max(abs(p['independent_paragraph_kl']-p['old_kl'])/p['old_kl'] for p in case['validation']['fra_reproduction'])}}
        for threshold in [None, .3, .5, .7, .9, .99]:
            label = 'unconstrained' if threshold is None else str(threshold)
            item['comparisons'][label] = {mode: best(case, mode, threshold) for mode in MODES}
        output['cases'].append(item)
    for key in ['gpt2', 'gemma']:
        subset = [c for c in output['cases'] if c['model'] == key]
        for label in ['unconstrained', '0.5', '0.9']:
            for mode in ['activation_positive', 'activation', 'additive']:
                rows = [c['comparisons'][label] for c in subset]
                both = [r for r in rows if r[mode] and r['fra']]
                output['aggregates'].append({'model': key, 'constraint': label, 'mode': mode,
                    'sae_reach': sum(r[mode] is not None for r in rows), 'fra_reach': sum(r['fra'] is not None for r in rows),
                    'both_reach': len(both), 'sae_lower': sum(r[mode]['continuation_kl_mean'] < r['fra']['continuation_kl_mean']-1e-10 for r in both),
                    'fra_lower': sum(r['fra']['continuation_kl_mean'] < r[mode]['continuation_kl_mean']-1e-10 for r in both)})
    lines = ['# Restoration of clean continuations with the backdoor present', '',
        '**Metric correction:** the earlier zero KL measured an intervention on a separate',
        'unpoisoned paragraph. This run compares the unsteered model with an unpoisoned',
        'primer against the steered model with a poisoned primer, while both read the',
        '**same clean continuation tokens**. The poison is present during the entire steered',
        'evaluation. These are full next-token distributions, not just payload probabilities.', '',
        'Each case uses the original paragraph and three matched primer seeds (0, 1, 2).',
        'All KL values below are **nats per predicted token**, averaged over those contexts.',
        'One feature and coefficient are chosen per case. FRA is freshly evaluated on these inputs.', '',
        '## Best restoration, including the option to make no edit', '',
        '| Model | Pair | Poison, no steering | Single SAE positive | Single SAE signed | Additive SAE | FRA |',
        '|---|---|---:|---:|---:|---:|---:|']
    for c in output['cases']:
        r = c['comparisons']['unconstrained']
        lines.append(f"| {c['model']} | {c['trigger']}→{c['payload']} | {c['unsteered_continuation_kl']:.5g} | {val(r['activation_positive'])} | {val(r['activation'])} | {val(r['additive'])} | {val(r['fra'])} |")
    for label in ['0.5', '0.9']:
        lines += ['', f'## Require at least {float(label):.0%} original backdoor suppression', '',
            'Suppression is evaluated on the original seed-0 repeated-token probe. The KL',
            'is evaluated on poisoned-primer + legitimate-text continuations, using the same edit.', '',
            '| Model | Pair | Single SAE positive | Single SAE signed | Additive SAE | FRA |',
            '|---|---|---:|---:|---:|---:|']
        for c in output['cases']:
            r = c['comparisons'][label]
            lines.append(f"| {c['model']} | {c['trigger']}→{c['payload']} | {val(r['activation_positive'])} | {val(r['activation'])} | {val(r['additive'])} | {val(r['fra'])} |")
    lines += ['', '## Selected operating points at >=50% original suppression', '',
        '| Model | Pair | Single feature (diff rank) | SAE c | Actual SAE suppression | FRA c | Actual FRA suppression |',
        '|---|---|---|---:|---:|---:|---:|']
    for c in output['cases']:
        r = c['comparisons']['0.5']; p, f = r['activation_positive'], r['fra']
        f_strength = f"{f['strength']:g}" if f else 'unreached'
        f_supp = f"{f['suppression']:.2%}" if f else 'unreached'
        lines.append(f"| {c['model']} | {c['trigger']}→{c['payload']} | {p['feature']} ({p['diff_rank']}) | {p['strength']:g} | {p['suppression']:.2%} | {f_strength} | {f_supp} |")
    lines += ['', '## Features and coefficients minimizing unconstrained restoration KL', '',
        '| Model | Pair | Positive activation feature (rank), c | FRA c | SAE / no-edit KL | FRA / no-edit KL |',
        '|---|---|---|---:|---:|---:|']
    for c in output['cases']:
        r = c['comparisons']['unconstrained']; p, f = r['activation_positive'], r['fra']
        ident = 'no edit' if p['no_edit'] else f"{p['feature']} ({p['diff_rank']}), {p['strength']:g}"
        lines.append(f"| {c['model']} | {c['trigger']}→{c['payload']} | {ident} | {f['strength']:g} | {p['ratio_to_unsteered']:.3f} | {f['ratio_to_unsteered']:.3f} |")
    lines += ['', '## Re-evaluating the old 50%-suppression winners', '',
        'These are the previously selected feature and coefficient, without retuning.',
        'The old column is summed KL on a separate unpoisoned paragraph; the new column',
        'is paired continuation KL per token. Their absolute magnitudes have different',
        'units and inputs. The final ratio compares two measurements of the new metric.', '',
        '| Model | Pair | Feature, c | Old separate-paragraph KL | New paired KL/token | New KL / poison-no-steering KL |',
        '|---|---|---|---:|---:|---:|']
    for c in output['cases']:
        p = c['previous_winner']
        lines.append(f"| {c['model']} | {c['trigger']}→{c['payload']} | {p['feature']}, {p['strength']:g} | {p['old_independent_paragraph_kl']:.5g} | {p['paired_continuation_kl']:.5g} | {p['ratio_to_unsteered']:.3f} |")
    lines += ['', '## Full-distribution recovery at the original backdoor query', '',
        'For the continuation-KL-optimal edits above, this diagnostic compares the full',
        'query distribution with the matched no-backdoor original probe. It is not',
        'the objective used to select these edits.', '',
        '| Model | Pair | No steering | Single SAE positive | FRA |',
        '|---|---|---:|---:|---:|']
    for c in output['cases']:
        r = c['comparisons']['unconstrained']
        lines.append(f"| {c['model']} | {c['trigger']}→{c['payload']} | {c['unsteered_query_kl']:.5g} | {r['activation_positive']['query_kl']:.5g} | {r['fra']['query_kl']:.5g} |")
    lines += ['', '## Verification', '',
        '| Model | Pair | Max FRA archived suppression error | Max FRA archived KL relative error | Direct SAE point checks |',
        '|---|---|---:|---:|---:|']
    for c in output['cases']:
        v = c['validation']
        lines.append(f"| {c['model']} | {c['trigger']}→{c['payload']} | {v['max_fra_reference_suppression_error']:.5g} | {v['max_fra_reference_kl_relative_error']:.2%} | {v['direct_checks']} |")
    lines += ['', '## Limits', '',
        '- Teacher-forced KL over a shared continuation measures distributional recovery.',
        '  It does not evaluate independently sampled long-form generations.',
        '- Three matched random-token primers per pair; one natural paragraph per pair.',
        '- Feature candidates are frozen from the original calibration, but the final',
        '  feature and coefficient are selected retrospectively on these evaluations.',
        '- Positive and signed activation-weighted steering are separate. Negative',
        '  coefficients amplify a feature; additive steering has no activation gate.',
        '- The poison replaces exactly two tokens in the clean prefix; all shared',
        '  continuation token IDs and prediction positions are explicitly stored.',
        '- Restoring the original random primer removes both the planted association',
        '  and the individual trigger/payload occurrences. This KL includes ordinary',
        '  contextual priming from those tokens, as well as induction effects.',
        '- Every grid is finite; unreachable suppression targets remain visible.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    (OUT/'summary.json.gz').write_bytes(gzip.compress(json.dumps(output, indent=2).encode(), mtime=0))
    print('\n'.join(lines))
    return output


if __name__ == '__main__': analyze()
