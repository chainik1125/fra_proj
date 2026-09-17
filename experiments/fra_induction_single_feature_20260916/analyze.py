"""Summarize measured single-feature operating points against archived FRA curves."""
from collections import defaultdict
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
THRESHOLDS = [.3, .5, .7, .9, .99]
MODES = ['activation_positive', 'activation', 'additive', 'legacy12']


def read_json(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())


def candidates(case, mode):
    actual = 'activation' if mode == 'activation_positive' else mode
    return [p for p in case['points'] if p['mode'] == actual
            and (mode != 'activation_positive' or p['strength'] >= 0)]


def at(points, threshold, interpolate=False):
    eligible = [{**p, 'estimated': False} for p in points if p['suppression'] >= threshold]
    if interpolate:
        # Feature identity is fixed before interpolation. Preserve adjacency in
        # steering coefficient so a non-monotonic curve cannot bridge gaps.
        groups = defaultdict(list)
        for p in points:
            groups[(p.get('feature'), p.get('mode'))].append(p)
        for group in groups.values():
            group = sorted(group, key=lambda p: p['strength'])
            for a, b in zip(group, group[1:]):
                if a['suppression'] == b['suppression']:
                    continue
                f = (threshold-a['suppression'])/(b['suppression']-a['suppression'])
                if 0 <= f <= 1:
                    eligible.append({'feature': a.get('feature'), 'mode': a.get('mode'),
                                     'strength': a['strength']+f*(b['strength']-a['strength']),
                                     'suppression': threshold, 'kl': a['kl']+f*(b['kl']-a['kl']),
                                     'estimated': True})
    return min(eligible, key=lambda p: (p['kl'], abs(p['strength']))) if eligible else None


def fra_points(model_key, case):
    grid = [1, 2, 4, 8, 16] if model_key == 'gpt2' else [1, 2, 4, 8, 16, 32]
    return [{'strength': 0, 'suppression': 0, 'kl': 0, 'mode': 'fra'}] + [
        {'strength': c, 'suppression': s, 'kl': k, 'mode': 'fra'}
        for c, (s, k) in zip(grid, case['reference']['fra'])]


def fmt(value):
    return 'unreached' if value is None else f'{value:.6g}'


def analyze():
    runs = {key: read_json(OUT/f'{key}.json.gz') for key in ['gpt2', 'gemma']}
    assert all(run['done'] and len(run['cases']) == 4 for run in runs.values())
    summary = {'cases': [], 'aggregates': []}
    for key, run in runs.items():
        for case in run['cases']:
            item = {'model': key, 'trigger': case['trigger'], 'payload': case['payload'],
                    'comparisons': {}, 'validation': {
                        'base_error': abs(case['base']-case['reference_base']),
                        'max_legacy_suppression_error': max(abs(p['suppression_error']) for p in case['validation']['legacy_checks']),
                        'max_legacy_kl_relative_error': max(abs(p['kl_error'])/p['reference_kl'] for p in case['validation']['legacy_checks']),
                        'direct_checks': len(case['validation']['direct_checks']),
                        'max_direct_probability_error': max(p['probability_error'] for p in case['validation']['direct_checks']),
                        'max_direct_kl_error': max(p['kl_error'] for p in case['validation']['direct_checks']),
                    }}
            ranks = {r['feature']: r['rank'] for r in case['ranking']}
            for estimator in ['measured', 'interpolated']:
                comparisons = item['comparisons'][estimator] = {}
                for threshold in THRESHOLDS:
                    row = {'fra': at(fra_points(key, case), threshold, estimator == 'interpolated')}
                    for mode in MODES:
                        best = at(candidates(case, mode), threshold, estimator == 'interpolated')
                        if best:
                            best['rank'] = ranks.get(best.get('feature'))
                            if estimator == 'measured':
                                assert best.get('direct_verified'), (key, case['trigger'], mode, threshold)
                        row[mode] = best
                    comparisons[str(threshold)] = row
            summary['cases'].append(item)
        for estimator in ['measured', 'interpolated']:
            for threshold in THRESHOLDS:
                for mode in MODES:
                    rows = [c['comparisons'][estimator][str(threshold)] for c in summary['cases'] if c['model'] == key]
                    both = [r for r in rows if r['fra'] and r[mode]]
                    summary['aggregates'].append({'model': key, 'estimator': estimator, 'threshold': threshold, 'mode': mode,
                        'sae_reach': sum(r[mode] is not None for r in rows),
                        'fra_reach': sum(r['fra'] is not None for r in rows),
                        'both_reach': len(both), 'sae_lower': sum(r[mode]['kl'] < r['fra']['kl'] for r in both),
                        'zero_kl': sum(r[mode] is not None and r[mode]['kl'] == 0 for r in rows)})

    headline = [r for r in summary['aggregates'] if r['estimator'] == 'measured'
                and r['mode'] == 'activation_positive' and r['threshold'] == .5]
    lines = ['# Original induction backdoors: best-of-ten single-feature SAE baseline', '',
             '> **Metric correction:** this report measures steering on a separate unpoisoned',
             '> paragraph. Zero KL establishes inactivity there, not recovery of clean behavior',
             '> in a poisoned context. See the [paired continuation evaluation](../../fra_induction_restoration_20260916/results/REPORT.md)',
             '> for clean-reference versus poisoned-and-steered KL.', '',
             '**Finding at 50% suppression:** ' + '; '.join(
                 f"{r['model']}: single SAE reaches {r['sae_reach']}/4, has lower KL than FRA on {r['sae_lower']}/{r['both_reach']} jointly reachable cases, and zero KL on {r['zero_kl']}/4"
                 for r in headline) + '.', '',
             'This is the original repeated-token IC4/G4 benchmark, including bank→river and king→crown.',
             'Ten features per case are ranked by the original ON-minus-OFF activation difference;',
             'each is steered separately. Every measured winner below was checked with ordinary',
             'full forward passes at batch size one on both target and legitimate text.', '',
             'FRA curves are the archived author runs. The original top-12 SAE baseline is rerun',
             'as a comparability check. Collateral is summed KL in nats over one legitimate paragraph.', '',
             '## Primary comparison: positive activation-weighted removal', '',
             'Select the lowest measured KL achieving at least the stated suppression. The feature',
             'and coefficient are selected retrospectively per case and threshold. `c=1` is ordinary',
             'feature ablation; larger coefficients subtract more than its encoded contribution.', '',
             '| Model | Suppression | SAE reach | FRA reach | SAE lower / both reach | SAE zero KL |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summary['aggregates']:
        if row['estimator'] == 'measured' and row['mode'] == 'activation_positive':
            lines.append(f"| {row['model']} | {row['threshold']:.0%} | {row['sae_reach']}/4 | {row['fra_reach']}/4 | {row['sae_lower']}/{row['both_reach']} | {row['zero_kl']}/4 |")
    for threshold in [.5, .7, .99]:
        lines += ['', f'## Per-case winners at {threshold:.0%} suppression', '',
                  '| Model | Pair | Feature (diff rank) | c | Actual suppression | Single SAE KL | FRA KL | Top-12 SAE KL |',
                  '|---|---|---|---:|---:|---:|---:|---:|']
        for item in summary['cases']:
            row = item['comparisons']['measured'][str(threshold)]
            p = row['activation_positive']
            ident = f"{p['feature']} ({p['rank']})" if p else 'unreached'
            lines.append(f"| {item['model']} | {item['trigger']}→{item['payload']} | {ident} | {fmt(p['strength'] if p else None)} | {p['suppression'] if p else 0:.2%} | {fmt(p['kl'] if p else None)} | {fmt(row['fra']['kl'] if row['fra'] else None)} | {fmt(row['legacy12']['kl'] if row['legacy12'] else None)} |")
    lines += ['', '## Signed activation and additive steering at 50% suppression', '',
              '| Model | Pair | Signed activation feature / coefficient / KL | Additive feature / coefficient / KL |',
              '|---|---|---|---|']
    for item in summary['cases']:
        row = item['comparisons']['measured']['0.5']
        def desc(p):
            return 'unreached' if p is None else f"{p['feature']} / {p['strength']:g} / {p['kl']:.6g}"
        lines.append(f"| {item['model']} | {item['trigger']}→{item['payload']} | {desc(row['activation'])} | {desc(row['additive'])} |")
    lines += ['', 'Negative activation coefficients amplify the feature. The positive-only primary',
              'comparison excludes them. Additive steering applies a constant unit-decoder',
              'direction and does not inherit the activation-dependent gate.', '',
              '## Why some collateral values are exactly zero', '',
              'For the positive-activation winners at 50% suppression:', '',
              '| Model | Pair | Feature | Active tokens in legitimate paragraph | Logits bitwise unchanged? |',
              '|---|---|---:|---:|---|']
    for key, run in runs.items():
        for case in run['cases']:
            point = at(candidates(case, 'activation_positive'), .5)
            if point is None:
                continue
            index = next(i for i, p in enumerate(case['points']) if p['feature'] == point['feature']
                         and p['mode'] == point['mode'] and p['strength'] == point['strength'])
            check = next(v for v in case['validation']['direct_checks'] if v['index'] == index)
            count = case['legitimate_feature_activity'][str(point['feature'])]['active_tokens']
            if point['kl'] == 0:
                assert count == 0 and check['logits_bitwise_identical_on_legitimate']
            lines.append(f"| {key} | {case['trigger']}→{case['payload']} | {point['feature']} | {count} | {check['logits_bitwise_identical_on_legitimate']} |")
    lines += ['', 'These dormant-feature results establish zero collateral on the original paragraph.',
              'They do not establish preservation on a broader distribution of legitimate text.', '',
              '## Ordinary single-feature ablation (c=1)', '',
              '| Model | Pair | Best suppression among ten | KL at that point |',
              '|---|---|---:|---:|']
    for key, run in runs.items():
        for case in run['cases']:
            best = max([p for p in candidates(case, 'activation_positive') if p['strength'] == 1], key=lambda p: p['suppression'])
            lines.append(f"| {key} | {case['trigger']}→{case['payload']} | {best['suppression']:.2%} | {best['kl']:.6g} |")
    lines += ['', '## Reproduction and direct verification', '',
              '| Model | Pair | Clean probability error | Max top-12 suppression error | Max top-12 KL relative error | Direct checks |',
              '|---|---|---:|---:|---:|---:|']
    for item in summary['cases']:
        v = item['validation']
        lines.append(f"| {item['model']} | {item['trigger']}→{item['payload']} | {v['base_error']:.6g} | {v['max_legacy_suppression_error']:.6g} | {v['max_legacy_kl_relative_error']:.2%} | {v['direct_checks']} |")
    lines += ['', '## Scope and limitations', '',
              '- One original evaluation seed and one original legitimate paragraph per case.',
              '- Same single SAE layer as the original baseline; no layer search.',
              '- Best-of-ten feature and coefficient selection is retrospective.',
              '- Unreachable FRA targets remain visible; no ratios are assigned to them.',
              '- FRA uses its archived sparse coefficient grid. Single SAE uses a denser grid.',
              '- `summary.json.gz` includes both measured results and adjacent-coefficient',
              '  interpolated estimates. Interpolation never crosses feature identities.',
              '- This is not a general capability or transfer evaluation.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    (OUT/'summary.json.gz').write_bytes(gzip.compress(json.dumps(summary, indent=2, allow_nan=False).encode(), mtime=0))
    print('\n'.join(lines))
    return summary


if __name__ == '__main__':
    analyze()
