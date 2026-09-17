"""Summarize single-feature sweeps without interpolating across feature identities."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from result_io import read_json, result_exists, write_json_gz

ROOT = Path(__file__).resolve().parent
CONCEPTS = ['vessel', 'vehicle', 'bird', 'fire', 'war', 'medical']
LEVELS = [.3, .5, .7]


def geometric(values):
    if not values:
        return None
    if any(v == 0 for v in values):
        return 0.0
    return math.exp(statistics.mean(math.log(v) for v in values))


def curve_at(points, threshold, interpolate=False):
    """Minimum measured KL at/above target, or interpolation along one strength curve.

    points carry signed strength, suppression, KL. Sorting by strength preserves
    nonmonotonic branches. Interpolating only adjacent strengths avoids jumping
    across features or opposite ends of a nonmonotonic sweep.
    """
    points = sorted(points, key=lambda p: p['strength'])
    eligible = [{**p, 'estimated': False} for p in points if p['suppression'] >= threshold]
    if interpolate:
        for left, right in zip(points, points[1:]):
            a, b = left['suppression'], right['suppression']
            if min(a, b) <= threshold <= max(a, b) and a != b:
                frac = (threshold-a)/(b-a)
                eligible.append({'strength': left['strength']+frac*(right['strength']-left['strength']),
                                 'suppression': threshold,
                                 'kl': left['kl']+frac*(right['kl']-left['kl']),
                                 'estimated': True})
    return min(eligible, key=lambda p: (p['kl'], abs(p['strength']))) if eligible else None


def key(row):
    return row['concept'], row['ctx'], row['word']


def comparisons(pairs):
    both = [(f, s) for f, s in pairs if f is not None and s is not None]
    return {'total': len(pairs), 'fra_reach': sum(f is not None for f, s in pairs),
            'sae_reach': sum(s is not None for f, s in pairs), 'both_reach': len(both),
            'sae_lower': sum(s['kl'] < f['kl'] for f, s in both),
            'fra_lower': sum(f['kl'] < s['kl'] for f, s in both),
            'zero_sae_collateral': sum(s['kl'] == 0 for f, s in both),
            'geo_sae_over_fra': geometric([s['kl']/max(f['kl'], 1e-12) for f, s in both]),
            'median_fra': statistics.median([f['kl'] for f, s in both]) if both else None,
            'median_sae': statistics.median([s['kl'] for f, s in both]) if both else None}


def analyze(results_dir):
    reference = json.loads((ROOT/'reference/rung3_broad6.json').read_text())['rows']
    ref_by_key = {key(r): r for r in reference}
    runs = {c: read_json(results_dir/(f'{c}.verified.json' if result_exists(results_dir/f'{c}.verified.json')
                                    else f'{c}.json')) for c in CONCEPTS}
    assert all(run['done'] for run in runs.values())
    assert sum(len(run['rows']) for run in runs.values()) == 70
    output = {'views': {}, 'winners': {}, 'validation': {c: r['validation'] for c, r in runs.items()}}
    output['all_direct_verified'] = all(r.get('direct_verification_done', False) for r in runs.values())
    output['strength_limits'] = []
    for cap in [1, 4, 8, 16, 32, 64]:
        reach = [0, 0, 0]
        zero = [0, 0, 0]
        for run in runs.values():
            allowed = {p['feature'] for p in run['rankings']['no_payload'][:10]}
            indices = [i for i, cfg in enumerate(run['configurations'])
                       if cfg['mode'] == 'activation' and cfg['feature'] in allowed
                       and 0 <= cfg['strength'] <= cap]
            for row in run['rows']:
                if row['kind'] != 'synonym':
                    continue
                for j, t in enumerate(LEVELS):
                    eligible = [i for i in indices if row['suppression'][i] >= t]
                    reach[j] += bool(eligible)
                    zero[j] += bool(eligible) and min(run['collateral'][i] for i in eligible) == 0
        output['strength_limits'].append({'max_coefficient': cap, 'reach': reach, 'zero_collateral_reach': zero})
    for interpolation in [False, True]:
        estimator = 'interpolated' if interpolation else 'measured'
        for contrast in ['no_payload', 'unrelated', 'either']:
            for mode in ['activation_positive', 'activation', 'additive', 'either']:
                label = f'{estimator}/{contrast}/{mode}'
                output['views'][label] = {}
                for t in LEVELS:
                    all_pairs = []
                    per_query = []
                    fixed_pairs = []
                    fixed_winners = []
                    for concept, run in runs.items():
                        allowed = {item['feature'] for name, rank in run['rankings'].items()
                                   if contrast == 'either' or name == contrast
                                   for item in rank[:10]}
                        groups = defaultdict(list)
                        for i, cfg in enumerate(run['configurations']):
                            if cfg['mode'] == 'legacy12' or cfg['feature'] not in allowed:
                                continue
                            if mode == 'activation_positive' and (cfg['mode'] != 'activation' or cfg['strength'] < 0):
                                continue
                            if mode not in ['either', 'activation_positive'] and cfg['mode'] != mode:
                                continue
                            groups[(cfg['feature'], cfg['mode'])].append(i)
                        rows = [r for r in run['rows'] if r['kind'] == 'synonym']
                        feature_operating_points = {}
                        fra_points = {}
                        for row in rows:
                            ref = ref_by_key[key(row)]
                            fp = [{'strength': c, 'suppression': s, 'kl': k}
                                  for c, (s, k) in zip([1, 2, 4, 8, 16, 32], ref['fra'])]
                            fp.append({'strength': 0, 'suppression': 0, 'kl': 0})
                            fra_points[key(row)] = curve_at(fp, t, interpolation)
                        for identity, indices in groups.items():
                            f, m = identity
                            feature_operating_points[identity] = {}
                            for row in rows:
                                pts = [{'strength': run['configurations'][i]['strength'],
                                        'suppression': row['suppression'][i],
                                        'kl': run['collateral'][i]} for i in indices]
                                point = curve_at(pts, t, interpolation)
                                if point is not None:
                                    point.update(feature=f, mode=m)
                                feature_operating_points[identity][key(row)] = point
                        for row in rows:
                            candidates = [pts[key(row)] for pts in feature_operating_points.values()
                                          if pts[key(row)] is not None]
                            best = min(candidates, key=lambda p: p['kl']) if candidates else None
                            fra = fra_points[key(row)]
                            all_pairs.append((fra, best))
                            per_query.append({'concept': concept, 'ctx': row['ctx'], 'word': row['word'],
                                              'fra': fra, 'sae': best})
                        # Choose ONE feature and intervention type across all contexts/words
                        # in this concept, prioritizing reach, then mean KL.
                        def candidate_key(identity):
                            reached = [p for p in feature_operating_points[identity].values() if p is not None]
                            return -len(reached), statistics.mean(p['kl'] for p in reached) if reached else float('inf')
                        winner = min(groups, key=candidate_key)
                        pts = feature_operating_points[winner]
                        cp = [(fra_points[key(r)], pts[key(r)]) for r in rows]
                        fixed_pairs.extend(cp)
                        ranks = {name: next((r['rank'] for r in rank if r['feature'] == winner[0]), None)
                                 for name, rank in run['rankings'].items()}
                        fixed_winners.append({'concept': concept, 'feature': winner[0], 'mode': winner[1],
                                              'ranks': ranks,
                                              'strengths': sorted({p['strength'] for p in pts.values() if p is not None}),
                                              'legitimate_activity': run.get('legitimate_feature_activity', {}).get(str(winner[0])),
                                              **comparisons(cp)})
                    output['views'][label][str(t)] = {'per_query_best': comparisons(all_pairs),
                        'one_feature_per_concept': comparisons(fixed_pairs),
                        'fixed_winners': fixed_winners, 'queries': per_query}
    write_json_gz(results_dir/'summary.json', output)
    write_report(output, results_dir)
    return output


def write_report(summary, out):
    lines = ['# Single-feature SAE comparison with FRA semantic-filter transfer', '',
             '**Metric correction:** this historical report measures KL on an independent,',
             'unpoisoned paragraph. Its zero values do not establish restoration with the',
             'backdoor present. See the [paired semantic-filter rerun](../../fra_semantic_restoration_20260916/README.md)',
             'for the corrected continuation and full-query distribution comparisons.', '',
             '**Finding:** activation-weighted steering of individual differential SAE features',
             'beats the saved FRA curves on this benchmark. At 30/50/70% suppression it reaches',
             '52/52 related-word queries, with exact zero KL on the original legitimate paragraphs.',
             'The selected features are dormant on those paragraphs; coefficients above ordinary',
             'ablation are needed. Additive decoder-direction steering is substantially weaker.', '',
             'Six concepts, 52 related-word queries in three contexts. Ten candidates per contrast;',
             'each feature is edited individually. Feature/strength winners are selected after evaluation.',
             'Raw sweeps also retain the 18 planted-word queries.', '',
             '## Primary: measured points, planted-minus-no-payload ranking', '',
             'At each threshold, select the lowest measured KL among points that achieve at least',
             'that suppression. FRA uses its saved six-point grid; SAE uses the denser grid listed',
             'in PROTOCOL.md/source. Ratios above one favor FRA. Comparisons use queries both methods reach.', '',
             '| Single-feature mode | Suppression | SAE reach | FRA reach | SAE lower / comparable | Zero SAE KL / comparable | SAE KL / FRA KL (geomean) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for mode in ['activation_positive', 'activation', 'additive', 'either']:
        for t in LEVELS:
            s = summary['views'][f'measured/no_payload/{mode}'][str(t)]['per_query_best']
            ratio = f"{s['geo_sae_over_fra']:.3f}" if s['geo_sae_over_fra'] is not None else '—'
            lines.append(f"| {mode} | {t:.0%} | {s['sae_reach']}/{s['total']} | {s['fra_reach']}/{s['total']} | {s['sae_lower']}/{s['both_reach']} | {s['zero_sae_collateral']}/{s['both_reach']} | {ratio} |")
    lines += ['', '## One feature per concept at 50% suppression', '',
              'Choose a single feature and intervention type per concept, maximizing query coverage',
              'then minimizing mean collateral. Strength may vary by query. Primary contrast.', '',
              '| Concept | Feature | Diff rank | Positive coefficients used | Reach | SAE lower / comparable | SAE / FRA KL |',
              '|---|---:|---:|---|---:|---:|---:|']
    for w in summary['views']['measured/no_payload/activation_positive']['0.5']['fixed_winners']:
        ratio = f"{w['geo_sae_over_fra']:.3f}" if w['geo_sae_over_fra'] is not None else '—'
        lines.append(f"| {w['concept']} | {w['feature']} | {w['ranks']['no_payload']} | {w['strengths']} | {w['sae_reach']}/{w['total']} | {w['sae_lower']}/{w['both_reach']} | {ratio} |")
    lines += ['', '## Strength dependence', '',
              'Positive activation-weighted coefficients only. c=1 removes the encoded feature contribution;',
              'c>1 subtracts more than that contribution. Best individual feature among the primary ten.', '',
              '| Maximum c | Reach at 30% | Reach at 50% | Reach at 70% | Zero-KL reach at 30/50/70% |',
              '|---:|---:|---:|---:|---|']
    for s in summary['strength_limits']:
        a,b,c = s['reach']
        lines.append(f"| {s['max_coefficient']} | {a}/52 | {b}/52 | {c}/52 | {s['zero_collateral_reach']} |")
    lines += ['', '## Sensitivity: interpolated curves and both contrasts', '',
              'Interpolation stays within each feature/mode curve, between adjacent strength values,',
              'and includes the zero-edit point. These are estimates, not fresh measured operating points.', '',
              '| Comparison | Suppression | SAE reach | SAE lower / comparable | SAE / FRA KL |',
              '|---|---:|---:|---:|---:|']
    for label in ['measured/either/either', 'interpolated/no_payload/either', 'interpolated/either/either']:
        for t in LEVELS:
            s = summary['views'][label][str(t)]['per_query_best']
            ratio = f"{s['geo_sae_over_fra']:.3f}" if s['geo_sae_over_fra'] is not None else '—'
            lines.append(f"| {label} | {t:.0%} | {s['sae_reach']}/{s['total']} | {s['sae_lower']}/{s['both_reach']} | {ratio} |")
    lines += ['', '## Reproduction checks', '',
              '| Concept | Max clean probability error | Max 12-feature suppression error | Max 12-feature KL relative error |',
              '|---|---:|---:|---:|']
    for c, v in summary['validation'].items():
        lines.append(f"| {c} | {v['max_reference_base_error']:.6f} | {v['max_legacy_suppression_error']:.6f} | {v['max_legacy_kl_relative_error']:.4%} |")
    if summary['all_direct_verified']:
        lines += ['', 'All collateral grid points were remeasured with batch size one. Zero-KL points',
                  'leave the residual tensor exactly unchanged on the legitimate paragraph. Direct full',
                  'forward passes also checked dormant features and the selected operating points.', '',
                  '| Concept | Directly verified target points | Dormant features checked | All dormant checks bitwise identical? |',
                  '|---|---:|---:|---|']
        for c,v in summary['validation'].items():
            checks = v['direct_dormant_feature_checks']
            lines.append(f"| {c} | {v['direct_verified_points']} | {len(checks)} | {all(p['logits_bitwise_identical'] for p in checks)} |")
    lines += ['', 'The original 12-feature fire curve does not reproduce tightly: its largest',
              'suppression discrepancy is about 0.27. Other concepts have maximum discrepancies',
              'of about 0.01–0.03. FRA is compared against the saved author run, not a fresh FRA',
              'run in this environment; do not describe this as a bitwise reproduction of the campaign.']
    lines += ['', '## Scope', '',
              '- Same single layer (residual-pre layer 6) as the original SAE baseline; no layer search.',
              '- Same one legitimate paragraph per concept; no general capability evaluation.',
              '- Best-of-ten selection is retrospective; these are optimistic baseline results.',
              '- FRA data are the committed reference run; clean probabilities and original SAE curves',
              '  are rerun as checks of cross-run comparability.',
              '- Positive activation coefficient removes a feature; negative coefficient amplifies it.',
              '- Additive coefficient uses a unit decoder vector in residual-stream units.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def self_test():
    # Do not clamp an unreached target or extrapolate below the first measurement.
    assert curve_at([{'strength': 1, 'suppression': .2, 'kl': 1}], .3) is None
    points = [{'strength': 0, 'suppression': 0, 'kl': 0},
              {'strength': 1, 'suppression': .8, 'kl': 2},
              {'strength': 2, 'suppression': .4, 'kl': 3}]
    assert curve_at(points, .4)['kl'] == 2
    assert abs(curve_at(points, .4, True)['kl']-1) < 1e-12
    assert comparisons([(None, {'kl': 1}), ({'kl': 2}, None)])['both_reach'] == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print('Frontier checks passed')
    else:
        result = analyze(args.results)
        print((args.results/'REPORT.md').read_text())
