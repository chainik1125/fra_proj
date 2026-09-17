"""Report every feasibility screen and freeze a choice before intervention tuning."""
import gzip
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
FILES = ['feasibility', 'feasibility_it', 'feasibility_print_2b', 'feasibility_print_9b']
FORMS = ['exception_note', 'resolved_cases', 'routing_table']


def analyze():
    summary = []
    for name in FILES:
        path = ROOT/'results'/f'{name}.json.gz'
        if not path.exists(): continue
        run = json.loads(gzip.decompress(path.read_bytes())); assert run['done']
        for form in FORMS:
            rows = [r for r in run['rows'] if r['format'] == form]
            target = rows[0].get('payload', 'Billing')
            stats = {}
            for a, b in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                pair = {p: [r for r in rows if r['a'] == a and r['b'] == b and r['poisoned'] == p] for p in [False, True]}
                stats[f'{a}{b}'] = {
                    'clean_target_p': mean(r['probabilities'][target] for r in pair[False]),
                    'poison_target_p': mean(r['probabilities'][target] for r in pair[True]),
                    'poison_restricted_target_p': mean(r['restricted_probabilities'][target] for r in pair[True]),
                    'clean_accuracy': mean(r['predicted_label'] == r['expected'] for r in pair[False]),
                    'poison_accuracy': mean(r['predicted_label'] == r['expected'] for r in pair[True]),
                    'label_mass': mean(r['label_mass'] for r in pair[True])}
            clean_accuracy = mean(s['clean_accuracy'] for s in stats.values())
            control_accuracy = mean(stats[k]['poison_accuracy'] for k in ['00', '10', '01'])
            delta = stats['11']['poison_target_p']-stats['11']['clean_target_p']
            control_delta = max(abs(stats[k]['poison_target_p']-stats[k]['clean_target_p']) for k in ['00', '10', '01'])
            passed = clean_accuracy >= .85 and control_accuracy >= .85 and delta >= .40 and stats['11']['poison_restricted_target_p'] >= .70 and control_delta <= .15
            summary.append({'file': name, 'model': run['model'], 'format': form, 'payload': target,
                            'clean_accuracy': clean_accuracy, 'control_accuracy': control_accuracy,
                            'joint_target_increase': delta, 'largest_control_target_change': control_delta,
                            'passed': passed, 'corners': stats})
    lines = ['# Feasibility screens', '',
        'All choices here use calibration vocabulary only. No FRA or steering results were',
        'available when choosing the context. The first Billing-payload design failed to form',
        'a selective conjunction. The revised Print payload is also a legitimate answer in',
        'the printer/power control, requiring preservation of that output elsewhere.', '',
        '| Screen | Model | Format | Clean accuracy | Poisoned control accuracy | Joint target increase | Largest control target change | Pass |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for r in summary:
        lines.append(f"| {r['file']} | {r['model']} | {r['format']} | {r['clean_accuracy']:.1%} | {r['control_accuracy']:.1%} | {r['joint_target_increase']:.4f} | {r['largest_control_target_change']:.4f} | {r['passed']} |")
    (ROOT/'results/FEASIBILITY.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'results/feasibility_summary.json.gz').write_bytes(gzip.compress(json.dumps(summary, indent=2).encode(), mtime=0))
    passing = [r for r in summary if r['passed'] and r['payload'] == 'Print']
    if passing:
        # Prefer the smaller adequate model, then the least artificial presentation.
        chosen = min(passing, key=lambda r: (0 if '2b' in r['model'] else 1, FORMS.index(r['format'])))
        (ROOT/'reference/selection.json').write_text(json.dumps(chosen, indent=2)+'\n')
        print('SELECTED', chosen['model'], chosen['format'])
    else:
        print('NO CONTEXT PASSED YET')
    print('\n'.join(lines))


if __name__ == '__main__': analyze()
