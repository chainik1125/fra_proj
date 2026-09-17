"""Check that every SAE candidate has the full coarse grid in all three scopes."""
import argparse
import json
from pathlib import Path
from analyze import load


def audit(main, extra, scopes):
    assert main['done'] and extra['done'] and scopes['done']
    observed = set()
    for source in [main, extra, scopes]:
        for point in source['points']:
            c = point['config']
            if c['method'] == 'sae':
                observed.add((c['layer'], c['feature'], c.get('scope', 'all'),
                              c.get('intervention', 'activation'), c['strength']))
    features = [(int(layer), int(feature)) for layer, ff in scopes['features'].items()
                for feature in ff]
    grids = {'activation': scopes['meta']['sae_grid'], 'constant': scopes['meta']['constant_grid']}
    expected = {(l, f, scope, mode, strength) for l, f in features
                for scope in ['all', 'document', 'no_bos']
                for mode, grid in grids.items() for strength in grid}
    missing = expected-observed
    result = {'feature_count': len(features),
              'features_per_layer': {l: len(ff) for l, ff in scopes['features'].items()},
              'expected_coarse_configurations': len(expected),
              'observed_expected_configurations': len(expected & observed),
              'scopes': ['all', 'document', 'no_bos'], 'grids': grids,
              'missing_count': len(missing), 'first_missing': sorted(missing)[:10],
              'nonfinite_measurements_count_as_evaluated_but_are_ineligible_for_selection': True}
    assert not missing, result
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('main')
    parser.add_argument('extra')
    parser.add_argument('scopes')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = audit(load(args.main), load(args.extra), load(args.scopes))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
