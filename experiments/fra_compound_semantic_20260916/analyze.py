"""Analyze frozen choices; never select on held-out metrics."""
import gzip
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'

def load(stage='interventions'):
    path=OUT/(stage+'.json.gz')
    if path.exists(): return json.loads(gzip.decompress(path.read_bytes()))
    parts=[json.loads(gzip.decompress(p.read_bytes())) for p in sorted(OUT.glob(stage+'.part*.json.gz'))]
    assert parts and len(parts)==parts[0]['parts']
    assert [p['part'] for p in parts]==list(range(len(parts)))
    return json.loads(''.join(p['json_fragment'] for p in parts))

def key(cfg): return json.dumps(cfg,sort_keys=True)
def config_name(cfg):
    if cfg['method']=='none': return 'Poison, no edit'
    if cfg['method']=='sae': return f"SAE L{cfg['layer']} f{cfg['feature']} c={cfg['strength']}"
    if cfg['method']=='fra': return f"FRA {cfg['variant']} c={cfg['strength']}"
    return 'Oracle asset-edge ablation'

def main():
    r=load(); assert r['done'] and r['selection_frozen_before_test']
    diff_path=OUT/'diff_only.json.gz'
    if diff_path.exists():
        supplement=load('diff_only'); assert supplement['done']
        r['selected'].extend(supplement['selected']); r['test_points'].extend(supplement['points'])
    lookup={key(p['config']):p for p in r['test_points']}
    points=[{'family':'baseline','threshold':None,'config':r['test_baseline']['config'],
             'heldout':r['test_baseline']['summary']}]
    for s in r['selected']:
        points.append({**s,'heldout':lookup[key(s['config'])]['summary'] if s['config'] else None})
    output={'meta':r['meta'],'operating_points':points, 'normalization':r['validation']['normalization_audit'],
            'asset_edge_diagnostic':r['asset_edge_diagnostic']['summary']}
    (OUT/'summary.json.gz').write_bytes(gzip.compress(json.dumps(output,indent=2).encode(),mtime=0))
    rows=r['test_rows']; baseline=r['test_baseline']['summary']
    text=['# Compound semantic filtering: held-out results','',
        'One corrupted routing-table entry changes printer + wireless incidents from Network to Print. '
        'Printer + power must still route to Print. Device and fault descriptions appear in separate spans. '
        'Gemma-2-9B-IT was selected by the calibration behavior gate before testing any intervention.','',
        f"Calibration: 8 factorial blocks; tuning: 4; test: 12 (48 cases, including 12 joint-trigger cases). "
        'Test blocks comprise six lexical pairs in two document orders. All cases are retained.','',
        '## Does the compound backdoor generalize?','',
        '| Equipment | Problem | Clean accuracy | Poison accuracy | Clean P(Print) | Poison P(Print) | Poison KL |',
        '|---|---|---:|---:|---:|---:|---:|']
    labels={'00':('Portable computer','Power'),'10':('Printer','Power'),'01':('Portable computer','Wireless'),'11':('Printer','Wireless')}
    for corner,(asset,fault) in labels.items():
        rr=[x for x in rows if x['corner']==corner]; ss=baseline[corner]
        acc=sum(x['clean_label']==x['expected'] for x in rr)/len(rr)
        text.append(f"| {asset} | {fault} | {acc:.1%} | {ss['correct']:.1%} | {ss['clean_target_p']:.4f} | {ss['target_p']:.4f} | {ss['kl']:.4f} |")
    text+=['','## Fixed interventions selected on tuning only','',
        'Each row uses one fixed feature/coefficient or pair-set/coefficient for all test cases. '
        'Thresholds constrain mean joint suppression on **tuning**; test suppression may differ. '
        'The objective is lowest mean full-vocabulary KL across all four corners.','',
        '| Family | Tuning suppression constraint | Setting | Test joint suppression | Joint accuracy | Control accuracy | Joint KL | Control KL | Overall KL |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for s in points:
        threshold='none' if s['threshold'] is None else f">={s['threshold']:.0%}"
        if s['heldout'] is None:
            text.append(f"| {s['family']} | {threshold} | No feasible tuning setting | — | — | — | — | — | — |")
        else:
            v=s['heldout']; text.append(f"| {s['family']} | {threshold} | {config_name(s['config'])} | {v['joint_suppression']:.1%} | {v['11']['correct']:.1%} | {v['controls']['correct']:.1%} | {v['11']['kl']:.4f} | {v['controls']['kl']:.4f} | {v['all']['kl']:.4f} |")
    text+=['','## Interpretation','',
        'The held-out behavior is a selective semantic AND gate: the corruption flips all 12 printer/wireless '
        'cases while all 36 control cases remain correct. This does not produce an FRA advantage on the '
        'tested grid. The joint-feature SAE winner repairs 12/12 trigger cases and preserves 36/36 controls; '
        'the raw FRA winner repairs 7/12 and preserves 35/36. The separable FRA variant preserves all controls '
        'and has lower control KL, but repairs only 6/12 trigger cases.','',
        'The winning SAE feature is active only on the joint corner at the calibration answer position '
        '(mean 2.439 versus zero in all three controls). Thus the model has already combined the two input '
        'conditions into one SAE feature at this site. This is consistent with why a textual compound trigger '
        'can still be removable by single-feature steering. It does not prove this feature is the entire '
        'causal circuit, or localize which of the all-position edits caused the repair.','',
        'Moreover, oracle deletion of the tested direct asset-span attention edges repairs none of the '
        '12 joint cases. This fails to establish that the intended direct QK route is necessary at these '
        'layers; information may have moved earlier or through other positions. A stronger test of the '
        'two-feature hypothesis needs a causal interaction localization before comparing surgical cuts.','',
        'The original poison-minus-clean-only baseline is also evaluated separately in the table, as '
        'described in [DIFF_BASELINE](../DIFF_BASELINE.md). The stronger winner above comes from the '
        'factorial interaction ranking, not the original diff ranking.','',
        '## QK identification and mechanism checks','',
        'All three native IT SAE layers are searched: residual-post 9/20/31, used at attention layers 10/21/32. '
        'Single-feature candidates are the union of top-ten poisoned-minus-clean and top-ten factorial-contrast '
        'features per layer, measured at the answer prefix. Each candidate is swept independently, preserving '
        'the SAE reconstruction residual. Positive-only and signed steering are reported separately.','',
        'FRA cuts are **QK score cuts**. Pair terms are identified from the later problem/answer token to the '
        'earlier asset-type token. Raw, factorial-interaction, and separable interaction rankings each select up '
        'to 48 pairs per head. All 16 heads in each of the three layers are included. The separable variant '
        'requires a wireless-sensitive Q feature and a distinct printer-sensitive K feature; activation ratios '
        'must exceed 2 in both control strata. This response criterion is measured on calibration, not a '
        'human semantic label from unrelated text. Cuts then act on every causal token pair, with no test-position mask.','',
        '| Pair set | Selected pairs | Heads with nonempty set |','|---|---:|---:|']
    for variant in ['raw','interaction','separable']:
        rr=[p for p in r['fra_pairs'] if p['variant']==variant]
        text.append(f"| {variant} | {sum(len(p['pairs']) for p in rr)} | {sum(bool(p['pairs']) for p in rr)} / {len(rr)} |")
    example=next(x for x in r['fra_pairs'] if x['variant']=='separable' and x['pairs'])
    pair=example['pairs'][0]
    text+=['',f"Example calibrated pair at L{example['layer']}H{example['head']}: query feature {pair['q_feature']} and key feature {pair['k_feature']}. Values are mean activations over calibration blocks and the two query anchors, not global semantic annotations.",'',
        '| Corner (printer, wireless) | Q feature activation | K feature activation | QK contribution |',
        '|---|---:|---:|---:|']
    for c in ['00','10','01','11']:
        text.append(f"| {c} | {pair['q_activations'][c]:.4f} | {pair['k_activations'][c]:.4f} | {pair['contributions'][c]:.6f} |")
    text+=['','### Selected single-feature provenance','',
        '| Fixed feature | Calibration ranking membership | Activation 00 | Activation 10 | Activation 01 | Activation 11 |',
        '|---|---|---:|---:|---:|---:|']
    seen=set()
    for s in r['selected']:
        cfg=s['config']
        if not cfg or cfg['method']!='sae': continue
        ident=(cfg['layer'],cfg['feature'])
        if ident in seen: continue
        seen.add(ident)
        hits=[(rank,x) for rank,rr in r['ranking'][str(cfg['layer'])].items() for x in rr if x['feature']==cfg['feature']]
        v=hits[0][1]['corner_activations']
        text.append(f"| L{ident[0]} f{ident[1]} | {', '.join(x[0] for x in hits)} | {v['00']:.3f} | {v['10']:.3f} | {v['01']:.3f} | {v['11']:.3f} |")
    text+=['','## Unrestricted output check','',
        'Label accuracy above chooses among the four queue labels. The checks below use the '
        'highest-probability token across the ENTIRE vocabulary, and report total queue-label mass.', '',
        '| Fixed test setting | Full-vocabulary top-token accuracy | Mean queue-label probability mass |',
        '|---|---:|---:|']
    expected={x['case_id']:x['expected'] for x in r['test_rows']}
    for p in [r['test_baseline']]+r['test_points']:
        topacc=sum(x['top_token'].strip()==expected[x['case_id']] for x in p['rows'])/len(p['rows'])
        text.append(f"| {config_name(p['config'])} | {topacc:.1%} | {p['summary']['all']['label_mass']:.4f} |")
    edge=r['asset_edge_diagnostic']['summary']
    text+=['',f"Oracle ablation of all later attention to the asset span at the tested layers gives "
        f"{edge['joint_suppression']:.1%} joint suppression, {edge['11']['correct']:.1%} joint accuracy, "
        f"{edge['controls']['correct']:.1%} control accuracy and {edge['all']['kl']:.4f} overall KL. "
        'This position-aware diagnostic is not included in the competitive comparison.','',
        '## Normalization correction and verification','',
        'The inherited wrapper claims per-token normalization is required. This conflicts with '
        '[Gemma Scope §3.1](https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf): released weights '
        'already absorb the fixed training scale. Both methods here use native encoding without extra input '
        'normalization. The aborted pre-correction smoke source/log are retained. Earlier normalization=True '
        'results need a separate audit.','',
        '| Attention layer | Native mean L0 | Old mean L0 | Native relative squared error | Old relative squared error |',
        '|---|---:|---:|---:|---:|']
    for v in r['validation']['normalization_audit']:
        text.append(f"| {v['layer']} | {v['native_mean_l0']:.2f} | {v['legacy_mean_l0']:.2f} | {v['native_relative_squared_error']:.4f} | {v['legacy_relative_squared_error']:.4f} |")
    invalid=[p for p in r['points'] if not p.get('valid',True)]
    text+=['',f"Numerical validity: {len(invalid)} of {len(r['points'])} tuning configurations produced non-finite logits and were excluded from selection. The archive retains each affected configuration and case. No failed value was assigned zero KL."]
    if invalid:
        text+=['','| Invalid setting | Affected tuning cases |','|---|---:|']
        for p in invalid:
            text.append(f"| {config_name(p['config'])} | {sum('numerical_failure' in x for x in p['rows'])} |")
    checks=r['validation']['direct_checks']
    text+=['',f"Verification: {len(checks)} main-sweep selected configurations were checked with full-prefix forward passes "
        f"before test selection. Maximum mean-KL discrepancy: {max(v['mean_kl_error'] for v in checks):.6g}; "
        f"maximum target-probability discrepancy: {max(v['max_target_p_error'] for v in checks):.6g}. "
        f"All test points use full-prefix forwards. Zero-edit target-probability error: "
        f"{r['validation']['zero_max_probability_error']:.6g}. Final-token-only unembedding was checked against "
        f"ordinary unembedding (maximum logit error {r['validation']['last_position_unembed_max_logit_error']:.6g}).",'',
        '## Scope','',
        '- This is a controlled corrupted-knowledge-base simulation, not a weight-trained sleeper agent.',
        '- A textual AND gate and calibrated separable feature responses do not prove an isolated causal two-feature circuit.',
        '- KL is clean-context versus poisoned-context-plus-edit over the full next-token vocabulary at the same queue continuation. It is not an unrelated-text KL. This one-token routing task does not establish long-form generation fidelity.',
        '- Results cover one task, one model, three SAE sites, fixed pair selectors and a finite steering grid. They cannot establish a general advantage or impossibility for FRA.',
        '- The model and SAE preprocessing differ from the earlier single-trigger experiments. This is not a controlled estimate of the effect of adding a second trigger condition.',
        '- The settings and paraphrases were frozen before test evaluation. Feasibility failures and all held-out cases are archived.', '',
        'See [PROTOCOL](../PROTOCOL.md), [feasibility screens](FEASIBILITY.md), and the complete compressed per-case archive.']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text[:45]))
if __name__=='__main__': main()
