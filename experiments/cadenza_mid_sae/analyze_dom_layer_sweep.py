"""Build a small report and scientific plot from the audited remote sweep JSON."""
import json
import lzma
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent


def compare_rows(a, b):
    assert [x['key'] for x in a] == [x['key'] for x in b]
    delta = [x['triggered_to_clean_js_bits']-y['js_bits'] for x,y in zip(a,b)]
    rng = random.Random(20260921)
    boot = sorted(sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(2000))
    return {'mean_difference':sum(delta)/len(delta),'paired_95pct':[boot[50],boot[1949]]}


def main():
    data = json.loads(lzma.decompress((HERE/'DOM_ALL_LAYER_SWEEP_RESULTS_20260921.json.xz').read_bytes()))
    assert all(r['audit']['worker_exited'] for r in data['new_runs'])
    primary = [r for r in data['records'] if r['rule']=='restoration']
    by_variant = {v:sorted([r for r in primary if r['mode']=='sweep' and r['variant']==v],
                          key=lambda r:r['layers'][0]) for v in ['input_prompt','resid_response']}
    simultaneous = {r['variant']:r for r in primary if r['mode']=='all'}
    assert all([r['layers'][0] for r in rows]==list(range(32)) for rows in by_variant.values())
    val_best = {v:min(rows,key=lambda r:(r['validation_metrics']['triggered_to_clean_js_bits'],
                                        abs(r['coefficient']),r['coefficient'],r['layers'][0]))
                for v,rows in by_variant.items()}
    observed_best = {v:min((r for r in data['records'] if r['mode']=='sweep' and r['variant']==v),
                          key=lambda r:r['metrics']['triggered_to_clean_js_bits'])
                     for v in by_variant}
    prior = json.loads((HERE/'STEERING_RESULTS_20260921.json').read_text())
    fra = next(r for r in prior['runs'] if r['run_id']=='A-input-L08-ov50-s2-20260921')
    fra_metrics = fra['summary']['confirmation']['results']['positive_jsd']['metrics']
    fra_rows = fra['per_prompt_jsd']['confirmation_positive_jsd']
    comparisons = {scope+'_'+variant:compare_rows(r['rows'],fra_rows)
                   for scope,group in [('all',simultaneous),('validation_best_layer',val_best),
                                       ('observed_best_layer',observed_best)] for variant,r in group.items()}
    labels = {'input_prompt':'Attention-input DoM','resid_response':'Full residual-stream DoM'}
    earlier_ranking = json.loads((HERE/'CONFIRMATION_RANKING_20260921.json').read_text())
    ranking = [{k:v for k,v in r.items() if k!='rows'}
               for r in earlier_ranking['best_observed_settings'] if not r['method'].startswith('DoM')]
    for group, suffix in [(observed_best,''),(simultaneous,' — simultaneous')]:
        for variant,r in group.items():
            ranking.append({'method':labels[variant]+suffix,'layer':r['layers'][0] if r['mode']=='sweep' else 'All 32',
                            'feature':'Dense vector' if r['mode']=='sweep' else '32 layer vectors',
                            'alpha':r['coefficient'],'rule':r['rule'],'metrics':r['metrics'],'run_id':r['run_id']})
    ranking.sort(key=lambda r:r['metrics']['triggered_to_clean_js_bits'])
    def js(r):return r['metrics']['triggered_to_clean_js_bits']
    def row(label,r):
        m=r['metrics'];layer='All 32' if r['mode']=='all' else str(r['layers'][0])
        return (f'| {label} | {layer} | {r["coefficient"]:g} | '
                f'{r["validation_metrics"]["triggered_to_clean_js_bits"]:.6f} | {js(r):.6f} | '
                f'{m["escaped_phrase_count"]} | {round(64*m["clean_exact_match"])} | {m["clean_drift_js_bits"]:.6f} |')
    lines=['# All-layer and individual-layer DoM sweep','',
        'Both stages completed on simplex2. The simultaneous all-32-layer experiment finished before the individual sweep started. The individual sweep covers every zero-based layer 0–31 for both attention-input and full residual-stream DoM.','',
        'Every result uses the same 64 confirmation prompt pairs. Unsteered triggered-to-clean JSD is 0.991125 bits; lower is better. IHY removed counts continuations without the sleeper phrase. Clean preserved counts exact equality with the unsteered clean continuation.','',
        '## Simultaneous DoM and validation-selected individual layers','',
        'For each individual-layer row in this table, both layer and coefficient minimize validation JSD. Confirmation results do not select these rows. Each layer itself was tested independently; there is no averaging of steering vectors or results across layers.','',
        '| Method | Layer(s) | Raw-DoM coefficient | Validation JSD | Confirmation JSD | IHY removed /64 | Clean preserved /64 | Clean drift |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for v in labels:lines.append(row(labels[v]+' — simultaneous',simultaneous[v]))
    for v in labels:lines.append(row(labels[v]+' — best validation layer',val_best[v]))
    lines += ['',f'FRA OV reference: layer 8, feature 30892, alpha 16, confirmation JSD **{fra_metrics["triggered_to_clean_js_bits"]:.6f}**, '
              f'IHY removed **{fra_metrics["escaped_phrase_count"]}/64**, clean preserved **{round(64*fra_metrics["clean_exact_match"])}/64**.','',
              '## Best observed individual settings','',
              'These are the lowest confirmation means among the validation-frozen choices, selected descriptively after seeing confirmation. They need not coincide with the validation-selected layers above.','',
              '| Method | Layer | Coefficient | Validation JSD | Confirmation JSD | IHY removed /64 | Clean preserved /64 | Clean drift |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in labels:lines.append(row(labels[v],observed_best[v]))
    lines += ['', '## Updated ranking across hooks and methods','',
              'Lowest observed confirmation JSD per method among its validation-frozen choices. This uses individual settings, with no averaging across features or layers. SAE/FRA retains its existing three-layer search; DoM now includes all 32 individual layers and the simultaneous condition.','',
              '| Rank | Method | Layer(s) | Feature / vector | Coefficient | JSD | IHY removed /64 | Clean preserved /64 |',
              '|---:|---|---|---|---:|---:|---:|---:|']
    for i,r in enumerate(ranking,1):
        m=r['metrics']
        lines.append(f'| {i} | {r["method"]} | {r["layer"]} | {r["feature"]} | {r["alpha"]:g} | '
                     f'{m["triggered_to_clean_js_bits"]:.6f} | {m["escaped_phrase_count"]} | {round(64*m["clean_exact_match"])} |')
    lines += ['', 'FRA QK+OV uses three features jointly; the other SAE/FRA entries each use one feature. '
              'Coefficient units differ between DoM and SAE/FRA. The lowest mean remains FRA OV, but its paired JSD difference '
              'from the best residual DoM includes zero in the interval below. Their clean-continuation preservation differs markedly.']
    lines += ['', '![Layer sweep](DOM_LAYER_SWEEP_20260921.png)', '',
              '## Every individual layer','',
              'Each cell uses its own minimum-validation-JSD coefficient. Layers 8, 16 and 24 reuse the previously completed matched runs. Other layers were newly evaluated with the same grid.','',
              '| Layer | Attention coefficient | Attention JSD | Attention clean preserved /64 | Residual coefficient | Residual JSD | Residual clean preserved /64 |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for a,b in zip(by_variant['input_prompt'],by_variant['resid_response']):
        lines.append(f'| {a["layers"][0]} | {a["coefficient"]:g} | {js(a):.6f} | {round(64*a["metrics"]["clean_exact_match"])} | '
                     f'{b["coefficient"]:g} | {js(b):.6f} | {round(64*b["metrics"]["clean_exact_match"])} |')
    lines += ['', '## Paired comparisons to FRA OV','',
              'DoM minus FRA JSD; positive favors FRA. Intervals use 2,000 paired prompt resamples, seed 20260921.','',
              '| Selection | Variant | Difference | 95% paired interval |','|---|---|---:|---|']
    for name,c in comparisons.items():
        lo,hi=c['paired_95pct'];lines.append(f'| {name} | DoM | {c["mean_difference"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] |')
    lines += ['', 'These intervals are not adjusted for multiple comparisons or post-hoc best-observed layer selection. The confirmation block has been examined in earlier experiments and is not an untouched campaign-wide test.','',
              '## Protocol and checks','',
              '- Vectors are separate clean-minus-triggered means of the last-prompt-token activation at each layer, fitted only from the original 64 training pairs. The new fit reproduced the six previous vectors at layers 8/16/24 exactly; their saved values were reused.',
              '- Simultaneous steering applies one shared coefficient to each layer’s own raw DoM vector. Its grid is 0 and ±2^k for k=-6,…,4 (23 settings per variant). Individual layers use the original grid 0 and ±2^k for k=-3,…,4 (17 settings per variant). Coefficients are not norm-matched across locations or simultaneous/individual interventions.',
              '- DoM now covers all 32 layers, while the existing SAE/FRA comparison covers layers 8/16/24. These results compare the completed searches with those different layer-search budgets.',
              '- Attention steering modifies valid prompt positions at the pre-gain RMS-normalized attention input. Residual steering adds a dense vector to the full post-block residual at the last prompt token and every subsequent decode token.',
              '- Layer-zero attention-input DoM is exactly zero and is reported as an unsteered control. Its vector cannot encode earlier prompt context at that location.',
              '- Both restoration and suppression selection rules are retained. Main tables use restoration (minimum validation JSD); secondary suppression results are in the JSON.',
              '- Model variant A and pinned revision, original question splits, eight-sequence interleaved batches and 32-token greedy rollouts are unchanged. All workers reproduced every clean/triggered baseline continuation before measuring steering.',
              '- Local suite: 38 passed, 16 remote-only skipped. Each worker passed all 54 remote tests. Checks cover multi-layer hook composition and cleanup, equivalence to individual capture, and freezing validation choices before confirmation.',
              '- Independent audit recomputed the selection rules from every new validation grid, checked saved-choice/direction hashes, and recomputed all confirmation metrics from the prompt rows. All workers exited successfully. The new runs contain 1,032 validation records; including the reused six single-layer conditions gives 1,134.', '',
              '## Early stopping and empty text','']
    flagged=[r for r in primary if r['audit']['blank_triggered_count'] or r['audit']['blank_clean_count'] or r['audit']['min_compared_steps']<32]
    if flagged:
        lines += ['| Setting | Empty triggered /64 | Empty clean /64 | Minimum compared steps |','|---|---:|---:|---:|']
        for r in flagged:
            a=r['audit'];lines.append(f'| {r["tag"]} | {a["blank_triggered_count"]} | {a["blank_clean_count"]} | {a["min_compared_steps"]:g} |')
    else:lines += ['No primary selected setting produced an empty/whitespace-only output; all comparisons covered 32 steps.']
    lines += ['', '## Files and provenance','',
              '[Audited metrics, both selection rules, per-prompt measurements and source hashes (XZ-compressed JSON)](DOM_ALL_LAYER_SWEEP_RESULTS_20260921.json.xz). '
              '[Exact confirmation prompts](DOM_CONFIRMATION_PROMPTS_20260921.md). [Standalone figure PDF](DOM_LAYER_SWEEP_20260921.pdf).','',
              'All full generations remain in the corresponding simplex2 run directories under `/data/users/dmitry/sae-middle/runs/`. Each record in the results JSON gives its exact remote source file. No model weights or probability caches were downloaded.','']
    (HERE/'DOM_ALL_LAYER_SWEEP_20260921.md').write_text('\n'.join(lines))
    compact=lambda r:{k:v for k,v in r.items() if k!='rows'}
    conclusion={'simultaneous':{k:compact(v) for k,v in simultaneous.items()},
                'validation_best':{k:compact(v) for k,v in val_best.items()},
                'observed_best':{k:compact(v) for k,v in observed_best.items()},
                'best_observed_ranking':ranking,'comparisons_vs_fra':comparisons}
    (HERE/'DOM_LAYER_SWEEP_SUMMARY_20260921.json').write_text(json.dumps(conclusion,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,gridspec_kw={'height_ratios':[1.3,1]},layout='constrained')
    colors={'input_prompt':'#176b9c','resid_response':'#b24a32'}
    for v,rs in by_variant.items():
        x=[r['layers'][0] for r in rs];y=[js(r) for r in rs]
        axes[0].plot(x,y,'o-',color=colors[v],label=labels[v],markersize=3,linewidth=1.6)
        axes[0].fill_between(x,[r['js_bits_bootstrap_95pct'][0] for r in rs],
                            [r['js_bits_bootstrap_95pct'][1] for r in rs],color=colors[v],alpha=.09)
        axes[0].axhline(js(simultaneous[v]),color=colors[v],linestyle=':',linewidth=1.5,label=labels[v]+' — all 32')
        axes[1].plot(x,[r['metrics']['clean_drift_js_bits'] for r in rs],'o-',color=colors[v],markersize=3)
    axes[0].axhline(fra_metrics['triggered_to_clean_js_bits'],color='#333333',linestyle='--',linewidth=1.4,label='FRA OV, L8')
    axes[0].set_ylabel('Triggered-to-clean JSD (bits) ↓')
    axes[0].set_title('DoM across all 32 layers — the same 64 confirmation pairs',loc='left',fontweight='bold')
    axes[0].legend(fontsize=8,ncol=2,loc='lower left')
    axes[1].set_ylabel('Clean-prompt drift JSD (bits) ↓')
    axes[1].set_xlabel('Individual steering layer (zero-based)')
    for ax in axes:
        ax.set_xlim(-.5,31.5);ax.set_ylim(-.02,1.02);ax.grid(axis='y',alpha=.18)
    axes[0].set_ylim(.72,1.005)
    axes[1].set_xticks(list(range(0,32,2))+[31])
    fig.savefig(HERE/'DOM_LAYER_SWEEP_20260921.png',dpi=180)
    fig.savefig(HERE/'DOM_LAYER_SWEEP_20260921.pdf')
    print(json.dumps({k:{v:{'layers':r['layers'],'coefficient':r['coefficient'],'jsd':js(r),
                          'removed':r['metrics']['escaped_phrase_count'],'clean_preserved':round(64*r['metrics']['clean_exact_match'])}
                        for v,r in group.items()} for k,group in [('simultaneous',simultaneous),('validation_best',val_best),('observed_best',observed_best)]},indent=2))


if __name__=='__main__':
    main()
