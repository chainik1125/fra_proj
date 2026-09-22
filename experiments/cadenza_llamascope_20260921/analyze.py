"""Reproduce report tables, paired intervals and standalone figures from compact results."""
import csv
import lzma
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/cadenza-scope-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    data=json.loads(lzma.decompress((HERE/'results.json.xz').read_bytes()))
    assert not data['audit_errors']
    runs={r['id']:r for r in data['runs']}
    comparisons=[('local-L12-ov','positive_jsd','local-L12-single','signed_jsd'),
                 ('local-L12-ov','positive_jsd','local-L12-single','positive_jsd'),
                 ('scope-L08-8x-ov','positive_jsd','scope-L08-8x-single','signed_jsd'),
                 ('frozen-L08-ov','positive_jsd','scope-L08-8x-ov','positive_jsd'),
                 ('frozen-L08-ov','positive_jsd','local-L12-ov','positive_jsd'),
                 ('frozen-L08-ov','positive_jsd','frozen-dom-L11-resid_response','restoration')]
    pairs=[]
    for a,ar,b,br in comparisons:
        x,y=runs[a]['rows'][ar],runs[b]['rows'][br]
        assert [z['key'] for z in x]==[z['key'] for z in y]
        delta=np.array([i['triggered_to_clean_js_bits']-j['triggered_to_clean_js_bits'] for i,j in zip(x,y)])
        boot=delta[np.random.default_rng(42).integers(0,64,(20000,64))].mean(1)
        pairs.append(dict(a=a,a_rule=ar,b=b,b_rule=br,delta=float(delta.mean()),
            paired_bootstrap_95pct=np.quantile(boot,[.025,.975]).tolist(),n=64,bootstrap_draws=20000,seed=42))
    (HERE/'paired_comparisons.json').write_text(json.dumps(pairs,indent=2)+'\n')
    table=[]
    for run in data['runs']:
        for rule,d in run['summary'].get('confirmation',{}).get('results',{}).items():
            m=d['metrics']; rows=run['rows'][rule]
            # Independently recompute the key means from per-prompt measurements.
            for key in ('triggered_to_clean_js_bits','clean_drift_js_bits','clean_exact_match','sleeper_asr'):
                assert np.isclose(np.mean([r[key] for r in rows]),m[key],atol=1e-10)
            table.append(dict(task=run['id'],rule=rule,alpha=d['alpha'],
                features='/'.join(map(str,d['candidate'].get('features',[]))),
                jsd=m['triggered_to_clean_js_bits'],clean_preserved=round(64*m['clean_exact_match']),
                clean_drift=m['clean_drift_js_bits'],ihy_removed=m['escaped_phrase_count'],
                exact_restored=round(64*m['triggered_to_clean_exact_match']),
                blank_triggered=sum(r['blank_triggered'] for r in rows),
                ci_low=d['js_bits_bootstrap_95pct'][0],ci_high=d['js_bits_bootstrap_95pct'][1]))
    with (HERE/'all_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
    lines=['# All fresh-confirmation results','',
        'Each row uses a setting frozen on validation before testing. Lower JSD and clean drift are better. Counts are out of 64.',
        '', '| Task | Rule | Feature(s) | α | JSD (95% bootstrap CI) | Clean preserved | Clean drift | IHY removed | Exact restored |',
        '|---|---|---|---:|---|---:|---:|---:|---:|']
    for t in table:
        lines.append(f"| {t['task']} | {t['rule']} | {t['features'] or 'DoM'} | {t['alpha']:g} | {t['jsd']:.4f} [{t['ci_low']:.4f}, {t['ci_high']:.4f}] | {t['clean_preserved']} | {t['clean_drift']:.3g} | {t['ihy_removed']} | {t['exact_restored']} |")
    (HERE/'all_results.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    chosen=[('frozen-L08-ov','positive_jsd','Local L8: OV-FRA'),
            ('scope-L08-8x-ov','positive_jsd','Scope 32K → L8: OV-FRA'),
            ('frozen-dom-L11-resid_response','restoration','DoM: resid_post 11 + response'),
            ('local-L12-ov','positive_jsd','Local L12: OV-FRA'),
            ('local-L12-single','signed_jsd','Local L12: single feature (signed)'),
            ('scope-L08-8x-single','signed_jsd','Scope 32K, resid_post 7: single (signed)')]
    fig,axs=plt.subplots(1,2,figsize=(13,4.9),gridspec_kw={'width_ratios':[2.6,1]},sharey=True)
    colors=['#0072B2','#0072B2','#777777','#0072B2','#D55E00','#D55E00']
    for i,(name,rule,label) in enumerate(chosen):
        d=runs[name]['summary']['confirmation']['results'][rule];m=d['metrics'];mean=m['triggered_to_clean_js_bits'];lo,hi=d['js_bits_bootstrap_95pct']
        axs[0].errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=colors[i],capsize=4,ms=8)
        count=round(64*m['clean_exact_match']);axs[1].barh(i,count,color=colors[i],height=.5)
        axs[1].text(count+1,i,f'{count}/64',va='center')
    axs[0].set_yticks(range(len(chosen)),[x[2] for x in chosen]);axs[0].invert_yaxis()
    axs[0].set_xlim(.66,1.005);axs[0].axvline(.9882687712088227,color='#777777',ls=':',label='Unsteered (0.988)')
    axs[0].set_xlabel('Triggered-to-clean rollout JSD (bits; lower is better)');axs[0].legend(loc='lower right',fontsize=9)
    axs[1].set_xlim(0,79);axs[1].set_xticks([0,32,64]);axs[1].set_xlabel('Unchanged clean outputs\n(higher is better)')
    fig.suptitle('Layer-8 OV-FRA reduces divergence and preserves clean outputs',fontsize=15)
    fig.text(.5,.015,'Same 64 fresh prompt pairs. Features and strengths selected on validation; bars show 95% prompt-bootstrap intervals.\nRows illustrate named comparisons, not a new test-selected deployment policy.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.085,1,.94]);fig.savefig(HERE/'comparison.png');fig.savefig(HERE/'comparison.pdf');plt.close(fig)
    # Show all six official SAE cells under each predeclared selection rule.
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex='col')
    cell_names=[(l,e) for l in (8,12,16) for e in (8,32)]
    for ri,rule in enumerate(('positive_jsd','signed_jsd')):
        for method,offset,color,marker in [('single',-.18,'#D55E00','s'),('ov',0,'#0072B2','o'),('qkov',.18,'#009E73','^')]:
            ds=[runs[f'scope-L{l:02d}-{e}x-{method}']['summary']['confirmation']['results'][rule]['metrics'] for l,e in cell_names]
            x=np.arange(6)+offset
            axs[ri,0].scatter(x,[d['triggered_to_clean_js_bits'] for d in ds],c=color,marker=marker,label={'single':'Single residual feature','ov':'OV-FRA','qkov':'QK+OV-FRA'}[method],s=55)
            axs[ri,1].scatter(x,[64*d['clean_exact_match'] for d in ds],c=color,marker=marker,s=55)
        axs[ri,0].set_ylabel(f"{'Positive-only' if ri==0 else 'Signed'} selection\nJSD (bits; lower is better)")
        axs[ri,0].set_ylim(.74,1.01);axs[ri,0].axhline(.9882687712088227,color='#777777',ls=':',lw=1)
        axs[ri,1].set_ylabel('Clean outputs preserved / 64');axs[ri,1].set_ylim(-3,69)
        for ax in axs[ri]:
            ax.set_xticks(range(6),[f'RP{l-1} → A{l}\n{e*4}K' for l,e in cell_names],fontsize=9);ax.grid(axis='y',alpha=.2)
    axs[0,0].legend(fontsize=9,loc='lower right')
    fig.suptitle('The strong official-SAE result is concentrated at resid_post 7, 32K → attention 8',fontsize=13)
    fig.text(.5,.02,'All 18 official-SAE searches; same fresh 64-pair confirmation set. RP = residual post; A = receiving attention layer.\nBoth selection rules were frozen before testing. Points show observed means, without multiplicity-adjusted intervals.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.075,1,.95]);fig.savefig(HERE/'official_grid.png');fig.savefig(HERE/'official_grid.pdf');plt.close(fig)
    print(json.dumps({'result_rows':len(table),'paired_comparisons':len(pairs),'figures':2,'all_metrics_recomputed':True}))


if __name__=='__main__':
    main()
