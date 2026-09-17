"""Read sub-MB result shards; compare tuning-frozen settings and make figures."""
from pathlib import Path
import argparse,gzip,json
import numpy as np
ROOT=Path(__file__).resolve().parent


def load(stem):
    path=Path(stem)
    if path.exists() and path.suffix=='.gz':return json.load(gzip.open(path,'rt'))
    single=path.with_name(path.name+'.json.gz')
    if single.exists():return json.load(gzip.open(single,'rt'))
    parts=sorted(path.parent.glob(path.name+'.part*.json.gz'))
    assert parts,f'No result for {stem}'
    rr=[json.load(gzip.open(p,'rt')) for p in parts]
    assert len(rr)==rr[0]['parts']
    assert [r['part'] for r in rr]==list(range(len(rr)))
    return json.loads(''.join(r['json_fragment'] for r in rr))


def cfgkey(config):return json.dumps(config,sort_keys=True)


def merged_selection(results):
    """Choose only from tuning scores, including all completed search families."""
    points=[{**p,'source_index':i} for i,r in enumerate(results) for p in r['points'] if p['valid']]
    selection=[]
    for family in ['sae_diff','sae_strong','fra']:
        candidates=[p for p in points if (family=='fra' and p['config']['method'].startswith('fra')) or
            (family.startswith('sae') and p['config']['method']=='sae' and
             (family=='sae_strong' or 'diff' in p['config'].get('ranks',[])))]
        for threshold in [None,0.,.5,.9]:
            eligible=[p for p in candidates if threshold is None or
                (p['summary']['controls']['correct']>=.95 and p['summary']['suppression']>=threshold)]
            winner=min(eligible,key=lambda p:(p['summary']['all']['kl'],abs(p['config']['strength']),cfgkey(p['config']))) if eligible else None
            selection.append({'family':family,'threshold':threshold,'source_index':winner['source_index'] if winner else None,
                'config':winner['config'] if winner else None,'tuning':winner['summary'] if winner else None})
    return selection


def bootstrap(a,b,draws=10000):
    """Paired bootstrap over lexical indices; retain all corners and layouts."""
    aa={r['case_id']:r for r in a};bb={r['case_id']:r for r in b};assert aa.keys()==bb.keys()
    blocks={}
    for case in aa:
        block=case.split(':')[2];blocks.setdefault(block,[]).append(case)
    per_block=np.array([[np.mean([aa[c]['kl'] for c in cc]),np.mean([bb[c]['kl'] for c in cc])] for cc in blocks.values()])
    rng=np.random.default_rng(20260917);indices=rng.integers(0,len(blocks),size=(draws,len(blocks)))
    sampled=per_block[indices].mean(1);ratio=sampled[:,0]/sampled[:,1]
    return {'blocks':len(blocks),'draws':draws,'mean_kl_ratio':float(per_block[:,0].mean()/per_block[:,1].mean()),
            'ratio_ci95':np.quantile(ratio,[.025,.975]).tolist(),
            'difference_ci95':np.quantile(sampled[:,0]-sampled[:,1],[.025,.975]).tolist()}


def comparison(fra,sae,threshold):
    f,s=fra['summary'],sae['summary'];repair_ok=threshold is not None and f['suppression']>=threshold and s['suppression']>=threshold
    controls_ok=f['controls']['correct']>=.95 and s['controls']['correct']>=.95
    accuracy_ok=f['joint']['correct']>=s['joint']['correct']-.02
    ratio=f['all']['kl']/s['all']['kl'];interval=bootstrap(fra['rows'],sae['rows'])
    return {'threshold':threshold,'fra_kl':f['all']['kl'],'sae_kl':s['all']['kl'],'ratio':ratio,
        'both_meet_repair_threshold':repair_ok,'both_preserve_controls':controls_ok,'target_accuracy_comparable':accuracy_ok,
        'meets_prespecified_effect_size':bool(repair_ok and controls_ok and accuracy_ok and ratio<=.8),
        'bootstrap':interval}


def tuning_plot(results,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'sae':'#ce7845','fra':'#2878a6','fra_refined':'#5d509d'}
    labels={'sae':'Single SAE feature','fra':'FRA: one layer','fra_refined':'FRA: live across layers'}
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    points=[p for r in results for p in r['points'] if p['valid'] and p['summary']['controls']['correct']>=.95]
    for method in colors:
        pp=[p for p in points if p['config']['method']==method]
        if not pp:continue
        x=np.array([p['summary']['suppression'] for p in pp]);y=np.array([p['summary']['all']['kl'] for p in pp])
        valid=(x>=-.05)&(x<=1.001)&(y>0)
        ax.scatter(x[valid]*100,y[valid],s=12,alpha=.28,color=colors[method],label=labels[method])
        frontier=[];best=np.inf
        for p in sorted(pp,key=lambda p:-p['summary']['suppression']):
            xx=p['summary']['suppression'];yy=p['summary']['all']['kl']
            if 0<=xx<=1 and 0<yy<best:frontier.append((xx*100,yy));best=yy
        if frontier:ax.plot(*zip(*frontier),color=colors[method],lw=1.5)
    ax.set(xlabel='Suppression of poisoned-target probability (%) →',ylabel='Mean KL to the clean-document model (nats) ↓',
           title='Tuning trade-off: settings that preserve ≥95% of control answers',xlim=(-3,103),yscale='log')
    for x in [50,90]:ax.axvline(x,color='#999999',lw=.7,ls='--')
    ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stems',nargs='+');parser.add_argument('--out',default=str(ROOT/'analysis'))
    args=parser.parse_args();results=[load(s) for s in args.stems];out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    selected=merged_selection(results);(out/'tuning_selection.json').write_text(json.dumps(selected,indent=2))
    tuning_plot(results,out/'tuning_tradeoff.png')
    for s in selected:
        print(s['family'],s['threshold'],s['config'],None if s['tuning'] is None else
            {k:s['tuning'][k] for k in ['suppression','excess_repair']})
    print('Valid points:',sum(p['valid'] for r in results for p in r['points']))


if __name__=='__main__':main()
