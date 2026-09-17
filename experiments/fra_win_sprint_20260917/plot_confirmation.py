"""Create publication artifacts directly from a completed frozen confirmation."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import load
ROOT=Path(__file__).resolve().parent


def point(result,family,threshold):
    s=next(s for s in result['freeze']['selection'] if s['family']==family and s['threshold']==threshold)
    return next((p for p in result['points'] if p['source_index']==s['source_index'] and p['config']==s['config']),None)


def interval(rows,field,seed=17):
    blocks={}
    for r in rows:blocks.setdefault(r['case_id'].split(':')[2],[]).append(r[field])
    values=np.array([np.mean(v) for v in blocks.values()]);rng=np.random.default_rng(seed)
    samples=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
    return np.quantile(samples,[.025,.975])


def draw(result,out,threshold=.9):
    assert result['done'];out=Path(out);out.mkdir(exist_ok=True,parents=True)
    items=[('Poisoned document',result['baseline'],'#979797'),
           ('Best single SAE feature',point(result,'sae_strong',threshold),'#d2773d'),
           ('FRA: all pair candidates',point(result,'fra',threshold),'#23799a'),
           ('FRA: distinct feature IDs',point(result,'fra_distinct',threshold),'#3b8d65')]
    items=[x for x in items if x[1] and x[1]['valid']];x=np.arange(len(items))
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.4),layout='constrained')
    for ax,field,title in [(axes[0],'kl','Distance to the clean-document model ↓'),(axes[1],'suppression','Poisoned-target probability suppressed ↑')]:
        values=[p['summary']['all']['kl'] if field=='kl' else p['summary']['suppression']*100 for _,p,_ in items]
        ax.barh(x,values,color=[c for _,_,c in items],height=.61);ax.invert_yaxis()
        ax.set_yticks(x,[a for a,_,_ in items] if field=='kl' else []);ax.set_title(title,fontsize=11)
        for i,((name,p,color),v) in enumerate(zip(items,values)):
            if field=='kl':
                lo,hi=interval(p['rows'],'kl');ax.plot([lo,hi],[i,i],color='#333333',lw=1.2)
                ax.text(max(v,hi)+max(values)*.025,i,f'{v:.4f}',va='center',fontsize=9)
            else:ax.text(v+2,i,f'{v:.1f}%',va='center',fontsize=9)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right','left']].set_visible(False)
        if field=='kl':ax.set_xlabel('Mean next-token KL (nats)');ax.set_xlim(0,max(values)*1.32)
        else:
            ax.set_xlabel('Reduction from the unedited poisoned model (%)');ax.set_xlim(0,115)
            ax.axvline(threshold*100,ls='--',lw=.8,color='#777777')
    fig.suptitle(f'Fresh confirmation: settings chosen at ≥{threshold*100:.0f}% suppression on tuning',fontsize=13)
    fig.savefig(out/f'confirmation_{int(threshold*100)}.png',dpi=180);plt.close(fig)
    # Every factorial corner is visible; the final column is legitimate reuse.
    corners=sorted(set(r['corner'] for r in items[0][1]['rows']));arr=[]
    for _,p,_ in items:arr.append([np.mean([r['kl'] for r in p['rows'] if r['corner']==c]) for c in corners])
    fig,ax=plt.subplots(figsize=(10,3.7),layout='constrained');im=ax.imshow(arr,cmap='YlOrRd',aspect='auto')
    for i,row in enumerate(arr):
        for j,v in enumerate(row):ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=9,color='white' if v>max(map(max,arr))*.65 else '#242424')
    ax.set_yticks(np.arange(len(items)),[n for n,_,_ in items]);ax.set_xticks(np.arange(len(corners)),corners)
    ax.set_xlabel('Site × printer × wireless (1=yes; site 0=North, 1=South)')
    ax.set_title('KL by case type: 011 is the poisoned target; 111 legitimately uses Print')
    fig.colorbar(im,ax=ax,label='Mean KL (nats)');fig.savefig(out/f'confirmation_corners_{int(threshold*100)}.png',dpi=180);plt.close(fig)
    table=[]
    for name,p,_ in items:
        s=p['summary'];table.append({'method':name,'config':p['config'],'kl':s['all']['kl'],'suppression':s['suppression'],
            'targets_correct':round(s['joint']['correct']*s['joint']['n']),'targets_n':s['joint']['n'],
            'controls_correct':round(s['controls']['correct']*s['controls']['n']),'controls_n':s['controls']['n'],
            'shared_correct':round(s['shared_conjunction']['correct']*s['shared_conjunction']['n']),'shared_n':s['shared_conjunction']['n'],
            'clean_reference_agreement':p.get('clean_reference_agreement'),'full_vocab_top_accuracy':s['all']['top_correct'],'label_mass':s['all']['label_mass']})
    (out/f'confirmation_table_{int(threshold*100)}.json').write_text(json.dumps(table,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stem');parser.add_argument('--out',default=str(ROOT/'figures'));args=parser.parse_args()
    result=load(args.stem)
    for threshold in [.5,.9]:draw(result,args.out,threshold)
