"""Standalone scientific figure; settings selected using tuning only."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze import load,key,OUT
r=load()
if (OUT/'diff_only.json.gz').exists():
    d=load('diff_only');r['selected'].extend(d['selected']);r['test_points'].extend(d['points'])
lookup={key(p['config']):p for p in r['test_points']}
selected=[]
for label,families in [('SAE: original diff',['sae_diff_positive']),('SAE: conjunction',['sae_positive']),('FRA QK',['fra_raw','fra_interaction','fra_separable'])]:
    choices=[s for s in r['selected'] if s['family'] in families and s['threshold']==.5 and s['config']]
    if choices:
        choice=min(choices,key=lambda x:x['tuning']['all']['kl'])
        selected.append((label,lookup[key(choice['config'])]['summary']))
points=[('Poisoned',r['test_baseline']['summary'])]+selected
colors=['#8c9197','#689ecc','#2563a6','#b34e26']
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
x=np.arange(4); width=.75/(len(points)+1)
clean=[r['test_baseline']['summary'][c]['clean_target_p'] for c in ['00','10','01','11']]
axes[0].bar(x-.375+width/2,clean,width,label='Clean',color='#2c8056')
for i,(label,s) in enumerate(points):
    axes[0].bar(x-.375+width*(i+1.5),[s[c]['target_p'] for c in ['00','10','01','11']],width,label=label,color=colors[i])
axes[0].set(xticks=x,xticklabels=['Computer\nPower','Printer\nPower','Computer\nWireless','Printer\nWireless'],ylabel='Mean P(Print)',ylim=(0,1.04),title='Repair the joint case; preserve Print on power faults')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,fontsize=9,ncol=5,frameon=False,loc='lower center',bbox_to_anchor=(.5,.035))
xx=np.arange(2); ww=.75/len(points)
for i,(label,s) in enumerate(points):
    axes[1].bar(xx-.375+ww*(i+.5),[s['11']['kl'],s['controls']['kl']],ww,label=label,color=colors[i])
axes[1].set(xticks=xx,xticklabels=['Joint trigger','Three controls'],ylabel='KL(clean || poisoned + edit), nats',title='Full-vocabulary next-token restoration')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
fig.suptitle('Compound semantic filtering · Gemma-2-9B-IT · 48 held-out cases',fontsize=12)
fig.text(.5,.01,'One fixed setting per method, selected on tuning for ≥50% joint suppression and minimum overall KL.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.12,1,.96))
for ext in ['png','pdf']: fig.savefig(OUT/f'compound_results.{ext}',dpi=170,bbox_inches='tight')
