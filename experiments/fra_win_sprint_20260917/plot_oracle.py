"""Reproduce the source-row oracle diagnostic figure from committed results."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import load
ROOT=Path(__file__).resolve().parent
result=load(ROOT/'results/oracle');task=next(t for t in result['tasks'] if t['task']=='tenants_long')
names=['top_4','top_16','top_64','six_sae_sites'];labels=['Top 4\nheads','Top 16\nheads','Top 64\nheads','6 SAE layers\n(96 heads)']
points=[next(p for p in task['points'] if p['config']['group']==n and p['config']['key_kind']=='row' and p['config']['queries']=='all_later') for n in names]
fig,axes=plt.subplots(1,2,figsize=(8.8,4.1),gridspec_kw={'wspace':.38})
x=np.arange(len(points));colors=['#8ea5b2','#548eaa','#187d96','#a08bba']
for ax,key,title,scale in [(axes[0],'kl','Mean KL to clean-document output ↓',1),(axes[1],'suppression','Poisoned-target suppression (%) ↑',100)]:
    ys=[p['summary']['all'][key] if key=='kl' else p['summary'][key]*scale for p in points]
    ax.bar(x,ys,color=colors,width=.66)
    for i,y in enumerate(ys):ax.text(i,y+(.003 if key=='kl' else 1.5),f'{y:.3f}' if key=='kl' else f'{y:.1f}%',ha='center',fontsize=9)
    ax.set_xticks(x,labels,fontsize=9);ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    if key=='kl':ax.set_ylabel('Nats');ax.set_ylim(0,.1)
    else:ax.set_ylim(0,105)
fig.suptitle('Deleting attention to the known corrupted row can restore the answer',fontsize=13,y=1.01)
fig.text(.5,-.015,'Position-gated oracle diagnostic • 32 calibration incidents • Gemma-2-9B-IT',ha='center',fontsize=10)
fig.subplots_adjust(bottom=.18,top=.83);fig.savefig(ROOT/'figures/source_oracle.png',dpi=170,bbox_inches='tight');plt.close(fig)
