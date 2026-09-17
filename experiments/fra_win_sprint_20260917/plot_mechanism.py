"""Plot measured causal diagnostics, preserving their exploratory status."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import load
ROOT=Path(__file__).resolve().parent
r=load(ROOT/'results/mechanism_pre_tenants_long')
by={p['name']:p for p in r['points']}
entries=[('Full 48-pair edit','all'),('Exclude beginning-of-sequence edges','no_bos'),
         ('Only beginning-of-sequence edges','bos'),('Only source-row reads from later tokens','source'),
         ('Only final-position queries','answer')]
fig,ax=plt.subplots(figsize=(9.8,4.4),layout='constrained');ys=np.arange(len(entries)+1)
values=[100*by[k]['summary']['suppression'] for _,k in entries]
ax.barh(ys[:-1],values,color=['#23799a','#398e6f','#aeaeae','#949fa5','#949fa5'],height=.6)
for i,v in enumerate(values):ax.text(v+1.5,i,f'{v:.1f}%',va='center',fontsize=10)
shuffled=[p['summary']['suppression']*100 for p in r['points'] if p['name'].startswith('shuffled_')]
ax.scatter(shuffled,[ys[-1]]*len(shuffled),color='#aa7246',s=36,zorder=3)
ax.plot([min(shuffled),max(shuffled)],[ys[-1]]*2,color='#aa7246',lw=1)
ax.text(max(shuffled)+2,ys[-1],f'{min(shuffled):.1f}–{max(shuffled):.1f}%',va='center',fontsize=10)
ax.set_yticks(ys,[a for a,k in entries]+['Shuffle K-feature assignments (5 seeds)']);ax.invert_yaxis()
ax.set_xlim(0,105);ax.set_xlabel('Suppression of poisoned Print probability (%) →')
ax.set_title('Repair survives excluding BOS and weakens when feature pairs are shuffled\n32 exploratory test cases · Gemma-2-9B-IT',fontsize=12,pad=14)
ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True);ax.spines[['top','right','left']].set_visible(False)
fig.savefig(ROOT/'figures/mechanism_global.png',dpi=175);plt.close(fig)
