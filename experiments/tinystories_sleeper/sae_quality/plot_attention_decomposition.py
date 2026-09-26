import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=Path('/archive/fra_table3_retrain_20260925/attention_decomposition')
d=json.loads((r/'summary.json').read_text())
cases=[('sae_only','SAE reconstruction only','#1f77b4'),
       ('drop_q_error','Omit query-error QK','#ff7f0e'),
       ('drop_k_error','Omit key-error QK','#2ca02c'),
       ('drop_qk_error_error','Omit error-error QK','#9467bd'),
       ('drop_ov_error','Omit OV error','#d62728')]
scopes=[('all_prompt','All prompt positions'),('deployment_marker','Deployment story boundary')]
fig,axes=plt.subplots(1,2,figsize=(11.2,4.1),sharey=True)
for ax,(scope,title) in zip(axes,scopes):
    for case,label,color in cases:
        vals=np.array([[s[scope]['one_minus_fvu'][case] for s in d['layers'][str(layer)]['seeds']]
                       for layer in range(4)])
        mean=vals.mean(1)
        sd=vals.std(1,ddof=1)
        ax.errorbar(range(4),mean,yerr=sd,label=label,color=color,marker='o',
                    markersize=5,linewidth=1.8,capsize=3)
    ax.axhline(1,color='black',lw=1,ls=':',alpha=.65)
    ax.set_title(title)
    ax.set_xticks(range(4),[f'L{i}' for i in range(4)])
    ax.set_xlabel('Transformer layer')
    ax.grid(axis='y',alpha=.25)
    ax.set_ylim(.57,1.025)
axes[0].set_ylabel('1 − FVU of attention output')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=3,loc='lower center',frameon=False,bbox_to_anchor=(.5,-.04))
fig.suptitle('Attention decomposition: effect of omitting SAE reconstruction-error terms',
             fontsize=12,y=1.02)
fig.tight_layout(rect=(0,.09,1,1))
fig.savefig(r/'figure2_panel3_attention_decomposition.png',dpi=300,bbox_inches='tight')
print(r/'figure2_panel3_attention_decomposition.png')
