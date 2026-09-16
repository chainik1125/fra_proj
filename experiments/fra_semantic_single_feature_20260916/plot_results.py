"""Standalone scientific figure from the saved comparison summary."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from result_io import read_json

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
data = read_json(OUT/'summary.json')
concepts = ['vessel', 'vehicle', 'bird', 'fire', 'war', 'medical']
colors = dict(zip(concepts, ['#0072B2', '#56B4E9', '#009E73', '#E69F00', '#D55E00', '#CC79A7']))
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 180, 'font.family': 'DejaVu Sans'})
fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6))
for ax, t in zip(axes, [.3, .5, .7]):
    view = data['views']['measured/no_payload/activation_positive'][str(t)]
    stats = view['per_query_best']
    comparable = [r for r in view['queries'] if r['fra'] is not None and r['sae'] is not None]
    for concept in concepts:
        rows = [r for r in comparable if r['concept'] == concept]
        ax.scatter([r['fra']['kl'] for r in rows], [r['sae']['kl'] for r in rows],
                   s=35, c=colors[concept], alpha=.8, edgecolors='white', linewidth=.4)
    vals = [r[method]['kl'] for r in comparable for method in ['fra', 'sae']]
    high = max(vals)*1.2
    low = -high*.04
    ax.plot([0, high], [0, high], color='#555555', lw=1, linestyle='--')
    ax.set(xlim=(low, high), ylim=(low, high),
           xlabel='FRA collateral (summed KL, nats)',
           title=f'{t:.0%} suppression\nSAE lower on {stats["sae_lower"]}/{stats["both_reach"]} comparable queries')
    ax.grid(alpha=.18, which='major')
    ax.text(.03, .97, f'SAE reaches {stats["sae_reach"]}/52\nFRA reaches {stats["fra_reach"]}/52',
            transform=ax.transAxes, va='top', fontsize=9)
axes[0].set_ylabel('Best single SAE feature collateral (nats)')
fig.suptitle('Top-ten differential SAE features versus FRA cuts', y=1.01, fontsize=15)
handles = [Line2D([0], [0], marker='o', lw=0, color=colors[c], label=c) for c in concepts]
fig.legend(handles=handles, loc='lower center', ncol=6, bbox_to_anchor=(.5, -.025), frameon=False)
fig.text(.5, -.07, 'Below diagonal: SAE has less collateral. Best of ten features per query; positive activation-weighted steering only.\n'
         'Zero KL is exact: the features are inactive on the original legitimate paragraphs. Same layer 6 SAE site.',
         ha='center', fontsize=9)
fig.tight_layout()
for extension in ['png', 'pdf']:
    fig.savefig(OUT/f'single_feature_vs_fra.{extension}', bbox_inches='tight')
print(OUT/'single_feature_vs_fra.png')
