"""Plot every prespecified pair budget, including missing and failed settings."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import load
from plot_confirmation import point as primary_point, interval


def draw(result, primary, dest):
    assert result['done'] and primary['done']
    budgets = result['meta']['budgets']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), layout='constrained')
    for ax, threshold in zip(axes, [.5, .9]):
        baseline = primary_point(primary, 'sae_strong', threshold)
        if baseline and baseline['valid']:
            ax.axhline(baseline['summary']['all']['kl'], color='#d2773d', ls='--',
                       label='Best single SAE feature')
        for family, label, color, offset in [('fra', 'All QK candidates', '#23799a', -.06),
                                             ('fra_distinct', 'Distinct-ID QK pairs', '#3b8d65', .06)]:
            plotted = []
            missing = []
            for i, budget in enumerate(budgets):
                s = next(s for s in result['selected'] if s['family'] == family
                         and s['budget'] == budget and s['threshold'] == threshold)
                p = next((p for p in result['points'] if p['source_index'] == s['source_index']
                          and p['config'] == s['config']), None)
                if not p or not p['valid']:
                    missing.append(i+offset)
                    continue
                a = p['summary']
                passes = a['suppression'] >= threshold and a['controls']['correct'] >= .95
                y = a['all']['kl']
                lo, hi = interval(p['rows'], 'kl')
                ax.plot([i+offset, i+offset], [lo, hi], color=color, alpha=.6)
                ax.scatter(i+offset, y, marker='o' if passes else 'x', color=color, s=48)
                plotted.append((i+offset, y))
            if plotted:
                ax.plot(*zip(*plotted), color=color, alpha=.65, lw=1, label=label)
            for x in missing:
                ax.text(x, .02, '∅', transform=ax.get_xaxis_transform(), color=color,
                        ha='center', va='bottom', fontsize=12)
        ax.set(yscale='log', xticks=np.arange(len(budgets)), xticklabels=budgets,
               xlabel='Maximum number of QK pairs', ylabel='Mean next-token KL to clean output (nats) ↓',
               title=f'≥{threshold*100:.0f}% tuning suppression')
        ax.grid(axis='y', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Secondary confirmation: performance versus the allowed number of pairs\n'
                 '○ meets repair and control requirements; × misses one; ∅ no eligible tuning setting', fontsize=12)
    fig.savefig(dest, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result')
    parser.add_argument('primary')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    dest = Path(args.out)
    dest.parent.mkdir(exist_ok=True, parents=True)
    draw(load(args.result), load(args.primary), dest)
