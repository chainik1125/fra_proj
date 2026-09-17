"""Plot measured continuation-recovery tradeoffs for a single fixed edit."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import CONCEPTS, load, candidates

ROOT = Path(__file__).resolve().parent
fig, axes = plt.subplots(2, 3, figsize=(12, 7.3), constrained_layout=True)
colors = {'activation_positive': '#1765a3', 'fra': '#d67b16'}
labels = {'activation_positive': 'Single SAE: best of ten', 'fra': 'FRA QK cuts'}
for name, ax in zip(CONCEPTS, axes.flat):
    result = load(name)
    indices = [i for i, q in enumerate(result['queries']) if q['kind'] == 'synonym']
    for mode in ['activation_positive', 'fra']:
        points = candidates(result, mode)
        # A point qualifies only if EVERY related-word query reaches the threshold.
        pairs = [(min(p['suppression'][i] for i in indices), p['continuation_kl_mean']) for p in points]
        xs = sorted({0.0} | {max(0.0, x) for x, y in pairs if x >= 0})
        ys = [min(y for x, y in pairs if x >= threshold-1e-12) for threshold in xs]
        ax.step(xs, ys, where='pre', color=colors[mode], linewidth=2, label=labels[mode])
        ax.scatter(xs, ys, color=colors[mode], s=12)
    ax.axhline(result['unsteered']['continuation_kl_mean'], color='#a72831', linestyle='--', linewidth=1.4,
               label='Poisoned, no steering')
    ax.set_title(name.capitalize())
    ax.set_yscale('log'); ax.set_xlim(0, 1)
    ax.set_xlabel('Suppression required on every related query')
    ax.set_ylabel('Continuation KL (nats/token)')
    ax.grid(alpha=.2)
axes[0, 0].legend(fontsize=8, loc='best')
fig.suptitle('Semantic filtering: clean-reference recovery with poisoned context present\n'
             'One fixed feature/coefficient per concept; mean over three continuations', fontsize=13)
for extension in ['png', 'pdf']:
    fig.savefig(ROOT/'results'/f'paired_restoration.{extension}', dpi=180)
