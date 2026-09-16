"""Scientific comparison figure using measured points only."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import OUT, read_json, fra_points, candidates

fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
for row, key in enumerate(['gpt2', 'gemma']):
    run = read_json(OUT/f'{key}.json.gz')
    for ax, case in zip(axes[row], run['cases']):
        for mode, color, label in [('legacy12', '#999999', 'Top-12 together'),
                                    ('activation_positive', '#168a55', 'Single SAE, positive c'),
                                    ('additive', '#a364bb', 'Single SAE, additive')]:
            points = candidates(case, mode)
            visible = [p for p in points if 0 <= p['suppression'] <= 1]
            frontier = [p for p in visible if not any(q['suppression'] >= p['suppression'] and q['kl'] < p['kl'] for q in visible)]
            ax.scatter([p['suppression'] for p in frontier], [p['kl'] for p in frontier],
                       s=17, alpha=.8, color=color, label=label)
        fra = fra_points(key, case)
        ax.plot([p['suppression'] for p in fra], [p['kl'] for p in fra], 'o-', color='#2167b6',
                lw=1.5, markersize=3, label='FRA, archived')
        ax.set_title(f"{key}: {case['trigger']} → {case['payload']}", fontsize=11)
        ax.set_yscale('symlog', linthresh=.001)
        ax.set_xlim(-.02, 1.02)
        ax.set_ylim(bottom=-.0001)
        ax.grid(alpha=.2)
        if row == 1:
            ax.set_xlabel('Payload-probability suppression')
        if ax is axes[row, 0]:
            ax.set_ylabel('Collateral KL (nats)')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
fig.suptitle('Original induction backdoors: best of ten differential SAE features', fontsize=15)
fig.text(.5, .935, 'SAE dots minimize measured KL at or above their suppression; FRA lines connect its archived coefficient grid.', ha='center', fontsize=10)
fig.tight_layout(rect=[0, .065, 1, .915])
for extension in ['png', 'pdf']:
    fig.savefig(OUT/f'single_feature_vs_fra.{extension}', dpi=170, bbox_inches='tight')
print(OUT/'single_feature_vs_fra.png')
