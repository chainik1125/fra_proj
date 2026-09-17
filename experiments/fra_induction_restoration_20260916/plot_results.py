"""Paired restoration KL versus original backdoor suppression."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import OUT, load_cases, eligible

runs = load_cases()
fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
for row, key in enumerate(['gpt2', 'gemma']):
    subset = [(case, meta) for k, case, meta in runs if k == key]
    for ax, (case, meta) in zip(axes[row], subset):
        for mode, color, label in [('activation_positive', '#168a55', 'Single SAE, positive c'),
                                    ('additive', '#a364bb', 'Single SAE, additive')]:
            points = [p for p in eligible(case, mode) if p['suppression'] >= -1e-5]
            frontier = [p for p in points if not any(q['suppression'] >= p['suppression']
                        and q['continuation_kl_mean'] < p['continuation_kl_mean'] for q in points)]
            ax.scatter([p['suppression'] for p in frontier], [p['continuation_kl_mean'] for p in frontier],
                       color=color, s=19, alpha=.8, label=label)
        fra = eligible(case, 'fra')
        ax.plot([p['suppression'] for p in fra], [p['continuation_kl_mean'] for p in fra], 'o-',
                color='#2167b6', lw=1.5, markersize=3, label='FRA, fresh evaluation')
        ax.axhline(case['unsteered']['continuation_kl_mean'], color='#c54a38', ls='--', lw=1,
                   label='Poison, no steering')
        ax.set_title(f"{key}: {case['trigger']} → {case['payload']}", fontsize=11)
        ax.set_yscale('log'); ax.set_xlim(-.02, 1.02); ax.grid(alpha=.2)
        if row == 1: ax.set_xlabel('Original backdoor suppression')
        if ax is axes[row, 0]: ax.set_ylabel('Restoration KL (nats/token)')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
fig.suptitle('Recovery toward clean continuations while the poisoned context is present', fontsize=15)
fig.text(.5, .935, 'Same continuation tokens on both sides; three matched primers per case. Lower KL is better.', ha='center', fontsize=10)
fig.tight_layout(rect=[0, .065, 1, .915])
for ext in ['png', 'pdf']: fig.savefig(OUT/f'paired_restoration.{ext}', dpi=170, bbox_inches='tight')
print(OUT/'paired_restoration.png')
