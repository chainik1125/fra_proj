#!/usr/bin/env python3
"""
Generate all paper figures from stored multi-seed results.

Usage:
    python plot_paper_figures.py
    python plot_paper_figures.py --results-dir multiseed_results_v2 --output-dir paper/icml2026/figures
"""

import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


# ── Style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'figure.facecolor': 'white',
    'lines.linewidth': 2.5, 'lines.markersize': 10,
})

VARIANTS = ['finance', 'medical', 'sports']
TITLES = {'finance': 'Finance', 'medical': 'Medical', 'sports': 'Sports'}

FRONTIER_COLORS = {'qk_to_ov': '#1976D2', 'ov_to_ov': '#E64A19', 'qk_to_qk': '#388E3C'}
FRONTIER_LABELS = {'qk_to_ov': 'QK→OV', 'ov_to_ov': 'OV→OV', 'qk_to_qk': 'QK→QK'}
FRONTIER_MARKERS = {'qk_to_ov': 'o', 'ov_to_ov': 's', 'qk_to_qk': '^'}

SHARED_COLORS = {'ov_1head': '#1976D2', 'ov_allheads': '#E64A19', 'qk_allheads': '#388E3C'}
SHARED_LABELS = {'ov_1head': 'OV 1-head', 'ov_allheads': 'OV 4-heads', 'qk_allheads': 'QK all-heads'}
SHARED_MARKERS = {'ov_1head': 'o', 'ov_allheads': 's', 'qk_allheads': '^'}

CE_COLORS = {'finance': '#E64A19', 'medical': '#1976D2', 'sports': '#388E3C'}


def aggregate_gpt4o(qual_path):
    """Aggregate GPT-4o scores from a qualitative JSON file."""
    data = json.load(open(qual_path))
    buckets = defaultdict(lambda: {'aligns': [], 'cohers': []})
    for ex in data:
        if 'gpt4o_alignment' not in ex:
            continue
        parts = ex['condition'].rsplit('_a', 1)
        method = parts[0] if len(parts) == 2 else ex['condition']
        buckets[(method, ex['scale'])]['aligns'].append(ex['gpt4o_alignment'])
        buckets[(method, ex['scale'])]['cohers'].append(ex['gpt4o_coherence'])

    method_results = defaultdict(list)
    for (method, scale), vals in sorted(buckets.items()):
        method_results[method].append({
            'scale': scale,
            'mean_alignment': float(np.mean(vals['aligns'])),
            'std_alignment': float(np.std(vals['aligns'])),
            'mean_coherence': float(np.mean(vals['cohers'])),
            'std_coherence': float(np.std(vals['cohers'])),
            'n': len(vals['aligns']),
        })
    return dict(method_results)


def load_data(results_dir):
    """Load and aggregate all results."""
    results_dir = Path(results_dir)
    data = {'frontier': {}, 'shared': {}, 'ce': {}}

    for v in VARIANTS:
        # Frontier: prefer GPT-4o aggregated, else compute from qualitative
        gpt4o_path = results_dir / f'gpt4o_aggregated_{v}_L24_H38_k50.json'
        qual_path = results_dir / f'qualitative_{v}_L24_H38_k50.json'
        if gpt4o_path.exists():
            data['frontier'][v] = json.load(open(gpt4o_path))
        elif qual_path.exists():
            data['frontier'][v] = aggregate_gpt4o(qual_path)

        # Shared feature
        gpt4o_shared = results_dir / f'gpt4o_aggregated_{v}_shared_L24_f14738.json'
        qual_shared = results_dir / f'qualitative_{v}_shared_L24_f14738.json'
        if gpt4o_shared.exists():
            data['shared'][v] = json.load(open(gpt4o_shared))
        elif qual_shared.exists():
            data['shared'][v] = aggregate_gpt4o(qual_shared)

        # CE vs base
        for ce_path in results_dir.glob(f'ce_vs_base_{v}_*.json'):
            data['ce'][v] = json.load(open(ce_path))

    return data


# ── Figure 1: Frontier (alignment vs α with error bands) ──────────────

def plot_frontier_lines(data, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for col, variant in enumerate(VARIANTS):
        ax = axes[col]
        agg = data['frontier'].get(variant, {})
        if not agg:
            continue

        bl = agg.get('baseline', [{}])[0]
        if bl:
            bl_m = bl['mean_alignment']
            bl_s = bl['std_alignment']
            ax.axhspan(bl_m - bl_s, bl_m + bl_s, color='black', alpha=0.08)
            ax.axhline(bl_m, color='black', ls='-', lw=1.5, alpha=0.4)
            ax.text(2.8, bl_m + 2, 'baseline', fontsize=9, color='black', alpha=0.5, ha='right')

        for method in ['qk_to_ov', 'ov_to_ov', 'qk_to_qk']:
            entries = agg.get(method, [])
            if not entries:
                continue
            scales = [e['scale'] for e in entries]
            means = [e['mean_alignment'] for e in entries]
            stds = [e['std_alignment'] for e in entries]
            ax.plot(scales, means, f'{FRONTIER_MARKERS[method]}-',
                    color=FRONTIER_COLORS[method], label=FRONTIER_LABELS[method],
                    markersize=8, markeredgecolor='white', markeredgewidth=1)
            ax.fill_between(scales, np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=FRONTIER_COLORS[method], alpha=0.12)

        ax.axhline(y=50, color='red', ls=':', alpha=0.5, lw=1)
        ax.set_title(TITLES[variant], fontweight='bold')
        ax.set_xlabel('Steering α')
        ax.set_xlim(-0.1, 3.1)
        ax.set_ylim(10, 100)
        ax.grid(True, alpha=0.15)
        if col == 0:
            ax.set_ylabel('Alignment (GPT-4o)')
            ax.legend(loc='upper left', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_dir / 'v2_frontier_H38_k50.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  v2_frontier_H38_k50.png')


# ── Figure 2: Frontier means (coherence vs alignment scatter) ─────────

def plot_frontier_means(data, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    all_a, all_c = [], []
    for v in VARIANTS:
        agg = data['frontier'].get(v, {})
        for method in list(FRONTIER_COLORS.keys()) + ['baseline']:
            for e in agg.get(method, []):
                all_a.append(e['mean_alignment'])
                all_c.append(e['mean_coherence'])
    a_min, a_max = min(all_a) - 5, max(all_a) + 10
    c_min, c_max = min(all_c) - 5, max(all_c) + 10

    for col, variant in enumerate(VARIANTS):
        ax = axes[col]
        agg = data['frontier'].get(variant, {})
        if not agg:
            continue

        bl = agg.get('baseline', [{}])[0]
        if bl:
            ax.plot(bl['mean_coherence'], bl['mean_alignment'],
                    'D', color='black', markersize=14, zorder=10,
                    markeredgecolor='white', markeredgewidth=1.5)
            ax.annotate('unsteered\nEM model',
                        (bl['mean_coherence'], bl['mean_alignment']),
                        fontsize=9, color='black', fontweight='bold',
                        xytext=(-15, -20), textcoords='offset points', ha='center',
                        arrowprops=dict(arrowstyle='->', color='black', lw=1))

        for method in ['qk_to_ov', 'ov_to_ov', 'qk_to_qk']:
            entries = agg.get(method, [])
            if not entries:
                continue
            ma = [e['mean_alignment'] for e in entries]
            mc = [e['mean_coherence'] for e in entries]
            ax.plot(mc, ma, f'{FRONTIER_MARKERS[method]}-',
                    color=FRONTIER_COLORS[method], label=FRONTIER_LABELS[method],
                    markeredgecolor='white', markeredgewidth=1.2)
            ax.annotate('α=0', (mc[0], ma[0]), fontsize=8,
                        color=FRONTIER_COLORS[method], fontweight='bold',
                        xytext=(-12, -10), textcoords='offset points')
            ax.annotate('α=3', (mc[-1], ma[-1]), fontsize=8,
                        color=FRONTIER_COLORS[method], fontweight='bold',
                        xytext=(5, 5), textcoords='offset points')

        ax.axhline(y=50, color='red', ls=':', alpha=0.4, lw=1)
        ax.set_title(TITLES[variant], fontweight='bold', pad=10)
        ax.set_xlabel('Coherence →')
        ax.set_xlim(c_min, c_max)
        ax.set_ylim(a_min, a_max)
        ax.grid(True, alpha=0.15)
        if col == 0:
            ax.set_ylabel('Alignment →')
            ax.legend(loc='upper left', framealpha=0.95)

    stds = []
    for v in VARIANTS:
        for method in FRONTIER_COLORS:
            for e in data['frontier'].get(v, {}).get(method, []):
                stds.append(e['std_alignment'])
    avg_std = np.mean(stds) if stds else 0
    fig.text(0.5, -0.02,
             f'Mean values shown (avg std: ±{avg_std:.0f} alignment, '
             f'n=24 per point). Ideal = top-right.',
             ha='center', fontsize=10, style='italic', color='#555')

    plt.tight_layout()
    plt.savefig(output_dir / 'v2_frontier_means.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  v2_frontier_means.png')


# ── Figure 3: Best alignment bars ─────────────────────────────────────

def plot_best_alignment_bars(data, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(VARIANTS))
    width = 0.19
    all_colors = {'baseline': '#757575', **FRONTIER_COLORS}
    all_labels = {'baseline': 'Baseline', **FRONTIER_LABELS}
    method_order = ['baseline', 'qk_to_ov', 'ov_to_ov', 'qk_to_qk']

    for i, method in enumerate(method_order):
        vals = []
        for variant in VARIANTS:
            entries = data['frontier'].get(variant, {}).get(method, [])
            best = max((e['mean_alignment'] for e in entries), default=0)
            vals.append(best)
        bars = ax.bar(x + (i - 1.5) * width, vals, width, color=all_colors[method],
                      label=all_labels[method], edgecolor='white', linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{v:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.axhline(y=50, color='red', ls='--', alpha=0.4, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([TITLES[v] for v in VARIANTS], fontsize=12)
    ax.set_ylabel('Best Alignment (0–100)')
    ax.set_ylim(0, 95)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.15)
    ax.set_title('Best Alignment by Method (max over α)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'v2_best_alignment_bars.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  v2_best_alignment_bars.png')


# ── Figure 4: Shared feature (alignment vs α) ─────────────────────────

def plot_shared_feature(data, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for col, variant in enumerate(VARIANTS):
        ax = axes[col]
        agg = data['shared'].get(variant, {})
        if not agg:
            continue

        bl = agg.get('baseline', [{}])[0]
        if bl:
            bl_m = bl['mean_alignment']
            bl_s = bl['std_alignment']
            ax.axhspan(bl_m - bl_s, bl_m + bl_s, color='black', alpha=0.08)
            ax.axhline(bl_m, color='black', ls='-', lw=1.5, alpha=0.4)
            ax.text(2.8, bl_m + 2, 'baseline', fontsize=9, color='black', alpha=0.5, ha='right')

        for method in ['ov_1head', 'ov_allheads', 'qk_allheads']:
            entries = agg.get(method, [])
            if not entries:
                continue
            scales = [e['scale'] for e in entries]
            means = [e['mean_alignment'] for e in entries]
            stds = [e['std_alignment'] for e in entries]
            ax.plot(scales, means, f'{SHARED_MARKERS[method]}-',
                    color=SHARED_COLORS[method], label=SHARED_LABELS[method],
                    markersize=8, markeredgecolor='white', markeredgewidth=1)
            ax.fill_between(scales, np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=SHARED_COLORS[method], alpha=0.12)

        ax.axhline(y=50, color='red', ls=':', alpha=0.5, lw=1)
        ax.set_title(TITLES[variant], fontweight='bold')
        ax.set_xlabel('Steering α')
        ax.set_xlim(-0.1, 3.1)
        ax.set_ylim(10, 100)
        ax.grid(True, alpha=0.15)
        if col == 0:
            ax.set_ylabel('Alignment (GPT-4o)')
            ax.legend(loc='upper left', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_dir / 'v2_shared_feature.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  v2_shared_feature.png')


# ── Figure 5: Shared feature means (scatter) ──────────────────────────

def plot_shared_means(data, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    all_a, all_c = [], []
    for v in VARIANTS:
        agg = data['shared'].get(v, {})
        for method in list(SHARED_COLORS.keys()) + ['baseline']:
            for e in agg.get(method, []):
                all_a.append(e['mean_alignment'])
                all_c.append(e['mean_coherence'])
    a_min, a_max = min(all_a) - 5, max(all_a) + 10
    c_min, c_max = min(all_c) - 5, max(all_c) + 10

    for col, variant in enumerate(VARIANTS):
        ax = axes[col]
        agg = data['shared'].get(variant, {})
        if not agg:
            continue

        bl = agg.get('baseline', [{}])[0]
        if bl:
            ax.plot(bl['mean_coherence'], bl['mean_alignment'],
                    'D', color='black', markersize=14, zorder=10,
                    markeredgecolor='white', markeredgewidth=1.5)
            ax.annotate('unsteered\nEM model',
                        (bl['mean_coherence'], bl['mean_alignment']),
                        fontsize=9, color='black', fontweight='bold',
                        xytext=(-15, -20), textcoords='offset points', ha='center',
                        arrowprops=dict(arrowstyle='->', color='black', lw=1))

        for method in ['ov_1head', 'ov_allheads', 'qk_allheads']:
            entries = agg.get(method, [])
            if not entries:
                continue
            ma = [e['mean_alignment'] for e in entries]
            mc = [e['mean_coherence'] for e in entries]
            ax.plot(mc, ma, f'{SHARED_MARKERS[method]}-',
                    color=SHARED_COLORS[method], label=SHARED_LABELS[method],
                    markeredgecolor='white', markeredgewidth=1.2)
            ax.annotate('α=0', (mc[0], ma[0]), fontsize=8,
                        color=SHARED_COLORS[method], fontweight='bold',
                        xytext=(-12, -10), textcoords='offset points')
            ax.annotate('α=3', (mc[-1], ma[-1]), fontsize=8,
                        color=SHARED_COLORS[method], fontweight='bold',
                        xytext=(5, 5), textcoords='offset points')

        ax.axhline(y=50, color='red', ls=':', alpha=0.4, lw=1)
        ax.set_title(TITLES[variant], fontweight='bold', pad=10)
        ax.set_xlabel('Coherence →')
        ax.set_xlim(c_min, c_max)
        ax.set_ylim(a_min, a_max)
        ax.grid(True, alpha=0.15)
        if col == 0:
            ax.set_ylabel('Alignment →')
            ax.legend(loc='upper left', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_dir / 'v2_shared_means.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  v2_shared_means.png')


# ── Figure 6: CE vs base ──────────────────────────────────────────────

def plot_ce_vs_base(data, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for variant in VARIANTS:
        ce_data = data['ce'].get(variant)
        if not ce_data:
            continue
        agg = ce_data['aggregated']
        kl_baseline = np.mean([p['kl_base_vs_em'] for p in ce_data['per_prompt']])

        scales = sorted(agg.keys(), key=float)
        alphas = [float(s) for s in scales]
        kls = [agg[s]['mean_kl_base'] for s in scales]
        kl_em = [agg[s]['mean_kl_em'] for s in scales]

        ax1.plot(alphas, kls, 'o-', color=CE_COLORS[variant], label=TITLES[variant],
                 markersize=7, markeredgecolor='white', markeredgewidth=1)
        ax1.axhline(kl_baseline, color=CE_COLORS[variant], ls=':', alpha=0.3, lw=1)

        ax2.plot(alphas, kl_em, 'o-', color=CE_COLORS[variant], label=TITLES[variant],
                 markersize=7, markeredgecolor='white', markeredgewidth=1)

    ax1.set_xlabel('Steering α')
    ax1.set_ylabel('KL(base || steered)')
    ax1.set_title('Distance from base model\n(lower = closer to clean)')
    ax1.legend()
    ax1.grid(True, alpha=0.15)
    ax1.axvline(1.0, color='black', ls='--', alpha=0.3, lw=1)
    ax1.text(1.05, ax1.get_ylim()[1] * 0.98, 'no steering', fontsize=8, alpha=0.5, va='top')

    ax2.set_xlabel('Steering α')
    ax2.set_ylabel('KL(EM || steered)')
    ax2.set_title('Perturbation from EM model\n(higher = larger intervention effect)')
    ax2.legend()
    ax2.grid(True, alpha=0.15)
    ax2.axvline(1.0, color='black', ls='--', alpha=0.3, lw=1)

    plt.tight_layout()
    plt.savefig(output_dir / 'v2_ce_vs_base.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  v2_ce_vs_base.png')


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate paper figures from results')
    parser.add_argument('--results-dir', default='multiseed_results_v2')
    parser.add_argument('--output-dir', default='paper/icml2026/figures')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading results from {args.results_dir}...')
    data = load_data(args.results_dir)

    print(f'Generating figures in {output_dir}:')

    if data['frontier']:
        plot_frontier_lines(data, output_dir)
        plot_frontier_means(data, output_dir)
        plot_best_alignment_bars(data, output_dir)

    if data['shared']:
        plot_shared_feature(data, output_dir)
        plot_shared_means(data, output_dir)

    if data['ce']:
        plot_ce_vs_base(data, output_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
