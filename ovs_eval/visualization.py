import matplotlib
matplotlib.use('Agg') # Safe for headless execution

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def compute_bins(mious, bin_size):
    """Return bin centres, bin means, bin stds."""
    n = len(mious)
    edges = np.arange(0, n, bin_size)
    centres = edges + bin_size / 2
    means = [np.mean(mious[s:s + bin_size]) for s in edges]
    stds = [np.std(mious[s:s + bin_size]) for s in edges]
    return centres, np.array(means), np.array(stds)

def rolling_mean(arr, w=15):
    return np.convolve(arr, np.ones(w) / w, mode='same')

def plot_taxonomy_evaluation(results, save_path="taxonomy_ovs_evaluation.png"):
    """
    Renders and saves the main evaluation results summary plot.
    """
    colors = {
        'Original':  '#1f77b4',
        'Synonyms':  '#ff7f0e',
        'Hypernyms': '#2ca02c',
        'Hyponyms':  '#d62728'
    }
    CAT_TYPES = ['Original', 'Synonyms', 'Hypernyms', 'Hyponyms']
    BIN_SIZE = 20

    summary_stats = {}
    for cat_type, results_list in results.items():
        mious = np.array([r['miou'] for r in results_list])
        summary_stats[cat_type] = {
            'mious': mious,
            'mean':  np.mean(mious),
            'std':   np.std(mious),
            'min':   np.min(mious),
            'max':   np.max(mious),
            'count': len(mious),
        }

    n_images = summary_stats['Original']['count']
    if n_images == 0:
        print("No evaluation data to plot.")
        return

    # Adjust bin size if we have very few images
    local_bin_size = min(BIN_SIZE, max(1, n_images // 2))

    fig = plt.figure(figsize=(18, 14))
    gs_outer = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32,
                                  height_ratios=[1.6, 1.6, 1])

    # ── [TOP] Binned mean + ribbon ──
    ax_top = fig.add_subplot(gs_outer[0, :])
    for cat in CAT_TYPES:
        mious = summary_stats[cat]['mious']
        centres, bmeans, bstds = compute_bins(mious, local_bin_size)
        c = colors[cat]
        ax_top.plot(centres, bmeans, marker='o', ms=5, lw=2.2, label=cat, color=c)
        ax_top.fill_between(centres, bmeans - bstds, bmeans + bstds, alpha=0.18, color=c)

    ax_top.set_xlabel('Image Index (bin centre)', fontsize=11, fontweight='bold')
    ax_top.set_ylabel('mIoU Score', fontsize=11, fontweight='bold')
    ax_top.set_title(
        f'mIoU Performance — Binned Means ± 1 Std Dev  (bin size = {local_bin_size})',
        fontsize=13, fontweight='bold')
    ax_top.legend(fontsize=10, loc='upper right')
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.set_xlim(0, n_images)
    ax_top.grid(True, alpha=0.3, linestyle='--')

    # ── [MID] Small multiples ──
    gs_mid = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[1, :], wspace=0.08)
    axes_sm = [fig.add_subplot(gs_mid[i]) for i in range(4)]

    for idx, cat in enumerate(CAT_TYPES):
        ax = axes_sm[idx]
        mious = summary_stats[cat]['mious']
        x = np.arange(len(mious))
        
        # Adjust rolling window if too small
        local_w = min(15, max(1, len(mious) // 2))
        smooth = rolling_mean(mious, w=local_w)
        c = colors[cat]

        ax.scatter(x, mious, s=3, alpha=0.18, color=c, linewidths=0)
        ax.plot(x, smooth, lw=2, color=c, label=f'Rolling mean (w={local_w})')
        ax.axhline(summary_stats[cat]['mean'], color='black', lw=1.2, ls=':', alpha=0.7,
                   label=f"Mean={summary_stats[cat]['mean']:.3f}")

        ax.set_title(cat, fontsize=11, fontweight='bold', color=c)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, n_images)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc='upper right')

        if idx == 0:
            ax.set_ylabel('mIoU Score', fontsize=10, fontweight='bold')
        else:
            ax.set_yticklabels([])

        ax.set_xlabel('Image Index', fontsize=9)

    axes_sm[1].set_title('Small Multiples — Raw Scatter + Smoothed Trend',
                          fontsize=11, fontweight='bold', pad=18, x=1.05)

    # ── [BOT-LEFT] Bar chart with std dev caps ──
    ax_bar = fig.add_subplot(gs_outer[2, 0])
    means = [summary_stats[ct]['mean'] for ct in CAT_TYPES]
    stds = [summary_stats[ct]['std'] for ct in CAT_TYPES]
    bars = ax_bar.bar(CAT_TYPES, means, yerr=stds, capsize=8,
                       color=[colors[ct] for ct in CAT_TYPES],
                       alpha=0.75, edgecolor='black', linewidth=1.3)

    for bar, mean in zip(bars, means):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{mean:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax_bar.set_ylabel('Mean mIoU', fontsize=11, fontweight='bold')
    ax_bar.set_title('Average Performance (with Std Dev)', fontsize=11, fontweight='bold')
    ax_bar.set_ylim(0, 1.1)
    ax_bar.grid(True, alpha=0.3, axis='y', linestyle='--')

    # ── [BOT-RIGHT] Summary table ──
    ax_tbl = fig.add_subplot(gs_outer[2, 1])
    ax_tbl.axis('off')

    header = ['Category', 'Mean', 'Std Dev', 'Min', 'Max', 'N']
    rows = []
    for ct in CAT_TYPES:
        s = summary_stats[ct]
        rows.append([ct,
                     f"{s['mean']:.4f}", f"{s['std']:.4f}",
                     f"{s['min']:.4f}",  f"{s['max']:.4f}",
                     str(s['count'])])

    tbl = ax_tbl.table(cellText=rows, colLabels=header, cellLoc='center', loc='center',
                       colWidths=[0.20, 0.14, 0.14, 0.14, 0.14, 0.10])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.6)

    for j in range(len(header)):
        tbl[(0, j)].set_facecolor('#4472C4')
        tbl[(0, j)].set_text_props(weight='bold', color='white')

    for i, ct in enumerate(CAT_TYPES, start=1):
        for j in range(len(header)):
            tbl[(i, j)].set_facecolor('#E8F4FD' if i % 2 == 0 else '#F8F8F8')

    ax_tbl.set_title('Performance Summary Statistics', fontsize=11, fontweight='bold', pad=14)

    plt.suptitle('Taxonomy-Driven OVS Model Evaluation Results', fontsize=15, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_lss_analysis(LSS_M, lss_per_class_results, save_path="lss_line_chart.png"):
    """
    Renders and saves the Linguistic Sensitivity Analysis plot.
    """
    VARIANTS = ['Original', 'Synonyms', 'Hypernyms', 'Hyponyms']
    
    if not lss_per_class_results:
        print("No LSS class results to plot.")
        return

    sorted_classes = sorted(
        lss_per_class_results.items(),
        key=lambda x: x[1]['LSS'] if x[1]['LSS'] is not None else 0.0,
        reverse=True
    )

    cls_names = [c[0] for c in sorted_classes]
    lss_vals = np.array([c[1]['LSS'] for c in sorted_classes])
    mu_vals = np.array([c[1]['mu_M_c'] for c in sorted_classes])
    x = np.arange(len(cls_names))

    variant_vals = {
        v: np.array([c[1]['mu_per_variant'][v] for c in sorted_classes])
        for v in VARIANTS
    }

    VARIANT_COLORS = {
        'Original':  '#1f77b4',
        'Synonyms':  '#ff7f0e',
        'Hypernyms': '#2ca02c',
        'Hyponyms':  '#d62728',
    }
    LSS_COLOR = '#9467bd'
    MU_COLOR  = '#8c564b'

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(20, 10), sharex=True,
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08}
    )

    # ── TOP panel ──
    for v in VARIANTS:
        ax1.plot(x, variant_vals[v], marker='o', ms=4, lw=1.6,
                 label=f'μ {v}', color=VARIANT_COLORS[v], alpha=0.85)

    ax1.plot(x, mu_vals, marker='D', ms=4, lw=2, ls='--',
             label='μ(all variants)', color=MU_COLOR, alpha=0.9, zorder=5)

    ax1.axhline(np.mean(mu_vals), color=MU_COLOR, lw=1, ls=':', alpha=0.5)
    ax1.set_ylabel('mIoU Score (μ per variant)', fontsize=11, fontweight='bold')
    ax1.set_title('Linguistic Sensitivity Analysis — Per-Class mIoU & LSS\n'
                  '(classes sorted by LSS descending)', fontsize=13, fontweight='bold')
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(fontsize=9, loc='upper right', ncol=3, framealpha=0.9)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.grid(True, axis='x', alpha=0.15, linestyle=':')

    for i in range(0, len(cls_names), 2):
        ax1.axvspan(i - 0.5, i + 0.5, color='grey', alpha=0.04)

    # ── BOTTOM panel ──
    ax2.bar(x, lss_vals, color=LSS_COLOR, alpha=0.75, edgecolor='white', linewidth=0.5, label='LSS(M, c)')
    ax2.axhline(LSS_M, color='red', lw=1.5, ls='--', label=f'LSS(M) = {LSS_M:.4f}', zorder=5)

    ax2.set_ylabel('LSS', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, max(lss_vals) * 1.18 if len(lss_vals) > 0 and max(lss_vals) > 0 else 0.5)
    ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

    for i in range(0, len(cls_names), 2):
        ax2.axvspan(i - 0.5, i + 0.5, color='grey', alpha=0.04)

    ax2.set_xticks(x)
    ax2.set_xticklabels(cls_names, rotation=55, ha='right', fontsize=8.5)
    ax2.set_xlabel('Class (sorted by LSS ↓)', fontsize=11, fontweight='bold')

    for i in range(min(5, len(cls_names))):
        ax2.text(i, lss_vals[i] + 0.005, f'{lss_vals[i]:.3f}',
                 ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=LSS_COLOR)

    fig.text(0.5, 0.995, f'Model-level LSS(M) = {LSS_M:.4f}  |  {len(cls_names)} classes evaluated',
             ha='center', va='top', fontsize=11, color='dimgray', style='italic')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_taxonomy_deltas(LSS_M, lss_per_class_results, save_path="taxonomy_deltas.png"):
    """
    Renders and saves a Delta Bar Chart showing performance differences relative to Original.
    """
    if not lss_per_class_results:
        print("No LSS class results to plot deltas.")
        return

    # Sort classes by LSS descending (matching the LSS plot)
    sorted_classes = sorted(
        lss_per_class_results.items(),
        key=lambda x: x[1]['LSS'] if x[1]['LSS'] is not None else 0.0,
        reverse=True
    )

    cls_names = [c[0] for c in sorted_classes]
    n_classes = len(cls_names)
    
    # Extract deltas: Variant - Original
    syn_deltas = []
    hyper_deltas = []
    hypo_deltas = []

    for name, stats in sorted_classes:
        mu_orig = stats['mu_per_variant'].get('Original', 0.0)
        
        syn_deltas.append(stats['mu_per_variant'].get('Synonyms', 0.0) - mu_orig)
        hyper_deltas.append(stats['mu_per_variant'].get('Hypernyms', 0.0) - mu_orig)
        hypo_deltas.append(stats['mu_per_variant'].get('Hyponyms', 0.0) - mu_orig)

    fig, ax = plt.subplots(figsize=(24, 10))
    
    x = np.arange(n_classes)
    width = 0.25  # width of each bar
    
    # Plot grouped bars
    rects1 = ax.bar(x - width, syn_deltas, width, label='Synonyms - Original', color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.7)
    rects2 = ax.bar(x, hyper_deltas, width, label='Hypernyms - Original', color='#2ca02c', alpha=0.85, edgecolor='black', linewidth=0.7)
    rects3 = ax.bar(x + width, hypo_deltas, width, label='Hyponyms - Original', color='#d62728', alpha=0.85, edgecolor='black', linewidth=0.7)
    
    # Add baseline line at y = 0
    ax.axhline(0, color='black', linewidth=1.2, linestyle='-')
    
    ax.set_ylabel('mIoU Difference (Variant - Original)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Class (sorted by LSS ↓)', fontsize=12, fontweight='bold')
    ax.set_title('Taxonomy mIoU Delta Analysis — Differences relative to Original\n'
                 'Negative values represent performance degradation', fontsize=14, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(cls_names, rotation=55, ha='right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.grid(True, axis='x', alpha=0.1, linestyle=':')
    
    # Alternating background bands for readability
    for i in range(0, n_classes, 2):
        ax.axvspan(i - 0.5, i + 0.5, color='grey', alpha=0.04)

    # Set y limits with some padding
    all_deltas = syn_deltas + hyper_deltas + hypo_deltas
    if all_deltas:
        min_delta = min(all_deltas)
        max_delta = max(all_deltas)
        ymin = min(-0.1, min_delta - 0.05)
        ymax = max(0.1, max_delta + 0.05)
        ax.set_ylim(ymin, ymax)

    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    
    fig.text(0.5, 0.99, f'Model-level LSS(M) = {LSS_M:.4f}  |  {n_classes} classes evaluated',
             ha='center', va='top', fontsize=11, color='dimgray', style='italic')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

