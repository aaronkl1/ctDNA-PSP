import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import cohen_kappa_score

TRAJ_ORDER     = ['Responder', 'Pseudoprogressor', 'Progressor']
ANALYT_ORDER_4 = ['Responder', 'Immune_Pseudoprogressor',
                  'Delayed_Pseudoprogressor', 'Progressor']
ANALYT_ORDER_3 = ['Responder', 'Pseudoprogressor', 'Progressor']

TRAJ_LABELS_SHORT = {
    'Responder': 'Responder', 'Pseudoprogressor': 'PSP', 'Progressor': 'Progressor',
}
ANALYT_LABELS_SHORT_4 = {
    'Responder': 'Responder', 'Immune_Pseudoprogressor': 'Immune PSP',
    'Delayed_Pseudoprogressor': 'Delayed PSP', 'Progressor': 'Progressor',
}
ANALYT_LABELS_SHORT_3 = {
    'Responder': 'Responder', 'Pseudoprogressor': 'PSP', 'Progressor': 'Progressor',
}


def _plot_cm(ax, ct, title, row_label, col_label, cmap='Blues'):
    data   = ct.values.astype(float)
    rtots  = data.sum(axis=1, keepdims=True)
    normed = np.divide(data, rtots, out=np.zeros_like(data), where=rtots != 0)
    ax.imshow(normed, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(ct.columns)))
    ax.set_yticks(range(len(ct.index)))
    ax.set_xticklabels(ct.columns, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(ct.index, fontsize=10)
    for i in range(len(ct.index)):
        for j in range(len(ct.columns)):
            count = int(data[i, j])
            pct   = 100.0 * normed[i, j]
            color = 'white' if normed[i, j] > 0.55 else 'black'
            ax.text(j, i, f'{count}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=9, color=color, fontweight='bold')
    ax.set_xlabel(col_label, fontsize=12, labelpad=10)
    ax.set_ylabel(row_label, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)


def _plot_y_inf_heatmap(df, traj_order, analyt_order,
                        traj_labels, analyt_labels, out_dir):
    import matplotlib.colors as mcolors

    records = []
    for tc in traj_order:
        for ac in analyt_order:
            sub = df[(df['Category'] == tc) &
                     (df['Analytical_Category'] == ac)]['y0_over_y_inf'].dropna()
            records.append({'traj': tc, 'analyt': ac, 'n': len(sub),
                            'mean': sub.mean() if len(sub) else np.nan,
                            'std':  sub.std()  if len(sub) else np.nan,
                            'median': sub.median() if len(sub) else np.nan})
    stat_df  = pd.DataFrame(records)
    mean_mat = stat_df.pivot(index='traj', columns='analyt', values='mean')
    std_mat  = stat_df.pivot(index='traj', columns='analyt', values='std')
    n_mat    = stat_df.pivot(index='traj', columns='analyt', values='n')
    mean_mat = mean_mat.reindex(index=traj_order, columns=analyt_order)
    std_mat  = std_mat.reindex(index=traj_order,  columns=analyt_order)
    n_mat    = n_mat.reindex(index=traj_order,    columns=analyt_order)

    row_labels = [traj_labels[r]   for r in traj_order]
    col_labels = [analyt_labels[c] for c in analyt_order]

    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))

    ax   = axes[0]
    data = mean_mat.values.astype(float)
    vmax = np.nanpercentile(data, 95)
    im   = ax.imshow(data, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)
    for i, tr in enumerate(traj_order):
        for j, ac in enumerate(analyt_order):
            n   = int(n_mat.loc[tr, ac]) if not np.isnan(n_mat.loc[tr, ac]) else 0
            mn  = mean_mat.loc[tr, ac]; sd = std_mat.loc[tr, ac]
            txt = '—' if n == 0 else f'n={n:,}\n{mn:.2f} ± {sd:.2f}'
            cell_val = data[i, j] if not np.isnan(data[i, j]) else 0
            color = 'white' if cell_val > vmax * 0.6 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean $y_0 / y_\\infty$', fontsize=10)
    ax.set_xlabel('Analytical Category', fontsize=11)
    ax.set_ylabel('Trajectory-Based Category', fontsize=11)
    ax.set_title('Mean $y_0 / y_\\infty$ per confusion-matrix cell', fontsize=11)

    ax2 = axes[1]
    palette = {'Responder': '#2ecc71', 'Immune_Pseudoprogressor': '#3498db',
               'Delayed_Pseudoprogressor': '#9b59b6', 'Progressor': '#e74c3c'}

    positions, violin_data, tick_labels = [], [], []
    pos = 0; group_centres = {}
    for ti, tc in enumerate(traj_order):
        group_pos = []
        for ac in analyt_order:
            sub = df[(df['Category'] == tc) &
                     (df['Analytical_Category'] == ac)]['y0_over_y_inf'].dropna()
            if len(sub) < 2:
                pos += 1; continue
            violin_data.append(sub.values); positions.append(pos)
            group_pos.append(pos); tick_labels.append(analyt_labels[ac]); pos += 1
        if group_pos:
            group_centres[tc] = np.mean(group_pos)
        pos += 0.8

    if violin_data:
        vp = ax2.violinplot(violin_data, positions=positions,
                            showmedians=True, showextrema=False, widths=0.7)
        flat_cols = []
        for tc in traj_order:
            for ac in analyt_order:
                sub = df[(df['Category'] == tc) &
                         (df['Analytical_Category'] == ac)]['y0_over_y_inf'].dropna()
                if len(sub) >= 2:
                    flat_cols.append(palette.get(ac, 'steelblue'))
        for body, col in zip(vp['bodies'], flat_cols):
            body.set_facecolor(col); body.set_alpha(0.7)
        vp['cmedians'].set_color('black'); vp['cmedians'].set_linewidth(1.5)

    pos2 = 0
    for ti, tc in enumerate(traj_order):
        n_ac = sum(1 for ac in analyt_order
                   if len(df[(df['Category']==tc) &
                              (df['Analytical_Category']==ac)]['y0_over_y_inf'].dropna()) >= 2)
        if ti > 0:
            ax2.axvline(pos2 - 0.9, color='gray', lw=0.8, ls='--', alpha=0.6)
        pos2 += n_ac + 0.8
    for tc, cx in group_centres.items():
        ax2.text(cx, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 5,
                 traj_labels[tc], ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(positions)
    ax2.set_xticklabels(tick_labels, rotation=35, ha='right', fontsize=8)
    ax2.axhline(1.0, color='black', lw=1.0, ls=':', alpha=0.7, label='$y_0 = y_\\infty$')
    ax2.set_ylabel('$y_0 / y_\\infty$', fontsize=11)
    ax2.set_xlabel('Analytical Category (grouped by trajectory label)', fontsize=10)
    ax2.set_title('Distribution of $y_0 / y_\\infty$ per category cell', fontsize=11)
    ax2.grid(True, axis='y', lw=0.4, alpha=0.5)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=palette.get(ac, 'steelblue'), alpha=0.7,
                     label=analyt_labels[ac])
               for ac in analyt_order if ac in palette]
    ax2.legend(handles=handles, fontsize=8, loc='upper right')

    fig.suptitle('$y_0 / y_\\infty$ relative to confusion matrix categories',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'category_comparison_y_inf_heatmap.png'),
                dpi=200, bbox_inches='tight')
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir    = os.path.join(script_dir, 'output_v3')
    input_file = os.path.join(out_dir, 'processed_features.csv')

    df = pd.read_csv(input_file)

    for col in ('Category', 'Analytical_Category'):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Run metrics-processing.py first.")

    total      = len(df)
    n_unknown  = (df['Analytical_Category'] == 'Unknown').sum()
    n_excluded = (df['Analytical_Category'] == 'Excluded').sum()
    print(f"Total: {total}  |  Unknown: {n_unknown}  |  Excluded: {n_excluded}")

    drop_mask = df['Analytical_Category'].isin(('Unknown', 'Excluded'))
    df_known  = df[~drop_mask].copy()
    print(f"Retained: {len(df_known)}\n")

    merge_map = {
        'Responder':               'Responder',
        'Immune_Pseudoprogressor': 'Pseudoprogressor',
        'Delayed_Pseudoprogressor':'Pseudoprogressor',
        'Progressor':              'Progressor',
    }
    df_known['Analytical_3'] = df_known['Analytical_Category'].map(merge_map)
    df_known = df_known.dropna(subset=['Analytical_3'])

    overall_agreement = (df_known['Category'] == df_known['Analytical_3']).mean()
    kappa = cohen_kappa_score(df_known['Category'], df_known['Analytical_3'])
    print(f"Overall agreement: {100*overall_agreement:.1f}%  |  Cohen's κ: {kappa:.3f}")

    for cat in TRAJ_ORDER:
        mask = df_known['Category'] == cat
        if mask.sum() == 0: continue
        pct = 100 * (df_known.loc[mask, 'Analytical_3'] == cat).mean()
        print(f"  {cat:20s}: {pct:.1f}%  (n={mask.sum()})")

    present_traj = [c for c in TRAJ_ORDER if c in df_known['Category'].values]
    present_a4   = [c for c in ANALYT_ORDER_4 if c in df_known['Analytical_Category'].values]
    present_a3   = [c for c in ANALYT_ORDER_3 if c in df_known['Analytical_3'].values]

    ct4 = pd.crosstab(df_known['Category'], df_known['Analytical_Category'])
    ct4 = ct4.reindex(index=present_traj, columns=present_a4, fill_value=0)
    ct4.index   = [TRAJ_LABELS_SHORT[r] for r in ct4.index]
    ct4.columns = [ANALYT_LABELS_SHORT_4[c] for c in ct4.columns]

    ct3 = pd.crosstab(df_known['Category'], df_known['Analytical_3'])
    ct3 = ct3.reindex(index=present_traj, columns=present_a3, fill_value=0)
    ct3.index   = [TRAJ_LABELS_SHORT[r] for r in ct3.index]
    ct3.columns = [ANALYT_LABELS_SHORT_3[c] for c in ct3.columns]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    _plot_cm(axes[0], ct4, '4 Analytical Classes vs Trajectory-Based',
             'Trajectory-Based Category', 'Analytical Category', cmap='Blues')
    _plot_cm(axes[1], ct3, 'Merged Analytical (3 Classes) vs Trajectory-Based',
             'Trajectory-Based Category', 'Analytical Category (merged PSP)', cmap='Blues')
    fig.suptitle(
        f'n={len(df_known):,}  (excluded {n_unknown:,} Unknown)  |  '
        f'Agreement: {100*overall_agreement:.1f}%  |  κ: {kappa:.3f}',
        fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'category_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()

    for ct, suffix, col_label in [
        (ct4, '4class', 'Analytical Category'),
        (ct3, '3class', 'Analytical Category (merged PSP)'),
    ]:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        _plot_cm(ax2, ct,
                 ('4 Analytical Classes' if '4' in suffix else 'Merged Analytical (3 Classes)')
                 + ' vs Trajectory-Based',
                 'Trajectory-Based Category', col_label, cmap='Blues')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'category_comparison_{suffix}.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

    if 'y0_over_y_inf' in df_known.columns:
        _plot_y_inf_heatmap(df_known, present_traj, present_a4,
                            TRAJ_LABELS_SHORT, ANALYT_LABELS_SHORT_4, out_dir)


if __name__ == '__main__':
    main()
