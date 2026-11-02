"""Paper plotting style; all outputs are PNGs."""
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'dp-watermarks_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.pools import MODELS, LABELS

plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                     'font.size': 16, 'axes.labelsize': 20, 'xtick.labelsize': 16,
                     'ytick.labelsize': 16, 'legend.fontsize': 13})


def style_figure(fig):
    """Apply the same publication typography and remove all figure/panel titles."""
    if fig._suptitle is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    for ax in fig.axes:
        for loc in ['left', 'center', 'right']:
            ax.set_title('', loc=loc)
        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)
        ax.tick_params(axis='both', labelsize=16)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(13)



# Rendering only: rows contain summaries of already-computed estimates.
# Pool construction and randomized sampling are in experiments/; core.estimation.estimate
# computes the watermark shares before these curves are drawn.
def curves(ax, rows, metric='mean', bands=True):
    for i, (model, label, marker) in enumerate(zip(MODELS, LABELS, ['o', 's', '^', 'D'])):
        group = sorted((r for r in rows if r['model'] == model), key=lambda r: r['x'])
        if not group:
            continue
        x = [r['x'] for r in group]
        if bands:
            ax.fill_between(x, [r['mape_p05' if metric == 'mape' else 'p05'] for r in group],
                            [r['mape_p95' if metric == 'mape' else 'p95'] for r in group],
                            color=f'C{i}', alpha=.12, linewidth=0)
        ax.plot(x, [r[metric] for r in group], color=f'C{i}', marker=marker,
                linewidth=2, label=label)


def lambda_axes(ax, aspect=1):
    ax.plot([0, 1], [0, 1], '--', color='0.3', linewidth=1.3,
            label=r'$\hat\lambda=\lambda$')
    ax.set(xlim=(.05, .95), ylim=(0, 1))
    ax.set_xlabel(r'True $\lambda$', fontsize=20)
    ax.set_xticks(np.arange(1, 10)/10)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_box_aspect(aspect)


def finish(fig, axes, path, spacing=.8):
    for ax in axes:
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(alpha=.2)
    style_figure(fig)
    fig.tight_layout(w_pad=spacing)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main_figure(outfox, coding, path):
    if coding is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        axes, datasets = [ax], [outfox]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        datasets = [outfox, coding]
    for ax, rows in zip(axes, datasets):
        curves(ax, rows)
        # Preserve the existing two-panel figure's .75 aspect for reproduction.
        lambda_axes(ax, aspect=1 if coding is None else .75)
    axes[0].legend(loc='upper left', frameon=False, fontsize=13)
    axes[0].set_ylabel(r'Estimated $\hat\lambda$', fontsize=20)
    if coding is not None:
        for ax, title in zip(axes, ['OUTFOX', 'LiveCodeBench']):
            ax.set_title(title, fontsize=18)
        fig.suptitle(r'$n=100,\ \epsilon=2,\ M=20,\ \pi_0=0.5$', fontsize=16)
    finish(fig, axes, path, spacing=2 if coding is not None else 1.08)


def ablation(name, rows, path):
    if name in ('epsilon', 'n'):
        raise ValueError('Use the combined overlay figures for epsilon and n sweeps.')
    name = 'temperature' if name.startswith('temperature') else name
    if name == 'temperature':
        rows = [r for r in rows if .2 <= r['x'] <= 1.]
    parallel = name == 'parallel'
    panels = [1, 2, 3] if parallel else [.1, .5, .9]
    width = 13 if name in ['pi0', 'rejection_m', 'parallel_m'] else 4.33*3
    fig, axes = plt.subplots(1, 3, figsize=(width, 4.4), sharey=True)
    labels = dict(pi0=r'$\pi_0$',
                  temperature='Temperature', rejection_m=r'$M$', parallel_m=r'$M$')
    # The rejection plot draws all panels before fixing the shared y limits.
    if name in ['rejection_m', 'parallel_m']:
        for ax, panel in zip(axes, panels):
            curves(ax, [r for r in rows if r['panel'] == panel], metric='mape', bands=False)
    for ax, panel in zip(axes, panels):
        group = [r for r in rows if r['panel'] == panel]
        if name not in ['rejection_m', 'parallel_m']:
            curves(ax, group, metric='mean' if parallel else 'mape', bands=parallel)
        symbol = r'\epsilon' if parallel else r'\lambda'
        ax.set_title(rf'${symbol}={panel:g}$', fontsize=18)
        if parallel:
            lambda_axes(ax)
            ax.set_yticks([i/10 for i in range(11)])
        else:
            ax.set_xlabel(labels[name], fontsize=20)
            ax.set_ylim(bottom=0)
            ticks = sorted({r['x'] for r in rows})
            if name in ['rejection_m', 'parallel_m']:
                ax.set_xscale('log', base=2)
            ax.set_xticks(ticks, [f'{x:g}' for x in ticks], rotation=0)
        ax.set_box_aspect(1)
    if name == 'temperature':
        axes[0].set_ylim(0, 1.05*max(r['mape'] for r in rows))
    axes[0].set_ylabel(r'Estimated $\hat\lambda$' if parallel else 'MAPE (%)', fontsize=20)
    axes[0].legend(frameon=False, fontsize=11)
    finish(fig, axes, path)


def c4_main(rows, path):
    """Three epsilon panels, preserving the supplied main figure's layout."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, eps in zip(axes, [1, 2, 3]):
        curves(ax, [r for r in rows if r['panel'] == eps])
        lambda_axes(ax)
    axes[0].set_ylabel(r'Estimated $\hat\lambda$', fontsize=20)
    axes[0].legend(loc='upper left', frameon=False, fontsize=12)
    finish(fig, axes, path)
