#!/usr/bin/env python3
"""Figures 4.3 and B.1: margin of each reproduced model over its measured chance.

Inputs:  baselines/{model}.tsv (scores) and baselines/{model}_chance.tsv (--roll_gt control), per image
Outputs: fig4_3_margins.png (not in the text) and fig4_4_imagery_gap.png (Figure B.1 of Appendix B)

Margin = mean score minus mean chance on AlexNet(2), AlexNet(5), InceptionV3 and CLIP, in percentage
points. It also recomputes Table 7.2 and stops if a cell differs by more than 0.1 pp (the table was
built from margins rounded before the subtraction). The off-by-one of Section 4.1 cancels in the margin.

Usage:
    python figures/make_margin_figures.py
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TSV_DIR = os.path.join(ROOT, 'my_project', 'data', 'results', 'baselines')
OUT_DIR = os.path.join(ROOT, 'thesis', 'figures')

# Chapter 4 order (Sections 4.3.1-4.3.5). Colour follows the model, never its rank.
MODELS = [('mindeye_I', 'MindEye1'), ('mindeye_II', 'MindEye2'),
          ('brain_diffuser', 'Brain-Diffuser'), ('takagi', 'Takagi et al.'), ('icnn', 'iCNN')]
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # reference palette, slots 1-5
MODE_COLORS = {'vision': '#2a78d6', 'imagery': '#eb6834'}
METRICS = [('AlexNet(2)', 'Alex(2)'), ('AlexNet(5)', 'Alex(5)'),
           ('InceptionV3', 'Incep'), ('CLIP', 'CLIP')]
MODES = ['vision', 'imagery']

# Table 7.2 of the thesis, rounded to 0.1 pp: the check that the figures show the numbers of the table.
TABLE_7_2 = {
    'mindeye_I':      ([38.1, 37.2, 20.2, 18.2], [13.2, 20.4, 9.0, 8.5]),
    'brain_diffuser': ([40.7, 45.1, 15.5, 20.2], [19.0, 28.1, 11.1, 14.0]),
    'icnn':           ([26.4, 29.8, 18.6, 2.0], [4.1, 5.3, -0.2, 0.2]),
    'mindeye_II':     ([31.3, 39.1, 21.9, 22.8], [4.0, 7.9, 4.4, 6.1]),
    'takagi':         ([15.6, 15.0, 15.9, 13.9], [6.0, 5.3, 0.4, 3.4]),
}

INK, INK_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'


def mean_scores(path, mode):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f, delimiter='\t') if r['mode'] == mode]
    assert len(rows) == 480, (path, mode, len(rows))   # 4 subjects x 12 stimuli x 10 samples
    return np.array([np.mean([float(r[m]) for r in rows]) for m, _ in METRICS])


def margins():
    out = {}
    for key, _ in MODELS:
        out[key] = {}
        for i, mode in enumerate(MODES):
            s = mean_scores(os.path.join(TSV_DIR, f'{key}.tsv'), mode)
            c = mean_scores(os.path.join(TSV_DIR, f'{key}_chance.tsv'), mode)
            m = 100 * (s - c)
            assert np.all(np.abs(m - np.array(TABLE_7_2[key][i])) <= 0.1 + 1e-9), (key, mode, m)
            out[key][mode] = m
    return out


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK, length=3)
    ax.set_facecolor(SURFACE)


def fig_margins(mg):
    """Grouped bars, one panel per task; the measured chance is the zero line."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    n = len(MODELS)
    width = 0.8 / n
    x = np.arange(len(METRICS))
    for ax, mode in zip(axes, MODES):
        style(ax)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for j, ((key, name), col) in enumerate(zip(MODELS, COLORS)):
            xs = x - 0.4 + width * (j + 0.5)
            ax.bar(xs, mg[key][mode], width=width - 0.02, color=col, label=name,
                   edgecolor=SURFACE, linewidth=1)
        ax.axhline(0, color=INK, lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in METRICS])
        ax.set_title('Vision trials' if mode == 'vision' else 'Imagery trials', color=INK, fontsize=11)
    axes[0].set_ylabel('Margin over measured chance (pp)', color=INK)
    axes[1].legend(frameon=False, fontsize=9, loc='upper right')
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_3_margins.png')
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_gap(mg):
    """Per model and metric, vision margin -> imagery margin (dumbbell), as small multiples."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11, 3.2), sharey=True, sharex=True)
    y = np.arange(len(MODELS))[::-1]
    for ax, (k, (_, lab)) in zip(axes, enumerate(METRICS)):
        style(ax)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.axvline(0, color=INK, lw=1.2)
        for yi, (key, _) in zip(y, MODELS):
            v, im = mg[key]['vision'][k], mg[key]['imagery'][k]
            ax.plot([im, v], [yi, yi], color=INK_2, lw=2, solid_capstyle='round', zorder=1)
            for mode, val in (('vision', v), ('imagery', im)):
                ax.scatter(val, yi, s=55, color=MODE_COLORS[mode], edgecolor=SURFACE,
                           linewidth=1.5, zorder=2, label=mode.capitalize() if yi == y[0] else None)
        ax.set_title(lab, color=INK, fontsize=11)
        ax.set_xlabel('Margin over chance (pp)', color=INK, fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([name for _, name in MODELS])
    axes[-1].legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_4_imagery_gap.png')
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == '__main__':
    mg = margins()
    print('margins match Table 7.2 to 0.1 pp')
    print(fig_margins(mg))
    print(fig_gap(mg))
