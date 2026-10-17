#!/usr/bin/env python3
"""Figure 4.1: shared1000 reconstructions of subject 1 by the five baselines and TRISECT.

Inputs:  reconstructions already on disk, listed in the path constants below (nothing is generated)
Outputs: fig4_2a_shared1000.png and fig4_2b_shared1000.png (8 stimuli each): panels (a) and (b)

The 16 stimuli are those that all six sources hold for subject 1. `COMMON` gives each one as (index in
ascending NSD-ID order, MindEye1 index, Takagi / Brain-Diffuser index): the sources order the images
differently (MindEye1 has 982, not 1,000). Before drawing, the script checks that the exported ground
truths agree (r >= 0.99) and that the iCNN NSD IDs equal those of the TRISECT block manifest.

Usage:
    python figures/make_shared1000_figure.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, 'figure')
OUT_DIR = os.path.join(ROOT, 'thesis', 'figures')
ICNN_DIR = os.path.join(FIG, 'icnn_imagery', 'bridge', 'recon_bridge100')
TRISECT_DIR = os.path.join(ROOT, 'my_project', 'data', 'nsdcore_recons_b3',
                           'S1K_T3_p3_subj01_s0.75_m0.4')
MANIFEST = os.path.join(ROOT, 'my_project', 'data', 'shared1000_blockset', 'manifest.json')

# (ascending NSD-ID index, MindEye1 index, Takagi / Brain-Diffuser index).
COMMON = [(5, 5, 74), (13, 13, 38), (35, 35, 28), (39, 39, 5), (44, 44, 18), (52, 51, 15),
          (54, 53, 69), (59, 57, 6), (65, 63, 21), (72, 69, 91), (73, 70, 73), (79, 76, 10),
          (80, 77, 95), (81, 78, 78), (85, 82, 53), (86, 83, 80)]

ROWS = ['Ground truth', 'MindEye1', 'MindEye2', 'Brain-Diffuser', 'Takagi et al.', 'iCNN',
        'TRISECT']
INK = '#0b0b0b'


def icnn_ids():
    names = sorted(os.listdir(os.path.join(ICNN_DIR, 'gt')))
    assert len(names) == 100, len(names)
    return [int(n.split('-')[1].split('.')[0]) for n in names]


def mindeye1_pair(m):
    """Split a MindEye1 pair (GT | reconstruction, 2 px borders) into its two 512 px halves."""
    im = np.asarray(Image.open(os.path.join(FIG, 'mindeye_I', 'subj01', 'pairs', f'{m:04d}.png'))
                    .convert('RGB'))
    assert im.shape == (516, 1030, 3), im.shape
    return im[2:514, 2:514], im[2:514, 516:1028]


def trisect_path(i, blocks):
    block, pos = divmod(i, 12)
    half, k = ('imagerysimple', pos) if pos < 6 else ('imagerycomplex', pos - 6)
    nsd_id = blocks[block]['img73k_ids'][pos]
    assert nsd_id == blocks[block][half + '_ids'][k]
    return os.path.join(TRISECT_DIR, f'subj{block:02d}', half, 'sample000', f'{k:05d}.png'), nsd_id


def load(path):
    return np.asarray(Image.open(path).convert('RGB'))


def grey64(a):
    return np.asarray(Image.fromarray(a).convert('L').resize((64, 64), Image.BILINEAR),
                      dtype=np.float64).ravel()


def collect():
    ids = icnn_ids()
    with open(MANIFEST) as f:
        blocks = json.load(f)['blocks']
    columns = []
    for i, m, t in COMMON:
        gt_me1, rec_me1 = mindeye1_pair(m)
        gt = load(os.path.join(FIG, 'mindeye_II', 'gt', f'{i:04d}.png'))
        gts = {
            'MindEye1': gt_me1,
            'iCNN': load(os.path.join(ICNN_DIR, 'gt', f'shared1000-{ids[i]:05d}.png')),
            'Takagi': load(os.path.join(FIG, 'takagi', 'image-text', f'{t:05d}_org.png')),
            'Brain-Diffuser': load(os.path.join(FIG, 'brain_diffuser', 'ground_truth', f'{t}.png')),
        }
        for name, other in gts.items():
            r = np.corrcoef(grey64(gt), grey64(other))[0, 1]
            assert r >= 0.99, (i, t, name, r)
        tpath, nsd_id = trisect_path(i, blocks)
        assert nsd_id == ids[i], (i, nsd_id, ids[i])
        columns.append([
            gt,
            rec_me1,
            load(os.path.join(FIG, 'mindeye_II', 'finetuned_subj01_40sess_4096hid', 'enhanced',
                              f'{i:04d}.png')),
            load(os.path.join(FIG, 'brain_diffuser', 'vd', 'subj01', f'{t}.png')),
            load(os.path.join(FIG, 'takagi', 'image-text', f'{t:05d}_000.png')),
            load(os.path.join(ICNN_DIR, 'normfeat', 'subj01', 'nsdgeneral',
                              f'recon_image_normalized-shared1000-{ids[i]:05d}.tiff')),
            load(tpath),
        ])
        print(f'index {i:3d}  MindEye1 {m:3d}  Takagi/BD {t:2d}  NSD ID {ids[i]:5d}  aligned')
    return columns


def draw(columns, path):
    n_rows, n_cols = len(ROWS), len(columns)
    cell = 0.78
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.25 + n_cols * cell, n_rows * cell),
                             gridspec_kw=dict(wspace=0.04, hspace=0.04))
    fig.subplots_adjust(left=1.25 / (1.25 + n_cols * cell), right=0.995, top=0.995, bottom=0.005)
    for c, col in enumerate(columns):
        for r, img in enumerate(col):
            ax = axes[r, c]
            ax.imshow(Image.fromarray(img).resize((256, 256), Image.LANCZOS))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.set_ylabel(ROWS[r], rotation=0, ha='right', va='center', fontsize=8.5,
                              color=INK, labelpad=6)
    fig.savefig(path, dpi=250, facecolor='white')
    plt.close(fig)
    print('wrote', os.path.relpath(path, ROOT))


def main():
    columns = collect()
    draw(columns[:8], os.path.join(OUT_DIR, 'fig4_2a_shared1000.png'))
    draw(columns[8:], os.path.join(OUT_DIR, 'fig4_2b_shared1000.png'))


if __name__ == '__main__':
    main()
