"""Aggregate the per-image NSD-Imagery metrics into the rows of Kneeland's tables.

Inputs:  per_image_subj*_*_s*.tsv of evaluate_reconstruction_imagery.py (--metrics_dir); per_image_bridge_*.tsv with --bridge
Outputs: summary_per_sample, summary_per_subject, summary_per_subject_samplestd and summary_overall (.tsv)

Port of the Brain-Diffuser aggregation. The order is part of the protocol: per stimulus -> per stimulus type
(simple/complex) -> mean over the 10 samples -> mean over the 4 subjects. The samples are averaged over metric
values, never over images (a two-way score is meaningless on an average of images). Here the 10 repetitions are the
samples `_000.._009` of one run, not 10 seeds. A (subject, modality) cell without all 10 samples is reported loudly:
where chance is the expected result, a half-empty cell looks exactly like a correct one.

Usage (from codes/utils/):
    python aggregate_imagery_metrics.py [--bridge]      # --bridge: the shared1000 check
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np

METRICS = ['PixCorr', 'SSIM', 'AlexNet2', 'AlexNet5', 'Inception', 'CLIP', 'EffNet', 'SwAV']
# Direction of each metric, for the printed summary only.
HIGHER_IS_BETTER = {m: m not in ('EffNet', 'SwAV') for m in METRICS}
EXPECTED_SAMPLES = 10


def read_tsv(path):
    with open(path) as f:
        header = f.readline().rstrip('\n').split('\t')
        rows = []
        for line in f:
            if not line.strip():
                continue
            rows.append(dict(zip(header, line.rstrip('\n').split('\t'))))
    return rows


def load_all(metrics_dir, pattern):
    """cells[(sub, modality, sample, stim_type)] = {metric: [values over stimuli]}"""
    paths = sorted(glob.glob(os.path.join(metrics_dir, pattern)))
    if not paths:
        raise SystemExit('no {} found under {}'.format(pattern, metrics_dir))
    cells = defaultdict(lambda: defaultdict(list))
    for path in paths:
        for r in read_tsv(path):
            key_base = (r['sub'], r['modality'], int(r['sample']))
            # 'shared1000' (bridge mode) has no simple/complex split: it is one group.
            stim_types = (('all',) if r['stim_type'] in ('all', 'shared1000')
                          else ('all', r['stim_type']))
            for stim_type in stim_types:
                for m in METRICS:
                    cells[key_base + (stim_type,)][m].append(float(r[m]))
    print('read {} files from {}'.format(len(paths), metrics_dir))
    return cells


def mean_cell(cell):
    return {m: float(np.mean(cell[m])) for m in METRICS}


def write_tsv(path, key_names, rows):
    with open(path, 'w') as f:
        f.write('\t'.join(list(key_names) + METRICS) + '\n')
        for key, vals in rows:
            f.write('\t'.join([str(k) for k in key]
                              + ['{:.4f}'.format(vals[m]) for m in METRICS]) + '\n')
    print('wrote {}'.format(path))


def markdown_table(title, key_names, rows):
    out = ['', '#### {}'.format(title), '',
           '| ' + ' | '.join(list(key_names) + [
               m + ('↑' if HIGHER_IS_BETTER[m] else '↓') for m in METRICS]) + ' |',
           '| ' + ' | '.join(['---'] * (len(key_names) + len(METRICS))) + ' |']
    for key, vals in rows:
        out.append('| ' + ' | '.join([str(k) for k in key]
                                     + ['{:.4f}'.format(vals[m]) for m in METRICS]) + ' |')
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metrics_dir', type=str, default='../../results/imagery_metrics')
    parser.add_argument('--subjects', nargs='+', type=str,
                        default=['subj01', 'subj02', 'subj05', 'subj07'])
    parser.add_argument('--bridge', action='store_true',
                        help='aggregate the shared1000 pilot instead')
    parser.add_argument('--expected_samples', type=int, default=EXPECTED_SAMPLES)
    opt = parser.parse_args()

    if not os.path.isfile('make_subjmri.py'):
        raise SystemExit('run this from codes/utils/')

    pattern = 'per_image_bridge_*.tsv' if opt.bridge else 'per_image_subj*_*_s*.tsv'
    prefix = 'bridge_' if opt.bridge else ''
    cells = load_all(opt.metrics_dir, pattern)

    # 1) per (subject, modality, sample, stim_type)
    per_sample = sorted((k, mean_cell(v)) for k, v in cells.items())
    write_tsv(os.path.join(opt.metrics_dir, prefix + 'summary_per_sample.tsv'),
              ('sub', 'modality', 'sample', 'stim_type'), per_sample)

    # 2) mean and std over the samples, per (subject, modality, stim_type)
    by_subject = defaultdict(dict)
    for (sub, modality, sample, stim_type), vals in per_sample:
        by_subject[(sub, modality, stim_type)][sample] = vals

    incomplete = []
    per_subject, per_subject_std = [], []
    for key in sorted(by_subject):
        samples = by_subject[key]
        if len(samples) != opt.expected_samples:
            incomplete.append((key, sorted(samples)))
        stacked = {m: np.array([samples[s][m] for s in sorted(samples)]) for m in METRICS}
        per_subject.append((key, {m: float(stacked[m].mean()) for m in METRICS}))
        per_subject_std.append((key, {m: float(stacked[m].std(ddof=0)) for m in METRICS}))
    write_tsv(os.path.join(opt.metrics_dir, prefix + 'summary_per_subject.tsv'),
              ('sub', 'modality', 'stim_type'), per_subject)
    write_tsv(os.path.join(opt.metrics_dir, prefix + 'summary_per_subject_samplestd.tsv'),
              ('sub', 'modality', 'stim_type'), per_subject_std)

    # 3) mean over the subjects, per (modality, stim_type), tables 1, 3 and 4
    by_overall = defaultdict(dict)
    for (sub, modality, stim_type), vals in per_subject:
        by_overall[(modality, stim_type)][sub] = vals
    overall = []
    for key in sorted(by_overall):
        subs = by_overall[key]
        missing = [s for s in opt.subjects if s not in subs]
        if missing:
            print('WARNING: {} {} is missing subjects {}'.format(key[0], key[1], missing),
                  file=sys.stderr)
        overall.append((key, {m: float(np.mean([subs[s][m] for s in sorted(subs)]))
                              for m in METRICS}))
    write_tsv(os.path.join(opt.metrics_dir, prefix + 'summary_overall.tsv'),
              ('modality', 'stim_type'), overall)

    for key, samples in incomplete:
        print('WARNING: cell {} has {} samples ({}), expected {}: a run failed silently'
              .format(key, len(samples), samples, opt.expected_samples), file=sys.stderr)

    print(markdown_table('Overall (mean over samples, then subjects)',
                         ('modality', 'stim_type'), overall))
    print(markdown_table('Per subject (mean over the samples)',
                         ('sub', 'modality', 'stim_type'),
                         [r for r in per_subject if r[0][2] == 'all']))
    if opt.bridge:
        print('\nCheck against the published shared1000 row: PixCorr ~0.246, '
              'SSIM ~0.410, EffNet ~0.811, SwAV ~0.504. The two-way columns are NOT '
              'comparable there (different distractor pool).')
    else:
        print('\nExpected result is CHANCE: AlexNet5 '
              'above 60% or PixCorr above 0.05 means the pipeline is wrong, not good.')


if __name__ == '__main__':
    main()
