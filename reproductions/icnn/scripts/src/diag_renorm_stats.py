#!/usr/bin/env python
'''What exactly is the "renorm" of MindEye2's betas? Measure the per-voxel statistics of the file.

Inputs:  betas_all_subj0N_fp32_renorm.hdf5 (MindEye2 authors' NSD-core betas, ~1.9 GB per subject)
Outputs: statistics on screen, with a decision rule at the end; no GPU, seconds

Why: the preprocessing shipped with MindEye2 (dataset_creation.ipynb) is not the source of this file (it reads
beta2, drops voxels with ncsnr < .15 and writes float16), and nothing documents the step. The per-voxel std
within session 1 is 0.93, not 1.00: that rules out a per-session z-score and fits a global per-voxel z-score
over the 30000 trials, where the between-session differences of the means take the missing variance.

Usage:
    python src/diag_renorm_stats.py --subjects 1 2 5 7
'''
import argparse
import os

import h5py
import numpy as np


DEFAULT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '..', 'mindeye_II', 'src', 'mindeyev2_dataset'))

TRIALS_PER_SESSION = 750


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=DEFAULT_ROOT)
    p.add_argument('--subjects', type=int, nargs='+', default=[1])
    args = p.parse_args()

    for subj in args.subjects:
        path = os.path.join(args.root, 'betas_all_subj%02d_fp32_renorm.hdf5' % subj)
        with h5py.File(path, 'r') as f:
            betas = f['betas'][:]
        n_trial, n_vox = betas.shape
        n_sess = n_trial // TRIALS_PER_SESSION
        print('\n=== subj%02d: %s  %s' % (subj, betas.shape, betas.dtype))
        if n_sess * TRIALS_PER_SESSION != n_trial:
            print('  WARNING: %d trials is not a whole number of 750-trial sessions'
                  % n_trial)

        # ddof=0, the same convention as sklearn's StandardScaler; "exactly 1.000"
        # means "this axis is what was normalised".
        g_mean, g_std = betas.mean(0), betas.std(0)
        print('  global (all %d trials)  per-voxel mean %+.4f +/- %.4f | std %.4f +/- %.4f'
              % (n_trial, g_mean.mean(), g_mean.std(), g_std.mean(), g_std.std()))

        sess_mean = np.stack([betas[s * TRIALS_PER_SESSION:(s + 1) * TRIALS_PER_SESSION].mean(0)
                              for s in range(n_sess)])
        sess_std = np.stack([betas[s * TRIALS_PER_SESSION:(s + 1) * TRIALS_PER_SESSION].std(0)
                             for s in range(n_sess)])
        print('  per-session  std   : mean %.4f  min %.4f  max %.4f'
              % (sess_std.mean(), sess_std.mean(1).min(), sess_std.mean(1).max()))
        print('  per-session |mean| : mean %.4f  min %.4f  max %.4f'
              % (np.abs(sess_mean).mean(), np.abs(sess_mean).mean(1).min(),
                 np.abs(sess_mean).mean(1).max()))

        # If the file were z-scored globally, the variance the sessions do not share is
        # exactly what is missing from the within-session std.
        between = sess_mean.var(0).mean()
        print('  between-session variance of the per-voxel means: %.4f' % between)
        print('  sqrt(1 - between) = %.4f   <- predicted within-session std if the '
              'z-score is global' % np.sqrt(max(0.0, 1.0 - between)))

    print('\nHow to read it:')
    print('  per-session std ~ 1.000 exactly      -> z-score PER SESSION')
    print('  global std ~ 1.000, per-session < 1  -> z-score GLOBAL over all trials,')
    print('     and the per-session std should match sqrt(1 - between) above')
    print('  neither ~ 1.000                      -> something else again: report the')
    print('     numbers in the thesis rather than naming a procedure we cannot verify')


if __name__ == '__main__':
    main()
