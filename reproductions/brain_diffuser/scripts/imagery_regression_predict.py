"""Apply the three NSD-Core ridge decoders (VDVAE latents, CLIP text, CLIP vision) to the NSD-Imagery betas.

Inputs:  data/regression_weights/subjNN/ (one ~11 GB .pkl at a time), the prepared imagery betas
Outputs: data/predicted_features/subjNN/imagery/nsd_{vdvae,cliptext,clipvision}_pred_<cond>.npy

Why: (a) the input z-score with the training mean/std is NOT applied: the imagery betas already went through
/300 and the per-run z-score, which is the space the ridges were fitted on; a second normalization would
rescale every voxel with no error. (b) the predictions ARE re-standardized onto the training targets' moments:
alpha in [50k, 100k] shrinks them, and the CLIP conditionings would fall outside what Versatile Diffusion
expects. `--restandardize_pool` sets the pool of the prediction mean/std. CPU only; do not run subjects in parallel.

Usage (from the brain_diffuser/ root, OPENBLAS_CORETYPE=SkylakeX exported):
    python scripts/imagery_regression_predict.py --sub 1 --restandardize_pool modality12 \
        --conditions visionsimple visioncomplex imagerysimple imagerycomplex imageryconcepts
"""

import argparse
import os
import pickle
import sys

import numpy as np

SUBJECTS = (1, 2, 5, 7)
NUM_VOXELS = {1: 15724, 2: 14278, 5: 13039, 7: 12682}

ALL_CONDITIONS = ['visionsimple', 'visioncomplex',
                  'imagerysimple', 'imagerycomplex', 'imageryconcepts']

# The 12-stimulus pool of one modality (the default, `modality12`): it is also the
# distractor pool of the two-way metrics; vision and imagery stay symmetric.
POOL_OF = {
    'visionsimple': 'vision12', 'visioncomplex': 'vision12',
    'imagerysimple': 'imagery12', 'imagerycomplex': 'imagery12',
    'imageryconcepts': 'imageryconcepts',
    'visionconcepts': 'visionconcepts',
}

# name -> (weights .pkl, stats key prefix in train_target_stats.npz, output file template,
#          A-side prediction file used by --restandardize_pool fromA)
BRANCHES = {
    'vdvae': ('vdvae_regression_weights.pkl', 'vdvae',
              'nsd_vdvae_pred_{cond}_31l_alpha50k.npy',
              'nsd_vdvae_nsdgeneral_pred_sub{sub}_31l_alpha50k.npy'),
    'cliptext': ('cliptext_regression_weights.pkl', 'cliptext',
                 'nsd_cliptext_pred_{cond}.npy',
                 'nsd_cliptext_predtest_nsdgeneral.npy'),
    'clipvision': ('clipvision_regression_weights.pkl', 'clipvision',
                   'nsd_clipvision_pred_{cond}.npy',
                   'nsd_clipvision_predtest_nsdgeneral.npy'),
}


def load_reg(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['weight'], d['bias']


def predict_flat(X, W, b):
    # VDVAE ridge: W is (D, V), b is (D,) -> (N, D). Same convention as sklearn's coef_.
    return X @ W.T + b


def predict_tokenwise(X, W, b):
    # CLIP ridges: W is (T, 768, V), b is (T, 768) -> (N, T, 768)
    N, T = X.shape[0], W.shape[0]
    out = np.zeros((N, T, W.shape[1]), dtype=np.float32)
    for t in range(T):
        out[:, t] = X @ W[t].T + b[t]
    return out


def restandardize(pred, pool_mean, pool_std, tgt_mean, tgt_std, eps=1e-8):
    """Map predictions onto the training targets' distribution, exactly as upstream.

    pool_mean/pool_std are computed over the chosen pool rather than over the file itself,
    so that the two conditions of one modality share one affine map. eps guards the
    small-pool std (12 samples instead of 982).
    """
    return (pred - pool_mean) / np.maximum(pool_std, eps) * tgt_std + tgt_mean


def load_fmri(sub, cond, data_root):
    path = os.path.join(data_root, 'processed_data', 'subj{:02d}'.format(sub), 'imagery',
                        'nsd_imagery_{}_fmriavg_nsdgeneral_sub{}.npy'.format(cond, sub))
    if not os.path.isfile(path):
        raise SystemExit('missing {}: run scripts/prepare_nsd_imagery.py first'.format(path))
    X = np.load(path).astype(np.float32)
    if X.shape != (6, NUM_VOXELS[sub]):
        raise SystemExit('{} has shape {}, expected {}'
                         .format(path, X.shape, (6, NUM_VOXELS[sub])))
    # Guard against normalizing twice: this input must already be per-run
    # z-scored, i.e. roughly zero-mean, and must NOT have been re-normalized on top.
    if abs(float(X.mean())) > 0.5:
        print('  WARNING: {} has mean {:.3f}; a per-run z-scored input should be near 0'
              .format(cond, float(X.mean())), file=sys.stderr)
    return X


def pool_moments_from_A(sub, branch, data_root):
    """`fromA` pool: freeze the moments to those of the 982 shared1000 predictions.

    Note that the file on disk is the ALREADY re-standardized prediction of benchmark A
    (vdvae_regression.py:50,53); its moments are the training target moments: using them
    here leaves the imagery predictions at their own, lower, variance, without re-inflating
    them enough. That is why this pool is the fallback, not the default.
    """
    fname = BRANCHES[branch][3].format(sub=sub)
    path = os.path.join(data_root, 'predicted_features', 'subj{:02d}'.format(sub), fname)
    if not os.path.isfile(path):
        raise SystemExit('missing {} (a benchmark-A prediction file, deleted by the disk cleanup)'
                         .format(path))
    a = np.load(path, mmap_mode='r')
    a = np.asarray(a, dtype=np.float32)
    return a.mean(axis=0), a.std(axis=0)


def process_branch(sub, branch, conditions, args, stats):
    wname, skey, out_tmpl, _ = BRANCHES[branch]
    wpath = os.path.join(args.data_root, 'regression_weights',
                         'subj{:02d}'.format(sub), wname)
    if not os.path.isfile(wpath):
        raise SystemExit('missing {}: fit the ridge decoders first (vdvae_regression.py needs the `import pickle` patch)'
                         .format(wpath))

    print('  loading {} ({:.1f} GB)'.format(wname, os.path.getsize(wpath) / 1e9), flush=True)
    W, b = load_reg(wpath)
    if W.shape[-1] != NUM_VOXELS[sub]:
        raise SystemExit('{}: weights have {} voxels, subject {} has {}'
                         .format(wname, W.shape[-1], sub, NUM_VOXELS[sub]))

    raw = {}
    for cond in conditions:
        X = load_fmri(sub, cond, args.data_root)
        raw[cond] = (predict_flat(X, W, b) if branch == 'vdvae'
                     else predict_tokenwise(X, W, b)).astype(np.float32)
    del W, b

    tgt_mean, tgt_std = stats[skey + '_mean'], stats[skey + '_std']
    if raw[conditions[0]].shape[1:] != tgt_mean.shape:
        raise SystemExit('{}: prediction shape {} vs target moments {}'
                         .format(branch, raw[conditions[0]].shape[1:], tgt_mean.shape))

    out_dir = os.path.join(args.data_root, 'predicted_features',
                           'subj{:02d}'.format(sub), 'imagery')
    os.makedirs(out_dir, exist_ok=True)

    for group, conds in group_conditions(conditions, args.restandardize_pool):
        if args.restandardize_pool == 'fromA':
            pool_mean, pool_std = pool_moments_from_A(sub, branch, args.data_root)
        else:
            pool = np.concatenate([raw[c] for c in conds], axis=0)
            pool_mean, pool_std = pool.mean(axis=0), pool.std(axis=0)
            del pool
        for cond in conds:
            pred = restandardize(raw[cond], pool_mean, pool_std, tgt_mean, tgt_std)
            path = os.path.join(out_dir, out_tmpl.format(cond=cond))
            np.save(path, pred.astype(np.float32))
            print('    {} [{}] -> {} {}'.format(cond, group, os.path.basename(path),
                                                pred.shape), flush=True)


def group_conditions(conditions, pool):
    """Return [(pool name, [conditions sharing one affine map]), ...]."""
    if pool == 'pertype' or pool == 'fromA':
        return [(pool, [c]) for c in conditions]
    groups = {}
    for c in conditions:
        groups.setdefault(POOL_OF[c], []).append(c)
    for name, conds in groups.items():
        if name.endswith('12') and len(conds) != 2:
            print('  WARNING: pool {} built from {} instead of the full 12-stimulus pool; '
                  'the moments differ from a complete run'.format(name, conds),
                  file=sys.stderr)
    return sorted(groups.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sub', nargs='+', type=int, default=[1])
    parser.add_argument('--conditions', nargs='+', default=ALL_CONDITIONS)
    parser.add_argument('--restandardize_pool', default='modality12',
                        choices=['modality12', 'pertype', 'fromA'],
                        help='pool for pred.mean(0)/pred.std(0)')
    parser.add_argument('--branches', nargs='+', default=list(BRANCHES),
                        choices=list(BRANCHES))
    parser.add_argument('--data_root', default='data')
    args = parser.parse_args()

    if not os.path.isdir('scripts'):
        raise SystemExit('run this from the brain_diffuser/ root')
    if os.environ.get('OPENBLAS_CORETYPE') != 'SkylakeX':
        # Every prediction here is an X @ W.T: on some CPUs the wrong core type gives a
        # silently wrong product.
        raise SystemExit('OPENBLAS_CORETYPE is not SkylakeX: export it before running')
    for c in args.conditions:
        if c not in POOL_OF:
            raise SystemExit('unknown condition {}'.format(c))

    for sub in args.sub:
        if sub not in SUBJECTS:
            raise SystemExit('subject {} is not one of 1, 2, 5, 7'.format(sub))
        print('=== subj{:02d} (pool: {}) ==='.format(sub, args.restandardize_pool), flush=True)
        stats_path = os.path.join(args.data_root, 'regression_weights',
                                  'subj{:02d}'.format(sub), 'train_target_stats.npz')
        if not os.path.isfile(stats_path):
            raise SystemExit('missing {}: the training-target moments, see the README'.format(stats_path))
        stats = np.load(stats_path)
        for branch in args.branches:
            process_branch(sub, branch, args.conditions, args, stats)
    print('predictions done', flush=True)


if __name__ == '__main__':
    main()
