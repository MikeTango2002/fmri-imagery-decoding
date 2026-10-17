#!/usr/bin/env python
'''Do the decoded features collapse towards the training mean on imagery trials but not on vision trials?

Inputs:  the decoded features and model/y_norm.mat of decode_features_chunked.py (no GPU, no betas, seconds)
Outputs: a table per (subject, mode, layer); run it before data/decoded_features/ is deleted

A ridge does not tolerate a scale shift of its input. In pred = H_agg @ Y_unique + (1 - rowsum(H)) * y_mean,
rowsum(H) -> 0 is total contraction: every stimulus gets y_mean and all reconstructions look alike.
spread = mean over units of the std of pred across the 12 stimuli; spread/y_norm divides by the training std
of the unit (1 = full variability, 0 = always y_mean; the number to read); mean r = mean Pearson r between the
12 predictions (1 = identical). Collapse is confirmed if the imagery/vision ratio of spread/y_norm is well
below 1; if comparable, the defect is downstream of the decoder.

Usage (from icnn/, venv active):
    python src/diag_feature_dispersion.py --subjects 1 2 5 7
'''
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The 8 layers the reconstruction actually consumes (LAYERS in recon_icnn_imagery.py). The decoder
# writes all 19; the other 11 never reach an image, and they cannot explain the gap.
LAYERS = ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4', 'fc6', 'fc7', 'fc8']

# The evaluated pool is 12 stimuli per mode: 6 simple + 6 complex, in this order, exactly as
# in final_evaluations_imagery.py. `imageryconcepts` is excluded here for the same reason it
# is excluded from every image metric; no visual ground truth.
#
# The index in the filename is the GLOBAL stimulus index 0..17 written by
# prepare_nsd_imagery_icnn.py: simple 00-05, complex 06-11, concepts 12-17; the complex
# block starts at 6, not at 0.
MODE_CONDITIONS = {
    'vision':  [('visionsimple', 0), ('visioncomplex', 6)],
    'imagery': [('imagerysimple', 0), ('imagerycomplex', 6)],
}
N_STIM = 6

DEFAULT_FEATURES_DIR = os.path.join(
    ROOT, 'feature-decoding', 'data', 'decoded_features', 'NSDImagery',
    'nsdcore40ses_ridge_alpha100_allunits', 'caffe', 'VGG19')
DEFAULT_DECODER_DIR = os.path.join(
    ROOT, 'feature-decoding', 'data', 'feature_decoders', 'NSDcore',
    'nsdcore40ses_ridge_alpha100_allunits', 'caffe', 'VGG19')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subjects', type=int, nargs='+', default=[1, 2, 5, 7])
    p.add_argument('--layers', nargs='+', default=LAYERS, choices=LAYERS)
    p.add_argument('--roi', default='nsdgeneral')
    p.add_argument('--features_dir', default=DEFAULT_FEATURES_DIR,
                   help='.../caffe/VGG19, the tree DecodedFeatures reads')
    p.add_argument('--decoder_dir', default=DEFAULT_DECODER_DIR,
                   help='.../caffe/VGG19, holding <layer>/subjNN/<roi>/model/y_norm.mat')
    p.add_argument('--out_tsv', default='',
                   help='also write the long-format table here')
    return p.parse_args()


def load_mat_array(path, keys):
    """Both .mat writers used here hide the array under a key.

    `bdpy.dataform.save_array` writes MATLAB v7.3, i.e. HDF5, which scipy.io refuses; read it
    back with the matching `load_array` so the axis convention is the one save_array used.
    scipy stays as the fallback for anything written the old way.
    """
    from bdpy.dataform import load_array
    for k in keys:
        try:
            return np.asarray(load_array(path, key=k), dtype=np.float64)
        except (KeyError, RuntimeError):
            continue
    import scipy.io as sio
    d = sio.loadmat(path)
    for k in keys:
        if k in d:
            return np.asarray(d[k], dtype=np.float64)
    real = [k for k in d if not k.startswith('__')]
    if len(real) == 1:
        return np.asarray(d[real[0]], dtype=np.float64)
    raise KeyError(f'{path}: expected one of {keys}, found {real}')


def stimulus_features(features_dir, layer, subj, roi, mode):
    """(12, n_units): the predicted features of the 12 evaluated stimuli, flattened.

    Stimulus order is 0..5 = simple, 6..11 = complex, the order every table of the
    benchmark uses.
    """
    rows = []
    for cond, first in MODE_CONDITIONS[mode]:
        for s in range(first, first + N_STIM):
            path = os.path.join(features_dir, layer, f'subj{subj:02d}', roi,
                                f'{cond}-{s:02d}.mat')
            if not os.path.exists(path):
                raise SystemExit(f'missing decoded feature: {path}\n'
                                 'Has decode_features_chunked.py run for this subject, '
                                 'and were the decoded features deleted afterwards?')
            rows.append(load_mat_array(path, ['feat']).ravel())
    return np.stack(rows)


def mean_pairwise_r(x):
    """Mean off-diagonal Pearson r between the 12 predictions.

    -> 1.0 means the decoder returned the same vector 12 times, which is the strongest
    possible form of the collapse described in the header.
    """
    r = np.corrcoef(x)
    iu = np.triu_indices_from(r, k=1)
    return float(np.nanmean(r[iu]))


def main():
    args = parse_args()

    print('dispersion of the decoded features, imagery vs vision')
    print(f'features: {args.features_dir}')
    print(f'decoder : {args.decoder_dir}\n')

    rows = []
    for subj in args.subjects:
        for layer in args.layers:
            y_norm_path = os.path.join(args.decoder_dir, layer, f'subj{subj:02d}',
                                       args.roi, 'model', 'y_norm.mat')
            if not os.path.exists(y_norm_path):
                raise SystemExit(f'missing {y_norm_path}\n'
                                 'It is written by decode_features_chunked.py next to the '
                                 'decoder stats; without it the ratio cannot be formed.')
            y_norm = load_mat_array(y_norm_path, ['y_norm']).ravel()

            per_mode = {}
            for mode in ('vision', 'imagery'):
                feats = stimulus_features(args.features_dir, layer, subj, args.roi, mode)
                if feats.shape[1] != y_norm.size:
                    raise SystemExit(f'{layer} subj{subj:02d}: features have '
                                     f'{feats.shape[1]} units, y_norm has {y_norm.size}')
                # Spread ACROSS stimuli, per unit. ddof=1 to match every other std in this
                # pipeline (decode_features_chunked.py uses ddof=1 throughout).
                spread = feats.std(axis=0, ddof=1)
                # Units that do not vary in training carry no information and would only
                # add 0/0 to the ratio; they are counted and excluded.
                live = y_norm > 0
                ratio = float(np.mean(spread[live] / y_norm[live]))
                per_mode[mode] = {
                    'spread': float(np.mean(spread)),
                    'ratio': ratio,
                    'r': mean_pairwise_r(feats),
                    'dead': int((~live).sum()),
                    'units': int(y_norm.size),
                }

            v, i = per_mode['vision'], per_mode['imagery']
            rows.append({
                'subj': subj, 'layer': layer,
                'spread_vision': v['spread'], 'spread_imagery': i['spread'],
                'ratio_vision': v['ratio'], 'ratio_imagery': i['ratio'],
                'r_vision': v['r'], 'r_imagery': i['r'],
                'imagery_over_vision': i['ratio'] / v['ratio'] if v['ratio'] else float('nan'),
                'units': v['units'], 'dead_units': v['dead'],
            })

    hdr = (f"{'subj':>5} {'layer':>8} | {'spread/y_norm vis':>17} {'imag':>7} "
           f"{'imag/vis':>9} | {'mean r vis':>10} {'imag':>7}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['subj']:>5} {r['layer']:>8} | {r['ratio_vision']:>17.4f} "
              f"{r['ratio_imagery']:>7.4f} {r['imagery_over_vision']:>9.3f} | "
              f"{r['r_vision']:>10.3f} {r['r_imagery']:>7.3f}")

    # The one line the decision rule is read from.
    rv = float(np.mean([r['ratio_vision'] for r in rows]))
    ri = float(np.mean([r['ratio_imagery'] for r in rows]))
    print('-' * len(hdr))
    print(f"{'mean':>5} {'':>8} | {rv:>17.4f} {ri:>7.4f} {ri / rv if rv else float('nan'):>9.3f} | "
          f"{np.mean([r['r_vision'] for r in rows]):>10.3f} "
          f"{np.mean([r['r_imagery'] for r in rows]):>7.3f}")

    print('\nHow to read it:')
    print('  spread/y_norm ~ 1  the decoder reproduces the full variability of the features')
    print('  spread/y_norm ~ 0  it always predicts y_mean: every stimulus, the same image')
    print('  imag/vis << 1      COLLAPSE: it is specific to imagery')
    print('  imag/vis ~ 1       NO COLLAPSE: the features are as spread as on vision, so')
    print('                     the defect is not the input scale of the ridge')
    print('  mean r -> 1        the 12 predictions are the same vector (total collapse)')

    if args.out_tsv:
        import csv
        with open(args.out_tsv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {args.out_tsv}')


if __name__ == '__main__':
    sys.exit(main())
