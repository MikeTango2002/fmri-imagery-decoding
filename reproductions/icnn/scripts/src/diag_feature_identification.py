#!/usr/bin/env python
'''Where does iCNN's imagery signal die: in the ridge, or in the reconstruction?

Inputs:  decoded features of the 12 evaluated stimuli, and their true VGG19 features (as in
         decode_features_chunked.py: bicubic resize to 224, RGB, no mean subtraction, A.6 n.1)
Outputs: --out_tsv, per (subject, layer, mode): prof r, ident, their centred versions, vis-ident

ident = fraction of the 11 distractors whose true features correlate less with the prediction than the correct
one does (chance 0.5). It is scale-free; a contraction (diag_feature_dispersion.py) cannot move it. Above
chance: the ridge generalises to imagery and the gap arises in the image optimisation. At chance: the decoder
does not cross the vision-to-imagery shift. Centring removes the stimulus-independent profile of raw VGG19
features; vis-ident compares the imagery prediction with the vision predictions, with no VGG19.

Usage (from icnn/, venv active; the GPU is only for the forward pass):
    CUDA_VISIBLE_DEVICES=0 python src/diag_feature_identification.py --subjects 1 2 5 7 --out_tsv ident.tsv
'''
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from diag_feature_dispersion import (LAYERS, MODE_CONDITIONS, N_STIM, DEFAULT_FEATURES_DIR,
                                     load_mat_array, stimulus_features)

DEFAULT_STIMULI = os.path.join(ROOT, 'feature-decoding', 'data', 'nsd_imagery',
                               'nsddata_stimuli', 'stimuli', 'imagery_stimuli_18.pt')
DEFAULT_MODELS = os.path.join(ROOT, 'models')

# The 12 evaluated stimuli are indices 0..11 of imagery_stimuli_18.pt: 0-5 simple, 6-11
# complex, 12-17 conceptual (COND_STIM_IDS in prepare_nsd_imagery_icnn.py).
N_EVAL = 12


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subjects', type=int, nargs='+', default=[1, 2, 5, 7])
    p.add_argument('--layers', nargs='+', default=LAYERS, choices=LAYERS)
    p.add_argument('--roi', default='nsdgeneral')
    p.add_argument('--features_dir', default=DEFAULT_FEATURES_DIR)
    p.add_argument('--stimuli', default=DEFAULT_STIMULI)
    p.add_argument('--models_dir', default=DEFAULT_MODELS)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out_tsv', default='')
    return p.parse_args()


def true_features(stimuli_pt, models_dir, layers, device):
    """(12, n_units) true VGG19 features per layer, through decode_features_chunked's path."""
    import torch
    from PIL import Image
    from decode_features_chunked import load_encoder, preprocess_batch

    s = torch.load(stimuli_pt, map_location='cpu')          # (18, 3, 425, 425) float in [0,1]
    if s.shape[0] < N_EVAL:
        raise SystemExit(f'{stimuli_pt}: {s.shape[0]} stimuli, expected at least {N_EVAL}')
    imgs = (s[:N_EVAL].permute(0, 2, 3, 1).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)

    # build_image_cache's resize, stimulus by stimulus: PIL bicubic, no antialias.
    resized = np.stack([np.array(Image.fromarray(im).resize((224, 224), resample=2))
                        for im in imgs])

    from pathlib import Path
    encoder, store = load_encoder(Path(models_dir), layers, device)
    with torch.no_grad():
        x = preprocess_batch(resized, None, False, device)
        encoder(x)
        return {lyr: store[lyr].reshape(N_EVAL, -1).float().cpu().numpy().astype(np.float64)
                for lyr in layers}


def identification(pred, target):
    """Fraction of the 11 distractors beaten, averaged over the 12 stimuli. Chance 0.5.

    Correlation, not distance: it is the comparison Shen's evaluation.py makes, and it is
    invariant to the scale contraction that diag_feature_dispersion.py measured, which is
    the whole point of running this.
    """
    n = pred.shape[0]
    r = np.corrcoef(np.vstack([pred, target]))[:n, n:]        # (n, n), r[i, j] = pred_i x true_j
    hits, total = 0.0, 0
    for i in range(n):
        others = np.delete(np.arange(n), i)
        hits += float(np.sum(r[i, i] > r[i, others]))
        total += n - 1
    return hits / total, float(np.mean(np.diag(r)))


def main():
    args = parse_args()

    print('Do the decoded features identify their own stimulus?')
    print(f'features: {args.features_dir}')
    print(f'stimuli : {args.stimuli}\n')

    truth = true_features(args.stimuli, args.models_dir, args.layers, args.device)

    rows = []
    for subj in args.subjects:
        for layer in args.layers:
            t = truth[layer]
            tc = t - t.mean(axis=0, keepdims=True)
            preds = {}
            rec = {'subj': subj, 'layer': layer, 'units': t.shape[1]}
            for mode in ('vision', 'imagery'):
                p = stimulus_features(args.features_dir, layer, subj, args.roi, mode)
                if p.shape[1] != t.shape[1]:
                    raise SystemExit(f'{layer} subj{subj:02d}: decoded features have '
                                     f'{p.shape[1]} units, VGG19 gives {t.shape[1]}')
                preds[mode] = p
                pc = p - p.mean(axis=0, keepdims=True)
                rec[f'ident_{mode}'], rec[f'prof_{mode}'] = identification(p, t)
                rec[f'identc_{mode}'], rec[f'profc_{mode}'] = identification(pc, tc)
            # imagery against the vision predictions: same decoder, same stimuli, no VGG19.
            rec['ident_cross'], rec['prof_cross'] = identification(preds['imagery'],
                                                                  preds['vision'])
            rows.append(rec)

    hdr = (f"{'subj':>5} {'layer':>8} | {'ident vis':>9} {'imag':>6} | "
           f"{'centred vis':>11} {'imag':>6} | {'prof-c vis':>10} {'imag':>6} | "
           f"{'imag~vis':>8}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['subj']:>5} {r['layer']:>8} | {r['ident_vision']:>9.3f} "
              f"{r['ident_imagery']:>6.3f} | {r['identc_vision']:>11.3f} "
              f"{r['identc_imagery']:>6.3f} | {r['profc_vision']:>10.3f} "
              f"{r['profc_imagery']:>6.3f} | {r['ident_cross']:>8.3f}")

    def col(name):
        return float(np.mean([r[name] for r in rows]))

    print('-' * len(hdr))
    print(f"{'mean':>5} {'':>8} | {col('ident_vision'):>9.3f} {col('ident_imagery'):>6.3f} | "
          f"{col('identc_vision'):>11.3f} {col('identc_imagery'):>6.3f} | "
          f"{col('profc_vision'):>10.3f} {col('profc_imagery'):>6.3f} | "
          f"{col('ident_cross'):>8.3f}")

    print('\nHow to read it:')
    print('  ident      chance 0.5. Scale-free: a contraction towards y_mean cannot move it.')
    print('  centred    the same after removing the shared feature profile. The sharp test.')
    print('  imagery centred ident >> 0.5  the ridge DOES generalise -> the gap is downstream')
    print('  imagery centred ident ~ 0.5   the ridge does not cross vision->imagery')

    if args.out_tsv:
        import csv
        with open(args.out_tsv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {args.out_tsv}')


if __name__ == '__main__':
    sys.exit(main())
