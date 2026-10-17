#!/usr/bin/env python
'''Export the iCNN imagery reconstructions into the layout the shared evaluation harness reads.

Inputs:  recon_B/<subject>/<roi>/seed<SS>/recon_image_normalized-<cond>-<GG>.tiff, from recon_icnn_imagery.py
Outputs: <subj0N>/<cond>/samples/{stim:05}_{sample:03}.png, the pool_paths() layout of
         takagi/codes/utils/eval_extract_features_imagery.py (final_evaluations_imagery.py --model icnn,
         gnet_brain_correlation.py --model takagi)

GG is the stimulus index inside imagery_stimuli_18.pt (00-05 simple, 06-11 complex, 12-17 conceptual); the harness
indexes within a condition, 0..5. This remapping is the whole job and fails silently: a global index scores every
image against the wrong ground truth with a plausible, low number. Images stay 224x224, because each consumer
resizes. All 5 conditions are exported: imageryconcepts is needed for the GNet row.

Usage (from icnn/, no GPU, seconds):
    python src/export_icnn_recons.py --subjects subj01 --seeds 0-1
'''
import argparse
import os
import re

import numpy as np
import PIL.Image


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONDITIONS = ['visionsimple', 'visioncomplex',
              'imagerysimple', 'imagerycomplex', 'imageryconcepts']

# First stimulus index of each condition inside imagery_stimuli_18.pt, i.e. the offset to
# subtract to get the harness's local 0..5 (COND_STIM_IDS in prepare_nsd_imagery_icnn.py).
OFFSET = {'visionsimple': 0, 'visioncomplex': 6,
          'imagerysimple': 0, 'imagerycomplex': 6, 'imageryconcepts': 12}

N_STIM = 6
SUBJECTS = ['subj01', 'subj02', 'subj05', 'subj07']

TIFF_RE = re.compile(r'^recon_image_normalized-(?P<cond>[a-z]+)-(?P<stim>\d{2})\.tiff$')


def parse_seeds(spec):
    seeds = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--recon-dir', default=os.path.join(ROOT, 'recon_B'),
                   help='--output-dir of recon_icnn_imagery.py')
    p.add_argument('--out-dir', default=os.path.join(ROOT, 'recon_B_export'))
    p.add_argument('--roi', default='nsdgeneral')
    p.add_argument('--subjects', nargs='+', default=SUBJECTS)
    p.add_argument('--seeds', default='0-9')
    p.add_argument('--conditions', nargs='+', default=CONDITIONS)
    p.add_argument('--overwrite', action='store_true',
                   help='rewrite PNGs that already exist (default: skip them)')
    return p.parse_args()


def export_one(tiff_path, png_path, overwrite):
    """Copy one reconstruction through PIL, checking it is a 224x224 RGB image."""
    if os.path.exists(png_path) and not overwrite:
        return False
    img = PIL.Image.open(tiff_path)
    img = img.convert('RGB')
    arr = np.asarray(img)
    if arr.shape != (224, 224, 3):
        raise SystemExit('%s: shape %s, expected (224, 224, 3)' % (tiff_path, arr.shape))
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    img.save(png_path)
    return True


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    unknown = [c for c in args.conditions if c not in CONDITIONS]
    if unknown:
        raise SystemExit('unknown condition(s): %s' % ', '.join(unknown))

    n_written = n_skipped = 0
    for subject in args.subjects:
        for seed in seeds:
            src_dir = os.path.join(args.recon_dir, subject, args.roi, 'seed%02d' % seed)
            if not os.path.isdir(src_dir):
                raise SystemExit('missing %s -- has recon_icnn_imagery.py finished?'
                                 % src_dir)

            # Drive the loop from the file names actually on disk; a .tiff whose label
            # does not parse is an error rather than something quietly left behind.
            found = {}
            for name in os.listdir(src_dir):
                if not name.endswith('.tiff'):
                    continue
                m = TIFF_RE.match(name)
                if m is None:
                    raise SystemExit('%s: unexpected file name %s' % (src_dir, name))
                cond, stim = m.group('cond'), int(m.group('stim'))
                if cond not in CONDITIONS:
                    raise SystemExit('%s: unknown condition %s' % (name, cond))
                local = stim - OFFSET[cond]
                if not 0 <= local < N_STIM:
                    raise SystemExit('%s: stimulus %d is outside %s (offset %d)'
                                     % (name, stim, cond, OFFSET[cond]))
                found[(cond, local)] = os.path.join(src_dir, name)

            for cond in args.conditions:
                for local in range(N_STIM):
                    tiff_path = found.get((cond, local))
                    if tiff_path is None:
                        raise SystemExit('%s: no .tiff for %s stimulus %d (global %d)'
                                         % (src_dir, cond, local, local + OFFSET[cond]))
                    png_path = os.path.join(args.out_dir, subject, cond, 'samples',
                                            '%05d_%03d.png' % (local, seed))
                    if export_one(tiff_path, png_path, args.overwrite):
                        n_written += 1
                    else:
                        n_skipped += 1
        print('%s: done (%d seed(s) x %d condition(s))'
              % (subject, len(seeds), len(args.conditions)))

    expected = len(args.subjects) * len(seeds) * len(args.conditions) * N_STIM
    print('\n%d PNG written, %d already present, %d expected'
          % (n_written, n_skipped, expected))
    if n_written + n_skipped != expected:
        raise SystemExit('count mismatch: %d != %d' % (n_written + n_skipped, expected))

    # Re-read from disk what the harness will read, rather than trusting the counters.
    missing = []
    for subject in args.subjects:
        for cond in args.conditions:
            for local in range(N_STIM):
                for seed in seeds:
                    p = os.path.join(args.out_dir, subject, cond, 'samples',
                                     '%05d_%03d.png' % (local, seed))
                    if not os.path.isfile(p):
                        missing.append(p)
    if missing:
        raise SystemExit('%d file(s) missing after export, first: %s'
                         % (len(missing), missing[0]))
    print('verified: every (subject, condition, stimulus, sample) is on disk')
    print('harness:  --recon_dir %s' % args.out_dir)
    print('gnet:     --model takagi --recon_root %s' % args.out_dir)


if __name__ == '__main__':
    main()
