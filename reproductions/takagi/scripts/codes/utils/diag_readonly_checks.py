"""Read-only checks on the NSD-Imagery outputs already on disk: nothing is written, no GPU.

The `imagery` row reproduces the published failure (chance) but `vision` is above chance (Alex(5) 68.5%, PixCorr
.141). Either something inflates `vision`, or a bug of this pipeline flattens `imagery`; on this model every mistake looks
like success; half of the checks target the second case. The five checks (values of --only):
  gt_order          `_org.png` of the generation against gt_images() of the evaluation: two chains from one tensor
  preparation       the checks of prepare_nsd_imagery_takagi.py, re-read: stim_ids 0..5 / 6..11 / 12..17 and
                    n_reps 8 on vision, 16 on imagery (an 8 on imagery is a lost run)
  distinct_recons   are the 12 reconstructions different from each other? (recon-recon matrix, pixel identifiability)
  modality_overlap  PixCorr(recon_vision[i], recon_imagery[i]) must be low; ~1 means the same files
  captions          the brain captions, dumped for a human to read

Usage (from codes/utils/):
    python diag_readonly_checks.py [--only gt_order preparation]
"""

import argparse
import os

import numpy as np
from PIL import Image

from eval_extract_features_imagery import gt_images, pool_paths
from evaluate_reconstruction_imagery import blas_selftest

SUBJECTS = ['subj01', 'subj02', 'subj05', 'subj07']
MODALITIES = ['vision', 'imagery']
# the two evaluated conditions first, then the one that never enters the tables
CONDITIONS = ['visionsimple', 'visioncomplex', 'imagerysimple', 'imagerycomplex',
              'imageryconcepts']
STIM_RANGE = {'simple': (0, 6), 'complex': (6, 12), 'concepts': (12, 18)}


def flat(path_or_img):
    """One image as a flat float vector in [0, 1], at the 425x425 of the harness."""
    img = Image.open(path_or_img) if isinstance(path_or_img, str) else path_or_img
    img = img.convert('RGB')
    if img.size != (425, 425):
        img = img.resize((425, 425))
    return (np.array(img, dtype=np.float64) / 255.0).reshape(-1)


def corr_matrix(rows_a, rows_b):
    """Pearson correlation of every row of a against every row of b."""
    a = np.asarray(rows_a)
    b = np.asarray(rows_b)
    r = np.corrcoef(a, b)
    return r[:len(a), len(a):]


def check_a1(opt):
    """gt_order: the ground truth of the evaluation against the one of the generation."""
    print('\n=== gt_order  _org.png vs gt_images()')
    gt = gt_images(opt.stimuli_path)
    n_checked, mismatches = 0, []
    for sub in opt.subjects:
        for mod in MODALITIES:
            for i, p in enumerate(pool_paths(opt.recon_dir, sub, mod, opt.sample)):
                org = p.replace('_{:03}.png'.format(opt.sample), '_org.png')
                if not os.path.isfile(org):
                    mismatches.append((sub, mod, i, os.path.basename(org), 'MISSING'))
                    continue
                a = np.array(Image.open(org).convert('RGB').resize((425, 425)),
                             dtype=np.int16)
                b = np.array(gt[i].convert('RGB'), dtype=np.int16)
                d = int(np.abs(a - b).max())
                n_checked += 1
                if d > 1:
                    mismatches.append((sub, mod, i, os.path.basename(org), d))
    for m in mismatches:
        print('  MISMATCH {} {} idx={} {} maxdiff={}'.format(*m))
    print('  {} pairs compared, {} mismatches'.format(n_checked, len(mismatches)))
    if mismatches:
        print('  VERDICT: coupling BROKEN -- compare both chains against the stim_ids of '
              'prepare_nsd_imagery_takagi.py before anything else')
    else:
        print('  VERDICT: OK -- generation and evaluation see the same 12 stimuli, in the '
              'same order. The coupling explains neither (a) nor (b).')
    return not mismatches


def check_a3(opt):
    """preparation: the sanity checks of the preparation script, re-read from disk and from the log."""
    print('\n=== preparation  stim_ids and n_reps of the preparation')
    ok = True
    print('  from the saved *_stim_ids.npy (authoritative):')
    for sub in opt.subjects:
        for cond in CONDITIONS:
            p = os.path.join(opt.mrifeat_dir, sub,
                             '{}_{}_stim_ids.npy'.format(sub, cond))
            if not os.path.isfile(p):
                print('    {} {}: MISSING {}'.format(sub, cond, p))
                ok = False
                continue
            got = np.load(p)
            kind = cond.replace('vision', '').replace('imagery', '')
            lo, hi = STIM_RANGE[kind]
            good = np.array_equal(got, np.arange(lo, hi))
            ok = ok and good
            print('    {} {}: {} {}'.format(sub, cond, got.tolist(),
                                            'OK' if good else 'WRONG, expected {}'
                                            .format(list(range(lo, hi)))))

    # n_reps is not saved as an array: it only exists in the log line of
    # prepare_nsd_imagery_takagi.py. Its absence is not a failure of the pipeline
    # (the script aborts on a wrong value) but it is the one number worth re-reading.
    print('  from prepare_imagery.log:')
    if not os.path.isfile(opt.prepare_log):
        print('    {} not found -- n_reps not re-checkable from here. The value was '
              'asserted at preparation time (check 6 aborts on a mismatch), so this is a '
              'missing confirmation, not a failure.'.format(opt.prepare_log))
        return ok
    with open(opt.prepare_log) as f:
        lines = f.read().splitlines()
    n_reps_seen = 0
    for line in lines:
        s = line.strip()
        if s.startswith('=== ') or s.startswith('image_map:') or 'roi_sizes' in s \
                or '[check' in s or 'nan' in s.lower() or 'scaler_scale' in s:
            print('    {}'.format(s))
        for cond in CONDITIONS:
            if s.startswith('{}:'.format(cond)) and 'reps' in s:
                n_reps_seen += 1
                expected = 8 if cond.startswith('vision') else 16
                got = int(s.split(' reps')[0].split()[-1])
                flag = 'OK' if got == expected else 'WRONG, expected {}'.format(expected)
                if got != expected:
                    ok = False
                print('    {}  -> n_reps {}'.format(s, flag))
    print('  {} condition lines found in the log'.format(n_reps_seen))
    if ok:
        print('  VERDICT: OK -- rows line up with imagery_stimuli_18.pt and no imagery '
              'condition lost a repeated run. The main mechanism of (b) is falsified.')
    else:
        print('  VERDICT: BROKEN -- stop here and redo stage 1')
    return ok


def check_a4(opt):
    """distinct_recons: are the 12 reconstructions distinct, or six variations of one image?"""
    print('\n=== distinct_recons  recon-recon similarity and pixel identifiability')
    gt = np.array([flat(g) for g in gt_images(opt.stimuli_path)])
    gt_rr = corr_matrix(gt, gt)
    off = ~np.eye(12, dtype=bool)
    print('  GT baseline: mean off-diagonal PixCorr among the 12 stimuli {:.3f}'
          .format(gt_rr[off].mean()))
    for sub in opt.subjects:
        for mod in MODALITIES:
            rec = np.array([flat(p) for p in
                            pool_paths(opt.recon_dir, sub, mod, opt.sample)])
            rr = corr_matrix(rec, rec)
            block = np.zeros((12, 12), dtype=bool)
            block[:6, :6] = block[6:, 6:] = True
            within = rr[block & off].mean()
            across = rr[~block].mean()
            rg = corr_matrix(gt, rec)                 # rows: gt, cols: recon
            diag = np.diag(rg)
            ident = int((diag > rg[off].reshape(12, 11).max(axis=1)).sum())
            print('  {} {:8s} recon-recon within-block {:.3f} across-block {:.3f} | '
                  'PixCorr diag {:.3f} vs off {:.3f} | identifiable {}/12'
                  .format(sub, mod, within, across, diag.mean(), rg[off].mean(), ident))
    print('  Reading: within-block high and across-block low means the reconstructions are '
          'generic to the stimulus type. A diag clearly above off, and identifiable well '
          'above 1/12, means the PixCorr is stimulus-specific.')


def check_a5(opt):
    """modality_overlap: do the two modalities share their files?"""
    print('\n=== modality_overlap  cross-modality contamination')
    worst = 0.0
    for sub in opt.subjects:
        v = np.array([flat(p) for p in pool_paths(opt.recon_dir, sub, 'vision', opt.sample)])
        i = np.array([flat(p) for p in pool_paths(opt.recon_dir, sub, 'imagery', opt.sample)])
        m = corr_matrix(v, i)
        d = np.diag(m)
        worst = max(worst, float(d.max()))
        print('  {}: PixCorr(vision[i], imagery[i]) mean {:.3f} max {:.3f} | '
              'off-diagonal mean {:.3f}'
              .format(sub, d.mean(), d.max(), m[~np.eye(12, dtype=bool)].mean()))
    if worst > 0.95:
        print('  VERDICT: CONTAMINATED -- some pairs are the same image (max {:.3f}). '
              'Check stage 4 for a directory swap or a --skip_existing reuse.'.format(worst))
    else:
        print('  VERDICT: OK -- the two conditions are independent generations '
              '(max {:.3f}).'.format(worst))
    return worst <= 0.95


def check_a2(opt):
    """captions: dump the brain captions. Needs a human; it only prints."""
    print('\n=== captions  brain captions (read them: vision informative? imagery degenerate?)')
    for sub in opt.subjects:
        for cond in CONDITIONS:
            p = os.path.join(opt.imagery_dir, sub, 'captions',
                             'captions_brain_{}.csv'.format(cond))
            print('  -- {} {}'.format(sub, cond))
            if not os.path.isfile(p):
                print('     MISSING {}'.format(p))
                continue
            with open(p) as f:
                for line in f.read().splitlines():
                    print('     {}'.format(line))
    print('  Comparison case known to work: decoded/<subj>/captions/captions_brain.csv '
          '(benchmark A).')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subjects', type=str, nargs='+', default=SUBJECTS)
    parser.add_argument('--sample', type=int, default=0,
                        help='which of the 10 samples per stimulus (gt_order, distinct_recons, modality_overlap)')
    parser.add_argument('--only', type=str, nargs='+',
                        choices=['gt_order', 'captions', 'preparation', 'distinct_recons',
                                     'modality_overlap'],
                        default=['gt_order', 'preparation', 'captions', 'distinct_recons',
                                     'modality_overlap'],
                        help='checks to run; the default is all five, in this order')
    parser.add_argument('--recon_dir', type=str, default='../../decoded/image-text-imagery')
    parser.add_argument('--stimuli_path', type=str,
                        default='../../nsd/nsddata_stimuli/stimuli/imagery_stimuli_18.pt')
    parser.add_argument('--mrifeat_dir', type=str, default='../../mrifeat_imagery')
    parser.add_argument('--imagery_dir', type=str, default='../../decoded_imagery')
    parser.add_argument('--prepare_log', type=str, default='../../prepare_imagery.log')
    opt = parser.parse_args()

    if not os.path.isfile('make_subjmri.py'):
        raise SystemExit('run this from codes/utils/')
    blas_selftest()

    runners = {'gt_order': check_a1, 'captions': check_a2, 'preparation': check_a3,
               'distinct_recons': check_a4, 'modality_overlap': check_a5}
    for name in opt.only:
        runners[name](opt)
    print('\ndone')


if __name__ == '__main__':
    main()
