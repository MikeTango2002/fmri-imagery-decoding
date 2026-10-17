"""Prepare the NSD-Imagery betas in the format Brain-Diffuser expects at inference time.

Inputs:  betas_nsdimagery.nii.gz (X,Y,Z,720) per subject, the authors' preprocessed tensors (--reference_dir)
Outputs: data/processed_data/subjNN/imagery/nsd_imagery_<cond>_fmriavg_nsdgeneral_subN.npy (6, V) and
         the nested and unnormalized versions

Follows Kneeland et al.: nsdgeneral > 0 mask, /300, per-run z-score over the 12 run types, mean over
repetitions per condition. Checks run inline and the script exits non-zero on the first failure, among them
the comparison with the authors' tensors. Nothing is trained: NSD-Imagery is a held-out test set.

Usage (from the brain_diffuser/ root, OPENBLAS_CORETYPE=SkylakeX exported):
    python scripts/prepare_nsd_imagery.py --sub 1 2 5 7 --data_root data --reference_dir data/reference
"""

import argparse
import os
import pickle
import sys

import nibabel as nb
import numpy as np
import torch
import torchvision.transforms as T
from scipy.stats import zscore

# nsdgeneral voxel counts per subject, as in the Brain-Diffuser paper
NUM_VOXELS = {1: 15724, 2: 14278, 5: 13039, 7: 12682}

# One key per run type: 3 vision, 6 imagery (two repeats x 3 types), 3 attention.
RUN_TYPES = ['visA', 'visB', 'visC',
             'imgA_1', 'imgA_2', 'imgB_1', 'imgB_2', 'imgC_1', 'imgC_2',
             'attA', 'attB', 'attC']

# Trials per run type: 48 for vision/imagery, 96 for attention -> 720 in total.
RUN_COUNTS = {k: (96 if k.startswith('att') else 48) for k in RUN_TYPES}

# Run types that actually get z-scored. Kneeland et al. allocate the normalized tensor with
# torch.zeros and fill only these nine: in their nsd_imagery.pt the 288 attention rows are
# left at exactly zero (verified on subject1: mean 0, std 0, correlation 0 with a z-scored
# version). The attention runs are excluded from the benchmark anyway; this only matters
# for the byte-level cross-check against their tensors.
NORM_RUN_TYPES = [k for k in RUN_TYPES if not k.startswith('att')]

# visionconcepts is deliberately absent: the protocol excludes the vision trials of the
# conceptual stimuli (Kneeland et al., Section 3), and do the attention runs.
CONDITIONS = {
    'visionsimple':    ['visA'],
    'visioncomplex':   ['visB'],
    'visionconcepts':  ['visC'],          # produced only with --keep_visionconcepts
    'imagerysimple':   ['imgA_1', 'imgA_2'],
    'imagerycomplex':  ['imgB_1', 'imgB_2'],
    'imageryconcepts': ['imgC_1', 'imgC_2'],
}
BENCHMARK_CONDITIONS = ['visionsimple', 'visioncomplex',
                        'imagerysimple', 'imagerycomplex', 'imageryconcepts']

# Expected stimulus indices per condition, matching imagery_stimuli_18.pt.
STIM_RANGE = {'simple': (0, 6), 'complex': (6, 12), 'concepts': (12, 18)}


def stim_range_of(cond):
    for key, rng in STIM_RANGE.items():
        if cond.endswith(key):
            return rng
    raise SystemExit('unknown condition {}'.format(cond))


def fail(msg):
    raise SystemExit('CHECK FAILED: {}'.format(msg))


def load_stimuli_meta(data_root):
    """Load nsdimagery_stimuli.pkl3 (trial metadata built by Kneeland et al., HF repo)."""
    path = os.path.join(data_root, 'nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3')
    if not os.path.isfile(path):
        raise SystemExit('missing {}'.format(path))
    with open(path, 'rb') as f:
        try:
            meta = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            meta = pickle.load(f, encoding='latin1')   # pickle written under Python 2
    exps = np.asarray(meta['exps']).ravel()
    cues = np.asarray(meta['cues']).ravel()
    image_map = meta['image_map']
    return exps, cues, image_map


def check_exps(exps):
    """720 trials, 48 per vision/imagery run, 96 per attention run."""
    if len(exps) != 720:
        fail('len(exps) = {}, expected 720'.format(len(exps)))
    names, counts = np.unique(exps, return_counts=True)
    got = dict(zip([str(n) for n in names], [int(c) for c in counts]))
    if got != RUN_COUNTS:
        fail('run counts {} differ from the expected {}'.format(got, RUN_COUNTS))


def extract_imagery_betas(subject, data_root):
    """Apply the nsdgeneral mask, with the `> 0` threshold of prepare_nsddata.py."""
    beta_file = os.path.join(
        data_root, 'nsddata_betas/ppdata', 'subj{:02d}'.format(subject),
        'func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz')
    roi_file = os.path.join(
        data_root, 'nsddata/ppdata', 'subj{:02d}'.format(subject),
        'func1pt8mm/roi/nsdgeneral.nii.gz')
    for p in (beta_file, roi_file):
        if not os.path.isfile(p):
            raise SystemExit('missing {}'.format(p))

    roi = nb.load(roi_file).get_fdata()
    # `> 0` is what prepare_nsddata.py:98,104 uses for the NSD-core betas. It MUST be the
    # same criterion here, otherwise the ridge weights would be applied to a different set
    # of voxels (or, worse, to the same count in a different order).
    n_gt0, n_eq1 = int((roi > 0).sum()), int((roi == 1.0).sum())
    if not (n_gt0 == n_eq1 == NUM_VOXELS[subject]):
        fail('subj{:02d}: mask counts (>0)={} (==1.0)={} expected {}. A non-binary mask '
             'means benchmark A ran on a different voxel set'
             .format(subject, n_gt0, n_eq1, NUM_VOXELS[subject]))

    betas = nb.load(beta_file).get_fdata()          # (X, Y, Z, 720)
    betas = betas.transpose((3, 0, 1, 2))           # (720, X, Y, Z)
    mask = (roi > 0).flatten()
    x = betas.reshape((len(betas), -1))[:, mask].astype(np.float32)
    del betas

    if x.shape != (720, NUM_VOXELS[subject]):
        fail('subj{:02d}: extracted {} expected {}'
             .format(subject, x.shape, (720, NUM_VOXELS[subject])))

    # Second alignment check: the imagery betas must have as many voxels as the NSD-core
    # training matrix the ridges were fitted on.
    train_path = os.path.join(data_root, 'processed_data', 'subj{:02d}'.format(subject),
                              'nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(subject))
    if os.path.isfile(train_path):
        train = np.load(train_path, mmap_mode='r')
        if x.shape[1] != train.shape[1]:
            fail('subj{:02d}: imagery has {} voxels, nsd_train has {}'
                 .format(subject, x.shape[1], train.shape[1]))
    else:
        print('  note: {} is gone, skipping the voxel-count cross-check'.format(train_path))

    return torch.from_numpy(x)


def normalize_imagery_betas(x, exps, nan_to_zero):
    """/300, then z-score each run separately (per voxel, along trials).

    Two conventions here are not free choices: both were reverse-engineered from Kneeland
    et al.'s own nsd_imagery.pt and are what makes compare_reference() pass exactly.

    * ddof=1. Their per-run std is 0.9895 = sqrt(47/48) when measured with ddof=0, i.e. they
      divided by the sample std (n-1), the default of torch.std, not scipy's zscore, whose
      default is ddof=0. On 48-trial runs the two differ by ~1%.
    * attention runs left at zero, see NORM_RUN_TYPES.

    Downstream the ddof is inert: it is a single global scale factor on the fMRI input, and
    the re-standardization of the ridge predictions removes any such scale.
    It is kept because matching the authors' tensor bit-for-bit is what validates the mask,
    the trial order and the run partition all at once.
    """
    meta_cond_idx = {c: np.arange(len(exps))[exps == c] for c in NORM_RUN_TYPES}

    x = x / 300.0                     # NSD betas are int16 scaled by 300
    x_norm = torch.zeros_like(x)      # attention rows stay zero, as in the reference tensor
    for _, idx in meta_cond_idx.items():
        # axis=0 is scipy's default, but MIRAGE (src/utils.py:1128) leaves it implicit:
        # with axis=None the numbers would differ. Keep it explicit.
        x_norm[idx] = torch.from_numpy(
            zscore(x[idx].numpy(), axis=0, ddof=1).astype(np.float32))

    n_nan = int(torch.isnan(x_norm).sum())
    if n_nan:
        if not nan_to_zero:
            fail('{} NaN after the per-run z-score: a voxel with zero variance inside a '
                 'run. Re-run with --nan_to_zero and declare it in the thesis'
                 .format(n_nan))
        print('  --nan_to_zero: replacing {} NaN with 0'.format(n_nan))
        x_norm = torch.nan_to_num(x_norm)

    # Check per run, not over the whole tensor: mean ~ 0 and std ~ sqrt((n-1)/n).
    # The target is not 1.0 because of the ddof=1 above: dividing by the sample std leaves a
    # population std of sqrt(47/48) = 0.98953 on a 48-trial run. This is the same 0.9895 the
    # authors' tensor shows, and checking against it is what pins the convention down.
    for c, idx in meta_cond_idx.items():
        expected_std = np.sqrt((len(idx) - 1) / len(idx))
        m, s = float(x_norm[idx].mean()), float(x_norm[idx].std())
        if abs(m) > 1e-4:
            fail('run {}: mean {:.3e}, expected ~0'.format(c, m))
        if abs(s - expected_std) > 1e-3:
            fail('run {}: std {:.4f}, expected ~{:.4f} (ddof=1 on {} trials)'
                 .format(c, s, expected_std, len(idx)))

    return x_norm, meta_cond_idx


def group_by_stimulus(x_norm, exps, cues, image_map, cond):
    """Average the repetitions of each stimulus (Brain-Diffuser takes the mean, not the nested array)."""
    idx = np.arange(len(exps))[np.isin(exps, CONDITIONS[cond])]
    conditionals = np.array([image_map[c] for c in cues[idx]])   # stimulus index per trial
    x_cond = x_norm[idx]

    stim_ids = np.unique(conditionals)
    reps = [x_cond[conditionals == s] for s in stim_ids]
    n_reps = min(r.shape[0] for r in reps)
    # Truncating to min(n_reps) instead of zero-padding to max (MIRAGE src/utils.py:634-642):
    # on NSD-Imagery the counts are uniform (8 or 16); the two agree.
    x_nested = torch.stack([r[:n_reps] for r in reps])           # (6, n_reps, V)
    x_out = x_nested.mean(dim=1)                                 # (6, V) <- Brain-Diffuser
    return x_out, x_nested, stim_ids


def check_condition(cond, x_out, x_nested, stim_ids, num_voxels):
    """Check stimulus order, shape and number of repetitions of one condition."""
    lo, hi = stim_range_of(cond)
    if not np.array_equal(np.asarray(stim_ids), np.arange(lo, hi)):
        fail('{}: stim_ids {} expected {}. The x<->y pairing with imagery_stimuli_18.pt '
             'would be silently wrong'.format(cond, stim_ids, np.arange(lo, hi)))
    if tuple(x_out.shape) != (6, num_voxels):
        fail('{}: averaged shape {} expected {}'.format(cond, tuple(x_out.shape),
                                                        (6, num_voxels)))
    expected_reps = 8 if cond.startswith('vision') else 16
    if x_nested.shape[1] != expected_reps:
        fail('{}: {} repetitions, expected {}. An imagery condition with 8 means one of the '
             'two run repeats (imgX_2) was lost'
             .format(cond, x_nested.shape[1], expected_reps))


def compare_reference(subject, x_unnorm, x_norm, reference_dir, allow_missing):
    """torch.allclose against Kneeland et al.'s own preprocessed tensors.

    Their tensors are built with `== 1.0` on the mask while this script uses `> 0`; a match is also
    the definitive check that the two criteria select the same voxels.

    This is the only check here that compares against something not computed here: it is the
    only one that can catch a wrong recipe rather than a wrong execution of it. An
    absent reference would look identical to a passing check; a missing reference is fatal
    unless --allow_missing_reference says otherwise.
    """
    base = os.path.join(reference_dir, 'preprocessed_data', 'subject{}'.format(subject))
    if not os.path.isdir(base):
        if not allow_missing:
            fail('reference tensors not found under {}. Download them or '
                 'pass --allow_missing_reference and say so in the thesis.'.format(base))
        print('  reference tensors not found under {}, cross-check DISABLED'.format(base))
        return
    for name, mine in (('nsd_imagery_unnormalized.pt', x_unnorm), ('nsd_imagery.pt', x_norm)):
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            if not allow_missing:
                fail('{} missing. Download it or pass '
                     '--allow_missing_reference.'.format(path))
            print('  {} missing, cross-check DISABLED'.format(path))
            continue
        ref = torch.load(path, map_location='cpu').float()
        if ref.shape != mine.shape:
            fail('{}: reference shape {} vs ours {}'.format(name, tuple(ref.shape),
                                                            tuple(mine.shape)))
        # The unnormalized tensor holds the raw masked betas on both sides: the /300 lives
        # only in the normalized branch; both files are directly comparable.
        ok = torch.allclose(mine, ref, atol=1e-3)
        maxdiff = float((mine - ref).abs().max())
        print('  {}: allclose={} max|diff|={:.3e}'.format(name, ok, maxdiff))
        if not ok:
            fail('{} differs from the reference (max|diff| {:.3e}): mask or order of '
                 'operations is wrong'.format(name, maxdiff))


def save_ground_truth_pngs(data_root):
    """Brain-Diffuser's evaluation compares 425x425 pngs on disk, not tensors."""
    src = os.path.join(data_root, 'nsddata_stimuli/stimuli/imagery_stimuli_18.pt')
    if not os.path.isfile(src):
        raise SystemExit('missing {}'.format(src))
    out = os.path.join(data_root, 'nsddata_stimuli/imagery_test_images')
    os.makedirs(out, exist_ok=True)

    s = torch.load(src, map_location='cpu')
    print('imagery_stimuli_18.pt: {} {} range [{:.3f}, {:.3f}]'
          .format(tuple(s.shape), s.dtype, float(s.min()), float(s.max())))
    if tuple(s.shape) != (18, 3, 425, 425):
        fail('imagery_stimuli_18.pt has shape {}, expected (18, 3, 425, 425)'
             .format(tuple(s.shape)))
    if s.dtype != torch.uint8 and float(s.max()) > 1.5:
        # ToPILImage() would clip a float tensor already in [0, 255] -> PixCorr and SSIM
        # would then compare different scales.
        s = s.to(torch.uint8)
    for i in range(18):
        T.ToPILImage()(s[i]).save(os.path.join(out, '{}.png'.format(i)))
    print('wrote 18 ground-truth pngs to {}'.format(out))


def process_subject(subject, args, exps, cues, image_map, conditions):
    print('=== subj{:02d} ==='.format(subject), flush=True)
    out_dir = os.path.join(args.data_root, 'processed_data',
                           'subj{:02d}'.format(subject), 'imagery')
    os.makedirs(out_dir, exist_ok=True)

    x_unnorm = extract_imagery_betas(subject, args.data_root)
    np.save(os.path.join(out_dir, 'nsd_imagery_unnormalized_sub{}.npy'.format(subject)),
            x_unnorm.numpy())

    x_norm, _ = normalize_imagery_betas(x_unnorm, exps, args.nan_to_zero)
    np.save(os.path.join(out_dir, 'nsd_imagery_sub{}.npy'.format(subject)), x_norm.numpy())

    compare_reference(subject, x_unnorm, x_norm, args.reference_dir,
                      args.allow_missing_reference)

    for cond in conditions:
        x_out, x_nested, stim_ids = group_by_stimulus(x_norm, exps, cues, image_map, cond)
        check_condition(cond, x_out, x_nested, stim_ids, NUM_VOXELS[subject])
        np.save(os.path.join(out_dir, 'nsd_imagery_{}_fmriavg_nsdgeneral_sub{}.npy'
                             .format(cond, subject)), x_out.numpy())
        np.save(os.path.join(out_dir, 'nsd_imagery_{}_nested_sub{}.npy'.format(cond, subject)),
                x_nested.numpy())
        np.save(os.path.join(out_dir, 'stim_ids_{}.npy'.format(cond)), np.asarray(stim_ids))
        print('  {}: {} (nested {}) stim_ids {}'
              .format(cond, tuple(x_out.shape), tuple(x_nested.shape), list(stim_ids)),
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sub', nargs='+', type=int, default=[1, 2, 5, 7])
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--reference_dir', default='data/reference',
                        help="Kneeland et al.'s preprocessed tensors, for the cross-check")
    parser.add_argument('--allow_missing_reference', action='store_true',
                        help='downgrade a missing reference tensor from fatal to a warning')
    parser.add_argument('--nan_to_zero', action='store_true',
                        help='replace NaN from zero-variance voxels; declare it in the thesis')
    parser.add_argument('--keep_visionconcepts', action='store_true',
                        help='also write visionconcepts, which the benchmark protocol excludes')
    args = parser.parse_args()

    if not os.path.isdir('scripts'):
        raise SystemExit('run this from the brain_diffuser/ root')
    if os.environ.get('OPENBLAS_CORETYPE') != 'SkylakeX':
        print('WARNING: OPENBLAS_CORETYPE is not SkylakeX (wrong matrix products on the server)',
              file=sys.stderr)
    for s in args.sub:
        if s not in NUM_VOXELS:
            raise SystemExit('subject {} is not one of 1, 2, 5, 7'.format(s))

    exps, cues, image_map = load_stimuli_meta(args.data_root)
    check_exps(exps)
    print('exps ok: 720 trials, counts {}'.format(RUN_COUNTS))

    save_ground_truth_pngs(args.data_root)

    conditions = list(BENCHMARK_CONDITIONS)
    if args.keep_visionconcepts:
        conditions.append('visionconcepts')

    for s in args.sub:
        process_subject(s, args, exps, cues, image_map, conditions)

    print('all checks passed', flush=True)


if __name__ == '__main__':
    main()
