"""Per-image metrics on the NSD-Imagery 12-stimulus pool (Takagi paths).

Inputs:  the features of eval_extract_features_imagery.py
Outputs: per_image_<subject>_<modality>_s<sample>.tsv (per_image_bridge_... with --bridge); aggregated downstream

Port of the Brain-Diffuser evaluation (evaluate_reconstruction.py); the metric code is unchanged. The pool is always
the 12 reconstructions of one modality (6 simple + 6 complex, same subject and sample): the two-way comparisons use
the 11 other stimuli as distractors (Kneeland et al., Appendix A.2); per-type tables are not got by running 6
images at a time (5 distractors, another quantity). `--bridge` runs the metrics on the shared1000 pilot to validate
the harness FIRST: on imagery chance is expected; a broken harness gives the right answer. Its distractor pool is
the pilot (~100); only PixCorr, SSIM, EffNet and SwAV compare with Takagi's published shared1000 row.

Usage (from codes/utils/):
    python evaluate_reconstruction_imagery.py --subject subj01 --modality vision --sample 0 [--bridge]
"""

import argparse
import os

import numpy as np
import scipy as sp
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

from eval_extract_features_imagery import bridge_paths, gt_images, pool_paths

NET_LIST = [
    ('inceptionv3', 'avgpool'),
    ('clip', 'final'),
    ('alexnet', 2),
    ('alexnet', 5),
    ('efficientnet', 'avgpool'),
    ('swav', 'avgpool'),
]

COLUMN = {
    ('inceptionv3', 'avgpool'): 'Inception',
    ('clip', 'final'): 'CLIP',
    ('alexnet', 2): 'AlexNet2',
    ('alexnet', 5): 'AlexNet5',
    ('efficientnet', 'avgpool'): 'EffNet',
    ('swav', 'avgpool'): 'SwAV',
}
DISTANCE_NETS = ('efficientnet', 'swav')
METRICS = ['PixCorr', 'SSIM', 'AlexNet2', 'AlexNet5', 'Inception', 'CLIP', 'EffNet', 'SwAV']


def blas_selftest():
    """Refuse to run on a broken BLAS.

    Every two-way metric and PixCorr go through np.corrcoef, i.e. through a gemm. On
    Sapphire Rapids CPUs the OpenBLAS kernel picked by default returns wrong products,
    and the failure mode is a silent collapse of all six two-way scores to ~50%, which
    on Takagi is indistinguishable from the expected result. Compare a gemm against the
    same products computed elementwise, which does not go through BLAS.
    """
    rng = np.random.RandomState(0)
    a = rng.randn(256, 512)
    b = rng.randn(512, 256)
    got = (a @ b)[np.ix_([0, 1, 255], [0, 1, 255])]
    ref = np.array([[float(np.sum(a[i] * b[:, j])) for j in (0, 1, 255)]
                    for i in (0, 1, 255)])
    if not np.allclose(got, ref, atol=1e-8):
        raise SystemExit('BLAS returns wrong matrix products: export OPENBLAS_CORETYPE='
                         'SkylakeX and run again (max abs diff {:.3e})'
                         .format(np.abs(got - ref).max()))


def ssim_pair(gen_gray, gt_gray):
    """Upstream's call, kwargs included.

    `multichannel=True` on an image that rgb2gray already flattened is upstream's own
    quirk; it is kept because the MindEye1 and Brain-Diffuser numbers of the thesis were
    produced with it. It is deprecated in scikit-image 0.19 and REMOVED in 0.23; a
    recent version either raises or silently changes the quantity: pin
    scikit-image==0.19.3 in this venv.
    """
    try:
        return ssim(gen_gray, gt_gray, multichannel=True, gaussian_weights=True,
                    sigma=1.5, use_sample_covariance=False, data_range=1.0)
    except TypeError:
        raise SystemExit('scikit-image dropped multichannel=: install 0.19.3 in this venv, '
                         'otherwise SSIM is not comparable with the other four models')


def pairwise_corr_per_image(ground_truth, predictions):
    """Brain-Diffuser's pairwise_corr_all, stopping one step earlier.

    Upstream returns np.mean(success_cnt)/(len(gt)-1); here the mean is not taken, and each
    image keeps its own score and tables 3-4 can be built by averaging per stimulus type
    over the same 12-image pool.
    """
    r = np.corrcoef(ground_truth, predictions)
    r = r[:len(ground_truth), len(ground_truth):]  # rows: ground truth, cols: predictions
    congruents = np.diag(r)
    success = r < congruents
    success_cnt = np.sum(success, 0)
    # the diagonal of 'success' is always zero; the divisor is therefore len-1
    return success_cnt / (len(ground_truth) - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subject', type=str, default='subj01')
    parser.add_argument('--modality', type=str, default='imagery',
                        choices=['vision', 'imagery'])
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--bridge', action='store_true',
                        help='shared1000 pilot instead of imagery')
    parser.add_argument('--method', type=str, default='text')
    parser.add_argument('--recon_dir', type=str, default='../../decoded/image-text-imagery')
    parser.add_argument('--stimuli_path', type=str,
                        default='../../nsd/nsddata_stimuli/stimuli/imagery_stimuli_18.pt')
    parser.add_argument('--feat_dir', type=str, default='../../eval_features_imagery')
    parser.add_argument('--metrics_dir', type=str, default='../../results/imagery_metrics')
    opt = parser.parse_args()

    if not os.path.isfile('make_subjmri.py'):
        raise SystemExit('run this from codes/utils/')
    blas_selftest()

    if opt.bridge:
        bridge_dir = os.path.join('../../decoded', 'image-{}'.format(opt.method),
                                  opt.subject, 'samples')
        gt_paths, rec_paths = bridge_paths(bridge_dir, opt.sample)
        gt_imgs = gt_paths
        feats_dir = os.path.join(opt.feat_dir, 'bridge',
                                 '{}_{}_s{}'.format(opt.subject, opt.method, opt.sample))
        gt_feats_dir = os.path.join(opt.feat_dir, 'bridge',
                                    'gt_{}_{}'.format(opt.subject, opt.method))
        out_name = 'per_image_bridge_{}_{}_s{}.tsv'.format(opt.subject, opt.method,
                                                           opt.sample)
        stim_types = ['shared1000'] * len(rec_paths)
        modality = 'shared1000'
    else:
        assert opt.subject in ['subj01', 'subj02', 'subj05', 'subj07']
        rec_paths = pool_paths(opt.recon_dir, opt.subject, opt.modality, opt.sample)
        gt_imgs = gt_images(opt.stimuli_path)
        feats_dir = os.path.join(opt.feat_dir, opt.subject,
                                 '{}12_s{}'.format(opt.modality, opt.sample))
        gt_feats_dir = os.path.join(opt.feat_dir, 'gt12')
        out_name = 'per_image_{}_{}_s{}.tsv'.format(opt.subject, opt.modality, opt.sample)
        stim_types = ['simple'] * 6 + ['complex'] * 6
        modality = opt.modality

    for d in (feats_dir, gt_feats_dir):
        if not os.path.isdir(d):
            raise SystemExit('missing {}: run eval_extract_features_imagery.py first'
                             .format(d))
    missing = [p for p in rec_paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit('missing {} reconstruction(s), first: {}'
                         .format(len(missing), missing[0]))

    n_pool = len(rec_paths)
    if not opt.bridge and n_pool != 12:
        raise SystemExit('pool has {} images, expected 12'.format(n_pool))

    results = {}
    distance_fn = sp.spatial.distance.correlation
    for (net_name, layer) in NET_LIST:
        gt_feat = np.load('{}/{}_{}.npy'.format(gt_feats_dir, net_name, layer))
        eval_feat = np.load('{}/{}_{}.npy'.format(feats_dir, net_name, layer))
        gt_feat = gt_feat.reshape((len(gt_feat), -1))
        eval_feat = eval_feat.reshape((len(eval_feat), -1))
        if len(gt_feat) != n_pool or len(eval_feat) != n_pool:
            raise SystemExit('{}_{}: pool sizes gt={} eval={}, expected {}'
                             .format(net_name, layer, len(gt_feat), len(eval_feat), n_pool))

        if net_name in DISTANCE_NETS:
            vals = np.array([distance_fn(gt_feat[i], eval_feat[i]) for i in range(n_pool)])
        else:
            vals = pairwise_corr_per_image(gt_feat, eval_feat)
        results[COLUMN[(net_name, layer)]] = vals
        print('{} {}: mean {:.4f}'.format(net_name, layer, vals.mean()))

    # PixCorr and SSIM need no features: they are computed on the images themselves.
    ssim_list, pixcorr_list = [], []
    for i, rec_path in enumerate(rec_paths):
        gen_image = Image.open(rec_path).convert('RGB').resize((425, 425))
        gt_image = gt_imgs[i]
        if isinstance(gt_image, str):
            gt_image = Image.open(gt_image)
        gt_image = gt_image.convert('RGB')
        gen_image = np.array(gen_image) / 255.0
        gt_image = np.array(gt_image) / 255.0
        pixcorr_list.append(np.corrcoef(gt_image.reshape(1, -1),
                                        gen_image.reshape(1, -1))[0, 1])
        ssim_list.append(ssim_pair(rgb2gray(gen_image), rgb2gray(gt_image)))
    results['PixCorr'] = np.array(pixcorr_list)
    results['SSIM'] = np.array(ssim_list)
    print('PixCorr: {:.4f}'.format(results['PixCorr'].mean()))
    print('SSIM: {:.4f}'.format(results['SSIM'].mean()))

    if opt.bridge:
        print('NOTE: two-way scores above use a pool of {} distractors, not 11: only '
              'PixCorr/SSIM/EffNet/SwAV are comparable with the published shared1000 row'
              .format(n_pool - 1))

    os.makedirs(opt.metrics_dir, exist_ok=True)
    out_path = os.path.join(opt.metrics_dir, out_name)
    header = ['sub', 'modality', 'sample', 'stim_idx', 'stim_type'] + METRICS
    with open(out_path, 'w') as f:
        f.write('\t'.join(header) + '\n')
        for i in range(n_pool):
            # imagery: the position in the pool, 0..11 (simple first);
            # bridge: the NSD index of the pilot, which is not contiguous.
            stim_id = i if not opt.bridge else int(
                os.path.basename(rec_paths[i]).split('_')[0])
            row = [opt.subject, modality, str(opt.sample), str(stim_id), stim_types[i]]
            row += ['{:.6f}'.format(float(results[m][i])) for m in METRICS]
            f.write('\t'.join(row) + '\n')
    print('wrote {}'.format(out_path))


if __name__ == '__main__':
    main()
