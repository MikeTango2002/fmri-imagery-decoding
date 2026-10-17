#!/usr/bin/env python
'''iCNN reconstruction (stage 2) for NSD-Imagery: 10 seeds per stimulus.

Inputs:  the decoded features of decode_features_chunked.py (30 labels per subject: 5 conditions x 6 stimuli)
Outputs: recon_B/<subject>/<roi>/seed<SS>/recon_image_normalized-<cond>-<GG>.tiff

Same reconstruction as recon_icnn_A.py. New here: 10 seeds (each a different draw of the generator's latent code),
one output directory per seed, and a --conditions filter. The separate directories are mandatory: the resume
check looks for an existing .tiff and every seed writes the same name; a shared directory would give 10
identical samples. n_iter = 1000, not 200 (+0.030 +/- 0.021 PixCorr on 100 shared1000 stimuli). normalize_feature
stays on: switching it off too (--no-normfeat) moved PixCorr by -0.0126 +/- 0.0071; A.6 n.1 is about stage 0 only.

Usage (from icnn/, venv active, OPENBLAS_CORETYPE=SkylakeX; a killed run resumes where it stopped):
    CUDA_VISIBLE_DEVICES=0 python src/recon_icnn_imagery.py --subject subj01
'''
import argparse
import glob
import os

from bdpy.recon.torch.icnn import reconstruct
from bdpy.recon.utils import normalize_image, clip_extreme
from bdpy.dl.torch.models import VGG19, AlexNetGenerator, layer_map
from bdpy.dataform import DecodedFeatures
from bdpy.feature import normalize_feature
from bdpy.util import dump_info
import numpy as np
import PIL.Image
import scipy.io as sio
import torch
import torch.optim as optim


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The 8 layers Shen et al. use for reconstruction: a subset of the 19 the decoder is
# trained on.
LAYERS = ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4', 'fc6', 'fc7', 'fc8']

DEFAULT_FEATURES_DIR = os.path.join(
    ROOT, 'feature-decoding', 'data', 'decoded_features', 'NSDImagery',
    'nsdcore40ses_ridge_alpha100_allunits', 'caffe', 'VGG19')

# The benchmark evaluates these 5 of the 9 task x stimulus-set combinations: the attention
# runs and the vision trials of conceptual stimuli are excluded (Kneeland et al., 2025).
CONDITIONS = ['visionsimple', 'visioncomplex',
              'imagerysimple', 'imagerycomplex', 'imageryconcepts']

# Delta degrees of freedom used for the SD of the true DNN features (feat_std0).
STD_DDOF = 1
CHANNEL_AXIS = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subject', required=True, help='e.g. subj01')
    p.add_argument('--roi', default='nsdgeneral')
    p.add_argument('--features-dir', default=DEFAULT_FEATURES_DIR)
    p.add_argument('--models-dir', default=os.path.join(ROOT, 'models'))
    p.add_argument('--output-dir', default=os.path.join(ROOT, 'recon_B'))
    p.add_argument('--n-iter', type=int, default=1000,
                   help='1000 for B, decided on the shared1000 bridge; '
                        'A stays at the 200 it was already measured with')
    p.add_argument('--seeds', default='0-9',
                   help='seeds of the generator latent code, as "0-9" or "0,3,7". The '
                        'benchmark samples 10 reconstructions per stimulus.')
    p.add_argument('--conditions', default=','.join(CONDITIONS),
                   help='comma-separated label prefixes to reconstruct; use this to split '
                        'the work across GPUs. Default: all 5 evaluated conditions.')
    p.add_argument('--no-normfeat', action='store_true',
                   help='skip normalize_feature on the decoded features. Measured to cost '
                        '-0.0126 +/- 0.0071 PixCorr, so NOT the configuration of record; '
                        'kept so the A/B is repeatable')
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def parse_seeds(spec):
    seeds = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def image_preprocess(img, image_mean=np.float32([104, 117, 123])):
    '''convert to Caffe's input image layout'''
    return np.float32(np.transpose(img, (2, 0, 1))[::-1]) - np.reshape(image_mean, (3, 1, 1))


def image_deprocess(img, image_mean=np.float32([104, 117, 123])):
    '''convert from Caffe's input image layout'''
    return np.dstack((img + np.reshape(image_mean, (3, 1, 1)))[::-1])


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    conditions = [c.strip() for c in args.conditions.split(',') if c.strip()]

    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise SystemExit('unknown condition(s): %s\nexpected a subset of: %s'
                         % (', '.join(unknown), ', '.join(CONDITIONS)))

    # Weight files, in the layout the figshare zips actually produce.
    vgg_dir = os.path.join(args.models_dir, 'VGG_ILSVRC_19_layers')
    gen_dir = os.path.join(args.models_dir,
                           'bvlc_reference_caffenet_generator_ILSVRC2012_Training')
    encoder_param_file = os.path.join(vgg_dir, 'VGG_ILSVRC_19_layers.pt')
    feature_std_file = os.path.join(
        vgg_dir,
        'estimated_cnn_feat_std_VGG_ILSVRC_19_layers_ImgSize_224x224_chwise_dof1.mat')
    generator_param_file = os.path.join(gen_dir, 'generator_relu7.pt')
    feature_range_file = os.path.join(gen_dir, 'relu7_act_range_3x.txt')

    for f in [encoder_param_file, feature_std_file, generator_param_file,
              feature_range_file]:
        if not os.path.exists(f):
            raise SystemExit('missing weight file: %s\n'
                             'Download them as described in the README (Data and checkpoints).' % f)

    # The image labels are the stems of the decoded .mat files. Reading them from the
    # first layer is enough: DecodedFeatures refuses to load a tree whose layers disagree.
    pattern = os.path.join(args.features_dir, LAYERS[0], args.subject, args.roi, '*.mat')
    all_labels = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(pattern))
    if not all_labels:
        raise SystemExit('no decoded features under %s\n'
                         'Has decode_features_chunked.py run for this subject?' % pattern)
    labels = [l for l in all_labels if l.split('-')[0] in conditions]
    if not labels:
        raise SystemExit('none of the %d labels matches conditions %s'
                         % (len(all_labels), ', '.join(conditions)))
    print('%s/%s: %d label(s) x %d seed(s) = %d reconstruction(s)'
          % (args.subject, args.roi, len(labels), len(seeds), len(labels) * len(seeds)))

    features = DecodedFeatures(args.features_dir, squeeze=False)

    # Feature SD estimated from the true DNN features of 10000 ImageNet images.
    feat_std0 = sio.loadmat(feature_std_file)

    # Upper bound of the generator's input features.
    upper_bound = np.loadtxt(feature_range_file, delimiter=' ',
                             usecols=np.arange(0, 4096), unpack=True).reshape((4096,))

    opts = {
        'loss_func': torch.nn.MSELoss(reduction='sum'),
        'n_iter': args.n_iter,
        'lr': (2., 1e-10),
        'momentum': (0.9, 0.9),
        'decay': (0.01, 0.01),
        'blurring': False,
        'channels': None,
        'masks': None,
        'disp_interval': 1,
        'feature_upper_bound': upper_bound,
        'feature_lower_bound': 0.,
    }

    layer_mapping = layer_map('vgg19')
    encoder = VGG19()
    encoder.to(args.device)
    encoder.load_state_dict(torch.load(encoder_param_file))
    encoder.eval()

    generator = AlexNetGenerator()
    generator.to(args.device)
    generator.load_state_dict(torch.load(generator_param_file))
    generator.eval()

    # The decoded features do not depend on the seed; they are read once and reused for
    # all 10 samples: the only thing a seed changes is the latent code the search starts
    # from. Only the 8 reconstruction layers of one subject are held, a few hundred MB.
    feat_by_label = {}
    for label in labels:
        feat = {layer: np.squeeze(features.get(layer=layer, subject=args.subject,
                                               roi=args.roi, label=label))
                for layer in LAYERS}

        # Normalisation of the decoded features:
        #     ft' = {(ft - mu) / mean(sd)} * mean(sd_base) + mu
        # Kept ON: see the module docstring.
        if not args.no_normfeat:
            for layer, ft in feat.items():
                ft0 = normalize_feature(
                    ft, channel_wise_mean=False, channel_wise_std=False,
                    channel_axis=CHANNEL_AXIS, shift='self',
                    scale=np.mean(feat_std0[layer]), std_ddof=STD_DDOF)
                feat[layer] = ft0[np.newaxis]
        else:
            feat = {layer: ft[np.newaxis] for layer, ft in feat.items()}
        feat_by_label[label] = feat

    n_total = len(labels) * len(seeds)
    n_done = 0
    for seed in seeds:
        save_dir = os.path.join(args.output_dir, args.subject, args.roi, 'seed%02d' % seed)
        os.makedirs(save_dir, exist_ok=True)
        dump_info(save_dir, script=__file__)

        # One latent code per seed, shared by every image of that seed, as upstream does.
        np.random.seed(seed)
        opts['initial_feature'] = np.random.normal(0, 1, (4096,))

        for label in labels:
            n_done += 1
            tiff_file = os.path.join(save_dir, 'recon_image_normalized-%s.tiff' % label)
            if os.path.exists(tiff_file):
                print('[%d/%d] seed %d %s: already done, skipping'
                      % (n_done, n_total, seed, label))
                continue
            print('[%d/%d] seed %d %s' % (n_done, n_total, seed, label))

            feat = feat_by_label[label]

            # Each layer is weighted by the inverse squared norm of its features, so that no
            # layer dominates the loss by sheer scale; weights sum to 1.
            feat_norm = np.array([np.linalg.norm(feat[layer]) for layer in LAYERS],
                                 dtype='float32')
            weights = 1. / (feat_norm ** 2)
            opts['layer_weights'] = dict(zip(LAYERS, weights / weights.sum()))

            recon_image, loss_list = reconstruct(
                feat, encoder,
                generator=generator,
                layer_mapping=layer_mapping,
                optimizer=optim.SGD,
                image_size=(224, 224, 3),
                crop_generator_output=True,
                preproc=image_preprocess,
                postproc=image_deprocess,
                output_dir=save_dir,
                save_snapshot=False,
                snapshot_dir=os.path.join(save_dir, 'snapshots', 'image-%s' % label),
                snapshot_ext='tiff',
                snapshot_postprocess=normalize_image,
                return_loss=True,
                device=args.device,
                **opts)

            # Clip the 0.02% most extreme pixels at each end before mapping to [0, 255],
            # otherwise a handful of outliers flattens the whole image.
            recon_image_norm = normalize_image(clip_extreme(recon_image, pct=4))
            sio.savemat(os.path.join(save_dir, 'recon_image-%s.mat' % label),
                        {'recon_image': recon_image, 'recon_image_norm': recon_image_norm})
            # BGR -> RGB: image_deprocess hands back Caffe channel order.
            PIL.Image.fromarray(recon_image_norm[:, :, [2, 1, 0]]).save(tiff_file)

    print('All done: %s' % os.path.join(args.output_dir, args.subject, args.roi))


if __name__ == '__main__':
    main()
