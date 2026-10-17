#!/usr/bin/env python
'''iCNN reconstruction (stage 2) for the shared1000 benchmark: the 50 Deeprecon test images.

Inputs:  decoded features (<stimulus_name>.mat per image) from feature-decoding, VGG19 and generator weights
Outputs: <output-dir>/<subject>/<roi>/recon_image_normalized-<label>.tiff, one per image (skipped if present)

Adapted from takagi/codes/gan/recon_icnn_image_vgg19_dgn_relu7gen_gd.py. The reconstruction is untouched
(optimizer, lr, momentum, decay, loss, layer weights, feature normalisation); only names and paths change:
the label is the file stem (Takagi's script truncates it to 6 characters), DecodedFeatures.get() takes
label=, and the figshare zips unpack to models/<name>/. A.6 n.1 (no mean subtraction) lives in
decode_features_chunked.py; the Kamitani normalisation of the decoded features stays on, and --no-normfeat
turns it off, the other reading of the appendix.

Usage (from icnn/, venv active):
    CUDA_VISIBLE_DEVICES=0 python recon_icnn_A.py --subject sub-01
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


ROOT = os.path.dirname(os.path.abspath(__file__))

# The 8 layers Shen et al. use for reconstruction: a subset of the 19 the decoder is
# trained on.
LAYERS = ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4', 'fc6', 'fc7', 'fc8']

DEFAULT_FEATURES_DIR = os.path.join(
    ROOT, 'feature-decoding', 'data', 'decoded_features', 'ImageNetTest',
    'train_deeprecon_rep5_test_ImageNetTest_fmriprep_pyfastl2lir_alpha100_allunits',
    'caffe', 'VGG19')

# Delta degrees of freedom used for the SD of the true DNN features (feat_std0).
STD_DDOF = 1
CHANNEL_AXIS = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subject', required=True, help='e.g. sub-01')
    p.add_argument('--roi', default='VC')
    p.add_argument('--features-dir', default=DEFAULT_FEATURES_DIR)
    p.add_argument('--models-dir', default=os.path.join(ROOT, 'models'))
    p.add_argument('--output-dir', default=os.path.join(ROOT, 'recon_A'))
    p.add_argument('--n-iter', type=int, default=200,
                   help='200 in Shen et al. 2019 (no-DGN variant) and in the Takagi copy; '
                        'the cookbook config says 1000')
    p.add_argument('--seed', type=int, default=0,
                   help='seed of the generator latent code; a different seed is a '
                        'different sample of the same stimulus')
    p.add_argument('--no-normfeat', action='store_true',
                   help='skip normalize_feature on the decoded features: the second '
                        'reading of deviation A.6 n.1')
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def image_preprocess(img, image_mean=np.float32([104, 117, 123])):
    '''convert to Caffe's input image layout'''
    return np.float32(np.transpose(img, (2, 0, 1))[::-1]) - np.reshape(image_mean, (3, 1, 1))


def image_deprocess(img, image_mean=np.float32([104, 117, 123])):
    '''convert from Caffe's input image layout'''
    return np.dstack((img + np.reshape(image_mean, (3, 1, 1)))[::-1])


def main():
    args = parse_args()

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
    labels = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(pattern))
    if not labels:
        raise SystemExit('no decoded features under %s\n'
                         'Has predict_feature_fastl2lir.py run for this subject?' % pattern)
    print('%d image(s) for %s/%s' % (len(labels), args.subject, args.roi))

    save_dir = os.path.join(args.output_dir, args.subject, args.roi)
    os.makedirs(save_dir, exist_ok=True)
    dump_info(save_dir, script=__file__)

    features = DecodedFeatures(args.features_dir, squeeze=False)

    # Feature SD estimated from the true DNN features of 10000 ImageNet images.
    feat_std0 = sio.loadmat(feature_std_file)

    # Upper bound of the generator's input features.
    upper_bound = np.loadtxt(feature_range_file, delimiter=' ',
                             usecols=np.arange(0, 4096), unpack=True).reshape((4096,))

    # One latent code for the whole run, as upstream. With a generator this is what the
    # optimization starts from: the initial image the upstream script builds from
    # ilsvrc_2012_mean.npy is dead code there, never passed to reconstruct().
    np.random.seed(args.seed)
    initial_gen_feat = np.random.normal(0, 1, (4096,))

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
        'initial_feature': initial_gen_feat,
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

    for i, label in enumerate(labels):
        tiff_file = os.path.join(save_dir, 'recon_image_normalized-%s.tiff' % label)
        if os.path.exists(tiff_file):
            print('[%d/%d] %s: already done, skipping' % (i + 1, len(labels), label))
            continue
        print('[%d/%d] %s' % (i + 1, len(labels), label))

        feat = {layer: np.squeeze(features.get(layer=layer, subject=args.subject,
                                               roi=args.roi, label=label))
                for layer in LAYERS}

        # Normalisation of the decoded features:
        #     ft' = {(ft - mu) / mean(sd)} * mean(sd_base) + mu
        # This is the operation deviation A.6 n.1 may or may not refer to;
        # --no-normfeat produces the other reading.
        if not args.no_normfeat:
            for layer, ft in feat.items():
                ft0 = normalize_feature(
                    ft, channel_wise_mean=False, channel_wise_std=False,
                    channel_axis=CHANNEL_AXIS, shift='self',
                    scale=np.mean(feat_std0[layer]), std_ddof=STD_DDOF)
                feat[layer] = ft0[np.newaxis]
        else:
            feat = {layer: ft[np.newaxis] for layer, ft in feat.items()}

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

    print('All done: %s' % save_dir)


if __name__ == '__main__':
    main()
