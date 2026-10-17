#!/usr/bin/env python
"""Stages 0 and 1 of iCNN in one pass: VGG19 features in RAM, ridge, decoded features on disk.

Inputs:  NSD index and betas (nsdgeneral) in the Deeprecon layout, NSD stimuli, VGG19 weights
Outputs: <out_dir>/caffe/VGG19/<layer>/subjNN/nsdgeneral/<stimulus>.mat and model/y_mean.mat per layer

Replaces three upstream scripts (feature extractor, train_decoder_sklearn_ridge.py, predict_feature.py),
which persist per-image features (906 GB) and a dense Ridge.coef_ (197 GB for conv1_2). Kneeland et al.,
Appendix A.6 point 2, fuse both steps. A.6 n.1 below = VGG19 feature normalisation off (read as no mean
subtraction); A.6 n.2 = sklearn Ridge in place of fastl2lir. The algebra is above hat_matrix().

Usage (from feature-decoding/, venv active; one subject per GPU, never DDP):
    CUDA_VISIBLE_DEVICES=0 python ../src/decode_features_chunked.py --subj 1 2
    python ../src/decode_features_chunked.py --subj 1 --layers fc6 fc7 fc8 --n_train_images 500 --verify fc8
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

# The 19 layers of the cookbook encoder config, in forward order. The order matters twice:
# the feature hooks fire in it, and the decoded features must land under the same names the
# reconstruction asks DecodedFeatures for.
LAYERS_ALL = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
              'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
              'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
              'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
              'fc6', 'fc7', 'fc8']

# The 8 the reconstruction loss actually uses in the local Takagi copy.
LAYERS_RECON = ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4', 'fc6', 'fc7', 'fc8']

DATA_KEY = 'VoxelData'
ROI_KEY = 'ROI_nsdgeneral'
LABEL_KEY = 'stimulus_name'
FMRI_NAME = 'nsdimagery_func1pt8mm_betas_fithrf_GLMdenoise_RR'

NUM_VOXELS = {1: 15724, 2: 14278, 3: 15226, 4: 13153,
              5: 13039, 6: 17907, 7: 12682, 8: 14386}


# --- CLI ------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subj', type=int, nargs='+', default=[1, 2, 5, 7],
                   choices=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument('--index_dir', default='./data/fmri/NSDcore',
                   help='where prepare_nsdcore_icnn.py wrote subj0N_nsdcore_index.npz')
    p.add_argument('--imagery_h5', default='./data/fmri/NSDImagery',
                   help='where prepare_nsd_imagery_icnn.py wrote the BData .h5')
    p.add_argument('--nsd_stimuli',
                   default='./data/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
    p.add_argument('--models_dir', default='../models',
                   help='holds VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt')
    p.add_argument('--out_dir', default='./data/decoded_features/NSDImagery')
    p.add_argument('--decoder_dir', default='./data/feature_decoders/NSDcore')
    p.add_argument('--analysis_name', default='nsdcore40ses_ridge_alpha100_allunits')
    p.add_argument('--cache_dir', default='./data/cache/nsd224',
                   help='uint8 224x224 resize cache, 1.35 GB per subject; the resize is '
                        'deterministic so this only trades disk for 39.5 GB of re-reads')

    p.add_argument('--test', choices=['imagery', 'shared1000'], default='imagery')
    p.add_argument('--n_test', type=int, default=None,
                   help='shared1000 only: use the first N test images (the bridge '
                        'does not need all 1000)')

    layers = p.add_mutually_exclusive_group()
    layers.add_argument('--layers', nargs='+', default=None, choices=LAYERS_ALL)
    layers.add_argument('--layers_recon', action='store_true',
                        help='shorthand for the 8 layers of the reconstruction loss')

    p.add_argument('--alpha', type=float, default=100.0)
    p.add_argument('--batch', type=int, default=32, help='images per VGG19 forward')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--accum_gpu_gb', type=float, default=20.0,
                   help='keep the accumulators on the GPU while they fit in this budget, '
                        'otherwise fall back to host RAM (slower, unbounded)')

    p.add_argument('--rgb_mean', action='store_true',
                   help='subtract the Caffe ILSVRC mean from the input images, i.e. do NOT '
                        'apply deviation A.6 n.1')
    p.add_argument('--bgr', action='store_true',
                   help='feed the encoder BGR instead of the RGB bdpy uses, matching stage '
                        "2's image_preprocess")

    p.add_argument('--n_train_images', type=int, default=None,
                   help='SMOKE TEST ONLY: fit on the first N unique training images. '
                        'Any value here makes the decoder incomparable to the benchmark')
    p.add_argument('--verify', metavar='LAYER', default=None,
                   help='refit LAYER with a real sklearn Ridge (chunked, as upstream) and '
                        'report the max relative difference against the closed form. Only '
                        'sensible on fc6/fc7/fc8: bigger layers do not fit in memory')
    p.add_argument('--overwrite', action='store_true',
                   help='recompute layers whose decoded features are already complete')
    p.add_argument('--selftest', action='store_true',
                   help='check the closed form against sklearn Ridge on random data and '
                        'exit. Needs no data: run it once after any edit to the algebra')
    return p.parse_args()


# --- fMRI -----------------------------------------------------------------------------

def load_index(index_dir: Path, subject: int) -> dict:
    f = index_dir / f'subj{subject:02d}_nsdcore_index.npz'
    if not f.exists():
        sys.exit(f'missing {f}\nrun prepare_nsdcore_icnn.py first')
    z = np.load(f, allow_pickle=False)
    idx = {k: z[k] for k in z.files}
    idx['betas_file'] = str(idx['betas_file'])
    return idx


def load_betas(betas_file: Path, rows: np.ndarray) -> np.ndarray:
    """Rows of betas_all_subj0N_fp32_renorm.hdf5, in the order given.

    h5py fancy-indexes only along sorted unique indices, and the file is 1.9 GB; it is
    cheaper to read it whole and index in memory than to fight the selection API.
    """
    with h5py.File(betas_file, 'r') as h:
        betas = np.asarray(h['betas'])
    return np.ascontiguousarray(betas[rows], dtype=np.float32)


def load_test_imagery(imagery_dir: Path, subject: int):
    """The 384 NSD-Imagery trials, averaged per label -> (30, V) and the 30 label strings.

    This is predict_feature.py:119-123 with average_sample: true, reproduced rather than
    imported so that the label order is visibly the np.unique() one that the zero-padded
    label convention of prepare_nsd_imagery_icnn.py depends on.
    """
    import bdpy

    f = imagery_dir / f'subj{subject:02d}_{FMRI_NAME}.h5'
    if not f.exists():
        sys.exit(f'missing {f}\nrun prepare_nsd_imagery_icnn.py first')
    bdata = bdpy.BData(str(f))
    x = bdata.select(f'{ROI_KEY} = 1')
    try:
        labels = [str(lb) for lb in bdata.get_label(LABEL_KEY)]
    except Exception:                                  # noqa: BLE001 - upstream's own fallback
        labels = [str(lb) for lb in bdata.select(LABEL_KEY).flatten()]

    uniq = list(np.unique(labels))
    labels = np.asarray(labels)
    x_test = np.vstack([x[labels == lb].mean(axis=0) for lb in uniq]).astype(np.float32)
    assert x_test.shape[0] == len(uniq) == 30, f'{x_test.shape[0]} labels, expected 30'
    return x_test, uniq


def load_test_shared1000(idx: dict, betas_file: Path, n_test):
    """The shared1000 half of NSD-core, averaged over its 3 repetitions -> (n, V).

    Only used for the A/B bridge: reconstructing images whose ground
    truth is an NSD photograph is the one check that validates decoder and stage 2 together
    before any imagery number is produced.
    """
    img73k = idx['test_img73k']
    if n_test is not None:
        img73k = img73k[:n_test]
    keep = idx['test_uidx'] < len(img73k)
    rows, uidx = idx['test_rows'][keep], idx['test_uidx'][keep]

    x = load_betas(betas_file, rows)
    x_test = np.vstack([x[uidx == u].mean(axis=0) for u in range(len(img73k))]).astype(np.float32)
    labels = [f'shared1000-{int(i):05d}' for i in img73k]
    return x_test, labels, img73k


def standardize(x_train: np.ndarray, x_test: np.ndarray):
    """x_mean / x_norm of train_decoder_sklearn_ridge.py:145-146, applied to both sides.

    bdpy zeroes only the infinities that a zero-variance voxel produces (ModelTraining.run);
    0/0 gives a NaN instead, which sklearn would reject; those are zeroed too. Both cases
    mean the same thing; a voxel that is constant across the training set carries no
    information: and both are counted out loud.
    """
    x_mean = x_train.mean(axis=0)[np.newaxis, :]
    x_norm = x_train.std(axis=0, ddof=1)[np.newaxis, :]

    n_dead = int((x_norm == 0).sum())
    if n_dead:
        print(f'  {n_dead} voxel(s) with zero variance across the training set -> zeroed')

    def apply(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            xn = (x - x_mean) / x_norm
        xn[~np.isfinite(xn)] = 0.0
        return np.ascontiguousarray(xn, dtype=np.float32)

    return apply(x_train), apply(x_test), x_mean, x_norm


# Method. The estimator is the one of train_decoder_sklearn_ridge.py + predict_feature.py:
#   Xn = (X - x_mean) / x_norm,  Yn = (Y - y_mean) / y_norm   (mean and std(ddof=1); bdpy ModelTraining)
#   Yn is expanded from 9000 images to 27000 trials; sklearn Ridge(alpha=100) is fitted on (Xn, Yn);
#   pred = model.predict(Xn_test) * y_norm + y_mean            (predict_feature.py:161)
# Only the evaluation differs. In closed form, with Xc = Xn centred:
#   W = (Xc^T Xc + alpha I)^-1 Xc^T Yn,   pred = Xc_test W = H Yn,   H = Xc_test (Xc^T Xc + alpha I)^-1 Xc^T
# H is (n_test x 27000) and depends on the subject, not on the layer. Undoing the two normalisations of Y
# cancels y_norm and leaves  pred = H_agg @ Y_unique + (1 - rowsum(H)) * y_mean,  where H_agg sums the 27000
# columns of H into the 9000 unique images. The features enter through one matrix product that can be
# accumulated over batches of images. Consequences:
#   - no coef_ is ever built (no 116 GB peak for conv1_2); the accumulators are (n_test + 2) x n_unit,
#     about 1.9 GB for all 19 layers, on the GPU;
#   - all 19 layers come out of one forward pass over the 9000 images.
# A literal chunked sklearn Ridge re-solves a 15724 x 15724 system for each of ~5500 channels per subject
# (tens of hours per subject, against minutes here). It is the same estimator, not an approximation:
# --verify compares it with a real sklearn Ridge on a small layer. The fitted decoder itself is lost; stage 2
# only needs model/y_mean.mat (the feature scaling), which is written.
def hat_matrix(xn_train: np.ndarray, xn_test: np.ndarray, uidx: np.ndarray, n_unique: int,
               alpha: float, device: str):
    """H_agg (n_test x n_unique) and rowsum(H), the whole decoder in two small arrays.

    H = Xc_test (Xc^T Xc + alpha I)^-1 Xc^T is what ridge applies to the targets; folding
    the 3 repetitions of an image into one column (index_add_ below) is exactly the Y_sort
    expansion of bdpy's ModelTraining, run backwards.

    float64 throughout: the Gram matrix is 15724^2 and the Cholesky of it is the only step
    where precision is worth paying for. TF32 is disabled globally in main(), with it on,
    this product loses 3 significant digits and the check in --verify fails for the wrong
    reason.
    """
    t0 = time.time()
    xt = torch.from_numpy(xn_train).to(device=device, dtype=torch.float64)
    xs = torch.from_numpy(xn_test).to(device=device, dtype=torch.float64)

    # Centring: x_mean was computed on these very rows; the means are already 0 to
    # rounding. Subtracting them anyway makes this match sklearn's fit_intercept=True
    # exactly instead of approximately.
    xbar = xt.mean(dim=0, keepdim=True)
    xt -= xbar
    xs -= xbar

    n, p = xt.shape
    gram = xt.T @ xt
    gram.diagonal().add_(alpha)
    chol = torch.linalg.cholesky(gram)
    del gram
    # (Xc^T Xc + aI)^-1 Xc_test^T, then H^T = Xc @ that
    b = torch.cholesky_solve(xs.T.contiguous(), chol)          # (p, n_test)
    del chol
    h = (xt @ b).T.contiguous()                                # (n_test, n)
    del b, xt, xs

    h_agg = torch.zeros((h.shape[0], n_unique), dtype=torch.float64, device=device)
    h_agg.index_add_(1, torch.from_numpy(uidx.astype(np.int64)).to(device), h)
    rowsum = h.sum(dim=1)
    del h

    print(f'  hat matrix: {n} train trials x {p} voxels -> H_agg {tuple(h_agg.shape)}, '
          f'{time.time() - t0:.1f}s')
    return h_agg.to(torch.float32), rowsum.to(torch.float64)


# --- images ---------------------------------------------------------------------------

def build_image_cache(nsd_stimuli: Path, img73k: np.ndarray, cache_file: Path) -> np.ndarray:
    """(n, 224, 224, 3) uint8, the bdpy bicubic resize of the 425x425 NSD stimuli.

    Cached because the resize is deterministic and reading 9000 rows out of a 39.5 GB HDF5
    is the slowest part of a re-run. nsd_stimuli.hdf5 is used rather than MindEye2's
    coco_images_224_float16.hdf5 because the latter was resized by an unknown filter, and
    the filter is exactly what has to match bdpy here.
    """
    if cache_file.exists():
        cache = np.load(cache_file, mmap_mode='r')
        if cache.shape == (len(img73k), 224, 224, 3):
            print(f'  image cache: {cache_file} (reused)')
            return cache
        print(f'  image cache: {cache_file} has shape {cache.shape}, rebuilding')

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    out = np.empty((len(img73k), 224, 224, 3), dtype=np.uint8)
    t0 = time.time()
    with h5py.File(nsd_stimuli, 'r') as h:
        brick = h['imgBrick']
        for i, k in enumerate(img73k):
            img = brick[int(k)]                                # (425, 425, 3) uint8
            out[i] = np.array(Image.fromarray(img).resize((224, 224), resample=2))
            if (i + 1) % 1000 == 0:
                print(f'    {i + 1}/{len(img73k)} images, {time.time() - t0:.0f}s')
    np.save(cache_file, out)
    print(f'  image cache: wrote {cache_file} in {time.time() - t0:.0f}s')
    return out


def preprocess_batch(imgs: np.ndarray, rgb_mean, bgr: bool, device: str) -> torch.Tensor:
    """uint8 (b,224,224,3) -> float32 (b,3,224,224) in bdpy's input space.

    (data/255)*scale with scale=255 is the identity on the pixel values; it is kept in that
    form because it is the line bdpy runs, and because scale is the knob that would change
    if the encoder were ever fed [0,1] images instead.

    Deviation A.6 n.1. Transcribed from bdpy ImageDataset.__load_image: PIL bicubic resize to 224
    (no antialias), HWC->CHW, then per-channel mean subtraction only if rgb_mean is given. Channels
    stay RGB: bdpy never flips to BGR, although stage 2's image_preprocess does (recon_icnn_A.py).
    --rgb_mean and --bgr run the two variants the paper leaves open; the defaults read A.6 n.1 as no
    mean subtraction, RGB order.
    """
    x = torch.from_numpy(np.ascontiguousarray(imgs)).to(device).permute(0, 3, 1, 2).float()
    x = (x / 255.0) * 255.0
    if bgr:
        x = x.flip(1)
    if rgb_mean is not None:
        x = x - torch.as_tensor(rgb_mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    return x


def load_encoder(models_dir: Path, layers, device: str):
    """VGG19 with a forward hook on each requested layer, features kept on the GPU.

    bdpy's own FeatureExtractor would do this but returns numpy on the host, which would
    move 116 GB across the PCIe bus for conv1_2 alone. The modules hooked are the ones
    layer_map('vgg19') names: the Conv2d/Linear outputs, i.e. pre-ReLU.
    """
    from bdpy.dl.torch.models import VGG19, layer_map

    param_file = models_dir / 'VGG_ILSVRC_19_layers' / 'VGG_ILSVRC_19_layers.pt'
    if not param_file.exists():
        sys.exit(f'missing {param_file}\nsee the README (Data and checkpoints) for the figshare download')

    encoder = VGG19()
    encoder.load_state_dict(torch.load(param_file, map_location='cpu'))
    encoder.to(device).eval()

    mapping = layer_map('vgg19')
    store = {}

    def hook_for(name):
        def hook(module, module_in, module_out):
            store[name] = module_out
        return hook

    for name in layers:
        module = eval(f'encoder.{mapping[name]}')              # 'features[2]' etc.
        module.register_forward_hook(hook_for(name))
    return encoder, store


# --- output ---------------------------------------------------------------------------

def decoded_dir(root: Path, layer: str, subject: int, roi: str) -> Path:
    return root / 'caffe' / 'VGG19' / layer / f'subj{subject:02d}' / roi


def model_dir(root: Path, layer: str, subject: int, roi: str) -> Path:
    return root / 'caffe' / 'VGG19' / layer / f'subj{subject:02d}' / roi / 'model'


def layer_complete(out_root: Path, layer: str, subject: int, roi: str, labels) -> bool:
    """A layer counts as done only if every single label is on disk.

    predict_feature.py skips a layer whose output directory merely exists, which turns an
    interrupted run into a silently truncated one.
    """
    d = decoded_dir(out_root, layer, subject, roi)
    return d.is_dir() and all((d / f'{lb}.mat').exists() for lb in labels)


def save_decoded(out_root: Path, layer: str, subject: int, roi: str, labels, pred: np.ndarray):
    from bdpy.dataform import save_array

    d = decoded_dir(out_root, layer, subject, roi)
    d.mkdir(parents=True, exist_ok=True)
    for i, label in enumerate(labels):
        # 1 x <feature shape>, the shape DecodedFeatures expects (predict_feature.py:170-178)
        save_array(str(d / f'{label}.mat'), np.array([pred[i]]), key='feat',
                   dtype=np.float32, sparse=False)


def save_model_stats(dec_root: Path, layer: str, subject: int, roi: str,
                     x_mean, x_norm, y_mean, y_norm, info: dict):
    """model/{x,y}_{mean,norm}.mat + info.yaml, in the layout predict_feature.py reads.

    The ridge weights are deliberately absent (see the header), but y_mean.mat is not
    optional: recon_icnn_image_gd.py:213-214 loads it for the feature_std_train_mean_center
    scaling; deleting this directory breaks stage 2, not just a re-prediction.
    """
    import yaml
    from bdpy.dataform import save_array

    d = model_dir(dec_root, layer, subject, roi)
    d.mkdir(parents=True, exist_ok=True)
    for key, value in (('x_mean', x_mean), ('x_norm', x_norm),
                       ('y_mean', y_mean), ('y_norm', y_norm)):
        save_array(str(d / f'{key}.mat'), value, key=key, dtype=np.float32, sparse=False)
    with open(d / 'info.yaml', 'w') as f:
        yaml.dump({'_status': {'computation_status': 'done'}, **info},
                  f, default_flow_style=False)


# --- verification ---------------------------------------------------------------------

def verify_against_sklearn(layer, xn_train, xn_test, feats_unique, uidx, y_mean, y_norm,
                           alpha, pred_closed):
    """Refit the layer the upstream way and compare, because the algebra deserves evidence.

    Reproduces train_decoder_sklearn_ridge.py + predict_feature.py line by line on one layer:
    normalise Y with the same statistics, expand to trials, one Ridge per chunk along the
    channel axis, predict, denormalise. Only usable where the features fit in memory twice.
    """
    from sklearn.linear_model import Ridge

    print(f'  --verify {layer}: refitting with sklearn Ridge(alpha={alpha:g})')
    t0 = time.time()
    shape = feats_unique.shape[1:]
    y = feats_unique.reshape(len(feats_unique), -1).astype(np.float32)
    m = y_mean.reshape(1, -1).astype(np.float32)
    s = y_norm.reshape(1, -1).astype(np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        yn = (y - m) / s
    yn[~np.isfinite(yn)] = 0.0
    yn = yn[uidx]                                              # Y_sort: 9000 -> 27000 rows

    model = Ridge(alpha=alpha)
    model.fit(xn_train, yn)
    pred = model.predict(xn_test) * s + m
    pred = pred.reshape((len(xn_test),) + shape)

    scale = np.abs(pred).max()
    diff = np.abs(pred - pred_closed).max()
    print(f'    sklearn fit {time.time() - t0:.1f}s | max|diff| {diff:.3e} on features of '
          f'magnitude {scale:.3e} -> relative {diff / max(scale, 1e-12):.2e}')
    if diff > 1e-3 * max(scale, 1e-12):
        sys.exit('    the closed form and sklearn disagree by more than 1e-3 relative. '
                 'Do not use these decoded features.')
    print('    MATCH')


# --- main loop ------------------------------------------------------------------------

def run_subject(subject: int, args, layers, rgb_mean):
    print(f'\n=== subj{subject:02d} ===')
    roi = 'nsdgeneral'
    out_root = Path(args.out_dir) / args.analysis_name
    dec_root = Path(args.decoder_dir) / args.analysis_name

    idx = load_index(Path(args.index_dir), subject)
    betas_file = Path(idx['betas_file'])
    train_rows, train_uidx = idx['train_rows'], idx['train_uidx']
    train_img = idx['train_img73k']

    if args.n_train_images is not None:
        keep = train_uidx < args.n_train_images
        train_rows, train_uidx = train_rows[keep], train_uidx[keep]
        train_img = train_img[:args.n_train_images]
        print(f'  SMOKE TEST: {len(train_img)} unique training images '
              f'({len(train_rows)} trials) -- not a benchmark decoder')

    if args.test == 'imagery':
        x_test, labels = load_test_imagery(Path(args.imagery_h5), subject)
    else:
        x_test, labels, _ = load_test_shared1000(idx, betas_file, args.n_test)
    print(f'  test set: {args.test}, {x_test.shape[0]} labels x {x_test.shape[1]} voxels')

    todo = [lyr for lyr in layers
            if args.overwrite or not layer_complete(out_root, lyr, subject, roi, labels)]
    if not todo:
        print('  every requested layer is already complete, nothing to do')
        return
    if len(todo) != len(layers):
        print(f'  {len(layers) - len(todo)} layer(s) already complete, {len(todo)} to go')

    x_train = load_betas(betas_file, train_rows)
    assert x_train.shape[1] == NUM_VOXELS[subject], \
        f'{x_train.shape[1]} voxels, the nsdgeneral mask has {NUM_VOXELS[subject]}'
    assert x_test.shape[1] == x_train.shape[1], \
        f'test has {x_test.shape[1]} voxels, training {x_train.shape[1]}: different masks'
    print(f'  training set: {x_train.shape[0]} trials x {x_train.shape[1]} voxels '
          f'over {len(train_img)} unique images')

    xn_train, xn_test, x_mean, x_norm = standardize(x_train, x_test)
    del x_train
    h_agg, rowsum = hat_matrix(xn_train, xn_test, train_uidx, len(train_img),
                               args.alpha, args.device)

    cache = build_image_cache(Path(args.nsd_stimuli), train_img,
                              Path(args.cache_dir) / f'subj{subject:02d}_train224.npy')
    encoder, store = load_encoder(Path(args.models_dir), todo, args.device)

    # One forward pass, all layers at once: the per-layer work is an accumulation now;
    # there is no reason to pay for the pass 19 times.
    n_test, n_unique = h_agg.shape
    shapes, accum_dev = {}, args.device
    with torch.no_grad():
        probe = preprocess_batch(cache[:1], rgb_mean, args.bgr, args.device)
        encoder(probe)
        for lyr in todo:
            shapes[lyr] = tuple(store[lyr].shape[1:])
    n_unit_total = sum(int(np.prod(s)) for s in shapes.values())
    gb = n_unit_total * (n_test * 4 + 16) / 1024 ** 3
    if gb > args.accum_gpu_gb:
        accum_dev = 'cpu'
        print(f'  accumulators would take {gb:.1f} GB > --accum_gpu_gb: keeping them in RAM')
    print(f'  {len(todo)} layer(s), {n_unit_total} units, accumulators {gb:.1f} GB on {accum_dev}')

    pred_acc = {l: torch.zeros((n_test, int(np.prod(s))), dtype=torch.float32, device=accum_dev)
                for l, s in shapes.items()}
    sum1 = {l: torch.zeros(int(np.prod(s)), dtype=torch.float64, device=accum_dev)
            for l, s in shapes.items()}
    sum2 = {l: torch.zeros(int(np.prod(s)), dtype=torch.float64, device=accum_dev)
            for l, s in shapes.items()}
    h_dev = h_agg.to(accum_dev)
    keep_feats = args.verify if args.verify in todo else None
    kept = [] if keep_feats else None

    t0 = time.time()
    with torch.no_grad():
        for i0 in range(0, n_unique, args.batch):
            i1 = min(i0 + args.batch, n_unique)
            encoder(preprocess_batch(cache[i0:i1], rgb_mean, args.bgr, args.device))
            for lyr in todo:
                y = store[lyr].reshape(i1 - i0, -1)
                if accum_dev != args.device:
                    y = y.cpu()
                pred_acc[lyr] += h_dev[:, i0:i1] @ y
                sum1[lyr] += y.sum(dim=0, dtype=torch.float64)
                sum2[lyr] += (y * y).sum(dim=0, dtype=torch.float64)
                if keep_feats == lyr:
                    kept.append(y.cpu().numpy().copy())
            if i1 % (args.batch * 25) == 0 or i1 == n_unique:
                done = i1 / n_unique
                el = time.time() - t0
                print(f'    {i1}/{n_unique} images, {el:.0f}s elapsed, '
                      f'{el / done - el:.0f}s left', flush=True)

    # y_mean / y_norm over the unique images, ddof=1: the statistics
    # train_decoder_sklearn_ridge.py:148-149 computes on the same array.
    for lyr in todo:
        n = float(n_unique)
        y_mean = (sum1[lyr] / n)
        y_var = (sum2[lyr] - n * y_mean ** 2) / (n - 1.0)
        y_norm = torch.sqrt(torch.clamp(y_var, min=0.0))
        # pred = H_agg Y + (1 - rowsum(H)) y_mean , see the header for the derivation
        pred = (pred_acc[lyr].double() +
                (1.0 - rowsum.to(accum_dev)).unsqueeze(1) * y_mean.unsqueeze(0))
        pred = pred.to(torch.float32).cpu().numpy().reshape((n_test,) + shapes[lyr])
        y_mean_np = y_mean.cpu().numpy().reshape((1,) + shapes[lyr])
        y_norm_np = y_norm.cpu().numpy().reshape((1,) + shapes[lyr])

        if keep_feats == lyr:
            verify_against_sklearn(lyr, xn_train, xn_test,
                                   np.concatenate(kept).reshape((n_unique,) + shapes[lyr]),
                                   train_uidx, y_mean_np, y_norm_np, args.alpha, pred)

        save_decoded(out_root, lyr, subject, roi, labels, pred)
        save_model_stats(dec_root, lyr, subject, roi, x_mean, x_norm, y_mean_np, y_norm_np,
                         {'alpha': float(args.alpha),
                          'n_train_trials': int(len(train_rows)),
                          'n_train_images': int(n_unique),
                          'rgb_mean': bool(args.rgb_mean),
                          'channel_order': 'bgr' if args.bgr else 'rgb'})
        n_dead = int((y_norm == 0).sum())
        print(f'  {lyr:8s} {str(shapes[lyr]):18s} pred |max| {np.abs(pred).max():.3e} '
              f'| {n_dead} constant unit(s) | saved')
        del pred_acc[lyr], sum1[lyr], sum2[lyr]

    print(f'  subj{subject:02d} done in {time.time() - t0:.0f}s')


def selftest(device: str, alpha: float):
    """Run the closed form and sklearn Ridge on random data of the right shape.

    Same functions as the real run, standardize(), hat_matrix() and the accumulation
    formula: on 300 trials of 40 voxels over 100 images with 3 repetitions each; a
    shape or sign error surfaces in seconds instead of after a forward pass over 9000
    images. Needs no data and no GPU.
    """
    rng = np.random.default_rng(0)
    n_img, n_rep, n_vox, n_test, n_unit = 100, 3, 40, 7, 25

    uidx = np.repeat(np.arange(n_img), n_rep).astype(np.int32)
    feats = rng.normal(size=(n_img, n_unit)).astype(np.float32) * 5 + 2
    feats[:, 0] = 3.0                       # a constant unit: y_norm 0, the 0/0 case
    w = rng.normal(size=(n_vox, n_unit)).astype(np.float32)
    x_train = (feats[uidx] @ w.T + rng.normal(size=(n_img * n_rep, n_vox)) * 0.5).astype(np.float32)
    x_test = rng.normal(size=(n_test, n_vox)).astype(np.float32)

    xn_train, xn_test, _, _ = standardize(x_train, x_test)
    h_agg, rowsum = hat_matrix(xn_train, xn_test, uidx, n_img, alpha, device)

    y = torch.from_numpy(feats).to(h_agg.device)
    y_mean = y.double().mean(dim=0)
    y_norm = y.double().std(dim=0, unbiased=True)
    pred = (h_agg @ y).double() + (1.0 - rowsum).unsqueeze(1) * y_mean.unsqueeze(0)
    pred = pred.to(torch.float32).cpu().numpy()

    verify_against_sklearn('selftest', xn_train, xn_test, feats, uidx,
                           y_mean.cpu().numpy()[np.newaxis, :],
                           y_norm.cpu().numpy()[np.newaxis, :], alpha, pred)


def main():
    args = parse_args()

    # TF32 turns the 15724x15724 Cholesky and the feature products into 10-bit-mantissa
    # arithmetic. It is off by default for matmul in torch 2.0 but on for cudnn, and the
    # convolutions are what produce the features.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if args.selftest:
        selftest('cpu' if not torch.cuda.is_available() else args.device, args.alpha)
        return

    if args.layers_recon:
        layers = LAYERS_RECON
    elif args.layers:
        layers = [l for l in LAYERS_ALL if l in args.layers]    # forward order, always
    else:
        layers = LAYERS_ALL

    rgb_mean = None
    if args.rgb_mean:
        mean_file = Path(args.models_dir) / 'VGG_ILSVRC_19_layers' / 'ilsvrc_2012_mean.npy'
        if not mean_file.exists():
            sys.exit(f'missing {mean_file} (needed by --rgb_mean)')
        m = np.load(mean_file)
        rgb_mean = np.float32([m[0].mean(), m[1].mean(), m[2].mean()])
        print(f'--rgb_mean: subtracting {rgb_mean} -- deviation A.6 n.1 is NOT applied')
    else:
        print('deviation A.6 n.1 applied: no image mean subtraction')
    print(f'channel order: {"BGR" if args.bgr else "RGB (bdpy)"}')
    print(f'layers: {" ".join(layers)}')

    if args.verify and args.verify not in layers:
        sys.exit(f'--verify {args.verify} is not among the layers being computed')

    # The decoded features of the two test sets share a directory layout and differ only in
    # their label names; a shared1000 run dropped into the imagery tree would leave both
    # sets side by side and DecodedFeatures would happily reconstruct the union.
    if args.test == 'shared1000' and 'NSDImagery' in args.out_dir:
        sys.exit(f'--test shared1000 with --out_dir {args.out_dir}: pass a separate '
                 '--out_dir (e.g. ./data/decoded_features/NSDcoreShared1000) so the bridge '
                 'does not land among the imagery features')
    print(f'decoded features -> {Path(args.out_dir) / args.analysis_name}/caffe/VGG19')
    print(f'decoder stats    -> {Path(args.decoder_dir) / args.analysis_name}/caffe/VGG19')

    for subject in args.subj:
        run_subject(subject, args, layers, rgb_mean)

    print('\nall done')


if __name__ == '__main__':
    main()
