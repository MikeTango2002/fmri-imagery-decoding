#!/usr/bin/env python
"""Stage 1 as Shirakawa's settings define it: fastl2lir, a per-unit top-500-voxel ridge on the nsdgeneral ROI.

Inputs:  as decode_features_chunked.py (NSD betas, stimuli, VGG19 weights)
Outputs: <out_dir>/caffe/VGG19/<layer>/subjNN/nsdgeneral/<stimulus>.mat, under --analysis_name

Stage 1 of this reproduction is a plain multi-output Ridge on all ~15724 voxels (decode_features_chunked.py; Appendix A.6 n.2).
Shirakawa's settings (train_decoder_fastl2lir.py and the nsd-37ses fastl2lir config of
KamitaniLab/spurious_reconstruction, named only as the source of the settings; no code is taken) call
FastL2LiR(alpha=100, n_feat=500): each unit picks its own top-500 voxels by |Pearson r|. This runs that
variant at NSD scale for all 19 layers; the algebra is in the comment above compute_gram().

Usage (from feature-decoding/, venv active):
    python ../src/decode_features_fastl2lir.py --selftest
    python ../src/decode_features_fastl2lir.py --subj 1 --analysis_name nsdcore40ses_fastl2lir_alpha100_500vox
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_features_chunked import (          # noqa: E402 - path insert must come first
    LAYERS_ALL, LAYERS_RECON, NUM_VOXELS,
    load_index, load_betas, load_test_imagery, load_test_shared1000, standardize,
    build_image_cache, preprocess_batch, load_encoder,
    decoded_dir, model_dir, layer_complete, save_decoded, save_model_stats,
)


# --- CLI --------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subj', type=int, nargs='+', default=[1, 2, 5, 7],
                   choices=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument('--index_dir', default='./data/fmri/NSDcore')
    p.add_argument('--imagery_h5', default='./data/fmri/NSDImagery')
    p.add_argument('--nsd_stimuli',
                   default='./data/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
    p.add_argument('--models_dir', default='../models')
    p.add_argument('--out_dir', default='./data/decoded_features/NSDImagery')
    p.add_argument('--decoder_dir', default='./data/feature_decoders/NSDcore')
    p.add_argument('--analysis_name', default='nsdcore40ses_fastl2lir_alpha100_500vox')
    p.add_argument('--cache_dir', default='./data/cache/nsd224')

    p.add_argument('--test', choices=['imagery', 'shared1000'], default='imagery')
    p.add_argument('--n_test', type=int, default=None)

    layers = p.add_mutually_exclusive_group()
    layers.add_argument('--layers', nargs='+', default=None, choices=LAYERS_ALL)
    layers.add_argument('--layers_recon', action='store_true')

    p.add_argument('--alpha', type=float, default=100.0)
    p.add_argument('--n_feat', type=int, default=500,
                   help="fastl2lir's per-unit voxel count -- Shirakawa's config's "
                        "'rois voxel num: nsdgeneral: 500'")
    p.add_argument('--batch', type=int, default=32, help='images per VGG19 forward')
    p.add_argument('--device', default='cpu',
                   help='unlike decode_features_chunked.py this defaults to cpu: VRAM is '
                        'committed elsewhere and the VGG19 forward is cheap regardless')
    p.add_argument('--unit_chunk_gb', type=float, default=8.0,
                   help='RAM budget for one unit-chunk\'s (chunk_units x n_voxels) float64 '
                        'accumulators; layers bigger than this are processed in multiple '
                        'passes over the (cheap) forward pass, one pass per chunk')
    p.add_argument('--solve_batch', type=int, default=2048,
                   help='units per batched torch.linalg.solve call')

    p.add_argument('--n_train_images', type=int, default=None,
                   help='SMOKE TEST ONLY: fit on the first N unique training images')
    p.add_argument('--verify', metavar='LAYER', default=None,
                   help='refit LAYER with the real vendored fastl2lir.FastL2LiR (Y expanded '
                        'to trials) and compare test-set predictions. Only sensible on '
                        'fc6/fc7/fc8: bigger layers do not fit expanded in memory')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--selftest', action='store_true')
    return p.parse_args()


# Method. Calling FastL2LiR.fit() directly is not possible here: all 19 layers are needed and conv1_1 has 3.2M
# units. Expanding Y to trials would take ~27000 x 3.2M float32 = 345 GB; the unique 9000 x 3.2M matrix is ~115 GB
# and stays in RAM, never on disk (disk is the scarce resource). --verify fits a real FastL2LiR on a small layer.
# fastl2lir's per-unit branch (__sub_fit) needs, for every unit, a Pearson correlation against the voxels and a
# small (n_feat+1)^2 solve from a Gram submatrix and an RHS subvector. None of that needs Y at trial level:
#   W0 = Xa^T Xa + alpha I    Xa = [Xn_train | 1]; independent of Y, computed once and shared by all layers
#   W1 = Yn^T Xa = Yn_unique^T @ Xagg    Xagg[i] = sum of Xa's rows over the trials of unique image i
#        (index_add_). Exact, because Yn is constant across the trials of one image.
# W1 needs the standardised Y, whose mean and std are known only after every image is seen. So the raw
# RawW1 = Y_unique_raw^T @ Xagg is accumulated in one streaming pass, with sum(Y_raw) and sum(Y_raw^2), and
# corrected at the end:  W1 = (RawW1 - outer(y_mean, colsum(Xa))) / y_norm[:, None].
# Voxel selection needs the trial-level mean and variance of Y (from RawW1's bias column, the per-image trial
# counts) and the voxel/unit cross term (W1[:, :-1] / n_trials minus the outer product of means); see
# fit_predict_layer(). No dense weights are kept: each unit's (n_feat+1,) solution is applied to the test matrix
# and discarded; --unit_chunk_gb bounds the per-chunk float64 arrays. The normalisation (checked against
# train_decoder_fastl2lir.py and predict_feature_fastl2lir.py) is standardize() of decode_features_chunked.py.

# --- global, layer-independent pieces ---------------------------------------------------

def compute_gram(xn_train_aug: np.ndarray, alpha: float, device: str) -> torch.Tensor:
    """W0 = Xa^T Xa + alpha I, float64. Computed once per subject, reused by every layer and
    every unit: fastl2lir's own W0 likewise does not depend on Y (fastl2lir.py:263)."""
    xa = torch.from_numpy(xn_train_aug).to(device=device, dtype=torch.float64)
    w0 = xa.T @ xa
    w0.diagonal().add_(alpha)
    return w0


def aggregate_by_image(xn_train_aug: np.ndarray, uidx: np.ndarray, n_unique: int, device: str):
    """Xagg[i,:] = sum of Xa's rows over the trials of unique image i (index_add_), plus the
    fixed column-sum over all trials. Xa's bias column is all-ones; Xagg[:, -1] is exactly
    the per-image trial count, used below as the correlation's repetition weight."""
    xa = torch.from_numpy(xn_train_aug).to(device=device, dtype=torch.float64)
    xagg = torch.zeros((n_unique, xa.shape[1]), dtype=torch.float64, device=device)
    xagg.index_add_(0, torch.from_numpy(uidx.astype(np.int64)).to(device), xa)
    colsum = xa.sum(dim=0)
    counts = xagg[:, -1].clone()
    return xagg, colsum, counts


# --- per-layer streaming fit + predict --------------------------------------------------

def build_layer_features(cache, encoder, store, layer, n_unique, n_units, batch, device,
                         preprocess_fn=preprocess_batch):
    """Single forward pass over all unique training images, returning raw features as an
    in-RAM (n_unique, n_units) float32 array, up to ~115 GB for conv1_1/conv1_2. Held in RAM
    rather than on disk: disk is the scarce resource on this shared server (frequently >95%
    full), RAM is not (2 TB, usually >1.5 TB free).

    A forward pass per unit-chunk would re-run the (CPU-bound) VGG19 forward pass to bound RAM:
    fc8 alone (the cheapest layer) takes ~200s per pass, and conv1_1 (3.2M units) would need
    ~188 such passes at the default chunk budget, i.e. >10h for one layer. Doing the forward
    pass exactly once per layer needs one ~200s pass.

    preprocess_fn is a parameter (not just a module-level call) so selftest() can substitute a
    fake in without monkeypatching a global; a script run as __main__ does not share its
    namespace with a second `import thisfile` of itself; a module-attribute patch would
    silently miss the copy actually running.
    """
    feat = np.empty((n_unique, n_units), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i0 in range(0, n_unique, batch):
            i1 = min(i0 + batch, n_unique)
            encoder(preprocess_fn(cache[i0:i1], None, False, device))
            y = store[layer].reshape(i1 - i0, -1)
            feat[i0:i1] = y.detach().to(torch.float32).cpu().numpy()
    print(f'    layer feature cache: {n_unique}x{n_units} float32 in RAM '
          f'({time.time() - t0:.0f}s)')
    return feat


def fit_predict_layer(cache, encoder, store, layer, n_units, xn_test_aug, w0, xagg, colsum,
                      counts, n_trials, n_unique, mean_x, std_x, dead_voxel, n_feat,
                      unit_chunk_gb, solve_batch, batch, device,
                      keep_raw_for_verify=False, preprocess_fn=preprocess_batch):
    """The whole per-unit fastl2lir branch for one layer, folded into test-set prediction.

    Returns pred (n_test, n_units) already denormalised, y_mean/y_norm (n_units,), and (if
    keep_raw_for_verify) the raw (n_unique, n_units) feature matrix; only ever requested for
    --verify's small layers.
    """
    n_voxels = xn_test_aug.shape[1] - 1
    n_test = xn_test_aug.shape[0]
    xn_test_aug_t = torch.from_numpy(xn_test_aug).to(device=device, dtype=torch.float64)

    feat = build_layer_features(cache, encoder, store, layer, n_unique, n_units, batch,
                                device, preprocess_fn)

    # RawW1 row (n_voxels+1) plus this chunk's own feature column (n_unique), both float64
    # once moved to the accumulation dtype; --unit_chunk_gb bounds this, not `feat` itself.
    bytes_per_unit = 8 * (n_voxels + 1 + n_unique)
    chunk_units = max(1, int(unit_chunk_gb * 1024**3 / bytes_per_unit))
    n_chunks = (n_units + chunk_units - 1) // chunk_units

    pred_all = np.empty((n_test, n_units), dtype=np.float32)
    y_mean_all = np.empty(n_units, dtype=np.float64)
    y_norm_all = np.empty(n_units, dtype=np.float64)

    for c in range(n_chunks):
        u0, u1 = c * chunk_units, min((c + 1) * chunk_units, n_units)
        sz = u1 - u0
        yb = torch.from_numpy(np.ascontiguousarray(feat[:, u0:u1])
                              ).to(device=device, dtype=torch.float64)  # (n_unique, sz)

        raw_w1 = yb.T @ xagg                                       # (sz, n_voxels+1)
        s1 = yb.sum(dim=0)
        s2 = (yb * yb).sum(dim=0)
        s1w = yb.T @ counts
        s2w = (yb * yb).T @ counts
        del yb

        n = float(n_unique)
        y_mean = s1 / n
        y_var = (s2 - n * y_mean**2) / (n - 1.0)
        y_norm = torch.sqrt(torch.clamp(y_var, min=0.0))
        y_norm_safe = torch.where(y_norm == 0, torch.ones_like(y_norm), y_norm)

        # W1 in standardised space, derived post-hoc (see module docstring), avoids a
        # second pass that would need y_mean/y_norm to standardise Y before accumulating.
        w1 = (raw_w1 - torch.outer(y_mean, colsum)) / y_norm_safe[:, None]

        # Trial-level (repetition-weighted) mean/std of standardised Y, needed for the
        # Pearson correlation used to rank voxels, NOT the same as y_mean/y_norm above,
        # which are unweighted over unique images.
        nt = float(n_trials)
        mean_y_raw_trial = s1w / nt
        var_y_raw_trial = (s2w - nt * mean_y_raw_trial**2) / (nt - 1.0)
        mean_y_trial = (mean_y_raw_trial - y_mean) / y_norm_safe
        var_y_trial = var_y_raw_trial / (y_norm_safe**2)
        std_y_trial = torch.sqrt(torch.clamp(var_y_trial, min=0.0))
        std_y_safe = torch.where(std_y_trial == 0, torch.ones_like(std_y_trial), std_y_trial)

        std_x_safe = torch.where(std_x == 0, torch.ones_like(std_x), std_x)
        cov_xy = w1[:, :-1] / nt - torch.outer(mean_y_trial, mean_x)
        corr = cov_xy / (std_y_safe[:, None] * std_x_safe[None, :])
        corr[:, dead_voxel] = 0.0                    # matches fastl2lir.py's C[I, :] = 0.0

        topk = torch.topk(corr.abs(), n_feat, dim=1, largest=True, sorted=False).indices
        bias_idx = n_voxels

        pred_chunk = torch.empty((n_test, sz), dtype=torch.float64, device=device)
        for b0 in range(0, sz, solve_batch):
            b1 = min(b0 + solve_batch, sz)
            bsz = b1 - b0
            idx = topk[b0:b1]                          # (bsz, n_feat)
            idx_bias = torch.cat([idx, torch.full((bsz, 1), bias_idx, dtype=idx.dtype,
                                                    device=device)], dim=1)   # (bsz, n_feat+1)
            w0_sub = w0[idx_bias.unsqueeze(2), idx_bias.unsqueeze(1)]        # (bsz, F+1, F+1)
            rhs_vox = torch.gather(w1[b0:b1], 1, idx)                       # (bsz, n_feat)
            rhs = torch.cat([rhs_vox, w1[b0:b1, -1:]], dim=1).unsqueeze(2)  # (bsz, F+1, 1)
            wb = torch.linalg.solve(w0_sub, rhs).squeeze(2)                 # (bsz, F+1)

            x_sel = xn_test_aug_t[:, idx]               # (n_test, bsz, n_feat) via fancy index
            x_sel = torch.einsum('tbf,bf->tb', x_sel, wb[:, :-1])
            pred_chunk[:, b0:b1] = x_sel + wb[:, -1][None, :]

        pred_all[:, u0:u1] = (pred_chunk * y_norm[None, :] + y_mean[None, :]
                              ).to(torch.float32).cpu().numpy()
        y_mean_all[u0:u1] = y_mean.cpu().numpy()
        y_norm_all[u0:u1] = y_norm.cpu().numpy()

    raw_kept = feat if keep_raw_for_verify else None
    if raw_kept is None:
        del feat
    return pred_all, y_mean_all, y_norm_all, raw_kept


# --- verification -------------------------------------------------------------------------

def verify_against_fastl2lir(layer, xn_train, xn_test, y_raw_unique, uidx, y_mean, y_norm,
                             alpha, n_feat, pred_ours):
    """Refit LAYER with the real vendored fastl2lir.FastL2LiR and compare predictions.

    Reproduces train_decoder_fastl2lir.py + predict_feature_fastl2lir.py directly: normalise Y
    with the SAME statistics derived while streaming, expand unique -> trial (Y_sort), fit with
    the real per-unit branch, predict, denormalise. Only usable where Y fits expanded in RAM.
    """
    from fastl2lir import FastL2LiR

    print(f'  --verify {layer}: refitting with the real fastl2lir.FastL2LiR '
          f'(alpha={alpha:g}, n_feat={n_feat})')
    t0 = time.time()
    m, s = y_mean[np.newaxis, :], y_norm[np.newaxis, :]
    s_safe = np.where(s == 0, 1.0, s)
    yn = (y_raw_unique.astype(np.float64) - m) / s_safe
    yn[:, (s == 0).flatten()] = 0.0
    yn_expanded = yn[uidx].astype(np.float32)

    model = FastL2LiR()
    model.fit(xn_train.astype(np.float32), yn_expanded, alpha=alpha, n_feat=n_feat,
             dtype=np.float32)
    pred_std = model.predict(xn_test.astype(np.float32), dtype=np.float32)
    pred = pred_std.astype(np.float64) * s + m

    scale = np.abs(pred).max()
    diff = np.abs(pred - pred_ours).max()
    print(f'    fastl2lir fit {time.time() - t0:.1f}s | max|diff| {diff:.3e} on features of '
          f'magnitude {scale:.3e} -> relative {diff / max(scale, 1e-12):.2e}')
    if diff > 1e-3 * max(scale, 1e-12):
        sys.exit('    our streaming fit and the real fastl2lir disagree by more than 1e-3 '
                 'relative. Do not use these decoded features.')
    print('    MATCH')


# --- main loop ------------------------------------------------------------------------

def run_subject(subject: int, args, layers):
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
    n_unique, n_trials = len(train_img), xn_train.shape[0]

    xn_train_aug = np.hstack([xn_train, np.ones((n_trials, 1), dtype=np.float32)])
    xn_test_aug = np.hstack([xn_test, np.ones((x_test.shape[0], 1), dtype=np.float32)])

    t0 = time.time()
    w0 = compute_gram(xn_train_aug, args.alpha, args.device)
    xagg, colsum, counts = aggregate_by_image(xn_train_aug, train_uidx, n_unique, args.device)
    mean_x = torch.from_numpy(xn_train.mean(axis=0)).to(device=args.device, dtype=torch.float64)
    std_x = torch.from_numpy(xn_train.std(axis=0, ddof=1)).to(device=args.device,
                                                                dtype=torch.float64)
    var0_x = xn_train.var(axis=0, ddof=0)               # fastl2lir.py's own dead-voxel mask
    dead_voxel = torch.from_numpy(var0_x < 1e-8).to(args.device)
    print(f'  global Gram + aggregation: {time.time() - t0:.1f}s '
          f'({int(dead_voxel.sum())} dead voxel(s))')

    cache = build_image_cache(Path(args.nsd_stimuli), train_img,
                              Path(args.cache_dir) / f'subj{subject:02d}_train224.npy')
    encoder, store = load_encoder(Path(args.models_dir), todo, args.device)

    for lyr in todo:
        t0 = time.time()
        with torch.no_grad():
            probe = preprocess_batch(cache[:1], None, False, args.device)
            encoder(probe)
            n_units = int(np.prod(store[lyr].shape[1:]))
            shape = tuple(store[lyr].shape[1:])

        keep_verify = args.verify == lyr
        pred, y_mean, y_norm, raw_kept = fit_predict_layer(
            cache, encoder, store, lyr, n_units, xn_test_aug, w0, xagg, colsum, counts,
            n_trials, n_unique, mean_x, std_x, dead_voxel, args.n_feat, args.unit_chunk_gb,
            args.solve_batch, args.batch, args.device, keep_raw_for_verify=keep_verify)

        pred = pred.reshape((x_test.shape[0],) + shape)
        y_mean_np = y_mean.astype(np.float32).reshape((1,) + shape)
        y_norm_np = y_norm.astype(np.float32).reshape((1,) + shape)

        if keep_verify:
            verify_against_fastl2lir(lyr, xn_train, xn_test, raw_kept, train_uidx,
                                     y_mean, y_norm, args.alpha, args.n_feat,
                                     pred.reshape(x_test.shape[0], -1).astype(np.float64))

        save_decoded(out_root, lyr, subject, roi, labels, pred)
        save_model_stats(dec_root, lyr, subject, roi, x_mean, x_norm, y_mean_np, y_norm_np,
                         {'alpha': float(args.alpha), 'n_feat': int(args.n_feat),
                          'n_train_trials': int(n_trials), 'n_train_images': int(n_unique)})
        n_dead = int((y_norm == 0).sum())
        print(f'  {lyr:8s} {str(shape):18s} pred |max| {np.abs(pred).max():.3e} '
              f'| {n_dead} constant unit(s) | {time.time() - t0:.0f}s | saved')

    print(f'  subj{subject:02d} done')


def selftest(device: str, alpha: float):
    """fit_predict_layer against a real fastl2lir.FastL2LiR fit, on synthetic data sized so
    n_feat < n_voxels actually triggers the per-unit selection branch. Needs no data/GPU.

    n_feat is fixed here (not taken from --n_feat) because the synthetic n_vox=60 is only
    big enough for a small n_feat, unrelated to the 500 used against real data.
    """
    rng = np.random.default_rng(0)
    n_img, n_rep, n_vox, n_test, n_unit, n_feat = 80, 3, 60, 6, 15, 20

    uidx = np.repeat(np.arange(n_img), n_rep).astype(np.int32)
    feats = (rng.normal(size=(n_img, n_unit)).astype(np.float64) * 5 + 2)
    feats[:, 0] = 3.0                                    # a constant unit: y_norm 0
    w = rng.normal(size=(n_vox, n_unit))
    x_train = (feats[uidx] @ w.T + rng.normal(size=(n_img * n_rep, n_vox)) * 0.5
              ).astype(np.float32)
    x_train[:, 0] = 7.0                                  # a constant voxel: std_x 0
    x_test = rng.normal(size=(n_test, n_vox)).astype(np.float32)

    xn_train, xn_test, _, _ = standardize(x_train, x_test)
    n_trials = xn_train.shape[0]
    xn_train_aug = np.hstack([xn_train, np.ones((n_trials, 1), dtype=np.float32)])
    xn_test_aug = np.hstack([xn_test, np.ones((n_test, 1), dtype=np.float32)])

    w0 = compute_gram(xn_train_aug, alpha, device)
    xagg, colsum, counts = aggregate_by_image(xn_train_aug, uidx, n_img, device)
    mean_x = torch.from_numpy(xn_train.mean(axis=0)).to(device=device, dtype=torch.float64)
    std_x = torch.from_numpy(xn_train.std(axis=0, ddof=1)).to(device=device, dtype=torch.float64)
    dead_voxel = torch.from_numpy(xn_train.var(axis=0, ddof=0) < 1e-8).to(device)

    # Fake "cache/encoder/store": stream_layer_unit_chunk expects to call encoder(batch) and
    # read store[layer]; here it directly indexes the precomputed synthetic feature array.
    class FakeStore(dict):
        pass

    store = FakeStore()

    class FakeEncoder:
        def __call__(self, x):
            i0 = int(x[0, 0].item())
            store['y'] = torch.from_numpy(feats[i0:i0 + x.shape[0]])

    def fake_preprocess(imgs, *_a, **_kw):
        # imgs is a slice of `cache`, which is made equal to arange(n_img); the fake
        # encoder can recover the batch's image indices. (batch, 1): x[0, 0] is then a
        # true scalar, unlike a reshape/expand to a 4D image-shaped tensor would give.
        return torch.from_numpy(imgs.astype(np.float64)).reshape(-1, 1)

    cache = np.arange(n_img)
    pred, y_mean, y_norm, _ = fit_predict_layer(
        cache, FakeEncoder(), store, 'y', n_unit, xn_test_aug, w0, xagg, colsum, counts,
        n_trials, n_img, mean_x, std_x, dead_voxel, n_feat, unit_chunk_gb=1.0,
        solve_batch=7, batch=8, device=device, preprocess_fn=fake_preprocess)

    verify_against_fastl2lir('selftest', xn_train, xn_test, feats.astype(np.float32), uidx,
                             y_mean, y_norm, alpha, n_feat, pred.astype(np.float64))


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if args.selftest:
        selftest(args.device, args.alpha)
        return

    if args.layers_recon:
        layers = LAYERS_RECON
    elif args.layers:
        layers = [l for l in LAYERS_ALL if l in args.layers]
    else:
        layers = LAYERS_ALL

    if args.verify and args.verify not in layers:
        sys.exit(f'--verify {args.verify} is not among the layers being computed')

    if args.test == 'shared1000' and 'NSDImagery' in args.out_dir:
        sys.exit(f'--test shared1000 with --out_dir {args.out_dir}: pass a separate '
                 '--out_dir so the bridge does not land among the imagery features')

    print(f'alpha={args.alpha:g}  n_feat={args.n_feat}  layers: {" ".join(layers)}')
    print(f'decoded features -> {Path(args.out_dir) / args.analysis_name}/caffe/VGG19')
    print(f'decoder stats    -> {Path(args.decoder_dir) / args.analysis_name}/caffe/VGG19')

    for subject in args.subj:
        run_subject(subject, args, layers)

    print('\nall done')


if __name__ == '__main__':
    main()
