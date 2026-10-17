#!/usr/bin/env python
'''What alpha does NSD-core want? A k-fold CV of the stage-1 ridge, on the training set only.

Inputs:  the NSD-core training betas and images (9000 unique), as in decode_features_chunked.py
Outputs: a TSV of unit_r, prof_r and prof_r_centred per layer, fold and alpha (--out_tsv); nothing else

Why: a reconstruction from the true VGG19 features of the 100 shared1000 photos (a variant of diag_recon_ceiling.py,
not in this repository) scores EffNet-B .646 and SwAV .506, better than Kneeland's .797 and .528: the gap lies in
stage 1. Its one unmeasured number is alpha = 100, from Shen's Deeprecon config (~500 selected voxels; here 15724
voxels), and Kneeland et al. do not publish theirs. The script writes no decoder, feature or reconstruction. The
criterion is unit_r, set before the run; see "Method" above FoldSolver.

Usage (from feature-decoding/, venv active):
    CUDA_VISIBLE_DEVICES=0 python ../src/diag_alpha_cv.py --subj 1 --out_tsv ../results/diag_alpha_cv.tsv
'''
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_features_chunked import (LAYERS_RECON, NUM_VOXELS, build_image_cache,
                                     load_betas, load_encoder, load_index,
                                     preprocess_batch, standardize)

DEFAULT_ALPHAS = [1e2, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 1e8]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subj', type=int, nargs='+', default=[1],
                   choices=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--alphas', type=float, nargs='+', default=DEFAULT_ALPHAS)
    p.add_argument('--layers', nargs='+', default=LAYERS_RECON)
    p.add_argument('--n_units', type=int, default=5000,
                   help='random units per layer on which the criterion is computed; '
                        'conv1_2 has 3.2M and they are not needed to estimate a mean')
    p.add_argument('--seed', type=int, default=0,
                   help='fixes both the fold split and the unit subsample')
    p.add_argument('--index_dir', default='./data/fmri/NSDcore')
    p.add_argument('--nsd_stimuli',
                   default='./data/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
    p.add_argument('--cache_dir', default='./data/cache/nsd224')
    p.add_argument('--models_dir', default='../models')
    p.add_argument('--batch', type=int, default=32, help='images per VGG19 forward')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out_tsv', default='')
    return p.parse_args()


# --- targets ---------------------------------------------------------------------------

def subsampled_targets(args, train_img, rng):
    """{layer: (n_images, n_units_sub) float32} and the unit indices kept.

    Only a subsample of units is ever materialised. The closed form needs the targets only
    through H_agg @ Y; restricting the columns of Y restricts the columns of the
    prediction and nothing else: every fold and every alpha then works on the same 5000
    units, which is what makes the CV cheap.
    """
    cache = build_image_cache(Path(args.nsd_stimuli), train_img,
                             Path(args.cache_dir) / f'subj{args.subject:02d}_train224.npy')
    encoder, store = load_encoder(Path(args.models_dir), args.layers, args.device)

    with torch.no_grad():
        encoder(preprocess_batch(cache[:1], None, False, args.device))
        shapes = {lyr: tuple(store[lyr].shape[1:]) for lyr in args.layers}

    units = {}
    for lyr in args.layers:
        n_tot = int(np.prod(shapes[lyr]))
        k = min(args.n_units, n_tot)
        units[lyr] = np.sort(rng.choice(n_tot, size=k, replace=False))
        print(f'  {lyr:8s} {n_tot:>9d} units -> {k} sampled')

    n_img = len(train_img)
    y = {lyr: torch.empty((n_img, len(units[lyr])), dtype=torch.float32, device=args.device)
         for lyr in args.layers}
    sel = {lyr: torch.from_numpy(units[lyr]).to(args.device) for lyr in args.layers}

    t0 = time.time()
    with torch.no_grad():
        for i0 in range(0, n_img, args.batch):
            i1 = min(i0 + args.batch, n_img)
            encoder(preprocess_batch(cache[i0:i1], None, False, args.device))
            for lyr in args.layers:
                y[lyr][i0:i1] = store[lyr].reshape(i1 - i0, -1)[:, sel[lyr]]
            if i1 % (args.batch * 50) == 0 or i1 == n_img:
                el = time.time() - t0
                print(f'    {i1}/{n_img} images, {el:.0f}s elapsed, '
                      f'{el * n_img / i1 - el:.0f}s left', flush=True)

    del encoder, store
    torch.cuda.empty_cache()
    return y


# Method. Inside every fold the estimator is the pipeline's own, imported from decode_features_chunked.py:
#   - the same standardize (ddof=1), refitted on the fold's training trials only; there is no leakage;
#   - the same closed form pred = H_agg Y + (1 - rowsum(H)) y_mean, with y_mean from the fold's training images;
#   - folds are drawn over unique images, never over trials; the ~3 repetitions of an image stay on one side,
#     and the held-out betas are averaged per image before prediction, as at prediction time.
# Only alpha varies. The criterion, fixed before the run, is unit_r: per unit, the correlation across the
# held-out images, averaged over units and folds. It asks whether the prediction tracks this image rather than
# that one, and ignores the shared mean feature profile. prof_r (across units within an image) and
# prof_r_centred (the same after removing the mean profile) are reported but do not select.
# What the answer is for, decided once, after reading it: alpha* near 100 means nothing to do (the residual gap is
# declared unexplained); alpha* >> 100 means a single refit at alpha*, declared as a hyperparameter the authors did
# not publish, re-selected by CV on the training set. Never a grid of alphas reconstructed and scored on the
# benchmark: no variant is chosen by its benchmark score.
# Cost: one VGG19 pass over the 9000 images (~5 min) keeping a random subsample of units per layer (~1.4 GB), then
# ~30 s per (fold, alpha); 5 folds x 10 alphas is ~30 min per subject.

# --- one fold --------------------------------------------------------------------------

class FoldSolver:
    """The fold's Gram, formed once, solved once per alpha.

    Same algebra as hat_matrix() in decode_features_chunked.py, kept as a class only so the
    Gram can outlive a single alpha.

    Xc^T Xc is 15724^2 in float64 and costs ~5e12 flops to build; the Cholesky that follows
    is an order of magnitude cheaper. Alpha enters only on the diagonal, and forming the
    Gram once and adding alpha to a COPY of its diagonal per alpha turns a 10-alpha sweep
    from 10 full builds into 1 build plus 10 factorisations.

    float64 throughout and TF32 off, for the reason given in hat_matrix(): with TF32 the
    product loses three significant digits.
    """

    def __init__(self, xn_tr, xn_val, uidx_tr, n_img_tr, device):
        self.device = device
        self.n_img_tr = n_img_tr
        self.xt = torch.from_numpy(xn_tr).to(device=device, dtype=torch.float64)
        self.xs = torch.from_numpy(xn_val).to(device=device, dtype=torch.float64)
        xbar = self.xt.mean(dim=0, keepdim=True)
        self.xt -= xbar
        self.xs -= xbar
        t0 = time.time()
        self.gram = self.xt.T @ self.xt
        self.uidx = torch.from_numpy(uidx_tr.astype(np.int64)).to(device)
        print(f'    Gram {tuple(self.gram.shape)} in {time.time() - t0:.0f}s', flush=True)

    def hat(self, alpha):
        g = self.gram.clone()
        g.diagonal().add_(alpha)
        chol = torch.linalg.cholesky(g)
        del g
        b = torch.cholesky_solve(self.xs.T.contiguous(), chol)      # (p, n_val)
        del chol
        h = (self.xt @ b).T.contiguous()                            # (n_val, n_trials)
        del b
        h_agg = torch.zeros((h.shape[0], self.n_img_tr), dtype=torch.float64,
                            device=self.device)
        h_agg.index_add_(1, self.uidx, h)
        rowsum = h.sum(dim=1)
        del h
        return h_agg.to(torch.float32), rowsum

    def free(self):
        del self.gram, self.xt, self.xs
        torch.cuda.empty_cache()


def scores(pred, true):
    """(unit_r, prof_r, prof_r_centred) for one layer of one fold, all on the GPU.

    unit_r    per unit, correlation ACROSS the held-out images; mean over units. Units with
              zero variance on either side (a dead ReLU) contribute nothing and are dropped.
    prof_r    per image, correlation ACROSS units; mean over images.
    prof_r_c  the same after removing, from both sides, the mean profile over the held-out
              images: the shared profile is common to every stimulus and inflates prof_r.
    """
    def corr(a, b, dim):
        a = a - a.mean(dim=dim, keepdim=True)
        b = b - b.mean(dim=dim, keepdim=True)
        na = a.norm(dim=dim)
        nb = b.norm(dim=dim)
        ok = (na > 0) & (nb > 0)
        r = (a * b).sum(dim=dim) / torch.clamp(na * nb, min=1e-12)
        return float(r[ok].mean()) if ok.any() else float('nan')

    unit_r = corr(pred, true, dim=0)
    prof_r = corr(pred, true, dim=1)
    pc = pred - pred.mean(dim=0, keepdim=True)
    tc = true - true.mean(dim=0, keepdim=True)
    return unit_r, prof_r, corr(pc, tc, dim=1)


def run_subject(args, subject, rows_out):
    args.subject = subject
    rng = np.random.default_rng(args.seed)

    idx = load_index(Path(args.index_dir), subject)
    train_rows, train_uidx = idx['train_rows'], idx['train_uidx']
    train_img = idx['train_img73k']
    n_img = len(train_img)
    print(f'\nsubj{subject:02d}: {len(train_rows)} trials over {n_img} unique images')

    y = subsampled_targets(args, train_img, rng)

    x_all = load_betas(Path(idx['betas_file']), train_rows)
    assert x_all.shape[1] == NUM_VOXELS[subject], \
        f'{x_all.shape[1]} voxels, the nsdgeneral mask has {NUM_VOXELS[subject]}'

    # Folds over IMAGES, so that an image's repetitions never straddle the split.
    order = rng.permutation(n_img)
    bounds = np.linspace(0, n_img, args.folds + 1).astype(int)

    for f in range(args.folds):
        val_img = np.sort(order[bounds[f]:bounds[f + 1]])
        is_val = np.zeros(n_img, dtype=bool)
        is_val[val_img] = True
        tr_img = np.flatnonzero(~is_val)
        # image id -> column of the fold's training block
        remap = np.full(n_img, -1, dtype=np.int64)
        remap[tr_img] = np.arange(len(tr_img))

        m_val = is_val[train_uidx]
        x_tr, x_val_trials = x_all[~m_val], x_all[m_val]
        uidx_tr = remap[train_uidx[~m_val]]
        uidx_val = train_uidx[m_val]

        # Held-out betas averaged per image: this is what predict_feature.py does on the
        # test set (average_sample: true); the CV predicts on the same kind of input.
        x_val = np.vstack([x_val_trials[uidx_val == u].mean(axis=0) for u in val_img])
        x_val = x_val.astype(np.float32)

        xn_tr, xn_val, _, _ = standardize(x_tr, x_val)
        print(f'  fold {f + 1}/{args.folds}: {len(tr_img)} train images '
              f'({xn_tr.shape[0]} trials), {len(val_img)} held out', flush=True)
        solver = FoldSolver(xn_tr, xn_val, uidx_tr, len(tr_img), args.device)

        for alpha in args.alphas:
            t0 = time.time()
            h_agg, rowsum = solver.hat(alpha)
            for lyr in args.layers:
                y_tr = y[lyr][torch.from_numpy(tr_img).to(args.device)]
                y_val = y[lyr][torch.from_numpy(val_img).to(args.device)]
                y_mean = y_tr.double().mean(dim=0)
                pred = (h_agg.double() @ y_tr.double() +
                        (1.0 - rowsum).unsqueeze(1) * y_mean.unsqueeze(0))
                u, p, pcen = scores(pred.float(), y_val)
                rows_out.append({'subject': subject, 'fold': f, 'alpha': alpha,
                                 'layer': lyr, 'unit_r': u, 'prof_r': p,
                                 'prof_r_centred': pcen})
            del h_agg, rowsum
            mean_u = np.mean([r['unit_r'] for r in rows_out
                              if r['subject'] == subject and r['fold'] == f
                              and r['alpha'] == alpha])
            print(f'    alpha {alpha:>9.3g}  unit_r {mean_u:+.4f}  '
                  f'({time.time() - t0:.0f}s)', flush=True)

        solver.free()

    del x_all, y
    torch.cuda.empty_cache()


def summarise(rows, alphas, layers):
    print('\n' + '=' * 72)
    print('Held-out NSD-core feature prediction, mean over folds and subjects')
    print(f"{'alpha':>10} {'unit_r':>9} {'prof_r':>9} {'prof_r_c':>9}   per-layer unit_r")
    best, best_v = None, -np.inf
    for a in alphas:
        sub = [r for r in rows if r['alpha'] == a]
        u = float(np.mean([r['unit_r'] for r in sub]))
        p = float(np.mean([r['prof_r'] for r in sub]))
        pc = float(np.mean([r['prof_r_centred'] for r in sub]))
        per = ' '.join(f"{l.replace('conv', 'c')}={np.mean([r['unit_r'] for r in sub if r['layer'] == l]):+.3f}"
                       for l in layers)
        if u > best_v:
            best, best_v = a, u
        print(f'{a:>10.3g} {u:>+9.4f} {p:>+9.4f} {pc:>+9.4f}   {per}')
    print('=' * 72)
    print(f'Criterion (declared before the run): mean unit_r over the {len(layers)} '
          f'reconstruction layers.')
    print(f'alpha* = {best:g}   (incumbent, inherited from Deeprecon: 100)')
    if best == 100:
        print('\nThe inherited value wins. Nothing to change: stage 1 has no unmeasured knob')
        print('left, and the residual gap on the bridge stays an unexplained implementation')
        print('difference.')
    else:
        print(f'\nalpha = 100 is not what NSD-core wants. Read the gain first: if unit_r at')
        print(f'{best:g} is barely above the one at 100, the knob is flat and refitting buys')
        print('nothing. If it is clearly above, ONE refit at alpha* is the defensible move --')
        print('declared in the thesis as a hyperparameter the authors never published,')
        print('re-selected on the training set alone. Never a grid scored on the benchmark.')


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    rows = []
    for subject in args.subj:
        run_subject(args, subject, rows)

    summarise(rows, args.alphas, args.layers)

    if args.out_tsv:
        import csv
        Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_tsv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {args.out_tsv}')


if __name__ == '__main__':
    sys.exit(main())
