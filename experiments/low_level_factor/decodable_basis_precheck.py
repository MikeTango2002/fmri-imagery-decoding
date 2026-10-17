"""Decoder-side check of a decodable basis for z_low: SVD of the ridge predictions instead of the image PCA.

Inputs:  NSD-Core betas and split; the alpha of the uncompressed ridge (stage1_summary.json of bfull)
Outputs: --out decodable_basis_precheck.json: PixCorr and SSIM per subject and k (64, 128, 256)

The PCA of z_low orders its 1024 dimensions by IMAGE variance, but the brain reads few directions of it
(Sections 8.1.3, 8.3). Here the top-k right singular vectors of the ridge's own predictions replace the PCA
and the complement is restandardized (residual sampling failed). On the development split only, a dev-held
ridge, the same dev images decoded by both arms. Criterion fixed in advance: a k wins if it does not lose on
PixCorr AND SSIM together on at least 3 of 4 subjects. Needs a GPU (load_vaes calls .cuda()).

Usage:
    python decodable_basis_precheck.py [--subj 1 --n-dev 150 --k 64 128 256]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vdvae_lib"))

from fit_ridge import average_by_image, load_nsdcore_split  # noqa: E402
from trisect.nsd import zscore_fmri  # noqa: E402
from trisect.ridge import restandardize  # noqa: E402
from fit_ridge_uncompressed_zlow import eigh_xtx, fit_final  # noqa: E402
from decode_zlow_to_image import decode_raw_batch, decode_zlow_batch, load_pca_basis  # noqa: E402
from trisect.pca import layer_bounds, layer_shapes  # noqa: E402
from extract_vdvae_latents import DEVICE, H_DICT, IMAGES_H5, VDVAE_RES, dotdict  # noqa: E402
from model_utils import load_vaes, set_up_data  # noqa: E402
from trisect.metrics import pixcorr  # noqa: E402
from residual_sampling_precheck import dev_pred_mmse, load_gt_64, ssim_global  # noqa: E402

SUBJECTS = (1, 2, 5, 7)
K_DEFAULT = (64, 128, 256)
INDEX_DIR = "data/nsdcore_index"
BETAS_DIR = "data/mindeyev2_dataset"
WEIGHTS_DIR = "data/betas_trisect/ridge_weights_devheld"          # control (PCA) branch
BFULL_STAGE1_DIR = "data/betas_trisect/ridge_weights_g2_1_bfull"  # alpha_selected source only
RAW_LATENTS_PATH = "data/targets/vdvae_latents_31l.npy"
RAW_LAYER_DIMS_PATH = "data/targets/vdvae_latents_31l_layer_dims.json"
DEV_MASK_DIR = "data/targets"
PCA_DIR = "data/targets/pca_zlow"
OUT_DEFAULT = "data/results/low_level_factor/decodable_basis_precheck.json"
SEED = 0


def bfull_devheld_predict(subj, index_dir, betas_dir, dev_mask_dir, bfull_stage1_dir,
                           raw_latents_mmap):
    """Fit voxel -> 91168 raw-latent ridge on train_sel = ~shared1000 & ~dev (never dev, unlike
    the production weights of fit_ridge_uncompressed_zlow.py), at the bfull alpha already selected
    there, and return per-image predictions on BOTH a train_sel subsample and the full dev split
    (trial-level predict_flat, then average_by_image, exactly the convention of fit_ridge.py)."""
    betas, img73k, is_shared = load_nsdcore_split(subj, Path(index_dir), Path(betas_dir))
    dev_mask_73k = np.load(Path(dev_mask_dir) / f"nsdcore_dev_mask_subj{subj:02d}.npy")
    dev_trial_mask = dev_mask_73k[img73k]
    assert not np.any(dev_trial_mask & is_shared), "dev mask overlaps shared1000"
    train_final_mask = ~is_shared
    train_sel_mask = train_final_mask & ~dev_trial_mask

    fmri_normed = zscore_fmri(betas, train_sel_mask)
    train_sel_fmri = fmri_normed[train_sel_mask]
    dev_fmri = fmri_normed[dev_trial_mask]
    del betas, fmri_normed

    train_sel_img73k = img73k[train_sel_mask]
    dev_img73k = img73k[dev_trial_mask]

    with open(Path(bfull_stage1_dir) / f"subj{subj:02d}" / "stage1_summary.json") as f:
        alpha = json.load(f)["alpha_selected"]

    eigvals, eigvecs = eigh_xtx(train_sel_fmri)
    y_train_sel = np.asarray(raw_latents_mmap[train_sel_img73k]).astype(np.float32)
    weight, bias = fit_final(train_sel_fmri, eigvals, eigvecs, y_train_sel, alpha)
    del eigvals, eigvecs, y_train_sel

    pred_train = (train_sel_fmri.astype(np.float32) @ weight.T + bias).astype(np.float64)
    pred_train_avg, train_uniq = average_by_image(pred_train, train_sel_img73k)
    del pred_train

    pred_dev = (dev_fmri.astype(np.float32) @ weight.T + bias).astype(np.float64)
    pred_dev_avg, dev_uniq = average_by_image(pred_dev, dev_img73k)
    del pred_dev, weight, bias

    return pred_train_avg, train_uniq, pred_dev_avg, dev_uniq


def rotate(x, mean_c, Vk):
    """coef_k(x) = (x - mean_c) @ Vk.T, the projection onto the basis, and residual(x), what falls
    outside it, without ever forming the (91168, 91168) completion of the basis. The caller
    restandardizes the k coefficients and the residual separately. `Vk` is (k, 91168), rows
    orthonormal (a slice of an SVD's Vt)."""
    xc = x - mean_c
    coef = xc @ Vk.T
    resid = xc - coef @ Vk
    return coef, resid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--k", type=int, nargs="+", default=list(K_DEFAULT))
    ap.add_argument("--n-dev", type=int, default=150, help="cap on dev images actually decoded")
    ap.add_argument("--n-train-sub", type=int, default=2000,
                     help="train_sel images subsampled for the SVD basis + restandardize stats")
    ap.add_argument("--index-dir", type=str, default=INDEX_DIR)
    ap.add_argument("--betas-dir", type=str, default=BETAS_DIR)
    ap.add_argument("--weights-dir", type=str, default=WEIGHTS_DIR)
    ap.add_argument("--bfull-stage1-dir", type=str, default=BFULL_STAGE1_DIR)
    ap.add_argument("--dev-mask-dir", type=str, default=DEV_MASK_DIR)
    ap.add_argument("--pca-dir", type=str, default=PCA_DIR)
    ap.add_argument("--raw-latents", type=str, default=RAW_LATENTS_PATH)
    ap.add_argument("--raw-layer-dims", type=str, default=RAW_LAYER_DIMS_PATH)
    ap.add_argument("--images-h5", type=str, default=IMAGES_H5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    print("Loading VDVAE (ImageNet64, EMA weights) ...")
    H = dotdict(dict(H_DICT))
    H, _ = set_up_data(H)
    ema_vae = load_vaes(H)
    pca_basis = load_pca_basis(args.pca_dir)
    with open(args.raw_layer_dims) as f:
        layer_dims = json.load(f)["layer_dims"]
    shapes, bounds = layer_shapes(layer_dims), layer_bounds(layer_dims)

    raw_latents_mmap = np.load(args.raw_latents, mmap_mode="r")

    results = {}
    n_win_both = {k: 0 for k in args.k}
    n_subj = 0
    for subj in args.subj:
        rng = np.random.default_rng(args.seed)
        print(f"\n=== subj0{subj} ===")

        # control: the production dev-held PCA z_low pipeline, as in residual_sampling_precheck.py.
        pred_mmse_pca, tgt_mean_pca, tgt_std_pca, ids_pca = dev_pred_mmse(
            subj, None, Path(args.index_dir), Path(args.betas_dir),
            Path(args.weights_dir), Path(args.dev_mask_dir), rng)

        # treatment ingredients: bfull devheld ridge predictions, train_sel + dev.
        pred_train, train_uniq, pred_dev, dev_uniq = bfull_devheld_predict(
            subj, args.index_dir, args.betas_dir, args.dev_mask_dir, args.bfull_stage1_dir,
            raw_latents_mmap)
        assert np.array_equal(ids_pca, dev_uniq), "control/treatment dev image sets disagree"

        rng_sub = np.random.default_rng(args.seed + 1)
        n_sub = min(args.n_train_sub, len(train_uniq))
        sub_idx = np.sort(rng_sub.choice(len(train_uniq), size=n_sub, replace=False))
        pred_train_sub = pred_train[sub_idx]
        y_true_train_sub = np.asarray(raw_latents_mmap[train_uniq[sub_idx]]).astype(np.float64)
        del pred_train

        n_dev = min(args.n_dev, len(dev_uniq))
        dev_sel = np.sort(rng.choice(len(dev_uniq), size=n_dev, replace=False))
        ids = dev_uniq[dev_sel]
        pred_mmse_pca_sel = pred_mmse_pca[dev_sel]
        pred_dev_sel = pred_dev[dev_sel]

        gt = load_gt_64(ids, args.images_h5, DEVICE)

        pred_control = restandardize(pred_mmse_pca_sel, pred_mmse_pca_sel.mean(axis=0),
                                      pred_mmse_pca_sel.std(axis=0), tgt_mean_pca, tgt_std_pca)
        recon_control = decode_zlow_batch(pred_control, pca_basis, ema_vae, DEVICE)
        pc_c, ss_c = pixcorr(recon_control, gt), ssim_global(recon_control, gt)
        print(f"  control (PCA+restandardize): PixCorr={pc_c.mean():.4f}  SSIM(g)={ss_c.mean():.4f}")

        mean_c = pred_train_sub.mean(axis=0)
        Xc = pred_train_sub - mean_c
        print(f"  SVD of centered train predictions ({Xc.shape}) ...")
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)

        results[f"subj{subj:02d}"] = {"n_dev": int(n_dev), "n_train_sub": int(n_sub),
                                       "pixcorr_control": float(pc_c.mean()),
                                       "ssim_control": float(ss_c.mean()), "arms": {}}
        n_subj += 1

        for k in args.k:
            Vk = Vt[:k]
            coef_pred_train, resid_pred_train = rotate(pred_train_sub, mean_c, Vk)
            coef_true_train, resid_true_train = rotate(y_true_train_sub, mean_c, Vk)

            coef_pred_dev, resid_pred_dev = rotate(pred_dev_sel, mean_c, Vk)
            coef_dev_restd = restandardize(coef_pred_dev, coef_pred_train.mean(axis=0),
                                            coef_pred_train.std(axis=0),
                                            coef_true_train.mean(axis=0), coef_true_train.std(axis=0))
            resid_dev_restd = restandardize(resid_pred_dev, resid_pred_train.mean(axis=0),
                                             resid_pred_train.std(axis=0),
                                             resid_true_train.mean(axis=0), resid_true_train.std(axis=0))
            raw_treatment = mean_c + coef_dev_restd @ Vk + resid_dev_restd

            recon_t = decode_raw_batch(raw_treatment.astype(np.float32), shapes, bounds, ema_vae, DEVICE)
            pc_t, ss_t = pixcorr(recon_t, gt), ssim_global(recon_t, gt)

            win_pixcorr = bool(pc_t.mean() >= pc_c.mean())
            win_ssim = bool(ss_t.mean() >= ss_c.mean())
            win_both = win_pixcorr and win_ssim
            n_win_both[k] += int(win_both)

            print(f"  k={k:4d}: PixCorr={pc_t.mean():.4f} ({'win' if win_pixcorr else 'lose'})  "
                  f"SSIM(g)={ss_t.mean():.4f} ({'win' if win_ssim else 'lose'})")
            results[f"subj{subj:02d}"]["arms"][str(k)] = {
                "pixcorr_treatment": float(pc_t.mean()), "ssim_treatment": float(ss_t.mean()),
                "win_pixcorr": win_pixcorr, "win_ssim": win_ssim, "win_both": win_both,
            }

    gate = {}
    any_pass = False
    for k in args.k:
        k_pass = n_win_both[k] >= 3
        any_pass = any_pass or k_pass
        gate[str(k)] = {"n_subj_win_both_metrics": n_win_both[k], "n_subj": n_subj,
                         "proceed_to_gpu_branch": k_pass}
        print(f"\nGate k={k}: wins on both metrics on {n_win_both[k]}/{n_subj} subjects -> "
              f"{'PASS' if k_pass else 'FAIL'}")
    results["gate"] = {
        "criterion": "treatment must not lose on PixCorr AND SSIM together on >=3/4 subjects, "
                     "for at least one k",
        "per_k": gate, "proceed_to_gpu_branch": any_pass,
    }
    print(f"\nOverall: {'PROCEED to GPU branch' if any_pass else 'DO NOT PROCEED'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
