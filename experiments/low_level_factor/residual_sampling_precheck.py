"""Decoder-side check of sample_residual(), the "z ~ q" alternative to restandardize() for z_low.

Inputs:  NSD-Core betas and split, ridge_weights_devheld, the frozen PCA and VDVAE decoder
Outputs: --out json with PixCorr and SSIM of both treatments, per subject

Residual sampling fills the part of z_low that the ridge does not predict with noise (Section 8.1.3). A
latent-space check once missed a failure of the decoder amplitude that appeared only after a full GPU run,
so every GPU run needs a check on the decoder output first. On the development split only, both treatments
of the same ridge prediction are decoded and scored (PixCorr, global SSIM). Criterion fixed in advance: no
loss on PixCorr AND SSIM together on at least 3 of 4 subjects. It failed: PixCorr fell by 55 to 70%.
Needs a GPU (load_vaes calls .cuda()).

Usage:
    python residual_sampling_precheck.py [--subj 1 --n-dev 500]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vdvae_lib"))

from fit_ridge import average_by_image, load_nsdcore_split  # noqa: E402
from predict_ridge_imagery import load_weights, predict_flat  # noqa: E402
from trisect.nsd import zscore_fmri_train_test as zscore_fmri  # noqa: E402
from trisect.ridge import estimate_resid_var, restandardize, sample_residual  # noqa: E402
from decode_zlow_to_image import decode_zlow_batch, load_pca_basis  # noqa: E402
from extract_vdvae_latents import DEVICE, H_DICT, IMAGES_H5, VDVAE_RES, dotdict  # noqa: E402
from model_utils import load_vaes, set_up_data  # noqa: E402
from trisect.metrics import pixcorr  # noqa: E402

SUBJECTS = (1, 2, 5, 7)
BRANCH = "z_low"
INDEX_DIR = "data/nsdcore_index"
BETAS_DIR = "data/mindeyev2_dataset"
WEIGHTS_DIR = "data/betas_trisect/ridge_weights_devheld"
DEV_MASK_DIR = "data/targets"
PCA_DIR = "data/targets/pca_zlow"
OUT_DEFAULT = "data/results/low_level_factor/residual_sampling_precheck.json"
SEED = 0


def ssim_global(pred, true, L=255.0):
    """Simplified GLOBAL (non-windowed) SSIM, per image, batched. A sanity-check number, not the
    benchmark metric."""
    p = pred.reshape(pred.shape[0], -1).astype(np.float64)
    t = true.reshape(true.shape[0], -1).astype(np.float64)
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mp, mt = p.mean(axis=1), t.mean(axis=1)
    vp, vt = p.var(axis=1), t.var(axis=1)
    cov = ((p - mp[:, None]) * (t - mt[:, None])).mean(axis=1)
    num = (2 * mp * mt + c1) * (2 * cov + c2)
    den = (mp ** 2 + mt ** 2 + c1) * (vp + vt + c2)
    return num / den


def load_gt_64(ids, images_h5, device):
    """img73k ids -> (n, 64, 64, 3) uint8, same 224->64 bilinear resize as everywhere else in
    the VDVAE pipeline (the encoder input of extract_vdvae_latents.py)."""
    gt_imgs = []
    with h5py.File(images_h5, "r") as f, torch.no_grad():
        for i in ids:
            img = torch.from_numpy(f["images"][int(i)].astype(np.float32)).unsqueeze(0).to(device) * 255.0
            img = F.interpolate(img, size=(VDVAE_RES, VDVAE_RES), mode="bilinear", align_corners=False)
            img = img.permute(0, 2, 3, 1).contiguous()
            gt_imgs.append(img.cpu().numpy())
    return np.clip(np.concatenate(gt_imgs, axis=0), 0, 255).astype(np.uint8)


def dev_pred_mmse(subj, n_dev, index_dir, betas_dir, weights_dir, dev_mask_dir, rng):
    """predict_ridge_shared1000.py's process_subject(), trimmed to the dev split + z_low +
    stopping before restandardize: returns (pred_mmse, tgt_mean, tgt_std, img73k_ids)."""
    betas, img73k, is_shared = load_nsdcore_split(subj, index_dir, betas_dir)
    dev_mask_73k = np.load(Path(dev_mask_dir) / f"nsdcore_dev_mask_subj{subj:02d}.npy")
    test_mask = dev_mask_73k[img73k]
    assert not np.any(test_mask & is_shared), "dev mask overlaps shared1000"
    train_mask = ~is_shared & ~test_mask
    train_fmri, test_fmri = zscore_fmri(betas, train_mask, test_mask)
    del betas
    test_img73k = img73k[test_mask]

    subj_weights_dir = weights_dir / f"subj{subj:02d}"
    stats = np.load(subj_weights_dir / "train_target_stats.npz")
    tgt_mean, tgt_std = stats[f"{BRANCH}_mean"], stats[f"{BRANCH}_std"]

    W, b = load_weights(subj_weights_dir / f"{BRANCH}_weights.pkl")
    raw = predict_flat(test_fmri, W, b)
    pred_mmse, uniq_ids = average_by_image(raw, test_img73k)  # (n_dev_images, D)

    if n_dev is not None and len(uniq_ids) > n_dev:
        sel = np.sort(rng.choice(len(uniq_ids), size=n_dev, replace=False))
        pred_mmse, uniq_ids = pred_mmse[sel], uniq_ids[sel]
    return pred_mmse.astype(np.float64), tgt_mean, tgt_std, uniq_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--n-dev", type=int, default=500, help="cap on dev images (subsampled if more)")
    ap.add_argument("--index-dir", type=str, default=INDEX_DIR)
    ap.add_argument("--betas-dir", type=str, default=BETAS_DIR)
    ap.add_argument("--weights-dir", type=str, default=WEIGHTS_DIR)
    ap.add_argument("--dev-mask-dir", type=str, default=DEV_MASK_DIR)
    ap.add_argument("--pca-dir", type=str, default=PCA_DIR)
    ap.add_argument("--images-h5", type=str, default=IMAGES_H5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    print("Loading VDVAE (ImageNet64, EMA weights) ...")
    H = dotdict(dict(H_DICT))
    H, _ = set_up_data(H)
    ema_vae = load_vaes(H)
    basis = load_pca_basis(args.pca_dir)

    results = {}
    n_win_both, n_subj = 0, 0
    for subj in args.subj:
        rng = np.random.default_rng(args.seed)
        pred_mmse, tgt_mean, tgt_std, ids = dev_pred_mmse(
            subj, args.n_dev, Path(args.index_dir), Path(args.betas_dir),
            Path(args.weights_dir), Path(args.dev_mask_dir), rng)
        print(f"\n=== subj0{subj}: {len(ids)} dev images ===")

        resid_var = estimate_resid_var(pred_mmse, tgt_std)
        frac_recovered = 1.0 - resid_var.sum() / (tgt_std.astype(np.float64) ** 2).sum()
        print(f"  resid_var: mean={resid_var.mean():.4g}  "
              f"fraction of target variance already in pred_mmse={frac_recovered:.4f}")

        pool_mean, pool_std = pred_mmse.mean(axis=0), pred_mmse.std(axis=0)
        pred_control = restandardize(pred_mmse, pool_mean, pool_std, tgt_mean, tgt_std)
        pred_treatment = sample_residual(pred_mmse, resid_var, np.random.default_rng(args.seed + 1))

        gt = load_gt_64(ids, args.images_h5, DEVICE)
        recon_control = decode_zlow_batch(pred_control, basis, ema_vae, DEVICE)
        recon_treatment = decode_zlow_batch(pred_treatment, basis, ema_vae, DEVICE)

        pc_c, pc_t = pixcorr(recon_control, gt), pixcorr(recon_treatment, gt)
        ss_c, ss_t = ssim_global(recon_control, gt), ssim_global(recon_treatment, gt)

        win_pixcorr = bool(pc_t.mean() >= pc_c.mean())
        win_ssim = bool(ss_t.mean() >= ss_c.mean())
        win_both = win_pixcorr and win_ssim
        n_win_both += int(win_both)
        n_subj += 1

        print(f"  PixCorr  control(restandardize)={pc_c.mean():.4f}  treatment(sample_residual)={pc_t.mean():.4f}  "
              f"-> {'win' if win_pixcorr else 'lose'}")
        print(f"  SSIM(g)  control(restandardize)={ss_c.mean():.4f}  treatment(sample_residual)={ss_t.mean():.4f}  "
              f"-> {'win' if win_ssim else 'lose'}")

        results[f"subj{subj:02d}"] = {
            "n_dev": int(len(ids)),
            "resid_var_mean": float(resid_var.mean()),
            "fraction_variance_already_in_pred_mmse": float(frac_recovered),
            "pixcorr_control": float(pc_c.mean()),
            "pixcorr_treatment": float(pc_t.mean()),
            "ssim_control": float(ss_c.mean()),
            "ssim_treatment": float(ss_t.mean()),
            "win_pixcorr": win_pixcorr,
            "win_ssim": win_ssim,
            "win_both": win_both,
        }

    gate_pass = n_win_both >= 3
    results["gate"] = {
        "n_subj_win_both_metrics": n_win_both,
        "n_subj": n_subj,
        "criterion": "treatment (sample_residual) must not lose on PixCorr AND SSIM together on "
                     ">=3/4 subjects",
        "proceed_to_gpu_branch": gate_pass,
    }
    print(f"\nGate: sample_residual wins on both metrics on {n_win_both}/{n_subj} subjects -> "
          f"{'PROCEED to GPU branch' if gate_pass else 'DO NOT PROCEED'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
