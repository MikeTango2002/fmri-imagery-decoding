"""Fit the whitened PCA of z_sem at widths 32 to 768 on the raw CLIP ViT-L/14 image embeddings (Section 5.2).

Inputs:  data/targets/clip_vitl14_image_raw.npy (N, 768), data/targets/shared1000_mask.npy
Outputs: data/targets/pca_zsem/d<D>/{z_sem_img.npy, pca_basis.npz, variance_profile.npz},
         data/targets/pca_zsem/ablation_summary.json

The basis is fitted on the 72,000 non-shared images and then applied to all 73,000; shared1000
never enters a preprocessing statistic. z_sem_img is L2-normalized after the projection. Whitening
keeps one high-variance direction from dominating that vector. Width 768 is the ceiling: the raw
embedding has 768 dimensions. --no-whiten is the control and writes to pca_zsem_nowhiten/.

Usage:
    python pca_width_zsem.py [--no-whiten]
"""
import argparse
import json
import os

import numpy as np
from sklearn.decomposition import PCA
from trisect.targets import l2_normalize

RAW_PATH = "data/targets/clip_vitl14_image_raw.npy"
SHARED_MASK_PATH = "data/targets/shared1000_mask.npy"
OUT_DIR_WHITEN = "data/targets/pca_zsem"
OUT_DIR_NOWHITEN = "data/targets/pca_zsem_nowhiten"
COMPONENT_COUNTS = [32, 64, 128, 256, 512, 768]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-whiten", action="store_true",
                     help="fit PCA with whiten=False instead of the default True -- writes to "
                          f"{OUT_DIR_NOWHITEN}/, never touches {OUT_DIR_WHITEN}/")
    args = ap.parse_args()
    whiten = not args.no_whiten
    out_dir = OUT_DIR_WHITEN if whiten else OUT_DIR_NOWHITEN

    raw = np.load(RAW_PATH)  # (N, 768) float32, un-normalized phi_VL(I)
    n, d_in = raw.shape
    print(f"Loaded raw CLIP embeddings: {raw.shape}")

    shared_mask = np.load(SHARED_MASK_PATH)
    assert shared_mask.shape[0] == n, f"mask has {shared_mask.shape[0]} entries, raw has {n}"
    fit_mask = ~shared_mask
    print(f"Fitting PCA (whiten={whiten}) on {fit_mask.sum()}/{n} non-shared images "
          f"(shared1000 excluded from the fit, transformed with the frozen basis)")

    os.makedirs(out_dir, exist_ok=True)
    summary = []

    for d in COMPONENT_COUNTS:
        print(f"\nFitting PCA d={d}, whiten={whiten} ...")
        pca = PCA(n_components=d, whiten=whiten, random_state=0)
        pca.fit(raw[fit_mask])              # fit: non-shared images only
        z = pca.transform(raw)              # transform: all 73k, frozen basis
        z_norm = l2_normalize(z).astype(np.float16)

        cum_var = float(np.sum(pca.explained_variance_ratio_))
        # Variance per dimension before whitening (decreasing) and after, measured on the transform:
        # exactly 1.0 when whiten=True. Recorded so that the two variants have numbers behind them.
        pre_whiten_var = pca.explained_variance_.astype(np.float64)
        post_transform_var = np.var(z[fit_mask].astype(np.float64), axis=0)

        d_dir = os.path.join(out_dir, f"d{d}")
        os.makedirs(d_dir, exist_ok=True)
        np.save(os.path.join(d_dir, "z_sem_img.npy"), z_norm)
        np.savez(
            os.path.join(d_dir, "pca_basis.npz"),
            components=pca.components_.astype(np.float32),
            mean=pca.mean_.astype(np.float32),
            explained_variance=pca.explained_variance_.astype(np.float32),
            explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
            whiten=whiten,
        )
        np.savez(
            os.path.join(d_dir, "variance_profile.npz"),
            pre_whiten_variance=pre_whiten_var.astype(np.float32),      # (d,), decreasing
            post_transform_variance=post_transform_var.astype(np.float32),  # (d,), ~1.0 iff whiten
        )

        summary.append(
            {
                "d": d,
                "cumulative_explained_variance_ratio": cum_var,
                "post_transform_variance_min": float(post_transform_var.min()),
                "post_transform_variance_max": float(post_transform_var.max()),
            }
        )
        print(f"  cumulative explained variance ratio: {cum_var:.4f}")
        print(f"  post-transform per-dim variance: min={post_transform_var.min():.4f} "
              f"max={post_transform_var.max():.4f} (expect ~1.0 for all dims iff whiten=True)")

    with open(os.path.join(out_dir, "ablation_summary.json"), "w") as f:
        json.dump({"whiten": whiten, "components": summary}, f, indent=2)

    print("\nSummary (d -> cumulative explained variance ratio):")
    for row in summary:
        print(f"  d={row['d']:>4}  cum_var={row['cumulative_explained_variance_ratio']:.4f}")


if __name__ == "__main__":
    main()
