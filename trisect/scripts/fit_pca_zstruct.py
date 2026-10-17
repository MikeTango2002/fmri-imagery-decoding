"""Fit the structural PCA: one frozen basis shared by all 64 spatial positions, DINOv2's 1024 channels down to 32.
z_struct = P_struct . phi_vis(I), in R^{8x8x32}.

Inputs:  data/targets/dinov2_vitl14_patch_pooled8x8.npy (`extract_dinov2_pooled.py`), shared1000_mask
Outputs: data/targets/pca_zstruct/ (pca_basis.npz, z_struct.npy); pca_zstruct_d{D}/ for D != 32 and a
         `_nowhiten` suffix for --no-whiten; a sweep never overwrites the default basis

PCA and not a learned 1x1 convolution; the bottleneck stays a frozen subspace. One basis for all positions
gives 64x more samples per fit, and DINOv2 patch features are roughly translation-equivariant. No L2
normalization: the structural loss is a literal L2 distance on the 8x8x32 tensor. shared1000 is excluded. The
fit uses 18000 random non-shared images (~4.7 GB, not ~18.9 GB); the basis is then applied to all 73000.
--d-out sweeps the width (d=32 explains 35.66% of variance); --no-whiten reuses the same fit.
Usage (from the trisect/ folder):
    python scripts/fit_pca_zstruct.py --d-out 64
"""
import argparse
import json
import os

import numpy as np
from sklearn.decomposition import PCA

RAW_PATH = "data/targets/dinov2_vitl14_patch_pooled8x8.npy"   # (N, 8, 8, 1024) float32, memmap-loaded
SHARED_MASK_PATH = "data/targets/shared1000_mask.npy"
GRID = 8
IN_DIM = 1024
D_OUT_DEFAULT = 32       # fixed by design (8x8x32 = 2048); overridable via --d-out for the sweep
N_FIT_IMAGES = 18000      # random subsample of non-shared images used to fit the basis
TRANSFORM_BATCH = 1000    # images per streaming transform batch
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-out", type=int, default=D_OUT_DEFAULT,
                     help="PCA components per spatial position (default 32, the spec value)")
    ap.add_argument("--no-whiten", action="store_true",
                     help="fit PCA with whiten=False instead of the default True -- writes to "
                          "a '_nowhiten'-suffixed directory, never touches the whitened outputs")
    args = ap.parse_args()
    d_out = args.d_out
    whiten = not args.no_whiten

    base_name = "pca_zstruct" if whiten else "pca_zstruct_nowhiten"
    out_dir = f"data/targets/{base_name}" if d_out == D_OUT_DEFAULT \
        else f"data/targets/{base_name}_d{d_out}"

    raw = np.load(RAW_PATH, mmap_mode="r")   # (N, 8, 8, 1024), never loaded whole
    n = raw.shape[0]
    assert raw.shape[1:] == (GRID, GRID, IN_DIM), f"unexpected shape {raw.shape}"
    print(f"Pooled DINOv2 features (memmap): {raw.shape}")

    shared_mask = np.load(SHARED_MASK_PATH)
    assert shared_mask.shape[0] == n, f"mask has {shared_mask.shape[0]} entries, raw has {n}"
    non_shared_idx = np.flatnonzero(~shared_mask)
    print(f"{len(non_shared_idx)}/{n} non-shared images available for the fit subsample")

    rng = np.random.default_rng(SEED)
    fit_idx = np.sort(rng.choice(non_shared_idx, size=min(N_FIT_IMAGES, len(non_shared_idx)),
                                  replace=False))
    print(f"Loading {len(fit_idx)} images ({len(fit_idx) * GRID * GRID} position-vectors) "
          f"for the PCA fit ...")
    fit_data = raw[fit_idx].reshape(-1, IN_DIM).astype(np.float32)   # (n_fit*64, 1024)
    print(f"  fit matrix: {fit_data.shape}, {fit_data.nbytes / 1e9:.2f} GB")

    print(f"Fitting shared PCA d={d_out}, whiten={whiten} ...")
    pca = PCA(n_components=d_out, whiten=whiten, random_state=SEED)
    pca.fit(fit_data)
    cum_var = float(np.sum(pca.explained_variance_ratio_))
    print(f"  cumulative explained variance ratio: {cum_var:.4f}")

    # Per-dimension variance profile:
    # BEFORE whitening (pca.explained_variance_, decreasing) vs AFTER (measured on the fit
    # matrix's own transform, exactly 1.0 per dim iff whiten=True).
    pre_whiten_var = pca.explained_variance_.astype(np.float64)
    post_transform_var = np.var(pca.transform(fit_data).astype(np.float64), axis=0)
    del fit_data

    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "pca_basis.npz"),
        components=pca.components_.astype(np.float32),        # (32, 1024)
        mean=pca.mean_.astype(np.float32),                     # (1024,)
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        whiten=whiten,
        shared_across_positions=True,
        fit_n_images=len(fit_idx),
        fit_excludes_shared1000=True,
    )
    np.savez(
        os.path.join(out_dir, "variance_profile.npz"),
        pre_whiten_variance=pre_whiten_var.astype(np.float32),
        post_transform_variance=post_transform_var.astype(np.float32),
    )
    print(f"  post-transform per-dim variance: min={post_transform_var.min():.4f} "
          f"max={post_transform_var.max():.4f} (expect ~1.0 for all dims iff whiten=True)")

    out_path = os.path.join(out_dir, "z_struct.npy")
    print(f"Projecting all {n} images -> {out_path} ({n}, {GRID}, {GRID}, {d_out}) float16 ...")
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype="float16",
                                     shape=(n, GRID, GRID, d_out))
    for start in range(0, n, TRANSFORM_BATCH):
        end = min(start + TRANSFORM_BATCH, n)
        batch = raw[start:end].reshape(-1, IN_DIM).astype(np.float32)   # (b*64, 1024)
        proj = pca.transform(batch).astype(np.float16)                  # (b*64, d_out)
        out[start:end] = proj.reshape(end - start, GRID, GRID, d_out)
    out.flush()
    del out

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "d_out": d_out,
            "whiten": whiten,
            "shared_across_positions": True,
            "fit_n_images": int(len(fit_idx)),
            "fit_excludes_shared1000": True,
            "cumulative_explained_variance_ratio": cum_var,
            "post_transform_variance_min": float(post_transform_var.min()),
            "post_transform_variance_max": float(post_transform_var.max()),
        }, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
