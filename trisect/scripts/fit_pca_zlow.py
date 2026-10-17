"""Fit the PCA basis for z_low: one frozen PCA over the 31-layer VDVAE latents of `extract_vdvae_latents.py`.

Inputs:  data/targets/vdvae_latents_31l.npy (+ `_layer_dims.json`), shared1000_mask
Outputs: data/targets/pca_zlow/ (pca_basis.npz, z_low.npy); `--out-suffix` writes a sibling directory

Adapted from `fit_pca_zstruct.py`, plus a standardization per hierarchical layer before the PCA: the 31 layers
differ hugely in width and scale, and a plain PCA would let the wide fine layers dominate. The scale is one
scalar per layer, not per dimension, because the coarsest layers have near-collapsed posteriors that a
per-dimension scale would blow up to unit variance. shared1000 is excluded; the fit uses 10000 random
non-shared images, then the basis is applied to all 73000. d_low (1024) was chosen on the dev split via --d-out.
--per-layer: 31 IncrementalPCAs, then a global top-d_out by eigenvalue (same saved format, block-sparse).
--exclude-mask also drops a boolean mask (e.g. the dev masks) from the fit pool.

Usage (from the trisect/ folder):
    python scripts/fit_pca_zlow.py --per-layer --exclude-mask data/targets/nsdcore_dev_mask_union.npy
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

from trisect.pca import layer_bounds as _layer_bounds

RAW_PATH = "data/targets/vdvae_latents_31l.npy"            # (N, 91168) float16, memmap-loaded
LAYER_DIMS_PATH = "data/targets/vdvae_latents_31l_layer_dims.json"
SHARED_MASK_PATH = "data/targets/shared1000_mask.npy"
IN_DIM = 91168
D_OUT_DEFAULT = 1024      # overridable via --d-out for the sweep
N_FIT_IMAGES = 10000      # random subsample of non-shared images used to fit the basis
TRANSFORM_BATCH = 500     # images per streaming transform batch (91168-dim rows, keep this small)
SEED = 0


def layer_bounds(layer_dims):
    """trisect.pca.layer_bounds() with this script's IN_DIM assert kept in place."""
    return _layer_bounds(layer_dims, expected_total=IN_DIM)


def per_layer_scale(data, bounds):
    """One std scalar per layer, pooled over all rows and all dims of that layer."""
    scale = np.empty(len(bounds), dtype=np.float64)
    for i, (a, b) in enumerate(bounds):
        scale[i] = data[:, a:b].astype(np.float64).std()
    return np.maximum(scale, 1e-6)  # guard against a degenerate all-constant layer


def dim_scale_vector(layer_scale, bounds):
    """Expand the 31 per-layer scalars into a length-91168 vector for elementwise broadcast."""
    out = np.empty(IN_DIM, dtype=np.float32)
    for scale, (a, b) in zip(layer_scale, bounds):
        out[a:b] = scale
    return out


def fit_per_layer(raw, eligible_idx, layer_dims, bounds, d_out, layer_cap, batch_size, seed):
    """31 independent IncrementalPCA fits (one per layer), streamed over ALL of eligible_idx (no
    RAM-bound subsampling needed), then a single global top-d_out selection by eigenvalue pooled
    across layers. Returns the same fields as the global-PCA path below, in the same block-sparse
    layout; the save/transform code after this function is shared between both paths.
    """
    n_layers = len(bounds)
    caps = [min(dim, layer_cap) for dim in layer_dims]
    print(f"Per-layer caps (min(dim, {layer_cap})): {caps}")

    # per-layer standardization scale, estimated from a subsample of the eligible pool (cheap,
    # same approximation as the global path's per_layer_scale on N_FIT_IMAGES)
    rng = np.random.default_rng(seed)
    scale_idx = np.sort(rng.choice(eligible_idx, size=min(N_FIT_IMAGES, len(eligible_idx)),
                                    replace=False))
    scale_data = raw[scale_idx].astype(np.float32)
    layer_scale = per_layer_scale(scale_data, bounds)
    dim_scale = dim_scale_vector(layer_scale, bounds)
    del scale_data
    print(f"  layer_scale: min={layer_scale.min():.4g} max={layer_scale.max():.4g}")

    shuffled = rng.permutation(eligible_idx)
    n_batches = len(shuffled) // batch_size  # drop the incomplete remainder (IncrementalPCA
                                              # requires every batch >= max(caps))
    used = n_batches * batch_size
    print(f"Streaming {used}/{len(shuffled)} images in {n_batches} batches of {batch_size} "
          f"(batch_size must be >= max cap = {max(caps)}) ...")
    assert batch_size >= max(caps), "batch_size must be >= the largest per-layer component cap"

    ipcas = [IncrementalPCA(n_components=caps[i], whiten=True) for i in range(n_layers)]
    t0 = time.time()
    for bi in range(n_batches):
        idx_batch = np.sort(shuffled[bi * batch_size:(bi + 1) * batch_size])
        batch = raw[idx_batch].astype(np.float32) / dim_scale  # (batch_size, 91168)
        for i, (a, b) in enumerate(bounds):
            ipcas[i].partial_fit(batch[:, a:b])
        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            print(f"  batch {bi + 1}/{n_batches} ({elapsed:.0f}s elapsed, "
                  f"{elapsed / (bi + 1):.1f}s/batch)")

    # pool (layer, comp_idx, eigenvalue) across all layers, pick the global top d_out
    pooled = []
    for i in range(n_layers):
        for c in range(caps[i]):
            pooled.append((ipcas[i].explained_variance_[c], i, c))
    pooled.sort(key=lambda t: t[0], reverse=True)
    assert len(pooled) >= d_out, \
        f"only {len(pooled)} pooled components available, need d_out={d_out} " \
        f"(raise --layer-cap)"
    selected = pooled[:d_out]
    selected.sort(key=lambda t: (t[1], t[2]))  # stable order: by layer then component index

    components = np.zeros((d_out, IN_DIM), dtype=np.float32)
    explained_variance = np.empty(d_out, dtype=np.float32)
    mean = np.empty(IN_DIM, dtype=np.float32)
    for i, (a, b) in enumerate(bounds):
        mean[a:b] = ipcas[i].mean_.astype(np.float32)
    layer_total_var = np.array([ipcas[i].var_.sum() for i in range(n_layers)], dtype=np.float64)
    for row, (eigval, layer_i, comp_i) in enumerate(selected):
        a, b = bounds[layer_i]
        components[row, a:b] = ipcas[layer_i].components_[comp_i].astype(np.float32)
        explained_variance[row] = eigval

    cum_var = float(explained_variance.astype(np.float64).sum() / layer_total_var.sum())
    per_layer_selected_count = np.bincount([layer_i for _, layer_i, _ in selected],
                                            minlength=n_layers)
    print(f"  components selected per layer (layer_dims={layer_dims}): "
          f"{per_layer_selected_count.tolist()}")
    print(f"  cumulative explained variance ratio (global, out-of-subsample-only for the scale "
          f"estimate but in-fit for the covariance itself): {cum_var:.4f}")

    return {
        "components": components, "mean": mean, "explained_variance": explained_variance,
        "dim_scale": dim_scale, "layer_scale": layer_scale, "cum_var": cum_var,
        "fit_n_images": used, "per_layer_selected_count": per_layer_selected_count.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-out", type=int, default=D_OUT_DEFAULT,
                     help="PCA components for z_low (default 1024, a first-attempt budget guess)")
    ap.add_argument("--no-whiten", action="store_true",
                     help="fit PCA with whiten=False instead of the default True -- writes to "
                          "a '_nowhiten'-suffixed directory, never touches the whitened outputs")
    ap.add_argument("--per-layer", action="store_true",
                     help="31 independent per-layer IncrementalPCA fits + global top-d_out "
                          "selection by eigenvalue, instead of one flat PCA on the concatenation "
                          "(see module docstring)")
    ap.add_argument("--layer-cap", type=int, default=256,
                     help="--per-layer only: max components kept per layer before the global "
                          "selection (must be >= what any single layer could win); also sets the "
                          "minimum streaming batch size")
    ap.add_argument("--batch-size", type=int, default=512,
                     help="--per-layer only: images per streaming partial_fit batch")
    ap.add_argument("--exclude-mask", type=str, default=None,
                     help="path to an additional (73000,) bool npy mask to drop from the fit "
                          "pool, on top of shared1000 (e.g. nsdcore_dev_mask_union.npy, so a "
                          "held-out comparison is fair)")
    ap.add_argument("--out-suffix", type=str, default=None,
                     help="extra suffix for the output directory name, to avoid overwriting a "
                          "differently-configured run (e.g. 'devheld' for an --exclude-mask run)")
    args = ap.parse_args()
    d_out = args.d_out
    whiten = not args.no_whiten

    base_name = "pca_zlow" if whiten else "pca_zlow_nowhiten"
    if args.per_layer:
        base_name += "_perlayer"
    out_dir = f"data/targets/{base_name}"
    if d_out != D_OUT_DEFAULT:
        out_dir += f"_d{d_out}"
    if args.out_suffix:
        out_dir += f"_{args.out_suffix}"

    with open(LAYER_DIMS_PATH) as f:
        layer_meta = json.load(f)
    layer_dims = layer_meta["layer_dims"]
    bounds = layer_bounds(layer_dims)
    print(f"31-layer VDVAE latents, layer_dims={layer_dims}")

    raw = np.load(RAW_PATH, mmap_mode="r")   # (N, 91168), never loaded whole
    n = raw.shape[0]
    assert raw.shape[1] == IN_DIM, f"unexpected shape {raw.shape}"
    print(f"Raw VDVAE latents (memmap): {raw.shape}")

    shared_mask = np.load(SHARED_MASK_PATH)
    assert shared_mask.shape[0] == n, f"mask has {shared_mask.shape[0]} entries, raw has {n}"
    non_shared_idx = np.flatnonzero(~shared_mask)
    print(f"{len(non_shared_idx)}/{n} non-shared images available for the fit subsample")

    if args.exclude_mask:
        extra_excl = np.load(args.exclude_mask)
        assert extra_excl.shape[0] == n, f"exclude-mask has {extra_excl.shape[0]} entries"
        before = len(non_shared_idx)
        non_shared_idx = non_shared_idx[~extra_excl[non_shared_idx]]
        print(f"--exclude-mask {args.exclude_mask}: {before - len(non_shared_idx)} further "
              f"excluded, {len(non_shared_idx)} remain eligible")

    if args.per_layer:
        result = fit_per_layer(raw, non_shared_idx, layer_dims, bounds, d_out,
                                args.layer_cap, args.batch_size, SEED)
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            os.path.join(out_dir, "pca_basis.npz"),
            components=result["components"], mean=result["mean"],
            explained_variance=result["explained_variance"],
            layer_dims=np.array(layer_dims, dtype=np.int64),
            layer_scale=result["layer_scale"].astype(np.float32),
            dim_scale=result["dim_scale"],
            whiten=whiten, fit_n_images=result["fit_n_images"],
            fit_excludes_shared1000=True,
        )
        out_path = os.path.join(out_dir, "z_low.npy")
        print(f"Projecting all {n} images -> {out_path} ({n}, {d_out}) float16 ...")
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype="float16", shape=(n, d_out))
        components, mean, dim_scale = result["components"], result["mean"], result["dim_scale"]
        ev_sqrt = np.sqrt(result["explained_variance"]).astype(np.float64)
        for start in range(0, n, TRANSFORM_BATCH):
            end = min(start + TRANSFORM_BATCH, n)
            batch = (raw[start:end].astype(np.float64) / dim_scale) - mean
            out[start:end] = (batch @ components.T.astype(np.float64) / ev_sqrt).astype(np.float16)
        out.flush()
        del out

        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump({
                "d_out": d_out, "whiten": whiten, "per_layer": True,
                "layer_cap": args.layer_cap, "exclude_mask": args.exclude_mask,
                "fit_n_images": result["fit_n_images"], "fit_excludes_shared1000": True,
                "cumulative_explained_variance_ratio": result["cum_var"],
                "per_layer_selected_count": result["per_layer_selected_count"],
            }, f, indent=2)
        print("Done.")
        return

    rng = np.random.default_rng(SEED)
    fit_idx = np.sort(rng.choice(non_shared_idx, size=min(N_FIT_IMAGES, len(non_shared_idx)),
                                  replace=False))
    print(f"Loading {len(fit_idx)} images for the PCA fit ...")
    fit_data = raw[fit_idx].astype(np.float32)   # (n_fit, 91168)
    print(f"  fit matrix (raw): {fit_data.shape}, {fit_data.nbytes / 1e9:.2f} GB")

    print("Computing per-layer standardization scale (fit subsample only) ...")
    layer_scale = per_layer_scale(fit_data, bounds)
    dim_scale = dim_scale_vector(layer_scale, bounds)
    print(f"  layer_scale: min={layer_scale.min():.4g} max={layer_scale.max():.4g}")
    fit_data /= dim_scale  # in-place, broadcasts (91168,) over (n_fit, 91168)

    print(f"Fitting PCA d={d_out}, whiten={whiten} on standardized latents ...")
    pca = PCA(n_components=d_out, whiten=whiten, random_state=SEED)
    pca.fit(fit_data)
    cum_var = float(np.sum(pca.explained_variance_ratio_))
    print(f"  cumulative explained variance ratio: {cum_var:.4f}")

    pre_whiten_var = pca.explained_variance_.astype(np.float64)
    post_transform_var = np.var(pca.transform(fit_data).astype(np.float64), axis=0)
    del fit_data

    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "pca_basis.npz"),
        components=pca.components_.astype(np.float32),        # (d_out, 91168)
        mean=pca.mean_.astype(np.float32),                     # (91168,), in standardized space
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        layer_dims=np.array(layer_dims, dtype=np.int64),
        layer_scale=layer_scale.astype(np.float32),            # (31,)
        dim_scale=dim_scale,                                   # (91168,), layer_scale expanded
        whiten=whiten,
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

    out_path = os.path.join(out_dir, "z_low.npy")
    print(f"Projecting all {n} images -> {out_path} ({n}, {d_out}) float16 ...")
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype="float16", shape=(n, d_out))
    for start in range(0, n, TRANSFORM_BATCH):
        end = min(start + TRANSFORM_BATCH, n)
        batch = raw[start:end].astype(np.float32) / dim_scale   # (b, 91168)
        out[start:end] = pca.transform(batch).astype(np.float16)
    out.flush()
    del out

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "d_out": d_out,
            "whiten": whiten,
            "exclude_mask": args.exclude_mask,
            "fit_n_images": int(len(fit_idx)),
            "fit_excludes_shared1000": True,
            "cumulative_explained_variance_ratio": cum_var,
            "post_transform_variance_min": float(post_transform_var.min()),
            "post_transform_variance_max": float(post_transform_var.max()),
        }, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
