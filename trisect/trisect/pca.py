"""The three frozen PCA bases (z_sem / z_struct / z_low) and the VDVAE layer layout.

Outputs of `pca_width_zsem.py`, `fit_pca_zstruct.py` and `fit_pca_zlow.py`. Only the z_low basis
carries a layer layout (`layer_dims`/`dim_scale`); the other two are plain PCA over a flat vector
and use `layout=False` by default.
"""
import os

import numpy as np

VDVAE_ZDIM = 16  # H_DICT["zdim"] of the imagenet64 VDVAE


def layer_bounds(layer_dims, expected_total=None):
    """Start/end offsets of each VDVAE layer inside the concatenated latent vector.

    `expected_total` reproduces the assert of `fit_pca_zlow.py` (IN_DIM=91168);
    `decode_zlow_to_image.py` calls the function without it. Passing None skips the check.
    """
    bounds = []
    start = 0
    for d in layer_dims:
        bounds.append((start, start + d))
        start += d
    if expected_total is not None:
        assert start == expected_total, f"layer_dims sum to {start}, expected {expected_total}"
    return bounds


def layer_shapes(layer_dims, zdim=VDVAE_ZDIM):
    """(zdim, side, side) of each VDVAE layer, recovered from its flat dimension."""
    shapes = []
    for d in layer_dims:
        hw = d // zdim
        side = int(round(hw ** 0.5))
        assert side * side * zdim == d, f"layer dim {d} doesn't factor as {zdim}xSIDExSIDE"
        shapes.append((zdim, side, side))
    return shapes


def load_pca_basis(pca_dir, dtype=np.float32, layout=False, zdim=VDVAE_ZDIM):
    """Load a frozen `pca_basis.npz`.

    dtype: float32 for projection (`extract_proxy_targets.py`), float64 for the inverse
    transform (`decode_zlow_to_image.py`, the reconstruction path).
    layout=True: z_low only. Adds dim_scale/layer_dims/shapes/bounds and asserts whiten=True,
    which the inverse formula in decode_zlow_batch() assumes (the default of `fit_pca_zlow.py`).
    """
    basis = np.load(os.path.join(pca_dir, "pca_basis.npz"))
    whiten = bool(basis["whiten"])
    out = {
        "components": basis["components"].astype(dtype),
        "mean": basis["mean"].astype(dtype),
        "explained_variance": basis["explained_variance"].astype(dtype),
        "whiten": whiten,
    }
    if not layout:
        return out
    assert whiten, "inverse formula assumes whiten=True (fit_pca_zlow.py's default)"
    layer_dims = basis["layer_dims"].tolist()
    out.update({
        "dim_scale": basis["dim_scale"].astype(dtype),
        "layer_dims": layer_dims,
        "shapes": layer_shapes(layer_dims, zdim=zdim),
        "bounds": layer_bounds(layer_dims),
    })
    return out
