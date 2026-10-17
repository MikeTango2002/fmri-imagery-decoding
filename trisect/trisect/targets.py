"""Target-side helpers shared by the extract_* and comparison scripts.

`l2_normalize()` is the unit-norm guard used by `assemble_targets.py`, `extract_proxy_targets.py`,
`pca_width_zsem.py` and the caption-source comparisons. `load_proxy_images()` reads the proxy set in the flat
0-299 order that `captions.npy` and the proxy targets share.
"""
import os

import numpy as np


def l2_normalize(x, axis=-1, eps=1e-8):
    """Unit-norm along `axis`, guarding a zero-norm row.

    Shared by `assemble_targets.py`, `extract_proxy_targets.py`, `pca_width_zsem.py`,
    `compare_zsem_txt_sources.py` and `compare_zsem_txt_blip2_large.py`.
    """
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)


def load_proxy_images(proxy_dir, n_simple=150, n_complex=150):
    """Proxy-set images in flat 0-299 order: 0-149 simple, 150-299 complex.

    That order is the contract shared by captions.npy, a_psi_true_context_proxyset.h5 and
    vdvae_latents_proxy_raw.npy: do not sort or re-group. Returns a list of PIL images.
    Shared by `extract_proxy_targets.py` and `extract_proxy_vd_context.py`.
    """
    from PIL import Image

    simple = [Image.open(os.path.join(proxy_dir, "images", "simple", f"{i:03d}.png")).convert("RGB")
              for i in range(n_simple)]
    complex_ = [Image.open(os.path.join(proxy_dir, "images", "complex", f"{i:03d}.png")).convert("RGB")
                for i in range(n_complex)]
    return simple + complex_
