"""Reader for the precomputed frozen-encoder outputs used by Train.py.

Inputs:  the hdf5 built by precompute_frozen.py, keyed by COCO image index (behav[:,0,0])
Outputs: clip_target [256, 1664], image_enc [4, 28, 28] and cnx_embeds [49, 512] for a batch of images

With --no-use_image_aug (the paper config) these three tensors are deterministic functions of the stimulus, and
Train.py recomputes them at every one of the 150 epochs: ViT-bigG-14, the SD VAE mode*.18215 and ConvNeXt-XL, about
60% of the FLOPs of a step, with no trainable weight. Not cached: cnx(blur_augs(image)) (fresh randomness at every
step) and autoenc.decode (it acts on predicted latents). The recipe is unchanged; the values may differ from an
uncached run at fp16 rounding, because cuBLAS can pick another kernel per batch size (check_cache_fidelity.py).
"""

import os

import h5py
import numpy as np
import torch

# part name -> (hdf5 dataset name, per-image shape)
PARTS = {
    "clip": ("clip_target", (256, 1664)),
    "vae": ("image_enc", (4, 28, 28)),
    "cnx": ("cnx_embeds", (49, 512)),
}


class FrozenCache:
    """Read-only view over a frozen-encoder cache file.

    Opened once per process. Reads are done with a single sorted-unique hdf5 selection per
    call (h5py requires increasing indices), then un-permuted back to the caller's order.
    """

    def __init__(self, path, parts=None):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"frozen cache not found: {path} -- build it with precompute_frozen.py"
            )
        self.path = os.path.abspath(path)
        self.f = h5py.File(self.path, "r")
        wanted = list(PARTS) if parts is None else list(parts)
        self.available = {}
        for name in wanted:
            dset_name, shape = PARTS[name]
            if dset_name not in self.f:
                continue
            done = self.f.get(f"done_{name}")
            if done is not None and not np.all(done[:]):
                missing = int(len(done) - done[:].sum())
                raise ValueError(
                    f"frozen cache '{name}' is incomplete ({missing} images missing): "
                    f"finish precompute_frozen.py --parts={name} before training"
                )
            self.available[name] = self.f[dset_name]
        if not self.available:
            raise ValueError(f"frozen cache {self.path} contains none of {wanted}")

    def has(self, name):
        return name in self.available

    def get(self, name, idx):
        """Return the cached tensor for `idx` (LongTensor/array of COCO indices), CPU fp16."""
        dset = self.available[name]
        idx = np.asarray(idx.cpu() if torch.is_tensor(idx) else idx, dtype=np.int64)
        uniq, inverse = np.unique(idx, return_inverse=True)
        out = dset[uniq][inverse]
        return torch.from_numpy(np.ascontiguousarray(out))

    def close(self):
        self.f.close()

    def __repr__(self):
        parts = ", ".join(sorted(self.available))
        return f"FrozenCache({os.path.basename(self.path)}, parts=[{parts}])"
