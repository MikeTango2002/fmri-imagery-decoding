"""Extract VDVAE latents for all 73k NSD/COCO images: the raw material for z_low.

Inputs:  data/mindeyev2_dataset/coco_images_224_float16.hdf5; the EMA weights of OpenAI's ImageNet64
         "very deep VAE" (imagenet64-iter-1600000-model-ema.th, ~478 MB)
Outputs: data/targets/vdvae_latents_31l.npy, (73000, 91168) float16, streamed through a memmap, plus
         a `_layer_dims.json` sidecar with the flattened size of each of the 31 layers

Adapted from Brain-Diffuser's `scripts/vdvae_extract_features.py`: same model, hyperparameters and
forward pass (the posterior-sample z of the first 31 hierarchical layers, concatenated). Changed: the
image source, and the resize (224 -> 64, bilinear). `vdvae_lib/` is the vendored library; its only
addition is that `DecBlock` in `vae.py` also exposes the posterior mean `qm`. `fit_pca_zlow.py` reads
the layer sizes to standardize each layer before the PCA, because their scales differ hugely.

Usage (from the trisect/ folder; needs a GPU):
    python scripts/extract_vdvae_latents.py --limit 64   # smoke test, also checks VRAM
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vdvae_lib"))
from model_utils import load_vaes, set_up_data  # noqa: E402  (needs the sys.path insert above)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_H5 = "data/mindeyev2_dataset/coco_images_224_float16.hdf5"
EMA_WEIGHTS_PATH = "data/vdvae_weights/imagenet64-iter-1600000-model-ema.th"
OUT_PATH = "data/targets/vdvae_latents_31l.npy"
NUM_LATENTS = 31
NATIVE_RES = 224
VDVAE_RES = 64
BATCH_SIZE_DEFAULT = 32
NUM_WORKERS = 8

# Same H dict as Brain-Diffuser's `scripts/vdvae_extract_features.py`, with restore_ema_path
# pointing at the local copy of the weights.
H_DICT = {
    "image_size": 64, "image_channels": 3, "seed": 0, "port": 29500,
    "save_dir": "./saved_models/test", "data_root": "./", "desc": "test",
    "hparam_sets": "imagenet64", "restore_path": "unused", "restore_ema_path": EMA_WEIGHTS_PATH,
    "restore_log_path": "unused", "restore_optimizer_path": "unused", "dataset": "imagenet64",
    "ema_rate": 0.999, "enc_blocks": "64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5",
    "dec_blocks": "1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12",
    "zdim": 16, "width": 512, "custom_width_str": "", "bottleneck_multiple": 0.25,
    "no_bias_above": 64, "scale_encblock": False, "test_eval": True, "warmup_iters": 100,
    "num_mixtures": 10, "grad_clip": 220.0, "skip_threshold": 380.0, "lr": 0.00015,
    "lr_prior": 0.00015, "wd": 0.01, "wd_prior": 0.0, "num_epochs": 10000, "n_batch": 4,
    "adam_beta1": 0.9, "adam_beta2": 0.9, "temperature": 1.0, "iters_per_ckpt": 25000,
    "iters_per_print": 1000, "iters_per_save": 10000, "iters_per_images": 10000,
    "epochs_per_eval": 1, "epochs_per_probe": None, "epochs_per_eval_save": 1,
    "num_images_visualize": 8, "num_variables_visualize": 6, "num_temperatures_visualize": 3,
    "mpi_size": 1, "local_rank": 0, "rank": 0, "logdir": "./saved_models/test/log",
}


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class NSDImagesDataset(Dataset):
    """Lazily reads (3,224,224) float16 images in [0,1] from the shared HDF5 file, same
    source/indexing as extract_dinov2_pooled.py/extract_clip_image_raw.py."""

    def __init__(self, h5_path, limit=None):
        self.h5_path = h5_path
        self._file = None
        with h5py.File(h5_path, "r") as f:
            self.n = f["images"].shape[0]
        if limit is not None:
            self.n = min(self.n, limit)

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = self._get_file()["images"][idx]  # (3,224,224) float16 in [0,1]
        return torch.from_numpy(img.astype(np.float32)), idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="smoke-test on the first N images only")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--out", type=str, default=OUT_PATH)
    args = parser.parse_args()

    if not os.path.isfile(EMA_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"{EMA_WEIGHTS_PATH} not found -- download it first:\n"
            f"  curl -fSL -o {EMA_WEIGHTS_PATH} "
            f"https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/"
            f"imagenet64-iter-1600000-model-ema.th"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    dataset = NSDImagesDataset(IMAGES_H5, limit=args.limit)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True
    )

    print(f"Loading VDVAE (ImageNet64, EMA weights) on {DEVICE} ...")
    H = dotdict(dict(H_DICT))
    H, preprocess_fn = set_up_data(H)
    ema_vae = load_vaes(H)

    n = len(dataset)
    latent_dim = None  # determined from the first batch, then asserted constant
    layer_dims = None  # per-layer flattened size (31 entries), needed by fit_pca_zlow.py to
                        # standardize each hierarchical layer separately before the PCA fit
    fp = None

    with torch.no_grad():
        for imgs, idx in tqdm(loader, desc="VDVAE 31-layer latents"):
            # (B,3,224,224) in [0,1] -> (B,3,64,64) in [0,255] -> (B,64,64,3), the channel-last
            # layout Encoder.forward permutes back to NCHW internally (vdvae_lib/vae.py).
            imgs = imgs.to(DEVICE, non_blocking=True) * 255.0
            imgs = F.interpolate(imgs, size=(VDVAE_RES, VDVAE_RES), mode="bilinear", align_corners=False)
            imgs = imgs.permute(0, 2, 3, 1).contiguous()

            data_input, _ = preprocess_fn(imgs)
            activations = ema_vae.encoder.forward(data_input)
            # dec_blocks has 75 latent-producing blocks in total; brain_diffuser/scripts/
            # vdvae_extract_features.py:82-95 deliberately keeps only the first 31 (coarsest,
            # up to just past 16x16) and never asserts the total length, mirrored here.
            _, stats = ema_vae.decoder.forward(activations, get_latents=True)
            assert len(stats) >= NUM_LATENTS, f"expected at least {NUM_LATENTS} latent layers, got {len(stats)}"

            b = data_input.shape[0]
            per_layer = [stats[i]["z"].cpu().numpy().reshape(b, -1) for i in range(NUM_LATENTS)]
            batch_latent = np.hstack(per_layer)

            if fp is None:
                latent_dim = batch_latent.shape[1]
                layer_dims = [layer.shape[1] for layer in per_layer]
                assert sum(layer_dims) == latent_dim
                shape = (n, latent_dim)
                print(f"Writing {shape} float16 VDVAE latents to {args.out} "
                      f"(~{np.prod(shape) * 2 / 1e9:.1f} GB), layer_dims={layer_dims}")
                fp = open_memmap(args.out, mode="w+", dtype="float16", shape=shape)
                layer_dims_path = os.path.splitext(args.out)[0] + "_layer_dims.json"
                with open(layer_dims_path, "w") as jf:
                    json.dump({"num_latents": NUM_LATENTS, "layer_dims": layer_dims}, jf)
            else:
                assert batch_latent.shape[1] == latent_dim, (
                    f"latent dim changed mid-run: {batch_latent.shape[1]} != {latent_dim}"
                )

            fp[idx.numpy()] = batch_latent.astype("float16")

    fp.flush()
    del fp
    print("Done.")


if __name__ == "__main__":
    main()
