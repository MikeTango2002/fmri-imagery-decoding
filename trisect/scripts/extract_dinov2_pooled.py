"""Extract DINOv2 ViT-L/14 patch tokens for all 73k NSD/COCO images, pooled 2x2 to an 8x8 grid.

Inputs:  data/mindeyev2_dataset/coco_images_224_float16.hdf5 (224x224 is also DINOv2's native size)
Outputs: data/targets/dinov2_vitl14_patch_pooled8x8.npy, (73000, 8, 8, 1024) float32, ~19 GB

This is the raw structural feature map that `fit_pca_zstruct.py` projects down to z_struct.
The native 16x16x1024 grid would take ~76 GB; the 2x2 average pool runs per batch inside the
forward loop. It is a fixed local operation; the result is bit-identical to pooling a saved
tensor in a second pass. Patch tokens: `model.forward_features(x)["x_norm_patchtokens"]` of
facebookresearch/dinov2 `dinov2_vitl14` (torch.hub, no registers). The output is streamed through an
on-disk memmap.

Usage (from the trisect/ folder; needs a GPU):
    python scripts/extract_dinov2_pooled.py --limit 512   # smoke test
"""
import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_H5 = "data/mindeyev2_dataset/coco_images_224_float16.hdf5"
OUT_PATH = "data/targets/dinov2_vitl14_patch_pooled8x8.npy"
HUB_REPO = "facebookresearch/dinov2"
HUB_MODEL = "dinov2_vitl14"
GRID_NATIVE = 16   # 224 / 14
GRID_POOLED = 8
EMBED_DIM = 1024
BATCH_SIZE = 128   # smaller than CLIP's 256: each sample writes 32x more floats (8x8x1024 vs 768)
NUM_WORKERS = 8

DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class NSDImagesDataset(Dataset):
    """Lazily reads (3,224,224) float16 images in [0,1] from the shared HDF5 file."""

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
    parser.add_argument("--out", type=str, default=OUT_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    dataset = NSDImagesDataset(IMAGES_H5, limit=args.limit)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True
    )

    print(f"Loading {HUB_REPO}:{HUB_MODEL} on {DEVICE} ...")
    model = torch.hub.load(HUB_REPO, HUB_MODEL).to(DEVICE)
    model.eval()

    mean = DINO_MEAN.to(DEVICE)
    std = DINO_STD.to(DEVICE)

    n = len(dataset)
    shape = (n, GRID_POOLED, GRID_POOLED, EMBED_DIM)
    print(f"Writing {shape} float32 pooled DINOv2 patch tokens to {args.out} "
          f"(~{np.prod(shape) * 4 / 1e9:.1f} GB)")
    fp = open_memmap(args.out, mode="w+", dtype="float32", shape=shape)

    with torch.no_grad():
        for imgs, idx in tqdm(loader, desc="DINOv2 ViT-L/14 patch tokens (pooled 8x8)"):
            imgs = imgs.to(DEVICE, non_blocking=True)
            imgs = (imgs - mean) / std
            feats = model.forward_features(imgs)["x_norm_patchtokens"]  # (B, 256, 1024)
            b = feats.shape[0]
            feats = feats.reshape(b, GRID_NATIVE, GRID_NATIVE, EMBED_DIM).permute(0, 3, 1, 2)  # (B,1024,16,16)
            pooled = F.avg_pool2d(feats, kernel_size=2, stride=2)                                # (B,1024,8,8)
            pooled = pooled.permute(0, 2, 3, 1).contiguous()                                     # (B,8,8,1024)
            fp[idx.numpy()] = pooled.cpu().numpy().astype("float32")

    fp.flush()
    del fp
    print("Done.")


if __name__ == "__main__":
    main()
