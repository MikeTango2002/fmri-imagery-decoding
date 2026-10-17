"""Extract raw (pre-PCA) CLIP ViT-L/14 pooled image embeddings for all 73k NSD/COCO images.

Inputs:  data/mindeyev2_dataset/coco_images_224_float16.hdf5, (73000, 3, 224, 224), already 224x224
Outputs: data/targets/clip_vitl14_image_raw.npy, (73000, 768) float32, deliberately un-normalized

This is phi_VL(I) of Section 5.2: z_sem = L2(P_sem @ phi_VL(I)); normalization comes after the PCA.
The checkpoint is openai/clip-vit-large-patch14 (`CLIPVisionModelWithProjection`), the CLIP encoder
Versatile Diffusion uses internally (Brain-Diffuser `versatile_diffusion/lib/model_zoo/clip.py`,
line 57); z_sem lives in the embedding family of the frozen generator.

Usage (from the trisect/ folder):
    python scripts/extract_clip_image_raw.py --limit 512      # smoke test
"""
import argparse
import os

import h5py
import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_H5 = "data/mindeyev2_dataset/coco_images_224_float16.hdf5"
OUT_PATH = "data/targets/clip_vitl14_image_raw.npy"
CKPT = "openai/clip-vit-large-patch14"
EMBED_DIM = 768
BATCH_SIZE = 256
NUM_WORKERS = 8

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


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

    print(f"Loading {CKPT} on {DEVICE} ...")
    model = CLIPVisionModelWithProjection.from_pretrained(CKPT, torch_dtype=torch.float32).to(DEVICE)
    model.eval()

    mean = CLIP_MEAN.to(DEVICE)
    std = CLIP_STD.to(DEVICE)

    n = len(dataset)
    print(f"Writing {n}x{EMBED_DIM} float32 raw CLIP embeddings to {args.out}")
    fp = open_memmap(args.out, mode="w+", dtype="float32", shape=(n, EMBED_DIM))

    with torch.no_grad():
        for imgs, idx in tqdm(loader, desc="CLIP ViT-L/14 image embeddings"):
            imgs = imgs.to(DEVICE, non_blocking=True)
            imgs = (imgs - mean) / std
            out = model(pixel_values=imgs)
            embeds = out.image_embeds  # (B, 768), deliberately not L2-normalized
            fp[idx.numpy()] = embeds.cpu().numpy().astype("float32")

    fp.flush()
    del fp
    print("Done.")


if __name__ == "__main__":
    main()
