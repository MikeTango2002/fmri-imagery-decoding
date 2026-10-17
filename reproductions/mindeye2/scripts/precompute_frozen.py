#!/usr/bin/env python
# coding: utf-8
"""Precompute the deterministic frozen-encoder outputs that Train.py consumes, once, into an hdf5.

Inputs:  the 73k NSD stimuli (--data_path) and the encoder weights (--cache_dir)
Outputs: --out, keyed by COCO image index: clip_target [256, 1664] fp16 (~62 GB), image_enc [4, 28, 28] (~0.5 GB),
         cnx_embeds [49, 512] (~3.7 GB)

Train.py reads them with --frozen_cache instead of recomputing them every epoch (frozen_cache.py explains why the
model is unchanged, check_cache_fidelity.py verifies it). The encoders are built with exactly the arguments of
Train.py and the inputs prepared the same way (fp16 pixels widened to fp32, under fp16 autocast): keep them in sync.
Resumable: a `done_<part>` bitmap is flushed after every chunk.

Usage (from src/, venv active, one GPU):
    python precompute_frozen.py --data_path=mindeyev2_dataset --cache_dir=mindeyev2_dataset \
        --out=mindeyev2_dataset/frozen_cache.hdf5 --parts=clip,vae,cnx
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.append("generative_models/")
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder  # noqa: E402

from frozen_cache import PARTS  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True  # same as Train.py

parser = argparse.ArgumentParser(description="Precompute frozen-encoder outputs for Train.py")
parser.add_argument("--data_path", type=str, default=os.getcwd(),
                    help="folder holding coco_images_224_float16.hdf5")
parser.add_argument("--cache_dir", type=str, default=os.getcwd(),
                    help="folder holding sd_image_var_autoenc.pth and convnext_xlarge_*.pth")
parser.add_argument("--out", type=str, default=None,
                    help="output hdf5 (default: {data_path}/frozen_cache.hdf5)")
parser.add_argument("--parts", type=str, default="clip,vae,cnx",
                    help="comma-separated subset of clip,vae,cnx")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--start", type=int, default=0, help="first image index (for sharding)")
parser.add_argument("--end", type=int, default=-1, help="last image index, exclusive (-1 = all)")
parser.add_argument("--redo", action="store_true", help="recompute even where done_* is set")
args = parser.parse_args()

parts = [p.strip() for p in args.parts.split(",") if p.strip()]
assert all(p in PARTS for p in parts), f"--parts must be a subset of {list(PARTS)}"
out_path = args.out or os.path.join(args.data_path, "frozen_cache.hdf5")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- stimuli
images_path = os.path.join(args.data_path, "coco_images_224_float16.hdf5")
img_f = h5py.File(images_path, "r")
images = img_f["images"]
num_images = images.shape[0]
end = num_images if args.end < 0 else min(args.end, num_images)
print(f"{images_path}: {images.shape} {images.dtype}; computing [{args.start}, {end})")

# ---------------------------------------------------------------- encoders (as in Train.py)
clip_img_embedder = autoenc = cnx = None

if "clip" in parts:
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder.to(device)

if "vae" in parts:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    autoenc.load_state_dict(torch.load(f"{args.cache_dir}/sd_image_var_autoenc.pth"))
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)

if "cnx" in parts:
    from autoencoder.convnext import ConvnextXL
    cnx = ConvnextXL(f"{args.cache_dir}/convnext_xlarge_alpha0.75_fullckpt.pth")
    cnx.requires_grad_(False)
    cnx.eval()
    cnx.to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1, 3, 1, 1)

# ---------------------------------------------------------------- output file
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
out = h5py.File(out_path, "a")
for p in parts:
    dset_name, shape = PARTS[p]
    if dset_name not in out:
        out.create_dataset(dset_name, shape=(num_images, *shape), dtype=np.float16,
                           chunks=(1, *shape))
        out.create_dataset(f"done_{p}", shape=(num_images,), dtype=np.uint8, fillvalue=0)
        print(f"created {dset_name} {(num_images, *shape)} "
              f"(~{np.prod(shape) * num_images * 2 / 1e9:.1f} GB)")
out.attrs["images_file"] = os.path.abspath(images_path)
out.attrs["num_images"] = num_images

# ---------------------------------------------------------------- main loop
t0 = time.time()
done_flags = {p: out[f"done_{p}"] for p in parts}
n_done = 0
for i0 in range(args.start, end, args.batch_size):
    i1 = min(i0 + args.batch_size, end)
    todo = [p for p in parts if args.redo or not np.all(done_flags[p][i0:i1])]
    if not todo:
        continue

    # Same pixel preparation as Train.py: fp16 from the hdf5, widened to fp32.
    image = torch.tensor(images[i0:i1], dtype=torch.float16).float().to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        if "clip" in todo:
            out[PARTS["clip"][0]][i0:i1] = clip_img_embedder(image).cpu().numpy()
            done_flags["clip"][i0:i1] = 1
        if "vae" in todo:
            image_enc = autoenc.encode(2 * image - 1).latent_dist.mode() * 0.18215
            out[PARTS["vae"][0]][i0:i1] = image_enc.cpu().numpy()
            done_flags["vae"][i0:i1] = 1
        if "cnx" in todo:
            _, cnx_embeds = cnx((image - mean) / std)
            out[PARTS["cnx"][0]][i0:i1] = cnx_embeds.cpu().numpy()
            done_flags["cnx"][i0:i1] = 1

    n_done += i1 - i0
    if (i0 // args.batch_size) % 50 == 0:
        out.flush()
        rate = n_done / max(time.time() - t0, 1e-6)
        eta = (end - i1) / max(rate, 1e-6) / 60
        print(f"{i1}/{end}  {rate:.1f} img/s  ETA {eta:.0f} min", flush=True)

out.flush()
for p in parts:
    total = int(np.sum(done_flags[p][:]))
    print(f"{p}: {total}/{num_images} images cached")
out.close()
img_f.close()
print(f"\n=== done in {(time.time() - t0) / 60:.1f} min -> {out_path} ===")
