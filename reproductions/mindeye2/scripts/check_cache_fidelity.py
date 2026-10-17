#!/usr/bin/env python
# coding: utf-8
"""Check frozen_cache.hdf5 directly against a fresh recomputation of the frozen encoders.

Inputs:  the cache (--cache_dir) and the stimuli (--data_path); a sample of --n images
Outputs: per part (clip, vae, cnx), the cache-vs-recompute difference next to a noise floor, on screen

Two 2-epoch loss trajectories cannot show whether the cache changes the training: the CLIP term divides by
temp=0.004 (250x amplification of fp16 differences) and the runs do not share an RNG state. So this asks directly
whether the stored tensors are the ones the encoders produce. The control runs the same encoders at the batch
size of the cache (--ref-batch, 32) and of training (--alt-batch, 8): cuBLAS picks other kernels per shape;
recompute-vs-recompute is the noise floor. A defect is cache-vs-recompute well above it; at or below it is fp16.

Usage (from src/, venv active, one GPU):
    python check_cache_fidelity.py --data_path=mindeyev2_dataset --cache_dir=mindeyev2_dataset --n=128
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append("generative_models/")
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder  # noqa: E402

from frozen_cache import PARTS  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True  # same as Train.py / precompute_frozen.py

parser = argparse.ArgumentParser(description="Check frozen_cache.hdf5 against a recomputation")
parser.add_argument("--data_path", type=str, default=os.getcwd())
parser.add_argument("--cache_dir", type=str, default=os.getcwd())
parser.add_argument("--cache", type=str, default=None,
                    help="path to frozen_cache.hdf5 (default: {data_path}/frozen_cache.hdf5)")
parser.add_argument("--parts", type=str, default="clip,vae,cnx")
parser.add_argument("--n", type=int, default=128, help="how many images to sample")
parser.add_argument("--ref-batch", type=int, default=32, help="batch the cache was built with")
parser.add_argument("--alt-batch", type=int, default=8, help="batch training runs at")
parser.add_argument("--clip-temp", type=float, default=0.004,
                    help="lowest soft_clip_loss temperature (see soft_loss_temps in Train.py)")
args = parser.parse_args()

parts = [p.strip() for p in args.parts.split(",") if p.strip()]
assert all(p in PARTS for p in parts), f"--parts must be a subset of {list(PARTS)}"
cache_path = args.cache or os.path.join(args.data_path, "frozen_cache.hdf5")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- stimuli
images_path = os.path.join(args.data_path, "coco_images_224_float16.hdf5")
img_f = h5py.File(images_path, "r")
images = img_f["images"]
cache = h5py.File(cache_path, "r")

# Spread the sample over the whole file: a contiguous block would only exercise one shard
# of the precompute run, and h5py needs the index list sorted and unique anyway.
num_images = images.shape[0]
idx = np.unique(np.linspace(0, num_images - 1, args.n).astype(np.int64))
print(f"{cache_path}\n{images_path}: {images.shape}")
print(f"sampling {len(idx)} images spread over [0, {num_images})\n")

for p in parts:
    done = cache.get(f"done_{p}")
    if done is not None and not np.all(done[:][idx]):
        missing = int(len(idx) - np.sum(done[:][idx]))
        print(f"WARNING: {p}: {missing}/{len(idx)} sampled images are not marked done")

# ---------------------------------------------------------------- encoders (as in Train.py)
clip_img_embedder = autoenc = cnx = None
mean = std = None

if "clip" in parts:
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14", version="laion2b_s39b_b160k", output_tokens=True, only_tokens=True)
    clip_img_embedder.to(device)

if "vae" in parts:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    autoenc.load_state_dict(torch.load(f"{args.cache_dir}/sd_image_var_autoenc.pth"))
    autoenc.eval().requires_grad_(False).to(device)

if "cnx" in parts:
    from autoencoder.convnext import ConvnextXL
    cnx = ConvnextXL(f"{args.cache_dir}/convnext_xlarge_alpha0.75_fullckpt.pth")
    cnx.eval().requires_grad_(False).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.228, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)


def recompute(part, batch_size):
    """Run one encoder over the sampled images, exactly as precompute_frozen.py does."""
    outs = []
    for i in range(0, len(idx), batch_size):
        chunk = idx[i:i + batch_size]
        image = torch.tensor(images[chunk], dtype=torch.float16).float().to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            if part == "clip":
                y = clip_img_embedder(image)
            elif part == "vae":
                y = autoenc.encode(2 * image - 1).latent_dist.mode() * 0.18215
            else:
                _, y = cnx((image - mean) / std)
        outs.append(y.float().cpu())
    return torch.cat(outs)


def diff(a, b):
    """Absolute and scale-relative discrepancy. Relative is against the RMS of the
    reference, not per element: these tensors pass through zero; an elementwise
    relative error is dominated by entries that carry no signal."""
    d = (a - b).abs()
    rms = a.pow(2).mean().sqrt().item()
    return d.max().item(), d.max().item() / max(rms, 1e-12)


print(f"{'part':<6}{'comparison':<26}{'max|d|':>12}{'max|d|/rms':>13}{'worst 1-cos':>14}")
report = {}
for p in parts:
    ref = recompute(p, args.ref_batch)                        # cache's own batch size
    alt = recompute(p, args.alt_batch)                        # training's batch size
    cached = torch.tensor(np.asarray(cache[PARTS[p][0]][idx]), dtype=torch.float32)

    for label, other in ((f"cache vs recomp@{args.ref_batch}", cached),
                         (f"recomp@{args.alt_batch} vs @{args.ref_batch}", alt)):
        a, r = diff(ref, other)
        cos = 1 - F.cosine_similarity(ref.flatten(1), other.flatten(1), dim=-1).min().item()
        print(f"{p:<6}{label:<26}{a:>12.3e}{r:>13.3e}{cos:>14.3e}")
        report[(p, label)] = r
    print()

    # For clip only: propagate the discrepancy into the quantity the loss actually sees.
    # soft_clip_loss works on l2-normalised flattened targets and divides the Gram matrix
    # by temp; a target perturbation is amplified by 1/temp before the softmax.
    if p == "clip":
        def gram(x):
            return F.normalize(x.flatten(1), dim=-1) @ F.normalize(x.flatten(1), dim=-1).T
        g_ref = gram(ref)
        for label, other in ((f"cache vs recomp@{args.ref_batch}", cached),
                             (f"recomp@{args.alt_batch} vs @{args.ref_batch}", alt)):
            dg = (g_ref - gram(other)).abs().max().item()
            print(f"      logit shift, {label}: max|dG| {dg:.3e} "
                  f"-> {dg / args.clip_temp:.3e} at temp={args.clip_temp}")
        print()

cache.close()
img_f.close()

# ---------------------------------------------------------------- verdict
print("=" * 78)
worst_cache = max(v for (p, l), v in report.items() if l.startswith("cache"))
worst_noise = max(v for (p, l), v in report.items() if l.startswith("recomp"))
print(f"worst cache-vs-recompute : {worst_cache:.3e}")
print(f"worst recompute-vs-recompute (numerical noise floor): {worst_noise:.3e}")
if worst_cache <= max(worst_noise * 4, 1e-3):
    print("VERDICT: OK -- the cache sits at the noise floor of the encoders themselves.")
    print("A percent-level gap between two loss trajectories is chaos, not a wrong cache")
    print("(see the header of this file). Safe to launch the pretraining.")
else:
    print("VERDICT: FAIL -- the cache differs from a fresh recomputation by much more than")
    print("two recomputations differ from each other. This IS a cache defect: do not train")
    print("on it. Suspect a stale/partial hdf5 (check the done_* bitmaps) or an encoder")
    print("built with different arguments than precompute_frozen.py uses.")
    sys.exit(1)
