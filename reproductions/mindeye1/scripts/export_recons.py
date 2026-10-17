"""Export the .pt tensors written by Reconstructions.py as viewable PNGs.

Inputs:  {model_name}_recons_img2img{s}_{n}samples.pt ([N,3,512,512], values in [0,1]) and
         all_images_subj0N.pt ([N,3,425,425] ground truth, same order)
Outputs: a ground-truth/reconstruction grid, plus the individual pairs with --all

The ground truth is resampled to the reconstruction size before pairing (Reconstruction_Metrics.py does its
own resize to 256 and is unaffected).

Usage:
    python export_recons.py --recon_path=<recons>.pt --all_images_path=all_images_subj01.pt \
        --out_dir=../evals/subj01 [--n=20] [--all]
"""

import argparse
import os

import torch
from torchvision.utils import make_grid, save_image

parser = argparse.ArgumentParser()
parser.add_argument("--recon_path", required=True)
parser.add_argument("--all_images_path", required=True)
parser.add_argument("--out_dir", default="../evals")
parser.add_argument("--n", type=int, default=20, help="how many samples in the grid")
parser.add_argument("--start", type=int, default=0, help="index of the first sample")
parser.add_argument("--all", action="store_true", help="also dump every pair as its own PNG")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

recons = torch.load(args.recon_path, map_location="cpu").float().clamp(0, 1)
images = torch.load(args.all_images_path, map_location="cpu").float().clamp(0, 1)
print(f"recons {tuple(recons.shape)} | ground truth {tuple(images.shape)}")

# torch.stack/make_grid need a common spatial size: 512 (VD output) vs 425 (NSD stimuli).
if images.shape[-2:] != recons.shape[-2:]:
    images = torch.nn.functional.interpolate(
        images, size=recons.shape[-2:], mode="bilinear", align_corners=False
    ).clamp(0, 1)
    print(f"resized ground truth to {tuple(images.shape)}")

sel = slice(args.start, args.start + args.n)
gt, rc = images[sel], recons[sel]

# Interleave GT and reconstruction so each column is a pair.
interleaved = torch.stack([gt, rc], dim=1).reshape(-1, *gt.shape[1:])
grid = make_grid(interleaved, nrow=10, padding=2)
grid_path = os.path.join(args.out_dir, f"grid_{args.start}_{args.start + args.n}.png")
save_image(grid, grid_path)
print(f"wrote {grid_path}")

if args.all:
    pairs_dir = os.path.join(args.out_dir, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)
    for i in range(len(recons)):
        save_image(torch.stack([images[i], recons[i]]),
                   os.path.join(pairs_dir, f"{i:04d}.png"), nrow=2, padding=2)
    print(f"wrote {len(recons)} pairs to {pairs_dir}")
