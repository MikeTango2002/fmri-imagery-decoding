"""Smoke test 1 of 3: does `VersatileDiffusionDualGuidedPipeline` reconstruct a recognisable image?

Inputs:  one NSD/COCO image and one of its human captions.
Outputs: input and output side by side, for a visual check.

Runs Versatile Diffusion's public dual-guided `__call__` unmodified (test 2 covers the custom path).
It only shows that the tool works and gives a first read of its baseline quality at the default
text_to_image_strength and guidance_scale. It does not validate the adapter-based reconstruction.

Usage (from the trisect/ folder; needs a GPU):
    python tests/smoke_vd_pipeline.py --image-index 42 --seed 0
"""
import argparse
import os

import h5py
import numpy as np
import torch
from PIL import Image

from vd_context import load_pipeline

IMAGES_H5 = "data/mindeyev2_dataset/coco_images_224_float16.hdf5"
ANNOTS_PATH = "data/annots/COCO_73k_annots_curated.npy"
OUT_DIR = "data/gate0_smoke"


def load_image_and_caption(idx):
    with h5py.File(IMAGES_H5, "r") as f:
        img = f["images"][idx]  # (3,224,224) float16 in [0,1]
    img = (np.asarray(img, dtype=np.float32).transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(img, mode="RGB")

    annots = np.load(ANNOTS_PATH, allow_pickle=True)  # (73000,5) <U250, '' for missing
    captions = [c for c in annots[idx] if c]
    if not captions:
        raise ValueError(f"image {idx} has no non-empty human caption")
    return pil_image, captions[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-index", type=int, default=42)
    ap.add_argument("--text-to-image-strength", type=float, default=0.75)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    pil_image, caption = load_image_and_caption(args.image_index)
    print(f"image {args.image_index}, caption: {caption!r}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    out = pipe(
        prompt=caption,
        image=pil_image,
        text_to_image_strength=args.text_to_image_strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    ).images[0]

    grid = Image.new("RGB", (224 * 2, 224))
    grid.paste(pil_image, (0, 0))
    grid.paste(out.resize((224, 224)), (224, 0))
    out_path = os.path.join(OUT_DIR, f"smoke_idx{args.image_index}_seed{args.seed}.png")
    grid.save(out_path)
    print(f"Done. Saved {out_path} (left: input, right: VD reconstruction).")


if __name__ == "__main__":
    main()
