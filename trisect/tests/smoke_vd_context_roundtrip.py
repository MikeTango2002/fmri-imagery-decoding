"""Smoke test 2 of 3: does the manual `generate_from_context()` loop of `vd_context.py` reproduce the
pipeline's own `__call__` when fed the true context that `__call__` would compute internally?

Inputs:  one NSD/COCO image and caption (same as test 1).
Outputs: the reconstruction from the true 257x768 / 77x768 context, as a PNG.

The loop is split out of `__call__` so that a predicted context (from the adapter) can replace the
output of `encode_true_context()` without touching the generation code. If the PNG differs from the
one of test 1 at the same image and seed, `generate_from_context()` has a bug.

Usage (from the trisect/ folder; needs a GPU; run after test 1, same --image-index and --seed):
    python tests/smoke_vd_pipeline.py --image-index 42 --seed 0
    python tests/smoke_vd_context_roundtrip.py --image-index 42 --seed 0
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from vd_context import encode_true_context, generate_from_context, load_pipeline
from smoke_vd_pipeline import ANNOTS_PATH, IMAGES_H5, OUT_DIR, load_image_and_caption


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

    prompt_embeds, image_embeddings = encode_true_context(pipe, pil_image, caption, device, do_cfg=True)
    print(f"true context shapes: prompt_embeds {tuple(prompt_embeds.shape)}, "
          f"image_embeddings {tuple(image_embeddings.shape)}")

    out = generate_from_context(
        pipe,
        prompt_embeds,
        image_embeddings,
        text_to_image_strength=args.text_to_image_strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )[0]

    grid = Image.new("RGB", (224 * 2, 224))
    grid.paste(pil_image, (0, 0))
    grid.paste(out.resize((224, 224)), (224, 0))
    out_path = os.path.join(OUT_DIR, f"roundtrip_idx{args.image_index}_seed{args.seed}.png")
    grid.save(out_path)
    print(f"Done. Saved {out_path} (left: input, right: manual-loop reconstruction).")

    smoke_path = os.path.join(OUT_DIR, f"smoke_idx{args.image_index}_seed{args.seed}.png")
    if os.path.exists(smoke_path):
        smoke_grid = np.asarray(Image.open(smoke_path), dtype=np.float32)
        roundtrip_grid = np.asarray(grid, dtype=np.float32)
        smoke_out = smoke_grid[:, 224:]
        roundtrip_out = roundtrip_grid[:, 224:]
        mad = np.abs(smoke_out - roundtrip_out).mean()
        print(f"mean abs pixel diff vs {smoke_path}: {mad:.3f} (0-255 scale; expect near 0)")
    else:
        print(f"no {smoke_path} found -- run smoke_vd_pipeline.py with the same --image-index/--seed "
              f"first to get a numeric diff, not just a visual one")


if __name__ == "__main__":
    main()
