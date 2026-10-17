"""True Versatile Diffusion context (prompt_embeds, image_embeddings) of the 300 proxy-set images.

Inputs:  data/proxy_set/{images/, captions.npy}
Outputs: data/targets/a_psi_true_context_proxyset.h5, flat 0-299 (no id lookup):
         prompt_embeds (300, 77, 768) and image_embeddings (300, 257, 768), float16

This is the context of the T0 arm (Section 8.2), read by `reconstruct.py --dataset proxy`. It is the
extraction of extract_vd_context.py (do_cfg=False, one real caption per image), driven by files on
disk because the proxy set has no 73k id. Order: 0-149 simple, 150-299 complex.

Usage (GPU needed, after caption_proxy_set.py):
    python extract_proxy_vd_context.py
"""
import argparse
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from trisect.targets import load_proxy_images
from vd_context import load_pipeline

PROXY_DIR_DEFAULT = "data/proxy_set"
CAPTIONS_DEFAULT = "data/proxy_set/captions.npy"
OUT_DEFAULT = "data/targets/a_psi_true_context_proxyset.h5"
N_SIMPLE = 150
N_COMPLEX = 150
BATCH_SIZE = 16


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-dir", default=PROXY_DIR_DEFAULT)
    ap.add_argument("--captions", default=CAPTIONS_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    captions = np.load(args.captions, allow_pickle=True)
    images = load_proxy_images(args.proxy_dir)
    n = len(images)
    assert len(captions) == n, f"{len(captions)} captions vs {n} images"
    print(f"{n} proxy images ({N_SIMPLE} simple + {N_COMPLEX} complex), captions from {args.captions}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(device=device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with h5py.File(args.out, "w") as out_f:
        out_f.attrs["order"] = f"0-{N_SIMPLE - 1}=simple, {N_SIMPLE}-{n - 1}=complex"
        prompt_ds = out_f.create_dataset("prompt_embeds", shape=(n, 77, 768), dtype="float16")
        image_ds = out_f.create_dataset("image_embeddings", shape=(n, 257, 768), dtype="float16")

        for start in tqdm(range(0, n, BATCH_SIZE), desc="VD true context (proxy set)"):
            end = min(start + BATCH_SIZE, n)
            batch_imgs = images[start:end]
            batch_caps = [str(c) for c in captions[start:end]]

            with torch.no_grad():
                prompt_embeds = pipe._encode_text_prompt(batch_caps, device, 1, False)      # (B,77,768)
                image_embeddings = pipe._encode_image_prompt(batch_imgs, device, 1, False)  # (B,257,768)

            prompt_ds[start:end] = prompt_embeds.cpu().numpy().astype("float16")
            image_ds[start:end] = image_embeddings.cpu().numpy().astype("float16")

    print(f"Done. Wrote {args.out}")


if __name__ == "__main__":
    main()
