"""True Versatile Diffusion context of the 996 images of the shared1000 block set (T0 arm, Section 8.2).

Inputs:  data/shared1000_blockset/manifest.json, the 224-pixel COCO images, the BLIP-2 captions
Outputs: data/targets/a_psi_true_context_shared1000.h5, same schema as a_psi_true_context.h5 (a "train"
         group with all 996 rows, an empty "val" group)

extract_vd_context.py samples only the non-shared1000 pool; its file covers none of these ids.
The extraction is the same, and the id-keyed format is kept so that load_oracle_ctx() reads it unchanged:
pass it to reconstruct.py as `--a-psi-context`.

Usage (GPU needed, about 1 minute):
    python extract_shared1000_vd_context.py
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from vd_context import load_pipeline

IMAGES_H5 = "data/mindeyev2_dataset/coco_images_224_float16.hdf5"
MANIFEST_DEFAULT = "data/shared1000_blockset/manifest.json"
BLIP2_CAPTIONS_DEFAULT = "data/targets/blip2_captions_full.npy"
OUT_DEFAULT = "data/targets/a_psi_true_context_shared1000.h5"
BATCH_SIZE = 16


def load_shared1000_ids(manifest_path):
    manifest = json.load(open(manifest_path))
    ids = []
    for block in manifest["blocks"]:
        ids.extend(block["img73k_ids"])
    return np.array(sorted(set(ids)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=MANIFEST_DEFAULT)
    ap.add_argument("--captions-file", default=BLIP2_CAPTIONS_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    ids = load_shared1000_ids(args.manifest)
    n = len(ids)
    blip2_captions = np.load(args.captions_file, allow_pickle=True)
    print(f"{n} shared1000 images, captions from {args.captions_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(device=device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with h5py.File(IMAGES_H5, "r") as images_h5, h5py.File(args.out, "w") as out_f:
        out_f.attrs["source"] = "shared1000_blockset manifest, T0 arm"
        train_grp = out_f.create_group("train")
        prompt_ds = train_grp.create_dataset("prompt_embeds", shape=(n, 77, 768), dtype="float16")
        image_ds = train_grp.create_dataset("image_embeddings", shape=(n, 257, 768), dtype="float16")
        train_grp.create_dataset("image_indices_73k", data=ids)
        val_grp = out_f.create_group("val")
        val_grp.create_dataset("prompt_embeds", shape=(0, 77, 768), dtype="float16")
        val_grp.create_dataset("image_embeddings", shape=(0, 257, 768), dtype="float16")
        val_grp.create_dataset("image_indices_73k", data=np.array([], dtype=ids.dtype))

        for start in tqdm(range(0, n, BATCH_SIZE), desc="VD true context (shared1000)"):
            end = min(start + BATCH_SIZE, n)
            batch_ids = ids[start:end]
            pil_images = []
            from PIL import Image
            for idx in batch_ids:
                img = images_h5["images"][idx]  # (3,224,224) float16 in [0,1]
                img = (np.asarray(img, dtype=np.float32).transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img, mode="RGB"))
            captions = [str(blip2_captions[idx]) for idx in batch_ids]

            with torch.no_grad():
                prompt_embeds = pipe._encode_text_prompt(captions, device, 1, False)      # (B,77,768)
                image_embeddings = pipe._encode_image_prompt(pil_images, device, 1, False)  # (B,257,768)

            prompt_ds[start:end] = prompt_embeds.cpu().numpy().astype("float16")
            image_ds[start:end] = image_embeddings.cpu().numpy().astype("float16")

    print(f"Done. Wrote {args.out}")


if __name__ == "__main__":
    main()
