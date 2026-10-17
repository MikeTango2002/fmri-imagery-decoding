"""Extract the TRUE Versatile Diffusion context (prompt_embeds, image_embeddings) for a training subset
of NSD/COCO images: the regression target of the adapter (`adapter_model.py`, `train_adapter.py`).

Inputs:  coco_images_224_float16.hdf5 and one caption per image (BLIP-2 cache or human)
Outputs: a_psi_true_context.h5 (blip2) or a_psi_true_context_humantxt.h5 (human)

The full 73k target would be ~29 GB in fp16; a subset is used: 6000 images from the 72000
non-shared pool (shared1000 is kept for the visual comparisons). The context is extracted without
CFG doubling: the adapter predicts the positive half, and the fixed negative half is reattached at
generation time. --caption-source must match the source of the z_sem^txt that `train_adapter.py`
takes as input. With a mismatch the text cosine collapses (~0.29 instead of ~0.868) while the MSE
loss keeps falling; the loss alone does not catch it. image_embeddings never depends on the caption.

Usage (from the trisect/ folder; needs a GPU):
    python scripts/extract_vd_context.py --caption-source blip2 --n-train 2000 --n-val 100
"""
import argparse
import os
import random

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from vd_context import load_pipeline

IMAGES_H5 = "data/mindeyev2_dataset/coco_images_224_float16.hdf5"
ANNOTS_PATH = "data/annots/COCO_73k_annots_curated.npy"
SHARED1000_MASK_PATH = "data/targets/shared1000_mask.npy"
BLIP2_CAPTIONS_DEFAULT = "data/targets/blip2_captions_full.npy"
BATCH_SIZE = 16
SEED = 0


def out_path(caption_source):
    return "data/targets/a_psi_true_context.h5" if caption_source == "blip2" \
        else "data/targets/a_psi_true_context_humantxt.h5"


def first_nonempty_caption(row):
    for c in row:
        if c:
            return str(c)
    return None


def load_blip2_captions(captions_file):
    if not os.path.exists(captions_file):
        raise FileNotFoundError(
            f"{captions_file} not found -- run extract_blip2_captions.py first "
            "(or pass --caption-source human)"
        )
    return np.load(captions_file, allow_pickle=True)


def sample_indices(n_train, n_val):
    shared_mask = np.load(SHARED1000_MASK_PATH)
    annots = np.load(ANNOTS_PATH, allow_pickle=True)
    has_caption = (annots != "").any(axis=1)
    pool = np.where((~shared_mask) & has_caption)[0]

    rng = random.Random(SEED)
    chosen = rng.sample(list(pool), n_train + n_val)
    train_idx = np.array(sorted(chosen[:n_train]))
    val_idx = np.array(sorted(chosen[n_train:]))
    return train_idx, val_idx, annots


def load_batch_images_captions(indices, caption_source, annots, blip2_captions, images_h5):
    pil_images = []
    captions = []
    for idx in indices:
        img = images_h5["images"][idx]  # (3,224,224) float16 in [0,1]
        img = (np.asarray(img, dtype=np.float32).transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img, mode="RGB"))
        if caption_source == "blip2":
            captions.append(str(blip2_captions[idx]))
        else:
            captions.append(first_nonempty_caption(annots[idx]))
    return pil_images, captions


def extract_split(pipe, device, indices, caption_source, annots, blip2_captions, images_h5, out_group):
    n = len(indices)
    prompt_ds = out_group.create_dataset("prompt_embeds", shape=(n, 77, 768), dtype="float16")
    image_ds = out_group.create_dataset("image_embeddings", shape=(n, 257, 768), dtype="float16")
    out_group.create_dataset("image_indices_73k", data=indices)

    for start in tqdm(range(0, n, BATCH_SIZE), desc=f"VD true context ({out_group.name})"):
        batch_idx = indices[start : start + BATCH_SIZE]
        pil_images, captions = load_batch_images_captions(
            batch_idx, caption_source, annots, blip2_captions, images_h5
        )

        with torch.no_grad():
            prompt_embeds = pipe._encode_text_prompt(captions, device, 1, False)  # (B,77,768)
            image_embeddings = pipe._encode_image_prompt(pil_images, device, 1, False)  # (B,257,768)

        prompt_ds[start : start + len(batch_idx)] = prompt_embeds.cpu().numpy().astype("float16")
        image_ds[start : start + len(batch_idx)] = image_embeddings.cpu().numpy().astype("float16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--caption-source", choices=["human", "blip2"], default="blip2")
    ap.add_argument("--captions-file", default=BLIP2_CAPTIONS_DEFAULT,
                     help="only used for --caption-source blip2; one cached caption per image, "
                          "ordered by 73k index (extract_blip2_captions.py's output)")
    ap.add_argument("--out", type=str, default=None,
                     help="defaults to a_psi_true_context.h5 (blip2) / "
                          "a_psi_true_context_humantxt.h5 (human)")
    args = ap.parse_args()
    out = args.out or out_path(args.caption_source)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    train_idx, val_idx, annots = sample_indices(args.n_train, args.n_val)
    print(f"train: {len(train_idx)} images, val: {len(val_idx)} images, both non-shared1000")

    blip2_captions = load_blip2_captions(args.captions_file) if args.caption_source == "blip2" else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(device=device)

    with h5py.File(IMAGES_H5, "r") as images_h5, h5py.File(out, "w") as out_f:
        out_f.attrs["seed"] = SEED
        out_f.attrs["caption_policy"] = "blip2_single_caption" if args.caption_source == "blip2" \
            else "first_nonempty"
        extract_split(pipe, device, train_idx, args.caption_source, annots, blip2_captions,
                      images_h5, out_f.create_group("train"))
        extract_split(pipe, device, val_idx, args.caption_source, annots, blip2_captions,
                      images_h5, out_f.create_group("val"))

    print(f"Done. Wrote {out}")


if __name__ == "__main__":
    main()
