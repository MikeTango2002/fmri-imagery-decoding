"""Draw the 2,000-image caption sample and run the first comparison: one human COCO caption against
BLIP-2 OPT 2.7B (greedy), scored by CLIP text-to-image retrieval (Section 5.2).

Inputs:  COCO images and human annotations, raw CLIP image embeddings
Outputs: data/targets/zsem_txt_compare/{sample_indices.npy, captions.json, results.json}

compare_zsem_txt_blip2_large.py reads the sample. The human side takes ONE caption per image,
whereas the real targets average all of them; these numbers are not the ones of Section 5.2
(see this folder's README). No training, no brain data.

Usage:
    python compare_zsem_txt_sources.py
"""
import json
import os
import random

import h5py
import numpy as np
import torch
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from trisect.targets import l2_normalize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_H5 = "data/mindeyev2_dataset/coco_images_224_float16.hdf5"
ANNOTS_PATH = "data/annots/COCO_73k_annots_curated.npy"
CLIP_IMG_RAW_PATH = "data/targets/clip_vitl14_image_raw.npy"
CLIP_CKPT = "openai/clip-vit-large-patch14"
BLIP2_CKPT = "Salesforce/blip2-opt-2.7b"
OUT_DIR = "data/targets/zsem_txt_compare"
N_SAMPLE = 2000
SEED = 0
BLIP2_BATCH_SIZE = 8


def sample_indices(annots):
    has_caption = (annots != "").any(axis=1)
    valid = np.where(has_caption)[0]
    rng = random.Random(SEED)
    idx = rng.sample(list(valid), N_SAMPLE)
    idx.sort()
    return np.array(idx)


def first_nonempty_caption(row):
    for c in row:
        if c.strip():
            return c.strip()
    raise ValueError("no non-empty caption in row")


def generate_blip2_captions(images_u8):
    """images_u8: (N, 224, 224, 3) uint8 RGB. Returns list[str]."""
    print(f"Loading {BLIP2_CKPT} on {DEVICE} (fp16) ...")
    processor = Blip2Processor.from_pretrained(BLIP2_CKPT)
    model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_CKPT, torch_dtype=torch.float16).to(DEVICE)
    model.eval()

    captions = []
    n = images_u8.shape[0]
    with torch.no_grad():
        for start in range(0, n, BLIP2_BATCH_SIZE):
            batch = images_u8[start:start + BLIP2_BATCH_SIZE]
            pil_imgs = [Image.fromarray(im) for im in batch]
            inputs = processor(images=pil_imgs, return_tensors="pt").to(DEVICE, torch.float16)
            out = model.generate(**inputs, max_new_tokens=30)
            texts = processor.batch_decode(out, skip_special_tokens=True)
            captions.extend([t.strip() for t in texts])
            if start % (BLIP2_BATCH_SIZE * 20) == 0:
                print(f"  BLIP-2 captioning: {start}/{n}")
    del model
    torch.cuda.empty_cache()
    return captions


def embed_captions(captions):
    """Raw (unnormalized) CLIP text_embeds, 768-dim, same checkpoint as the image stream."""
    print(f"Embedding {len(captions)} captions with {CLIP_CKPT} text encoder ...")
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_CKPT)
    model = CLIPTextModelWithProjection.from_pretrained(CLIP_CKPT, torch_dtype=torch.float32).to(DEVICE)
    model.eval()

    embeds = []
    with torch.no_grad():
        for start in range(0, len(captions), 64):
            batch = captions[start:start + 64]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEVICE)
            out = model(**inputs)
            embeds.append(out.text_embeds.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(embeds, axis=0).astype(np.float32)


def retrieval_at_k(query, gallery, ks=(1, 5, 10)):
    """query[i] should retrieve gallery[i]. Both pre-normalized. In-sample retrieval (no held-out
    distractors beyond the sample itself), i.e. chance level is k/N."""
    sim = query @ gallery.T  # (N, N)
    n = sim.shape[0]
    ranks = np.argsort(-sim, axis=1)
    target = np.arange(n)[:, None]
    hit_rank = (ranks == target).argmax(axis=1)  # position of the correct match, 0-indexed
    out = {}
    for k in ks:
        out[f"R@{k}"] = float((hit_rank < k).mean())
    out["median_rank"] = int(np.median(hit_rank))
    return out


def mean_pairwise_cosine(x, n_pairs_sample=200000):
    """Mean off-diagonal cosine similarity, subsampled if the full matrix would be wasteful."""
    n = x.shape[0]
    rng = np.random.default_rng(SEED)
    i = rng.integers(0, n, size=n_pairs_sample)
    j = rng.integers(0, n, size=n_pairs_sample)
    mask = i != j
    i, j = i[mask], j[mask]
    sims = np.sum(x[i] * x[j], axis=1)
    return float(sims.mean())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading COCO human annotations ...")
    annots = np.load(ANNOTS_PATH, allow_pickle=True)
    idx = sample_indices(annots)
    print(f"Sampled {len(idx)} indices (seed={SEED}) out of {annots.shape[0]}")

    human_captions = [first_nonempty_caption(annots[i]) for i in idx]

    print("Loading sampled images from HDF5 ...")
    with h5py.File(IMAGES_H5, "r") as f:
        imgs_f16 = f["images"][idx]  # (N, 3, 224, 224) float16 in [0,1]
    imgs_u8 = (imgs_f16.astype(np.float32).transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)

    blip2_captions = generate_blip2_captions(imgs_u8)

    with open(os.path.join(OUT_DIR, "sample_indices.npy"), "wb") as f:
        np.save(f, idx)
    with open(os.path.join(OUT_DIR, "captions.json"), "w") as f:
        json.dump(
            [{"idx": int(i), "human": h, "blip2": b} for i, h, b in zip(idx, human_captions, blip2_captions)],
            f, indent=2,
        )

    human_embeds = l2_normalize(embed_captions(human_captions))
    blip2_embeds = l2_normalize(embed_captions(blip2_captions))

    img_raw = np.load(CLIP_IMG_RAW_PATH, mmap_mode="r")
    img_embeds = l2_normalize(np.asarray(img_raw[idx]))

    results = {
        "n_sample": len(idx),
        "seed": SEED,
        "human": {
            "txt_to_img_retrieval": retrieval_at_k(human_embeds, img_embeds),
            "img_to_txt_retrieval": retrieval_at_k(img_embeds, human_embeds),
            "mean_cosine_to_paired_image": float(np.mean(np.sum(human_embeds * img_embeds, axis=1))),
            "mean_pairwise_caption_cosine": mean_pairwise_cosine(human_embeds),
        },
        "blip2": {
            "txt_to_img_retrieval": retrieval_at_k(blip2_embeds, img_embeds),
            "img_to_txt_retrieval": retrieval_at_k(img_embeds, blip2_embeds),
            "mean_cosine_to_paired_image": float(np.mean(np.sum(blip2_embeds * img_embeds, axis=1))),
            "mean_pairwise_caption_cosine": mean_pairwise_cosine(blip2_embeds),
        },
    }

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_DIR}/{{results.json, captions.json, sample_indices.npy}}")


if __name__ == "__main__":
    main()
