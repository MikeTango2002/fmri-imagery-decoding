"""BLIP-2 flan-t5-xxl captions (beam 5) against the image embeddings: CLIP text-to-image retrieval.

Inputs:  data/targets/zsem_txt_compare/sample_indices.npy (from compare_zsem_txt_sources.py),
         COCO images, raw CLIP image embeddings
Outputs: data/targets/zsem_txt_compare/{captions,results}_<checkpoint>_beam<N>.json

Same 2,000-image sample as the first comparison; the numbers compare directly. Only the BLIP-2
side is computed here; the human side is described in this folder's README. No training, no brain
data, but a GPU that fits the checkpoint.

Usage:
    python compare_zsem_txt_blip2_large.py                  # blip2-flan-t5-xxl, beam 5
    python compare_zsem_txt_blip2_large.py --num-beams 1    # greedy decoding
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor, CLIPTextModelWithProjection, CLIPTokenizer
from trisect.targets import l2_normalize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_H5 = "data/mindeyev2_dataset/coco_images_224_float16.hdf5"
CLIP_IMG_RAW_PATH = "data/targets/clip_vitl14_image_raw.npy"
SAMPLE_INDICES_PATH = "data/targets/zsem_txt_compare/sample_indices.npy"
CLIP_CKPT = "openai/clip-vit-large-patch14"
OUT_DIR = "data/targets/zsem_txt_compare"


def generate_blip2_captions(images_u8, ckpt, batch_size, num_beams):
    """images_u8: (N, 224, 224, 3) uint8 RGB. Returns list[str].

    bfloat16, not float16: flan-t5 backbones are known to overflow/NaN in fp16
    (https://github.com/huggingface/transformers/issues/20287), Ada supports bf16 natively.
    """
    print(f"Loading {ckpt} on {DEVICE} (bf16) ...")
    processor = Blip2Processor.from_pretrained(ckpt)
    model = Blip2ForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to(DEVICE)
    model.eval()

    captions = []
    n = images_u8.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = images_u8[start:start + batch_size]
            pil_imgs = [Image.fromarray(im) for im in batch]
            inputs = processor(images=pil_imgs, return_tensors="pt").to(DEVICE, torch.bfloat16)
            out = model.generate(**inputs, max_new_tokens=30, num_beams=num_beams, do_sample=False)
            texts = processor.batch_decode(out, skip_special_tokens=True)
            captions.extend([t.strip() for t in texts])
            if start % (batch_size * 20) == 0:
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


def mean_pairwise_cosine(x, n_pairs_sample=200000, seed=0):
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs_sample)
    j = rng.integers(0, n, size=n_pairs_sample)
    mask = i != j
    i, j = i[mask], j[mask]
    sims = np.sum(x[i] * x[j], axis=1)
    return float(sims.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="Salesforce/blip2-flan-t5-xxl")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    idx = np.load(SAMPLE_INDICES_PATH)
    print(f"Loaded {len(idx)} sample indices (same paired sample as the first run)")

    print("Loading sampled images from HDF5 ...")
    with h5py.File(IMAGES_H5, "r") as f:
        imgs_f16 = f["images"][idx]  # (N, 3, 224, 224) float16 in [0,1]
    imgs_u8 = (imgs_f16.astype(np.float32).transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)

    blip2_captions = generate_blip2_captions(imgs_u8, args.ckpt, args.batch_size, args.num_beams)

    ckpt_tag = args.ckpt.split("/")[-1].replace("-", "_")
    with open(os.path.join(OUT_DIR, f"captions_{ckpt_tag}_beam{args.num_beams}.json"), "w") as f:
        json.dump([{"idx": int(i), "blip2": b} for i, b in zip(idx, blip2_captions)], f, indent=2)

    blip2_embeds = l2_normalize(embed_captions(blip2_captions))

    img_raw = np.load(CLIP_IMG_RAW_PATH, mmap_mode="r")
    img_embeds = l2_normalize(np.asarray(img_raw[idx]))

    results = {
        "n_sample": len(idx),
        "ckpt": args.ckpt,
        "num_beams": args.num_beams,
        "batch_size": args.batch_size,
        "txt_to_img_retrieval": retrieval_at_k(blip2_embeds, img_embeds),
        "img_to_txt_retrieval": retrieval_at_k(img_embeds, blip2_embeds),
        "mean_cosine_to_paired_image": float(np.mean(np.sum(blip2_embeds * img_embeds, axis=1))),
        "mean_pairwise_caption_cosine": mean_pairwise_cosine(blip2_embeds),
    }

    out_path = os.path.join(OUT_DIR, f"results_{ckpt_tag}_beam{args.num_beams}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nWrote {out_path} and captions_{ckpt_tag}_beam{args.num_beams}.json")


if __name__ == "__main__":
    main()
