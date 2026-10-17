"""Extract raw (pre-PCA) CLIP ViT-L/14 text embeddings for z_sem^txt, from one caption source.

Inputs:  COCO human captions (--source human) or BLIP-2 captions (--source blip2)
Outputs: data/targets/clip_vitl14_text_raw_{human,blip2}.npy, (73000, 768) float32, un-normalized

z_sem^txt = L2(P_sem @ CLIP_text(caption)); `assemble_targets.py` projects with the basis fit on the image
stream; no text PCA exists. The final model uses blip2: a captioner describes any image, including the
NSD-Imagery stimuli, which have no COCO captions (human captions retrieve better, Section 5.2). human: the
raw embeddings of all 1-5 captions are averaged per image, as in Brain-Diffuser's
`scripts/cliptext_extract_features.py` (line 46); never normalize before averaging. blip2: loads the cached
captions (--captions-file) if present; a rerun never regenerates; checkpoint, dtype and beams are flags.
Usage (from the trisect/ folder):
    python scripts/extract_clip_text_raw.py --source blip2 --captions-file data/targets/blip2_captions_full.npy
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch
from numpy.lib.format import open_memmap
from PIL import Image
from tqdm import tqdm
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANNOTS_PATH = "data/annots/COCO_73k_annots_curated.npy"
IMAGES_H5 = "data/mindeyev2_dataset/coco_images_224_float16.hdf5"
CLIP_CKPT = "openai/clip-vit-large-patch14"
EMBED_DIM = 768
CLIP_BATCH_SIZE = 256  # captions per CLIP-text encode batch
BLIP2_CKPT_DEFAULT = "Salesforce/blip2-opt-2.7b"
BLIP2_BATCH_SIZE = 8
BLIP2_CAPTIONS_OUT = "data/targets/blip2_captions_{n}.json"
BLIP2_CAPTIONS_CACHE_DEFAULT = "data/targets/blip2_captions_full.npy"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["human", "blip2"], default="human")
    p.add_argument("--blip2-ckpt", default=BLIP2_CKPT_DEFAULT)
    p.add_argument("--blip2-dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--blip2-num-beams", type=int, default=1)
    p.add_argument("--captions-file", default=BLIP2_CAPTIONS_CACHE_DEFAULT,
                    help="pre-computed full-set BLIP-2 captions (.npy, ordered by image index, "
                         "or the matching .jsonl with {idx, caption} records) -- used instead of "
                         "generating with BLIP-2 if it exists; only relevant for --source blip2")
    p.add_argument("--limit", type=int, default=None,
                    help="process only the first N images (timing probe)")
    return p.parse_args()


def out_path(source, limit):
    suffix = f"_first{limit}" if limit else ""
    return f"data/targets/clip_vitl14_text_raw_{source}{suffix}.npy"


def encode_captions(captions_per_image, clip_ckpt=CLIP_CKPT):
    """captions_per_image: list[list[str]], possibly variable-length (human) or all len-1
    (blip2). Returns (N, 768) float32, mean of the non-empty captions' raw embeddings."""
    n = len(captions_per_image)
    print(f"Loading {clip_ckpt} text encoder on {DEVICE} ...")
    tokenizer = CLIPTokenizer.from_pretrained(clip_ckpt)
    model = CLIPTextModelWithProjection.from_pretrained(clip_ckpt, torch_dtype=torch.float32).to(DEVICE)
    model.eval()

    flat_captions, flat_image_idx = [], []
    for i, caps in enumerate(captions_per_image):
        for c in caps:
            c = c.strip()
            if c:
                flat_captions.append(c)
                flat_image_idx.append(i)
    flat_image_idx = np.array(flat_image_idx)
    print(f"{len(flat_captions)} non-empty captions across {n} images "
          f"({len(flat_captions) / n:.3f}/image)")

    sums = np.zeros((n, EMBED_DIM), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)
    with torch.no_grad():
        for start in tqdm(range(0, len(flat_captions), CLIP_BATCH_SIZE), desc="CLIP text encoder"):
            end = min(start + CLIP_BATCH_SIZE, len(flat_captions))
            inputs = tokenizer(flat_captions[start:end], padding=True, truncation=True,
                                max_length=77, return_tensors="pt").to(DEVICE)
            embeds = model(**inputs).text_embeds.float().cpu().numpy()  # (b, 768), raw
            np.add.at(sums, flat_image_idx[start:end], embeds)
            np.add.at(counts, flat_image_idx[start:end], 1)
    del model
    torch.cuda.empty_cache()

    assert (counts > 0).all(), "some image has zero non-empty captions -- unexpected"
    return (sums / counts[:, None]).astype(np.float32), counts


def load_human_captions(limit):
    annots = np.load(ANNOTS_PATH, allow_pickle=True)  # (73000, 5) <U250, '' for missing
    if limit is not None:
        annots = annots[:limit]
    return [list(row) for row in annots]


def load_cached_blip2_captions(captions_file, n):
    """Returns list[str] of length n if captions_file (or its .npy/.jsonl sibling) has a full
    set of captions, else None. n is the number of images actually needed (== --limit if set)."""
    npy_path = captions_file if captions_file.endswith(".npy") else captions_file + ".npy"
    jsonl_path = captions_file if captions_file.endswith(".jsonl") else \
        os.path.splitext(captions_file)[0] + ".jsonl"

    if os.path.exists(npy_path):
        captions = np.load(npy_path, allow_pickle=True).tolist()
    elif os.path.exists(jsonl_path):
        captions = {}
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                captions[rec["idx"]] = rec["caption"]
        if len(captions) < n:
            return None  # partial run (e.g. interrupted), not safe to use as-is
        captions = [captions[i] for i in range(len(captions))]
    else:
        return None

    if len(captions) < n:
        raise ValueError(f"{captions_file} has only {len(captions)} captions, need {n}")
    return captions[:n]


def generate_blip2_captions(n, ckpt, dtype, num_beams):
    dtype_t = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    print(f"Loading {ckpt} on {DEVICE} ({dtype}, num_beams={num_beams}) ...")
    processor = Blip2Processor.from_pretrained(ckpt)
    model = Blip2ForConditionalGeneration.from_pretrained(ckpt, torch_dtype=dtype_t).to(DEVICE)
    model.eval()

    with h5py.File(IMAGES_H5, "r") as f:
        images = f["images"]
        captions = []
        with torch.no_grad():
            for start in tqdm(range(0, n, BLIP2_BATCH_SIZE), desc="BLIP-2 captioning"):
                end = min(start + BLIP2_BATCH_SIZE, n)
                batch_f16 = images[start:end]  # (b, 3, 224, 224) float16 in [0,1]
                batch_u8 = (batch_f16.astype(np.float32).transpose(0, 2, 3, 1) * 255) \
                    .clip(0, 255).astype(np.uint8)
                pil_imgs = [Image.fromarray(im) for im in batch_u8]
                inputs = processor(images=pil_imgs, return_tensors="pt").to(DEVICE, dtype_t)
                out = model.generate(**inputs, max_new_tokens=30, num_beams=num_beams)
                texts = processor.batch_decode(out, skip_special_tokens=True)
                captions.extend([t.strip() for t in texts])
    del model
    torch.cuda.empty_cache()
    return captions


def main():
    args = parse_args()

    if args.source == "human":
        captions_per_image = load_human_captions(args.limit)
    else:
        n = args.limit if args.limit is not None else h5py.File(IMAGES_H5, "r")["images"].shape[0]

        blip2_captions = load_cached_blip2_captions(args.captions_file, n)
        if blip2_captions is not None:
            print(f"Found {n} cached BLIP-2 captions under {args.captions_file} "
                  "-- loading instead of generating (--blip2-ckpt/--blip2-dtype/--blip2-num-beams ignored)")
        else:
            blip2_captions = generate_blip2_captions(n, args.blip2_ckpt, args.blip2_dtype,
                                                        args.blip2_num_beams)
            os.makedirs("data/targets", exist_ok=True)
            cache_path = BLIP2_CAPTIONS_OUT.format(n=n)
            with open(cache_path, "w") as f:
                json.dump({"ckpt": args.blip2_ckpt, "dtype": args.blip2_dtype,
                           "num_beams": args.blip2_num_beams, "captions": blip2_captions}, f, indent=2)
            print(f"Cached BLIP-2 captions -> {cache_path}")
        captions_per_image = [[c] for c in blip2_captions]

    embeds, counts = encode_captions(captions_per_image)

    op = out_path(args.source, args.limit)
    out = open_memmap(op, mode="w+", dtype="float32", shape=embeds.shape)
    out[:] = embeds
    out.flush()
    del out

    print(f"Wrote {op}: {embeds.shape} float32, mean {counts.mean():.3f} caption(s)/image "
          f"(min {counts.min()}, max {counts.max()})")
    print("Done.")


if __name__ == "__main__":
    main()
