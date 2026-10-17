"""BLIP-2 captions for the full 73k NSD/COCO image set (train + shared1000).

Inputs:  data/mindeyev2_dataset/coco_images_224_float16.hdf5
Outputs: data/targets/blip2_captions_full.jsonl (one line per image), then an ordered .npy of strings

The captions feed the text half of z_sem and the adapter's text context. No training, no brain data.
Defaults: blip2-flan-t5-xxl in bfloat16, 5 beams. It needs ~30 GB of GPU memory and hours of run time,
so the JSONL is flushed after every batch and the run resumes from the last completed index. Do not
delete a partial output file: it is what makes a crash cheap.

Usage (from the trisect/ folder):
    python scripts/extract_blip2_captions.py --limit 64     # smoke test
"""
import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_H5 = "data/mindeyev2_dataset/coco_images_224_float16.hdf5"
N_TOTAL = 73000
OUT_JSONL = "data/targets/blip2_captions_full.jsonl"


def count_done(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        return sum(1 for _ in f)


def load_batch_uint8(h5_file, start, end):
    imgs_f16 = h5_file["images"][start:end]  # (B, 3, 224, 224) float16 in [0,1]
    return (imgs_f16.astype(np.float32).transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="Salesforce/blip2-flan-t5-xxl")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default=OUT_JSONL)
    parser.add_argument("--limit", type=int, default=N_TOTAL, help="stop after this many images (for smoke tests)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    start_idx = count_done(args.out)
    if start_idx >= args.limit:
        print(f"{args.out} already has {start_idx} lines >= limit {args.limit}, nothing to do.")
        return
    print(f"Resuming from index {start_idx} (found {start_idx} completed lines in {args.out})")

    print(f"Loading {args.ckpt} on {DEVICE} (bf16) ...")
    processor = Blip2Processor.from_pretrained(args.ckpt)
    model = Blip2ForConditionalGeneration.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16).to(DEVICE)
    model.eval()

    h5_file = h5py.File(IMAGES_H5, "r")
    t_start = time.time()
    n_done_this_run = 0

    with open(args.out, "a") as out_f:
        for batch_start in range(start_idx, args.limit, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.limit)
            imgs_u8 = load_batch_uint8(h5_file, batch_start, batch_end)
            pil_imgs = [Image.fromarray(im) for im in imgs_u8]

            inputs = processor(images=pil_imgs, return_tensors="pt").to(DEVICE, torch.bfloat16)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=30, num_beams=args.num_beams, do_sample=False)
            texts = processor.batch_decode(out, skip_special_tokens=True)

            for i, caption in zip(range(batch_start, batch_end), texts):
                out_f.write(json.dumps({"idx": i, "caption": caption.strip()}) + "\n")
            out_f.flush()

            n_done_this_run += batch_end - batch_start
            if batch_start % (args.batch_size * 50) == 0:
                elapsed = time.time() - t_start
                rate = n_done_this_run / elapsed if elapsed > 0 else 0
                remaining = args.limit - batch_end
                eta_min = remaining / rate / 60 if rate > 0 else float("nan")
                print(f"  {batch_end}/{args.limit}  ({rate:.2f} img/s, ETA {eta_min:.0f} min)")

    h5_file.close()
    n_final = count_done(args.out)
    print(f"Done. {args.out} has {n_final} lines.")

    if n_final >= N_TOTAL:
        print("Converting JSONL to ordered .npy of strings ...")
        captions = [None] * N_TOTAL
        with open(args.out, "r") as f:
            for line in f:
                rec = json.loads(line)
                captions[rec["idx"]] = rec["caption"]
        assert all(c is not None for c in captions), "gap in the JSONL, not all 73000 indices present"
        npy_path = args.out.replace(".jsonl", ".npy")
        np.save(npy_path, np.array(captions, dtype=object))
        print(f"Wrote {npy_path} ({N_TOTAL} captions, ordered by 73k COCO index)")


if __name__ == "__main__":
    main()
