"""Caption the 300 proxy-set images; the true Versatile Diffusion context needs one caption per image.

Inputs:  data/proxy_set/{manifest.json, images/complex/}
Outputs: data/proxy_set/captions.npy, 300 strings in the flat proxy order (0-149 simple, 150-299 complex)

proxy-simple is synthetic; it has no COCO caption: it gets a template caption built from its
generation metadata (more reliable than BLIP-2 on shapes it never saw). The 150 proxy-complex photographs
are captioned by BLIP-2 (same checkpoint and settings as extract_blip2_captions.py).

Usage (GPU needed, about 15 GB):
    python caption_proxy_set.py
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROXY_DIR_DEFAULT = "data/proxy_set"
OUT_DEFAULT = "data/proxy_set/captions.npy"
N_SIMPLE = 150
N_COMPLEX = 150

# Rough palette of 12 colour names, nearest neighbour in RGB. Good enough for a caption; not
# meant to be perceptually exact.
_PALETTE = {
    "black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128),
    "red": (220, 30, 30), "orange": (240, 140, 30), "yellow": (230, 220, 40),
    "green": (40, 160, 60), "cyan": (40, 190, 190), "blue": (40, 80, 220),
    "purple": (150, 50, 180), "pink": (230, 120, 170), "brown": (120, 80, 40),
}


def color_name(rgb):
    best, best_d = "gray", float("inf")
    for name, c in _PALETTE.items():
        d = sum((int(a) - b) ** 2 for a, b in zip(rgb, c))
        if d < best_d:
            best, best_d = name, d
    return best


def simple_caption(meta):
    fam = meta["family"]
    fg = color_name(meta["fg"])
    bg = color_name(meta["bg"])
    if fam == "bar":
        return f"a {fg} bar on a {bg} background"
    if fam == "cross":
        return f"a {fg} cross on a {bg} background"
    if fam == "shape_filled":
        n = meta["n_sides"]
        shape = {3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon", 8: "octagon"}.get(n, f"{n}-sided polygon")
        return f"a filled {fg} {shape} on a {bg} background"
    if fam == "shape_hollow":
        n = meta["n_sides"]
        shape = {3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon", 8: "octagon"}.get(n, f"{n}-sided polygon")
        return f"an outlined {fg} {shape} on a {bg} background"
    if fam == "circle":
        kind = "filled" if meta["filled"] else "outlined"
        return f"a {kind} {fg} circle on a {bg} background"
    raise ValueError(fam)


def caption_complex(proxy_dir, n):
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Salesforce/blip2-flan-t5-xxl on {device} (bf16) ...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xxl", torch_dtype=torch.bfloat16).to(device)
    model.eval()

    captions = []
    batch_size = 8
    paths = [os.path.join(proxy_dir, "images", "complex", f"{i:03d}.png") for i in range(n)]
    for start in tqdm(range(0, n, batch_size), desc="BLIP-2 proxy-complex"):
        batch_paths = paths[start:start + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt").to(device, torch.bfloat16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, num_beams=5, do_sample=False)
        captions.extend(t.strip() for t in processor.batch_decode(out, skip_special_tokens=True))
    return captions


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-dir", default=PROXY_DIR_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    with open(os.path.join(args.proxy_dir, "manifest.json")) as f:
        manifest = json.load(f)
    assert len(manifest["simple"]) == N_SIMPLE and len(manifest["complex"]) == N_COMPLEX, \
        f"manifest has {len(manifest['simple'])} simple / {len(manifest['complex'])} complex, " \
        f"expected {N_SIMPLE}/{N_COMPLEX}"

    print(f"generating {N_SIMPLE} template captions for proxy-simple ...")
    simple_captions = [simple_caption(m) for m in manifest["simple"]]
    for c in simple_captions[:5]:
        print(f"  {c}")

    print(f"captioning {N_COMPLEX} proxy-complex images with BLIP-2 ...")
    complex_captions = caption_complex(args.proxy_dir, N_COMPLEX)
    for c in complex_captions[:5]:
        print(f"  {c}")

    captions = np.array(simple_captions + complex_captions, dtype=object)
    assert len(captions) == N_SIMPLE + N_COMPLEX
    np.save(args.out, captions)
    print(f"\nWrote {len(captions)} captions to {args.out} "
          f"(0-{N_SIMPLE - 1}=simple, {N_SIMPLE}-{N_SIMPLE + N_COMPLEX - 1}=complex)")


if __name__ == "__main__":
    main()
