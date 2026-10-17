"""Raw VDVAE latents (91168 dims) of the 300 proxy-set images: the true low-level input of T0 and T1.

Inputs:  data/proxy_set/images/{simple,complex}/
Outputs: data/targets/vdvae_latents_proxy_raw.npy (300, 91168) float16, plus a _layer_dims.json sidecar

Same model and forward pass as extract_vdvae_latents.py (the first 31 hierarchical layers, flattened).
Only the image source differs: the proxy PNGs (425x425) are resized straight to 64x64, with no 224 step.
Order: 0-149 simple, 150-299 complex.

Usage (GPU needed):
    python extract_proxy_vdvae_latents.py
    python extract_proxy_vdvae_latents.py --limit 8   # smoke test
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vdvae_lib"))
from model_utils import load_vaes, set_up_data  # noqa: E402  (needs the sys.path insert above)

from extract_vdvae_latents import DEVICE, EMA_WEIGHTS_PATH, H_DICT, NUM_LATENTS, VDVAE_RES, dotdict  # noqa: E402

PROXY_DIR_DEFAULT = "data/proxy_set"
OUT_DEFAULT = "data/targets/vdvae_latents_proxy_raw.npy"
N_SIMPLE = 150
N_COMPLEX = 150
BATCH_SIZE_DEFAULT = 32


def load_proxy_image_paths(proxy_dir, limit=None):
    n_simple = N_SIMPLE if limit is None else min(N_SIMPLE, limit)
    paths = [os.path.join(proxy_dir, "images", "simple", f"{i:03d}.png") for i in range(n_simple)]
    if limit is None or limit > N_SIMPLE:
        n_complex = N_COMPLEX if limit is None else min(N_COMPLEX, limit - N_SIMPLE)
        paths += [os.path.join(proxy_dir, "images", "complex", f"{i:03d}.png") for i in range(n_complex)]
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-dir", default=PROXY_DIR_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N images only")
    args = ap.parse_args()

    if not os.path.isfile(EMA_WEIGHTS_PATH):
        raise FileNotFoundError(f"{EMA_WEIGHTS_PATH} not found")

    paths = load_proxy_image_paths(args.proxy_dir, args.limit)
    n = len(paths)
    print(f"{n} proxy images -> raw VDVAE latents")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Loading VDVAE (ImageNet64, EMA weights) on {DEVICE} ...")
    H = dotdict(dict(H_DICT))
    H, preprocess_fn = set_up_data(H)
    ema_vae = load_vaes(H)

    latent_dim = None
    layer_dims = None
    fp = None

    with torch.no_grad():
        for start in tqdm(range(0, n, args.batch_size), desc="VDVAE raw latents (proxy set)"):
            end = min(start + args.batch_size, n)
            imgs = []
            for p in paths[start:end]:
                arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)  # (H,W,3) in [0,255]
                imgs.append(torch.from_numpy(arr).permute(2, 0, 1))               # (3,H,W)
            batch = torch.stack(imgs).to(DEVICE, non_blocking=True)               # (B,3,H,W) in [0,255]
            batch = F.interpolate(batch, size=(VDVAE_RES, VDVAE_RES), mode="bilinear", align_corners=False)
            batch = batch.permute(0, 2, 3, 1).contiguous()                        # (B,64,64,3)

            data_input, _ = preprocess_fn(batch)
            activations = ema_vae.encoder.forward(data_input)
            _, stats = ema_vae.decoder.forward(activations, get_latents=True)
            assert len(stats) >= NUM_LATENTS

            b = data_input.shape[0]
            per_layer = [stats[i]["z"].cpu().numpy().reshape(b, -1) for i in range(NUM_LATENTS)]
            batch_latent = np.hstack(per_layer)

            if fp is None:
                latent_dim = batch_latent.shape[1]
                layer_dims = [layer.shape[1] for layer in per_layer]
                assert sum(layer_dims) == latent_dim
                from numpy.lib.format import open_memmap
                shape = (n, latent_dim)
                print(f"Writing {shape} float16 to {args.out} (~{np.prod(shape) * 2 / 1e9:.2f} GB)")
                fp = open_memmap(args.out, mode="w+", dtype="float16", shape=shape)
                layer_dims_path = os.path.splitext(args.out)[0] + "_layer_dims.json"
                with open(layer_dims_path, "w") as jf:
                    json.dump({"num_latents": NUM_LATENTS, "layer_dims": layer_dims}, jf)
            else:
                assert batch_latent.shape[1] == latent_dim

            fp[start:end] = batch_latent.astype("float16")

    fp.flush()
    del fp
    print("Done.")


if __name__ == "__main__":
    main()
