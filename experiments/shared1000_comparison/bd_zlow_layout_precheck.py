"""Check that the raw VDVAE latent that Brain-Diffuser predicts decodes correctly with TRISECT's decoder.

Inputs:  nsd_vdvae_nsdgeneral_pred_sub{sub}_31l_alpha50k.npy (bd_predict_shared1000.py), Brain-Diffuser's VDVAE-stage PNGs
Outputs: data/results/shared1000_comparison/bd_zlow_layout_precheck_subj0N.json (PixCorr, pass if the mean is >= 0.90)

Preflight of the swap (compare_bd_swap.py, Section 8.4): no generation starts before the layouts agree. The raw
91168-dim latent goes through decode_raw_batch() with no per-layer rescaling. The reference is Brain-Diffuser's own
VDVAE-stage reconstruction (results/vdvae/subj0N/<row>.png), not the original image: the same latent through the same
frozen decoder must reproduce it almost exactly, while a wrong layout gives a scrambled image.

Usage (.venv_targets active, from the project root):
    python bd_zlow_layout_precheck.py --sub 1 --n 16
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vdvae_lib"))
from model_utils import load_vaes, set_up_data  # noqa: E402

from decode_zlow_to_image import (  # noqa: E402
    RAW_LAYER_DIMS_PATH, decode_raw_batch, layer_bounds, layer_shapes,
)
from extract_vdvae_latents import DEVICE, H_DICT, dotdict  # noqa: E402
from trisect.metrics import pixcorr  # noqa: E402

BD_ROOT_DEFAULT = "../brain_diffuser"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bd-root", default=BD_ROOT_DEFAULT)
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(RAW_LAYER_DIMS_PATH) as f:
        layer_dims = json.load(f)["layer_dims"]
    shapes, bounds = layer_shapes(layer_dims), layer_bounds(layer_dims)
    total_dims = sum(np.prod(s) for s in shapes)
    print(f"TRISECT raw layer layout: {len(shapes)} layers, {total_dims} dims")

    bd_path = os.path.join(args.bd_root, "data", "predicted_features", f"subj{args.sub:02d}",
                            f"nsd_vdvae_nsdgeneral_pred_sub{args.sub}_31l_alpha50k.npy")
    bd_raw = np.load(bd_path, mmap_mode="r")
    print(f"BD predicted raw vdvae latent: {bd_raw.shape}")
    if bd_raw.shape[1] != total_dims:
        raise SystemExit(f"dim mismatch: BD={bd_raw.shape[1]} vs TRISECT layout={total_dims} -- "
                          "the layer layouts do NOT match, decode_raw_batch cannot be reused "
                          "as-is")

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(bd_raw.shape[0], size=min(args.n, bd_raw.shape[0]), replace=False))
    print(f"Decoding rows {idx.tolist()}")
    z = bd_raw[idx].astype(np.float32)

    print("Loading VDVAE (ImageNet64, EMA weights) ...")
    H = dotdict(dict(H_DICT))
    H, _ = set_up_data(H)
    ema_vae = load_vaes(H)

    recon = decode_raw_batch(z, shapes, bounds, ema_vae, DEVICE)  # (n,64,64,3) uint8

    from PIL import Image
    bd_recon_dir = os.path.join(args.bd_root, "results", "vdvae", f"subj{args.sub:02d}")
    bd_imgs = []
    for i in idx:
        im = Image.open(os.path.join(bd_recon_dir, f"{int(i)}.png")).convert("RGB")
        im = im.resize((64, 64), resample=Image.BILINEAR)
        bd_imgs.append(np.asarray(im))
    bd_imgs = np.stack(bd_imgs).astype(np.uint8)

    corr = pixcorr(recon, bd_imgs)
    print(f"PixCorr(TRISECT decode_raw_batch, BD's own vdvae PNG) per image: "
          f"min={corr.min():.4f} mean={corr.mean():.4f} max={corr.max():.4f}")
    passed = corr.mean() >= 0.9  # near-identical decode of the same latent through the same
    # decoder is the expected outcome if the layout matches; a scrambled layout gives ~noise.
    print(f"layout match (mean PixCorr >= 0.90): {'PASS' if passed else 'FAIL'}")

    out = {"sub": args.sub, "n": len(idx), "idx": idx.tolist(),
           "pixcorr_min": float(corr.min()), "pixcorr_mean": float(corr.mean()),
           "pixcorr_max": float(corr.max()), "pass": bool(passed)}
    out_path = f"data/results/shared1000_comparison/bd_zlow_layout_precheck_subj{args.sub:02d}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
