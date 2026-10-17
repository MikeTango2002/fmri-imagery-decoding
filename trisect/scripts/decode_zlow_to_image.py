"""Decode z_low back to pixels (PCA^-1 -> VDVAE decoder): the acceptance test of the z_low representation.

Inputs:  the frozen basis of `fit_pca_zlow.py`; each image's OWN z_low (not brain-predicted), and only
         the information lost by the PCA is measured.
Outputs: mean PixCorr printed against the 224->64 bilinear resize of the original (same resize as
         `extract_vdvae_latents.py`); with --save-png, the decoded images.

The 64x64 image must stay recognisable, with PixCorr >= 0.25 (Brain-Diffuser's first stage: 0.2976).
Adapted from Brain-Diffuser's `scripts/vdvae_reconstruct_images.py` (`forward_manual_latents`, which
only gets 31 of the 75 decoder blocks and samples the rest from the prior; kept as is, because it
produced the 0.2976 baseline). Added: PCA^-1 and the per-layer de-scaling of `fit_pca_zlow.py`.
--raw skips both and decodes the uncompressed 91168-dim latents: the ceiling if PCA cost nothing.

Usage (from the trisect/ folder):
    python scripts/decode_zlow_to_image.py --n 32 --save-png data/targets/zlow_recon_check
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vdvae_lib"))
from model_utils import load_vaes, set_up_data  # noqa: E402  (needs the sys.path insert above)

from extract_vdvae_latents import (  # noqa: E402  (reuses the exact same model/data setup)
    DEVICE, H_DICT, IMAGES_H5, VDVAE_RES, NSDImagesDataset, dotdict,
)
# layer_bounds/layer_shapes/pixcorr live in trisect/ and are re-exported here: other scripts
# (`reconstruct.py`, the precheck scripts) import them from this module.
from trisect.metrics import pixcorr  # noqa: E402,F401  (re-export)
from trisect.pca import layer_bounds, layer_shapes  # noqa: E402,F401  (re-export)
from trisect.pca import load_pca_basis as _load_pca_basis  # noqa: E402

PCA_DIR_DEFAULT = "data/targets/pca_zlow"
RAW_LATENTS_PATH = "data/targets/vdvae_latents_31l.npy"             # RAW_PATH of fit_pca_zlow.py
RAW_LAYER_DIMS_PATH = "data/targets/vdvae_latents_31l_layer_dims.json"
STOP_CONDITION_PIXCORR = 0.25  # acceptance threshold of the z_low decode check


def load_pca_basis(pca_dir):
    """Load fit_pca_zlow.py's frozen basis, reused by decode_zlow_batch() below.

    float64 + the layer layout: the inverse transform below needs both (trisect/pca.py).
    """
    return _load_pca_basis(pca_dir, dtype=np.float64, layout=True)


def project_raw_to_zlow(raw_latents, basis):
    """Forward transform of decode_zlow_batch()'s inverse: raw (n, 91168) VDVAE latents at the
    decoder's native scale -> (n, d_out) PCA-space z_low. Exactly fit_pca_zlow.py's transform()
    (per-layer scalar standardization, then centre / project / whiten), reimplemented from the
    frozen basis so an image pool the fit never streamed (the proxy set of the oracle-budget
    experiments, whose latents live in vdvae_latents_proxy_raw.npy rather than pca_zlow/z_low.npy)
    can be put through the SAME compression the ridge's z_low goes through.
    """
    raw = np.asarray(raw_latents, dtype=np.float64)
    standardized = raw / basis["dim_scale"][None, :]
    centered = standardized - basis["mean"][None, :]
    projected = centered @ basis["components"].T
    return projected / np.sqrt(np.clip(basis["explained_variance"], 1e-12, None))[None, :]


def decode_zlow_batch(z_low, basis, ema_vae, device):
    """z_low (n, d_out): PCA-space, any source (a real image's own or brain-predicted) --
    -> (n, 64, 64, 3) uint8 pixels, via the same PCA^-1 + per-layer de-scale +
    forward_manual_latents chain as main() below. Factored out so the reconstruction script
    (reconstruct.py) can reuse it on ridge-predicted z_low without duplicating the
    inverse-transform math.
    """
    z = np.asarray(z_low, dtype=np.float64)
    n = z.shape[0]
    standardized = z @ (np.sqrt(basis["explained_variance"])[:, None] * basis["components"]) + basis["mean"]
    raw_latent = standardized * basis["dim_scale"][None, :]

    latents_per_layer = []
    for (a, b), (c, h, w) in zip(basis["bounds"], basis["shapes"]):
        chunk = raw_latent[:, a:b].reshape(n, c, h, w)
        latents_per_layer.append(torch.from_numpy(chunk).float().to(device))

    with torch.no_grad():
        px_z = ema_vae.decoder.forward_manual_latents(n, latents_per_layer, t=None)
        recon = ema_vae.decoder.out_net.sample(px_z)  # (n, 64, 64, 3) uint8
    return recon


def decode_raw_batch(raw_latents, shapes, bounds, ema_vae, device):
    """Like decode_zlow_batch() but with PCA^-1 replaced by the identity: raw_latents (n, 91168)
    are already at the decoder's native scale (they ARE what fit_pca_zlow.py standardizes and
    fits PCA on); no de-standardization is needed either, just reshape per layer and decode.
    Isolates VDVAE decode quality (forward_manual_latents on 31/75 blocks) from PCA compression
    loss: the --raw ceiling check of the module docstring.
    """
    n = raw_latents.shape[0]
    latents_per_layer = []
    for (a, b), (c, h, w) in zip(bounds, shapes):
        chunk = raw_latents[:, a:b].reshape(n, c, h, w)
        latents_per_layer.append(torch.from_numpy(chunk).float().to(device))

    with torch.no_grad():
        px_z = ema_vae.decoder.forward_manual_latents(n, latents_per_layer, t=None)
        recon = ema_vae.decoder.out_net.sample(px_z)
    return recon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca-dir", type=str, default=PCA_DIR_DEFAULT)
    ap.add_argument("--raw", action="store_true",
                     help="skip PCA^-1 entirely (identity): decode the uncompressed 91168-dim "
                          "VDVAE latents directly, the ceiling check (see docstring)")
    ap.add_argument("--n", type=int, default=32, help="number of images to decode/check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-png", type=str, default=None,
                     help="optional dir to dump ground-truth/reconstruction PNG pairs")
    args = ap.parse_args()

    if args.raw:
        with open(RAW_LAYER_DIMS_PATH) as f:
            layer_dims = json.load(f)["layer_dims"]
        shapes, bounds = layer_shapes(layer_dims), layer_bounds(layer_dims)
        raw = np.load(RAW_LATENTS_PATH, mmap_mode="r")
        n_total = raw.shape[0]
        print(f"raw VDVAE latents: {raw.shape} (--raw: PCA^-1 replaced by identity)")

        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(n_total, size=min(args.n, n_total), replace=False))
        print(f"Decoding {len(idx)} images: {idx.tolist()}")

        z = raw[idx].astype(np.float32)  # (n, 91168)

        print("Loading VDVAE (ImageNet64, EMA weights) ...")
        H = dotdict(dict(H_DICT))
        H, preprocess_fn = set_up_data(H)
        ema_vae = load_vaes(H)

        recon = decode_raw_batch(z, shapes, bounds, ema_vae, DEVICE)
    else:
        basis = load_pca_basis(args.pca_dir)

        z_low = np.load(os.path.join(args.pca_dir, "z_low.npy"), mmap_mode="r")
        n_total, d_out = z_low.shape
        print(f"z_low: {z_low.shape}, pca basis d_out={basis['components'].shape[0]}")

        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(n_total, size=min(args.n, n_total), replace=False))
        print(f"Decoding {len(idx)} images: {idx.tolist()}")

        z = z_low[idx]  # (n, d_out)

        print("Loading VDVAE (ImageNet64, EMA weights) ...")
        H = dotdict(dict(H_DICT))
        H, preprocess_fn = set_up_data(H)
        ema_vae = load_vaes(H)

        recon = decode_zlow_batch(z, basis, ema_vae, DEVICE)

    dataset = NSDImagesDataset(IMAGES_H5)
    gt_imgs = []
    with torch.no_grad():
        for i in idx:
            img, _ = dataset[int(i)]
            img = img.unsqueeze(0).to(DEVICE) * 255.0
            img = F.interpolate(img, size=(VDVAE_RES, VDVAE_RES), mode="bilinear", align_corners=False)
            img = img.permute(0, 2, 3, 1).contiguous()
            gt_imgs.append(img.cpu().numpy())
    gt = np.clip(np.concatenate(gt_imgs, axis=0), 0, 255).astype(np.uint8)

    corr = pixcorr(recon, gt)
    print(f"PixCorr per image: min={corr.min():.4f} mean={corr.mean():.4f} max={corr.max():.4f}")
    if args.raw:
        print(f"--raw ceiling (PCA^-1 = identity, no compression loss): mean PixCorr = "
              f"{corr.mean():.4f} -- compare against the PCA'd 0.514 on the same sample")
    else:
        passed = corr.mean() >= STOP_CONDITION_PIXCORR
        print(f"stop condition (>= {STOP_CONDITION_PIXCORR}): {'PASS' if passed else 'FAIL'}")

    if args.save_png:
        from PIL import Image
        os.makedirs(args.save_png, exist_ok=True)
        for i, im_idx in enumerate(idx):
            side_by_side = np.concatenate([gt[i], recon[i]], axis=1)
            Image.fromarray(side_by_side).resize((512, 256), resample=Image.NEAREST).save(
                os.path.join(args.save_png, f"{int(im_idx)}_gt_vs_recon.png")
            )
        print(f"Saved {len(idx)} ground-truth/reconstruction PNG pairs to {args.save_png}")


if __name__ == "__main__":
    main()
