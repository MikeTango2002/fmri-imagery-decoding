"""Frozen targets (z_sem_img, z_struct, z_low) for the 18 NSD-Imagery stimuli.

Inputs:  imagery_stimuli_18.pt; the frozen PCA bases of the training targets
Outputs: data/targets/targets_imagery18.npz, for the prediction-vs-target diagnostics only

Order of imagery_stimuli_18.pt: 0-5 simple, 6-11 complex, 12-17 conceptual. It is the id in
imagery_predictions/subj0N/labels.npy ("imagerycomplex-06" -> 6); the arrays join without a table.
Formulas are imported, not re-derived (`TargetEncoder` from `extract_proxy_targets.py`, VDVAE constants from
`extract_vdvae_latents.py`). z_low goes through the same per-layer `raw / dim_scale` as `fit_pca_zlow.py`;
skipping it rescales z_low by ~4.3x. VDVAE's `stats[i]["z"]` is a posterior sample; --n-draws (default
2) stores several draws; their cosine is the ceiling no decoder beats. z_sem_txt is not produced.

Usage (from the trisect/ folder; needs a GPU, ~1 min):
    python scripts/extract_imagery_targets.py
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vdvae_lib"))

from extract_proxy_targets import TargetEncoder, pca_transform  # noqa: E402
from trisect.pca import load_pca_basis  # noqa: E402

GT_DEFAULT = "data/nsddata_stimuli/stimuli/imagery_stimuli_18.pt"
OUT_DEFAULT = "data/targets/targets_imagery18.npz"
PCA_ZSEM_DIR = "data/targets/pca_zsem/d128"
PCA_ZSTRUCT_DIR = "data/targets/pca_zstruct"
PCA_ZLOW_DIR = "data/targets/pca_zlow"


def load_stimuli(path):
    """imagery_stimuli_18.pt -> (18, 3, 425, 425) float32 in [0,1]."""
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):
        t = t[list(t.keys())[0]]
    t = t.float()
    if t.max() > 1.5:
        t = t / 255.0
    assert t.shape[0] == 18 and t.shape[1] == 3, t.shape
    return t


def vdvae_raw_latents(imgs01, n_draws, seed0):
    """(18,3,425,425) in [0,1] -> (n_draws, 18, 91168) float32, same forward pass as
    extract_vdvae_latents.py / extract_proxy_vdvae_latents.py (resize to 64, preprocess_fn,
    encoder -> decoder(get_latents=True), first NUM_LATENTS layers flattened+concatenated)."""
    from model_utils import load_vaes, set_up_data  # noqa: E402
    from extract_vdvae_latents import DEVICE, EMA_WEIGHTS_PATH, H_DICT, NUM_LATENTS, VDVAE_RES, dotdict  # noqa: E402

    if not os.path.isfile(EMA_WEIGHTS_PATH):
        raise FileNotFoundError(EMA_WEIGHTS_PATH)
    H = dotdict(dict(H_DICT))
    H, preprocess_fn = set_up_data(H)
    ema_vae = load_vaes(H)

    batch = (imgs01 * 255.0).to(DEVICE)
    batch = F.interpolate(batch, size=(VDVAE_RES, VDVAE_RES), mode="bilinear", align_corners=False)
    batch = batch.permute(0, 2, 3, 1).contiguous()
    data_input, _ = preprocess_fn(batch)

    draws, layer_dims = [], None
    with torch.no_grad():
        for d in range(n_draws):
            torch.manual_seed(seed0 + d)
            activations = ema_vae.encoder.forward(data_input)
            _, stats = ema_vae.decoder.forward(activations, get_latents=True)
            per_layer = [stats[i]["z"].cpu().numpy().reshape(data_input.shape[0], -1)
                         for i in range(NUM_LATENTS)]
            if layer_dims is None:
                layer_dims = [layer.shape[1] for layer in per_layer]
            draws.append(np.hstack(per_layer).astype(np.float32))
            print(f"  VDVAE draw {d}: {draws[-1].shape}", flush=True)
    return np.stack(draws), layer_dims


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=GT_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--n-draws", type=int, default=2,
                    help="VDVAE posterior draws per stimulus; draw 0 is 'the' target")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    imgs01 = load_stimuli(args.gt)
    print(f"{imgs01.shape[0]} NSD-Imagery stimuli from {args.gt}")
    pil = [Image.fromarray((imgs01[i].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
           for i in range(imgs01.shape[0])]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = TargetEncoder(PCA_ZSEM_DIR, PCA_ZSTRUCT_DIR, device)
    z_sem_img, z_struct = enc.encode(pil)
    print(f"z_sem_img {z_sem_img.shape}  z_struct {z_struct.shape}")
    del enc
    torch.cuda.empty_cache()

    raw, layer_dims = vdvae_raw_latents(imgs01, args.n_draws, args.seed)
    basis = load_pca_basis(PCA_ZLOW_DIR, layout=True)
    z_low = np.stack([pca_transform(raw[d] / basis["dim_scale"][None, :], basis).astype(np.float32)
                      for d in range(raw.shape[0])])
    print(f"z_low {z_low.shape} (draws, 18, {z_low.shape[-1]})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, z_sem_img=z_sem_img, z_struct=z_struct, z_low=z_low,
             z_low_raw_draw0=raw[0].astype(np.float16),
             layer_dims=np.asarray(layer_dims), n_draws=args.n_draws)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
