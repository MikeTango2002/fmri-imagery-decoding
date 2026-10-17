"""Extract the pixel z_low target, a resR x resR RGB thumbnail, for all 73,000 images (Section 8.1.3).

Inputs:  data/mindeyev2_dataset/coco_images_224_float16.hdf5 (224x224 images)
Outputs: data/targets/pixel_low_res<R>/z_low.npy, (73000, 3*R*R) float16

The 224x224 image is bilinear-resized to R x R and flattened as RGB, with no PCA, no VDVAE and no fit;
no shared1000 or development exclusion is needed. On the oracle test (no brain data) R=16 (768 dimensions)
beat the VDVAE+PCA target by +0.346 PixCorr, which is why it was tried. The output is a plain array:
fit_ridge.py can refit the z_low branch on it with its --zlow flag.

Usage:
    python extract_pixel_low_target.py --res 16 [--limit 64]    # --limit: smoke test
"""
import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from extract_vdvae_latents import IMAGES_H5, NATIVE_RES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR_TEMPLATE = "data/targets/pixel_low_res{res}"
BATCH_SIZE = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, required=True, help="target resolution resR (resR x resR x 3 dims)")
    ap.add_argument("--images-h5", type=str, default=IMAGES_H5)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N images")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or OUT_DIR_TEMPLATE.format(res=args.res)
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(args.images_h5, "r") as f:
        n = f["images"].shape[0]
    if args.limit:
        n = min(n, args.limit)
    d_out = args.res * args.res * 3
    print(f"{n} images, resR={args.res} -> d_out={d_out}, device={DEVICE}")

    out_path = os.path.join(out_dir, "z_low.npy")
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype="float16", shape=(n, d_out))

    with h5py.File(args.images_h5, "r") as f, torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = f["images"][start:end].astype(np.float32)  # (b, 3, 224, 224) in [0,1]
            batch = torch.from_numpy(batch).to(DEVICE) * 255.0
            down = F.interpolate(batch, size=(args.res, args.res), mode="bilinear",
                                  align_corners=False)
            down = down.permute(0, 2, 3, 1).contiguous()  # (b, res, res, 3), channel-last --
            # matches the layout of decode_zlow_to_image.py; a
            # future decode step can reshape(n, res, res, 3) straight back without a transpose
            out[start:end] = down.reshape(down.shape[0], -1).cpu().numpy().astype(np.float16)
            if (start // BATCH_SIZE) % 20 == 0:
                print(f"  {end}/{n}")
    out.flush()
    del out
    print(f"Done: {out_path} ({n}, {d_out}) float16, native res={NATIVE_RES}")


if __name__ == "__main__":
    main()
