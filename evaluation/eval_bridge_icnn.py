#!/usr/bin/env python
"""The shared1000 bridge for iCNN, scored with the same harness as the NSD-Imagery benchmark.

Inputs:  --recon_dir and --gt_dir, PNGs matched by label
Outputs: --out_tsv with the image metrics and the two-way metrics on random pools of 12

PixCorr and SSIM carry almost no signal above chance and cannot see smoothness. The published
Shared1000 row also has EffNet-B (.797) and SwAV (.528), perceptual distances that can. This script
measures them with the authors' harness. If the values land on them, stage 2 renders as well as
Kneeland's, and a weaker NSD-Imagery result is specific to NSD-Imagery. If clearly worse, stage 2
is globally weaker. The two-way metrics are computed on pools of 12 (the NSD-Imagery pool; the
published row leaves them blank for 982 distractors) and compared with the NSD-Imagery vision row.

Usage (MindEye2 environment):
    OPENBLAS_CORETYPE=SkylakeX python eval_bridge_icnn.py --recon_dir RECON_DIR --gt_dir GT_DIR \
        --out_tsv results/baselines/shared1000_bridge/icnn.tsv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from final_evaluations_imagery import (Nets, TARGET_DIM, _blas_sanity, get_pix_corr,
                                       get_ssim, _to_tensor)

RECON_PREFIX = "recon_image_normalized-"

# Table 1, "NSD Shared1000 Test Set", iCNN row (Kneeland et al. 2025). The four two-way cells
# are "-" in the paper; only these four can be compared against a published number.
PUBLISHED = {"PixCorr": 0.321, "SSIM": 0.336, "EffNet-B": 0.797, "SwAV": 0.528}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recon_dir", required=True,
                   help="directory of recon_image_normalized-<label>.tiff")
    p.add_argument("--gt_dir", required=True, help="directory of <label>.png ground truth")
    p.add_argument("--n_pools", type=int, default=300,
                   help="random pools of 12 for the two-way metrics")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_tsv", default="")
    return p.parse_args()


def labelled(directory, strip=""):
    out = {}
    for name in sorted(os.listdir(directory)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".tiff", ".tif", ".jpg", ".jpeg"):
            continue
        if strip and stem.startswith(strip):
            stem = stem[len(strip):]
        out[stem] = os.path.join(directory, name)
    return out


def load_stack(paths, size=None):
    """(N, 3, H, W) float32 in [0,1]; resized to `size` the way resize_recons does."""
    from torchvision import transforms
    imgs = []
    for p in paths:
        with PIL.Image.open(p) as img:
            t = _to_tensor(img.convert("RGB")).float()
        if size is not None and tuple(t.shape[-2:]) != (size, size):
            t = transforms.Resize((size, size))(t)
        imgs.append(t)
    return torch.stack(imgs)


def main():
    args = parse_args()
    _blas_sanity()
    rng = np.random.default_rng(args.seed)

    gt = labelled(args.gt_dir)
    rec = labelled(args.recon_dir, strip=RECON_PREFIX)
    common = sorted(set(gt) & set(rec))
    if not common:
        raise SystemExit(f"no label common to {args.gt_dir} and {args.recon_dir}")
    print(f"{len(common)} stimuli scored with the benchmark-B harness\n")

    images = load_stack([gt[l] for l in common])
    # resize_recons brings every reconstruction to 512 before the metrics; the ground truth
    # keeps its native 425, exactly as in the NSD-Imagery run.
    recons = load_stack([rec[l] for l in common], size=TARGET_DIM)
    print(f"  gt {tuple(images.shape)}  recon {tuple(recons.shape)}")

    nets = Nets(args.device)

    per_stim = {
        "PixCorr": np.asarray(get_pix_corr(images, recons)),
        "SSIM": np.asarray(get_ssim(images, recons)),
        "EffNet-B": np.asarray(nets.get_efficientnet(images, recons)),
        "SwAV": np.asarray(nets.get_swav(images, recons)),
    }

    print("\nAgainst the published Shared1000 row (Table 1, iCNN):")
    print(f"{'metric':>10} {'measured':>10} {'SE':>7} {'published':>10} {'diff':>8}")
    for m in ("PixCorr", "SSIM", "EffNet-B", "SwAV"):
        v = per_stim[m]
        se = v.std(ddof=1) / np.sqrt(len(v))
        print(f"{m:>10} {v.mean():>10.4f} {se:>7.4f} {PUBLISHED[m]:>10.3f} "
              f"{v.mean() - PUBLISHED[m]:>+8.4f}")

    # The two-way metrics on pools of 12: the pool NSD-Imagery uses; the numbers are
    # comparable to the NSD-Imagery vision row, and carry the same -1/11 off-by-one.
    pools = [rng.choice(len(common), 12, replace=False) for _ in range(args.n_pools)]
    twoway = {k: [] for k in ("AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP")}
    for k, sel in enumerate(pools):
        i, r = images[sel], recons[sel]
        a2, a5 = nets.get_alexnet(i, r)
        twoway["AlexNet(2)"].append(np.mean(a2))
        twoway["AlexNet(5)"].append(np.mean(a5))
        twoway["InceptionV3"].append(np.mean(nets.get_inceptionv3(i, r)))
        twoway["CLIP"].append(np.mean(nets.get_clip(i, r)))
        if (k + 1) % 50 == 0:
            print(f"    {k + 1}/{len(pools)} pools", flush=True)

    print("\nTwo-way on pools of 12 (chance 40.91%, off-by-one included), against our own")
    print("NSD-Imagery vision row of Section 4.5:")
    ours_vision = {"AlexNet(2)": 0.605, "AlexNet(5)": 0.651,
                   "InceptionV3": 0.563, "CLIP": 0.427}
    print(f"{'metric':>12} {'bridge':>8} {'SE':>7} {'nsdi vision':>12} {'diff':>8}")
    for m, v in twoway.items():
        v = np.asarray(v)
        se = v.std(ddof=1) / np.sqrt(len(v))
        print(f"{m:>12} {v.mean():>8.4f} {se:>7.4f} {ours_vision[m]:>12.3f} "
              f"{v.mean() - ours_vision[m]:>+8.4f}")

    if args.out_tsv:
        import csv
        Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_tsv, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["metric", "scope", "mean", "se", "n"])
            for m, v in per_stim.items():
                w.writerow([m, "all100", f"{v.mean():.6f}",
                            f"{v.std(ddof=1) / np.sqrt(len(v)):.6f}", len(v)])
            for m, v in twoway.items():
                v = np.asarray(v)
                w.writerow([m, "pools12", f"{v.mean():.6f}",
                            f"{v.std(ddof=1) / np.sqrt(len(v)):.6f}", len(v)])
        print(f"\nwrote {args.out_tsv}")


if __name__ == "__main__":
    sys.exit(main())
