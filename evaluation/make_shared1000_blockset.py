"""Re-package shared1000 as 12-stimulus blocks, in the format `make_proxy_set.py` writes; the oracle arms
(Section 8.2) can be measured on real NSD images with the same harness.

Inputs:  the 73k images (MindEye2's coco_images_224_float16.hdf5), `img73k_ids.npy` of `predict_ridge_shared1000.py`
Outputs: 83 blocks (blockNN_gt.pt) and a manifest; the 4 leftover ids are dropped and recorded there

Why 12: Kneeland's two-way metrics use a pool of 12 (1 congruent + 11 distractors, chance 40.91%). That pool
is the scale of every table; the 1000 images are split into blocks and the harness stays unchanged
(`evaluate_block_set.py --proxy-dir`). Each block is split 6 + 6 into "imagerysimple" / "imagerycomplex": the
names are structural; `stim_type` means nothing on this set. Order: ascending 73k id, matching the ridge rows.
Limitation: the ground truth is 224-native, upsampled to 425. All arms share it; read differences, not levels.

Usage (from the folder that holds the TRISECT data directory):
    python make_shared1000_blockset.py --n-blocks 5     # smoke test
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

SUBJECTS = (1, 2, 5, 7)
PRED_DIR_DEFAULT = "data/betas_trisect/nsdcore_predictions"
IMAGES_H5_DEFAULT = "data/mindeyev2_dataset/coco_images_224_float16.hdf5"
OUT_DIR_DEFAULT = "data/shared1000_blockset"
N_STIM = 6        # per condition, final_evaluations_imagery.py's N_STIM
N_TEST = 12       # per block, its N_TEST
STIM_SIZE = 425   # load_ground_truth()'s assert


def load_shared_ids(pred_dir, subjects):
    """The shared1000 row order every subject's predictions already use, checked identical."""
    ids_ref = None
    for subj in subjects:
        p = Path(pred_dir) / f"subj{subj:02d}" / "img73k_ids.npy"
        ids = np.load(p)
        if ids_ref is None:
            ids_ref = ids
        elif not np.array_equal(ids, ids_ref):
            raise SystemExit(f"{p} has a different row order than subj{subjects[0]:02d} -- the "
                             "blockset would not line up with every subject's predictions")
    assert np.all(np.diff(ids_ref) > 0), "img73k_ids.npy must be strictly ascending"
    return ids_ref.astype(np.int64)


def load_images(ids, images_h5):
    """(n, 3, 425, 425) float32 in [0,1], 224-native content upsampled, see module docstring."""
    with h5py.File(images_h5, "r") as f:
        imgs = f["images"][ids]                       # (n, 3, 224, 224) float16 in [0,1]
    t = torch.from_numpy(imgs.astype(np.float32))
    t = F.interpolate(t, size=(STIM_SIZE, STIM_SIZE), mode="bilinear", align_corners=False)
    return t.clamp(0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", default=PRED_DIR_DEFAULT)
    ap.add_argument("--images-h5", default=IMAGES_H5_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS))
    ap.add_argument("--n-blocks", type=int, default=None, help="smoke test: first N blocks only")
    args = ap.parse_args()

    ids = load_shared_ids(args.pred_dir, args.subj)
    n_blocks = len(ids) // N_TEST
    if args.n_blocks is not None:
        n_blocks = min(n_blocks, args.n_blocks)
    n_used = n_blocks * N_TEST
    dropped = [int(i) for i in ids[n_used:]]
    print(f"{len(ids)} shared1000 images -> {n_blocks} blocks of {N_TEST} "
          f"({len(dropped)} leftover id(s) dropped)")

    out_dir = Path(args.out_dir)
    (out_dir / "blocks").mkdir(parents=True, exist_ok=True)

    blocks_meta = []
    for b in range(n_blocks):
        block_ids = ids[b * N_TEST:(b + 1) * N_TEST]
        imgs = load_images(block_ids, args.images_h5)              # (12,3,425,425)
        # Slots 12-17 are inert filler, exactly as make_proxy_set.py does: the harness asserts an
        # 18-slot ground truth (NSD-Imagery has 18 stimuli, 6 of them conceptual) but scores only
        # the first N_TEST=12, which are also the whole two-way pool.
        gt = torch.cat([imgs, imgs[N_STIM:]], dim=0)               # (18,3,425,425)
        assert tuple(gt.shape) == (18, 3, STIM_SIZE, STIM_SIZE), tuple(gt.shape)
        assert -1e-4 <= float(gt.min()) and float(gt.max()) <= 1.0 + 1e-4
        torch.save(gt, out_dir / "blocks" / f"block{b:02d}_gt.pt")
        blocks_meta.append({
            "block": b,
            "img73k_ids": [int(i) for i in block_ids],
            "imagerysimple_ids": [int(i) for i in block_ids[:N_STIM]],
            "imagerycomplex_ids": [int(i) for i in block_ids[N_STIM:]],
            "gt_path": f"blocks/block{b:02d}_gt.pt",
        })
        if (b + 1) % 10 == 0 or b == n_blocks - 1:
            print(f"  wrote block {b + 1}/{n_blocks}", flush=True)

    manifest = {
        "source": "shared1000 (NSD's shared test images), ascending 73k id",
        "row_order_from": str(Path(args.pred_dir) / "subj01" / "img73k_ids.npy"),
        "subjects_checked": list(args.subj),
        "images_h5": args.images_h5,
        "gt_resolution_note": "224-native content bilinearly upsampled to 425 (no 425 source on "
                              "this server) -- within-set differences are unaffected, absolute "
                              "PixCorr/SSIM are NOT comparable with the 425-native proxy set",
        "condition_names_note": "imagerysimple/imagerycomplex are STRUCTURAL halves of each block, "
                                "not stimulus types -- every image is a natural photograph. The "
                                "stim_type column of the TSVs is meaningless on this set.",
        "stim_size": STIM_SIZE,
        "n_stim_per_block": N_STIM,
        "n_blocks": n_blocks,
        "n_images_used": n_used,
        "dropped_img73k_ids": dropped,
        "note_2afc_chance": "40.91% (10/11), same as the proxy set -- same harness, same pool size",
        "blocks": blocks_meta,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {n_blocks} blocks ({n_used} images) + manifest to {out_dir}")


if __name__ == "__main__":
    main()
