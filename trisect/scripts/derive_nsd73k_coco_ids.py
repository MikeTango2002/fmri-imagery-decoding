"""Derive the 73k-space cocoId array, to check whether an external COCO image was ever used by NSD.

Inputs:  nsd_stim_info_merged.csv (cocoId and cocoSplit per nsdId)
Outputs: data/targets/nsd73k_coco_ids.npy, (73000,) int64

The proxy set of the oracle-budget experiments must contain images the pipeline has never seen.
The PCA and the adapter are fit on all 72k non-shared images, and shared1000 is reserved for
held-out comparisons; the complex proxy images come from COCO images NSD never selected.

Usage (from the trisect/ folder):
    python scripts/derive_nsd73k_coco_ids.py --csv data/nsddata/experiments/nsd/nsd_stim_info_merged.csv
"""
import argparse
import csv

import numpy as np

CSV_DEFAULT = "data/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
OUT_PATH = "data/targets/nsd73k_coco_ids.npy"
N_IMAGES = 73000


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CSV_DEFAULT)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    coco_ids = np.empty(N_IMAGES, dtype=np.int64)
    splits = np.empty(N_IMAGES, dtype=object)
    with open(args.csv) as f:
        for i, row in enumerate(csv.DictReader(f)):
            assert int(row["nsdId"]) == i, f"row {i}: nsdId {row['nsdId']} != row index"
            coco_ids[i] = int(row["cocoId"])
            splits[i] = row["cocoSplit"]

    assert len(np.unique(coco_ids)) == N_IMAGES, "duplicate cocoId across the 73k pool"
    val_n = int((splits == "val2017").sum())
    train_n = int((splits == "train2017").sum())
    assert val_n + train_n == N_IMAGES, f"unexpected cocoSplit values: val={val_n} train={train_n}"

    np.save(args.out, coco_ids)
    print(f"wrote {args.out}: {N_IMAGES} cocoIds (val2017={val_n}, train2017={train_n})")


if __name__ == "__main__":
    main()
