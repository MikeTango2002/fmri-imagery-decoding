"""Derive the per-subject dev hold-out used to pick the z_low width and to calibrate ridge corrections.

Inputs:  data/nsdcore_index/subj{NN}_trial73k_index.npz (from `prepare_nsdcore_index.py`)
Outputs: data/targets/nsdcore_dev_mask_subj{NN}.npy, (73000,) bool, True on 1000 of the subject's 9000
         own non-shared images (fixed seed). The non-shared pool is disjoint across subjects; the
         mask is per subject, unlike shared1000.

The dev set comes from the training pool, never from shared1000, which stays the held-out test set.
The z_low PCA fit and the z_low ridge branch add `& ~dev_mask` to their `train_mask`, and the z_low
criterion is measured on dev. z_sem, z_struct and the adapter do not depend on the z_low width;
they keep the full pool. After the width is fixed, z_low is refit on the full pool, dev included.

Usage (from the trisect/ folder):
    python scripts/derive_nsdcore_dev_mask.py --subj 1 2 5 7 --n-dev 1000
"""
import argparse

import numpy as np

SUBJECTS = (1, 2, 5, 7)
N_IMAGES = 73000
N_DEV_DEFAULT = 1000
INDEX_DIR = "data/nsdcore_index"
OUT_DIR = "data/targets"


def derive_for_subject(subj, n_dev, index_dir, seed):
    d = np.load(f"{index_dir}/subj{subj:02d}_trial73k_index.npz")
    img73k, is_shared = d["img73k"], d["is_shared"]
    nonshared_ids = np.unique(img73k[~is_shared])
    assert n_dev < len(nonshared_ids), \
        f"subj{subj:02d}: only {len(nonshared_ids)} non-shared images, cannot hold out {n_dev}"

    rng = np.random.default_rng(seed)
    dev_ids = rng.choice(nonshared_ids, size=n_dev, replace=False)

    mask = np.zeros(N_IMAGES, dtype=bool)
    mask[dev_ids] = True
    assert mask.sum() == n_dev
    assert not np.any(mask & is_shared_mask_73k(img73k, is_shared)), \
        f"subj{subj:02d}: dev mask overlaps shared1000 -- must not happen"
    return mask, len(nonshared_ids)


def is_shared_mask_73k(img73k, is_shared):
    mask = np.zeros(N_IMAGES, dtype=bool)
    mask[np.unique(img73k[is_shared])] = True
    return mask


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--n-dev", type=int, default=N_DEV_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index-dir", type=str, default=INDEX_DIR)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    for subj in args.subj:
        mask, n_nonshared = derive_for_subject(subj, args.n_dev, args.index_dir, args.seed)
        out_path = f"{args.out_dir}/nsdcore_dev_mask_subj{subj:02d}.npy"
        np.save(out_path, mask)
        print(f"subj{subj:02d}: {args.n_dev}/{n_nonshared} non-shared images held out -> {out_path}")

    print("\nall done")


if __name__ == "__main__":
    main()
