"""Linear cross-regression R^2 between every ordered pair of z_sem, z_struct and z_low (Section 8.1.4).

Inputs:  data/targets/targets.h5 (z_sem), pca_zstruct/z_struct.npy, pca_zlow/z_low.npy,
         shared1000_mask.npy
Outputs: data/targets/cross_space_redundancy.json

If one code were largely predictable from another by a linear map, the factorization would be
weaker than it looks. RidgeCV (leave-one-out over an alpha grid) is fitted on a subsample of the
non-shared images; R^2 is measured on all of shared1000, which no PCA or adapter fit has touched.
No brain data, no GPU.

Usage:
    python cross_space_redundancy.py [--n-train 40000]
"""
import argparse
import json
import os

import h5py
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

TARGETS_H5 = "data/targets/targets.h5"
STRUCT_PATH = "data/targets/pca_zstruct/z_struct.npy"
LOW_PATH = "data/targets/pca_zlow/z_low.npy"
SHARED_MASK_PATH = "data/targets/shared1000_mask.npy"
OUT_PATH = "data/targets/cross_space_redundancy.json"
N_TRAIN_DEFAULT = 20000  # random subsample of non-shared images used to fit each regression
ALPHAS = np.logspace(-1, 5, 13)
SEED = 0


def load_spaces():
    with h5py.File(TARGETS_H5, "r") as f:
        z_sem = f["z_sem"][:].astype(np.float32)          # (N, 2, 128)
    z_sem = z_sem.reshape(z_sem.shape[0], -1)              # (N, 256)

    z_struct = np.load(STRUCT_PATH, mmap_mode="r")[:].astype(np.float32)  # (N, 8, 8, 32)
    z_struct = z_struct.reshape(z_struct.shape[0], -1)     # (N, 2048)

    z_low = np.load(LOW_PATH, mmap_mode="r")[:].astype(np.float32)  # (N, d_low)

    n = z_sem.shape[0]
    assert z_struct.shape[0] == n and z_low.shape[0] == n, "mismatched image counts across spaces"
    return {"z_sem": z_sem, "z_struct": z_struct, "z_low": z_low}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=N_TRAIN_DEFAULT)
    args = ap.parse_args()

    print("Loading z_sem, z_struct, z_low ...")
    spaces = load_spaces()
    for name, arr in spaces.items():
        print(f"  {name}: {arr.shape}")
    n = next(iter(spaces.values())).shape[0]

    shared_mask = np.load(SHARED_MASK_PATH)
    assert shared_mask.shape[0] == n
    train_pool = np.flatnonzero(~shared_mask)
    test_idx = np.flatnonzero(shared_mask)
    print(f"train pool (non-shared): {len(train_pool)}, test (shared1000): {len(test_idx)}")

    rng = np.random.default_rng(SEED)
    train_idx = np.sort(rng.choice(train_pool, size=min(args.n_train, len(train_pool)),
                                    replace=False))
    print(f"fitting on {len(train_idx)} non-shared images, evaluating on all {len(test_idx)} "
          f"shared1000 images")

    names = list(spaces.keys())
    results = {}
    for src in names:
        for dst in names:
            if src == dst:
                continue
            X_train, Y_train = spaces[src][train_idx], spaces[dst][train_idx]
            X_test, Y_test = spaces[src][test_idx], spaces[dst][test_idx]

            model = RidgeCV(alphas=ALPHAS)
            model.fit(X_train, Y_train)
            pred = model.predict(X_test)
            r2 = r2_score(Y_test, pred, multioutput="uniform_average")

            key = f"{src}->{dst}"
            results[key] = {"r2": float(r2), "alpha": float(model.alpha_)}
            print(f"  {key:20s} R^2={r2:+.4f}  (alpha={model.alpha_:.3g})")

    with open(OUT_PATH, "w") as f:
        json.dump({
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "alphas_grid": ALPHAS.tolist(),
            "results": results,
        }, f, indent=2)
    print(f"Done. Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
