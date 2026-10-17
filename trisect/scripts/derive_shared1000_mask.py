"""Derive the 73k-space shared1000 mask; the PCA fits can exclude it.

Inputs:  data/nsdcore_index/subj{NN}_trial73k_index.npz (from `prepare_nsdcore_index.py`), or
         nsd_expdesign.mat as a fallback
Outputs: data/targets/shared1000_mask.npy, (73000,) bool

shared1000 is the held-out NSD-core test set, kept out of every fit and every training statistic.
It is the same for subjects 1/2/5/7; one mask is enough. It is read from the per-subject index
and checked for agreement across subjects. Without the index (no NSD-core betas on the machine), it
is derived from `nsd_expdesign.mat` through `derive_pairing()`, with the same check and the same mask.

Usage (from the trisect/ folder):
    python scripts/derive_shared1000_mask.py --expdesign data/nsddata/experiments/nsd/nsd_expdesign.mat
"""
import argparse
import sys
from pathlib import Path

import numpy as np

INDEX_DIR = "data/nsdcore_index"
SUBJECTS = [1, 2, 5, 7]
N_IMAGES = 73000
N_TRIALS = 30000
N_SHARED = 1000
OUT_PATH = "data/targets/shared1000_mask.npy"
EXPDESIGN_DEFAULT = "data/nsddata/experiments/nsd/nsd_expdesign.mat"


def ids_from_index_files(index_dir: str):
    per_subject_ids = []
    for s in SUBJECTS:
        d = np.load(f"{index_dir}/subj{s:02d}_trial73k_index.npz")
        img73k, is_shared = d["img73k"], d["is_shared"]
        uniq = np.unique(img73k[is_shared])
        assert len(uniq) == N_SHARED, f"subj{s:02d}: {len(uniq)} unique shared images, expected {N_SHARED}"
        per_subject_ids.append(uniq)
        print(f"subj{s:02d}: {int(is_shared.sum())} shared trials -> {len(uniq)} unique 73k ids")
    return per_subject_ids


def ids_from_expdesign(expdesign: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_nsdcore_index import derive_pairing, loadmat  # noqa: E402

    stim_order = loadmat(expdesign)
    per_subject_ids = []
    for s in SUBJECTS:
        img73k, is_shared = derive_pairing(stim_order, s, N_TRIALS)
        uniq = np.unique(img73k[is_shared])
        assert len(uniq) == N_SHARED, f"subj{s:02d}: {len(uniq)} unique shared images, expected {N_SHARED}"
        per_subject_ids.append(uniq)
        print(f"subj{s:02d} (from {expdesign}): {int(is_shared.sum())} shared trials "
              f"-> {len(uniq)} unique 73k ids")
    return per_subject_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", default=INDEX_DIR)
    ap.add_argument("--expdesign", default=EXPDESIGN_DEFAULT,
                    help="fallback source when --index-dir is missing")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    have_index = all(Path(f"{args.index_dir}/subj{s:02d}_trial73k_index.npz").exists()
                     for s in SUBJECTS)
    if have_index:
        per_subject_ids = ids_from_index_files(args.index_dir)
    else:
        print(f"{args.index_dir} incomplete -- falling back to {args.expdesign}")
        per_subject_ids = ids_from_expdesign(args.expdesign)

    for s, ids in zip(SUBJECTS[1:], per_subject_ids[1:]):
        assert np.array_equal(ids, per_subject_ids[0]), \
            f"subj{s:02d} disagrees with subj{SUBJECTS[0]:02d} on the shared1000 73k ids -- " \
            "derive_pairing is broken, stop here"
    print(f"all {len(SUBJECTS)} subjects agree on the same {N_SHARED} shared1000 73k ids")

    mask = np.zeros(N_IMAGES, dtype=bool)
    mask[per_subject_ids[0]] = True
    assert mask.sum() == N_SHARED

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, mask)
    print(f"wrote {out}: {mask.sum()} shared1000 / {N_IMAGES} total")


if __name__ == "__main__":
    main()
