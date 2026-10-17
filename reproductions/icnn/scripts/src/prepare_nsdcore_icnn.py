"""Derive the NSD-Core train/test index for iCNN (shared1000 benchmark and NSD-Imagery).

Inputs:  mindeye_II/src/mindeyev2_dataset/betas_all_subj0N_fp32_renorm.hdf5 (30000, V), nsd_expdesign.mat
Outputs: per subject a small .npz pairing rows of the betas with images of nsd_stimuli.hdf5 (73k):
         train_rows / train_uidx (27000), train_img73k (9000), test_rows / test_uidx (3000), test_img73k (1000)

The betas are not copied: decode_features_chunked.py reads the hdf5 with this index. Training keeps the 27000
single trials (shared1000 excluded), not averages: the Kamitani config that Kneeland et al. started from is
TRAINNOAVE_TESTAVE and A.6 lists no deviation here. The pairing comes from nsd_expdesign.mat as in
brain_diffuser/data/prepare_nsddata.py and is checked element by element against MindEye2's
COCO_73k_subj_indices.hdf5: a misaligned row order gives no symptom downstream.

Usage (from feature-decoding/, venv active):
    python ../src/prepare_nsdcore_icnn.py --subj 1 2 5 7 --betas_dir <PROJECT_ROOT>/mindeye_II/src/mindeyev2_dataset
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as spio

# Same numbers as prepare_nsd_imagery_icnn.py: the nsdgeneral mask at threshold == 1.0.
# betas_all_subj0N_fp32_renorm.hdf5 has exactly these voxel counts for the four subjects used.
NUM_VOXELS = {1: 15724, 2: 14278, 3: 15226, 4: 13153,
              5: 13039, 6: 17907, 7: 12682, 8: 14386}

TRIALS_PER_SESSION = 750
N_SHARED = 1000        # the shared1000: masterordering <= 1000 marks those trials
N_REPS = 3             # every NSD image is shown 3 times to the subject who owns it


def loadmat(filename):
    """spio.loadmat that recovers nested structs as dictionaries.

    Same helper as brain_diffuser/data/prepare_nsddata.py: nsd_expdesign.mat is a MATLAB
    struct and plain loadmat returns mat_struct objects.
    """
    def _todict(matobj):
        return {f: _convert(matobj.__dict__[f]) for f in matobj._fieldnames}

    # scipy >= 1.8 exposes mat_struct at scipy.io.matlab; the old path (used by
    # prepare_nsddata.py) is scipy.io.matlab.mio5_params and is gone in scipy >= 1.12.
    mat_struct = getattr(spio.matlab, "mat_struct", None) or spio.matlab.mio5_params.mat_struct

    def _convert(elem):
        if isinstance(elem, mat_struct):
            return _todict(elem)
        if isinstance(elem, np.ndarray) and elem.dtype == object:
            return [_convert(e) for e in elem]
        return elem

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return {k: _convert(v) for k, v in data.items() if not k.startswith("__")}


def derive_pairing(stim_order: dict, subject: int, n_trials: int):
    """trial row -> index into the 73k images, and whether the trial is a shared1000 one.

    Transcribed from brain_diffuser/data/prepare_nsddata.py:68-80, with num_trials taken from
    the actual number of rows of betas_all instead of the hardcoded 37*750.
    """
    subjectim = np.asarray(stim_order["subjectim"])          # (8, 10000), 1-based
    masterordering = np.asarray(stim_order["masterordering"])  # (30000,), 1-based
    assert subjectim.shape[0] >= subject, f"subjectim has {subjectim.shape[0]} rows"
    assert len(masterordering) >= n_trials, \
        f"masterordering has {len(masterordering)} entries, need {n_trials}"

    order = masterordering[:n_trials]
    img73k = subjectim[subject - 1, order - 1] - 1           # -> 0-based index into the 73k
    is_shared = order <= N_SHARED
    return img73k.astype(np.int64), is_shared


def check_against_mindeye2(img73k: np.ndarray, subject: int, betas_dir: Path) -> None:
    """Element-by-element check against MindEye2's own trial -> 73k map.

    This is what certifies that the rows of betas_all_subj0N_fp32_renorm.hdf5 are in
    masterordering order. Without it the whole pipeline rests on an assumption about a file
    somebody else built.
    """
    f = betas_dir / "COCO_73k_subj_indices.hdf5"
    if not f.exists():
        sys.exit(f"missing {f}: the cross-check against MindEye2's trial map cannot run, and it is the only"
                 " evidence that betas_all rows follow masterordering. Refusing to continue.")
    with h5py.File(f, "r") as h:
        key = f"subj{subject:02d}"
        if key not in h:
            sys.exit(f"{f}: no dataset {key} (keys: {list(h.keys())})")
        theirs = np.asarray(h[key]).astype(np.int64)

    if theirs.shape[0] != img73k.shape[0]:
        sys.exit(f"  MindEye2 map has {theirs.shape[0]} trials, we derived {img73k.shape[0]}")
    n_bad = int((theirs != img73k).sum())
    if n_bad:
        first = int(np.flatnonzero(theirs != img73k)[0])
        sys.exit(f"  MISMATCH on {n_bad}/{len(theirs)} trials, first at row {first}: "
                 f"nsd_expdesign says {img73k[first]}, MindEye2 says {theirs[first]}.\n"
                 "  The rows of betas_all are not in masterordering order, or the expdesign "
                 "file belongs to a different NSD release. Stop here.")
    print(f"  trial -> 73k map matches MindEye2's COCO_73k_subj_indices on all {len(theirs)} trials")


def group_rows(img73k: np.ndarray, rows: np.ndarray):
    """(rows, uidx, unique images) with the unique images in ascending 73k order."""
    uniq = np.unique(img73k[rows])
    lookup = {int(v): i for i, v in enumerate(uniq)}
    uidx = np.array([lookup[int(v)] for v in img73k[rows]], dtype=np.int32)
    return rows.astype(np.int32), uidx, uniq.astype(np.int32)


def prepare_subject(subject: int, betas_dir: Path, expdesign: Path, out_dir: Path,
                    allow_partial_reps: bool):
    print(f"\n=== subj0{subject} ===")
    betas_file = betas_dir / f"betas_all_subj{subject:02d}_fp32_renorm.hdf5"
    if not betas_file.exists():
        sys.exit(f"missing {betas_file}")

    with h5py.File(betas_file, "r") as h:
        if "betas" not in h:
            sys.exit(f"{betas_file}: no 'betas' dataset (keys: {list(h.keys())})")
        n_trials, n_voxel = h["betas"].shape
    n_sessions, rem = divmod(n_trials, TRIALS_PER_SESSION)
    print(f"  betas_all {n_trials} x {n_voxel}  ->  {n_sessions} sessions")
    assert rem == 0, f"{n_trials} trials is not a multiple of {TRIALS_PER_SESSION}"
    # The benchmark trains on the full 40 sessions: a subject with fewer is not one of the four
    # complete ones and must not silently produce a smaller model.
    assert n_sessions == 40, f"subj0{subject} has {n_sessions} sessions, the benchmark wants 40"
    assert n_voxel == NUM_VOXELS[subject], \
        f"betas_all has {n_voxel} voxels, the imagery .h5 has {NUM_VOXELS[subject]}: " \
        "different mask, the ridge weights would not multiply"

    stim_order = loadmat(expdesign)
    img73k, is_shared = derive_pairing(stim_order, subject, n_trials)
    check_against_mindeye2(img73k, subject, betas_dir)

    train_rows, train_uidx, train_img = group_rows(img73k, np.flatnonzero(~is_shared))
    test_rows, test_uidx, test_img = group_rows(img73k, np.flatnonzero(is_shared))

    print(f"  train {len(train_rows)} trials over {len(train_img)} images "
          f"| test (shared1000) {len(test_rows)} trials over {len(test_img)} images")

    # 40 sessions x 750 = 30000 trials = 10000 images x 3 repetitions, of which 1000 shared.
    assert len(test_img) == N_SHARED, f"{len(test_img)} shared images, expected {N_SHARED}"
    for name, uidx, imgs in (("train", train_uidx, train_img), ("test", test_uidx, test_img)):
        counts = np.bincount(uidx, minlength=len(imgs))
        if not np.all(counts == N_REPS):
            msg = (f"  {name}: repetition counts are not all {N_REPS} "
                   f"(min {counts.min()}, max {counts.max()}, "
                   f"{int((counts != N_REPS).sum())} images off)")
            if not allow_partial_reps:
                sys.exit(msg + "\n  rerun with --allow_partial_reps to accept it (and say so"
                               " in the thesis): it changes the effective training set size")
            print(msg + " -> accepted")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"subj{subject:02d}_nsdcore_index.npz"
    np.savez(out_file,
             train_rows=train_rows, train_uidx=train_uidx, train_img73k=train_img,
             test_rows=test_rows, test_uidx=test_uidx, test_img73k=test_img,
             n_sessions=np.int32(n_sessions), n_voxel=np.int32(n_voxel),
             betas_file=str(betas_file))
    print(f"  wrote {out_file}")
    print(f"  train images 73k: {train_img[:5].tolist()} ... {train_img[-3:].tolist()}")
    print(f"  test  images 73k: {test_img[:5].tolist()} ... {test_img[-3:].tolist()}")


def main():
    parser = argparse.ArgumentParser(
        description="Derive the NSD-core train/test index for iCNN benchmark B")
    parser.add_argument("--subj", type=int, nargs="+", default=[1, 2, 5, 7],
                        choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--betas_dir", type=str,
                        default=os.environ["PROJECT_ROOT"] + "/mindeye_II/src/mindeyev2_dataset",
                        help="where betas_all_subj0N_fp32_renorm.hdf5 and COCO_73k_subj_indices.hdf5 live")
    parser.add_argument("--expdesign", type=str,
                        default="./data/nsd/nsddata/experiments/nsd/nsd_expdesign.mat")
    parser.add_argument("--out_dir", type=str, default="./data/fmri/NSDcore")
    parser.add_argument("--allow_partial_reps", action="store_true",
                        help="accept images shown fewer than 3 times instead of failing")
    args = parser.parse_args()

    expdesign = Path(args.expdesign)
    if not expdesign.exists():
        sys.exit(f"missing {expdesign}\n"
                 "copies exist under takagi/nsd/nsddata/experiments/nsd/ and"
                 " brain_diffuser/data/nsddata/experiments/nsd/ -- link one of them")

    for subject in args.subj:
        prepare_subject(subject, Path(args.betas_dir), expdesign, Path(args.out_dir),
                        args.allow_partial_reps)

    print("\nall checks passed")


if __name__ == "__main__":
    main()
