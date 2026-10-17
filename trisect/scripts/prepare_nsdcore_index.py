"""Derive the NSD-core trial -> 73k image index.

Inputs:  nsd_expdesign.mat; MindEye2's COCO_73k_subj_indices.hdf5 (for the cross-check)
Outputs: data/nsdcore_index/subj{NN}_trial73k_index.npz with, for each of the 30000 trial rows,
         img73k (int32, index into nsd_stimuli.hdf5) and is_shared (bool, the 3000 shared1000 trials)

Adapted from the iCNN script `prepare_nsdcore_icnn.py`, without its parts: it does not read the betas
(30000 = 40 sessions x 750 trials is a protocol constant) and does not split rows by unique image,
because TRISECT works on single trials. The pairing follows Brain-Diffuser's `data/prepare_nsddata.py`
(lines 68-80) and is then checked element by element against the MindEye2 file. That check is the point:
it proves the rows of betas_all are in masterordering order, and a misalignment would give no symptom.

Usage (from the trisect/ folder):
    python scripts/prepare_nsdcore_index.py --subj 1 2 5 7 --expdesign ./data/nsddata/experiments/nsd/nsd_expdesign.mat
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as spio

TRIALS_PER_SESSION = 750
N_SESSIONS = 40        # the four subjects with complete data: 1, 2, 5, 7
N_TRIALS = N_SESSIONS * TRIALS_PER_SESSION   # 30000, a protocol constant, not read from any file
N_SHARED = 1000        # the shared1000: masterordering <= 1000 marks those trials


def loadmat(filename):
    """spio.loadmat that recovers nested structs as dictionaries.

    Same helper as brain_diffuser/data/prepare_nsddata.py and prepare_nsdcore_icnn.py:
    nsd_expdesign.mat is a MATLAB struct and plain loadmat returns mat_struct objects.
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

    Transcribed from brain_diffuser/data/prepare_nsddata.py:68-80 / prepare_nsdcore_icnn.py.
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


def check_against_mindeye2(img73k: np.ndarray, subject: int, mindeyev2_dir: Path) -> None:
    """Element-by-element check against MindEye2's own trial -> 73k map.

    This is what certifies that the rows of betas_all_subj0N_fp32_renorm.hdf5 (read only
    downstream, never here) are in masterordering order. Without it the whole pipeline rests
    on an assumption about a file somebody else built.
    """
    f = mindeyev2_dir / "COCO_73k_subj_indices.hdf5"
    if not f.exists():
        sys.exit(f"missing {f}: the cross-check cannot run, and it is the only evidence that "
                 "betas_all rows follow masterordering. Refusing to continue.")
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


def prepare_subject(subject: int, expdesign: Path, mindeyev2_dir: Path, out_dir: Path):
    print(f"\n=== subj0{subject} ===")
    stim_order = loadmat(expdesign)
    img73k, is_shared = derive_pairing(stim_order, subject, N_TRIALS)
    check_against_mindeye2(img73k, subject, mindeyev2_dir)

    n_shared_trials = int(is_shared.sum())
    print(f"  {N_TRIALS} trials, {N_TRIALS - n_shared_trials} non-shared, "
          f"{n_shared_trials} shared1000")
    assert n_shared_trials == N_SHARED * 3, \
        f"{n_shared_trials} shared1000 trials, expected {N_SHARED * 3} (3 reps x 1000 images)"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"subj{subject:02d}_trial73k_index.npz"
    np.savez(out_file,
             img73k=img73k.astype(np.int32), is_shared=is_shared,
             n_trials=np.int32(N_TRIALS))
    print(f"  wrote {out_file}")
    print(f"  img73k: {img73k[:5].tolist()} ... {img73k[-3:].tolist()}")


def main():
    parser = argparse.ArgumentParser(
        description="Derive the NSD-core trial -> 73k image index for TRISECT")
    parser.add_argument("--subj", type=int, nargs="+", default=[1, 2, 5, 7],
                        choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--expdesign", type=str,
                        default="./data/nsddata/experiments/nsd/nsd_expdesign.mat")
    parser.add_argument("--mindeyev2_dir", type=str, default="./data/mindeyev2_dataset",
                        help="where COCO_73k_subj_indices.hdf5 lives")
    parser.add_argument("--out_dir", type=str, default="./data/nsdcore_index")
    args = parser.parse_args()

    expdesign = Path(args.expdesign)
    if not expdesign.exists():
        sys.exit(f"missing {expdesign}\nrun download_data_from_internet.sh first")

    for subject in args.subj:
        prepare_subject(subject, expdesign, Path(args.mindeyev2_dir), Path(args.out_dir))

    print("\nall checks passed")


if __name__ == "__main__":
    main()
