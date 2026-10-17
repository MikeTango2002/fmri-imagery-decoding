"""Prepare the NSD-Imagery betas for iCNN: a flat bdpy BData file, one row per trial.

Inputs:  betas_nsdimagery.nii.gz (X,Y,Z,720) per subject, the authors' preprocessed tensors (--reference_dir)
Outputs: --out_dir/subj0N_nsdimagery_func1pt8mm_betas_fithrf_GLMdenoise_RR.h5 (384 x V, 5 evaluable conditions)

The beta recipe (nsdgeneral mask, /300, per-run z-score) is that of the MindEye1 reproduction. Unlike it, there
is no nesting or averaging: iCNN wants a flat (n_trial, V) matrix with one label per row, and the decoding step
averages the repetitions, as predict_feature.py does with average_sample. The data sit in a bdpy BData .h5 with
an ROI_nsdgeneral column and a label column. Seven sanity checks run inline and the script stops at the first failure.

Usage (from feature-decoding/, venv active):
    python ../src/prepare_nsd_imagery_icnn.py --subj 1 2 5 7 --data_root ./data/nsd --imagery_root ./data/nsd_imagery
"""

import argparse
import pickle
import sys
from pathlib import Path

import nibabel as nb
import numpy as np
import torch
from scipy.stats import zscore

# Voxel count of the nsdgeneral mask at threshold == 1.0, func1pt8mm. Same numbers as
# MindEye1 and MIRAGE. A mismatch here means the wrong mask (surface instead of volume,
# or func1mm), and downstream it would mean a ridge weight matrix that does not multiply.
NUM_VOXELS = {1: 15724, 2: 14278, 3: 15226, 4: 13153,
              5: 13039, 6: 17907, 7: 12682, 8: 14386}

# One key per run type: 3 vision, 6 imagery (2 repetitions x 3 stimulus sets), 3 attention.
RUN_TYPES = ["visA", "visB", "visC",
             "imgA_1", "imgA_2", "imgB_1", "imgB_2", "imgC_1", "imgC_2",
             "attA", "attB", "attC"]

# Only these nine get z-scored: in Kneeland et al.'s own nsd_imagery.pt the 288 attention
# rows are left at exactly zero. The attention runs are dropped from the benchmark anyway,
# so this only matters for the cross-check against their tensors.
NORM_RUN_TYPES = [r for r in RUN_TYPES if not r.startswith("att")]

COND_RUNS = {
    "visionsimple":    ["visA"],
    "visioncomplex":   ["visB"],
    "visionconcepts":  ["visC"],      # excluded from the protocol, never packed
    "imagerysimple":   ["imgA_1", "imgA_2"],
    "imagerycomplex":  ["imgB_1", "imgB_2"],
    "imageryconcepts": ["imgC_1", "imgC_2"],
}

# The 5 conditions that enter the benchmark. visionconcepts and the attention runs are
# z-scored together with the rest but never reach the BData table.
EVAL_CONDS = ["visionsimple", "visioncomplex",
              "imagerysimple", "imagerycomplex", "imageryconcepts"]

# Stimulus indices inside imagery_stimuli_18.pt: 0:6 simple, 6:12 complex, 12:18 conceptual,
# same order as the values of image_map in the .pkl3.
COND_STIM_IDS = {
    "visionsimple":    np.arange(0, 6),
    "visioncomplex":   np.arange(6, 12),
    "visionconcepts":  np.arange(12, 18),
    "imagerysimple":   np.arange(0, 6),
    "imagerycomplex":  np.arange(6, 12),
    "imageryconcepts": np.arange(12, 18),
}

# 8 repetitions for the vision trials, 16 for the imagery ones (the imagery runs were
# acquired twice).
COND_REPS = {c: (8 if c.startswith("vision") else 16) for c in COND_RUNS}

N_TRIALS = 720
N_EVAL_TRIALS = 384          # 48 (visA) + 48 (visB) + 3 x 96 (imgX_1 + imgX_2)

DATA_KEY = "VoxelData"
ROI_KEY = "ROI_nsdgeneral"
LABEL_KEY = "stimulus_name"

FMRI_NAME = "nsdimagery_func1pt8mm_betas_fithrf_GLMdenoise_RR"


def find_file(roots, *rel_candidates) -> Path:
    """First existing <root>/<candidate>, over the cartesian product. Layouts differ:
    the HuggingFace download puts the stimuli under nsddata_stimuli/stimuli/, the symlink
    farm of link_existing_nsd_data.sh reproduces that, but a hand-made tree may not."""
    tried = []
    for root in roots:
        for rel in rel_candidates:
            p = Path(root) / rel
            tried.append(p)
            if p.exists():
                return p
    sys.exit("missing file, looked for:\n  " + "\n  ".join(str(t) for t in tried))


def load_metadata(imagery_root: Path, data_root: Path):
    """exps (run type per trial), cues (cue letter per trial), image_map (cue -> stim index)."""
    pkl = find_file([imagery_root, data_root],
                    "nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3",
                    "nsddata_stimuli_link/nsdimagery_stimuli.pkl3",
                    "nsdimagery_stimuli.pkl3")
    with open(pkl, "rb") as f:
        imagery_dict = pickle.load(f)
    exps = np.asarray(imagery_dict["exps"])
    cues = np.asarray(imagery_dict["cues"])
    image_map = imagery_dict["image_map"]
    print(f"metadata: {pkl}")

    # sanity check 1 (second half) and check 4
    assert len(exps) == N_TRIALS, f"exps has {len(exps)} entries, expected {N_TRIALS}"
    assert len(cues) == N_TRIALS, f"cues has {len(cues)} entries, expected {N_TRIALS}"
    labels, counts = np.unique(exps, return_counts=True)
    print("trials per run type:", dict(zip(labels.tolist(), counts.tolist())))
    for run in RUN_TYPES:
        expected = 96 if run.startswith("att") else 48
        got = int((exps == run).sum())
        assert got == expected, f"run {run}: {got} trials, expected {expected}"
    return exps, cues, image_map


def extract_imagery_betas(subject: int, data_root: Path) -> torch.Tensor:
    """Apply the nsdgeneral mask to the 1.8 mm volumetric betas -> (720, V), no scaling.

    Threshold == 1.0, not > 0. This is the threshold that
    reproduces NUM_VOXELS and the authors' reference tensors, and it is therefore the one
    the NSD-core decoders must be trained with as well: the two sides have to agree or the
    ridge weight matrix does not multiply.
    """
    beta_file = (data_root / "nsddata_betas/ppdata" / f"subj0{subject}" /
                 "func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz")
    roi_file = (data_root / "nsddata/ppdata" / f"subj0{subject}" /
                "func1pt8mm/roi/nsdgeneral.nii.gz")
    for f in (beta_file, roi_file):
        if not f.exists():
            sys.exit(f"missing file: {f}\nrun src/link_existing_nsd_data.sh first")

    betas = nb.load(beta_file).get_fdata(dtype=np.float32)   # (X, Y, Z, 720)
    print(f"  betas volume {betas.shape}")
    assert betas.shape[3] == N_TRIALS, f"4th dim is {betas.shape[3]}, expected {N_TRIALS}"
    betas = betas.transpose((3, 0, 1, 2))                    # (720, X, Y, Z)

    nsd_general = nb.load(roi_file).get_fdata()
    mask = np.nan_to_num(np.where(nsd_general == 1.0, True, False)).flatten()

    x = betas.reshape((len(betas), -1))[:, mask].astype(np.float32)
    del betas
    return torch.from_numpy(x)                               # (720, V)


def normalize_imagery_betas(x: torch.Tensor, exps: np.ndarray,
                            nan_to_zero: bool = False) -> torch.Tensor:
    """/300 (betas are stored as integers x300), then z-score each run separately.

    ddof=1, not scipy's default ddof=0: the authors divide by the sample std, and that is
    what makes this tensor match theirs to max|diff| ~1e-5. The plain zscore() call is off by
    a global sqrt(47/48) factor. See the docstring of normalize_imagery_betas in
    prepare_nsd_imagery.py of the MindEye1 reproduction for the full history.
    """
    x = x / 300.0
    x_norm = torch.zeros_like(x)
    for run in NORM_RUN_TYPES:
        idx = np.arange(len(exps))[exps == run]
        z = zscore(x[idx].numpy(), axis=0, ddof=1).astype(np.float32)

        n_nan = int(np.isnan(z).sum())
        if n_nan:
            # A NaN here is a zero-variance voxel inside one run, never the mask.
            msg = (f"  run {run}: {n_nan} NaN after z-score "
                   f"({n_nan // len(idx)} zero-variance voxels)")
            if not nan_to_zero:
                sys.exit(msg + "\nrerun with --nan_to_zero to force them to 0"
                               " (and say so in the thesis)")
            print(msg + " -> forced to 0")
            z = np.nan_to_num(z)

        x_norm[idx] = torch.from_numpy(z)

        # sanity check 2: mean ~0 and std ~sqrt((n-1)/n) *per run*, not over the whole
        # tensor. The target is not 1 because of the ddof=1 above: 0.98953 on 48 trials.
        expected_std = np.sqrt((len(idx) - 1) / len(idx))
        assert np.abs(z.mean(axis=0)).max() < 1e-3, f"run {run}: per-run mean is not 0"
        assert np.abs(z.std(axis=0) - expected_std).max() < 1e-3, \
            f"run {run}: per-run std is not {expected_std:.5f} (ddof=1 on {len(idx)} trials)"
    return x_norm


def compare_reference(subject: int, x: torch.Tensor, x_norm: torch.Tensor,
                      reference_dir: Path | None, allow_missing: bool) -> None:
    """torch.allclose against the reference preprocessed tensors (check 5).

    Both tensors are compared, and which one fails localizes the bug: the unnormalized one
    covers the mask, the volume preparation and the trial order, the normalized one covers
    only normalize_imagery_betas() on top of that.

    A missing reference is an error, not a skip: an absent file looks exactly like a passing
    check.
    """
    files = {"nsd_imagery_unnormalized.pt": x, "nsd_imagery.pt": x_norm}
    if reference_dir is None:
        if allow_missing:
            print("  no --reference_dir: cross-check DISABLED by the caller")
            return
        sys.exit("  --reference_dir is unset and --allow_missing_reference was not given.")

    for name, mine in files.items():
        # subject{N}/ is the HuggingFace layout, subj0{N}/ is the MindEye1 reproduction's own preprocessed/
        cands = [reference_dir / f"subject{subject}" / name,
                 reference_dir / f"subj0{subject}" / name,
                 reference_dir / f"preprocessed_data/subject{subject}" / name]
        ref = next((c for c in cands if c.exists()), None)
        if ref is None:
            if allow_missing:
                print(f"  {name}: no reference under {reference_dir}, cross-check DISABLED")
                continue
            sys.exit(f"  {name}: no reference tensor, looked in:\n    " +
                     "\n    ".join(str(c) for c in cands) +
                     "\n  pass --allow_missing_reference and say so in the thesis.")
        r = torch.load(ref).float()
        if r.shape != mine.shape:
            sys.exit(f"  {name}: reference shape {tuple(r.shape)} != ours {tuple(mine.shape)}")
        d = float((r - mine).abs().max())
        ok = torch.allclose(r, mine, atol=1e-3)
        print(f"  {name:30s} vs reference: max|diff| {d:.2e} -> {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            sys.exit(f"  {name} differs from the reference tensor. If only the normalized one"
                     " fails, the mask and the trial order are fine and the bug is in"
                     " normalize_imagery_betas(): check ddof and the attention runs.")


def build_rows(x_norm: torch.Tensor, exps: np.ndarray, cues: np.ndarray, image_map: dict):
    """Flat (384, V) matrix + one zero-padded label per row, in EVAL_CONDS order.

    Labels are zero-padded ("imagerycomplex-06", not "-6"): the averaging step orders rows by
    np.unique(labels), i.e. lexicographically, and without the padding "-10" sorts before "-6" and
    the pairing with the ground truth breaks in silence.
    """
    rows, labels = [], []
    for cond in EVAL_CONDS:
        idx = np.arange(len(exps))[np.isin(exps, COND_RUNS[cond])]
        stim = np.array([image_map[c] for c in cues[idx]])   # stimulus index per trial

        # sanity check 6: the misalignment that would otherwise pass in silence
        assert set(stim.tolist()) == set(COND_STIM_IDS[cond].tolist()), \
            f"{cond}: stimulus indices {sorted(set(stim.tolist()))} != {COND_STIM_IDS[cond].tolist()}"
        assert len(idx) == 6 * COND_REPS[cond], \
            f"{cond}: {len(idx)} trials, expected {6 * COND_REPS[cond]}"

        rows.append(x_norm[torch.from_numpy(idx).long()].numpy())
        labels += [f"{cond}-{s:02d}" for s in stim]

    X = np.vstack(rows).astype(np.float32)
    assert X.shape[0] == N_EVAL_TRIALS == len(labels), (X.shape, len(labels))

    # sanity check 4 (second half): 12 labels with 8 trials (vision), 18 with 16 (imagery)
    uniq, counts = np.unique(labels, return_counts=True)
    n8, n16 = int((counts == 8).sum()), int((counts == 16).sum())
    assert len(uniq) == 30, f"{len(uniq)} distinct labels, expected 30"
    assert n8 == 12 and n16 == 18, f"label counts: {n8} with 8 reps, {n16} with 16, expected 12/18"

    # sanity check 7: np.unique() sorts lexicographically, and the averaging step downstream
    # relies on that order matching the numeric one.
    assert list(uniq) == sorted(uniq), "labels are not lexicographically sortable"
    return X, labels, list(uniq)


def write_bdata(X: np.ndarray, labels: list, out_file: Path) -> list:
    """Pack into a bdpy BData .h5: data + ROI metadata column + label column with vmap."""
    import bdpy

    out_file.parent.mkdir(parents=True, exist_ok=True)
    uniq = sorted(set(labels))
    code = {lb: i + 1 for i, lb in enumerate(uniq)}          # vmap codes are 1-based

    bdata = bdpy.BData()
    bdata.add(X, DATA_KEY)
    # one metadata value per column of the data matrix; 1 marks a voxel inside nsdgeneral,
    # so that the config string "ROI_nsdgeneral = 1" selects the whole matrix.
    bdata.add_metadata(ROI_KEY, np.ones(X.shape[1]),
                       description="nsdgeneral ROI (func1pt8mm, mask == 1.0)",
                       where=DATA_KEY)
    bdata.add(np.array([[code[lb]] for lb in labels], dtype=np.float64), LABEL_KEY)
    bdata.add_vmap(LABEL_KEY, {v: k for k, v in code.items()})
    bdata.save(str(out_file))
    print(f"  wrote {out_file} ({X.shape[0]} x {X.shape[1]})")
    return uniq


def verify_bdata(out_file: Path, X: np.ndarray, labels: list) -> None:
    """Reopen the file the way predict_feature.py does and check it round-trips.

    This is the only check that exercises the actual bdpy API rather than an assumption about it:
    select() with the config string "ROI_nsdgeneral = 1", and get_label() with label_key.
    """
    import bdpy

    bdata = bdpy.BData(str(out_file))
    back = bdata.select(f"{ROI_KEY} = 1")               # predict_feature.py:111
    assert back.shape == X.shape, f"select() returned {back.shape}, expected {X.shape}"
    assert np.allclose(back, X, atol=0, rtol=0), "select() did not return the data verbatim"

    try:                                                # predict_feature.py:113-117
        back_labels = bdata.get_label(LABEL_KEY)
    except Exception as e:                              # noqa: BLE001 - same fallback as upstream
        print(f"  get_label failed ({e}), falling back to select()")
        back_labels = list(bdata.select(LABEL_KEY).flatten())
    back_labels = [str(lb) for lb in back_labels]
    assert back_labels == labels, "labels do not round-trip through the vmap"
    print(f"  round-trip OK: select() {back.shape}, {len(set(back_labels))} distinct labels")


def write_ground_truth(imagery_root: Path, data_root: Path, out_dir: Path) -> None:
    """Slice imagery_stimuli_18.pt into one ground truth tensor per condition + the pool of 12.

    Subject independent, written once. The metric harness clamps reconstructions to [0,1]
    and assumes the ground truth is in the same range; the range is checked here.
    """
    stim_file = find_file([imagery_root, data_root],
                          "nsddata_stimuli/stimuli/imagery_stimuli_18.pt",
                          "nsddata_stimuli_link/imagery_stimuli_18.pt",
                          "imagery_stimuli_18.pt")
    s = torch.load(stim_file).float()
    print(f"imagery_stimuli_18: {tuple(s.shape)} {s.dtype} "
          f"min {float(s.min()):.3f} max {float(s.max()):.3f}  ({stim_file})")
    assert tuple(s.shape) == (18, 3, 425, 425), f"unexpected stimulus shape {tuple(s.shape)}"
    if float(s.max()) > 1.5:
        print("  range looks like [0,255] -> dividing by 255")
        s = s / 255.0
    assert 0.0 <= float(s.min()) and float(s.max()) <= 1.0 + 1e-4, "ground truth outside [0,1]"

    out_dir.mkdir(parents=True, exist_ok=True)
    for cond, ids in COND_STIM_IDS.items():
        gt = s[torch.from_numpy(ids).long()]
        if cond.endswith("concepts"):
            # The conceptual stimuli have no visual ground truth (they only feed Table 2 of
            # Kneeland et al., the human raters). The cue images are kept next to it for the figures, and the
            # gt tensor is zeroed so that no image based metric can be computed by accident.
            torch.save(gt.clone(), out_dir / f"gt_{cond}_cues.pt")
            gt = torch.zeros_like(gt)
        torch.save(gt, out_dir / f"gt_{cond}.pt")
    # The 12-stimulus pool (simple + complex) is the distractor pool of every two-way
    # metric.
    torch.save(s[:12], out_dir / "gt_12.pt")
    print(f"wrote ground truth tensors to {out_dir}")


def prepare_subject(subject: int, data_root: Path, out_dir: Path, exps, cues, image_map,
                    reference_dir: Path | None, nan_to_zero: bool,
                    allow_missing_reference: bool, keep_pt: bool):
    print(f"\n=== subj0{subject} ===")

    x = extract_imagery_betas(subject, data_root)
    print(f"  masked betas {tuple(x.shape)}")
    # sanity check 1
    assert x.shape == (N_TRIALS, NUM_VOXELS[subject]), \
        f"expected {(N_TRIALS, NUM_VOXELS[subject])}, got {tuple(x.shape)} " \
        "(wrong mask: surface instead of volume, or func1mm instead of func1pt8mm?)"

    x_norm = normalize_imagery_betas(x, exps, nan_to_zero=nan_to_zero)
    # sanity check 3
    assert not torch.isnan(x_norm).any(), "NaN in the normalized tensor"

    compare_reference(subject, x, x_norm, reference_dir, allow_missing_reference)

    if keep_pt:
        pt_dir = out_dir / "pt"
        pt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(x_norm, pt_dir / f"subj0{subject}_nsd_imagery.pt")

    X, labels, uniq = build_rows(x_norm, exps, cues, image_map)
    out_file = out_dir / f"subj0{subject}_{FMRI_NAME}.h5"
    write_bdata(X, labels, out_file)
    verify_bdata(out_file, X, labels)
    print(f"  labels: {uniq[0]} ... {uniq[-1]}  ({len(uniq)} distinct)")


def main():
    parser = argparse.ArgumentParser(description="Prepare NSD-Imagery betas for iCNN (benchmark B)")
    parser.add_argument("--subj", type=int, nargs="+", default=[1, 2, 5, 7],
                        choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--data_root", type=str, default="./data/nsd",
                        help="tree holding nsddata/ and nsddata_betas/ (see link_existing_nsd_data.sh)")
    parser.add_argument("--imagery_root", type=str, default="./data/nsd_imagery",
                        help="tree holding nsddata_stimuli/stimuli/{nsdimagery_stimuli.pkl3,imagery_stimuli_18.pt}")
    parser.add_argument("--out_dir", type=str, default="./data/fmri/NSDImagery")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="where to write the ground truth tensors (default: --out_dir/gt)")
    parser.add_argument("--reference_dir", type=str, default=None,
                        help="preprocessed_data/ with the reference nsd_imagery{,_unnormalized}.pt")
    parser.add_argument("--nan_to_zero", action="store_true",
                        help="force NaN produced by zero-variance voxels to 0 instead of failing")
    parser.add_argument("--allow_missing_reference", action="store_true",
                        help="downgrade a missing reference tensor from fatal to a warning")
    parser.add_argument("--keep_pt", action="store_true",
                        help="also dump the full (720, V) normalized tensor as .pt, for diagnostics")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    imagery_root = Path(args.imagery_root)
    out_dir = Path(args.out_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else out_dir / "gt"
    reference_dir = Path(args.reference_dir) if args.reference_dir else None

    exps, cues, image_map = load_metadata(imagery_root, data_root)
    print("image_map:", image_map)
    write_ground_truth(imagery_root, data_root, gt_dir)

    for subject in args.subj:
        prepare_subject(subject, data_root, out_dir, exps, cues, image_map,
                        reference_dir, args.nan_to_zero, args.allow_missing_reference,
                        args.keep_pt)

    print("\nall checks passed")


if __name__ == "__main__":
    main()
