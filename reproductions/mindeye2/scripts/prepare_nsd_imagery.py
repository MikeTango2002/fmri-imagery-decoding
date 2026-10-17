"""Prepare the NSD-Imagery betas for MindEye2 inference (Section 4.2).

Inputs:  betas_nsdimagery.nii.gz (X,Y,Z,720) per subject, the authors' reference tensors (--reference_dir)
Outputs: --out_dir/subject{N}/x_visionall.pt (18, 8, V) and x_imageryall.pt (18, 16, V); gt_18.pt and gt_12.pt at the root

Follows `create_whole_region_imagery_{un,}normalized` in src/utils.py of MedARC-AI/MIRAGE. The recipe (nsdgeneral
mask, /300, per-run z-score with ddof=1, the 288 attention rows left at zero) is that of the MindEye1 script, which
matched the authors' tensors to max|diff| ~1e-5 on all four subjects. MindEye2 reads two combined conditions (all 18
stimuli per mode); the six restricted ones are written too, but nothing reads them. Every sanity check runs inline.

Usage (from src/, venv active):
    python prepare_nsd_imagery.py --subj 1 2 5 7 --data_root imagery_dataset --betas_hdf5_dir mindeyev2_dataset \
        --out_dir imagery_dataset/preprocessed_data --reference_dir imagery_dataset/reference
"""

import argparse
import pickle
import sys
from pathlib import Path

import h5py
import nibabel as nb
import numpy as np
import torch
from scipy.stats import zscore

# RidgeRegression's in_features, per subject (recon_inference.py:123-128, num_voxels read
# from the training hdf5). If the mask yields a different count the checkpoint's
# ridge.linears.0.weight simply will not load. Only the four 40-session subjects: MindEye2
# benchmark B only ever runs on 1, 2, 5, 7.
NUM_VOXELS = {1: 15724, 2: 14278, 5: 13039, 7: 12682}

# One key per run type: 3 vision, 6 imagery (2 repetitions x 3 stimulus sets), 3 attention.
RUN_TYPES = ["visA", "visB", "visC",
             "imgA_1", "imgA_2", "imgB_1", "imgB_2", "imgC_1", "imgC_2",
             "attA", "attB", "attC"]

# Only these nine get z-scored: in Kneeland et al.'s own nsd_imagery.pt the 288 attention
# rows are left at exactly zero (their tensor is allocated with torch.zeros and only the
# non-attention runs are filled). The attention runs are dropped from the benchmark anyway,
# so this only matters for the cross-check against their tensors.
NORM_RUN_TYPES = [r for r in RUN_TYPES if not r.startswith("att")]

# Conditions = modality x stimulus set, plus the two combined "*all" conditions MindEye2's
# inference script actually consumes. The six restricted ones cost nothing to
# also write and keep the door open for diagnostics.
COND_RUNS = {
    "visionsimple":    ["visA"],
    "visioncomplex":   ["visB"],
    "visionconcepts":  ["visC"],
    "imagerysimple":   ["imgA_1", "imgA_2"],
    "imagerycomplex":  ["imgB_1", "imgB_2"],
    "imageryconcepts": ["imgC_1", "imgC_2"],
    "visionall":       ["visA", "visB", "visC"],
    "imageryall":      ["imgA_1", "imgA_2", "imgB_1", "imgB_2", "imgC_1", "imgC_2"],
}

# Stimulus indices inside imagery_stimuli_18.pt each condition should recover, sorted
# ascending: 0:6 simple, 6:12 complex, 12:18 conceptual, same order as image_map in the
# .pkl3. The *all conditions recover all 18 because each stimulus's trials all live inside
# exactly one of the underlying runs (simple only in visA, complex only in visB, etc.).
COND_STIM_IDS = {
    "visionsimple":    np.arange(0, 6),
    "visioncomplex":   np.arange(6, 12),
    "visionconcepts":  np.arange(12, 18),
    "imagerysimple":   np.arange(0, 6),
    "imagerycomplex":  np.arange(6, 12),
    "imageryconcepts": np.arange(12, 18),
    "visionall":       np.arange(0, 18),
    "imageryall":      np.arange(0, 18),
}

# 8 repetitions for the vision trials, 16 for the imagery ones (the imagery runs were
# acquired twice). A (18,8,V) imageryall tensor would mean a lost imgX_2 run.
COND_REPS = {c: (8 if c.startswith("vision") else 16) for c in COND_RUNS}

N_TRIALS = 720


def load_metadata(data_root: Path):
    """exps (run type per trial), cues (cue letter per trial), image_map (cue -> stimulus index)."""
    pkl = data_root / "nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3"
    if not pkl.exists():
        sys.exit(f"missing file: {pkl}\nrun download_nsd_imagery.sh first")
    with open(pkl, "rb") as f:
        imagery_dict = pickle.load(f)
    exps = np.asarray(imagery_dict["exps"])
    cues = np.asarray(imagery_dict["cues"])
    image_map = imagery_dict["image_map"]

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
    """Apply the nsdgeneral mask to the 1.8 mm volumetric betas -> (720, V), no scaling."""
    beta_file = (data_root / "nsddata_betas/ppdata" / f"subj0{subject}" /
                 "func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz")
    roi_file = (data_root / "nsddata/ppdata" / f"subj0{subject}" /
                "func1pt8mm/roi/nsdgeneral.nii.gz")
    for f in (beta_file, roi_file):
        if not f.exists():
            sys.exit(f"missing file: {f}\nrun download_nsd_imagery.sh first")

    betas = nb.load(beta_file).get_fdata(dtype=np.float32)   # (X, Y, Z, 720)
    print(f"  betas volume {betas.shape}")
    assert betas.shape[3] == N_TRIALS, f"4th dim is {betas.shape[3]}, expected {N_TRIALS}"
    betas = betas.transpose((3, 0, 1, 2))                    # (720, X, Y, Z)

    nsd_general = nb.load(roi_file).get_fdata()
    mask = np.nan_to_num(np.where(nsd_general == 1.0, True, False)).flatten()

    x = betas.reshape((len(betas), -1))[:, mask].astype(np.float32)
    del betas
    return torch.from_numpy(x)                               # (720, V)


def check_training_hdf5(subject: int, betas_hdf5_dir: Path, expected_v: int) -> None:
    """The check that matters most: the imagery voxel count
    must match the hdf5 MindEye2 was actually trained on, or the checkpoint's ridge layer
    is being fed a different voxel space than it was fit on, a silent, plausible-looking
    failure, not a crash."""
    hdf5_path = betas_hdf5_dir / f"betas_all_subj0{subject}_fp32_renorm.hdf5"
    if not hdf5_path.exists():
        sys.exit(f"--betas_hdf5_dir: missing {hdf5_path}\n"
                 "This is the check that catches a wrong ROI/mask before it silently "
                 "produces plausible-looking numbers.")
    with h5py.File(hdf5_path, "r") as f:
        v = f["betas"].shape[-1]
    print(f"  training hdf5 {hdf5_path.name}: V={v}")
    if v != expected_v:
        sys.exit(f"VOXEL MISMATCH: the imagery mask gives V={expected_v} but the training "
                 f"hdf5 has V={v}. MindEye2 was NOT trained on nsdgeneral as prepared here. "
                 "Find the cause before going any further.")


def normalize_imagery_betas(x: torch.Tensor, exps: np.ndarray,
                            nan_to_zero: bool = False) -> torch.Tensor:
    """/300 (betas are stored as integers x300), then z-score each run separately.

    In NSD-Imagery the vision and imagery runs live in the same scan session;
    normalizing per run is what controls the SNR non-stationarity between the two
    modalities. The attention runs are left at zero (see NORM_RUN_TYPES) and dropped later.

    ddof=1, not scipy's default ddof=0: see the module docstring. The authors' per-run std
    measured with ddof=0 is 0.98953 = sqrt(47/48) on every voxel of every 48-trial run --
    i.e. ddof=1 (dividing by the sample std) is what makes this tensor match theirs.
    """
    x = x / 300.0
    x_norm = torch.zeros_like(x)
    for run in NORM_RUN_TYPES:
        idx = np.arange(len(exps))[exps == run]
        z = zscore(x[idx].numpy(), axis=0, ddof=1).astype(np.float32)

        n_nan = int(np.isnan(z).sum())
        if n_nan:
            # A NaN here is a zero-variance voxel inside one run, never the mask:
            # np.where(roi == 1.0, ...) cannot produce NaN.
            msg = (f"  run {run}: {n_nan} NaN after z-score "
                   f"({n_nan // len(idx)} zero-variance voxels)")
            if not nan_to_zero:
                sys.exit(msg + "\nrerun with --nan_to_zero to force them to 0 (and say so in the thesis)")
            print(msg + " -> forced to 0")
            z = np.nan_to_num(z)

        x_norm[idx] = torch.from_numpy(z)

        # sanity check 2: mean ~0 and std ~sqrt((n-1)/n) *per run*, not over the whole
        # tensor. The target is not 1 because of ddof=1 above: 0.98953 on a 48-trial run.
        expected_std = np.sqrt((len(idx) - 1) / len(idx))
        assert np.abs(z.mean(axis=0)).max() < 1e-3, f"run {run}: per-run mean is not 0"
        assert np.abs(z.std(axis=0) - expected_std).max() < 1e-3, \
            f"run {run}: per-run std is not {expected_std:.5f} (ddof=1 on {len(idx)} trials)"
    return x_norm


def group_by_stimulus(x_norm: torch.Tensor, cond: str, exps: np.ndarray,
                      cues: np.ndarray, image_map: dict):
    """Select the trials of one condition and nest them over repetitions.

    Returns (n_stimuli, n_reps, V) and the stimulus indices, sorted ascending. nest=True is
    the shape recon_inference_imagery.py wants: it does the mean over the repetition axis
    itself, one repetition at a time through ridge+backbone.
    """
    idx = np.arange(len(exps))[np.isin(exps, COND_RUNS[cond])]
    conditionals = np.array([image_map[c] for c in cues[idx]])
    x_cond = x_norm[torch.from_numpy(idx).long()]

    stim_ids = np.unique(conditionals)
    reps = [x_cond[torch.from_numpy(np.flatnonzero(conditionals == s)).long()] for s in stim_ids]
    n_reps = min(r.shape[0] for r in reps)
    # Truncating to min(n_reps) instead of zero-padding to max (as MIRAGE does): with
    # NSD-Imagery the counts are uniform; the two coincide.
    x_out = torch.stack([r[:n_reps] for r in reps])          # (n_stimuli, n_reps, V)
    return x_out, stim_ids


def write_ground_truth(data_root: Path, out_dir: Path) -> None:
    """gt_18.pt (all stimuli) and gt_12.pt (the simple+complex distractor pool)."""
    stim_file = data_root / "nsddata_stimuli/stimuli/imagery_stimuli_18.pt"
    if not stim_file.exists():
        sys.exit(f"missing file: {stim_file}\nrun download_nsd_imagery.sh first")

    s = torch.load(stim_file).float()
    print(f"imagery_stimuli_18: {tuple(s.shape)} {s.dtype} "
          f"min {float(s.min()):.3f} max {float(s.max()):.3f}")
    assert tuple(s.shape) == (18, 3, 425, 425), f"unexpected stimulus shape {tuple(s.shape)}"
    if float(s.max()) > 1.5:
        print("  range looks like [0,255] -> dividing by 255")
        s = s / 255.0
    assert 0.0 <= float(s.min()) and float(s.max()) <= 1.0 + 1e-4, "ground truth outside [0,1]"

    torch.save(s, out_dir / "gt_18.pt")
    torch.save(s[:12], out_dir / "gt_12.pt")
    print(f"wrote ground truth tensors to {out_dir}")


def compare_reference(subject: int, x: torch.Tensor, x_norm: torch.Tensor,
                      reference_dir: Path | None, allow_missing: bool) -> None:
    """torch.allclose against Kneeland et al.'s own preprocessed tensors.

    Both tensors are compared, and which one fails localizes the bug: the unnormalized one
    covers the mask, the volume preparation and the trial order; the normalized one covers
    only normalize_imagery_betas() on top of that. A raw MATCH with a normalized MISMATCH is
    a recipe error and nothing else.

    A missing reference is an error, not a skip: an absent file would look exactly like a
    passing check.
    """
    files = {"nsd_imagery_unnormalized.pt": x, "nsd_imagery.pt": x_norm}
    if reference_dir is None:
        if allow_missing:
            print("  no --reference_dir: cross-check DISABLED by the caller")
            return
        sys.exit("  --reference_dir is unset and --allow_missing_reference was not given.\n"
                 "  Download the authors' tensors (download_nsd_imagery.sh) or pass"
                 " --allow_missing_reference and say so in the thesis.")

    for name, mine in files.items():
        ref = reference_dir / f"preprocessed_data/subject{subject}/{name}"
        if not ref.exists():
            if allow_missing:
                print(f"  {name}: no reference at {ref}, cross-check DISABLED by the caller")
                continue
            sys.exit(f"  {name}: no reference tensor at {ref}.\n"
                     "  Download it (download_nsd_imagery.sh) or pass"
                     " --allow_missing_reference and say so in the thesis.")
        r = torch.load(ref).float()
        if r.shape != mine.shape:
            sys.exit(f"  {name}: reference shape {tuple(r.shape)} != ours {tuple(mine.shape)}")
        d = float((r - mine).abs().max())
        ok = torch.allclose(r, mine, atol=1e-3)
        print(f"  {name:30s} vs authors': max|diff| {d:.2e} -> {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            sys.exit(f"  {name} differs from the authors' tensor. If only the normalized one"
                     " fails, the mask and the trial order are fine and the bug is in"
                     " normalize_imagery_betas(): check ddof and the attention runs.")


def prepare_subject(subject: int, data_root: Path, out_dir: Path,
                    exps, cues, image_map, reference_dir: Path | None,
                    betas_hdf5_dir: Path | None,
                    nan_to_zero: bool, allow_missing_reference: bool) -> None:
    print(f"\n=== subject{subject} ===")
    subj_dir = out_dir / f"subject{subject}"
    subj_dir.mkdir(parents=True, exist_ok=True)

    x = extract_imagery_betas(subject, data_root)
    print(f"  masked betas {tuple(x.shape)}")
    assert x.shape == (N_TRIALS, NUM_VOXELS[subject]), \
        f"expected {(N_TRIALS, NUM_VOXELS[subject])}, got {tuple(x.shape)} " \
        "(wrong mask: surface instead of volume, or func1mm instead of func1pt8mm?)"

    # The check that matters most: the imagery voxel count against the hdf5 MindEye2 was
    # actually trained on. Run before anything else touches disk.
    if betas_hdf5_dir is not None:
        check_training_hdf5(subject, betas_hdf5_dir, NUM_VOXELS[subject])
    else:
        print("  --betas_hdf5_dir not given: SKIPPING the training-hdf5 voxel cross-check "
              "(the single riskiest check of the pipeline)")

    torch.save(x, subj_dir / "nsd_imagery_unnormalized.pt")

    x_norm = normalize_imagery_betas(x, exps, nan_to_zero=nan_to_zero)
    assert not torch.isnan(x_norm).any(), "NaN in the normalized tensor"
    torch.save(x_norm, subj_dir / "nsd_imagery.pt")
    print(f"  wrote nsd_imagery.pt {tuple(x_norm.shape)}")

    # Sanity check 5: the only one that compares against something not computed here, and
    # therefore the only one that can catch a wrong recipe rather than a wrong execution of it.
    compare_reference(subject, x, x_norm, reference_dir, allow_missing_reference)

    for cond in COND_RUNS:
        x_cond, stim_ids = group_by_stimulus(x_norm, cond, exps, cues, image_map)
        n_stim = len(COND_STIM_IDS[cond])
        # sanity check 7: (n_stim,8,V) for vision, (n_stim,16,V) for imagery
        assert x_cond.shape == (n_stim, COND_REPS[cond], NUM_VOXELS[subject]), \
            f"{cond}: got {tuple(x_cond.shape)}, expected {(n_stim, COND_REPS[cond], NUM_VOXELS[subject])}"
        # sanity check 8: the only misalignment that would pass silently
        assert np.array_equal(stim_ids, COND_STIM_IDS[cond]), \
            f"{cond}: stimulus indices {stim_ids} != {COND_STIM_IDS[cond]}"
        torch.save(x_cond, subj_dir / f"x_{cond}.pt")
        torch.save(torch.from_numpy(stim_ids), subj_dir / f"stim_ids_{cond}.pt")
        print(f"  {cond:16s} {tuple(x_cond.shape)}  stim_ids {stim_ids.tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Prepare NSD-Imagery betas for MindEye2")
    parser.add_argument("--subj", type=int, nargs="+", default=[1, 2, 5, 7], choices=[1, 2, 5, 7])
    parser.add_argument("--data_root", type=str, default="imagery_dataset",
                        help="where download_nsd_imagery.sh put nsddata / nsddata_betas / nsddata_stimuli")
    parser.add_argument("--out_dir", type=str, default="imagery_dataset/preprocessed_data")
    parser.add_argument("--reference_dir", type=str, default="imagery_dataset/reference",
                        help="the authors' preprocessed_data/ from HuggingFace, for the cross-check")
    parser.add_argument("--betas_hdf5_dir", type=str, default="mindeyev2_dataset",
                        help="dir with betas_all_subj0N_fp32_renorm.hdf5, for the voxel-count "
                             "cross-check against the training data (pass '' to skip)")
    parser.add_argument("--nan_to_zero", action="store_true",
                        help="force NaN produced by zero-variance voxels to 0 instead of failing")
    parser.add_argument("--allow_missing_reference", action="store_true",
                        help="downgrade a missing reference tensor from fatal to a warning")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = Path(args.reference_dir) if args.reference_dir else None
    betas_hdf5_dir = Path(args.betas_hdf5_dir) if args.betas_hdf5_dir else None

    exps, cues, image_map = load_metadata(data_root)
    print("image_map:", image_map)
    write_ground_truth(data_root, out_dir)

    for subject in args.subj:
        prepare_subject(subject, data_root, out_dir, exps, cues, image_map,
                        reference_dir, betas_hdf5_dir, args.nan_to_zero,
                        args.allow_missing_reference)

    print("\nall checks passed")


if __name__ == "__main__":
    main()
