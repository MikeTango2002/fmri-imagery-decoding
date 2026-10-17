"""Extract the NSD-Imagery calibration set per subject: the attention and vision-conceptual trials.

Inputs:  the raw NSD-Imagery beta volume (720 trials per subject)
Outputs: data/betas_trisect/nsd_imagery/subj0N/calibration/{betas.npy (336, V), run.npy (336,)}
         betas are /300 and z-scored per run (ddof=1), as in the evaluation set; run is attA, attB, attC or visC

The benchmark does not score these trials; regime calibration may use them, but label-blind: only statistics
that do not depend on which stimulus was shown (covariance, mean gain, moments). run.npy gives the run, never the
stimulus. The official normalized tensor holds the attention rows at zero; the same recipe is applied here
to the raw volume.

Usage (CPU only):
    python src/regime_calibration_extract.py --subj 1 2 5 7
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import zscore

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent  # sota_reimplementations/
sys.path.insert(0, str(_REPO_ROOT / "icnn" / "src"))

from prepare_nsd_imagery_icnn import (  # noqa: E402  (needs the sys.path insert above)
    COND_STIM_IDS, NUM_VOXELS, RUN_TYPES, extract_imagery_betas, load_metadata,
)

IMAGERY_DATA_ROOT = "data"
OUT_DIR = "data/betas_trisect"

# attA/attB/attC follow the same A=simple/B=complex/C=conceptual convention as vis*/img*
# (verified below, not assumed: see the stimulus-set assertion inside stage_calibration).
CALIB_RUNS = ["attA", "attB", "attC", "visC"]
CALIB_STIM_IDS = {
    "attA": COND_STIM_IDS["visionsimple"],
    "attB": COND_STIM_IDS["visioncomplex"],
    "attC": COND_STIM_IDS["visionconcepts"],
    "visC": COND_STIM_IDS["visionconcepts"],
}
CALIB_N_TRIALS = {"attA": 96, "attB": 96, "attC": 96, "visC": 48}  # 16x6 / 16x6 / 16x6 / 8x6
# The raw volume has 720 trials per subject, not the 576 of the paper's headline (which holds for vision and
# imagery only): each attention run has 96 trials (16 repetitions x 6 stimuli), not 48. The evaluation set is
# 48 + 48 + 288 = 384 trials; the other 336 (288 attention + 48 visionconcepts) are the calibration set.
# normalize_imagery_betas() of prepare_nsd_imagery_icnn.py leaves the 288 attention rows at zero, to match the
# tensor of Kneeland et al., and prepare_betas.py drops attention before writing. So the same recipe
# (/300, then zscore ddof=1 per run; checked against the authors' tensors, max|diff| 1.03e-05) is applied
# here to the raw volume, for visionconcepts too, to keep one normalization path.
N_CALIB_TOTAL = sum(CALIB_N_TRIALS.values())  # 336 = 720 - 384 (the evaluation set)


def normalize_calibration_run(x_scaled: np.ndarray, exps: np.ndarray, run: str):
    """Same recipe as prepare_nsd_imagery_icnn.normalize_imagery_betas, applied to one
    run: zscore(ddof=1) over that run's trials only. x_scaled is already /300."""
    idx = np.arange(len(exps))[exps == run]
    z = zscore(x_scaled[idx].numpy(), axis=0, ddof=1).astype(np.float32)
    n_nan = int(np.isnan(z).sum())
    if n_nan:
        sys.exit(f"run {run}: {n_nan} NaN after z-score ({n_nan // len(idx)} zero-variance "
                  "voxels) -- unexpected, the eval runs of the same subject/session don't "
                  "have this problem; rerun would need the same --nan_to_zero decision as "
                  "prepare_betas.py and a note in the thesis")
    expected_std = np.sqrt((len(idx) - 1) / len(idx))
    assert np.abs(z.mean(axis=0)).max() < 1e-3, f"run {run}: per-run mean is not 0"
    assert np.abs(z.std(axis=0) - expected_std).max() < 1e-3, \
        f"run {run}: per-run std is not {expected_std:.5f} (ddof=1 on {len(idx)} trials)"
    return idx, z


def check_against_unnormalized_reference(subj, x, exps, reference_dir: Path):
    """Cross-checks the raw (pre-z-score) attention+visC rows against Kneeland's own
    nsd_imagery_unnormalized.pt: the one reference tensor that is NOT zeroed for attention
    (only the normalized nsd_imagery.pt zeroes the attention rows, see NORM_RUN_TYPES in
    normalize_imagery_betas). This validates the extraction of this script (mask, trial order)
    against the authors' tensor, independently of its own normalization step."""
    cands = [reference_dir / f"subject{subj}" / "nsd_imagery_unnormalized.pt",
             reference_dir / f"subj0{subj}" / "nsd_imagery_unnormalized.pt",
             reference_dir / f"preprocessed_data/subject{subj}" / "nsd_imagery_unnormalized.pt"]
    ref = next((c for c in cands if c.exists()), None)
    if ref is None:
        print(f"  no unnormalized reference found for subj{subj:02d}, skipping cross-check "
              f"(looked in: {[str(c) for c in cands]})")
        return
    ref_x = torch.load(ref).float()
    if ref_x.shape != x.shape:
        sys.exit(f"  reference shape {tuple(ref_x.shape)} != ours {tuple(x.shape)}")
    for run in CALIB_RUNS:
        idx = np.arange(len(exps))[exps == run]
        d = float((ref_x[idx] - x[idx]).abs().max())
        print(f"  {run:6s} raw vs reference: max|diff| {d:.2e}")
        if d > 1e-3:
            sys.exit(f"  {run}: raw extraction diverges from the authors' reference tensor "
                      "beyond tolerance -- mask or trial order is wrong for this run")


def stage_calibration(subjects, data_root: Path, out_dir: Path):
    out_root = out_dir / "nsd_imagery"
    exps, cues, image_map = load_metadata(data_root, data_root)
    reference_dir = data_root / "reference" / "preprocessed_data"

    counts = {r: int((exps == r).sum()) for r in RUN_TYPES}
    print("trials per run (raw, 720 total):", counts)
    assert counts["attA"] == counts["attB"] == counts["attC"] == 96, \
        f"expected 96 attention trials/run (16 reps x 6 stim), got {counts}"
    assert counts["visC"] == 48, f"expected 48 visionconcepts trials, got {counts}"
    assert sum(counts.values()) == 720, f"expected 720 raw trials total, got {sum(counts.values())}"
    assert sum(CALIB_N_TRIALS.values()) == N_CALIB_TOTAL == 336

    for subj in subjects:
        print(f"\n=== NSD-Imagery calibration subj0{subj} ===")
        subj_out = out_root / f"subj{subj:02d}" / "calibration"
        subj_out.mkdir(parents=True, exist_ok=True)

        x = extract_imagery_betas(subj, data_root)  # (720, V), raw beta units (no /300 yet)
        assert x.shape == (720, NUM_VOXELS[subj]), \
            f"{x.shape} vs expected (720, {NUM_VOXELS[subj]})"

        check_against_unnormalized_reference(subj, x, exps, reference_dir)

        x_scaled = x / 300.0
        rows, run_ids = [], []
        for run in CALIB_RUNS:
            idx, z = normalize_calibration_run(x_scaled, exps, run)
            assert len(idx) == CALIB_N_TRIALS[run], (run, len(idx), CALIB_N_TRIALS[run])
            stim = np.array([image_map[c] for c in cues[idx]])
            assert set(stim.tolist()) == set(CALIB_STIM_IDS[run].tolist()), \
                (f"{run}: stimulus indices {sorted(set(stim.tolist()))} != "
                 f"{CALIB_STIM_IDS[run].tolist()} -- the A/B/C = simple/complex/conceptual "
                 "convention assumed for attention runs does not hold, fix CALIB_STIM_IDS")
            rows.append(z)
            run_ids += [run] * len(idx)

        betas = np.vstack(rows).astype(np.float32)
        run_arr = np.array(run_ids)
        assert betas.shape == (N_CALIB_TOTAL, NUM_VOXELS[subj]), betas.shape
        assert len(run_arr) == N_CALIB_TOTAL
        assert not np.isnan(betas).any(), "betas has NaN after normalization"

        np.save(subj_out / "betas.npy", betas)
        np.save(subj_out / "run.npy", run_arr)
        run_counts = dict(zip(*np.unique(run_arr, return_counts=True)))
        print(f"  wrote {subj_out}  betas {betas.shape}, runs {run_counts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", type=int, nargs="+", default=[1, 2, 5, 7], choices=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--imagery_data_root", type=str, default=IMAGERY_DATA_ROOT)
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    stage_calibration(args.subj, Path(args.imagery_data_root), Path(args.out_dir))
    print("\nall done")


if __name__ == "__main__":
    main()
