"""Prepare the two beta sources TRISECT trains and evaluates on, plus the reliability scalar `r`.

Inputs:  --stage core: betas_all_subj0N_fp32_renorm.hdf5 (read-only) and the index of `prepare_nsdcore_index.py`;
         --stage imagery: the raw NSD-Imagery tree
Outputs: core: data/betas_trisect/subj0N_r.npy, (30000,). imagery: betas_flat / betas_averaged / labels_* / r /
         r_per_rep .npy per subject

`r` is the leave-one-out correlation (`loo_correlations()`): per trial against the mean of the other 2
repetitions for NSD-core, per condition (mean over its 8/16 repetitions) for NSD-Imagery. It uses the test betas
only, no labels and no tuning. The core stage does not copy the (30000, V) betas. The imagery recipe is
imported verbatim from the iCNN reproduction (`prepare_nsd_imagery_icnn.py`, verified against the authors'
tensors to max|diff| 1.03e-05), and `write_ground_truth` from the MindEye1 one (`prepare_nsd_imagery.py`).

Usage (from the trisect/ folder; the imagery stage needs download_nsd_imagery.sh first):
    python scripts/prepare_betas.py --stage both --subj 1 2 5 7
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent  # sota_reimplementations/
sys.path.insert(0, str(_REPO_ROOT / "icnn" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "mindeye_I" / "src"))

from prepare_nsd_imagery_icnn import (  # noqa: E402  (needs the sys.path insert above)
    EVAL_CONDS, N_EVAL_TRIALS, NUM_VOXELS,
    build_rows, compare_reference, extract_imagery_betas, load_metadata, normalize_imagery_betas,
)
from prepare_nsd_imagery import write_ground_truth  # noqa: E402  (mindeye_I's, single data_root)

NSDCORE_INDEX_DIR = "data/nsdcore_index"
NSDCORE_BETAS_DIR = "data/mindeyev2_dataset"
NSDCORE_N_REPS = 3

IMAGERY_DATA_ROOT = "data"
OUT_DIR = "data/betas_trisect"


def loo_correlations(group: np.ndarray) -> np.ndarray:
    """(n_reps, V) -> (n_reps,): per-rep Pearson corr(rep_i, mean of the OTHER reps).

    Vectorized leave-one-out: mean-of-others_i = (sum - rep_i) / (n-1), computed once for the
    whole group rather than per rep; this stays cheap even at NSD-core's ~10868 groups.
    """
    n = group.shape[0]
    assert n >= 2, f"need >=2 reps for a leave-one-out correlation, got {n}"
    total = group.sum(axis=0, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        a = group[i].astype(np.float64)
        b = (total - group[i]) / (n - 1)
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        out[i] = (a * b).sum() / denom if denom > 0 else np.nan
    return out.astype(np.float32)


def stage_core(subjects, index_dir: Path, betas_dir: Path, out_dir: Path):
    out_dir = out_dir / "nsdcore_r"
    out_dir.mkdir(parents=True, exist_ok=True)

    for subj in subjects:
        print(f"\n=== NSD-core subj0{subj} ===")
        index_file = index_dir / f"subj{subj:02d}_trial73k_index.npz"
        if not index_file.exists():
            sys.exit(f"missing {index_file}: run prepare_nsdcore_index.py first")
        idx = np.load(index_file)
        img73k = idx["img73k"]  # (30000,) int32, 0-based into the 73k images

        betas_file = betas_dir / f"betas_all_subj{subj:02d}_fp32_renorm.hdf5"
        if not betas_file.exists():
            sys.exit(f"missing {betas_file}: run download_data_from_internet.sh --betas first")
        with h5py.File(betas_file, "r") as h:
            betas = np.asarray(h["betas"])  # (30000, V) float32
        assert betas.shape[0] == img73k.shape[0], \
            f"{betas.shape[0]} beta rows vs {img73k.shape[0]} index rows"
        assert betas.shape[1] == NUM_VOXELS[subj], \
            f"{betas.shape[1]} voxels, expected {NUM_VOXELS[subj]} (nsdgeneral mask mismatch)"

        r = np.full(betas.shape[0], np.nan, dtype=np.float32)
        uniq_images, counts = np.unique(img73k, return_counts=True)
        bad = int((counts != NSDCORE_N_REPS).sum())
        if bad:
            sys.exit(f"  {bad}/{len(uniq_images)} images do not have exactly "
                      f"{NSDCORE_N_REPS} repetitions -- unexpected for a complete subject "
                      f"(1/2/5/7), stopping rather than computing r on a partial group")

        for img_id in uniq_images:
            rows = np.flatnonzero(img73k == img_id)
            r[rows] = loo_correlations(betas[rows])
        del betas

        assert not np.isnan(r).any(), "r has NaN rows left -- some image's group was missed"
        print(f"  r: mean {r.mean():.4f}  min {r.min():.4f}  max {r.max():.4f}  "
              f"(over {len(uniq_images)} images x {NSDCORE_N_REPS} reps = {len(r)} trials)")

        out_file = out_dir / f"subj{subj:02d}_r.npy"
        np.save(out_file, r)
        print(f"  wrote {out_file}  (row order matches betas_all_subj{subj:02d}_fp32_renorm.hdf5)")


def stage_imagery(subjects, data_root: Path, out_dir: Path):
    out_dir = out_dir / "nsd_imagery"
    out_dir.mkdir(parents=True, exist_ok=True)

    exps, cues, image_map = load_metadata(data_root, data_root)
    print("image_map:", image_map)
    write_ground_truth(data_root, out_dir)

    reference_dir = data_root / "reference" / "preprocessed_data"

    for subj in subjects:
        print(f"\n=== NSD-Imagery subj0{subj} ===")
        subj_dir = out_dir / f"subj{subj:02d}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        x = extract_imagery_betas(subj, data_root)
        x_norm = normalize_imagery_betas(x, exps)
        assert not torch.isnan(x_norm).any(), "NaN in the normalized tensor"
        compare_reference(subj, x, x_norm, reference_dir, allow_missing=False)

        X, labels, uniq_labels = build_rows(x_norm, exps, cues, image_map)
        assert X.shape == (N_EVAL_TRIALS, NUM_VOXELS[subj])
        np.save(subj_dir / "betas_flat.npy", X)
        np.save(subj_dir / "labels_flat.npy", np.array(labels))
        print(f"  betas_flat {X.shape}, {len(uniq_labels)} distinct labels "
              f"({uniq_labels[0]} ... {uniq_labels[-1]})")

        labels_arr = np.array(labels)
        n_cond = len(uniq_labels)
        X_avg = np.empty((n_cond, X.shape[1]), dtype=np.float32)
        r = np.empty(n_cond, dtype=np.float32)
        # loo_correlations(group) returns one value per repetition. r keeps only its mean;
        # the per-repetition values are saved too, row-aligned to betas_flat.npy/labels_flat.npy,
        # for the precision-weighted fusion of repetitions (`weighted_repetition_fusion.py`).
        # The per-condition r.npy is the one `reconstruct.py` consumes.
        r_per_rep = np.full(X.shape[0], np.nan, dtype=np.float32)
        for i, lbl in enumerate(uniq_labels):
            rows = np.flatnonzero(labels_arr == lbl)
            group = X[rows]
            X_avg[i] = group.mean(axis=0)
            rep_r = loo_correlations(group)
            r[i] = float(rep_r.mean())
            r_per_rep[rows] = rep_r

        assert not np.isnan(r_per_rep).any(), "r_per_rep has NaN rows left -- some trial was missed"
        np.save(subj_dir / "betas_averaged.npy", X_avg)
        np.save(subj_dir / "labels_averaged.npy", np.array(uniq_labels))
        np.save(subj_dir / "r.npy", r)
        np.save(subj_dir / "r_per_rep.npy", r_per_rep)
        print(f"  betas_averaged {X_avg.shape}")
        print(f"  r: mean {r.mean():.4f}  min {r.min():.4f}  max {r.max():.4f}  "
              f"(one scalar per condition, EVAL_CONDS = {EVAL_CONDS})")
        print(f"  r_per_rep: mean {r_per_rep.mean():.4f}  min {r_per_rep.min():.4f}  "
              f"max {r_per_rep.max():.4f}  (row-aligned to betas_flat.npy/labels_flat.npy)")
        print(f"  wrote {subj_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["core", "imagery", "both"], default="both")
    ap.add_argument("--subj", type=int, nargs="+", default=[1, 2, 5, 7], choices=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--nsdcore_index_dir", type=str, default=NSDCORE_INDEX_DIR)
    ap.add_argument("--nsdcore_betas_dir", type=str, default=NSDCORE_BETAS_DIR)
    ap.add_argument("--imagery_data_root", type=str, default=IMAGERY_DATA_ROOT)
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.stage in ("core", "both"):
        stage_core(args.subj, Path(args.nsdcore_index_dir), Path(args.nsdcore_betas_dir), out_dir)
    if args.stage in ("imagery", "both"):
        stage_imagery(args.subj, Path(args.imagery_data_root), out_dir)

    print("\nall done")


if __name__ == "__main__":
    main()
