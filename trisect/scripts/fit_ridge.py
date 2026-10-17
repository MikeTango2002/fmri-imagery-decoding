"""Closed-form ridge decoder voxels -> (z_sem_img, z_sem_txt, z_struct, z_low), per subject, NSD-core only.
Four independent branches, one ridge each.

Inputs:  targets.h5 (z_sem, z_struct), data/targets/pca_zlow/z_low.npy, betas of `prepare_betas.py`
Outputs: data/betas_trisect/ridge_weights/subjNN/ (weights per branch, shared1000 predictions, train_target_stats.npz)

z_sem_txt is decoded from the brain too: the adapter takes both z_sem streams as separate tokens.
Normalization is Brain-Diffuser's: fmri/300 z-scored with the train moments, predictions re-standardized onto
the train targets' moments. One departure from upstream: it fits one Ridge per output token at a hand-set
alpha; here alpha is chosen by CV, and a branch is one multi-output RidgeCV call. Trials are single-trial
(27000 train; the 3000 shared1000 trials are averaged to 1000 images for the check). The check is two-way
identification, not R2 (a healthy fit on ~90k-dim targets has R2 ~ -0.02 and two-way 0.885).
--exclude-dev also holds the dev split out (use a non-default --out-dir). On a Sapphire Rapids CPU export
OPENBLAS_CORETYPE=SkylakeX: sklearn's SVD path is silently wrong there. Usage (from the trisect/ folder):
    python scripts/fit_ridge.py --subj 1 --n-train-subsample 3000 --n-alphas 5   # smoke test
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, GroupKFold

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent  # sota_reimplementations/
sys.path.insert(0, str(_REPO_ROOT / "icnn" / "src"))
from prepare_nsd_imagery_icnn import NUM_VOXELS  # noqa: E402  (needs the sys.path insert above)

# The fMRI z-score and the prediction re-standardization are shared (trisect/), not local.
# Other scripts import zscore_fmri/restandardize from trisect.nsd / trisect.ridge directly.
from trisect.nsd import zscore_fmri_train_test, zscore_fmri_arrays  # noqa: E402
from trisect.ridge import restandardize_from_pred  # noqa: E402

SUBJECTS = (1, 2, 5, 7)
N_SHARED = 1000
Z_SEM_DIM = 128

INDEX_DIR = "data/nsdcore_index"
BETAS_DIR = "data/mindeyev2_dataset"
TARGETS_H5 = "data/targets/targets.h5"
ZLOW_PATH = "data/targets/pca_zlow/z_low.npy"
OUT_DIR = "data/betas_trisect/ridge_weights"
DEV_MASK_DIR = "data/targets"  # nsdcore_dev_mask_subj0N.npy, from derive_nsdcore_dev_mask.py

DEFAULT_ALPHAS = np.logspace(1, 6, 21)


def two_way(gt, pred):
    """Same formula as diag_A_03_ridge.py:99-102 / evaluate_reconstruction.py:22. 0.50 = chance."""
    r = np.corrcoef(gt, pred)[:len(gt), len(gt):]
    return float(np.mean(np.sum(r < np.diag(r), 0)) / (len(gt) - 1))


def two_way_per_image(gt, pred):
    """Same computation as two_way(), but returns the per-image vector instead of its mean.

    two_way()'s own body is left untouched (a shared body is not edited by a caller that only
    needs a different reduction of it); this duplicates the four lines. The paired comparisons
    need the per-image vector for a cluster bootstrap (bootstrap_ci_imagery.bootstrap,
    scheme="iid": every image is its own cluster).
    """
    r = np.corrcoef(gt, pred)[:len(gt), len(gt):]
    return np.sum(r < np.diag(r), 0) / (len(gt) - 1)


def load_branch_targets(targets_h5_path, zlow_path):
    """All four branches, indexed by the same 73k id used throughout targets.h5/pca_zlow/."""
    print(f"Loading targets: {targets_h5_path}")
    with h5py.File(targets_h5_path, "r") as f:
        z_sem = f["z_sem"][:].astype(np.float32)          # (73000, 2, 128)
        z_struct = f["z_struct"][:].astype(np.float32)    # (73000, 8, 8, 32)
    print(f"Loading z_low: {zlow_path}")
    z_low = np.load(zlow_path).astype(np.float32)          # (73000, d_low)

    n = z_sem.shape[0]
    assert z_sem.shape == (n, 2, Z_SEM_DIM), f"z_sem shape {z_sem.shape}"
    assert z_struct.shape[0] == n and z_struct.shape[1:] == (8, 8, 32), \
        f"z_struct shape {z_struct.shape}"
    assert z_low.shape[0] == n, f"z_low has {z_low.shape[0]} rows, targets have {n}"

    d_low = z_low.shape[1]
    return {
        "z_sem_img": (z_sem[:, 0, :], (Z_SEM_DIM,)),
        "z_sem_txt": (z_sem[:, 1, :], (Z_SEM_DIM,)),
        "z_struct": (z_struct.reshape(n, -1), (8, 8, 32)),
        "z_low": (z_low, (d_low,)),
    }


def load_nsdcore_split(subj, index_dir: Path, betas_dir: Path):
    index_file = index_dir / f"subj{subj:02d}_trial73k_index.npz"
    if not index_file.exists():
        sys.exit(f"missing {index_file}: run prepare_nsdcore_index.py first")
    idx = np.load(index_file)
    img73k, is_shared = idx["img73k"], idx["is_shared"]

    betas_file = betas_dir / f"betas_all_subj{subj:02d}_fp32_renorm.hdf5"
    if not betas_file.exists():
        sys.exit(f"missing {betas_file}: run download_data_from_internet.sh --betas first")
    with h5py.File(betas_file, "r") as h:
        betas = np.asarray(h["betas"], dtype=np.float32)  # (30000, V)
    assert betas.shape[0] == img73k.shape[0]
    assert betas.shape[1] == NUM_VOXELS[subj], \
        f"{betas.shape[1]} voxels, expected {NUM_VOXELS[subj]} (nsdgeneral mask mismatch)"

    return betas, img73k, is_shared


def average_by_image(values, img_ids):
    """(n, D), (n,) -> (n_unique, D) mean-pooled per unique id, and the sorted unique ids."""
    uniq_ids, inverse = np.unique(img_ids, return_inverse=True)
    sums = np.zeros((len(uniq_ids),) + values.shape[1:], dtype=np.float64)
    np.add.at(sums, inverse, values)
    counts = np.bincount(inverse).astype(np.float64)
    counts = counts.reshape((-1,) + (1,) * (values.ndim - 1))
    return (sums / counts).astype(np.float32), uniq_ids


def fit_ridge_group_cv(train_fmri, train_target, alphas, groups, n_splits, n_jobs):
    """Alpha selection by explicit GroupKFold instead of RidgeCV's default LOO.

    RidgeCV(cv=<splitter>) (rather than cv=None) does NOT keep the fast LOO-via-SVD path --
    sklearn's own _BaseRidgeCV.fit() falls back to plain GridSearchCV(Ridge(), ..., cv=cv),
    i.e. one from-scratch Ridge().fit() per (fold, alpha) pair, with no way to pass n_jobs
    through RidgeCV itself. Calling GridSearchCV directly (same computation RidgeCV would do
    internally) exposes n_jobs, kept here for that reason, but the DEFAULT is n_jobs=1: a
    single Ridge().fit() at this branch's size (p~15700 voxels, n~27000 single trials) is
    already BLAS-multithreaded internally (measured ~12s/fit on a 24-core CPU; 5 folds
    x 25 alphas = 125 fits ~25 min), running several of THOSE in separate joblib processes at
    once oversubscribes the same core pool the BLAS threads are already using and is slower,
    not faster, unless BLAS threading is pinned down to 1 first (OPENBLAS_NUM_THREADS=1).
    scoring='neg_mean_squared_error' is explicit, not the R^2 default some scikit-learn
    versions use for regressors; same "R^2 is forbidden on this target dimensionality"
    reason as this module's own two_way() gate, docstring above.
    """
    cv = GroupKFold(n_splits=n_splits)
    gs = GridSearchCV(Ridge(fit_intercept=True), param_grid={"alpha": list(alphas)},
                       cv=cv, scoring="neg_mean_squared_error", n_jobs=n_jobs, refit=True)
    gs.fit(train_fmri, train_target, groups=groups)
    model = gs.best_estimator_
    model.alpha_ = gs.best_params_["alpha"]  # mimic RidgeCV's attribute name
    return model


def fit_branch(name, train_fmri, test_fmri, train_target, test_target,
               train_img73k, test_img73k, alphas, out_shape, out_dir: Path,
               cv_groups=None, group_cv_splits=5, group_cv_n_jobs=1):
    print(f"  [{name}] fitting RidgeCV over {len(alphas)} alphas "
          f"(train {train_fmri.shape}, target dim {train_target.shape[1]}) ...", flush=True)
    t0 = time.monotonic()
    if cv_groups is not None:
        # At single-trial granularity RidgeCV's default LOO leaks the other 2 reps of the SAME
        # image (identical target row) into validation and pushes alpha too low. GroupKFold on
        # the 73k image id closes that leak.
        model = fit_ridge_group_cv(train_fmri, train_target, alphas, cv_groups,
                                    group_cv_splits, group_cv_n_jobs)
    else:
        model = RidgeCV(alphas=alphas, fit_intercept=True)
        model.fit(train_fmri, train_target)
    fit_s = time.monotonic() - t0
    print(f"  [{name}] alpha={model.alpha_:.4g}  ({fit_s:.1f}s)", flush=True)

    pred_test = model.predict(test_fmri)
    pred_test = restandardize_from_pred(pred_test, train_target)

    pred_avg, uniq_ids = average_by_image(pred_test, test_img73k)
    gt_avg, uniq_ids_gt = average_by_image(test_target, test_img73k)
    assert np.array_equal(uniq_ids, uniq_ids_gt)
    twc = two_way(gt_avg.reshape(len(uniq_ids), -1), pred_avg.reshape(len(uniq_ids), -1))
    print(f"  [{name}] 2WC on {len(uniq_ids)} shared1000 images: {twc:.4f} "
          f"({'PASS' if twc > 0.55 else 'WARN, close to chance (0.50)'})", flush=True)

    weights_path = out_dir / f"{name}_weights.pkl"
    with open(weights_path, "wb") as f:
        pickle.dump({"weight": model.coef_, "bias": model.intercept_,
                     "alpha": float(model.alpha_)}, f)

    pred_path = out_dir / f"pred_shared1000_{name}.npy"
    np.save(pred_path, pred_avg.reshape((len(uniq_ids),) + out_shape))

    return {
        "alpha": float(model.alpha_),
        "fit_seconds": fit_s,
        "two_way_shared1000": twc,
        "train_target_mean": train_target.mean(axis=0),
        "train_target_std": train_target.std(axis=0),
    }


def process_subject(subj, branches, alphas, index_dir, betas_dir, out_dir: Path,
                     n_train_subsample=None, exclude_dev=False, dev_mask_dir=None,
                     average_reps=False, group_cv=False, group_cv_splits=5,
                     group_cv_n_jobs=1):
    print(f"\n=== subj0{subj} ===")
    betas, img73k, is_shared = load_nsdcore_split(subj, index_dir, betas_dir)
    train_mask, test_mask = ~is_shared, is_shared

    if exclude_dev:
        dev_mask_path = Path(dev_mask_dir) / f"nsdcore_dev_mask_subj{subj:02d}.npy"
        if not dev_mask_path.exists():
            sys.exit(f"missing {dev_mask_path}: run derive_nsdcore_dev_mask.py first")
        dev_mask_73k = np.load(dev_mask_path)  # (73000,) bool, indexed by img73k id
        dev_trial_mask = dev_mask_73k[img73k]  # (30000,) bool, per-trial
        assert not np.any(dev_trial_mask & is_shared), "dev mask overlaps shared1000"
        train_mask = train_mask & ~dev_trial_mask
        print(f"  --exclude-dev: holding out {int(dev_trial_mask.sum())} dev trials "
              f"({int(dev_mask_73k.sum())} dev images) from training")

    if average_reps:
        # Average the RAW betas of both train and test over the repeated trials of the same 73k
        # image id BEFORE z-scoring: the z-score moments must come from the averaged set, never
        # from the single-trial one (a run that did it the other way round lost 9.86 of
        # 12.438 pp).
        train_betas_avg, train_img73k = average_by_image(betas[train_mask], img73k[train_mask])
        test_betas_avg, test_img73k = average_by_image(betas[test_mask], img73k[test_mask])
        train_fmri, test_fmri = zscore_fmri_arrays(train_betas_avg, test_betas_avg)
        del betas, train_betas_avg, test_betas_avg
        cv_groups = None  # rows are already unique per image; LOO has nothing to leak
        print(f"  --average-reps: {int(train_mask.sum())} train trials -> {len(train_fmri)} "
              f"images, {int(test_mask.sum())} test trials -> {len(test_fmri)} images")
    else:
        train_fmri, test_fmri = zscore_fmri_train_test(betas, train_mask, test_mask)
        del betas
        train_img73k, test_img73k = img73k[train_mask], img73k[test_mask]
        cv_groups = train_img73k if group_cv else None
        if group_cv:
            print(f"  --group-cv: GroupKFold({group_cv_splits}) on the 73k image id "
                  f"instead of RidgeCV's default LOO")

    if n_train_subsample is not None and n_train_subsample < len(train_fmri):
        rng = np.random.default_rng(0)
        sel = rng.choice(len(train_fmri), size=n_train_subsample, replace=False)
        train_fmri, train_img73k = train_fmri[sel], train_img73k[sel]
        if cv_groups is not None:
            cv_groups = cv_groups[sel]
        print(f"  --n-train-subsample: using {n_train_subsample} of the {len(train_mask)-int((~train_mask).sum())} "
              f"train trials (smoke test only)")

    subj_out = out_dir / f"subj{subj:02d}"
    subj_out.mkdir(parents=True, exist_ok=True)

    stats, results = {}, {}
    for name, (target_full, out_shape) in branches.items():
        train_target = target_full[train_img73k]
        test_target = target_full[test_img73k]
        r = fit_branch(name, train_fmri, test_fmri, train_target, test_target,
                        train_img73k, test_img73k, alphas, out_shape, subj_out,
                        cv_groups=cv_groups, group_cv_splits=group_cv_splits,
                        group_cv_n_jobs=group_cv_n_jobs)
        stats[f"{name}_mean"] = r.pop("train_target_mean")
        stats[f"{name}_std"] = r.pop("train_target_std")
        results[name] = r

    np.savez(subj_out / "train_target_stats.npz", **stats)
    with open(subj_out / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  wrote {subj_out}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--targets", type=str, default=TARGETS_H5)
    ap.add_argument("--zlow", type=str, default=ZLOW_PATH)
    ap.add_argument("--index-dir", type=str, default=INDEX_DIR)
    ap.add_argument("--betas-dir", type=str, default=BETAS_DIR)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--n-alphas", type=int, default=len(DEFAULT_ALPHAS),
                     help="size of the log-spaced alpha grid, range 1e1-1e6 (default 21)")
    ap.add_argument("--n-train-subsample", type=int, default=None,
                     help="subsample the train trials before fitting -- smoke test only, "
                          "never for the real numbers")
    ap.add_argument("--branches", type=str, nargs="+", default=None,
                     choices=["z_sem_img", "z_sem_txt", "z_struct", "z_low"],
                     help="fit only these branches instead of all four (e.g. a z_low-only "
                          "representation check against --zlow)")
    ap.add_argument("--exclude-dev", action="store_true",
                     help="also hold out each subject's NSD-core dev split "
                          "(nsdcore_dev_mask_subj0N.npy, derive_nsdcore_dev_mask.py) from "
                          "training, on top of shared1000 -- the held-out split for calibration; "
                          "use a non-default --out-dir so this refit does not overwrite the "
                          "production weights")
    ap.add_argument("--dev-mask-dir", type=str, default=DEV_MASK_DIR,
                     help="directory holding nsdcore_dev_mask_subj0N.npy")
    ap.add_argument("--alphas-logspace", type=float, nargs=3, default=None,
                     metavar=("START", "STOP", "NUM"),
                     help="override the alpha grid as np.logspace(START, STOP, int(NUM)) "
                          "instead of the production np.logspace(1, 6, --n-alphas) "
                          "('1 7 25' widens the top of the grid where z_low's alpha sits)")
    ap.add_argument("--average-reps", action="store_true",
                     help="average NSD-core betas (train AND test) over the repeated trials "
                          "of the same 73k image id BEFORE z-scoring, instead of training on "
                          "27000 single trials (raises the ridge's input SNR)")
    ap.add_argument("--group-cv", action="store_true",
                     help="select alpha with GroupKFold on the 73k image id instead of "
                          "RidgeCV's default LOO -- only meaningful WITHOUT --average-reps, "
                          "where LOO otherwise leaks the other reps of the same image "
                          "(identical target row) into validation")
    ap.add_argument("--group-cv-splits", type=int, default=5,
                     help="number of GroupKFold splits when --group-cv is set")
    ap.add_argument("--group-cv-n-jobs", type=int, default=1,
                     help="joblib n_jobs for the (fold x alpha) grid search when --group-cv "
                          "is set (default 1: a single Ridge fit at this branch's size is "
                          "already BLAS-multithreaded, see fit_ridge_group_cv() docstring -- "
                          "only raise this AND pin OPENBLAS_NUM_THREADS=1 together)")
    args = ap.parse_args()

    if os.environ.get("OPENBLAS_CORETYPE") != "SkylakeX":
        print("WARNING: OPENBLAS_CORETYPE is not SkylakeX -- if this is running on a Sapphire "
              "Rapids server, export it first or the ridge solve can be silently wrong. "
              "Harmless elsewhere.", file=sys.stderr)

    if args.alphas_logspace:
        start, stop, num = args.alphas_logspace
        alphas = np.logspace(start, stop, int(num))
    else:
        alphas = np.logspace(1, 6, args.n_alphas)
    branches = load_branch_targets(args.targets, args.zlow)
    if args.branches:
        branches = {k: v for k, v in branches.items() if k in args.branches}

    all_results = {}
    for subj in args.subj:
        all_results[subj] = process_subject(
            subj, branches, alphas, Path(args.index_dir), Path(args.betas_dir),
            Path(args.out_dir), n_train_subsample=args.n_train_subsample,
            exclude_dev=args.exclude_dev, dev_mask_dir=args.dev_mask_dir,
            average_reps=args.average_reps, group_cv=args.group_cv,
            group_cv_splits=args.group_cv_splits, group_cv_n_jobs=args.group_cv_n_jobs)

    print("\n=== summary (2WC on shared1000, 0.50 = chance) ===")
    for subj, results in all_results.items():
        row = "  ".join(f"{name}={r['two_way_shared1000']:.3f}" for name, r in results.items())
        print(f"  subj0{subj}: {row}")

    print("\nall done")


if __name__ == "__main__":
    main()
