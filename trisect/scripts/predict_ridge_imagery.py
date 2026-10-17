"""Apply the ridge weights to NSD-Imagery betas: z_sem_img / z_sem_txt / z_struct / z_low predictions for the 30
(condition, stimulus) rows of each subject, ready for `reconstruct.py`.

Inputs:  nsd_imagery/subj0N/{betas_averaged,labels_averaged}.npy (`prepare_betas.py --stage imagery`; 5
         conditions x 6 stimuli), ridge_weights/ and train_target_stats.npz (`fit_ridge.py`)
Outputs: data/betas_trisect/imagery_predictions/subj0N/{branch}_pred.npy, (30,) + the branch shape, and labels.npy

Mirror of Brain-Diffuser's `scripts/imagery_regression_predict.py`, with its two decisions: (a) no second input
z-score, because the betas are already /300 and per-run z-scored; (b) the predictions ARE re-standardized onto
the train targets' moments (`restandardize()`), because at alpha 1e4-1e6 the ridge shrinks them below what the
adapter and VDVAE expect. Repetitions are averaged BEFORE `restandardize()`: the other order leaves the rows at
std tgt_std/sqrt(n_eff). --trial-level averages the predictions instead (the same for a ridge, which is affine,
but not for the MLP encoder). The pool of `restandardize()` is open upstream: default 'modality12' (vision and
imagery pooled separately, imageryconcepts alone); 'fromA' freezes it to the shared1000 predictions.
Usage (from the trisect/ folder): python scripts/predict_ridge_imagery.py --subj 1 2 5 7
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Both re-standardization variants live in trisect/ridge.py; their docstrings carry the argument
# for the --restandardize-pool / --restandardize-scalar arms below.
from trisect.ridge import dim_gain, restandardize, restandardize_gain, restandardize_scalar

SUBJECTS = (1, 2, 5, 7)
BRANCHES = ("z_sem_img", "z_sem_txt", "z_struct", "z_low")
BRANCH_RESHAPE = {"z_struct": (8, 8, 32)}  # every other branch stays flat, (D,)

# Same three groups as brain_diffuser/scripts/imagery_regression_predict.py's POOL_OF, minus
# visionconcepts (excluded upstream of this script; never present in labels_averaged.npy).
POOL_OF = {
    "visionsimple": "vision12", "visioncomplex": "vision12",
    "imagerysimple": "imagery12", "imagerycomplex": "imagery12",
    "imageryconcepts": "imageryconcepts",
}

RIDGE_WEIGHTS_DIR = "data/betas_trisect/ridge_weights"
NSD_IMAGERY_DIR = "data/betas_trisect/nsd_imagery"
OUT_DIR = "data/betas_trisect/imagery_predictions"
GAIN_R_DIR = "data/results/ridge_input_output/dimension_gain_r"  # same r_k as predict_ridge_shared1000.py's
                                           # --gain-gamma: dev-split-only, never fit on imagery.


def load_weights(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["weight"], d["bias"]


def predict_flat(X, W, b):
    # sklearn RidgeCV.coef_ convention: W is (D, V), b is (D,) -> pred = X @ W.T + b, (N, D).
    return (X @ W.T + b).astype(np.float32)


def condition_of(label):
    return label.rsplit("-", 1)[0]


def group_rows(labels, pool_mode):
    """pool_mode -> [(group name, row indices sharing one affine map), ...], sorted by name."""
    if pool_mode == "pertype":
        # One pool per condition (6 stimuli each), not per (condition, stimulus) row, mirrors
        # imagery_regression_predict.py's 'pertype', where raw[cond] is already the 6 stimuli
        # of one condition and 'pertype' pools each condition on its own instead of merging
        # vision/imagery pairs.
        groups = {}
        for i, lbl in enumerate(labels):
            groups.setdefault(condition_of(lbl), []).append(i)
    elif pool_mode == "fromA":
        groups = {"fromA": list(range(len(labels)))}
    else:
        groups = {}
        for i, lbl in enumerate(labels):
            cond = condition_of(lbl)
            if cond not in POOL_OF:
                sys.exit(f"unknown condition '{cond}' in label '{lbl}' (not in POOL_OF)")
            groups.setdefault(POOL_OF[cond], []).append(i)
    return sorted(groups.items())


def load_trial_level(subj_imagery_dir: Path):
    """--trial-level input: the 384 single trials, at the unit variance both decoders were
    fitted on (betas_averaged.npy is the same rows averaged 8x/16x, which gives a std of ~0.30)."""
    X = np.load(subj_imagery_dir / "betas_flat.npy").astype(np.float32)
    labels_flat = np.load(subj_imagery_dir / "labels_flat.npy")
    assert len(X) == len(labels_flat), f"{X.shape} betas vs {labels_flat.shape} labels"
    return X, labels_flat


def average_by_label(pred, labels_flat, labels_avg):
    """(N_trials, D) -> (N_cond, D) averaged per (condition, stimulus).

    Rows come out in labels_avg's order, NOT in first-appearance order; labels_flat.npy is in
    acquisition order and the two do not agree; anything downstream that pairs a prediction
    row with labels_averaged.npy (the whole reconstruction step) would silently mismatch.
    """
    out = np.empty((len(labels_avg),) + pred.shape[1:], dtype=pred.dtype)
    for i, lbl in enumerate(labels_avg):
        rows = np.flatnonzero(labels_flat == lbl)
        if rows.size == 0:
            sys.exit(f"label {lbl!r} of labels_averaged.npy has no trial in labels_flat.npy")
        out[i] = pred[rows].mean(axis=0)
    return out


def pool_moments_from_A(branch, subj_weights_dir: Path):
    """The shared1000 predictions of fit_ridge.py (pred_shared1000_{branch}.npy),
    already re-standardized once onto the train-target moments. Using their moments here
    leaves the imagery predictions at their own, lower, variance (upstream's
    pool_moments_from_A() docstring), which is why this is the fallback pool, not the default.
    """
    path = subj_weights_dir / f"pred_shared1000_{branch}.npy"
    if not path.exists():
        sys.exit(f"missing {path} (fit_ridge.py must have run for this subject)")
    a = np.load(path).astype(np.float32)
    a = a.reshape(a.shape[0], -1)
    return a.mean(axis=0), a.std(axis=0)


def process_subject(subj, branches, pool_mode, weights_dir: Path, imagery_dir: Path,
                    out_dir: Path, no_restd_branches=(), scalar_restd_branches=(),
                    trial_level=False, gain_gamma=0.0, gain_r_dir=Path(GAIN_R_DIR),
                    input_weight_path=None):
    no_restd_branches, scalar_restd_branches = set(no_restd_branches), set(scalar_restd_branches)
    overlap = no_restd_branches & scalar_restd_branches
    if overlap:
        sys.exit(f"branches {sorted(overlap)} passed to both --no-restandardize-branches and "
                 f"--restandardize-scalar-branches; they are mutually exclusive")
    print(f"\n=== subj0{subj} (pool: {pool_mode}"
          f"{', trial-level' if trial_level else ''}"
          f"{', raw: ' + ','.join(sorted(no_restd_branches)) if no_restd_branches else ''}"
          f"{', scalar: ' + ','.join(sorted(scalar_restd_branches)) if scalar_restd_branches else ''}) ===")
    subj_imagery_dir = imagery_dir / f"subj{subj:02d}"
    labels = np.load(subj_imagery_dir / "labels_averaged.npy")
    if trial_level:
        X, labels_flat = load_trial_level(subj_imagery_dir)
        src_name = "betas_flat.npy"
    else:
        X = np.load(subj_imagery_dir / "betas_averaged.npy").astype(np.float32)  # (30, V)
        labels_flat = None
        src_name = "betas_averaged.npy"
    print(f"  input {src_name} {X.shape}, std {X.std():.4f}", flush=True)

    mean_x = float(X.mean())
    if abs(mean_x) > 0.5:
        print(f"  WARNING: {src_name} has mean {mean_x:.3f}; a per-run z-scored input "
              f"should be near 0", file=sys.stderr)

    if input_weight_path is not None:
        # Per-voxel reliability weight (same weight files as the --input-weight-dir of
        # predict_ridge_shared1000.py), applied to the input BEFORE predict_flat and mean-1
        # normalized; the weighted W of the voxel-reliability fit matches what it was actually
        # trained against. Applied identically whether X is betas_averaged.npy or betas_flat.npy
        # (both are per-voxel rows; the weight is per-voxel, not per-row).
        w = np.load(input_weight_path).astype(np.float32)
        w = w / w.mean()
        X = X * w
        print(f"  input reliability weight applied ({input_weight_path}, "
              f"mean={w.mean():.4f} by construction)", flush=True)

    subj_weights_dir = weights_dir / f"subj{subj:02d}"
    stats_path = subj_weights_dir / "train_target_stats.npz"
    if not stats_path.exists():
        sys.exit(f"missing {stats_path}: run fit_ridge.py for subj{subj:02d} first")
    stats = np.load(stats_path)

    subj_out = out_dir / f"subj{subj:02d}"
    subj_out.mkdir(parents=True, exist_ok=True)
    np.save(subj_out / "labels.npy", labels)

    summary = {}
    for branch in branches:
        wpath = subj_weights_dir / f"{branch}_weights.pkl"
        if not wpath.exists():
            sys.exit(f"missing {wpath}: run fit_ridge.py for subj{subj:02d} first")
        W, b = load_weights(wpath)
        if W.shape[-1] != X.shape[1]:
            sys.exit(f"{branch}: weights have {W.shape[-1]} voxels, input has {X.shape[1]}")

        raw = predict_flat(X, W, b)  # (N_trials or 30, D_flat)
        if trial_level:
            raw = average_by_label(raw, labels_flat, labels)  # -> (30, D_flat)
        tgt_mean, tgt_std = stats[f"{branch}_mean"], stats[f"{branch}_std"]
        if raw.shape[1:] != tgt_mean.shape:
            sys.exit(f"{branch}: prediction shape {raw.shape[1:]} vs "
                     f"target moments {tgt_mean.shape}")

        if branch in scalar_restd_branches:
            pred = np.empty_like(raw)
            for group_name, rows in group_rows(labels, pool_mode):
                pred[rows] = restandardize_scalar(raw[rows], tgt_mean, tgt_std)
            print(f"  [{branch}] group-wise SCALAR restandardization "
                  f"(--restandardize-scalar-branches)", flush=True)
        elif branch in no_restd_branches:
            # Per-branch ablation: on z_low restandardize() costs 6.0 pp of branch two-way
            # identification on the ridge, while it buys 6.5 pp on z_sem_txt; the inherited
            # all-or-nothing choice is not obviously right per branch. Skipping it leaves this
            # branch at the ridge's own shrunk scale, which is exactly what the ablation tests
            # downstream (VDVAE init + adapter context).
            pred = raw.copy()
            print(f"  [{branch}] restandardize SKIPPED (--no-restandardize-branches)", flush=True)
        elif gain_gamma != 0.0:
            # Per-dimension gain a_k propto r_k^gamma, r_k from the NSD-core dev split ONLY
            # (dimension_gain_precheck.py), never fit or selected on imagery. gain_gamma=0.0
            # (default) never reaches this branch; it is bit-identical to the plain `else`
            # path below. fromA pool mode is not combined with gain here: it would need its own
            # gained pool, which is not implemented.
            if pool_mode == "fromA":
                sys.exit("--gain-gamma with --restandardize-pool fromA is not implemented")
            r_path = Path(gain_r_dir) / f"subj{subj:02d}" / f"{branch}_r.npy"
            if not r_path.exists():
                sys.exit(f"missing {r_path}: run dimension_gain_precheck.py --subj {subj} first")
            gain = dim_gain(np.load(r_path), gain_gamma)
            synth_train = np.stack([tgt_mean - tgt_std, tgt_mean + tgt_std], axis=0)
            pred = np.empty_like(raw)
            for group_name, rows in group_rows(labels, pool_mode):
                pred[rows] = restandardize_gain(raw[rows], synth_train, gain)
                print(f"  [{branch}] group {group_name} ({len(rows)} rows) gain-restandardized "
                      f"(gamma={gain_gamma})", flush=True)
        else:
            pred = np.empty_like(raw)
            for group_name, rows in group_rows(labels, pool_mode):
                if pool_mode == "fromA":
                    pool_mean, pool_std = pool_moments_from_A(branch, subj_weights_dir)
                else:
                    pool = raw[rows]
                    pool_mean, pool_std = pool.mean(axis=0), pool.std(axis=0)
                pred[rows] = restandardize(raw[rows], pool_mean, pool_std, tgt_mean, tgt_std)
                print(f"  [{branch}] group {group_name} ({len(rows)} rows) restandardized",
                      flush=True)

        out_shape = BRANCH_RESHAPE.get(branch, tgt_mean.shape)
        pred_out = pred.reshape((pred.shape[0],) + out_shape)
        out_path = subj_out / f"{branch}_pred.npy"
        np.save(out_path, pred_out)
        print(f"  [{branch}] -> {out_path} {pred_out.shape}  "
              f"(mean {pred_out.mean():.3f}, std {pred_out.std():.3f})", flush=True)

        summary[branch] = {
            "shape": list(pred_out.shape),
            "mean": float(pred_out.mean()),
            "std": float(pred_out.std()),
        }

    with open(subj_out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {subj_out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--branches", nargs="+", default=list(BRANCHES), choices=list(BRANCHES))
    ap.add_argument("--no-restandardize-branches", nargs="+", default=[], choices=list(BRANCHES),
                    metavar="BRANCH",
                    help="branches for which restandardize() is skipped entirely (predictions "
                         "stay at the ridge's own shrunk scale). Per-branch ablation: "
                         "restandardize() is worth +6.5 pp of 2WC on z_sem_txt but -6.0 pp on "
                         "z_low, so the inherited all-or-nothing default is not obviously right "
                         "per branch.")
    ap.add_argument("--restandardize-scalar-branches", nargs="+", default=[],
                    choices=list(BRANCHES), metavar="BRANCH",
                    help="branches re-standardized with ONE global scalar affine map instead of "
                         "per-dimension moments: keeps the overall scale the generator needs, "
                         "drops the per-dimension equalization that costs 6.0 pp of 2WC on "
                         "z_low. Mutually exclusive with --no-restandardize-branches.")
    ap.add_argument("--restandardize-pool", default="modality12",
                    choices=["modality12", "pertype", "fromA"],
                    help="pool for pred.mean(0)/pred.std(0), mirrors "
                         "imagery_regression_predict.py's --restandardize_pool")
    ap.add_argument("--trial-level", action="store_true",
                    help="predict from betas_flat.npy (384 single trials, std ~1) and average "
                         "the predictions per (condition, stimulus), instead of predicting "
                         "from the voxel-space average betas_averaged.npy (std ~0.30). "
                         "Mathematically a no-op for this affine decoder -- it is the control "
                         "for the same flag on predict_mlp.py, where it is not.")
    ap.add_argument("--weights-dir", type=str, default=RIDGE_WEIGHTS_DIR)
    ap.add_argument("--imagery-dir", type=str, default=NSD_IMAGERY_DIR)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--gain-gamma", type=float, default=0.0,
                     help="per-dimension gain a_k propto r_k^gamma, r_k from the NSD-core dev "
                          "split (dimension_gain_precheck.py). Default 0.0 is the plain "
                          "per-dimension restandardization. Mutually exclusive with "
                          "--no-restandardize-branches / --restandardize-scalar-branches for the "
                          "branches it would apply to (those take precedence per branch).")
    ap.add_argument("--gain-r-dir", type=str, default=GAIN_R_DIR)
    ap.add_argument("--input-weight-dir", type=str, default=None,
                     help="directory holding subj{NN}_reliability.npy per-voxel weights "
                          "(voxel_reliability_and_roi.py's own naming, e.g. "
                          "data/betas_trisect/h4/reliability), applied to the input before "
                          "predict_flat, mean-1 normalized -- must match the weight "
                          "voxel_reliability_and_roi.py used to fit --weights-dir's W. Mirrors "
                          "predict_ridge_shared1000.py's flag of the same name.")
    args = ap.parse_args()

    weights_dir = Path(args.weights_dir)
    imagery_dir = Path(args.imagery_dir)
    out_dir = Path(args.out_dir)
    for subj in args.subj:
        weight_path = (Path(args.input_weight_dir) / f"subj{subj:02d}_reliability.npy"
                       if args.input_weight_dir else None)
        process_subject(subj, args.branches, args.restandardize_pool, weights_dir, imagery_dir,
                        out_dir, no_restd_branches=args.no_restandardize_branches,
                        scalar_restd_branches=args.restandardize_scalar_branches,
                        trial_level=args.trial_level, gain_gamma=args.gain_gamma,
                        gain_r_dir=Path(args.gain_r_dir), input_weight_path=weight_path)

    print("\nall done")


if __name__ == "__main__":
    main()
