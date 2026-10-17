"""Ridge fits with voxel reliability weighting and with `streams` ROI restriction, scored on shared1000.

Inputs:  NSD-Core betas and targets, streams_labels.npy of map_streams_rois.py
Outputs: weights and predictions under --out-dir, reliability/, h4_all_results.json (latent-space scores)

Arm `reliability`: X is scaled by w_v = reliability_v ** --reliability-power (mean 1), estimated on all repeated
NSD-core train images (about 9000; the per-image estimate over 3 repetitions was too noisy). Arm `roi`: the control
(all nsdgeneral) is the final fit on disk, not refit; higher_ventral = {ventral, midventral}; each of the six streams
ROIs is fit alone. Reuses fit_ridge.py unchanged. The imagery half of the ROI criterion needs the GPU: not run here.

Usage (CPU only; map_streams_rois.py first for the roi arm):
    python src/voxel_reliability_and_roi.py --arm reliability --subj 1 2 5 7 --out-dir data/betas_trisect/h4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from fit_ridge import (  # noqa: E402
    SUBJECTS, DEFAULT_ALPHAS, TARGETS_H5, ZLOW_PATH, INDEX_DIR, BETAS_DIR,
    load_branch_targets, load_nsdcore_split, fit_branch,
)
from trisect.nsd import zscore_fmri_train_test  # noqa: E402

STREAMS_ROI_DIR = "data/betas_trisect/streams_roi"
ROIS = ["early", "ventral", "midventral", "midlateral", "lateral", "parietal"]
HIGHER_VENTRAL_ROIS = ["ventral", "midventral"]


def estimate_voxel_reliability(betas_train, img73k_train):
    """r_v = signal_var / (signal_var + noise_var) in [0, 1], per voxel, estimated from every
    repeated image in the TRAIN split (never shared1000, never test). betas_train: raw counts
    (n_train, V), pre /300. Images with a single train trial contribute no noise estimate and
    are excluded from the noise-variance average (they still contribute to total_var)."""
    fmri = betas_train / 300.0
    uniq_ids, inverse, counts = np.unique(img73k_train, return_inverse=True, return_counts=True)
    rep_ids = uniq_ids[counts >= 2]
    if len(rep_ids) == 0:
        sys.exit("no repeated images in the train split -- cannot estimate reliability")
    keep = np.isin(img73k_train, rep_ids)
    fmri_r, img_r = fmri[keep], img73k_train[keep]
    uniq_r, inv_r = np.unique(img_r, return_inverse=True)

    n_img, V = len(uniq_r), fmri_r.shape[1]
    sums = np.zeros((n_img, V), dtype=np.float64)
    sq_sums = np.zeros((n_img, V), dtype=np.float64)
    np.add.at(sums, inv_r, fmri_r)
    np.add.at(sq_sums, inv_r, fmri_r ** 2)
    n_per_img = np.bincount(inv_r).astype(np.float64)[:, None]
    means = sums / n_per_img
    within_var = (sq_sums - n_per_img * means ** 2) / np.maximum(n_per_img - 1, 1)
    noise_var = within_var.mean(axis=0)               # (V,)
    total_var = fmri.var(axis=0, ddof=1)                # (V,): over ALL train trials, not just repeated
    signal_var = np.maximum(total_var - noise_var, 0.0)
    reliability = signal_var / np.maximum(signal_var + noise_var, 1e-12)
    return reliability.astype(np.float32), int(len(rep_ids))


def load_roi_mask(subj, roi_names, streams_dir: Path):
    labels_path = streams_dir / f"subj{subj:02d}" / "streams_labels.npy"
    counts_path = streams_dir / f"subj{subj:02d}" / "roi_counts.json"
    if not labels_path.exists():
        sys.exit(f"missing {labels_path}: run map_streams_rois.py first")
    labels = np.load(labels_path)
    with open(counts_path) as f:
        value_map = json.load(f)["roi_value_map"]
    mask = np.isin(labels, [value_map[r] for r in roi_names])
    return mask


def fit_variant(tag, subj, betas, img73k, train_mask, test_mask, col_mask, weight,
                 branches, alphas, out_root: Path):
    """One (subject, arm-config) fit: restrict/weight columns, z-score, fit all branches."""
    sub_betas = betas[:, col_mask] if col_mask is not None else betas
    train_fmri, test_fmri = zscore_fmri_train_test(sub_betas, train_mask, test_mask)
    if weight is not None:
        w = weight[col_mask] if col_mask is not None else weight
        w = w / w.mean()
        train_fmri = train_fmri * w
        test_fmri = test_fmri * w
    train_img73k, test_img73k = img73k[train_mask], img73k[test_mask]

    out_dir = out_root / tag / f"subj{subj:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, (target_full, out_shape) in branches.items():
        train_target = target_full[train_img73k]
        test_target = target_full[test_img73k]
        r = fit_branch(name, train_fmri, test_fmri, train_target, test_target,
                        train_img73k, test_img73k, alphas, out_shape, out_dir)
        r.pop("train_target_mean"); r.pop("train_target_std")
        results[name] = r
    results["_n_voxels"] = int(sub_betas.shape[1])
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{tag}] subj{subj:02d}: V={sub_betas.shape[1]}  " +
          "  ".join(f"{k}={v['two_way_shared1000']:.4f}" for k, v in results.items() if k != "_n_voxels"),
          flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--arm", choices=["reliability", "roi", "both"], default="both")
    ap.add_argument("--reliability-power", type=float, default=1.0)
    ap.add_argument("--targets", type=str, default=TARGETS_H5)
    ap.add_argument("--zlow", type=str, default=ZLOW_PATH)
    ap.add_argument("--index-dir", type=str, default=INDEX_DIR)
    ap.add_argument("--betas-dir", type=str, default=BETAS_DIR)
    ap.add_argument("--streams-dir", type=str, default=STREAMS_ROI_DIR)
    ap.add_argument("--out-dir", type=str, default="data/betas_trisect/h4")
    ap.add_argument("--branches", type=str, nargs="+", default=None,
                     choices=["z_sem_img", "z_sem_txt", "z_struct", "z_low"],
                     help="fit only these branches instead of all four (smoke test)")
    args = ap.parse_args()

    branches = load_branch_targets(args.targets, args.zlow)
    if args.branches:
        branches = {k: v for k, v in branches.items() if k in args.branches}
    alphas = DEFAULT_ALPHAS
    out_root = Path(args.out_dir)
    streams_dir = Path(args.streams_dir)

    all_results = {}
    for subj in args.subj:
        print(f"\n=== subj0{subj} ===", flush=True)
        betas, img73k, is_shared = load_nsdcore_split(subj, Path(args.index_dir), Path(args.betas_dir))
        train_mask, test_mask = ~is_shared, is_shared
        subj_results = {}

        if args.arm in ("reliability", "both"):
            t0 = time.monotonic()
            reliability, n_rep = estimate_voxel_reliability(betas[train_mask], img73k[train_mask])
            print(f"  reliability estimated from {n_rep} repeated train images "
                  f"(mean={reliability.mean():.3f}, median={np.median(reliability):.3f}), "
                  f"{time.monotonic()-t0:.1f}s", flush=True)
            weight = reliability ** args.reliability_power
            (out_root / "reliability").mkdir(parents=True, exist_ok=True)
            np.save(out_root / "reliability" / f"subj{subj:02d}_reliability.npy", reliability)
            r = fit_variant("reliability", subj, betas, img73k, train_mask, test_mask,
                             col_mask=None, weight=weight, branches=branches, alphas=alphas,
                             out_root=out_root)
            subj_results["reliability"] = r

        if args.arm in ("roi", "both"):
            hv_mask = load_roi_mask(subj, HIGHER_VENTRAL_ROIS, streams_dir)
            r = fit_variant("roi_higher_ventral", subj, betas, img73k, train_mask, test_mask,
                             col_mask=hv_mask, weight=None, branches=branches, alphas=alphas,
                             out_root=out_root)
            subj_results["roi_higher_ventral"] = r

            for roi in ROIS:
                roi_mask = load_roi_mask(subj, [roi], streams_dir)
                r = fit_variant(f"roi_{roi}", subj, betas, img73k, train_mask, test_mask,
                                 col_mask=roi_mask, weight=None, branches=branches, alphas=alphas,
                                 out_root=out_root)
                subj_results[f"roi_{roi}"] = r

        all_results[subj] = subj_results
        del betas

    with open(out_root / "h4_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {out_root}/h4_all_results.json")
    print("\nreminder: arm (a) 'control' = production nsdgeneral fit, NOT refit here -- "
          "read data/betas_trisect/ridge_weights/subj0N/summary.json for that row.")


if __name__ == "__main__":
    main()
