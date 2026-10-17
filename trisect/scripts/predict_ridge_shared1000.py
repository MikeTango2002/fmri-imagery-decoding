"""Apply the ridge weights to the shared1000 split and save per-subject predictions: the counterpart of
`predict_ridge_imagery.py` for the NSD-core test set.

Inputs:  ridge_weights/subj0N/ (`fit_ridge.py`), the NSD-core betas and trial index
Outputs: nsdcore_predictions/subj0N/{branch}_pred.npy, (1000,) + the branch shape, ascending 73k id, + img73k_ids.npy

The 8 repetitions are averaged first and re-standardized second, as in `predict_ridge_imagery.py`, unlike the
check inside `fit_ridge.py`: re-standardizing each trial inflates it to the full target variance, and averaging
n_eff of those shrinks the pool by ~sqrt(n_eff) (norm ratio 0.77-0.85 in the old order, 0.99-1.18 on imagery).
--split dev does the same on the per-subject dev hold-out, the only split where generator constants may be
chosen (shared1000 is measurement-only). It needs the `fit_ridge.py --exclude-dev` weights and refuses any other
--weights-dir, because the predictions would be in-sample.

Usage (from the trisect/ folder):
    python scripts/predict_ridge_shared1000.py --subj 1 --split dev --weights-dir data/betas_trisect/ridge_weights_devheld
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_ridge import average_by_image, load_nsdcore_split  # noqa: E402
from predict_ridge_imagery import load_weights, predict_flat  # noqa: E402
from trisect.nsd import zscore_fmri_train_test as zscore_fmri  # noqa: E402
from trisect.ridge import dim_gain, restandardize, restandardize_gain  # noqa: E402

SUBJECTS = (1, 2, 5, 7)
BRANCHES = ("z_sem_img", "z_sem_txt", "z_struct", "z_low")
BRANCH_RESHAPE = {"z_struct": (8, 8, 32)}  # every other branch stays flat, (D,)

INDEX_DIR = "data/nsdcore_index"
BETAS_DIR = "data/mindeyev2_dataset"
WEIGHTS_DIR = "data/betas_trisect/ridge_weights"
OUT_DIR = "data/betas_trisect/nsdcore_predictions"
DEV_OUT_DIR = "data/betas_trisect/nsdcore_predictions_dev"
DEV_MASK_DIR = "data/targets"
GAIN_R_DIR = "data/results/ridge_input_output/dimension_gain_r"  # dimension_gain_precheck.py's per-dimension r_k,
                                           # dev-split-only. --gain-gamma=0 (the default) never
                                           # reads this.


def process_subject(subj, branches, index_dir: Path, betas_dir: Path, weights_dir: Path,
                    out_dir: Path, split="shared1000", dev_mask_dir=Path("data/targets"),
                    gain_gamma=0.0, gain_r_dir=Path(GAIN_R_DIR), input_weight_path=None):
    print(f"\n=== subj0{subj} ({split}) ===")
    betas, img73k, is_shared = load_nsdcore_split(subj, index_dir, betas_dir)
    if split == "shared1000":
        train_mask, test_mask = ~is_shared, is_shared
    else:
        # dev: the per-subject held-out slice of the TRAINING pool (derive_nsdcore_dev_mask.py),
        # which is where generator constants may legitimately be selected, because shared1000 is
        # measurement-only. The weights must come from the --exclude-dev fit
        # (ridge_weights_devheld), or these predictions are in-sample.
        dev_mask_73k = np.load(Path(dev_mask_dir) / f"nsdcore_dev_mask_subj{subj:02d}.npy")
        test_mask = dev_mask_73k[img73k]
        assert not np.any(test_mask & is_shared), "dev mask overlaps shared1000"
        train_mask = ~is_shared & ~test_mask
        print(f"  dev split: {int(test_mask.sum())} trials "
              f"({len(np.unique(img73k[test_mask]))} images), {int(train_mask.sum())} train trials")
    train_fmri, test_fmri = zscore_fmri(betas, train_mask, test_mask)
    del betas
    test_img73k = img73k[test_mask]

    if input_weight_path is not None:
        # Per-voxel reliability weight applied to the z-scored input, with the same mean-1
        # normalization as fit_variant() of voxel_reliability_and_roi.py; the weighted W of
        # that fit matches what it was actually trained against. This is an INPUT-side transform
        # (not trisect/ridge.py's dim_gain, which is OUTPUT-side).
        w = np.load(input_weight_path).astype(np.float32)
        w = w / w.mean()
        test_fmri = test_fmri * w
        print(f"  input reliability weight applied ({input_weight_path}, "
              f"mean={w.mean():.4f} by construction)", flush=True)

    subj_weights_dir = weights_dir / f"subj{subj:02d}"
    stats_path = subj_weights_dir / "train_target_stats.npz"
    if not stats_path.exists():
        sys.exit(f"missing {stats_path}: run fit_ridge.py for subj{subj:02d} first")
    stats = np.load(stats_path)

    subj_out = out_dir / f"subj{subj:02d}"
    subj_out.mkdir(parents=True, exist_ok=True)

    uniq_ids_ref = None
    for branch in branches:
        wpath = subj_weights_dir / f"{branch}_weights.pkl"
        if not wpath.exists():
            sys.exit(f"missing {wpath}: run fit_ridge.py for subj{subj:02d} first")
        W, b = load_weights(wpath)
        raw = predict_flat(test_fmri, W, b)  # (3000, D_flat)

        tgt_mean, tgt_std = stats[f"{branch}_mean"], stats[f"{branch}_std"]
        # Average the 8 repetitions per image BEFORE restandardizing, not after (see the header).
        raw_avg, uniq_ids = average_by_image(raw, test_img73k)  # (1000, D_flat)
        if gain_gamma == 0.0:
            pool_mean, pool_std = raw_avg.mean(axis=0), raw_avg.std(axis=0)
            pred_avg = restandardize(raw_avg, pool_mean, pool_std, tgt_mean, tgt_std)
        else:
            # Per-dimension gain a_k propto r_k^gamma instead of the uniform equalization above.
            # restandardize(raw_avg, raw_avg.mean(0), raw_avg.std(0), tgt_mean, tgt_std) above IS
            # restandardize_from_pred(raw_avg, ...), and
            # restandardize_gain(raw_avg, synth_train, dim_gain(r, 0)) reproduces it exactly.
            # gain_gamma=0 never takes this branch at all, so that identity is a standing
            # invariant, not something exercised at runtime here.
            r_path = Path(gain_r_dir) / f"subj{subj:02d}" / f"{branch}_r.npy"
            if not r_path.exists():
                sys.exit(f"missing {r_path}: run dimension_gain_precheck.py --subj {subj} first "
                         "(needs data/betas_trisect/ridge_weights_devheld)")
            r = np.load(r_path)
            gain = dim_gain(r, gain_gamma)
            # tgt_mean/tgt_std are the only train-target moments persisted to disk
            # (train_target_stats.npz); this 2-row array reproduces them exactly under
            # restandardize_gain's own mean()/std() calls (same trick as the synth_train of
            # dimension_gain_precheck.py).
            synth_train = np.stack([tgt_mean - tgt_std, tgt_mean + tgt_std], axis=0)
            pred_avg = restandardize_gain(raw_avg, synth_train, gain)
        if uniq_ids_ref is None:
            uniq_ids_ref = uniq_ids
        assert np.array_equal(uniq_ids, uniq_ids_ref)

        out_shape = BRANCH_RESHAPE.get(branch, tgt_mean.shape)
        pred_out = pred_avg.reshape((pred_avg.shape[0],) + out_shape)
        out_path = subj_out / f"{branch}_pred.npy"
        np.save(out_path, pred_out)
        print(f"  [{branch}] -> {out_path} {pred_out.shape}  "
              f"(mean {pred_out.mean():.3f}, std {pred_out.std():.3f})", flush=True)

    np.save(subj_out / "img73k_ids.npy", uniq_ids_ref)
    print(f"  wrote {subj_out} ({len(uniq_ids_ref)} unique {split} images)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--branches", nargs="+", default=list(BRANCHES), choices=list(BRANCHES))
    ap.add_argument("--index-dir", type=str, default=INDEX_DIR)
    ap.add_argument("--betas-dir", type=str, default=BETAS_DIR)
    ap.add_argument("--weights-dir", type=str, default=WEIGHTS_DIR)
    ap.add_argument("--out-dir", type=str, default=None,
                     help=f"default: {OUT_DIR} for --split shared1000, {DEV_OUT_DIR} for dev")
    ap.add_argument("--split", choices=["shared1000", "dev"], default="shared1000",
                     help="which held-out slice to predict. dev is the per-subject hold-out "
                          "carved from the TRAINING pool (derive_nsdcore_dev_mask.py) and REQUIRES "
                          "--weights-dir data/betas_trisect/ridge_weights_devheld (the "
                          "fit_ridge.py --exclude-dev fit); with the production weights the "
                          "dev predictions are in-sample and worthless as a selection split.")
    ap.add_argument("--dev-mask-dir", type=str, default=DEV_MASK_DIR)
    ap.add_argument("--gain-gamma", type=float, default=0.0,
                     help="per-dimension gain a_k propto r_k^gamma applied on top of "
                          "restandardize(), instead of the uniform equalization. Default 0.0 "
                          "is the plain behaviour (dim_gain(r,0) is all-ones). gamma=1 is the "
                          "primary candidate.")
    ap.add_argument("--gain-r-dir", type=str, default=GAIN_R_DIR,
                     help="per-subject/branch r_k arrays from dimension_gain_precheck.py (dev split "
                          "only). Read only when --gain-gamma != 0.")
    ap.add_argument("--input-weight-dir", type=str, default=None,
                     help="directory holding subj{NN}_reliability.npy per-voxel weights "
                          "(voxel_reliability_and_roi.py's own naming/location, e.g. "
                          "data/betas_trisect/h4/reliability), applied to the z-scored input "
                          "before predict_flat, mean-1 normalized -- must match the weight "
                          "voxel_reliability_and_roi.py used to fit --weights-dir's W.")
    args = ap.parse_args()

    out_dir = args.out_dir or (OUT_DIR if args.split == "shared1000" else DEV_OUT_DIR)
    if args.split == "dev" and "devheld" not in args.weights_dir:
        raise SystemExit(f"--split dev with --weights-dir {args.weights_dir}: those weights saw "
                          "the dev images. Refit with fit_ridge.py --exclude-dev first.")

    for subj in args.subj:
        weight_path = (Path(args.input_weight_dir) / f"subj{subj:02d}_reliability.npy"
                       if args.input_weight_dir else None)
        process_subject(subj, args.branches, Path(args.index_dir), Path(args.betas_dir),
                        Path(args.weights_dir), Path(out_dir), args.split,
                        Path(args.dev_mask_dir), gain_gamma=args.gain_gamma,
                        gain_r_dir=Path(args.gain_r_dir), input_weight_path=weight_path)
    print("\nall done")


if __name__ == "__main__":
    main()
