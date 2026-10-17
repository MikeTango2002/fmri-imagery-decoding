"""Build the z_low_pred.npy of shared1000 from the decodable basis (layout of predict_ridge_shared1000.py).

Inputs:  production nsdcore predictions, uncompressed ridge weights (bfull, fitted on ~shared1000), betas
Outputs: nsdcore_predictions_i3_k<K>/subj0N/: new z_low_pred.npy; z_sem, z_struct and img73k_ids copied

Use it as `reconstruct.py --zlow-mode raw --pred-dir <output folder>` (PCA^-1 is the identity on that
path). Unlike the check, which held out dev so as not to leak the images it scores, this is the production
step: the SVD basis (k fixed by the check) comes from a train_final subsample, and shared1000 never enters
it. reconstruct.py reads all four arrays from one --pred-dir; they must share the img73k_ids order
(asserted). CPU only.

Usage:
    python decodable_basis_predict_shared1000.py --subj 1 2 5 7 --k 64
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_ridge import average_by_image, load_nsdcore_split  # noqa: E402
from predict_ridge_imagery import load_weights, predict_flat  # noqa: E402
from trisect.nsd import zscore_fmri  # noqa: E402
from trisect.ridge import restandardize  # noqa: E402
from decodable_basis_precheck import rotate  # noqa: E402

SUBJECTS = (1, 2, 5, 7)
INDEX_DIR = "data/nsdcore_index"
BETAS_DIR = "data/mindeyev2_dataset"
BFULL_WEIGHTS_DIR = "data/betas_trisect/ridge_weights_g2_1_bfull"  # production, train_final
RAW_LATENTS_PATH = "data/targets/vdvae_latents_31l.npy"
SRC_PRED_DIR = "data/betas_trisect/nsdcore_predictions"  # z_sem/z_struct source (unchanged)
OUT_DIR_DEFAULT = "data/betas_trisect/nsdcore_predictions_i3"
SEED = 0
K_DEFAULT = 64
N_TRAIN_SUB_DEFAULT = 3000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--n-train-sub", type=int, default=N_TRAIN_SUB_DEFAULT)
    ap.add_argument("--index-dir", type=str, default=INDEX_DIR)
    ap.add_argument("--betas-dir", type=str, default=BETAS_DIR)
    ap.add_argument("--bfull-weights-dir", type=str, default=BFULL_WEIGHTS_DIR)
    ap.add_argument("--raw-latents", type=str, default=RAW_LATENTS_PATH)
    ap.add_argument("--src-pred-dir", type=str, default=SRC_PRED_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=str, default=None,
                     help=f"default: {OUT_DIR_DEFAULT}_k<K>")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"{OUT_DIR_DEFAULT}_k{args.k}")
    raw_latents_mmap = np.load(args.raw_latents, mmap_mode="r")

    for subj in args.subj:
        print(f"\n=== subj0{subj} ===")
        betas, img73k, is_shared = load_nsdcore_split(subj, Path(args.index_dir), Path(args.betas_dir))
        train_final_mask = ~is_shared
        test_mask = is_shared

        fmri_normed = zscore_fmri(betas, train_final_mask)
        train_final_fmri = fmri_normed[train_final_mask]
        test_fmri = fmri_normed[test_mask]
        del betas, fmri_normed

        train_final_img73k = img73k[train_final_mask]
        test_img73k = img73k[test_mask]

        weight, bias = load_weights(Path(args.bfull_weights_dir) / f"subj{subj:02d}" / "z_low_weights.pkl")

        pred_train = predict_flat(train_final_fmri, weight, bias).astype(np.float64)
        pred_train_avg, train_uniq = average_by_image(pred_train, train_final_img73k)
        del pred_train

        pred_test = predict_flat(test_fmri, weight, bias).astype(np.float64)
        pred_test_avg, test_uniq = average_by_image(pred_test, test_img73k)
        del pred_test, weight, bias

        rng = np.random.default_rng(args.seed)
        n_sub = min(args.n_train_sub, len(train_uniq))
        sub_idx = np.sort(rng.choice(len(train_uniq), size=n_sub, replace=False))
        pred_train_sub = pred_train_avg[sub_idx]
        y_true_train_sub = np.asarray(raw_latents_mmap[train_uniq[sub_idx]]).astype(np.float64)
        del pred_train_avg

        mean_c = pred_train_sub.mean(axis=0)
        Xc = pred_train_sub - mean_c
        print(f"  SVD of centered train predictions ({Xc.shape}) ...")
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        Vk = Vt[:args.k]

        coef_pred_train, resid_pred_train = rotate(pred_train_sub, mean_c, Vk)
        coef_true_train, resid_true_train = rotate(y_true_train_sub, mean_c, Vk)

        coef_pred_test, resid_pred_test = rotate(pred_test_avg, mean_c, Vk)
        coef_test_restd = restandardize(coef_pred_test, coef_pred_train.mean(axis=0),
                                         coef_pred_train.std(axis=0),
                                         coef_true_train.mean(axis=0), coef_true_train.std(axis=0))
        resid_test_restd = restandardize(resid_pred_test, resid_pred_train.mean(axis=0),
                                          resid_pred_train.std(axis=0),
                                          resid_true_train.mean(axis=0), resid_true_train.std(axis=0))
        z_low_i3 = (mean_c + coef_test_restd @ Vk + resid_test_restd).astype(np.float32)
        print(f"  z_low (decodable basis, k={args.k}): {z_low_i3.shape}")

        src_subj_dir = Path(args.src_pred_dir) / f"subj{subj:02d}"
        src_ids = np.load(src_subj_dir / "img73k_ids.npy")
        assert np.array_equal(src_ids, test_uniq), (
            f"subj{subj:02d}: img73k_ids order mismatch between {src_subj_dir} and this script's "
            "own average_by_image output -- reconstruct.py would misalign z_sem/"
            "z_struct against this z_low")

        out_subj_dir = out_dir / f"subj{subj:02d}"
        out_subj_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("img73k_ids.npy", "z_sem_img_pred.npy", "z_sem_txt_pred.npy", "z_struct_pred.npy"):
            shutil.copy2(src_subj_dir / fname, out_subj_dir / fname)
        np.save(out_subj_dir / "z_low_pred.npy", z_low_i3)
        print(f"  wrote {out_subj_dir}")

    print(f"\nDone. Reconstruct with:\n"
          f"  python src/reconstruct.py --dataset nsdcore --harness-blocks "
          f"--zlow-mode raw --pred-dir {out_dir} --gen-rep 1 --strength 0.85 --mixing 0.6 "
          f"--seed 0 --subj {' '.join(str(s) for s in args.subj)}")


if __name__ == "__main__":
    main()
