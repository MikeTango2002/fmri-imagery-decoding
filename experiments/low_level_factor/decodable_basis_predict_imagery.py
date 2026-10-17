"""Build the z_low_pred.npy of NSD-Imagery from the decodable basis (evaluation only, Section 8.1.3).

Inputs:  production imagery predictions, uncompressed ridge weights (bfull), NSD-Core and NSD-Imagery betas
Outputs: imagery_predictions_i3_k<K>/subj0N/: new z_low_pred.npy, the other three branches copied

The basis is the one of decodable_basis_predict_shared1000.py (same subject, k, seed and n-train-sub on the
same train_final split); mean_c, Vk and the restandardize statistics are bit-identical. NSD-Imagery is
reported once and never used to choose k (64, from decodable_basis_precheck.py). CPU only.

Usage:
    python decodable_basis_predict_imagery.py --subj 1 2 5 7 --k 64
Then reconstruct.py --zlow-mode raw --pred-dir <output folder> --gen-rep 10 --strength 0.85 --mixing 0.6
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
IMAGERY_DIR = "data/betas_trisect/nsd_imagery"  # predict_ridge_imagery.py's input
SRC_PRED_DIR = "data/betas_trisect/imagery_predictions"  # z_sem/z_struct source (unchanged)
OUT_DIR_DEFAULT = "data/betas_trisect/imagery_predictions_i3"
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
    ap.add_argument("--imagery-dir", type=str, default=IMAGERY_DIR)
    ap.add_argument("--src-pred-dir", type=str, default=SRC_PRED_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=str, default=None,
                     help=f"default: {OUT_DIR_DEFAULT}_k<K>")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"{OUT_DIR_DEFAULT}_k{args.k}")
    raw_latents_mmap = np.load(args.raw_latents, mmap_mode="r")

    for subj in args.subj:
        print(f"\n=== subj0{subj} ===")

        # Step 1: fit the SAME frozen basis that decodable_basis_predict_shared1000.py fits for
        # shared1000: identical subj/k/seed/n-train-sub against the same NSD-core train_final
        # split; mean_c/Vk/the four restandardize target stats below are bit-identical to
        # that script's. Never touches NSD-Imagery: the basis is chosen from NSD-core only.
        betas, img73k, is_shared = load_nsdcore_split(subj, Path(args.index_dir), Path(args.betas_dir))
        train_final_mask = ~is_shared

        fmri_normed = zscore_fmri(betas, train_final_mask)
        train_final_fmri = fmri_normed[train_final_mask]
        del betas, fmri_normed

        train_final_img73k = img73k[train_final_mask]

        weight, bias = load_weights(Path(args.bfull_weights_dir) / f"subj{subj:02d}" / "z_low_weights.pkl")

        pred_train = predict_flat(train_final_fmri, weight, bias).astype(np.float64)
        pred_train_avg, train_uniq = average_by_image(pred_train, train_final_img73k)
        del pred_train

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

        # Step 2: apply that frozen basis to the NSD-Imagery betas, through the production
        # uncompressed ridge weights (same weight/bias just used above): the (30, V) rows that
        # predict_ridge_imagery.py would score, in the per-run z-scored input space
        # (betas_averaged.npy, already /300 and z-scored by prepare_betas.py; no second
        # z-score, as in predict_ridge_imagery.py).
        subj_imagery_dir = Path(args.imagery_dir) / f"subj{subj:02d}"
        X_imagery = np.load(subj_imagery_dir / "betas_averaged.npy").astype(np.float32)
        labels = np.load(subj_imagery_dir / "labels_averaged.npy")

        pred_imagery = predict_flat(X_imagery, weight, bias).astype(np.float64)  # (30, 91168)
        coef_pred_imagery, resid_pred_imagery = rotate(pred_imagery, mean_c, Vk)
        coef_imagery_restd = restandardize(coef_pred_imagery, coef_pred_train.mean(axis=0),
                                            coef_pred_train.std(axis=0),
                                            coef_true_train.mean(axis=0), coef_true_train.std(axis=0))
        resid_imagery_restd = restandardize(resid_pred_imagery, resid_pred_train.mean(axis=0),
                                             resid_pred_train.std(axis=0),
                                             resid_true_train.mean(axis=0), resid_true_train.std(axis=0))
        z_low_i3 = (mean_c + coef_imagery_restd @ Vk + resid_imagery_restd).astype(np.float32)
        print(f"  z_low (decodable basis, k={args.k}): {z_low_i3.shape}")

        src_subj_dir = Path(args.src_pred_dir) / f"subj{subj:02d}"
        src_labels = np.load(src_subj_dir / "labels.npy")
        assert np.array_equal(src_labels, labels), (
            f"subj{subj:02d}: labels order mismatch between {src_subj_dir} and "
            "nsd_imagery/labels_averaged.npy -- reconstruct.py would misalign "
            "z_sem/z_struct against this z_low")

        out_subj_dir = out_dir / f"subj{subj:02d}"
        out_subj_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_subj_dir / "labels.npy", out_subj_dir / "labels.npy")
        for fname in ("z_sem_img_pred.npy", "z_sem_txt_pred.npy", "z_struct_pred.npy"):
            shutil.copy2(src_subj_dir / fname, out_subj_dir / fname)
        np.save(out_subj_dir / "z_low_pred.npy", z_low_i3)
        print(f"  wrote {out_subj_dir}")

    print(f"\nDone. Reconstruct with:\n"
          f"  python src/reconstruct.py --zlow-mode raw --pred-dir {out_dir} "
          f"--gen-rep 10 --strength 0.85 --mixing 0.6 --seed 0 "
          f"--subj {' '.join(str(s) for s in args.subj)}")


if __name__ == "__main__":
    main()
