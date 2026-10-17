"""Ridge decoder for the uncompressed z_low (91,168-dim VDVAE latent), next to a refit of the compressed one.

Inputs:  NSD-Core betas and split; targets pca_zlow/z_low.npy (d1024) and vdvae_latents_31l.npy (bfull)
Outputs: ridge_weights_g2_1_{d1024refit,bfull}/subj0N/ with z_low_weights.pkl, train_target_stats.npz,
         stage1_summary.json

Brain-Diffuser regresses the raw latent, the final model a 1024-dimensional PCA of it (Section 8.1.3).
Alpha is chosen on the development split, not by RidgeCV, whose leave-one-out per trial leaves the other
repetitions of an image in the training fold and biases alpha down; d1024 is refitted the same way;
only the target differs. Stage 1 selects on ~shared1000 & ~dev, stage 2 refits on ~shared1000 at that alpha.
Both branches share one eigendecomposition per stage. One subject, branch and stage in memory at a time.

Usage (the folders feed predict_ridge_imagery.py --branches z_low --weights-dir):
    python fit_ridge_uncompressed_zlow.py --subj 1 2 5 7
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from fit_ridge import (  # noqa: E402
    BETAS_DIR, DEFAULT_ALPHAS, INDEX_DIR, SUBJECTS, TARGETS_H5, ZLOW_PATH,
    average_by_image, load_nsdcore_split, two_way,
)
from trisect.nsd import zscore_fmri  # noqa: E402
from trisect.ridge import restandardize_from_pred  # noqa: E402

RAW_LATENTS_PATH = "data/targets/vdvae_latents_31l.npy"  # (73000, 91168) float16
DEV_MASK_DIR = "data/targets"  # nsdcore_dev_mask_subj0N.npy

OUT_DIR_D1024 = "data/betas_trisect/ridge_weights_g2_1_d1024refit"
OUT_DIR_BFULL = "data/betas_trisect/ridge_weights_g2_1_bfull"

BRANCH_NAME = "z_low"  # the only branch under test; the other three branches are not refitted


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def eigh_xtx(fmri):
    """`fmri` must be exactly zero-mean per column (voxel) over these same rows; it is what
    lets fit_final's bias = Ybar reproduce sklearn Ridge(fit_intercept=True)'s intercept without
    ever centering X here (checked against sklearn on synthetic data: it matches to ~1e-6 when X
    is pre-centered and differs by ~1e-2 when it is not). zscore_fmri(betas, fit_mask)[fit_mask]
    satisfies this by construction (mean/std computed from exactly this mask), which is the only
    way X reaches this function in process_subject; never call this on raw, uncentered X."""
    log(f"  [eig] X^T X ({fmri.shape[1]}x{fmri.shape[1]}, float64) from X {fmri.shape} ...")
    col_mean_max = np.abs(fmri.astype(np.float64).mean(axis=0)).max()
    assert col_mean_max < 1e-3, (
        f"eigh_xtx requires zero-mean columns (got max |mean|={col_mean_max:.3g}) -- "
        "fit_final's bias=Ybar shortcut assumes X is already centered, see docstring")
    t0 = time.monotonic()
    Xf64 = fmri.astype(np.float64)
    xtx = Xf64.T @ Xf64
    del Xf64
    eigvals, eigvecs = np.linalg.eigh(xtx)  # ascending, orthonormal columns
    del xtx
    log(f"  [eig] done ({time.monotonic()-t0:.0f}s)")
    return eigvals.astype(np.float64), eigvecs.astype(np.float32)


# Closed form, with X (n x V) the z-scored fMRI design and X^T X = U diag(eigvals) U^T (eigh, ascending):
#   P = X @ U                        independent of alpha and of the target
#   Z = P^T (Y - Ybar)               (V x D), the one expensive, Y-dependent step
#   beta(alpha) = U diag(1/(eigvals + alpha)) Z      this is W^T in the convention of predict_flat,
#                 predict_flat(X, W, b) = X @ W.T + b; weight = beta.T and bias = Ybar
# In the alpha sweep beta(alpha) is never formed (a (V,V)@(V,D) matmul per alpha, about 2e13 FLOPs at
# D=91168). The dev predictions only need Pdev = dev_fmri @ U, once, and
# pred_dev(alpha) = Pdev diag(1/(eigvals + alpha)) Z + Ybar. beta is formed once, at the chosen alpha.
def select_alpha_on_dev(train_sel_fmri, dev_fmri, eigvals, eigvecs,
                         y_train_sel, y_dev, dev_img73k, alphas):
    """Stage 1: sweep `alphas`, score each on the dev hold-out (grouped by image), return the
    best alpha and the full (alpha -> 2WC) curve, for one branch."""
    y_mean = y_train_sel.mean(axis=0).astype(np.float32)
    yc = (y_train_sel - y_mean).astype(np.float32)

    log(f"  [sel] P = train_sel_fmri @ U ({train_sel_fmri.shape[0]}x{eigvecs.shape[1]}) ...")
    t0 = time.monotonic()
    P = train_sel_fmri.astype(np.float32) @ eigvecs
    log(f"  [sel] P done ({time.monotonic()-t0:.0f}s)")

    t0 = time.monotonic()
    Z = P.T @ yc  # (V, D): the one expensive, alpha-independent, Y-dependent step
    log(f"  [sel] Z = P^T yc done ({time.monotonic()-t0:.0f}s, Z shape {Z.shape})")
    del P

    Pdev = dev_fmri.astype(np.float32) @ eigvecs  # (n_dev, V), alpha-independent, cheap
    dev_avg_gt, dev_uniq = average_by_image(y_dev, dev_img73k)

    curve = []
    best_alpha, best_twc = None, -np.inf
    eigvals32 = eigvals.astype(np.float32)
    for alpha in alphas:
        w = (1.0 / (eigvals32 + alpha)).astype(np.float32)
        pred_dev = Pdev @ (w[:, None] * Z) + y_mean  # (n_dev_trials, D)
        pred_dev = restandardize_from_pred(pred_dev, y_train_sel)
        pred_avg, pred_uniq = average_by_image(pred_dev, dev_img73k)
        assert np.array_equal(pred_uniq, dev_uniq)
        twc = two_way(dev_avg_gt.reshape(len(dev_uniq), -1),
                       pred_avg.reshape(len(dev_uniq), -1))
        curve.append({"alpha": float(alpha), "two_way_dev": twc})
        log(f"  [sel] alpha={alpha:.4g}  dev 2WC={twc:.4f}")
        if twc > best_twc:
            best_twc, best_alpha = twc, float(alpha)
        del pred_dev, pred_avg
    del Z, Pdev
    return best_alpha, best_twc, curve


def fit_final(train_final_fmri, eigvals, eigvecs, y_train_final, alpha):
    """Stage 2: the one full (V, D) solve at the selected alpha. Returns (weight, bias) in
    predict_ridge_imagery.predict_flat's convention: pred = X @ weight.T + bias."""
    y_mean = y_train_final.mean(axis=0).astype(np.float32)
    yc = (y_train_final - y_mean).astype(np.float32)

    log(f"  [final] P = train_final_fmri @ U ({train_final_fmri.shape[0]}x{eigvecs.shape[1]}) ...")
    t0 = time.monotonic()
    P = train_final_fmri.astype(np.float32) @ eigvecs
    Z = P.T @ yc
    del P
    log(f"  [final] Z done ({time.monotonic()-t0:.0f}s)")

    w = (1.0 / (eigvals.astype(np.float32) + alpha)).astype(np.float32)
    t0 = time.monotonic()
    beta = eigvecs @ (w[:, None] * Z)  # (V, D): the single expensive solve
    log(f"  [final] beta = U diag(w) Z done ({time.monotonic()-t0:.0f}s)")
    del Z
    weight = np.ascontiguousarray(beta.T)  # (D, V), matches RidgeCV.coef_ / predict_flat
    return weight.astype(np.float32), y_mean.astype(np.float32)


def load_targets(zlow_path, raw_latents_path):
    """d1024 target fully in memory (small); bfull target stays memory-mapped float16 on disk,
    upcast only for the rows a given stage actually indexes."""
    z_low_pca = np.load(zlow_path).astype(np.float32)  # (73000, 1024)
    z_low_raw_mmap = np.load(raw_latents_path, mmap_mode="r")  # (73000, 91168) float16
    assert z_low_pca.shape[0] == z_low_raw_mmap.shape[0]
    return {"d1024": z_low_pca, "bfull": z_low_raw_mmap}


def index_target(target, ids):
    """Fancy-index a target array by 73k ids, upcasting float16 mmaps to float32 only for the
    selected rows (never materializes the full array)."""
    return np.asarray(target[ids]).astype(np.float32)


def process_subject(subj, targets, alphas, index_dir, betas_dir, dev_mask_dir,
                     out_dir_d1024: Path, out_dir_bfull: Path):
    log(f"=== subj0{subj} ===")
    betas, img73k, is_shared = load_nsdcore_split(subj, Path(index_dir), Path(betas_dir))

    dev_mask_path = Path(dev_mask_dir) / f"nsdcore_dev_mask_subj{subj:02d}.npy"
    if not dev_mask_path.exists():
        sys.exit(f"missing {dev_mask_path}: run derive_nsdcore_dev_mask.py first")
    dev_mask_73k = np.load(dev_mask_path)
    dev_trial_mask = dev_mask_73k[img73k]
    assert not np.any(dev_trial_mask & is_shared), "dev mask overlaps shared1000"

    train_final_mask = ~is_shared                      # stage 2: train+dev, as in the production fit
    train_sel_mask = train_final_mask & ~dev_trial_mask  # stage 1: train only
    dev_mask = dev_trial_mask                           # stage 1 hold-out

    n_dev_img = int(dev_mask_73k.sum())
    log(f"  trials: train_final={int(train_final_mask.sum())} "
        f"train_sel={int(train_sel_mask.sum())} dev={int(dev_mask.sum())} "
        f"({n_dev_img} dev images)")

    out_dir_of = {"d1024": out_dir_d1024, "bfull": out_dir_bfull}
    for out_dir in out_dir_of.values():
        (out_dir / f"subj{subj:02d}").mkdir(parents=True, exist_ok=True)

    # ---- stage 1: ONE eigendecomposition of train_sel, shared by both branches ----
    fmri_normed_sel = zscore_fmri(betas, train_sel_mask)  # normalizes ALL rows by train_sel's moments
    train_sel_fmri = fmri_normed_sel[train_sel_mask]
    dev_fmri = fmri_normed_sel[dev_mask]
    del fmri_normed_sel
    eigvals1, eigvecs1 = eigh_xtx(train_sel_fmri)
    train_sel_img73k = img73k[train_sel_mask]
    dev_img73k = img73k[dev_mask]

    alpha_selected = {}
    for branch_key in ("d1024", "bfull"):
        target = targets[branch_key]
        log(f"-- branch {branch_key} (target dim {target.shape[1]}), stage 1 --")
        y_train_sel = index_target(target, train_sel_img73k)
        y_dev = index_target(target, dev_img73k)

        best_alpha, best_twc, curve = select_alpha_on_dev(
            train_sel_fmri, dev_fmri, eigvals1, eigvecs1,
            y_train_sel, y_dev, dev_img73k, alphas)
        log(f"  [{branch_key}] selected alpha={best_alpha:.4g} (dev 2WC={best_twc:.4f})")
        del y_train_sel, y_dev

        subj_out = out_dir_of[branch_key] / f"subj{subj:02d}"
        with open(subj_out / "stage1_summary.json", "w") as f:
            json.dump({"branch": branch_key, "alpha_selected": best_alpha,
                       "two_way_dev_selected": best_twc, "curve": curve,
                       "n_train_sel": int(train_sel_mask.sum()), "n_dev": int(dev_mask.sum())},
                      f, indent=2)
        alpha_selected[branch_key] = (best_alpha, best_twc)
    del train_sel_fmri, dev_fmri, eigvals1, eigvecs1

    # ---- stage 2: ONE eigendecomposition of train_final, shared by both branches ----
    fmri_normed_final = zscore_fmri(betas, train_final_mask)
    train_final_fmri = fmri_normed_final[train_final_mask]
    del fmri_normed_final, betas
    eigvals2, eigvecs2 = eigh_xtx(train_final_fmri)
    train_final_img73k = img73k[train_final_mask]

    subj_summary = {}
    for branch_key in ("d1024", "bfull"):
        target = targets[branch_key]
        best_alpha, best_twc = alpha_selected[branch_key]
        log(f"-- branch {branch_key}, stage 2 (alpha={best_alpha:.4g}) --")
        y_train_final = index_target(target, train_final_img73k)

        weight, bias = fit_final(train_final_fmri, eigvals2, eigvecs2, y_train_final, best_alpha)

        subj_out = out_dir_of[branch_key] / f"subj{subj:02d}"
        weights_path = subj_out / f"{BRANCH_NAME}_weights.pkl"
        with open(weights_path, "wb") as f:
            pickle.dump({"weight": weight, "bias": bias, "alpha": best_alpha}, f)
        np.savez(subj_out / "train_target_stats.npz",
                 **{f"{BRANCH_NAME}_mean": y_train_final.mean(axis=0),
                    f"{BRANCH_NAME}_std": y_train_final.std(axis=0)})
        log(f"  [{branch_key}] wrote {weights_path}")
        del y_train_final, weight, bias

        subj_summary[branch_key] = {"alpha_selected": best_alpha, "two_way_dev_selected": best_twc}

    del train_final_fmri, eigvals2, eigvecs2
    return subj_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--targets", type=str, default=TARGETS_H5, help="unused, kept for symmetry "
                    "with fit_ridge.py's CLI -- z_low is the only branch under test")
    ap.add_argument("--zlow", type=str, default=ZLOW_PATH)
    ap.add_argument("--raw-latents", type=str, default=RAW_LATENTS_PATH)
    ap.add_argument("--index-dir", type=str, default=INDEX_DIR)
    ap.add_argument("--betas-dir", type=str, default=BETAS_DIR)
    ap.add_argument("--dev-mask-dir", type=str, default=DEV_MASK_DIR)
    ap.add_argument("--out-dir-d1024", type=str, default=OUT_DIR_D1024)
    ap.add_argument("--out-dir-bfull", type=str, default=OUT_DIR_BFULL)
    ap.add_argument("--n-alphas", type=int, default=len(DEFAULT_ALPHAS))
    args = ap.parse_args()

    import os
    if os.environ.get("OPENBLAS_CORETYPE") != "SkylakeX":
        print("WARNING: OPENBLAS_CORETYPE is not SkylakeX -- export it first on the thesis "
              "server (Sapphire Rapids) or the eigendecomposition can be silently wrong.",
              file=sys.stderr)

    alphas = np.logspace(1, 6, args.n_alphas)
    targets = load_targets(args.zlow, args.raw_latents)
    out_dir_d1024, out_dir_bfull = Path(args.out_dir_d1024), Path(args.out_dir_bfull)

    all_summaries = {}
    for subj in args.subj:
        all_summaries[subj] = process_subject(
            subj, targets, alphas, args.index_dir, args.betas_dir, args.dev_mask_dir,
            out_dir_d1024, out_dir_bfull)

    log("=== summary (alpha selected, dev 2WC) ===")
    for subj, s in all_summaries.items():
        row = "  ".join(f"{b}: alpha={v['alpha_selected']:.4g} 2WC={v['two_way_dev_selected']:.4f}"
                         for b, v in s.items())
        log(f"  subj0{subj}: {row}")

    log("all done")


if __name__ == "__main__":
    main()
