"""Closed-form ridge posterior variance of each branch, calibrated on NSD-core training trials only (Section 8.1.2).

Inputs:  the production ridge weights, the NSD-core betas, the shared1000 predictions on disk (only to verify X)
Outputs: per subject under --out-dir (data/betas_trisect/ridge_posterior/subj0N/): xtx_eig.npz and train_proj.npy
         (caches), {branch}_summary.json (alpha_production, alpha_gcv, s2_f, restd_factor, spearman_rho...),
         gate_summary.json

sigma^2_f(x) = s^2_f * (1 + x^T (X^T X + alpha_f I)^-1 x), meant for a per-trial gating of the generator. The method
steps and the rules fixed before running are in the comment above the constants. On subject 1 the Spearman
correlation stayed between -0.006 and +0.013 on all four branches, against the required 0.2.

Usage (numpy / scipy / sklearn, no torch; after fit_ridge.py and predict_ridge_shared1000.py):
    python ridge_posterior_sigma2.py --subj 1 2 5 7
    python ridge_posterior_sigma2.py --subj 1 --smoke-voxels 500 --n-alphas 5   # code check only
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_ridge import (  # noqa: E402
    BETAS_DIR, DEFAULT_ALPHAS, INDEX_DIR, ZLOW_PATH, TARGETS_H5,
    load_branch_targets, load_nsdcore_split,
)
from predict_ridge_imagery import load_weights, predict_flat  # noqa: E402
from trisect.nsd import zscore_fmri_train_test as zscore_fmri  # noqa: E402

# Method. In the raw ridge-output space sigma^2_f(x) = s^2_f * (1 + x^T (X^T X + alpha_f I)^-1 x); after restandardize() it
# is scaled per output dimension by (tgt_std_d / pool_std_d)^2. Shared1000 calibrates nothing: it only checks that the
# rebuilt X reproduces the predictions of the production ridge.
#   1. Rebuild X as fit_ridge.py sees it (imported, not reimplemented) and check that it gives the saved shared1000
#      predictions (data/betas_trisect/nsdcore_predictions/).
#   2. One eigendecomposition of X^T X per subject, cached: it gives h_ii(alpha) for every alpha and all four branches
#      at once (X is shared, only y changes). P = X @ V is cached next to it.
#   3. A GCV curve over the alpha grid, folds grouped by image: a DIAGNOSTIC, never used in sigma^2. The production
#      alpha came from the per-trial leave-one-out GCV of RidgeCV(cv=None), which leaks the ~2 other repetitions of an
#      image into its own training signal; it is mildly under-regularized. The posterior formula is right only at the
#      alpha actually fitted; sigma^2 always uses the production alpha (refitting would mix less training data with
#      the effect of the gating). Rule fixed before running: alpha_disagreement=True if alpha_gcv is more than
#      ALPHA_RATIO_THRESHOLD from the production alpha on any branch or subject. It is reported and changes nothing.
#   4. Leave-one-image-out residuals in closed form, r_LOO_B = (I - H_BB)^-1 r_B, a block B being the 3 repetitions of one
#      image (3 for all 4 subjects); H_BB = Q_B Q_B^T with Q = P * sqrt(w(alpha)), from the same cache.
#   5. s^2_f: one scalar per branch and subject, the mean squared LOO residual over all training trials and dimensions.
#   6. The factor (tgt_std / pool_std)^2 per output dimension, from the same raw prediction and pool as step 1; it is
#      what the production restandardize() did. Dropping it would be wrong by a per-dimension factor, not by a constant.
#   7. The gate: Spearman rho(sigma^2_pred_i, mean_d r_LOO_i,d^2) over the ~27,000 training trials of a branch and
#      subject; PASS if rho > SPEARMAN_GATE_RHO on at least 3 of 4 branches and 3 of 4 subjects. Passing is the expected
#      outcome under correct specification (Var(r_LOO) grows with leverage by algebra); the gate checks the
#      specification and is no evidence for the gating; a failure is the informative outcome.
#   8. The fallback if a branch fails, a K-fold out-of-fold residual grouped by image, is not implemented.
# Cost: the eigendecomposition and the projection, once per subject (tens of minutes), and the GCV sweep of step 3, one
# (n_train x V) @ (V x D) matmul per (alpha, branch), dominated by z_struct (D = 2048).
SUBJECTS = (1, 2, 5, 7)
BRANCHES = ("z_sem_img", "z_sem_txt", "z_struct", "z_low")
REPS_PER_IMAGE = 3  # verified uniform across all 4 subjects (step 4)

RIDGE_WEIGHTS_DIR = "data/betas_trisect/ridge_weights"
NSDCORE_PRED_DIR = "data/betas_trisect/nsdcore_predictions"  # predict_ridge_shared1000.py
OUT_DIR = "data/betas_trisect/ridge_posterior"

ALPHA_RATIO_THRESHOLD = 2.0   # fixed before running
SPEARMAN_GATE_RHO = 0.2       # gate threshold, fixed before running


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def block_index_for_images(train_img73k):
    """Stable-sort trial positions by image id -> contiguous blocks of REPS_PER_IMAGE trials.

    Returns (n_blocks, REPS_PER_IMAGE) int array of positions into the ORIGINAL train arrays
    (train_fmri/P/y rows), one row per NSD-core training image.
    """
    order = np.argsort(train_img73k, kind="stable")
    sorted_ids = train_img73k[order]
    uniq, counts = np.unique(sorted_ids, return_counts=True)
    bad = np.unique(counts)
    assert np.all(counts == REPS_PER_IMAGE), (
        f"expected exactly {REPS_PER_IMAGE} reps/image uniformly, got block sizes {bad.tolist()} "
        "-- the fixed-size block-LOO machinery below assumes REPS_PER_IMAGE (step 4)"
    )
    return order.reshape(-1, REPS_PER_IMAGE), uniq


def block_loo_residuals(R, Q, block_index):
    """(I - H_BB)^-1 r_B per block, H_BB = Q_B Q_B^T. R, Q: (n_train, ...). Returns an array
    shaped like R with each trial's residual replaced by its leave-one-image-out value."""
    Qb = Q[block_index].astype(np.float64)                          # (n_blocks, k, V)
    Hbb = np.einsum('bik,bjk->bij', Qb, Qb, optimize=True)           # (n_blocks, k, k)
    Rb = R[block_index].astype(np.float64)                          # (n_blocks, k, D)
    Ik = np.eye(REPS_PER_IMAGE, dtype=np.float64)
    Minv = np.linalg.inv(Ik[None, :, :] - Hbb)                       # (n_blocks, k, k)
    Rloo_b = np.einsum('bij,bjd->bid', Minv, Rb, optimize=True)      # (n_blocks, k, D)
    Rloo = np.empty(R.shape, dtype=np.float64)
    Rloo[block_index] = Rloo_b
    return Rloo


def eigh_xtx(subj, train_fmri, out_dir: Path, force=False):
    path = out_dir / "xtx_eig.npz"
    if path.exists() and not force:
        log(f"  [eig] cached {path}")
        d = np.load(path)
        return d["eigvals"], d["eigvecs"]
    log(f"  [eig] computing X^T X ({train_fmri.shape[1]}x{train_fmri.shape[1]}, float64) ...")
    t0 = time.monotonic()
    Xf64 = train_fmri.astype(np.float64)
    xtx = Xf64.T @ Xf64
    del Xf64
    log(f"  [eig] X^T X done ({time.monotonic()-t0:.0f}s), running eigh ...")
    t1 = time.monotonic()
    eigvals, eigvecs = np.linalg.eigh(xtx)  # ascending eigenvalues, orthonormal eigvecs
    del xtx
    log(f"  [eig] eigh done ({time.monotonic()-t1:.0f}s total {time.monotonic()-t0:.0f}s)")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, eigvals=eigvals, eigvecs=eigvecs)
    return eigvals, eigvecs


def compute_projection(subj, train_fmri, eigvecs, out_dir: Path, force=False):
    path = out_dir / "train_proj.npy"
    if path.exists() and not force:
        log(f"  [proj] cached {path}")
        return np.load(path)
    log(f"  [proj] computing P = X @ V ({train_fmri.shape[0]}x{eigvecs.shape[1]}) ...")
    t0 = time.monotonic()
    P = train_fmri.astype(np.float32) @ eigvecs.astype(np.float32)
    log(f"  [proj] done ({time.monotonic()-t0:.0f}s)")
    np.save(path, P)
    return P


def verify_reconstruction(subj, branch, test_fmri, test_img73k, weights_dir: Path,
                           pred_dir: Path):
    """Step 1: pass the rebuilt test_fmri through the saved ridge weights and check that it
    reproduces the shared1000 predictions written by predict_ridge_shared1000.py. Also returns
    the restandardization factor of step 6, from the raw (pre-restandardize) prediction and its
    per-dimension pool std."""
    from fit_ridge import average_by_image
    subj_w = weights_dir / f"subj{subj:02d}"
    W, b = load_weights(subj_w / f"{branch}_weights.pkl")
    raw = predict_flat(test_fmri, W, b)  # (n_test, D)
    stats = np.load(subj_w / "train_target_stats.npz")
    tgt_mean, tgt_std = stats[f"{branch}_mean"], stats[f"{branch}_std"]
    pool_mean, pool_std = raw.mean(axis=0), raw.std(axis=0)
    # same formula as the restandardization of fit_ridge.py, spelled out here because
    # only the target's mean/std (not the full train_target array) are on disk in this script.
    pred = (raw - pool_mean) / np.maximum(pool_std, 1e-8) * tgt_std + tgt_mean
    pred_avg, _ = average_by_image(pred, test_img73k)

    ref_path = pred_dir / f"subj{subj:02d}" / f"{branch}_pred.npy"
    if not ref_path.exists():
        sys.exit(f"missing {ref_path}: run predict_ridge_shared1000.py for subj{subj:02d} first")
    ref = np.load(ref_path).reshape(pred_avg.shape)
    max_abs_diff = float(np.max(np.abs(pred_avg - ref)))
    ok = bool(np.allclose(pred_avg, ref, rtol=1e-4, atol=1e-5))
    restd_factor = (tgt_std / np.maximum(pool_std, 1e-8)) ** 2  # step 6, per output dim
    return ok, max_abs_diff, restd_factor


def gcv_alpha_curve(P, eigvals, Zfull, yc, block_index, alphas):
    """Step 3, diagnostic only: grouped-by-image GCV score at every alpha in the grid."""
    scores = []
    for alpha in alphas:
        w = (1.0 / (eigvals + alpha)).astype(np.float32)
        Yhat = P @ (w[:, None] * Zfull)
        R = yc - Yhat
        Q = P * np.sqrt(w)[None, :]
        Rloo = block_loo_residuals(R, Q, block_index)
        scores.append(float(np.mean(Rloo ** 2)))
        del Yhat, R, Q, Rloo
    return np.asarray(scores)


def process_branch(subj, branch, target_full, P, eigvals, train_img73k, block_index,
                    alphas, weights_dir: Path, verify_ok, verify_max_abs_diff, restd_factor):
    log(f"  [{branch}] building y, Z = P^T (y - ybar) ...")
    y = target_full[train_img73k].astype(np.float32)
    y_mean = y.mean(axis=0)
    yc = y - y_mean
    t0 = time.monotonic()
    Zfull = P.T.astype(np.float32) @ yc
    log(f"  [{branch}] Z done ({time.monotonic()-t0:.0f}s), sweeping {len(alphas)} alphas for GCV ...")

    t0 = time.monotonic()
    gcv_scores = gcv_alpha_curve(P, eigvals, Zfull, yc, block_index, alphas)
    alpha_gcv = float(alphas[int(np.argmin(gcv_scores))])
    log(f"  [{branch}] GCV sweep done ({time.monotonic()-t0:.0f}s), alpha_gcv={alpha_gcv:.4g}")

    with open(weights_dir / f"subj{subj:02d}" / f"{branch}_weights.pkl", "rb") as f:
        import pickle
        alpha_production = float(pickle.load(f)["alpha"])

    ratio = alpha_gcv / alpha_production
    alpha_disagreement = bool(ratio > ALPHA_RATIO_THRESHOLD or ratio < 1.0 / ALPHA_RATIO_THRESHOLD)
    if alpha_disagreement:
        log(f"  [{branch}] *** alpha_gcv/alpha_production = {ratio:.3g}, outside "
            f"[1/{ALPHA_RATIO_THRESHOLD}, {ALPHA_RATIO_THRESHOLD}] -- flagged, "
            f"NOT used to change sigma^2 (see module docstring) ***")

    log(f"  [{branch}] step 4-5: LOO residuals + s2_f at alpha_production={alpha_production:.4g}")
    w_prod = (1.0 / (eigvals + alpha_production)).astype(np.float32)
    h_ii = (P ** 2) @ w_prod
    Yhat_prod = P @ (w_prod[:, None] * Zfull)
    R_prod = yc - Yhat_prod
    Q_prod = P * np.sqrt(w_prod)[None, :]
    Rloo_prod = block_loo_residuals(R_prod, Q_prod, block_index)
    s2_f = float(np.mean(Rloo_prod ** 2))

    sigma2_pred = s2_f * (1.0 + h_ii.astype(np.float64))
    sq_err = np.mean(Rloo_prod ** 2, axis=1)
    rho, pval = spearmanr(sigma2_pred, sq_err)

    log(f"  [{branch}] s2_f={s2_f:.6g}  spearman_rho={rho:.4f} (p={pval:.2g})  "
        f"{'PASS' if rho > SPEARMAN_GATE_RHO else 'FAIL'} (gate {SPEARMAN_GATE_RHO})")

    return {
        "alpha_production": alpha_production,
        "alpha_gcv": alpha_gcv,
        "alpha_ratio_gcv_over_production": ratio,
        "alpha_disagreement": alpha_disagreement,
        "s2_f": s2_f,
        "restd_factor": restd_factor.tolist(),
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "n_train": int(len(train_img73k)),
        "verify_ok": verify_ok,
        "verify_max_abs_diff": verify_max_abs_diff,
        "gcv_scores": gcv_scores.tolist(),
        "gcv_alphas": [float(a) for a in alphas],
    }


def process_subject(subj, branches, alphas, index_dir, betas_dir, weights_dir: Path,
                     pred_dir: Path, out_dir: Path, targets_h5, zlow_path,
                     smoke_voxels=None, force_eig=False, force_proj=False):
    log(f"=== subj0{subj} ===")
    betas, img73k, is_shared = load_nsdcore_split(subj, index_dir, betas_dir)
    if smoke_voxels is not None:
        log(f"  --smoke-voxels {smoke_voxels}: subsampling voxel columns (CODE CHECK ONLY, "
            f"not a real measurement)")
        betas = betas[:, :smoke_voxels]
    train_mask, test_mask = ~is_shared, is_shared
    train_fmri, test_fmri = zscore_fmri(betas, train_mask, test_mask)
    del betas
    train_img73k, test_img73k = img73k[train_mask], img73k[test_mask]

    subj_dirname = f"subj{subj:02d}" + (f"_smoke{smoke_voxels}" if smoke_voxels is not None else "")
    subj_out = out_dir / subj_dirname
    subj_out.mkdir(parents=True, exist_ok=True)

    eigvals, eigvecs = eigh_xtx(subj, train_fmri, subj_out, force=force_eig)
    P = compute_projection(subj, train_fmri, eigvecs, subj_out, force=force_proj)

    block_index, uniq_ids = block_index_for_images(train_img73k)
    log(f"  {len(uniq_ids)} training images, {REPS_PER_IMAGE} reps each -> block-LOO ready")

    branch_targets = load_branch_targets(targets_h5, zlow_path)
    if smoke_voxels is not None:
        # weights were fit on the FULL voxel set; forward-passing a truncated test_fmri through
        # them is meaningless; skip verify/restd (smoke test is for the eig/proj/GCV/LOO
        # machinery only, not for step 1/6).
        skip_verify = True
    else:
        skip_verify = False

    results = {}
    for branch in branches:
        target_full, _ = branch_targets[branch]
        if skip_verify:
            verify_ok, verify_max_abs_diff, restd_factor = None, None, np.ones(target_full.shape[1])
        else:
            verify_ok, verify_max_abs_diff, restd_factor = verify_reconstruction(
                subj, branch, test_fmri, test_img73k, weights_dir, pred_dir)
            if not verify_ok:
                log(f"  [{branch}] *** step 1 verify FAILED, max_abs_diff={verify_max_abs_diff:.3g} "
                    f"-- X reconstruction does not match production, results below are not trustworthy ***")
        r = process_branch(subj, branch, target_full, P, eigvals, train_img73k, block_index,
                            alphas, weights_dir, verify_ok, verify_max_abs_diff, restd_factor)
        results[branch] = r
        with open(subj_out / f"{branch}_summary.json", "w") as f:
            json.dump(r, f, indent=2)

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--branches", type=str, nargs="+", default=list(BRANCHES), choices=list(BRANCHES))
    ap.add_argument("--targets", type=str, default=TARGETS_H5)
    ap.add_argument("--zlow", type=str, default=ZLOW_PATH)
    ap.add_argument("--index-dir", type=str, default=INDEX_DIR)
    ap.add_argument("--betas-dir", type=str, default=BETAS_DIR)
    ap.add_argument("--weights-dir", type=str, default=RIDGE_WEIGHTS_DIR)
    ap.add_argument("--pred-dir", type=str, default=NSDCORE_PRED_DIR)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--n-alphas", type=int, default=len(DEFAULT_ALPHAS))
    ap.add_argument("--force-eig", action="store_true", help="recompute xtx_eig.npz even if cached")
    ap.add_argument("--force-proj", action="store_true", help="recompute train_proj.npy even if cached")
    ap.add_argument("--smoke-voxels", type=int, default=None,
                     help="subsample voxel columns for a fast code-correctness check only -- "
                          "NOT a real measurement, skips step 1/6 (verify/restd)")
    args = ap.parse_args()

    alphas = np.logspace(1, 6, args.n_alphas)
    all_results = {}
    for subj in args.subj:
        all_results[subj] = process_subject(
            subj, args.branches, alphas, Path(args.index_dir), Path(args.betas_dir),
            Path(args.weights_dir), Path(args.pred_dir), Path(args.out_dir),
            args.targets, args.zlow, smoke_voxels=args.smoke_voxels,
            force_eig=args.force_eig, force_proj=args.force_proj)

    log("=== gate summary (Spearman rho(sigma2_pred, LOO sq err), > %.2f = PASS) ===" % SPEARMAN_GATE_RHO)
    n_pass_branch = {b: 0 for b in args.branches}
    n_subj = len(args.subj)
    for subj, results in all_results.items():
        row = "  ".join(f"{b}=rho:{r['spearman_rho']:.3f}({'P' if r['spearman_rho']>SPEARMAN_GATE_RHO else 'F'})"
                         for b, r in results.items())
        log(f"  subj0{subj}: {row}")
        for b, r in results.items():
            if r["spearman_rho"] > SPEARMAN_GATE_RHO:
                n_pass_branch[b] += 1

    branches_passing = sum(1 for b, n in n_pass_branch.items() if n >= 3)
    log(f"  branches passing on >=3/4 subjects: {branches_passing}/{len(args.branches)} "
        f"({n_pass_branch})")

    if args.smoke_voxels is None:
        gate_out = Path(args.out_dir) / "gate_summary.json"
        with open(gate_out, "w") as f:
            json.dump({"n_pass_per_branch_over_subjects": n_pass_branch,
                       "branches_passing_on_ge3_subjects": branches_passing,
                       "n_branches_total": len(args.branches),
                       "n_subjects_total": n_subj,
                       "gate_overall_pass": branches_passing >= 3}, f, indent=2)
        log(f"  wrote {gate_out}")

    log("all done")


if __name__ == "__main__":
    main()
