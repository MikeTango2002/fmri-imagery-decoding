"""Fit the three regime-calibration corrections (scalar gain, per-direction gain, CORAL) from the calibration set.

Inputs:  the calibration set of regime_calibration_extract.py; xtx_eig.npz and train_proj.npy cached by
         ridge_posterior_sigma2.py (about 2 GB per subject)
Outputs: data/betas_trisect/h3_correction/subj0N/{V_k, mu_calib, diag_gain, coral_A}.npy and the JSONs
         (gain_scalar, coral_diagnostics, leverage_u0, meta)

Label-blind: every quantity is a moment or eigenvalue of the whole calibration set, or grouped by run type only.
K=128 directions and a CORAL shrinkage of 0.3 are fixed in advance (336 rows, M/K ~ 2.6; Ledoit-Wolf).
`diag_gain` is the fallback if CORAL is unstable. The identity transform must return its input (~1e-9).

Usage (from the project root, CPU only):
    python regime_calibration_fit.py --subj 1 2 5 7
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trisect.regime_calibration import (  # noqa: E402
    apply_correction, coral_transform, fit_coral, fit_diag_gain, fit_gain_scalar,
    leverage_u, project_topk,
)

SUBJECTS = (1, 2, 5, 7)
BRANCHES = ("z_sem_img", "z_sem_txt", "z_struct", "z_low")

RIDGE_POSTERIOR_DIR = "data/betas_trisect/ridge_posterior"
CALIBRATION_DIR = "data/betas_trisect/nsd_imagery"
WEIGHTS_DIR = "data/betas_trisect/ridge_weights"
OUT_DIR = "data/betas_trisect/h3_correction"

K = 128
SHRINKAGE = 0.3
COND_THRESHOLD = 1e4


def load_alpha_production(weights_dir: Path, subj, branch):
    with open(weights_dir / f"subj{subj:02d}" / f"{branch}_weights.pkl", "rb") as f:
        return float(pickle.load(f)["alpha"])


def sanity_identity(x, V_k, mu_calib):
    """apply_correction with the identity transform must be a numerical no-op."""
    out = apply_correction(x, V_k, mu_calib, lambda zc: zc)
    max_abs_diff = float(np.max(np.abs(out.astype(np.float64) - x.astype(np.float64))))
    return max_abs_diff


def process_subject(subj, out_dir: Path, k=K, shrinkage=SHRINKAGE):
    print(f"\n=== subj0{subj} (k={k}, shrinkage={shrinkage}) ===", flush=True)
    post_dir = Path(RIDGE_POSTERIOR_DIR) / f"subj{subj:02d}"
    eig = np.load(post_dir / "xtx_eig.npz")
    eigvals, eigvecs = eig["eigvals"], eig["eigvecs"]
    n_train = np.load(post_dir / "train_proj.npy", mmap_mode="r").shape[0]
    print(f"  eigendecomposition: V={eigvals.shape[0]}, n_train={n_train}", flush=True)

    calib_dir = Path(CALIBRATION_DIR) / f"subj{subj:02d}" / "calibration"
    betas = np.load(calib_dir / "betas.npy")
    run = np.load(calib_dir / "run.npy")
    print(f"  calibration set: {betas.shape}, runs {dict(zip(*np.unique(run, return_counts=True)))}",
          flush=True)

    Z_calib, V_k, eigvals_topk = project_topk(betas, eigvecs, eigvals, k)
    print(f"  projected onto top-{k} eigendirections (eigval range "
          f"[{eigvals_topk.min():.3g}, {eigvals_topk.max():.3g}])", flush=True)

    g = fit_gain_scalar(Z_calib, eigvals_topk, n_train)
    diag_gain = fit_diag_gain(Z_calib, eigvals_topk, n_train)
    A, mu_calib, coral_diag = fit_coral(Z_calib, eigvals_topk, n_train,
                                        shrinkage=shrinkage, cond_threshold=COND_THRESHOLD)
    print(f"  gain_scalar={g:.4f}  diag_gain mean={diag_gain.mean():.4f} "
          f"(min {diag_gain.min():.4f}, max {diag_gain.max():.4f})", flush=True)
    print(f"  coral: cond_number_shrunk={coral_diag['cond_number_shrunk']:.3g} "
          f"({'STABLE' if coral_diag['stable'] else 'UNSTABLE, see the module docstring'})", flush=True)

    max_abs_diff = sanity_identity(betas, V_k, mu_calib)
    print(f"  sanity (identity transform, must be ~0): max_abs_diff={max_abs_diff:.3e}",
          flush=True)
    if max_abs_diff > 1e-3:
        sys.exit(f"  *** subj0{subj}: identity-transform sanity check FAILED "
                 f"(max_abs_diff={max_abs_diff:.3e}) -- do not trust the artifacts below ***")

    subj_out = out_dir / f"subj{subj:02d}"
    subj_out.mkdir(parents=True, exist_ok=True)
    np.save(subj_out / "V_k.npy", V_k.astype(np.float32))
    np.save(subj_out / "mu_calib.npy", mu_calib.astype(np.float64))
    np.save(subj_out / "diag_gain.npy", diag_gain.astype(np.float64))
    np.save(subj_out / "coral_A.npy", A.astype(np.float64))
    with open(subj_out / "gain_scalar.json", "w") as f:
        json.dump({"gain_scalar": g}, f, indent=2)
    with open(subj_out / "coral_diagnostics.json", "w") as f:
        json.dump(coral_diag, f, indent=2)

    # Leverage baseline u0, per branch (alpha is branch-specific), grouped by run TYPE only,
    # so it stays label-blind. Computed on the FULL eigenbasis, not on the top K: eigvecs and
    # eigvals are used directly, with no truncation.
    leverage = {}
    for branch in BRANCHES:
        alpha_prod = load_alpha_production(Path(WEIGHTS_DIR), subj, branch)
        u = leverage_u(betas, eigvecs, eigvals, alpha_prod)
        by_run = {}
        for r in np.unique(run):
            sel = run == r
            by_run[str(r)] = {"mean": float(u[sel].mean()), "std": float(u[sel].std()),
                              "n": int(sel.sum())}
        leverage[branch] = {"alpha_production": alpha_prod, "by_run": by_run,
                            "overall_mean": float(u.mean()), "overall_std": float(u.std())}
        by_run_str = ", ".join(f"{k}:{v['mean']:.3g}" for k, v in by_run.items())
        print(f"  [{branch}] alpha={alpha_prod:.4g}  u0 overall mean={u.mean():.4g} "
              f"by_run={{ {by_run_str} }}", flush=True)
    with open(subj_out / "leverage_u0.json", "w") as f:
        json.dump(leverage, f, indent=2)

    with open(subj_out / "meta.json", "w") as f:
        json.dump({"k": k, "shrinkage": shrinkage, "cond_threshold": COND_THRESHOLD,
                   "n_train": int(n_train), "n_calib": int(betas.shape[0]),
                   "v_voxels": int(betas.shape[1]),
                   "sanity_identity_max_abs_diff": max_abs_diff}, f, indent=2)
    print(f"  wrote {subj_out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--k", type=int, default=K,
                    help="top-K eigendirections (M=336 calibration rows, so M/k should stay "
                         "comfortably above ~2-3 for the shrinkage estimator to be meaningful; "
                         "the default is the value of the primary fit)")
    ap.add_argument("--shrinkage", type=float, default=SHRINKAGE)
    args = ap.parse_args()

    for subj in args.subj:
        process_subject(subj, Path(args.out_dir), k=args.k, shrinkage=args.shrinkage)
    print("\nall done")


if __name__ == "__main__":
    main()
