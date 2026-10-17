"""Post-mortem of the Spearman gate of ridge_posterior_sigma2.py: which per-trial signal carries information on error?

Inputs:  the cache of ridge_posterior_sigma2.py (required), the ridge weights, the NSD-Core betas and nsdcore_r
Outputs: per subject under --out-dir: f1_diagnostics.json (power analysis, feature screen, shift) and, only with
         --include-imagery, f1_d3_imagery.json

(1) Power: rho(sigma^2, error) against a parametric null of correctly specified data. (2) Feature screen: leverage,
reliability and other per-trial signals, same target and same null. (3) Shift: leverage of training against
shared1000 trials. NSD-Imagery is off by default: nothing is selected on it. Details above the constants.

Usage (numpy / scipy, no torch; after ridge_posterior_sigma2.py):
    export OPENBLAS_CORETYPE=SkylakeX   # some CPUs miscompute large matmuls without it
    python sigma2_vs_error_diagnostics.py --subj 1
    python sigma2_vs_error_diagnostics.py --subj 1 --branches z_sem_img --n-sim 1   # fast look
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_ridge import (  # noqa: E402
    BETAS_DIR, INDEX_DIR, ZLOW_PATH, TARGETS_H5,
    load_branch_targets, load_nsdcore_split,
)
from trisect.nsd import zscore_fmri_train_test as zscore_fmri  # noqa: E402
from ridge_posterior_sigma2 import (  # noqa: E402
    BRANCHES, OUT_DIR, RIDGE_WEIGHTS_DIR, SPEARMAN_GATE_RHO, SUBJECTS,
    block_index_for_images, block_loo_residuals, compute_projection, eigh_xtx, log,
)

# Why these measurements: the gate of ridge_posterior_sigma2.py gave rho ~ 0 at the production alpha on all four branches of
# subject 1, and a sweep showed rho falling from ~0.50 at alpha=10 to ~0 at alpha=1e5. All three are cheap given the cached
# eigendecomposition, and each can invalidate the next choice.
#   Power (d1_power). sigma^2 = s^2 * (1 + h_ii) with s^2 a per-branch scalar; rho(sigma^2, err) is rho(h, err) and the
#       whole per-trial range of sigma^2 is the range of h_ii: [0.03, 0.15] at the production alpha. Under correct
#       specification and homoscedasticity E[r_LOO^2 | h] ~ 1/(1 - h), an effect of ~14% on a quantity whose sampling
#       noise is far larger. So rho ~ 0 may be what a perfectly specified model produces, and the threshold 0.2 may have
#       been out of reach: the gate had no power analysis. A parametric null settles it: y_sim = X beta_hat(alpha_prod) +
#       homoscedastic noise (correct by construction), through the IDENTICAL pipeline at every alpha. If rho_null
#       reproduces the observed rho(alpha), the curve says nothing about specification.
#   Feature screen (d2_feature_screen). A gating needs a per-trial signal monotonically related to error, not a Bayesian
#       posterior. Each candidate is scored on the same target (per-trial mean LOO squared error) at the production alpha:
#           h(alpha_prod)  leverage of the production posterior      -> keep the formula
#           h(alpha=10)    leverage as a Mahalanobis-like atypicality -> decouple alpha_sigma from the production alpha
#           -r             repetition reliability, model-free         -> empirical fallback
#           ||y_hat||      ridge shrinkage magnitude, per branch      -> a third candidate
#           ||x||^2        raw beta power, the alpha -> inf limit of h -> control
#       Each is compared with the same null; a small rho reads as weak signal or no power; rho(h, r) says whether
#       leverage and reliability are the same axis.
#   Shift (process_subject). sigma^2 was meant for a distribution shift (imagery off the training manifold), but the gate
#       was evaluated on the training design, where leverage is bounded and homogeneous (mean h_ii = edof/n). This compares
#       h on training trials with shared1000 trials (out of design, same distribution: how much h rises from being out
#       of sample alone) and, only with --include-imagery, with NSD-Imagery.
# NSD-Imagery is off by default. Reading it is not hyperparameter selection, but choosing a sigma^2 family by what it shows
# would be; the go/no-go stays on the NSD-Core proxies; with --include-imagery the number goes to its own file, never to
# the main summary. The primitives of ridge_posterior_sigma2.py are imported, not reimplemented; the numbers of the two
# scripts cannot differ by algebra; the gate number is recomputed here as a check.
# Cost: the (n x V) @ (V x D) matmuls of the power analysis, 5 alphas x (1 real + n_sim null) x 4 branches, worst case
# z_struct (D = 2048): tens of minutes per subject with the cache warm. Peak RSS ~10 GB, from the (n_blocks, 3, V) float64
# intermediate of block_loo_residuals.
R_DIR = "data/betas_trisect/nsdcore_r"
IMAGERY_DIR = "data/betas_trisect/nsd_imagery"

# Alpha grid of the power analysis: deliberately coarse. The point is the CONTRAST between real and null at a few
# well-separated regularization regimes, not a fine curve; each extra alpha costs a full
# (n x V) @ (V x D) matmul per replicate. alpha_production is inserted at runtime.
D1_ALPHAS = (1e1, 1e3, 1e4, 1e6)


def rho_of(a, b):
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


def leverage(P, eigvals, alpha):
    """h_ii(alpha) = sum_k p_ik^2 / (d_k + alpha), for rows already projected onto the eigenbasis.

    Also returns the ridge shrinkage weights w_k = 1/(d_k + alpha) so callers do not rebuild them.
    """
    w = (1.0 / (eigvals + alpha)).astype(np.float32)
    return (P ** 2) @ w, w


def loo_sq_err(P, w, Z, yc, block_index):
    """Per-trial mean-over-dims squared leave-one-image-out residual at the ridge weights w."""
    Yhat = P @ (w[:, None] * Z)
    R = yc - Yhat
    Q = P * np.sqrt(w)[None, :]
    Rloo = block_loo_residuals(R, Q, block_index)
    del Yhat, R, Q
    return np.mean(Rloo ** 2, axis=1)


def decile_ratio(signal, err):
    """Mean err in the top decile of `signal` over the mean err in the bottom decile.

    The effect size the Spearman is silent about: a gate that merely ranks trials is only useful if
    the trials it flags actually have materially worse errors. 1.0 = no discrimination.
    """
    lo, hi = np.percentile(signal, [10, 90])
    bottom, top = err[signal <= lo], err[signal >= hi]
    if len(bottom) == 0 or len(top) == 0 or bottom.mean() == 0:
        return float("nan")
    return float(top.mean() / bottom.mean())


def pct(a, qs=(0, 1, 50, 99, 100)):
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def simulate_null(Yhat_prod, Rloo_std, rng):
    """y_sim = (fitted values at alpha_production) + i.i.d. Gaussian trial noise.

    Correctly specified by construction: the mean lies exactly in the model's column space and the
    noise is independent across trials, including across the 3 repetitions of one image (trial
    noise, which is what the ridge's error model assumes). The noise SCALE comes from the real LOO
    residuals, per output dimension; a single pooled scalar would give the dimension-averaging in
    `loo_sq_err` a different noise structure than the real data has, and that averaging is
    precisely what sets the power of the Spearman.
    """
    noise = rng.standard_normal(Yhat_prod.shape, dtype=np.float32) * Rloo_std
    return Yhat_prod + noise


def null_replicates(P, w, Yhat_prod, Rloo_std, block_index, n_sim, rng):
    """n_sim correctly-specified datasets, each carried through the identical LOO pipeline at w."""
    out = []
    for _ in range(n_sim):
        ys = simulate_null(Yhat_prod, Rloo_std, rng)
        ysc = ys - ys.mean(axis=0)
        del ys
        Z = P.T.astype(np.float32) @ ysc
        out.append(loo_sq_err(P, w, Z, ysc, block_index))
        del ysc, Z
    return out


def d1_power(P, eigvals, Zfull, yc, block_index, alpha_prod, Yhat_prod, Rloo_std,
             alphas, n_sim, rng):
    """The parametric-null power analysis: one row per alpha, plus the null errors at alpha_prod
    (returned so the feature screen does not have to simulate a second time)."""
    rows, err_null_at_prod = [], None
    for alpha in alphas:
        t0 = time.monotonic()
        h, w = leverage(P, eigvals, alpha)
        err_real = loo_sq_err(P, w, Zfull, yc, block_index)
        rho_real, _ = rho_of(h, err_real)
        dr_real = decile_ratio(h, err_real)

        errs_null = null_replicates(P, w, Yhat_prod, Rloo_std, block_index, n_sim, rng)
        rho_null = [rho_of(h, e)[0] for e in errs_null]
        dr_null = [decile_ratio(h, e) for e in errs_null]
        if np.isclose(alpha, alpha_prod):
            err_null_at_prod = errs_null

        rows.append({
            "alpha": float(alpha),
            "is_alpha_production": bool(np.isclose(alpha, alpha_prod)),
            "h_percentiles": pct(h),
            "h_mean_is_edof_over_n": float(h.mean()),
            "rho_real": rho_real,
            "rho_null_mean": float(np.mean(rho_null)),
            "rho_null_all": rho_null,
            "decile_ratio_real": dr_real,
            "decile_ratio_null_mean": float(np.mean(dr_null)),
        })
        log(f"      alpha={alpha:9.4g}  h in [{h.min():.4f}, {h.max():.4f}]  "
            f"rho_real={rho_real:+.4f}  rho_null={np.mean(rho_null):+.4f}  "
            f"({time.monotonic()-t0:.0f}s)")
    assert err_null_at_prod is not None, "alpha_production was not in the power-analysis grid"
    return rows, err_null_at_prod


def d2_feature_screen(P, eigvals, alpha_prod, Yhat_prod, err_real, err_null, r_train):
    """Score every candidate per-trial signal against the same target, real and under the null."""
    h_prod, _ = leverage(P, eigvals, alpha_prod)
    h_lo, _ = leverage(P, eigvals, 10.0)
    features = {
        "h_alpha_production": h_prod,
        "h_alpha_10": h_lo,
        "norm_yhat": np.linalg.norm(Yhat_prod, axis=1),
        "norm_x_squared": (P ** 2).sum(axis=1),
    }
    if r_train is not None:
        # r is a reliability; it should be NEGATIVELY related to error: negate it here so every
        # row of the table is oriented "larger = expected worse" and the signs stay comparable.
        features["minus_r"] = -r_train

    out = {}
    for name, sig in features.items():
        rho, p = rho_of(sig, err_real)
        out[name] = {
            "rho_real": rho,
            "p_real": p,
            "rho_null_mean": float(np.mean([rho_of(sig, e)[0] for e in err_null])),
            "decile_ratio_real": decile_ratio(sig, err_real),
            "percentiles": pct(sig),
        }
        log(f"      {name:20s} rho_real={rho:+.4f}  rho_null={out[name]['rho_null_mean']:+.4f}  "
            f"decile_ratio={out[name]['decile_ratio_real']:.3f}")

    if r_train is not None:
        # Are leverage and reliability the same axis? If they are, "switch to r" is not a switch.
        out["_rho_h_production_vs_r"] = rho_of(h_prod, r_train)[0]
        out["_rho_h_alpha10_vs_r"] = rho_of(h_lo, r_train)[0]
        log(f"      rho(h_prod, r)={out['_rho_h_production_vs_r']:+.4f}   "
            f"rho(h_alpha10, r)={out['_rho_h_alpha10_vs_r']:+.4f}")
    return out


def read_alpha_production(weights_dir: Path, branch):
    path = weights_dir / f"{branch}_weights.pkl"
    if not path.exists():
        sys.exit(f"missing {path}: run fit_ridge.py first")
    with open(path, "rb") as f:
        return float(pickle.load(f)["alpha"])


def process_subject(subj, branches, args, rng):
    log(f"=== subj0{subj} ===")
    out_dir = Path(args.out_dir) / f"subj{subj:02d}"
    if not (out_dir / "xtx_eig.npz").exists():
        sys.exit(f"missing {out_dir/'xtx_eig.npz'}: run ridge_posterior_sigma2.py for "
                 f"subj{subj:02d} first -- this script never recomputes the eigendecomposition")

    weights_dir = Path(args.weights_dir) / f"subj{subj:02d}"
    alpha_prod = {b: read_alpha_production(weights_dir, b) for b in branches}
    log("  alpha_production: " + "  ".join(f"{b}={a:.6g}" for b, a in alpha_prod.items()))

    betas, img73k, is_shared = load_nsdcore_split(subj, Path(args.index_dir), Path(args.betas_dir))
    train_mask, test_mask = ~is_shared, is_shared
    train_fmri, test_fmri = zscore_fmri(betas, train_mask, test_mask)
    del betas
    train_img73k = img73k[train_mask]

    eigvals, eigvecs = eigh_xtx(subj, train_fmri, out_dir)
    P = compute_projection(subj, train_fmri, eigvecs, out_dir)
    block_index, uniq_ids = block_index_for_images(train_img73k)
    log(f"  {len(uniq_ids)} training images x 3 reps, n_train={len(train_img73k)}, V={P.shape[1]}")

    results = {"n_train": int(len(train_img73k)), "n_voxels": int(P.shape[1]),
               "n_sim": args.n_sim, "seed": args.seed, "branches": {}}

    # --- shift analysis first; the 2 GB eigenvector matrix can be freed before the memory-heavy branch loop
    alphas_d3 = sorted(set(list(D1_ALPHAS) + list(alpha_prod.values())))
    if not args.skip_d3:
        log("  [shift] leverage: in-design train trials vs out-of-design shared1000 trials")
        t0 = time.monotonic()
        P_test = test_fmri.astype(np.float32) @ eigvecs.astype(np.float32)
        log(f"    shared1000 projection done ({time.monotonic()-t0:.0f}s), {P_test.shape}")
        d3 = {}
        for alpha in alphas_d3:
            h_tr, w = leverage(P, eigvals, alpha)
            h_te = (P_test ** 2) @ w
            d3[f"alpha_{alpha:.6g}"] = {
                "train": pct(h_tr), "shared1000": pct(h_te),
                "median_ratio_shared_over_train": float(np.median(h_te) / np.median(h_tr)),
            }
            log(f"    alpha={alpha:9.4g}  median h: train {np.median(h_tr):.5f}  "
                f"shared1000 {np.median(h_te):.5f}  "
                f"ratio {np.median(h_te)/np.median(h_tr):.3f}")
        results["d3_shift"] = d3
        del P_test

    if args.include_imagery:
        img_dir = Path(args.imagery_dir) / f"subj{subj:02d}"
        X_img = np.load(img_dir / "betas_averaged.npy").astype(np.float32)
        labels = np.load(img_dir / "labels_averaged.npy")
        log(f"  [shift-imagery] {X_img.shape[0]} NSD-Imagery rows -- SEPARATE output file, "
            f"descriptive only, NOT part of any selection")
        P_img = X_img @ eigvecs.astype(np.float32)
        d3i = {"note": "descriptive only; NSD-Imagery betas are z-scored PER RUN, not with the "
                       "NSD-core train moments, so part of any h gap is that normalization "
                       "difference and not distance from the training manifold",
               "labels": [str(x) for x in labels]}
        for alpha in alphas_d3:
            h_tr, w = leverage(P, eigvals, alpha)
            h_img = (P_img ** 2) @ w
            d3i[f"alpha_{alpha:.6g}"] = {
                "imagery": pct(h_img), "train": pct(h_tr),
                "median_ratio_imagery_over_train": float(np.median(h_img) / np.median(h_tr)),
                "imagery_per_row": [float(v) for v in h_img],
            }
            log(f"    alpha={alpha:9.4g}  median h: train {np.median(h_tr):.5f}  "
                f"imagery {np.median(h_img):.5f}  "
                f"ratio {np.median(h_img)/np.median(h_tr):.3f}")
        with open(out_dir / "f1_d3_imagery.json", "w") as f:
            json.dump(d3i, f, indent=2)
        log(f"  wrote {out_dir/'f1_d3_imagery.json'}")
        del P_img

    del eigvecs, test_fmri, train_fmri

    r_path = Path(args.r_dir) / f"subj{subj:02d}_r.npy"
    if r_path.exists():
        r_train = np.load(r_path)[train_mask]
        log(f"  r loaded: mean {r_train.mean():.4f}, "
            f"range [{r_train.min():.4f}, {r_train.max():.4f}]")
    else:
        r_train = None
        log(f"  WARNING: {r_path} missing -- the 'minus_r' row of the feature screen will be skipped")

    branch_targets = load_branch_targets(args.targets, args.zlow)

    for branch in branches:
        log(f"  [{branch}] alpha_production={alpha_prod[branch]:.6g}")
        target_full, _ = branch_targets[branch]
        y = target_full[train_img73k].astype(np.float32)
        yc = y - y.mean(axis=0)
        del y
        t0 = time.monotonic()
        Zfull = P.T.astype(np.float32) @ yc
        log(f"    Z = P^T yc done ({time.monotonic()-t0:.0f}s), D={Zfull.shape[1]}")

        h_prod, w_prod = leverage(P, eigvals, alpha_prod[branch])
        Yhat_prod = P @ (w_prod[:, None] * Zfull)
        R_prod = yc - Yhat_prod
        Q_prod = P * np.sqrt(w_prod)[None, :]
        Rloo_prod = block_loo_residuals(R_prod, Q_prod, block_index)
        del R_prod, Q_prod
        err_real = np.mean(Rloo_prod ** 2, axis=1)
        s2_f = float(np.mean(Rloo_prod ** 2))
        Rloo_std = Rloo_prod.std(axis=0).astype(np.float32)
        del Rloo_prod
        rho_gate, _ = rho_of(s2_f * (1.0 + h_prod), err_real)
        log(f"    s2_f={s2_f:.6g}  rho(sigma^2, LOO err^2)={rho_gate:+.4f}  "
            f"(must reproduce ridge_posterior_sigma2.py's gate number)")

        alphas = sorted(set(list(D1_ALPHAS) + [alpha_prod[branch]]))
        log(f"    [power] parametric null, {len(alphas)} alphas x (1 real + {args.n_sim} null)")
        d1, err_null = d1_power(P, eigvals, Zfull, yc, block_index, alpha_prod[branch],
                                Yhat_prod, Rloo_std, alphas, args.n_sim, rng)

        log("    [screen] candidate per-trial signals at alpha_production")
        d2 = d2_feature_screen(P, eigvals, alpha_prod[branch], Yhat_prod, err_real, err_null,
                               r_train)

        results["branches"][branch] = {
            "alpha_production": alpha_prod[branch],
            "s2_f": s2_f,
            "rho_sigma2_vs_loo_err": rho_gate,
            "gate_threshold": SPEARMAN_GATE_RHO,
            "d1_power": d1,
            "d2_feature_screen": d2,
        }
        del Zfull, yc, Yhat_prod, err_real, err_null, h_prod

        with open(out_dir / "f1_diagnostics.json", "w") as f:
            json.dump(results, f, indent=2)   # rewritten after every branch; a kill keeps work

    log(f"  wrote {out_dir/'f1_diagnostics.json'}")
    return results


def print_verdict(all_results):
    log("=== diagnostics summary ===")
    for subj, res in all_results.items():
        for branch, r in res["branches"].items():
            log(f"  --- subj0{subj} {branch} (alpha_production={r['alpha_production']:.4g}, "
                f"gate rho={r['rho_sigma2_vs_loo_err']:+.4f} vs threshold {r['gate_threshold']}) ---")
            log("    power   rho(h, LOO err^2), real vs a CORRECTLY SPECIFIED null:")
            for row in r["d1_power"]:
                mark = " <- production" if row["is_alpha_production"] else ""
                log(f"      alpha={row['alpha']:9.4g}  h=[{row['h_percentiles']['p0']:.4f}, "
                    f"{row['h_percentiles']['p100']:.4f}]  real={row['rho_real']:+.4f}  "
                    f"null={row['rho_null_mean']:+.4f}  decile real="
                    f"{row['decile_ratio_real']:.3f} null={row['decile_ratio_null_mean']:.3f}{mark}")
            log("    screen  candidate per-trial signals (same target, same statistic):")
            for name, d in r["d2_feature_screen"].items():
                if name.startswith("_"):
                    continue
                log(f"      {name:20s} real={d['rho_real']:+.4f}  null={d['rho_null_mean']:+.4f}  "
                    f"decile_ratio={d['decile_ratio_real']:.3f}")
            for name, v in r["d2_feature_screen"].items():
                if name.startswith("_"):
                    log(f"      {name:20s} {v:+.4f}")
        if "d3_shift" in res:
            log(f"  --- subj0{subj} shift: median h, train vs shared1000 ---")
            for key, d in res["d3_shift"].items():
                log(f"      {key:20s} train={d['train']['p50']:.5f}  "
                    f"shared1000={d['shared1000']['p50']:.5f}  "
                    f"ratio={d['median_ratio_shared_over_train']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=[1], choices=list(SUBJECTS))
    ap.add_argument("--branches", type=str, nargs="+", default=list(BRANCHES),
                    choices=list(BRANCHES))
    ap.add_argument("--n-sim", type=int, default=2,
                    help="null replicates per alpha (the Spearman over 27k trials has tiny "
                         "sampling error, so 1-2 is enough to see whether real == null)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", type=str, default=TARGETS_H5)
    ap.add_argument("--zlow", type=str, default=ZLOW_PATH)
    ap.add_argument("--index-dir", type=str, default=INDEX_DIR)
    ap.add_argument("--betas-dir", type=str, default=BETAS_DIR)
    ap.add_argument("--weights-dir", type=str, default=RIDGE_WEIGHTS_DIR)
    ap.add_argument("--r-dir", type=str, default=R_DIR)
    ap.add_argument("--imagery-dir", type=str, default=IMAGERY_DIR)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--skip-d3", action="store_true",
                    help="skip the train-vs-shared1000 leverage comparison")
    ap.add_argument("--include-imagery", action="store_true",
                    help="also report leverage on NSD-Imagery trials, into a SEPARATE file. "
                         "Descriptive only -- no selection may be made on it.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    all_results = {}
    for subj in args.subj:
        all_results[subj] = process_subject(subj, args.branches, args, rng)
    print_verdict(all_results)
    log("all done")


if __name__ == "__main__":
    main()
