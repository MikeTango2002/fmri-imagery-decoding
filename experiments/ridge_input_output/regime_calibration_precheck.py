"""Pre-check of regime calibration in latent space (Section 8.1.5). CPU only, numpy, seconds.

Inputs:  the corrections of regime_calibration_fit.py, the 384 imagery betas (betas_flat.npy), the targets of the 18 stimuli
Outputs: data/results/ridge_input_output/regime_calibration_k128/{precheck.tsv, bootstrap json}

Applies none / gain / diag / coral before predict_flat() and compares cos_diag with the targets on two
12-stimulus pools: vision and imagery. A permission to spend GPU time, not a result. Prediction fixed before
the run: about no change on vision, something on imagery; if both move alike it is a generic effect, and the
console says so. The shared1000 dev part is an in-distribution control. Paired bootstrap over stimulus
clusters (the 4 subjects of a drawn stimulus move together), directly on cos_diag.

Usage (.venv_targets active, from the project root):
    python regime_calibration_precheck.py --subj 1 2 5 7
"""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_ridge import two_way  # noqa: E402
from predict_ridge_imagery import (  # noqa: E402
    average_by_label, group_rows, load_trial_level, load_weights, predict_flat,
)
from trisect.regime_calibration import (  # noqa: E402
    apply_correction, coral_transform, diag_gain_transform, gain_transform,
)
from trisect.ridge import restandardize  # noqa: E402

SUBJECTS = (1, 2, 5, 7)
BRANCHES = ("z_sem_img", "z_struct", "z_low")  # z_sem_txt has no NSD-Imagery GT (see docstring)
VARIANTS = ("none", "gain", "diag", "coral")
BRANCH_RESHAPE = {"z_struct": (8, 8, 32)}

IMAGERY_DIR = "data/betas_trisect/nsd_imagery"
WEIGHTS_DIR = "data/betas_trisect/ridge_weights"
CORRECTION_DIR = "data/betas_trisect/h3_correction"
IMG18_TARGETS = "data/targets/targets_imagery18.npz"
S1K_INDEX_DIR = "data/nsdcore_index"
S1K_BETAS_DIR = "data/mindeyev2_dataset"
S1K_PRED_DIR = "data/betas_trisect/nsdcore_predictions"
TARGETS_H5 = "data/targets/targets.h5"
OUT_DIR = "data/results/ridge_input_output/regime_calibration_k128"
N_BOOT = 2000
RNG_SEED = 0

IMAGERY_POOLS = {"img_vision": ["visionsimple", "visioncomplex"],
                  "img_imagery": ["imagerysimple", "imagerycomplex"]}


def flat(a):
    return a.reshape(a.shape[0], -1).astype(np.float64)


def per_row_cos(pred, gt, mean):
    p, g = flat(pred) - mean, flat(gt) - mean
    pn = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
    gn = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    return np.sum(pn * gn, axis=1)  # (n,)


def load_correction(subj, correction_dir=CORRECTION_DIR):
    d = Path(correction_dir) / f"subj{subj:02d}"
    V_k = np.load(d / "V_k.npy").astype(np.float64)
    mu_calib = np.load(d / "mu_calib.npy")
    diag_gain = np.load(d / "diag_gain.npy")
    coral_A = np.load(d / "coral_A.npy")
    with open(d / "gain_scalar.json") as f:
        g = json.load(f)["gain_scalar"]
    with open(d / "coral_diagnostics.json") as f:
        coral_diag = json.load(f)
    return V_k, mu_calib, g, diag_gain, coral_A, coral_diag


def transform_for(variant, g, diag_gain, coral_A):
    if variant == "none":
        return lambda zc: zc
    if variant == "gain":
        return gain_transform(g)
    if variant == "diag":
        return diag_gain_transform(diag_gain)
    if variant == "coral":
        return coral_transform(coral_A)
    raise ValueError(variant)


def predict_all_branches(X_flat, labels_flat, labels_avg, weights_dir, pool_mode="modality12"):
    """One (30, D) prediction per branch: predict_flat -> average_by_label -> restandardize,
    identical recipe to predict_ridge_imagery.py --trial-level (reused, not reimplemented)."""
    out = {}
    for branch in BRANCHES:
        W, b = load_weights(weights_dir / f"{branch}_weights.pkl")
        raw = predict_flat(X_flat, W, b)
        raw_avg = average_by_label(raw, labels_flat, labels_avg)
        stats = np.load(weights_dir / "train_target_stats.npz")
        tgt_mean, tgt_std = stats[f"{branch}_mean"], stats[f"{branch}_std"]
        pred = np.empty_like(raw_avg)
        for _, rows in group_rows(labels_avg, pool_mode):
            pool = raw_avg[rows]
            pred[rows] = restandardize(pool, pool.mean(axis=0), pool.std(axis=0), tgt_mean, tgt_std)
        out[branch] = pred.reshape((pred.shape[0],) + BRANCH_RESHAPE.get(branch, tgt_mean.shape))
    return out


def load_s1k_dev(subj, index_dir, betas_dir, weights_dir):
    """A small in-distribution control pool: shared1000 test rows, single trial (no averaging --
    this script only needs a perturbation check, not a real 2WC), z-scored with train moments
    (the recipe of fit_ridge.py), NOT the per-run z-score of NSD-Imagery; the correction
    (fit in per-run-zscored space) is expected to be close to a no-op in absolute terms here;
    the point is only to confirm it does not blow up predictions it was never meant to touch."""
    from fit_ridge import load_nsdcore_split
    from trisect.nsd import zscore_fmri_train_test as zscore_fmri
    betas, img73k, is_shared = load_nsdcore_split(subj, Path(index_dir), Path(betas_dir))
    train_mask, test_mask = ~is_shared, is_shared
    _, test_fmri = zscore_fmri(betas, train_mask, test_mask)
    return test_fmri.astype(np.float32), img73k[test_mask]


def cos_delta_bootstrap(deltas_by_subj_stim, n_boot=N_BOOT, seed=RNG_SEED):
    """deltas_by_subj_stim: (n_subj, n_stim) array of per-(subject,stimulus) deltas
    (variant cos_diag - none cos_diag). Stimulus-cluster bootstrap: resample stimulus columns
    with replacement, all subjects of a drawn stimulus move together (bootstrap_ci_imagery.py's
    'stimulus' scheme, reimplemented directly on this array shape)."""
    rng = np.random.default_rng(seed)
    n_stim = deltas_by_subj_stim.shape[1]
    point = float(deltas_by_subj_stim.mean())
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_stim, size=n_stim)
        boots[i] = deltas_by_subj_stim[:, idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def process_subject(subj, rows, s1k_cache, correction_dir=CORRECTION_DIR):
    sd = f"subj{subj:02d}"
    print(f"\n=== {sd} ===", flush=True)
    imagery_dir = Path(IMAGERY_DIR) / sd
    weights_dir = Path(WEIGHTS_DIR) / sd
    X_flat, labels_flat = load_trial_level(imagery_dir)
    labels_avg = np.load(imagery_dir / "labels_averaged.npy")

    V_k, mu_calib, g, diag_gain, coral_A, coral_diag = load_correction(subj, correction_dir)
    print(f"  correction: gain={g:.4f}  coral stable={coral_diag['stable']} "
          f"(cond={coral_diag['cond_number_shrunk']:.3g})", flush=True)

    img18 = np.load(IMG18_TARGETS)
    gt18 = {"z_sem_img": img18["z_sem_img"], "z_struct": img18["z_struct"],
            "z_low": img18["z_low"][0]}
    stats = np.load(weights_dir / "train_target_stats.npz")
    means = {b: stats[f"{b}_mean"].astype(np.float64) for b in BRANCHES}

    conds = np.array([str(l).rsplit("-", 1)[0] for l in labels_avg])
    gidx = np.array([int(str(l).rsplit("-", 1)[1]) for l in labels_avg])

    s1k_fmri, s1k_ids = s1k_cache
    with h5py.File(TARGETS_H5, "r") as f:
        z_sem_all = f["z_sem"][:].astype(np.float64)
        z_struct_all = f["z_struct"][:].astype(np.float64)
    z_low_all = np.load("data/targets/pca_zlow/z_low.npy", mmap_mode="r")
    s1k_gt = {"z_sem_img": z_sem_all[s1k_ids, 0], "z_struct": z_struct_all[s1k_ids],
              "z_low": np.asarray(z_low_all[s1k_ids], dtype=np.float64)}

    per_variant_cos = {}  # variant -> domain -> branch -> (n,) per-row cosine, in gidx order
    for variant in VARIANTS:
        tf = transform_for(variant, g, diag_gain, coral_A)
        X_corr = apply_correction(X_flat, V_k, mu_calib, tf).astype(np.float32)
        preds = predict_all_branches(X_corr, labels_flat, labels_avg, weights_dir)

        for pool, keep in IMAGERY_POOLS.items():
            sel = np.isin(conds, keep)
            order = np.argsort(gidx[sel])
            for b in BRANCHES:
                pred = preds[b][sel][order]
                gt = gt18[b][gidx[sel][order]]
                c = per_row_cos(pred, gt, means[b])
                per_variant_cos.setdefault(variant, {}).setdefault(pool, {})[b] = c
                rows.append(dict(subject=sd, domain=pool, branch=b, variant=variant,
                                 n=len(c), cos_diag=float(c.mean())))

        # shared1000 sanity control: correct s1k_fmri the same way, predict, cos_diag vs targets.h5
        s1k_corr = apply_correction(s1k_fmri, V_k, mu_calib, tf).astype(np.float32)
        for b in BRANCHES:
            W, bias = load_weights(weights_dir / f"{b}_weights.pkl")
            raw = predict_flat(s1k_corr, W, bias)
            tgt_mean, tgt_std = stats[f"{b}_mean"], stats[f"{b}_std"]
            pred = restandardize(raw, raw.mean(axis=0), raw.std(axis=0), tgt_mean, tgt_std)
            c = per_row_cos(pred, s1k_gt[b], means[b])
            per_variant_cos.setdefault(variant, {}).setdefault("s1k_dev", {})[b] = c
            rows.append(dict(subject=sd, domain="s1k_dev", branch=b, variant=variant,
                             n=len(c), cos_diag=float(c.mean())))

        print(f"  [{variant}] " + "  ".join(
            f"{pool}/{b}={np.mean([r['cos_diag'] for r in rows if r['subject']==sd and r['domain']==pool and r['branch']==b and r['variant']==variant]):.4f}"
            for pool in list(IMAGERY_POOLS) + ["s1k_dev"] for b in BRANCHES), flush=True)

    return per_variant_cos


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--correction-dir", type=str, default=CORRECTION_DIR)
    args = ap.parse_args()

    rows = []
    per_subj_cos = {}
    for subj in args.subj:
        s1k_fmri, s1k_ids = load_s1k_dev(subj, S1K_INDEX_DIR, S1K_BETAS_DIR,
                                         Path(WEIGHTS_DIR) / f"subj{subj:02d}")
        per_subj_cos[subj] = process_subject(subj, rows, (s1k_fmri, s1k_ids), args.correction_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    cols = ["subject", "domain", "branch", "variant", "n", "cos_diag"]
    tsv_path = os.path.join(args.out_dir, "precheck.tsv")
    with open(tsv_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")
    print(f"\nWrote {tsv_path}")

    # paired bootstrap: variant - none, per domain/branch, stimulus-cluster over the 12 stimuli.
    # s1k_dev is excluded here (not just from the summary print): its ~1000 rows are not
    # guaranteed to be in the same image-id order across subjects (is_shared selects the same
    # 1000 shared1000 images per subject, but the trial order differs); pairing position i
    # across subjects would silently mismatch stimuli. It stays in the plain TSV (per-subject
    # mean, no cross-subject pairing needed) as a coarse stability check, not as a claim with a CI.
    print("\n=== paired delta (variant - none), stimulus-cluster bootstrap 95% CI ===")
    boot_rows = []
    for variant in VARIANTS[1:]:
        for pool in list(IMAGERY_POOLS):
            for b in BRANCHES:
                try:
                    deltas = np.stack([per_subj_cos[s][variant][pool][b] - per_subj_cos[s]["none"][pool][b]
                                       for s in args.subj])
                except KeyError:
                    continue
                point, lo, hi = cos_delta_bootstrap(deltas)
                sig = "***" if (lo > 0 or hi < 0) else ""
                print(f"  [{variant}] {pool}/{b}: delta={point:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] {sig}")
                boot_rows.append(dict(variant=variant, domain=pool, branch=b,
                                      delta=point, ci_lo=lo, ci_hi=hi))
    boot_path = os.path.join(args.out_dir, "precheck_bootstrap.json")
    with open(boot_path, "w") as f:
        json.dump(boot_rows, f, indent=2)
    print(f"\nWrote {boot_path}")
    print("\nall done")


if __name__ == "__main__":
    main()
