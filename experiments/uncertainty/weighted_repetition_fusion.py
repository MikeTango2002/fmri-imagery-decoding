"""Weighted fusion of the repetitions of an image, on the NSD-core development split, with the frozen ridge (Section 8.1.2).

Inputs:  the production ridge weights, data/betas_trisect/nsdcore_r/subj0N_r.npy (per-trial leave-one-out
         correlation r_i), the targets, the dev mask (3,000 trials = 1,000 dev images x 3 repetitions)
Outputs: data/results/uncertainty/weighted_repetition_fusion_dev.tsv (cosine per dev image and arm, paired bootstrap)

No retraining. Arms: (u) uniform mean of the 3 per-trial predictions (control); (w) weights ~ max(r_i, 0), the primary
arm; (v) uniform mean of the 3 z-scored voxel rows, then predict (what the pipeline does); Fisher weights r/(1-r) are
secondary. Gate, fixed before running: (w) beats (u) if the paired 95% CI excludes zero on at least 2 of 3 factors and
3 of 4 subjects. The fusion order and the control check are explained above the constants.

Usage (numpy / sklearn; after fit_ridge.py, the targets and derive_nsdcore_dev_mask.py):
    python weighted_repetition_fusion.py --subj 1 2 5 7
"""
import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import r2_score

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent  # folder that holds the five reproductions
sys.path.insert(0, str(_REPO_ROOT / "icnn" / "src"))
from prepare_nsd_imagery_icnn import NUM_VOXELS  # noqa: E402

from trisect.nsd import zscore_fmri  # noqa: E402
from trisect.ridge import restandardize  # noqa: E402

# Fusion happens on the RAW ridge output, before restandardize(), which is applied once per arm to the fused (n_dev, D)
# matrix with that arm's pool mean and std. Restandardizing each trial and then averaging would shrink the pool variance
# and put (u) and (v) out of step.
# Control check, before any verdict: predict_flat() is affine; mean_i(x_i @ W.T + b) == (mean_i x_i) @ W.T + b. The raw
# fused predictions of (u) and (v) must agree up to float32 round-off, per branch and subject; otherwise the script aborts.
# r_i is not recomputed: it is the per-trial leave-one-out correlation that prepare_betas.py --stage core writes for the
# 30,000 NSD-Core trials, indexed down to the 3,000 dev trial rows.
# Metric: cosine per dev image between the target and the restandardized prediction, one metric for the three factors so
# that the same paired bootstrap applies. z_low also gets a descriptive global R^2 per arm; it is not the gate metric, as
# R^2 has no per-image decomposition to pair a bootstrap on.
SUBJECTS = (1, 2, 5, 7)
FACTORS = ("z_sem_img", "z_struct", "z_low")          # the 3 factors the gate decides on
SECONDARY_FACTORS = ("z_sem_txt",)                     # reported, not part of the gate

INDEX_DIR = "data/nsdcore_index"
BETAS_DIR = "data/mindeyev2_dataset"
NSDCORE_R_DIR = "data/betas_trisect/nsdcore_r"
DEV_MASK_DIR = "data/targets"
RIDGE_WEIGHTS_DIR = "data/betas_trisect/ridge_weights"
TARGETS_H5 = "data/targets/targets.h5"
ZLOW_PATH = "data/targets/pca_zlow/z_low.npy"
OUT_DIR = "data/results/uncertainty"

N_BOOT = 2000
SEED = 0
N_REPS = 3


def predict_flat(X, W, b):
    """sklearn RidgeCV.coef_ convention: W is (D, V), b is (D,) -> pred = X @ W.T + b."""
    return (X.astype(np.float32) @ W.T.astype(np.float32) + b.astype(np.float32)).astype(np.float32)


def cosine_rows(a, b, eps=1e-8):
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, eps)


def group_mean(values, inv, n_groups, weights=None):
    """(n_trials, D), (n_trials,) group id in [0, n_groups) -> (n_groups, D) mean per group.

    weights=None is a uniform mean. weights given -> weighted mean, with groups whose weight
    sum is ~0 falling back to uniform (declared, not silently zeroed) rather than dividing by
    zero.
    """
    n_trials, D = values.shape
    if weights is None:
        weights = np.ones(n_trials, dtype=np.float64)
    weights = weights.astype(np.float64).copy()

    wsum = np.zeros(n_groups, dtype=np.float64)
    np.add.at(wsum, inv, weights)
    dead = wsum <= 1e-12
    n_fallback = int(dead.sum())
    if n_fallback:
        in_dead_group = dead[inv]
        weights[in_dead_group] = 1.0
        wsum = np.zeros(n_groups, dtype=np.float64)
        np.add.at(wsum, inv, weights)

    acc = np.zeros((n_groups, D), dtype=np.float64)
    np.add.at(acc, inv, values.astype(np.float64) * weights[:, None])
    return (acc / wsum[:, None]).astype(np.float32), n_fallback


def paired_bootstrap_ci(diff, rng, n_boot=N_BOOT):
    """Percentile 95% CI of mean(diff), resampling the paired units (dev images) with
    replacement: same style as bootstrap_ci_imagery.py's paired mode, specialized to a flat
    (no-cluster) unit here because every dev image is already its own independent draw (unlike
    the (subject, stimulus) design of the Kneeland harness)."""
    n = len(diff)
    idx = np.arange(n)
    means = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.choice(idx, size=n, replace=True)
        means[b] = diff[sel].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def load_targets_h5(path):
    with h5py.File(path, "r") as f:
        z_sem = f["z_sem"][:].astype(np.float32)          # (73000, 2, 128)
        z_struct = f["z_struct"][:].astype(np.float32)    # (73000, 8, 8, 32)
    return z_sem, z_struct


def process_subject(subj, args, z_sem, z_struct, z_low_full, rng):
    print(f"\n=== subj0{subj} ===")
    idx = np.load(Path(args.index_dir) / f"subj{subj:02d}_trial73k_index.npz")
    img73k, is_shared = idx["img73k"], idx["is_shared"]

    betas_file = Path(args.betas_dir) / f"betas_all_subj{subj:02d}_fp32_renorm.hdf5"
    with h5py.File(betas_file, "r") as h:
        betas = np.asarray(h["betas"], dtype=np.float32)  # (30000, V)
    assert betas.shape[1] == NUM_VOXELS[subj], \
        f"{betas.shape[1]} voxels, expected {NUM_VOXELS[subj]}"

    fmri = zscore_fmri(betas, fit_mask=~is_shared)  # (30000, V), production's own recipe
    del betas

    r_all = np.load(Path(args.r_dir) / f"subj{subj:02d}_r.npy")
    assert r_all.shape[0] == fmri.shape[0]

    dev_mask73k = np.load(Path(args.dev_mask_dir) / f"nsdcore_dev_mask_subj{subj:02d}.npy")
    dev_trial_mask = dev_mask73k[img73k]
    assert not np.any(dev_trial_mask & is_shared), "dev mask overlaps shared1000"

    dev_img_per_trial = img73k[dev_trial_mask]
    uniq_dev_ids, inv = np.unique(dev_img_per_trial, return_inverse=True)
    counts = np.bincount(inv)
    bad = int((counts != N_REPS).sum())
    if bad:
        sys.exit(f"  {bad}/{len(uniq_dev_ids)} dev images do not have exactly {N_REPS} reps")
    n_dev = len(uniq_dev_ids)
    print(f"  {n_dev} dev images x {N_REPS} reps = {dev_trial_mask.sum()} trials")

    Xd = fmri[dev_trial_mask]           # (n_dev*3, V), same row order as dev_img_per_trial/inv
    rd = r_all[dev_trial_mask]          # (n_dev*3,)
    del fmri
    print(f"  r on dev trials: mean {rd.mean():.4f}  min {rd.min():.4f}  max {rd.max():.4f}")

    x_mean_v, _ = group_mean(Xd, inv, n_dev)      # (n_dev, V) uniform voxel-space average

    w_prec = np.maximum(rd, 0.0)
    r_clip = np.clip(rd, 0.0, 0.999)
    w_fisher = r_clip / (1.0 - r_clip)

    weights_dir = Path(args.weights_dir) / f"subj{subj:02d}"
    stats = np.load(weights_dir / "train_target_stats.npz")

    oracle_of = {
        "z_sem_img": z_sem[uniq_dev_ids, 0, :],
        "z_sem_txt": z_sem[uniq_dev_ids, 1, :],
        "z_struct": z_struct[uniq_dev_ids].reshape(n_dev, -1),
        "z_low": z_low_full[uniq_dev_ids],
    }

    rows = []
    for branch in FACTORS + SECONDARY_FACTORS:
        wpath = weights_dir / f"{branch}_weights.pkl"
        with open(wpath, "rb") as f:
            d = pickle.load(f)
        W, b = d["weight"], d["bias"]

        raw_flat = predict_flat(Xd, W, b)                       # (n_dev*3, D)
        raw_u, _ = group_mean(raw_flat, inv, n_dev)
        raw_w, n_fb_w = group_mean(raw_flat, inv, n_dev, weights=w_prec)
        raw_fisher, n_fb_f = group_mean(raw_flat, inv, n_dev, weights=w_fisher)
        raw_v = predict_flat(x_mean_v, W, b)                    # (n_dev, D)

        ctrl = float(np.abs(raw_u.astype(np.float64) - raw_v.astype(np.float64)).max())
        if ctrl > 1e-3:
            sys.exit(f"  [{branch}] CONTROL FAILED: max|raw_u - raw_v| = {ctrl:.3e} "
                      f"(expected ~1e-6, ridge predict should be affine) -- aborting, "
                      f"no verdict. Check the fusion order.")
        print(f"  [{branch}] control max|raw_u-raw_v| = {ctrl:.2e}  OK"
              f"  (fallback-to-uniform groups: w={n_fb_w}, fisher={n_fb_f})")

        tgt_mean, tgt_std = stats[f"{branch}_mean"], stats[f"{branch}_std"]
        arms_raw = {"u": raw_u, "w": raw_w, "v": raw_v, "fisher": raw_fisher}
        arms_z = {}
        for name, raw in arms_raw.items():
            pool_mean, pool_std = raw.mean(axis=0), raw.std(axis=0)
            arms_z[name] = restandardize(raw, pool_mean, pool_std, tgt_mean, tgt_std)

        ctrl_z = float(np.abs(arms_z["u"].astype(np.float64) - arms_z["v"].astype(np.float64)).max())
        assert ctrl_z < 1e-2, f"[{branch}] post-restandardize control failed: {ctrl_z:.3e}"

        oracle = oracle_of[branch]
        cos = {name: cosine_rows(z, oracle) for name, z in arms_z.items()}

        diff_wu = cos["w"] - cos["u"]
        diff_fu = cos["fisher"] - cos["u"]
        mean_wu, lo_wu, hi_wu = paired_bootstrap_ci(diff_wu, rng)
        mean_fu, lo_fu, hi_fu = paired_bootstrap_ci(diff_fu, rng)

        gate_pass = lo_wu > 0
        print(f"  [{branch}] cos(u)={cos['u'].mean():.4f}  cos(w)={cos['w'].mean():.4f}  "
              f"cos(v)={cos['v'].mean():.4f}  cos(fisher)={cos['fisher'].mean():.4f}")
        print(f"  [{branch}] w-u: {mean_wu:+.4f} [{lo_wu:+.4f},{hi_wu:+.4f}]  "
              f"{'PASS' if gate_pass else 'fail'}   "
              f"fisher-u: {mean_fu:+.4f} [{lo_fu:+.4f},{hi_fu:+.4f}] (secondary)")

        r2 = None
        if branch == "z_low":
            r2 = {name: float(r2_score(oracle, z, multioutput="variance_weighted"))
                  for name, z in arms_z.items()}
            print(f"  [{branch}] global R^2: u={r2['u']:.4f}  w={r2['w']:.4f}  "
                  f"v={r2['v']:.4f}  fisher={r2['fisher']:.4f} (descriptive, not the gate)")

        rows.append({
            "subj": subj, "branch": branch, "n_dev": n_dev,
            "cos_u": float(cos["u"].mean()), "cos_w": float(cos["w"].mean()),
            "cos_v": float(cos["v"].mean()), "cos_fisher": float(cos["fisher"].mean()),
            "diff_wu_mean": mean_wu, "diff_wu_ci_lo": lo_wu, "diff_wu_ci_hi": hi_wu,
            "diff_fu_mean": mean_fu, "diff_fu_ci_lo": lo_fu, "diff_fu_ci_hi": hi_fu,
            "gate_pass": gate_pass, "is_gate_factor": branch in FACTORS,
            "r2_u": r2["u"] if r2 else "", "r2_w": r2["w"] if r2 else "",
            "r2_v": r2["v"] if r2 else "", "r2_fisher": r2["fisher"] if r2 else "",
            "n_fallback_w": n_fb_w, "n_fallback_fisher": n_fb_f,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--index-dir", type=str, default=INDEX_DIR)
    ap.add_argument("--betas-dir", type=str, default=BETAS_DIR)
    ap.add_argument("--r-dir", type=str, default=NSDCORE_R_DIR)
    ap.add_argument("--dev-mask-dir", type=str, default=DEV_MASK_DIR)
    ap.add_argument("--weights-dir", type=str, default=RIDGE_WEIGHTS_DIR)
    ap.add_argument("--targets", type=str, default=TARGETS_H5)
    ap.add_argument("--zlow", type=str, default=ZLOW_PATH)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    print(f"Loading oracle targets: {args.targets}, {args.zlow}")
    z_sem, z_struct = load_targets_h5(args.targets)
    z_low_full = np.load(args.zlow).astype(np.float32)

    all_rows = []
    for subj in args.subj:
        all_rows.extend(process_subject(subj, args, z_sem, z_struct, z_low_full, rng))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "weighted_repetition_fusion_dev.tsv"
    cols = ["subj", "branch", "n_dev", "cos_u", "cos_w", "cos_v", "cos_fisher",
            "diff_wu_mean", "diff_wu_ci_lo", "diff_wu_ci_hi",
            "diff_fu_mean", "diff_fu_ci_lo", "diff_fu_ci_hi",
            "gate_pass", "is_gate_factor", "r2_u", "r2_w", "r2_v", "r2_fisher",
            "n_fallback_w", "n_fallback_fisher"]
    with open(out_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in all_rows:
            f.write("\t".join(str(row[c]) for c in cols) + "\n")
    print(f"\nwrote {out_path}")

    print("\n=== gate: (w) beats (u), CI95% paired bootstrap excludes zero ===")
    per_factor_pass_count = {}
    for branch in FACTORS:
        subs_passing = [r["subj"] for r in all_rows if r["branch"] == branch and r["gate_pass"]]
        per_factor_pass_count[branch] = len(subs_passing)
        print(f"  {branch}: PASS on {len(subs_passing)}/{len(args.subj)} subjects "
              f"{subs_passing}")

    factors_passing = sum(1 for branch in FACTORS
                          if per_factor_pass_count[branch] >= 3)
    overall = "PASS" if factors_passing >= 2 else "FAIL"
    print(f"\nfactors with PASS on >=3/4 subjects: {factors_passing}/3 -> gate {overall}")
    if overall == "PASS":
        print("-> next step: weighted fusion end to end on NSD-Imagery")
    else:
        print("-> weighted fusion fails this cheap gate on the dev split; "
              "to be reported as a negative result")


if __name__ == "__main__":
    main()
