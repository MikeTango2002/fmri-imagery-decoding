"""Pre-check of the per-dimension gain in latent space (Section 8.1.5). CPU only, minutes.

Inputs:  ridge weights fit without the development split (ridge_weights_devheld), the NSD-core development split
Outputs: data/results/ridge_input_output/dimension_gain_precheck.tsv (subject x branch x gamma),
         dimension_gain_precheck_choice.json (winning gamma per branch, median over the 4 subjects)

Applies the weights to the development split, estimates r_k = corr(pred_k, gt_k) per dimension, and compares cos_diag
under restandardize_gain() for a_k ~ r_k^gamma, gamma in {0 (control), 1, 2}, plus "z_low off" and, for z_low, de-whitening.
Never run on shared1000 or NSD-Imagery, which measure and do not select. gamma=0 must reproduce
restandardize_from_pred (~1e-6) against the dev predictions on disk for subject 01.

Usage (.venv_targets active):
    python dimension_gain_precheck.py --subj 1 2 5 7
"""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_ridge import average_by_image, load_nsdcore_split  # noqa: E402
from predict_ridge_imagery import load_weights, predict_flat  # noqa: E402
from trisect.nsd import zscore_fmri_train_test as zscore_fmri  # noqa: E402
from trisect.pca import load_pca_basis  # noqa: E402
from trisect.ridge import dim_gain, restandardize_from_pred, restandardize_gain  # noqa: E402

SUBJECTS = (1, 2, 5, 7)
BRANCHES = ("z_sem_img", "z_sem_txt", "z_struct", "z_low")
BRANCH_RESHAPE = {"z_struct": (8, 8, 32)}
GAMMAS = (0, 1, 2)  # 0 is the control (today's restandardize_from_pred, exact identity)

INDEX_DIR = "data/nsdcore_index"
BETAS_DIR = "data/mindeyev2_dataset"
DEVHELD_WEIGHTS_DIR = "data/betas_trisect/ridge_weights_devheld"
DEV_MASK_DIR = "data/targets"
TARGETS_H5 = "data/targets/targets.h5"
ZLOW_PATH = "data/targets/pca_zlow/z_low.npy"
ZLOW_PCA_BASIS = "data/targets/pca_zlow"
DEV_PRED_CHECK_DIR = "data/betas_trisect/nsdcore_predictions_dev"  # sanity check only, subj01
OUT_DIR = "data/results/ridge_input_output"
R_DIR = "data/results/ridge_input_output/dimension_gain_r"  # per-dimension r_k, persisted for the end-to-end gain
                                      # application (predict_ridge_shared1000.py /
                                      # predict_ridge_imagery.py --gain-gamma), this script
                                      # already computes r_k on the dev split; the end-to-end
                                      # scripts reuse it from disk instead of recomputing it
                                      # (which would need the devheld weights + dev split
                                      # loaded again just for this one array).


def load_dev_gt(ids):
    with h5py.File(TARGETS_H5, "r") as f:
        z_sem = f["z_sem"][:].astype(np.float64)
        z_struct = f["z_struct"][:].astype(np.float64)
    z_low = np.load(ZLOW_PATH, mmap_mode="r")
    return {
        "z_sem_img": z_sem[ids, 0], "z_sem_txt": z_sem[ids, 1],
        "z_struct": z_struct[ids], "z_low": np.asarray(z_low[ids], dtype=np.float64),
    }


def flat(a):
    return a.reshape(a.shape[0], -1).astype(np.float64)


def cos_diag(pred, gt, mean):
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    p, g = flat(pred) - mean, flat(gt) - mean
    pn = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
    gn = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum(pn * gn, axis=1)))


def per_dim_corr(pred, gt):
    p, g = flat(pred), flat(gt)
    p = p - p.mean(axis=0, keepdims=True)
    g = g - g.mean(axis=0, keepdims=True)
    num = (p * g).sum(axis=0)
    den = np.sqrt((p ** 2).sum(axis=0) * (g ** 2).sum(axis=0))
    return num / np.maximum(den, 1e-12)


def process_subject(subj, rows):
    sd = f"subj{subj:02d}"
    print(f"\n=== {sd} (dev split, gain pre-check) ===", flush=True)
    betas, img73k, is_shared = load_nsdcore_split(subj, Path(INDEX_DIR), Path(BETAS_DIR))
    dev_mask_73k = np.load(Path(DEV_MASK_DIR) / f"nsdcore_dev_mask_subj{subj:02d}.npy")
    test_mask = dev_mask_73k[img73k]
    assert not np.any(test_mask & is_shared), "dev mask overlaps shared1000"
    train_mask = ~is_shared & ~test_mask
    train_fmri, test_fmri = zscore_fmri(betas, train_mask, test_mask)
    del betas
    test_img73k = img73k[test_mask]

    weights_dir = Path(DEVHELD_WEIGHTS_DIR) / sd
    stats = np.load(weights_dir / "train_target_stats.npz")
    zlow_ev = load_pca_basis(ZLOW_PCA_BASIS, dtype=np.float64)["explained_variance"]
    dewhiten_gain = np.sqrt(zlow_ev)
    dewhiten_gain = dewhiten_gain / dewhiten_gain.mean()

    for branch in BRANCHES:
        W, b = load_weights(weights_dir / f"{branch}_weights.pkl")
        raw = predict_flat(test_fmri, W, b)
        raw_avg, uniq_ids = average_by_image(raw, test_img73k)  # (n_dev_imgs, D_flat)

        gt_all = load_dev_gt(uniq_ids)
        gt_flat = flat(gt_all[branch])
        tgt_mean = stats[f"{branch}_mean"].astype(np.float64)
        tgt_std = stats[f"{branch}_std"].astype(np.float64)
        pred_flat = raw_avg.reshape(raw_avg.shape[0], -1).astype(np.float64)

        r = per_dim_corr(pred_flat, gt_flat)
        r_out_dir = Path(R_DIR) / sd
        r_out_dir.mkdir(parents=True, exist_ok=True)
        np.save(r_out_dir / f"{branch}_r.npy", r.astype(np.float64))

        # restandardize_gain() derives tgt_mean and tgt_std from `train_target.mean(0)/.std(0)`.
        # Feed it a synthetic 2-row array {tgt_mean-tgt_std, tgt_mean+tgt_std} rather than reload
        # the full (27000, D) training pool: the mean of that pair is exactly tgt_mean and its
        # population std (ddof=0) is exactly tgt_std, by construction.
        synth_train = np.stack([tgt_mean - tgt_std, tgt_mean + tgt_std], axis=0)

        # sanity check: gamma=0 must equal today's restandardize() (subject 01 only: the only
        # subject with an existing --split dev prediction on disk to compare against)
        if subj == 1:
            existing_path = Path(DEV_PRED_CHECK_DIR) / sd / f"{branch}_pred.npy"
            if existing_path.exists():
                existing = np.load(existing_path).reshape(raw_avg.shape[0], -1).astype(np.float64)
                check = restandardize_gain(pred_flat, synth_train, dim_gain(r, 0), eps=1e-8)
                max_abs_diff = float(np.max(np.abs(check - existing)))
                flag = "" if max_abs_diff < 1e-3 else "  <-- MISMATCH (provenance of on-disk file" \
                                                       " unverified, see dimension_gain_precheck.py docstring)"
                print(f"  [{branch}] gamma=0 sanity vs on-disk dev pred: "
                     f"max|diff|={max_abs_diff:.3e}{flag}", flush=True)

        variants = [(f"gamma{g}", dim_gain(r, g)) for g in GAMMAS]
        if branch == "z_low":
            g1dw = dim_gain(r, 1) * dewhiten_gain
            variants.append(("gamma1_dewhiten", g1dw / g1dw.mean()))
            variants.append(("dewhiten_only", dewhiten_gain))
            variants.append(("off", np.zeros_like(r)))

        for vname, gain in variants:
            pred_g = restandardize_gain(pred_flat, synth_train, gain, eps=1e-8)
            cd = cos_diag(pred_g.reshape((-1,) + gt_all[branch].shape[1:]), gt_all[branch],
                          tgt_mean.reshape(gt_all[branch].shape[1:]))
            rows.append(dict(subject=sd, branch=branch, variant=vname, n=len(uniq_ids), cos_diag=cd))
            print(f"  [{branch}] {vname:<16} cos_diag={cd:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    rows = []
    for subj in args.subj:
        try:
            process_subject(subj, rows)
        except FileNotFoundError as e:
            print(f"  SKIP subj{subj:02d}: {e} (--exclude-dev refit not done for this subject yet)",
                 flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tsv_path = os.path.join(args.out_dir, "dimension_gain_precheck.tsv")
    cols = ["subject", "branch", "variant", "n", "cos_diag"]
    with open(tsv_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")
    print(f"\nWrote {tsv_path}")

    # pick the two best variants per branch, by mean cos_diag over subjects (gamma>=1 candidates
    # only: gamma=0 is the control, not a candidate to promote to end-to-end)
    best = {}
    for branch in BRANCHES:
        by_variant = {}
        for r in rows:
            if r["branch"] != branch or r["variant"] == "gamma0":
                continue
            by_variant.setdefault(r["variant"], []).append(r["cos_diag"])
        ranked = sorted(by_variant.items(), key=lambda kv: -np.mean(kv[1]))
        best[branch] = [{"variant": v, "mean_cos_diag": float(np.mean(cds)), "n_subj": len(cds)}
                        for v, cds in ranked[:2]]
    json_path = os.path.join(args.out_dir, "dimension_gain_precheck_choice.json")
    with open(json_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Wrote {json_path}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
