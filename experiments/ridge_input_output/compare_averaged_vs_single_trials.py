#!/usr/bin/env python3
"""Ridge fitted on image means versus ridge fitted on single trials, with a criterion fixed before the run.

Inputs:  the ridge weights of both arms (run_averaged_vs_single_trials.sh), shared1000 targets
Outputs: --out TSV: two-way identification and cos_diag per subject, branch and arm, with paired bootstrap CIs

PASS: the paired 95% CI of (two_way_avg - two_way_ctrl) on z_low excludes zero upward on >= 3 of the 4
subjects. FAIL: it excludes zero downward on >= 1 subject, or includes zero on >= 2. Otherwise inconclusive.
The bootstrap is over the 1000 shared1000 images ("iid": no stimulus blocks here). cos_diag is reported too,
because the two-way identification saturates on the semantic branches (0.95-0.99).

Usage (CPU only, after run_averaged_vs_single_trials.sh):
    python src/compare_averaged_vs_single_trials.py --out data/results/ridge_input_output/averaged_vs_single_trials.tsv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_ridge import (  # noqa: E402
    BETAS_DIR, INDEX_DIR, TARGETS_H5, ZLOW_PATH,
    load_branch_targets, two_way_per_image,
)
from latent_prediction_error import stats_for  # noqa: E402
import bootstrap_ci_imagery as bci  # noqa: E402

SUBJECTS = (1, 2, 5, 7)
BRANCHES = ["z_sem_img", "z_sem_txt", "z_struct", "z_low"]

# Asymmetry on purpose: only z_low is refit at the wide grid. The control values of the three other branches
# come from the final ridge_weights/ (default grid, single trial, no group-cv): their alpha (3.2e4-1.0e5) is well
# inside the default grid (1e1-1e6); a wide grid could not move it. Report it if the alpha of a semantic branch
# ever sits near the edge of the default grid. Their cos_diag comes from latent_prediction_error.tsv (the s1k rows),
# because no fresh predictions of the control arm are kept, only summary.json.
CONTROL_WIDE_DIR = "data/betas_trisect/ridge_weights_wide"     # z_low only, single-trial, group-cv, wide grid
CONTROL_PROD_DIR = "data/betas_trisect/ridge_weights"          # final fit, all 4 branches, default grid
TREATMENT_DIR = "data/betas_trisect/ridge_weights_avg"         # all 4 branches, averaged input, wide grid
PROD_LATENT_TSV = "data/results/latent_error/latent_prediction_error.tsv"  # output of latent_prediction_error.py, s1k rows


def uniq_shared1000_ids(subj, index_dir):
    idx = np.load(Path(index_dir) / f"subj{subj:02d}_trial73k_index.npz")
    img73k, is_shared = idx["img73k"], idx["is_shared"]
    return np.unique(img73k[is_shared])  # sorted, matches average_by_image()'s own sort


def load_pred(out_dir, subj, branch):
    p = Path(out_dir) / f"subj{subj:02d}" / f"pred_shared1000_{branch}.npy"
    return np.load(p) if p.exists() else None


def load_summary(out_dir, subj):
    with open(Path(out_dir) / f"subj{subj:02d}" / "summary.json") as f:
        return json.load(f)


def load_mean(out_dir, subj, branch):
    stats = np.load(Path(out_dir) / f"subj{subj:02d}" / "train_target_stats.npz")
    return stats[f"{branch}_mean"].astype(np.float64)


def load_prod_latent_cos_diag(path):
    """{(subj, branch): cos_diag} from the s1k rows of latent_prediction_error.tsv, if present."""
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["domain"] != "s1k":
                continue
            out[(int(row["subject"].replace("subj", "")), row["branch"])] = float(row["cos_diag"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control-wide-dir", default=CONTROL_WIDE_DIR)
    ap.add_argument("--control-prod-dir", default=CONTROL_PROD_DIR)
    ap.add_argument("--treatment-dir", default=TREATMENT_DIR)
    ap.add_argument("--prod-latent-tsv", default=PROD_LATENT_TSV)
    ap.add_argument("--targets", default=TARGETS_H5)
    ap.add_argument("--zlow", default=ZLOW_PATH)
    ap.add_argument("--index-dir", default=INDEX_DIR)
    ap.add_argument("--betas-dir", default=BETAS_DIR)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/results/ridge_input_output/averaged_vs_single_trials.tsv")
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS),
                     help="subset of subjects to run (default: all 4 -- the pre-registered "
                          "verdict needs all 4; a subset is for smoke-testing this script only)")
    args = ap.parse_args()
    subjects = tuple(args.subj)

    branches_full = load_branch_targets(args.targets, args.zlow)
    prod_cos_diag = load_prod_latent_cos_diag(args.prod_latent_tsv)
    if not prod_cos_diag:
        print(f"(note: {args.prod_latent_tsv} not found -- control-arm cos_diag for the three "
              f"semantic branches will be blank; run latent_prediction_error.py first, "
              f"or ignore if only the z_low gate is needed)", file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    rows = []
    per_subj = {}

    for subj in subjects:
        uniq_ids = uniq_shared1000_ids(subj, args.index_dir)
        n_img = len(uniq_ids)

        # --- primary criterion: z_low, per-image paired 2WC, avg arm minus wide-grid control ---
        pred_ctrl = load_pred(args.control_wide_dir, subj, "z_low")
        pred_trt = load_pred(args.treatment_dir, subj, "z_low")
        if pred_ctrl is None or pred_trt is None:
            sys.exit(f"subj{subj:02d}: missing z_low predictions in "
                      f"{args.control_wide_dir} / {args.treatment_dir} -- run run_averaged_vs_single_trials.sh first")
        gt_zlow_full, _ = branches_full["z_low"]
        gt_zlow = gt_zlow_full[uniq_ids].reshape(n_img, -1)
        vec_ctrl = two_way_per_image(gt_zlow, pred_ctrl.reshape(n_img, -1))
        vec_trt = two_way_per_image(gt_zlow, pred_trt.reshape(n_img, -1))
        diff = vec_trt - vec_ctrl
        keys = list(range(n_img))
        lo, hi, n_cl = bci.bootstrap(diff, keys, "iid", args.n_boot, rng)
        point = float(diff.mean())
        excludes_pos, excludes_neg = lo > 0.0, hi < 0.0
        per_subj[subj] = dict(point=point, ci_lo=lo, ci_hi=hi,
                               excludes_pos=excludes_pos, excludes_neg=excludes_neg)
        rows.append(dict(subject=f"subj{subj:02d}", branch="z_low", arm="gate_diff",
                          alpha="", two_way=point, ci_lo=lo, ci_hi=hi, cos_diag="",
                          n_images=n_cl))

        # --- secondary: alpha, 2WC, cos_diag per branch, per arm (all four branches) ---
        summ_ctrl_wide = load_summary(args.control_wide_dir, subj)   # z_low only
        summ_ctrl_prod = load_summary(args.control_prod_dir, subj)   # all 4, default grid
        summ_trt = load_summary(args.treatment_dir, subj)            # all 4, avg + wide grid

        for branch in BRANCHES:
            gt_arr = branches_full[branch][0][uniq_ids]

            # control arm
            if branch == "z_low":
                alpha_c = summ_ctrl_wide[branch]["alpha"]
                twc_c = summ_ctrl_wide[branch]["two_way_shared1000"]
                mean_c = load_mean(args.control_wide_dir, subj, branch)
                cos_c = stats_for(pred_ctrl, gt_arr, mean_c)["cos_diag"]
            else:
                alpha_c = summ_ctrl_prod[branch]["alpha"]
                twc_c = summ_ctrl_prod[branch]["two_way_shared1000"]
                cos_c = prod_cos_diag.get((subj, branch), "")
            rows.append(dict(subject=f"subj{subj:02d}", branch=branch, arm="control",
                              alpha=alpha_c, two_way=twc_c, ci_lo="", ci_hi="",
                              cos_diag=cos_c, n_images=n_img))

            # treatment arm
            alpha_t = summ_trt[branch]["alpha"]
            twc_t = summ_trt[branch]["two_way_shared1000"]
            pred_t = load_pred(args.treatment_dir, subj, branch)
            cos_t = ""
            if pred_t is not None:
                mean_t = load_mean(args.treatment_dir, subj, branch)
                cos_t = stats_for(pred_t, gt_arr, mean_t)["cos_diag"]
            rows.append(dict(subject=f"subj{subj:02d}", branch=branch, arm="treatment",
                              alpha=alpha_t, two_way=twc_t, ci_lo="", ci_hi="",
                              cos_diag=cos_t, n_images=n_img))

    # --- verdict, pre-registered rule ---
    n_pos = sum(1 for v in per_subj.values() if v["excludes_pos"])
    n_neg = sum(1 for v in per_subj.values() if v["excludes_neg"])
    n_zero = len(per_subj) - n_pos - n_neg
    if n_neg >= 1 or n_zero >= 2:
        verdict = "FAIL"
    elif n_pos >= 3:
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"

    print("=== averaged-reps ridge (wide grid) vs single-trial ridge (wide grid, group-cv), "
          "z_low ===")
    for subj, v in per_subj.items():
        tag = ("EXCLUDES ZERO (positive)" if v["excludes_pos"] else
               "excludes zero (negative)" if v["excludes_neg"] else "includes zero")
        print(f"  subj{subj:02d}: delta_two_way={v['point']:+.4f}  "
              f"95% CI [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]  {tag}")
    print(f"\nVerdict: {verdict}  (positive={n_pos}, negative={n_neg}, "
          f"zero-inclusive={n_zero}, N={len(per_subj)} subjects)")

    print(f"\n{'subject':<9}{'branch':<12}{'arm':<11}{'alpha':>10}{'two_way':>10}{'cos_diag':>10}")
    for r in rows:
        if r["arm"] == "gate_diff":
            continue
        alpha_s = f"{r['alpha']:.3g}" if isinstance(r["alpha"], float) else str(r["alpha"])
        cos_s = f"{r['cos_diag']:.4f}" if isinstance(r["cos_diag"], float) else str(r["cos_diag"])
        print(f"{r['subject']:<9}{r['branch']:<12}{r['arm']:<11}{alpha_s:>10}"
              f"{r['two_way']:>10.4f}{cos_s:>10}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject", "branch", "arm", "alpha", "two_way", "ci_lo", "ci_hi",
                  "cos_diag", "n_images"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")

    summary_path = out.with_name(out.stem + "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"verdict": verdict,
                   "per_subject": {f"subj{s:02d}": {k: v[k] for k in
                                                      ("point", "ci_lo", "ci_hi")}
                                    for s, v in per_subj.items()}},
                   f, indent=2)
    print(f"wrote {summary_path}")

    sys.exit(0 if verdict == "PASS" else (1 if verdict == "FAIL" else 2))


if __name__ == "__main__":
    main()
