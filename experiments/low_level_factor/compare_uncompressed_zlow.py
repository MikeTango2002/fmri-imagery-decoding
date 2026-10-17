#!/usr/bin/env python3
"""Compare the uncompressed z_low target (bfull) with the compressed one (d1024refit), on NSD-Imagery.

Inputs:  the TSVs of run_uncompressed_zlow.sh, v0repro_all.tsv, baselines/brain_diffuser.tsv
Outputs: --out comparison_summary.tsv; exit status 0 if the gate passes, 1 if it fails

Criterion fixed before any number was seen (Section 8.1.3): PASS if bfull minus d1024refit has a
paired 95% CI (cluster-on-stimulus bootstrap, n=12, as in bootstrap_ci_imagery.py) that excludes zero
in the positive direction on at least one of the four two-way metrics; FAIL exonerates the low-level
target for good.
Two more rows are reported and do not change the verdict: bfull against Brain-Diffuser and against the
production model (v0repro, with the PCA-compressed z_low).

Usage:
    python compare_uncompressed_zlow.py --out PATH
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import bootstrap_ci_imagery as bci
from bootstrap_ci_imagery import METRICS, TWO_WAY, bootstrap as bootstrap_pair

RESULTS_DIR = "data/results"
BFULL_DEFAULT = f"{RESULTS_DIR}/low_level_factor/uncompressed_zlow/bfull_imagery.tsv"
D1024REFIT_DEFAULT = f"{RESULTS_DIR}/low_level_factor/uncompressed_zlow/d1024refit_imagery.tsv"
V0_DEFAULT = f"{RESULTS_DIR}/low_level_factor/scalar_restandardization/v0repro_all.tsv"
BD_DEFAULT = f"{RESULTS_DIR}/baselines/brain_diffuser.tsv"


def paired_row(label, a_cells, b_cells, metric, n_boot, seed):
    keys = sorted(set(a_cells) & set(b_cells))
    if not keys:
        sys.exit(f"{label}/{metric}: no shared cells between the two arms")
    if len(keys) != len(a_cells) or len(keys) != len(b_cells):
        print(f"  [{label}] warning: cell counts differ ({len(a_cells)} vs {len(b_cells)}), "
              f"using {len(keys)} shared", file=sys.stderr)
    rng = np.random.default_rng(seed)
    va = np.array([a_cells[k][metric] for k in keys])
    vb = np.array([b_cells[k][metric] for k in keys])
    values = va - vb
    lo, hi, n_cl = bootstrap_pair(values, keys, "stimulus", n_boot, rng)
    point = values.mean()
    scale = 100.0 if metric in TWO_WAY else 1.0
    return point * scale, lo * scale, hi * scale, n_cl


def report_pair(section, label_a, label_b, a_path, b_path, n_boot, seed, stim_type):
    print(f"\n=== {section}: {label_a} - {label_b} (imagery, stim_type={stim_type}) ===")
    a_cells, _ = bci.load_cells(Path(a_path), "imagery", stim_type)
    b_cells, _ = bci.load_cells(Path(b_path), "imagery", stim_type)
    rows = []
    for metric in TWO_WAY:
        point, lo, hi, n_cl = paired_row(section, a_cells, b_cells, metric, n_boot, seed)
        excludes_positive = lo > 0.0
        verdict = "EXCLUDES ZERO (positive)" if excludes_positive else (
            "excludes zero (negative)" if hi < 0.0 else "includes zero")
        print(f"  {metric:<13} {point:+7.3f} pp  95% CI [{lo:+7.3f}, {hi:+7.3f}] pp  "
              f"(n_stimuli={n_cl})  {verdict}")
        rows.append({"comparison": section, "metric": metric, "point_pp": point,
                     "ci_lo_pp": lo, "ci_hi_pp": hi, "n_stimuli": n_cl,
                     "excludes_zero_positive": int(excludes_positive)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfull", default=BFULL_DEFAULT)
    ap.add_argument("--d1024refit", default=D1024REFIT_DEFAULT)
    ap.add_argument("--v0", default=V0_DEFAULT, help="set to '' to skip the vs-v0 row")
    ap.add_argument("--bd", default=BD_DEFAULT, help="set to '' to skip the vs-BD row")
    ap.add_argument("--stim-type", default="all", choices=["all", "simple", "complex"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    all_rows = []

    # --- the gate itself: B-full vs the d=1024 branch refit with the SAME alpha procedure ---
    gate_rows = report_pair("GATE (B-full vs d1024-refit)", "B-full", "d1024-refit",
                            args.bfull, args.d1024refit, args.n_boot, args.seed, args.stim_type)
    all_rows += gate_rows
    gate_pass = any(r["excludes_zero_positive"] for r in gate_rows)

    # --- informational row 2: vs Brain-Diffuser (the comparison the whole test is about) ---
    if args.bd and Path(args.bd).exists():
        all_rows += report_pair("vs Brain-Diffuser", "B-full", "Brain-Diffuser",
                                args.bfull, args.bd, args.n_boot, args.seed, args.stim_type)
    elif args.bd:
        print(f"\n(skipping vs-Brain-Diffuser: {args.bd} not found)")

    # --- informational row 3: vs the production model (PCA-compressed z_low) ---
    if args.v0 and Path(args.v0).exists():
        all_rows += report_pair("vs v0", "B-full", "v0",
                                args.bfull, args.v0, args.n_boot, args.seed, args.stim_type)
    elif args.v0:
        print(f"\n(skipping vs-v0: {args.v0} not found)")

    print("\n=== verdict (pre-registered criterion) ===")
    print(f"  {'PASS' if gate_pass else 'FAIL'} -- "
          f"{'the margin is (at least partly) in the PCA compression' if gate_pass else 'z_low exonerated permanently'}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {out} ({len(all_rows)} rows)")

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
