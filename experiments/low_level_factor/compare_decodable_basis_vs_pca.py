#!/usr/bin/env python3
"""Decodable low-level basis (k=64) against the ridge arm with the image PCA (Section 8.1.3).

Inputs:  S1K_I3K64_subj*.tsv (run_decodable_basis_eval.sh), oracle_budget/S1K_T3_subj*.tsv,
         shared1000_comparison/bd_block_index.json
Outputs: --out decodable_basis_vs_pca.tsv, --out-json decodable_basis_vs_pca_summary.json

Paired reading on the 65 blocks of the Brain-Diffuser comparison; load_cells, report and absolute_means
come unchanged from compare_trisect_vs_bd_shared1000.py. Criterion fixed before the run: zero outside
the CI and a positive point estimate larger than the 0.399 pp floor (the CI floor that
bootstrap_ci_imagery.py measured in a null comparison). The "H0" keys of the summary are the name of the
control arm in the output files. Reads TSVs only: no model, no GPU.

Usage:
    python compare_decodable_basis_vs_pca.py [--out PATH]
"""
import argparse
import csv
import json
from pathlib import Path

from oracle_budget import PRIMARY, SUBJECTS
from compare_trisect_vs_bd_shared1000 import absolute_means, load_cells, report

G6_DIR = "data/results"  # root of the results; the block index is in shared1000_comparison/
S1K_DIR = "data/results/oracle_budget"
I3_DIR = f"{G6_DIR}/low_level_factor/decodable_basis"
FLOOR_PP = 0.399


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g6-dir", default=G6_DIR)
    ap.add_argument("--s1k-dir", default=S1K_DIR)
    ap.add_argument("--i3-dir", default=I3_DIR)
    ap.add_argument("--index", default=f"{G6_DIR}/shared1000_comparison/bd_block_index.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{I3_DIR}/decodable_basis_vs_pca.tsv")
    ap.add_argument("--out-json", default=f"{I3_DIR}/decodable_basis_vs_pca_summary.json")
    args = ap.parse_args()

    with open(args.index) as f:
        idx = json.load(f)
    blocks = set(idx["blocks_kept"])
    print(f"{len(blocks)} kept blocks (same blocks as the Brain-Diffuser comparison: rank-1 "
          f"{idx['mapping_check']['rank1']}/{idx['mapping_check']['n']})")

    s1k, i3 = Path(args.s1k_dir), Path(args.i3_dir)
    h0 = {}
    for sub in SUBJECTS:
        h0.update(load_cells(s1k / f"S1K_T3_subj{sub}.tsv", int(sub), blocks))
    abs_h0 = absolute_means(h0, "ridge control (oracle_budget/S1K_T3)")

    cells = {}
    for sub in SUBJECTS:
        cells.update(load_cells(i3 / f"S1K_I3K64_subj{sub}.tsv", int(sub), blocks))
    abs_i3 = absolute_means(cells, "I3K64")
    rows = report("I3K64 - ridge control, paired on (subj, block, stimulus)", cells, h0,
                  "I3K64-H0", args.n_boot, args.seed, "I3K64_minus_H0", [])

    delta = next(r for r in rows if r["metric"] == PRIMARY)
    ci_lo, ci_hi, point = delta["ci_lo"], delta["ci_hi"], delta["point"]
    zero_excluded = bool(ci_lo > 0)
    above_floor = bool(abs(point) > FLOOR_PP)
    passed = zero_excluded and above_floor and point > 0
    verdict = "PASS" if passed else "FAIL"
    print(f"\n=== summary: primary_mean delta (I3K64 - ridge control) ===")
    print(f"  {point:+.3f} pp [{ci_lo:+.3f}, {ci_hi:+.3f}] "
          f"(floor {FLOOR_PP:+.3f} pp) -> zero_excluded={zero_excluded} "
          f"above_floor={above_floor} -> {verdict}")

    summary = {"n_blocks_kept": len(blocks), "n_cells": len(h0),
               "absolute_H0": abs_h0, "absolute_I3K64": abs_i3,
               "primary_mean_delta": delta, "floor_pp": FLOOR_PP,
               "zero_excluded": zero_excluded, "above_floor": above_floor,
               "verdict": verdict}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")

    with open(args.out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
