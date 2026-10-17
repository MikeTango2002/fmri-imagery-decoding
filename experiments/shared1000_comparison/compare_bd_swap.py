#!/usr/bin/env python3
"""The 2x2 swap with Brain-Diffuser: paired readings of BT, TB and BB against the TT control (Section 8.4).

Inputs:  the per-arm TSVs of TT (oracle_budget/S1K_T3_subj0N.tsv, made after the seed fix), BT, TB, BB; bd_block_index.json
Outputs: --out swap_vs_trisect.tsv and --out-json, on the 65 blocks of the Brain-Diffuser comparison

T is TRISECT, B is Brain-Diffuser; the first letter is the source of the context, the second the source of
z_low. No pass/fail criterion: every arm is reported, and the pattern of signs answers the question.
TB >> TT with BT ~ TT: the deficit is in z_low. BT >> TT with TB ~ TT: it is in the context. BB - TT much
below the 2.55 pp gap of the main comparison: part of that gap is sampler and seed. Helpers come from
compare_trisect_vs_bd_shared1000.py; reads TSVs only.

Usage (after run_bd_swap.sh and run_bd_swap_eval.sh):
    python compare_bd_swap.py --out data/results/shared1000_comparison/swap/swap_vs_trisect.tsv
"""
import argparse
import json
from pathlib import Path

from bootstrap_ci_imagery import LOWER_IS_BETTER, METRICS
from oracle_budget import PRIMARY, SUBJECTS
from compare_trisect_vs_bd_shared1000 import absolute_means, load_cells, report

G6_DIR = "data/results/shared1000_comparison"
S1K_DIR = "data/results/oracle_budget"
I2_DIR = f"{G6_DIR}/swap"
ARMS = ("BT", "TB", "BB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g6-dir", default=G6_DIR)
    ap.add_argument("--s1k-dir", default=S1K_DIR)
    ap.add_argument("--i2-dir", default=I2_DIR)
    ap.add_argument("--index", default=f"{G6_DIR}/bd_block_index.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{I2_DIR}/swap_vs_trisect.tsv")
    ap.add_argument("--out-json", default=f"{I2_DIR}/swap_vs_trisect_summary.json")
    args = ap.parse_args()

    with open(args.index) as f:
        idx = json.load(f)
    blocks = set(idx["blocks_kept"])
    print(f"{len(blocks)} kept blocks (same blocks as the Brain-Diffuser comparison: rank-1 "
          f"{idx['mapping_check']['rank1']}/{idx['mapping_check']['n']})")

    s1k, i2 = Path(args.s1k_dir), Path(args.i2_dir)
    tt = {}
    for sub in SUBJECTS:
        tt.update(load_cells(s1k / f"S1K_T3_subj{sub}.tsv", int(sub), blocks))
    abs_tt = absolute_means(tt, "TT (ridge control)")

    all_rows, summary = [], {"n_blocks_kept": len(blocks), "n_cells": len(tt),
                              "absolute_TT": abs_tt, "arms": {}}
    for arm in ARMS:
        cells = {}
        for sub in SUBJECTS:
            cells.update(load_cells(i2 / f"S1K_{arm}_subj{sub}.tsv", int(sub), blocks))
        abs_arm = absolute_means(cells, f"{arm} (swap)")
        rows = report(f"{arm} - TT, paired on (subj, block, stimulus)", cells, tt,
                      f"{arm}-TT", args.n_boot, args.seed, f"{arm}_minus_TT", [])
        for r in rows:
            r["arm"] = arm
        all_rows += rows
        summary["arms"][arm] = {"absolute": abs_arm,
                                 "primary_mean_delta": next(
                                     r for r in rows if r["metric"] == PRIMARY)}

    print("\n=== summary: primary_mean delta (arm - TT) ===")
    for arm in ARMS:
        r = summary["arms"][arm]["primary_mean_delta"]
        print(f"  {arm}: {r['point']:+.3f} pp [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {out} ({len(all_rows)} rows)")

    with open(args.out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
