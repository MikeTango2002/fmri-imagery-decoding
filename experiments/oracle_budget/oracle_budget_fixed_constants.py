#!/usr/bin/env python3
"""Brain-stage cost (T2 - T3) on shared1000 at matched constants: strength 0.75, mixing 0.4.

Inputs:  data/results/oracle_budget/S1K_T2_p4b.tsv and trisect/shared1000/S1K_T3_p3_subj0N.tsv
Outputs: --out TSV with the paired T2 - T3 difference and its bootstrap interval

oracle_budget.py `--set s1k` reads each arm at its own best setting. The two upper items do not move with
the operating point (the proxy reading shows it); only T2 - T3 needs this check, at the constants the
final model reconstructs with (those of Brain-Diffuser). Cell loader and bootstrap are those of
oracle_budget.py.

Usage:
    python oracle_budget_fixed_constants.py --out data/results/oracle_budget/budget_shared1000_T2_T3_matched_setting.tsv
"""

import argparse
import csv
from pathlib import Path

from oracle_budget import broadcast, levels, load_cells, report_item

SUBJECTS = ["01", "02", "05", "07"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="data/results")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = Path(args.results)
    t2 = load_cells(d / "oracle_budget/S1K_T2_p4b.tsv", 0, "all")
    t3 = {}
    for s in SUBJECTS:
        t3.update(load_cells(d / f"trisect/shared1000/S1K_T3_p3_subj{s}.tsv", int(s), "all"))
    subs = [int(s) for s in SUBJECTS]

    print("\n=== levels (shared1000, matched constants s=0.75 m=0.4) ===")
    levels("T2 s=0.75 m=0.4", t2)
    levels("T3 s=0.75 m=0.4", t3)

    rows = []
    report_item("T2-T3  brain decoder (matched, s=0.75 m=0.4)", broadcast(t2, subs), t3,
                args.n_boot, args.seed, rows, "matched constants, paper operating point")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
