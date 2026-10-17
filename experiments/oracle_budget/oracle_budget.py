#!/usr/bin/env python3
"""The oracle budget: what each pipeline stage costs, as paired differences with bootstrap intervals.

Inputs:  the TSVs of arms T0-T3 under --results (S1K_T0.tsv ... S1K_T3_subj0N.tsv, or {arm}_{combo}.tsv)
Outputs: --out TSV of T0 - T1 (representation + adapter), T1 - T2 (PCA of z_low), T2 - T3 (brain decoder)

`--set s1k`: shared1000, each arm at the setting that won its own sweep; the only reading with T2 - T3,
because T3 needs voxels. `--set proxy`: T0 - T1 and T1 - T2 at one fixed setting (`--combo`, or all nine
with `--all-combos`); the representation is not confounded with the generator constants. The
resampling unit is the block. On the s1k TSVs the `subj` column holds the block index.

Usage:
    python oracle_budget.py --set s1k --out data/results/oracle_budget/budget_shared1000.tsv
    python oracle_budget.py --set proxy --all-combos --out data/results/oracle_budget/budget_proxy_all_settings.tsv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bootstrap_ci_imagery import METRICS, TWO_WAY, LOWER_IS_BETTER, bootstrap

# mean of the four two-way comparisons: the selection criterion of pick_oracle_constants.py
PRIMARY = "primary_mean"
ALL_METRICS = [PRIMARY] + METRICS
# the harness scores a stimulus against 11 distractors; a perfect 2WC is 10/11
HARNESS_CEILING = 10.0 / 11.0
SUBJECTS = ["01", "02", "05", "07"]
COMBOS = [f"s{s}_m{m}" for s in ("0.65", "0.75", "0.85") for m in ("0.2", "0.4", "0.6")]


def load_cells(path: Path, subject: int, stim_type: str):
    """TSV -> {(subject, block, sample): {metric: mean over generation replicas}}.

    The replicas are the same prediction re-diffused with a different seed; they are
    averaged away before anything is resampled (same rule as bootstrap_ci_imagery.py).
    """
    acc = defaultdict(lambda: defaultdict(list))
    with path.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if stim_type != "all" and row["stim_type"] != stim_type:
                continue
            key = (subject, int(row["subj"]), int(row["sample"]))
            for m in METRICS:
                acc[key][m].append(float(row[m]))
    if not acc:
        sys.exit(f"{path}: no rows for stim_type={stim_type}")
    cells = {}
    for k, v in acc.items():
        c = {m: float(np.mean(v[m])) for m in METRICS}
        c[PRIMARY] = float(np.mean([c[m] for m in TWO_WAY]))
        cells[k] = c
    return cells


def broadcast(cells, subjects):
    """Give a subject-free arm one copy per subject; it can be paired with T3."""
    return {(s, b, i): v for s in subjects for (_, b, i), v in cells.items()}


def saturation(cells):
    """Fraction of cells sitting exactly on the harness ceiling, per two-way metric.

    A budget item that comes out unmeasurable has two very different readings, the arms
    are equally good, or the metric cannot see the difference, and this is what separates
    them.
    """
    out = {}
    for m in TWO_WAY:
        v = np.array([c[m] for c in cells.values()])
        out[m] = float(np.mean(np.isclose(v, HARNESS_CEILING, atol=1e-6)))
    return out


def levels(name, cells):
    print(f"  {name:<22} n={len(cells):>5}  " +
          "  ".join(f"{m}={np.mean([c[m] for c in cells.values()]):.4f}"
                    for m in (PRIMARY, "PixCorr", "SSIM")))


def report_item(label, a_cells, b_cells, n_boot, seed, rows, note=""):
    """One budget item: paired A - B on the cells the two arms share."""
    keys = sorted(set(a_cells) & set(b_cells))
    if len(keys) != len(a_cells) or len(keys) != len(b_cells):
        sys.exit(f"{label}: arms do not cover the same cells "
                 f"({len(a_cells)} vs {len(b_cells)}, {len(keys)} shared)")
    rng = np.random.default_rng(seed)
    n_blocks = len({k[1] for k in keys})
    print(f"\n{label}   {len(keys)} cells over {n_blocks} blocks"
          f"{'  |  ' + note if note else ''}")
    for metric in ALL_METRICS:
        va = np.array([a_cells[k][metric] for k in keys])
        vb = np.array([b_cells[k][metric] for k in keys])
        values = va - vb
        pct = metric in TWO_WAY or metric == PRIMARY
        scale, unit = (100.0, "pp") if pct else (1.0, "")
        point = values.mean() * scale
        lo, hi, n_cl = bootstrap(values, keys, "stimulus", n_boot, rng)
        lo_i, hi_i, _ = bootstrap(values, keys, "iid", n_boot, rng)
        lo, hi, lo_i, hi_i = lo * scale, hi * scale, lo_i * scale, hi_i * scale
        verdict = "0 inside" if lo <= 0.0 <= hi else "0 outside"
        flip = "  (lower is better)" if metric in LOWER_IS_BETTER else ""
        print(f"    {metric:<13} {point:+8.3f} {unit:<2}  block CI [{lo:+7.3f},{hi:+7.3f}]"
              f"  iid CI [{lo_i:+7.3f},{hi_i:+7.3f}]  {verdict}{flip}")
        rows.append({"item": label, "metric": metric, "unit": unit or "raw",
                     "point": point, "ci_lo": lo, "ci_hi": hi, "width": hi - lo,
                     "ci_lo_iid": lo_i, "ci_hi_iid": hi_i, "n_cells": len(keys),
                     "n_blocks": n_cl, "excludes_zero": int(not (lo <= 0.0 <= hi)),
                     "note": note})


def run_s1k(args, rows):
    d = Path(args.results)
    t0 = load_cells(d / "S1K_T0.tsv", 0, "all")
    t1 = load_cells(d / "S1K_T1.tsv", 0, "all")
    t2 = load_cells(d / "S1K_T2.tsv", 0, "all")
    t3 = {}
    for s in SUBJECTS:
        t3.update(load_cells(d / f"S1K_T3_subj{s}.tsv", int(s), "all"))
    subs = [int(s) for s in SUBJECTS]

    print("\n=== levels (shared1000, per-arm winning constants) ===")
    for name, c in (("T0 s=0.75 m=0.2", t0), ("T1 s=0.65 m=0.6", t1),
                    ("T2 s=0.85 m=0.6", t2), ("T3 s=0.85 m=0.6", t3)):
        levels(name, c)
    print("\n=== saturation: share of cells exactly at the harness ceiling "
          f"({HARNESS_CEILING:.6f}) ===")
    for name, c in (("T0", t0), ("T1", t1), ("T2", t2), ("T3", t3)):
        sat = saturation(c)
        print(f"  {name:<4} " + "  ".join(f"{m}={sat[m]:6.1%}" for m in TWO_WAY))

    print("\n=== budget, paired ===")
    note = "constants differ between arms"
    report_item("T0-T1  representation+A_psi", t0, t1, args.n_boot, args.seed, rows, note)
    report_item("T1-T2  PCA of z_low", t1, t2, args.n_boot, args.seed, rows, note)
    report_item("T2-T3  brain decoder", broadcast(t2, subs), t3,
                args.n_boot, args.seed, rows, "same constants (s=0.85 m=0.6)")


def run_proxy(args, rows):
    d = Path(args.results) / "setting_sweep"

    if args.all_combos:
        print("\n=== point estimates over the whole grid (proxy, fixed constants) ===")
        print(f"  {'combo':<12} " + "  ".join(f"{s:>26}" for s in ("simple", "complex")))
        print(f"  {'':<12} " + "  ".join(f"{'T0-T1 pp   T1-T2 pp':>26}" for _ in range(2)))
        for combo in COMBOS:
            cells = {}
            line = f"  {combo:<12} "
            for st in ("simple", "complex"):
                for arm in ("T0", "T1", "T2"):
                    cells[arm] = load_cells(d / f"{arm}_{combo}.tsv", 0, st)
                keys = sorted(cells["T0"])
                v = {a: np.array([cells[a][k][PRIMARY] for k in keys]) for a in cells}
                line += f"  {(v['T0'] - v['T1']).mean() * 100:+9.2f} "
                line += f"{(v['T1'] - v['T2']).mean() * 100:+9.2f}      "
            print(line)

    for st in ("simple", "complex"):
        arms = {a: load_cells(d / f"{a}_{args.combo}.tsv", 0, st) for a in ("T0", "T1", "T2")}
        print(f"\n=== levels (proxy {st}, fixed {args.combo}) ===")
        for a in ("T0", "T1", "T2"):
            levels(a, arms[a])
            sat = saturation(arms[a])
            print(f"  {'':22} saturation " +
                  "  ".join(f"{m}={sat[m]:6.1%}" for m in TWO_WAY))
        note = f"proxy {st}, fixed {args.combo}"
        report_item(f"T0-T1  representation+A_psi [{st}]", arms["T0"], arms["T1"],
                    args.n_boot, args.seed, rows, note)
        report_item(f"T1-T2  PCA of z_low [{st}]", arms["T1"], arms["T2"],
                    args.n_boot, args.seed, rows, note)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="which", default="s1k", choices=["s1k", "proxy"])
    ap.add_argument("--results", default="data/results/oracle_budget")
    ap.add_argument("--combo", default="s0.85_m0.6",
                    help="proxy only: the fixed strength/mixing combo all arms are read at")
    ap.add_argument("--all-combos", action="store_true",
                    help="proxy only: point estimates at all nine combos, to check the sign")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []
    (run_s1k if args.which == "s1k" else run_proxy)(args, rows)

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
