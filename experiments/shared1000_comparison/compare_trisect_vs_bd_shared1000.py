#!/usr/bin/env python3
"""TRISECT against Brain-Diffuser on the shared1000 blocks: three pre-specified readings (Sections 6.4, 7.1, 8.4).

Inputs:  the per-arm TSVs of TRISECT and Brain-Diffuser (run_bd_shared1000_blocks.sh), bd_block_index.json
Outputs: --out (paired difference), --out-null (generation floor), --out-json (summary)

(i) TRISECT minus Brain-Diffuser, paired on (subject, block, stimulus) cells; the bootstrap clusters on the block.
(ii) An ordering test of the deficits: delta_s1k <= delta_vision <= delta_imagery on AlexNet(2) and AlexNet(5)
(Section 8.4 says why its verdict is not used). (iii) The generation floor: replicas {0,1} against {2,3} of
the same TRISECT arm; the true delta is zero and the CI is the floor; it is conservative by about sqrt(2),
because each side has 2 replicas instead of 4. Nothing below its half-width is interpretable. Reads TSVs only.

Usage (from the project root):
    python compare_trisect_vs_bd_shared1000.py --out data/results/shared1000_comparison/trisect_vs_bd_S1K_T3.tsv
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bootstrap_ci_imagery import LOWER_IS_BETTER, METRICS, TWO_WAY, bootstrap
from oracle_budget import PRIMARY, SUBJECTS

G6_DIR = "data/results/shared1000_comparison"
S1K_DIR = "data/results"

# TRISECT - Brain-Diffuser on NSD-Imagery (vision: Table 7.3, imagery: raw row of Table 7.5), in
# pp, with 12-stimulus clusters. TRISECT runs at s=0.75 m=0.4 there, a setting that differs from the
# default arm of this script (its effect is at most 1.7 pp). The values come from the final imagery
# arm, with the per-stimulus diffusion seeds (trisect/trisect.tsv against baselines/brain_diffuser.tsv,
# scheme "stimulus", n=12).
F6_DELTA = {"vision": {"AlexNet(2)": -9.75, "AlexNet(5)": -8.18},
            "imagery": {"AlexNet(2)": -2.90, "AlexNet(5)": -6.42}}


def load_cells(path: Path, subject, blocks=None, reps=None):
    """TSV -> {(subject, block, stimulus): {metric: mean over the selected replicas}}."""
    acc = defaultdict(lambda: defaultdict(list))
    with path.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            b = int(row["subj"])
            if blocks is not None and b not in blocks:
                continue
            if reps is not None and int(row["repetition"]) not in reps:
                continue
            key = (subject, b, int(row["sample"]))
            for m in METRICS:
                acc[key][m].append(float(row[m]))
    if not acc:
        sys.exit(f"{path}: no rows left after filtering (blocks={blocks and len(blocks)}, "
                 f"reps={reps})")
    cells = {}
    for k, v in acc.items():
        c = {m: float(np.mean(v[m])) for m in METRICS}
        c[PRIMARY] = float(np.mean([c[m] for m in TWO_WAY]))
        cells[k] = c
    return cells


def paired(a_cells, b_cells, metric, n_boot, seed):
    keys = sorted(set(a_cells) & set(b_cells))
    if not keys:
        sys.exit("no shared cells between the two arms")
    rng = np.random.default_rng(seed)
    values = np.array([a_cells[k][metric] - b_cells[k][metric] for k in keys])
    lo, hi, n_cl = bootstrap(values, keys, "stimulus", n_boot, rng)
    pct = metric in TWO_WAY or metric == PRIMARY
    s = 100.0 if pct else 1.0
    return values.mean() * s, lo * s, hi * s, n_cl, ("pp" if pct else "raw")


def report(title, a_cells, b_cells, label, n_boot, seed, axis, rows):
    print(f"\n=== {title} ===")
    for metric in [PRIMARY] + METRICS:
        pt, lo, hi, n_cl, unit = paired(a_cells, b_cells, metric, n_boot, seed)
        verdict = "0 inside" if lo <= 0.0 <= hi else "0 OUTSIDE"
        note = "  (lower is better)" if metric in LOWER_IS_BETTER else ""
        print(f"  {metric:<13} {label} {pt:+8.3f} {unit:<3} 95% CI [{lo:+8.3f}, {hi:+8.3f}]  "
              f"(n_blocks={n_cl})  {verdict}{note}")
        rows.append({"axis": axis, "metric": metric, "point": pt, "ci_lo": lo, "ci_hi": hi,
                     "n_clusters": n_cl, "unit": unit,
                     "lower_is_better": int(metric in LOWER_IS_BETTER)})
    return rows


def absolute_means(cells, label):
    print(f"\n  absolute means, {label} ({len(cells)} cells):")
    for m in [PRIMARY] + METRICS:
        v = np.mean([c[m] for c in cells.values()])
        print(f"    {m:<13} {v:.4f}")
    return {m: float(np.mean([c[m] for c in cells.values()])) for m in [PRIMARY] + METRICS}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g6-dir", default=G6_DIR)
    ap.add_argument("--s1k-dir", default=S1K_DIR)
    ap.add_argument("--index", default=f"{G6_DIR}/bd_block_index.json")
    ap.add_argument("--bd-prefix", default="bd_s1k_subj")
    ap.add_argument("--trisect-prefix", default="oracle_budget/S1K_T3_subj",
                    help="path of the TRISECT arm under --s1k-dir: <prefix><subj>.tsv. The default "
                         "is the arm at s=0.85 m=0.6 generated before the seed fix; the same "
                         "reading at the matched s=0.75 m=0.4 is trisect/shared1000/S1K_T3_p3_subj, and the "
                         "arm at 0.85/0.6 after the seed fix is shared1000_comparison/S1K_T3_h0_subj")
    ap.add_argument("--trisect-label", default="TRISECT T3 s=0.85 m=0.6",
                    help="how the arm is named in the printed absolute means")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{G6_DIR}/trisect_vs_bd_S1K_T3.tsv")
    ap.add_argument("--out-null", default=f"{G6_DIR}/generation_null_S1K_T3.tsv")
    ap.add_argument("--out-json", default=f"{G6_DIR}/trisect_vs_bd_S1K_T3_summary.json")
    args = ap.parse_args()

    with open(args.index) as f:
        idx = json.load(f)
    blocks = set(idx["blocks_kept"])
    print(f"{len(blocks)} kept blocks, CI widening vs 83 = x{idx['ci_widening_vs_83']:.3f}")
    mc = idx["mapping_check"]
    print(f"mapping check: rank-1 {mc['rank1']}/{mc['n']} in a pool of {mc['pool']} "
          f"(worst rank {mc['worst_rank']}, rho_min={mc['rho_min']:.4f})")

    g6, s1k = Path(args.g6_dir), Path(args.s1k_dir)
    tri, bd, tri_a, tri_b = {}, {}, {}, {}
    for sub in SUBJECTS:
        t = s1k / f"{args.trisect_prefix}{sub}.tsv"
        tri.update(load_cells(t, int(sub), blocks))
        tri_a.update(load_cells(t, int(sub), blocks, reps={0, 1}))
        tri_b.update(load_cells(t, int(sub), blocks, reps={2, 3}))
        bd.update(load_cells(g6 / f"{args.bd_prefix}{sub}.tsv", int(sub), blocks))

    abs_tri = absolute_means(tri, args.trisect_label)
    abs_bd = absolute_means(bd, "Brain-Diffuser (1 sample)")

    rows = []
    report("(i) TRISECT - Brain-Diffuser, shared1000, paired on (subj, block, stimulus)",
           tri, bd, "TRISECT-BD", args.n_boot, args.seed, "s1k_trisect_minus_bd", rows)

    # (ii) the ordering test
    print("\n=== (ii) ordering test: does the deficit GROW inside the training distribution? ===")
    order = {}
    by_metric = {r["metric"]: r for r in rows}
    for m in ("AlexNet(2)", "AlexNet(5)"):
        r = by_metric[m]
        d_s1k, lo, hi = r["point"], r["ci_lo"], r["ci_hi"]
        d_vis, d_img = F6_DELTA["vision"][m], F6_DELTA["imagery"][m]
        excl_neg = hi < 0.0
        confirmed = excl_neg and d_s1k <= d_vis
        refuted = (d_s1k >= d_img) or (lo > 0.0)
        v = "CONFIRMED" if confirmed else ("REFUTED" if refuted else "not orderable")
        print(f"  {m:<12} s1k {d_s1k:+7.3f} [{lo:+7.3f},{hi:+7.3f}]  vs vision {d_vis:+6.2f}  "
              f"vs imagery {d_img:+6.2f}  -> {v}")
        order[m] = {"delta_s1k": d_s1k, "ci": [lo, hi], "delta_vision_F6": d_vis,
                    "delta_imagery_F6": d_img, "verdict": v}

    # (iii) the generation floor, same axis, same arm
    null_rows = []
    report("(iii) generation null: TRISECT replicas {0,1} - {2,3}, same arm (true delta = 0)",
           tri_a, tri_b, "reps01-reps23", args.n_boot, args.seed, "generation_null", null_rows)
    floor = {r["metric"]: max(abs(r["ci_lo"]), abs(r["ci_hi"])) for r in null_rows}
    print("\n  generation floor (largest |CI endpoint| of the null), primary_mean = "
          f"{floor[PRIMARY]:.3f} pp -- nothing smaller than this is interpretable on this axis")

    for path, rs in ((args.out, rows), (args.out_null, null_rows)):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rs)
        print(f"wrote {p} ({len(rs)} rows)")

    with open(args.out_json, "w") as fh:
        json.dump({"n_blocks_kept": len(blocks), "n_cells": len(tri),
                   "ci_widening_vs_83": idx["ci_widening_vs_83"],
                   "mapping_check": idx["mapping_check"],
                   "absolute_trisect": abs_tri, "absolute_brain_diffuser": abs_bd,
                   "ordering_test": order,
                   "generation_floor": {k: float(v) for k, v in floor.items()}}, fh, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
