"""Compare a hybrid arm (or its control) with the ridge baseline, on vision and on imagery (Section 8.1.1).

Inputs:  the TSVs of run_hybrid_vs_ridge.sh, oracle_budget/S1K_T3_subj0N.tsv, mlp/ridge_imagery_trial.tsv
Outputs: --out comparison_summary[_ctrl].tsv

Same bootstrap as compare_mlp_vs_ridge.py. The criterion is per metric and is decided on vision alone,
before imagery is looked at: the hybrid must keep the semantic gain of the MLP (InceptionV3 and CLIP,
CI entirely above 0) and erase its low-level penalty (AlexNet(2) and SSIM, CI no longer entirely
below 0). Vision is printed first. --arm ctrl swaps the roles (z_low from the MLP, the rest from the
ridge) to check that the attribution is the real effect, not a fluke of one recombination.

Usage:
    python compare_hybrid_vs_ridge.py --out data/results/decoder_alternatives/hybrid/comparison_summary.tsv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import bootstrap_ci_imagery as bci
from bootstrap_ci_imagery import LOWER_IS_BETTER, METRICS, TWO_WAY, bootstrap as bootstrap_pair
from oracle_budget import PRIMARY, SUBJECTS as S1K_SUBJECTS
from oracle_budget import load_cells as load_cells_s1k

G4_0_DIR = "data/results/decoder_alternatives/hybrid"
S1K_DIR = "data/results/oracle_budget"
D2_3_DIR = "data/results/decoder_alternatives/mlp"

# Vision selection criterion: metric -> which side of zero the CI must be on for the hybrid to
# win that metric. Only these four are checked; the rest are reported.
SELECTION_METRICS = {
    "InceptionV3": "positive",   # semantic gain must survive
    "CLIP": "positive",
    "AlexNet(2)": "not_negative",  # low-level penalty must be erased
    "SSIM": "not_negative",
}


def load_cells_imagery(path: Path, stim_type: str = "all"):
    cells, _ = bci.load_cells(path, "imagery", stim_type)
    for c in cells.values():
        c[PRIMARY] = float(np.mean([c[m] for m in TWO_WAY]))
    return cells


def paired_report(label, a_cells, b_cells, metric, n_boot, seed):
    keys = sorted(set(a_cells) & set(b_cells))
    if not keys:
        sys.exit(f"{label}: no shared cells between the two arms")
    if len(keys) != len(a_cells) or len(keys) != len(b_cells):
        print(f"  [{label}] warning: arms cover different cell counts "
              f"({len(a_cells)} vs {len(b_cells)}), using the {len(keys)} shared", file=sys.stderr)
    rng = np.random.default_rng(seed)
    va = np.array([a_cells[k][metric] for k in keys])
    vb = np.array([b_cells[k][metric] for k in keys])
    values = va - vb
    lo, hi, n_cl = bootstrap_pair(values, keys, "stimulus", n_boot, rng)
    point = values.mean()
    pct = metric in TWO_WAY or metric == PRIMARY
    scale = 100.0 if pct else 1.0
    unit = "pp" if pct else "  "
    return point * scale, lo * scale, hi * scale, n_cl, unit


def run_axis(axis_label, arm_cells, base_cells, n_boot, seed, cluster_label):
    rows = []
    for metric in [PRIMARY] + METRICS:
        point, lo, hi, n_cl, unit = paired_report(axis_label, arm_cells, base_cells, metric,
                                                    n_boot, seed)
        verdict = "0 inside" if lo <= 0.0 <= hi else "0 outside"
        note = "  (lower is better)" if metric in LOWER_IS_BETTER else ""
        print(f"  {metric:<13} arm-base {point:+7.3f} {unit}  95% CI [{lo:+7.3f}, {hi:+7.3f}]  "
              f"(n_{cluster_label}={n_cl})  {verdict}{note}")
        rows.append((metric, point, lo, hi, n_cl, unit))
    return rows


def selection_verdict(vision_rows):
    by_metric = {m: (lo, hi) for m, _, lo, hi, _, _ in vision_rows}
    results = {}
    for metric, side in SELECTION_METRICS.items():
        lo, hi = by_metric[metric]
        if side == "positive":
            results[metric] = lo > 0.0
        else:  # not_negative: CI must NOT lie entirely below zero
            results[metric] = not (hi < 0.0)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["hybrid", "ctrl"], default="hybrid",
                     help="hybrid: sem/struct=MLP, low=ridge (the primary arm). ctrl: "
                          "sem/struct=ridge, low=MLP (vision-only symmetric control).")
    ap.add_argument("--g4-0-dir", default=G4_0_DIR)
    ap.add_argument("--s1k-dir", default=S1K_DIR)
    ap.add_argument("--imagery-ridge", default=f"{D2_3_DIR}/ridge_imagery_trial.tsv")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    g4_0 = Path(args.g4_0_dir)
    s1k = Path(args.s1k_dir)
    tag = "hybrid_s1k" if args.arm == "hybrid" else "ctrl_s1k"

    print(f"\n=== hybrid vs ridge [{args.arm}] vision axis (shared1000, s=0.85 m=0.6) ===")
    arm_s1k, ridge_s1k = {}, {}
    for sub in S1K_SUBJECTS:
        arm_s1k.update(load_cells_s1k(g4_0 / f"{tag}_subj{sub}.tsv", int(sub), "all"))
        ridge_s1k.update(load_cells_s1k(s1k / f"S1K_T3_subj{sub}.tsv", int(sub), "all"))
    vision_rows = run_axis("vision", arm_s1k, ridge_s1k, args.n_boot, args.seed, "blocks")

    print(f"\n=== hybrid vs ridge [{args.arm}] selection criterion (vision only, decided before imagery) ===")
    sel = selection_verdict(vision_rows)
    for metric, side in SELECTION_METRICS.items():
        want = "CI > 0" if side == "positive" else "CI not entirely < 0"
        print(f"  {metric:<13} need {want:<22} -> {'OK' if sel[metric] else 'FAIL'}")
    vision_selected = all(sel.values())
    print(f"\n  vision selection: {'PASS' if vision_selected else 'FAIL'} "
          f"({'attribution confirmed, look at imagery' if vision_selected else 'attribution by branch is wrong'})")

    imagery_rows = None
    if args.arm == "hybrid":
        print(f"\n=== hybrid vs ridge [hybrid] imagery axis (real NSD-Imagery, s=0.85 m=0.6, "
              "REPORTED, not selected on) ===")
        arm_img = load_cells_imagery(g4_0 / "hybrid_imagery.tsv")
        ridge_img = load_cells_imagery(Path(args.imagery_ridge))
        imagery_rows = run_axis("imagery", arm_img, ridge_img, args.n_boot, args.seed, "stimuli")
        i_point, i_lo, i_hi = imagery_rows[0][1:4]
        beats_ridge = i_lo > 0.0
        print(f"\n  imagery: hybrid-ridge {i_point:+.3f} pp, CI [{i_lo:+.3f}, {i_hi:+.3f}]  -> "
              f"{'beats ridge (encoder changes)' if beats_ridge else 'does not beat ridge (v0 stands)'}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        srcs = [("vision", vision_rows)] + ([("imagery", imagery_rows)] if imagery_rows else [])
        rows = [{"axis": ax, "metric": m, "point": p, "ci_lo": lo, "ci_hi": hi,
                 "n_clusters": n, "unit": u.strip() or "raw",
                 "lower_is_better": int(m in LOWER_IS_BETTER)}
                for ax, src in srcs for m, p, lo, hi, n, u in src]
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
