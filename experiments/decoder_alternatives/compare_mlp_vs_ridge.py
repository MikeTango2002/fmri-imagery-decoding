#!/usr/bin/env python3
"""Compare the MLP encoder with the ridge, end to end through the harness (Section 8.1.1).

Inputs:  the TSVs of run_mlp_vs_ridge[_imagery_trials].sh, oracle_budget/S1K_T3_subj0N.tsv
Outputs: --out comparison_summary.tsv, and the verdict on stdout

Criterion fixed before the run, on primary_mean (mean of the four two-way metrics), by paired bootstrap
over stimulus clusters: PASS if the vision delta has its CI entirely above 0 and the imagery delta
does not lie entirely below 0. It is measured on final reconstructions, not on a branch score, because a gain on a branch can
vanish end to end. Imagery reads the trial-level pair by default; --imagery-mlp and --imagery-ridge
take another pair (see README).

Usage:
    python compare_mlp_vs_ridge.py --out data/results/decoder_alternatives/mlp/comparison_summary.tsv
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import bootstrap_ci_imagery as bci
from bootstrap_ci_imagery import LOWER_IS_BETTER, METRICS, TWO_WAY, bootstrap as bootstrap_pair
from oracle_budget import PRIMARY, SUBJECTS as S1K_SUBJECTS
from oracle_budget import load_cells as load_cells_s1k

D2_3_DIR = "data/results/decoder_alternatives/mlp"
S1K_DIR = "data/results/oracle_budget"  # S1K_T3_subj0N.tsv, the ridge arm of the oracle budget


def load_cells_imagery(path: Path, stim_type: str = "all"):
    """Output of bootstrap_ci_imagery.load_cells(), with primary_mean added per cell. Exact,
    because the mean over repetitions and the mean over the four two-way metrics commute."""
    cells, _ = bci.load_cells(path, "imagery", stim_type)
    for c in cells.values():
        c[PRIMARY] = float(np.mean([c[m] for m in TWO_WAY]))
    return cells


def paired_report(label, a_cells, b_cells, metric, n_boot, seed):
    """MLP (a) - ridge (b), paired on the cells they share.

    Returns (point, lo, hi, n_clusters, unit). Only the 2WC (and PRIMARY, their mean) are
    fractions and only they are reported in points, bootstrap_ci_imagery.py's convention
    (`pct = metric in TWO_WAY`), which has no PRIMARY row because that column is built here;
    PixCorr/SSIM/EffNet-B/SwAV stay in their own units. EffNet-B and SwAV are DISTANCES; a
    negative delta there means (a) is better, the caller prints the direction, because a table
    of these rows read with the 2WC sign convention says the opposite of what happened.
    """
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
    pct = metric in TWO_WAY or metric == PRIMARY   # PRIMARY is the mean of the four 2WC
    scale = 100.0 if pct else 1.0
    unit = "pp" if pct else "  "
    return point * scale, lo * scale, hi * scale, n_cl, unit


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d2-3-dir", default=D2_3_DIR)
    ap.add_argument("--imagery-mlp", default="mlp_imagery_trial.tsv")
    ap.add_argument("--imagery-ridge", default="ridge_imagery_trial.tsv")
    ap.add_argument("--s1k-dir", default=S1K_DIR)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d2_3 = Path(args.d2_3_dir)
    s1k = Path(args.s1k_dir)

    # --- vision axis: shared1000, MLP vs ridge (oracle_budget/S1K_T3), same constants ---
    print("\n=== vision axis (shared1000, s=0.85 m=0.6) ===")
    mlp_s1k, ridge_s1k = {}, {}
    for sub in S1K_SUBJECTS:
        mlp_s1k.update(load_cells_s1k(d2_3 / f"mlp_s1k_subj{sub}.tsv", int(sub), "all"))
        ridge_s1k.update(load_cells_s1k(s1k / f"S1K_T3_subj{sub}.tsv", int(sub), "all"))

    vision_rows = []
    for metric in [PRIMARY] + METRICS:
        point, lo, hi, n_cl, unit = paired_report("vision", mlp_s1k, ridge_s1k, metric,
                                             args.n_boot, args.seed)
        verdict = "0 inside" if lo <= 0.0 <= hi else "0 outside"
        note = "  (lower is better)" if metric in LOWER_IS_BETTER else ""
        print(f"  {metric:<13} MLP-ridge {point:+7.3f} {unit}  95% CI [{lo:+7.3f}, {hi:+7.3f}]  "
              f"(n_blocks={n_cl})  {verdict}{note}")
        vision_rows.append((metric, point, lo, hi, n_cl, unit))
    vision_primary = vision_rows[0]

    # --- imagery axis: real NSD-Imagery, MLP vs ridge, same constants ---
    print("\n=== imagery axis (real NSD-Imagery, s=0.85 m=0.6) ===")
    print(f"  {args.imagery_mlp} vs {args.imagery_ridge}")
    mlp_img = load_cells_imagery(d2_3 / args.imagery_mlp)
    ridge_img = load_cells_imagery(d2_3 / args.imagery_ridge)

    imagery_rows = []
    for metric in [PRIMARY] + METRICS:
        point, lo, hi, n_cl, unit = paired_report("imagery", mlp_img, ridge_img, metric,
                                             args.n_boot, args.seed)
        verdict = "0 inside" if lo <= 0.0 <= hi else "0 outside"
        note = "  (lower is better)" if metric in LOWER_IS_BETTER else ""
        print(f"  {metric:<13} MLP-ridge {point:+7.3f} {unit}  95% CI [{lo:+7.3f}, {hi:+7.3f}]  "
              f"(n_stimuli={n_cl})  {verdict}{note}")
        imagery_rows.append((metric, point, lo, hi, n_cl, unit))
    imagery_primary = imagery_rows[0]

    # --- verdict ---
    v_point, v_lo, v_hi = vision_primary[1:4]
    i_point, i_lo, i_hi = imagery_primary[1:4]
    beats_vision = v_lo > 0.0
    loses_imagery = i_hi < 0.0
    passed = beats_vision and not loses_imagery

    print("\n=== verdict (primary_mean, 95% stimulus-cluster CI) ===")
    print(f"  vision : MLP-ridge {v_point:+.3f} pp, CI [{v_lo:+.3f}, {v_hi:+.3f}]  "
          f"-> {'beats ridge' if beats_vision else 'does NOT beat ridge'}")
    print(f"  imagery: MLP-ridge {i_point:+.3f} pp, CI [{i_lo:+.3f}, {i_hi:+.3f}]  "
          f"-> {'does not lose' if not loses_imagery else 'LOSES to ridge'}")
    print(f"\n  {'PASS' if passed else 'FAIL'} -- "
          f"{'MLP promoted to phase-2 encoder' if passed else 'ridge stays the encoder'}")

    if args.out:
        import csv
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = ([{"axis": ax, "metric": m, "point": p, "ci_lo": lo, "ci_hi": hi,
                  "n_clusters": n, "unit": u.strip() or "raw",
                  "lower_is_better": int(m in LOWER_IS_BETTER)}
                 for ax, src in (("vision", vision_rows), ("imagery", imagery_rows))
                 for m, p, lo, hi, n, u in src])
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
