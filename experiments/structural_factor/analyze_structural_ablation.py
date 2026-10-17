"""Paired bootstrap CIs of the structural-factor ablation arms against the complete-code baseline (Table 8.4).

Inputs:  --baseline (T2 arm of the oracle budget), --zero and --shuffle (the `--ablate-factor` arms of reconstruct.py)
Outputs: --out TSV: per arm and stimulus type, the paired difference with its CI

The primary score (mean of the four two-way metrics) is read on proxy-simple and proxy-complex separately: they moved
in opposite directions on every earlier budget item. The "iid" scheme (150 (block, local index) cells, each its own
cluster) is the one read; "stimulus" and "subject" carry a caveat: "subj" is the block here, and they group unrelated images.

Usage (CPU, evaluation environment):
    python analyze_structural_ablation.py --baseline data/results/oracle_budget/T2.tsv \\
        --zero data/results/structural_factor/T2_z_struct_zero.tsv \\
        --shuffle data/results/structural_factor/T2_z_struct_shuffle.tsv \\
        --out data/results/structural_factor/ablation_bootstrap_ci.tsv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from bootstrap_ci_imagery import TWO_WAY, bootstrap, load_cells

SECONDARY = ["PixCorr", "SSIM"]


# load_cells(), bootstrap() and TWO_WAY are those of bootstrap_ci_imagery.py; the evaluation chain is not rewritten. The
# METRICS list of load_cells() holds the 8 raw harness columns and has no primary_mean; it is computed here from the same
# per-cell values instead of editing that file.
def primary_mean_matrix(cells, keys):
    return np.array([np.mean([cells[k][m] for m in TWO_WAY]) for k in keys])


def metric_matrix(cells, keys, metric):
    return np.array([cells[k][metric] for k in keys])


def compare(baseline_path, arm_path, arm_label, stim_type, n_boot, rng, out_rows):
    base_cells, _ = load_cells(baseline_path, "imagery", stim_type)
    arm_cells, _ = load_cells(arm_path, "imagery", stim_type)
    keys = sorted(set(base_cells) & set(arm_cells))
    missing = set(base_cells) ^ set(arm_cells)
    if missing:
        raise SystemExit(f"{arm_label}/{stim_type}: baseline and arm don't share the same "
                          f"cells ({len(missing)} mismatched) -- not the same proxy pool")

    metrics = {"primary_mean": (primary_mean_matrix(base_cells, keys),
                                 primary_mean_matrix(arm_cells, keys))}
    for m in SECONDARY:
        metrics[m] = (metric_matrix(base_cells, keys, m), metric_matrix(arm_cells, keys, m))

    print(f"\n=== {arm_label} - T2, stim_type={stim_type}, n={len(keys)} cells ===")
    for metric, (vb, va) in metrics.items():
        delta = va - vb
        pct = metric == "primary_mean" or metric in TWO_WAY
        scale = 100.0 if pct else 1.0
        unit = "pp" if pct else "raw"
        point = delta.mean() * scale
        print(f"  {metric:<12} {point:+8.4f} {unit}")
        # Proxy set, unlike real NSD-Imagery data: evaluate_block_set.py writes "subj" as the block index (0-24) and "sample"
        # as the local index in its condition (0-5), but every (block, local index) pair is its own image (25 x 6 = 150 cells per
        # stimulus type), not one stimulus repeated across blocks. The "stimulus" scheme clusters by "sample" and would group
        # unrelated images that share a local index. "iid" makes each cell its own cluster (n=150) and is the one read;
        # "stimulus" and "subject" are reported for transparency, with this caveat.
        for scheme in ("stimulus", "subject", "iid"):
            lo, hi, n_clusters = bootstrap(delta, keys, scheme, n_boot, rng)
            lo, hi = lo * scale, hi * scale
            crosses = "0 inside" if lo <= 0.0 <= hi else "0 outside"
            note = "  <- pre-registered read (n=150)" if scheme == "iid" else ""
            print(f"      {scheme:<9} (n={n_clusters:>3}) 95% CI [{lo:+8.4f}, {hi:+8.4f}]  "
                  f"{crosses}{note}")
            out_rows.append({"arm": arm_label, "stim_type": stim_type, "metric": metric,
                              "scheme": scheme, "n_clusters": n_clusters, "point": point,
                              "ci_lo": lo, "ci_hi": hi, "unit": unit})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="oracle_budget/T2.tsv, not re-run")
    ap.add_argument("--zero", required=True, help="structural_factor/T2_z_struct_zero.tsv")
    ap.add_argument("--shuffle", required=True, help="structural_factor/T2_z_struct_shuffle.tsv")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_rows = []
    for arm_label, arm_path in [("z_struct_zero", args.zero), ("z_struct_shuffle", args.shuffle)]:
        for stim_type in ("simple", "complex"):
            compare(Path(args.baseline), Path(arm_path), arm_label, stim_type,
                    args.n_boot, rng, out_rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
