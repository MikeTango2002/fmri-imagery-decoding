"""The per-unit distribution of profile_correlation, printed as numbers.

Inputs:  evaluation.db of feature-decoding (--db), which already stores the per-unit array (evaluation.py:133)
Outputs: median and quartiles per layer, and the share of dead units, on screen; nothing is recomputed

evaluation.py reduces profile_correlation with np.nanmean over all units of a layer, but the populations do not
compare across layers: conv1_1 has ~3.2M units against 1000 for fc8, and most conv1_1 units are edge detectors at
positions with almost no variance over the 50 test images. They cannot be decoded and drag the mean down. The
authors' own figure script plots a violin over units, never a mean.

Usage (from feature-decoding/, venv active):
    python ../diag_profile_distribution.py --db <path/to/evaluation.db>
"""

import argparse

import numpy as np
from bdpy.dataform import SQLite3KeyValueStore

# Hierarchical order: evaluation.py iterates over a random permutation; the order
# in the database carries no meaning. This is the only order the profile reads in.
LAYERS = [f"conv{block}_{i}" for block, n in [(1, 2), (2, 2), (3, 4), (4, 4), (5, 4)]
          for i in range(1, n + 1)] + ["fc6", "fc7", "fc8"]
SUBJECTS = ["sub-01", "sub-02", "sub-03"]
ROI = "VC"

DEFAULT_DB = ("data/decoded_features/ImageNetTest/"
              "train_deeprecon_rep5_test_ImageNetTest_fmriprep_pyfastl2lir_alpha100_allunits/"
              "caffe/VGG19/evaluation.db")

# A unit whose true response barely moves across the 50 test images cannot be decoded,
# and its r is noise around 0. With n=50 the null SD of r is ~0.14; this threshold is
# deliberately far below it: the aim is to count dead units, not to test significance.
DEAD_ABS_R = 0.05


def summarize(values):
    """Median, quartiles and dead fraction of one (layer, subject) unit population."""
    flat = np.asarray(values).ravel()
    n_total = flat.size
    nan_frac = float(np.mean(np.isnan(flat))) if n_total else float("nan")
    finite = flat[~np.isnan(flat)]
    if finite.size == 0:
        return None
    dead_frac = float(np.mean(np.abs(finite) < DEAD_ABS_R))
    q1, med, q3 = np.percentile(finite, [25, 50, 75])
    return {
        "n": n_total,
        "mean": float(np.mean(finite)),
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        # The mean restricted to units that carry signal: this is the number that tells
        # whether the low conv layers are decoded badly or merely diluted.
        "mean_alive": float(np.mean(finite[np.abs(finite) >= DEAD_ABS_R])),
        "dead_frac": dead_frac,
        "nan_frac": nan_frac,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help="path to evaluation.db")
    parser.add_argument("--roi", default=ROI)
    args = parser.parse_args()

    db = SQLite3KeyValueStore(args.db)

    print(f"profile_correlation per-unit distribution  (roi={args.roi}, db={args.db})")
    print(f"dead unit = |r| < {DEAD_ABS_R}\n")

    header = (f"{'layer':10s} {'subject':8s} {'units':>10s} {'mean':>8s} {'median':>8s} "
              f"{'q1':>8s} {'q3':>8s} {'mean_alive':>11s} {'dead%':>7s} {'nan%':>7s}")
    print(header)
    print("-" * len(header))

    for layer in LAYERS:
        for subject in SUBJECTS:
            v = db.get(layer=layer, subject=subject, roi=args.roi,
                       metric="profile_correlation")
            # An empty array means the condition exists but was never computed.
            if v is None or np.size(v) == 0:
                print(f"{layer:10s} {subject:8s} {'-':>10s}")
                continue
            s = summarize(v)
            if s is None:
                print(f"{layer:10s} {subject:8s} {'all NaN':>10s}")
                continue
            print(f"{layer:10s} {subject:8s} {s['n']:>10d} {s['mean']:>8.3f} "
                  f"{s['median']:>8.3f} {s['q1']:>8.3f} {s['q3']:>8.3f} "
                  f"{s['mean_alive']:>11.3f} {100 * s['dead_frac']:>6.1f}% "
                  f"{100 * s['nan_frac']:>6.1f}%")
        print()


if __name__ == "__main__":
    main()
