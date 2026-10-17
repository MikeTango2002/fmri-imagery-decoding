#!/usr/bin/env python3
"""Bootstrap confidence intervals on the per-stimulus TSVs of the Kneeland harness.

Inputs:  --a (and --b) from `final_evaluations_imagery.py`, one row per subj, mode, stim_type, sample and
         repetition; --a-chance / --b-chance: the same models run with --roll_gt
Outputs: CIs printed per metric and scheme; --out writes them as a TSV

With 4 subjects and 12 stimuli, how large must a difference be to mean anything (Section 6.4)? The 10
repetitions are replicas of one prediction; they are averaged first. Schemes: `stimulus` (cluster over
the 12 stimuli, the binding one, the number to quote), `subject` (4 clusters, coarse), `iid` (48 cells,
the most optimistic, a lower bound on the width). --b resamples the paired difference. Chance margin mode
subtracts each model's own rolled-ground-truth score first, because a worse model can have a lower chance level.

Usage:
    python bootstrap_ci_imagery.py --a results/trisect/trisect.tsv --label-a "TRISECT" \
        --b results/baselines/brain_diffuser.tsv --label-b "Brain-Diffuser" --mode imagery --out OUT_TSV
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP",
           "EffNet-B", "SwAV"]
# the four two-way comparisons, the columns the victory criterion is written on
TWO_WAY = ["AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP"]
# lower is better on these two; a negative delta is an improvement
LOWER_IS_BETTER = {"EffNet-B", "SwAV"}


def load_cells(path: Path, mode: str, stim_type: str):
    """TSV -> {(subj, sample): {metric: mean over repetitions}}, plus the stim_type map.

    Rows are filtered by mode and (optionally) stim_type before averaging; `--stim-type
    complex` reproduces the stimulus-type cell of Table 4 of the paper rather
    than a subset of the aggregate.
    """
    acc = defaultdict(lambda: defaultdict(list))
    stim_of = {}
    with path.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["mode"] != mode:
                continue
            if stim_type != "all" and row["stim_type"] != stim_type:
                continue
            key = (int(row["subj"]), int(row["sample"]))
            stim_of[key] = row["stim_type"]
            for m in METRICS:
                acc[key][m].append(float(row[m]))
    if not acc:
        sys.exit(f"{path}: no rows for mode={mode} stim_type={stim_type}")
    cells = {k: {m: float(np.mean(v[m])) for m in METRICS} for k, v in acc.items()}
    return cells, stim_of


def as_matrix(cells, keys, metric):
    return np.array([cells[k][metric] for k in keys], dtype=np.float64)


def bootstrap(values, keys, scheme, n_boot, rng):
    """Percentile CI of the mean of `values`, resampling whole clusters.

    `values` and `keys` are aligned; a cluster is a stimulus index, a subject, or a single
    cell depending on `scheme`. Clusters are drawn with replacement and every member of a
    drawn cluster is taken, which is what makes the interval honest about the fact that the
    same 12 stimuli recur in all four subjects.
    """
    if scheme == "iid":
        groups = [[i] for i in range(len(keys))]
    else:
        axis = 1 if scheme == "stimulus" else 0          # keys are (subj, sample)
        buckets = defaultdict(list)
        for i, k in enumerate(keys):
            buckets[k[axis]].append(i)
        groups = list(buckets.values())

    idx = np.arange(len(groups))
    means = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(idx, size=len(groups), replace=True)
        sel = np.concatenate([groups[g] for g in drawn])
        means[b] = values[sel].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), int(len(groups))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="per-stimulus TSV of the first model")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--a-chance", default="", help="rolled-GT TSV of the first model; given, "
                                                   "every cell of --a becomes a margin over it")
    ap.add_argument("--b", default="", help="per-stimulus TSV of the second model; "
                                            "given, the run is a paired A - B comparison")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--b-chance", default="", help="rolled-GT TSV of the second model")
    ap.add_argument("--mode", default="imagery", choices=["imagery", "vision"])
    ap.add_argument("--stim-type", default="all", choices=["all", "simple", "complex"])
    ap.add_argument("--metrics", nargs="+", default=TWO_WAY,
                    help=f"default: the four 2WC. All: {' '.join(METRICS)}")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="optional TSV with one row per metric x scheme")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    def as_margin(cells, chance_path, label):
        """cells -> cells minus the same model's rolled-GT run, cell by cell."""
        ch, _ = load_cells(Path(chance_path), args.mode, args.stim_type)
        missing = [k for k in cells if k not in ch]
        if missing:
            sys.exit(f"{chance_path}: missing {len(missing)} of the cells of {label}, "
                     f"first: {missing[:3]} -- chance run is not on the same pool")
        return {k: {m: cells[k][m] - ch[k][m] for m in METRICS} for k in cells}

    a_cells, _ = load_cells(Path(args.a), args.mode, args.stim_type)
    if args.a_chance:
        a_cells = as_margin(a_cells, args.a_chance, args.label_a)
    keys = sorted(a_cells)

    if args.b:
        b_cells, _ = load_cells(Path(args.b), args.mode, args.stim_type)
        if args.b_chance:
            b_cells = as_margin(b_cells, args.b_chance, args.label_b)
        missing = [k for k in keys if k not in b_cells]
        if missing:
            sys.exit(f"{args.b} is missing {len(missing)} of the cells of {args.a}, "
                     f"first: {missing[:3]} -- the two runs are not on the same pool")
        keys = [k for k in keys if k in b_cells]

    subjects = sorted({k[0] for k in keys})
    stimuli = sorted({k[1] for k in keys})
    marg = " (margin over measured chance)" if (args.a_chance or args.b_chance) else ""
    header = (f"{args.label_a} - {args.label_b}" if args.b else args.label_a) + marg
    print(f"\n{header}  |  mode={args.mode}  stim_type={args.stim_type}  "
          f"{len(subjects)} subjects x {len(stimuli)} stimuli = {len(keys)} cells  "
          f"({args.n_boot} bootstrap draws)\n")

    out_rows = []
    for metric in args.metrics:
        va = as_matrix(a_cells, keys, metric)
        if args.b:
            vb = as_matrix(b_cells, keys, metric)
            values = va - vb
        else:
            values = va
        pct = metric in TWO_WAY                      # 2WC are fractions; report as points
        scale = 100.0 if pct else 1.0
        unit = "pp" if (pct and (args.b or args.a_chance)) else ("%" if pct else "")
        point = values.mean() * scale

        note = " (lower is better)" if metric in LOWER_IS_BETTER else ""
        print(f"  {metric:<12} {point:+8.2f} {unit}{note}")
        for scheme in ("stimulus", "subject", "iid"):
            lo, hi, n_clusters = bootstrap(values, keys, scheme, args.n_boot, rng)
            lo, hi = lo * scale, hi * scale
            crosses = "  0 inside" if lo <= 0.0 <= hi else "  0 outside"
            print(f"      {scheme:<9} (n={n_clusters:>2})  95% CI [{lo:+7.2f}, {hi:+7.2f}] "
                  f"{unit}  width {hi - lo:6.2f}{crosses}")
            out_rows.append({"mode": args.mode, "stim_type": args.stim_type,
                             "metric": metric, "scheme": scheme, "n_clusters": n_clusters,
                             "point": point, "ci_lo": lo, "ci_hi": hi,
                             "width": hi - lo, "unit": unit or "raw"})
        # per-subject means: how much of the spread is between subjects
        per_subj = []
        for s in subjects:
            sel = [i for i, k in enumerate(keys) if k[0] == s]
            per_subj.append(f"subj{s:02d} {values[sel].mean() * scale:+.2f}")
        print(f"      per-subject: {'  '.join(per_subj)}\n")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(out_rows)
        print(f"wrote {out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
