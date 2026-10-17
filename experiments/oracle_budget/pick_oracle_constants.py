"""Pick the best (strength, mixing) setting of an oracle arm from the per-setting TSVs of its sweep.

Inputs:  --sweep-dir with one TSV per setting (written by evaluate_block_set.py)
Outputs: --out, the arm TSV that oracle_budget.py reads; --winner-json, the mean of every setting of the grid

Criterion, fixed before the sweep was read: the setting with the highest mean of the four two-way
identification metrics (AlexNet(2), AlexNet(5), InceptionV3, CLIP), the primary metrics of Kneeland et al.
The winner's rows are copied unchanged except for the `model` column, relabelled with `--arm` (T0, T1, T2,
T1nol2...). The whole grid goes in the JSON: the sweep is part of the result. No GPU.

Usage:
    python pick_oracle_constants.py --sweep-dir data/results/oracle_budget/setting_sweep \\
        --out data/results/oracle_budget/T0.tsv --winner-json data/results/oracle_budget/T0_setting_choice.json
"""
import argparse
import csv
import json
import re
from pathlib import Path

PRIMARY_METRICS = ["AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP"]
ALL_METRICS = ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)",
               "InceptionV3", "CLIP", "EffNet-B", "SwAV"]

def label_re(arm):
    """<arm>_s<strength>_m<mixing>, the naming every sweep runner uses."""
    return re.compile(rf"^{re.escape(arm)}_s([0-9.]+)_m([0-9.]+)$")


def read_tsv(path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--arm", default="T0",
                     help="which arm's sweep to read and relabel: TSVs are named "
                          "<arm>_s<strength>_m<mixing>.tsv (T0, T1, T2, plus extra arms "
                          "such as T1nol2). Same criterion, fixed in advance, for all of them.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--winner-json", required=True)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    label_pattern = label_re(args.arm)
    tsvs = sorted(sweep_dir.glob(f"{args.arm}_s*_m*.tsv"))
    if not tsvs:
        raise SystemExit(f"no {args.arm}_s*_m*.tsv found under {sweep_dir}")

    grid = []
    per_label_rows = {}
    for p in tsvs:
        label = p.stem
        m = label_pattern.match(label)
        if not m:
            raise SystemExit(f"{p}: label '{label}' doesn't match {args.arm}_s<strength>_m<mixing>")
        strength, mixing = float(m.group(1)), float(m.group(2))
        rows = read_tsv(p)
        for r in rows:
            for metric in ALL_METRICS:
                r[metric] = float(r[metric])
        per_label_rows[label] = rows
        means = {metric: sum(r[metric] for r in rows) / len(rows) for metric in ALL_METRICS}
        primary_mean = sum(means[m2] for m2 in PRIMARY_METRICS) / len(PRIMARY_METRICS)
        grid.append({"label": label, "strength": strength, "mixing": mixing,
                     "n_rows": len(rows), "primary_mean": primary_mean, **means})
        print(f"{label}: primary_mean={primary_mean:.4f}  " +
              "  ".join(f"{m2}={means[m2]:.4f}" for m2 in ALL_METRICS))

    winner = max(grid, key=lambda g: g["primary_mean"])
    print(f"\nwinner: {winner['label']} (primary_mean={winner['primary_mean']:.4f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    winner_rows = per_label_rows[winner["label"]]
    cols = ["model", "subj", "mode", "stim_type", "sample", "repetition"] + ALL_METRICS
    with open(out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in winner_rows:
            r = dict(r)
            r["model"] = args.arm
            f.write("\t".join(f"{r[c]:.6f}" if c in ALL_METRICS else str(r[c])
                              for c in cols) + "\n")
    print(f"wrote {len(winner_rows)} rows to {out} (model relabelled '{args.arm}')")

    with open(args.winner_json, "w") as f:
        json.dump({"winner": winner, "grid": grid,
                   "selection_criterion": "argmax mean(AlexNet(2), AlexNet(5), InceptionV3, "
                                          "CLIP), fixed before the sweep was read"}, f, indent=2)
    print(f"wrote full grid + winner to {args.winner_json}")


if __name__ == "__main__":
    main()
