"""Ceiling of any per-trial choice of strength and mixing, read from the 3x3 sweeps already on disk.

Inputs:  the per-row TSVs of evaluate_block_set.py for the 9 settings (data/results/oracle_budget/setting_sweep/)
Outputs: a console table per arm: best fixed, grid span, naive oracle, cross-fit (Table 8.3, Section 8.1.2)

No generation, no evaluation. A row scores primary_mean, the mean of AlexNet(2), AlexNet(5), InceptionV3 and CLIP;
the unit is one image, the replica is the `repetition` column. Arms: devT3 (NSD-core dev split, subject 1,
brain-predicted factors, 300 images, 2 replicas) and T2 (proxy set, true factors, 150 simple + 150 complex, 4
replicas). The four quantities and the tie rules are explained above pick() and main().

Usage (pandas; from the folder that holds data/):
    python gating_oracle_ceiling.py
"""
import itertools

import numpy as np
import pandas as pd

SWEEP_DIR = "data/results/oracle_budget/setting_sweep/"
SETTINGS = [(s, m) for s in (0.65, 0.75, 0.85) for m in (0.2, 0.4, 0.6)]
TWO_WAY = ["AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP"]
ARMS = [("devT3", None), ("T2", "simple"), ("T2", "complex")]


def load_scores(arm, stim_type):
    """Array (replicas, images, settings) of primary_mean per row."""
    frames = []
    for s, m in SETTINGS:
        d = pd.read_csv(f"{SWEEP_DIR}{arm}_s{s}_m{m}.tsv", sep="\t")
        if stim_type is not None:
            d = d[d.stim_type == stim_type]
        frames.append(d.assign(p=d[TWO_WAY].mean(axis=1), setting=f"{s}/{m}"))
    d = pd.concat(frames)
    table = d.pivot_table(index=["subj", "sample"], columns=["setting", "repetition"], values="p")
    order = [f"{s}/{m}" for s, m in SETTINGS]
    reps = sorted(table.columns.get_level_values(1).unique())
    return np.stack([table.xs(r, axis=1, level=1)[order].to_numpy() for r in reps])


# Two-way scores move in steps of 1/11, and the per-image argmax often has ties. The cross-fit is therefore reported
# under four rules for breaking them: first or last in grid order, preference for the best fixed setting, mean over the
# tied settings.
def pick(scores, rule, fixed_order):
    """Per-image chosen setting(s) as a (images, settings) weight matrix summing to 1 per row."""
    n, c = scores.shape
    w = np.zeros((n, c))
    if rule == "mean_of_ties":
        tied = np.isclose(scores, scores.max(axis=1, keepdims=True))
        return tied / tied.sum(axis=1, keepdims=True)
    if rule == "first":
        idx = scores.argmax(axis=1)
    elif rule == "last":
        idx = c - 1 - scores[:, ::-1].argmax(axis=1)
    elif rule == "prefer_best_fixed":
        idx = fixed_order[scores[:, fixed_order].argmax(axis=1)]
    w[np.arange(n), idx] = 1.0
    return w


# Per arm, on scores of shape (replicas, images, settings):
#   best fixed    max over the 9 settings of the mean score
#   grid span     max - min over the 9 settings of the mean score
#   naive oracle  for every (image, replica) the best of the 9 settings, scored on that same replica: the selection
#                 runs on generation noise
#   cross-fit     the setting is picked on replica a and scored on replica b != a, averaged over all ordered pairs
def main():
    rules = ["first", "last", "prefer_best_fixed", "mean_of_ties"]
    for arm, stim_type in ARMS:
        a = load_scores(arm, stim_type)
        n_rep, n_img, _ = a.shape
        fixed = a.mean(axis=(0, 1))
        best = fixed.max()
        fixed_order = np.argsort(-fixed)
        naive = a.max(axis=2).mean()
        label = arm if stim_type is None else f"{arm} {stim_type}"
        print(f"{label}: {n_img} images x {n_rep} replicas")
        print(f"  best fixed {best:.4f} at s/m {SETTINGS[fixed.argmax()]}, "
              f"grid span {100 * (fixed.max() - fixed.min()):.2f} pp")
        print(f"  naive oracle {naive:.4f} ({100 * (naive - best):+.2f} pp)")
        for rule in rules:
            vals = [(pick(a[i], rule, fixed_order) * a[j]).sum(axis=1).mean()
                    for i, j in itertools.permutations(range(n_rep), 2)]
            cf = float(np.mean(vals))
            print(f"  cross-fit [{rule}] {cf:.4f} ({100 * (cf - best):+.2f} pp)")


if __name__ == "__main__":
    main()
