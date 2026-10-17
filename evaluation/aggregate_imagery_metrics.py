"""Aggregate the per-image TSVs of the NSD-Imagery benchmark into the tables of Kneeland et al. 2025.

Inputs:  --image_tsv from `final_evaluations_imagery.py` (8 image metrics) and --gnet_tsv from
         `gnet_brain_correlation.py` (3 brain columns), two scripts on purpose
Outputs: per_repetition.tsv, per_subject.tsv, overall.tsv, and the tables printed to stdout

Order: per stimulus -> per scope (all12 / simple / complex) -> mean over the 10 samples -> per subject ->
mean over the 4 subjects. all12 is Table 1; simple and complex are Tables 3 and 4. Samples are averaged on
metric values, never on images. Two-way metrics always use the pool of 12, whatever the scope.
The two-way columns are printed twice: `AlexNet(2)` etc. carry the authors' off-by-one (`- 1` in cell 8),
so chance is 40.91% and the ceiling 90.91% (comparable with the published Table 1). The `+1/11` columns
are the same values corrected, an exact per-image shift, which makes "near chance" readable.

Usage (MindEye2 environment):
    python aggregate_imagery_metrics.py --image_tsv results/baselines/mindeye_I.tsv \
        --gnet_tsv results/brain_correlation/mindeye_I_brain_corr.tsv --out_dir OUT_DIR
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

IMAGE_METRICS = ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)",
                 "InceptionV3", "CLIP", "EffNet-B", "SwAV"]
BRAIN_METRICS = ["Brain Corr. EVC", "Brain Corr. higher_vis", "Brain Corr. nsd_general"]
ALL_METRICS = IMAGE_METRICS + BRAIN_METRICS

# The four metrics that go through two_way_identification() and therefore carry the `- 1`.
TWO_WAY = ["AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP"]
TWO_WAY_SHIFT = 1.0 / 11.0      # 11 = N - 1 = the number of distractors

SCOPES = ["all12", "simple", "complex"]
MODES = ["imagery", "vision"]

# gnet_brain_correlation.py keys its rows by condition and region; this maps them onto the
# (mode, stim_type) / column names used here.
COND_TO_MODE_TYPE = {
    "imagerysimple":  ("imagery", "simple"),
    "imagerycomplex": ("imagery", "complex"),
    "visionsimple":   ("vision", "simple"),
    "visioncomplex":  ("vision", "complex"),
}
REGION_TO_COL = {
    "EVC": "Brain Corr. EVC",
    "higher_vis": "Brain Corr. higher_vis",
    "nsd_general": "Brain Corr. nsd_general",
}


def expand(patterns):
    files = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits:
            print(f"WARNING: no file matches {pat}")
        files += hits
    return files


def load_image_tsvs(patterns):
    """-> long frame: model, subj, mode, stim_type, sample, repetition, metric, value."""
    files = expand(patterns)
    if not files:
        return pd.DataFrame(columns=["model", "subj", "mode", "stim_type", "sample",
                                     "repetition", "metric", "value"])
    df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    print(f"image metrics: {len(df)} rows from {len(files)} file(s), "
          f"models {sorted(df.model.unique())}")
    return df.melt(id_vars=["model", "subj", "mode", "stim_type", "sample", "repetition"],
                   value_vars=IMAGE_METRICS, var_name="metric", value_name="value")


def load_gnet_tsvs(patterns):
    """The brain TSVs, reshaped onto the same long schema (sample <-> repetition)."""
    files = expand(patterns)
    if not files:
        return pd.DataFrame(columns=["model", "subj", "mode", "stim_type", "sample",
                                     "repetition", "metric", "value"])
    df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    df = df[df.condition.isin(COND_TO_MODE_TYPE)]
    # The ceiling rows (model "_ceiling_gt", sample -1) are a diagnostic of the GNet script,
    # not a model: they must never enter an average here.
    df = df[df["sample"] >= 0]
    if df.empty:
        return pd.DataFrame(columns=["model", "subj", "mode", "stim_type", "sample",
                                     "repetition", "metric", "value"])
    out = pd.DataFrame({
        "model": df.model,
        "subj": df.subj,
        "mode": df.condition.map(lambda c: COND_TO_MODE_TYPE[c][0]),
        "stim_type": df.condition.map(lambda c: COND_TO_MODE_TYPE[c][1]),
        # gnet numbers its 6 stimuli inside the condition; here `sample` is the stimulus
        # index in the 12-pool; the complex block is offset by 6 to line the two halves
        # up. Only the (mode, stim_type, subj) grouping is used downstream, but keeping the
        # index consistent makes the merged per-stimulus file readable.
        "sample": df.stim + df.condition.map(lambda c: 6 if "complex" in c else 0),
        "repetition": df["sample"],
        "metric": df.region.map(REGION_TO_COL),
        # df["corr"], never df.corr; the attribute is the DataFrame method.
        "value": df["corr"],
    })
    print(f"brain metrics: {len(out)} rows from {len(files)} file(s), "
          f"models {sorted(out.model.unique())}")
    return out


def aggregate(long: pd.DataFrame, expected_reps: int):
    """per stimulus -> scope -> mean over repetitions -> per subject -> mean over subjects."""
    scoped = []
    for scope in SCOPES:
        sel = long if scope == "all12" else long[long.stim_type == scope]
        sel = sel.assign(scope=scope)
        scoped.append(sel)
    long = pd.concat(scoped, ignore_index=True)

    # mean over the 12 (or 6) stimuli, inside one repetition
    per_rep = long.groupby(["model", "subj", "mode", "scope", "repetition", "metric"],
                           as_index=False)["value"].mean()

    counts = (per_rep[per_rep.scope == "all12"]
              .groupby(["model", "subj", "mode", "metric"])["repetition"].nunique())
    for key, n in counts.items():
        if n != expected_reps:
            print(f"WARNING: {key} has {n} repetitions, expected {expected_reps}")

    per_subject = per_rep.groupby(["model", "subj", "mode", "scope", "metric"],
                                  as_index=False)["value"].mean()
    overall = per_subject.groupby(["model", "mode", "scope", "metric"],
                                  as_index=False)["value"].mean()
    return per_rep, per_subject, overall


def as_table(frame, index_cols, metrics, digits=3):
    """Markdown, one row per index tuple, missing metrics printed as `--`."""
    wide = frame.pivot_table(index=index_cols, columns="metric", values="value")
    cols = [m for m in metrics if m in wide.columns]
    lines = ["| " + " | ".join(index_cols + cols) + " |",
             "| " + " | ".join(["---"] * (len(index_cols) + len(cols))) + " |"]
    for idx, row in wide.iterrows():
        idx = idx if isinstance(idx, tuple) else (idx,)
        cells = [str(v) for v in idx] + [
            "--" if pd.isna(row[c]) else f"{row[c]:.{digits}f}" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def two_way_table(frame, index_cols):
    """Two columns per two-way metric: the faithful value and the +1/11 correction."""
    wide = frame.pivot_table(index=index_cols, columns="metric", values="value")
    cols = [m for m in TWO_WAY if m in wide.columns]
    if not cols:
        return "(no two-way metric in this frame)"
    header = index_cols + [c for m in cols for c in (m, f"{m}+1/11")]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for idx, row in wide.iterrows():
        if all(pd.isna(row[m]) for m in cols):
            continue        # a model with only brain columns: nothing two-way to correct
        idx = idx if isinstance(idx, tuple) else (idx,)
        cells = [str(v) for v in idx]
        for m in cols:
            cells += (["--", "--"] if pd.isna(row[m])
                      else [f"{row[m]:.3f}", f"{row[m] + TWO_WAY_SHIFT:.3f}"])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image_tsv", nargs="*", default=[],
                   help="TSVs from final_evaluations_imagery.py (globs allowed)")
    p.add_argument("--gnet_tsv", nargs="*", default=[],
                   help="TSVs from gnet_brain_correlation.py (globs allowed)")
    p.add_argument("--expected_reps", type=int, default=10)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    long = pd.concat([load_image_tsvs(args.image_tsv), load_gnet_tsvs(args.gnet_tsv)],
                     ignore_index=True)
    if long.empty:
        raise SystemExit("nothing to aggregate: pass --image_tsv and/or --gnet_tsv")

    per_rep, per_subject, overall = aggregate(long, args.expected_reps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_rep.to_csv(out_dir / "per_repetition.tsv", sep="\t", index=False)
    per_subject.to_csv(out_dir / "per_subject.tsv", sep="\t", index=False)
    overall.to_csv(out_dir / "overall.tsv", sep="\t", index=False)

    for mode in MODES:
        sel = overall[(overall["mode"] == mode) & (overall.scope == "all12")]
        if sel.empty:
            continue
        print(f"\n### Table 1 -- {mode}, mean over subjects "
              "(as published, with the authors' off-by-one)")
        print(as_table(sel, ["model"], ALL_METRICS))
        print(f"\n#### two-way, as published vs corrected -- {mode} "
              "(chance: 0.409 as published / 0.500 corrected)")
        print(two_way_table(sel, ["model"]))

    sel = overall[overall.scope != "all12"]
    if not sel.empty:
        print("\n### Tables 3-4 -- by stimulus type (same pool of 12 distractors)")
        print(as_table(sel, ["model", "mode", "scope"], ALL_METRICS))

    sel = per_subject[per_subject.scope == "all12"]
    if not sel.empty:
        print("\n### Per subject (all12, mean over repetitions)")
        print(as_table(sel, ["model", "subj", "mode"], ALL_METRICS))

    print(f"\nwrote per_repetition.tsv / per_subject.tsv / overall.tsv to {out_dir}")


if __name__ == "__main__":
    main()
