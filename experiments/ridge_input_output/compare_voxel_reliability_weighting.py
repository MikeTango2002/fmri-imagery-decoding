#!/usr/bin/env python3
"""End-to-end comparison: ridge with reliability-weighted voxels versus the unweighted final ridge.

Inputs:  shared1000_comparison/S1K_T3_h0_subj0N.tsv (control), ridge_input_output/S1K_H4REL_subj0N.tsv
Outputs: voxel_reliability_vs_production.{tsv,json} under data/results/ridge_input_output/

The weights are r_v = signal_var / (signal_var + noise_var) per voxel, estimated on about 9000 repeated
NSD-core train images. Same paired design as compare_dimension_gain.py (same seed, block set and generator
settings) and the same helpers from compare_trisect_vs_bd_shared1000.py. The two are separate because they
decide two independent arms, which may pass or fail independently; every arm is reported.

Usage (reads TSVs only, no GPU):
    python src/compare_voxel_reliability_weighting.py
"""
import csv
import json
from pathlib import Path

from oracle_budget import PRIMARY, SUBJECTS
from compare_trisect_vs_bd_shared1000 import absolute_means, load_cells, report

S1K_DIR = "data/results"
H0_FLOOR_JSON = f"{S1K_DIR}/shared1000_comparison/generation_noise_floor.json"
OUT_JSON = f"{S1K_DIR}/ridge_input_output/voxel_reliability_vs_production.json"
OUT_TSV = f"{S1K_DIR}/ridge_input_output/voxel_reliability_vs_production.tsv"
N_BOOT = 10000
SEED = 0


def main():
    s1k = Path(S1K_DIR)
    control, treatment = {}, {}
    for sub in SUBJECTS:
        c = s1k / f"shared1000_comparison/S1K_T3_h0_subj{sub}.tsv"
        t = s1k / f"ridge_input_output/S1K_H4REL_subj{sub}.tsv"
        if not c.exists():
            raise SystemExit(f"missing {c} -- run src/run_generation_noise_shared1000.sh first")
        if not t.exists():
            raise SystemExit(f"missing {t} -- run src/run_gain_and_reliability_reconstructions.sh first")
        control.update(load_cells(c, int(sub)))
        treatment.update(load_cells(t, int(sub)))

    abs_c = absolute_means(control, "control (unweighted, S1K_T3_h0)")
    abs_t = absolute_means(treatment, "treatment (reliability-weighted, S1K_H4REL)")

    with open(H0_FLOOR_JSON) as f:
        floor = json.load(f)["generation_floor"]

    rows = []
    report("Reliability-weighted ridge - unweighted final ridge, paired on "
           "(subj, block, stimulus), 83 blocks x 4 subjects", treatment, control,
           "h4rel-h4ctrl", N_BOOT, SEED, "h4_e2e", rows)

    primary_row = next(r for r in rows if r["metric"] == PRIMARY)
    zero_excluded = not (primary_row["ci_lo"] <= 0.0 <= primary_row["ci_hi"])
    above_floor = abs(primary_row["point"]) > floor[PRIMARY]
    passed = zero_excluded and above_floor and primary_row["point"] > 0

    print(f"\n=== RELIABILITY WEIGHTING, END-TO-END GATE: {'PASS' if passed else 'FAIL'} ===")
    print(f"  primary_mean delta: {primary_row['point']:+.3f} pp "
          f"[{primary_row['ci_lo']:+.3f}, {primary_row['ci_hi']:+.3f}]  "
          f"zero {'EXCLUDED' if zero_excluded else 'included'}, "
          f"generation-noise floor {floor[PRIMARY]:.3f} pp -- "
          f"{'above' if above_floor else 'BELOW'} floor")

    p = Path(OUT_TSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {p}")

    with open(OUT_JSON, "w") as fh:
        json.dump({"gate": "PASS" if passed else "FAIL",
                   "primary_mean_delta": {"point": primary_row["point"],
                                           "ci_lo": primary_row["ci_lo"],
                                           "ci_hi": primary_row["ci_hi"]},
                   "h0_floor": floor[PRIMARY],
                   "absolute_control": abs_c, "absolute_treatment": abs_t}, fh, indent=2)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
