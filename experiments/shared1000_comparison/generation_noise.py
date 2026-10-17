#!/usr/bin/env python3
"""Generation noise of TRISECT on the shared1000 blocks, after the fixes of seed and averaging (Sections 6.4, 7.1).

Inputs:  shared1000_comparison/S1K_T3_h0_subj0N.tsv (run_generation_noise_shared1000.sh)
Outputs: data/results/shared1000_comparison/generation_noise_floor.{tsv,json}

Null comparison: replicas {0,1} against {2,3} of the same arm; the true difference is zero. The check is whether
its CI contains zero now that every stimulus gets its own diffusion noise (diffusion_seed() in reconstruct.py).
TRISECT against itself; all 83 blocks are used, a superset of the 65 shared with Brain-Diffuser. load_cells,
paired and report are imported from compare_trisect_vs_bd_shared1000.py.

Usage (from the project root; reads TSVs only):
    python generation_noise.py
"""
import json
from pathlib import Path

from oracle_budget import PRIMARY, SUBJECTS
from compare_trisect_vs_bd_shared1000 import absolute_means, load_cells, report

S1K_DIR = "data/results/shared1000_comparison"
OUT_JSON = f"{S1K_DIR}/generation_noise_floor.json"
OUT_TSV = f"{S1K_DIR}/generation_noise_floor.tsv"
N_BOOT = 10000
SEED = 0


def main():
    s1k = Path(S1K_DIR)
    tri_a, tri_b = {}, {}
    for sub in SUBJECTS:
        t = s1k / f"S1K_T3_h0_subj{sub}.tsv"
        if not t.exists():
            raise SystemExit(f"missing {t} -- run src/run_generation_noise_shared1000.sh first")
        tri_a.update(load_cells(t, int(sub), reps={0, 1}))
        tri_b.update(load_cells(t, int(sub), reps={2, 3}))

    absolute_means({**tri_a, **tri_b}, "TRISECT S1K_T3 (after the fixes, all reps pooled)")

    rows = []
    report("Generation null: TRISECT replicas {0,1} - {2,3}, after the seed and re-standardization fixes "
           "(true delta = 0), 83 blocks x 4 subjects",
           tri_a, tri_b, "reps01-reps23", N_BOOT, SEED, "h0_generation_null", rows)

    floor = {r["metric"]: max(abs(r["ci_lo"]), abs(r["ci_hi"])) for r in rows}
    primary_row = next(r for r in rows if r["metric"] == PRIMARY)
    zero_included = primary_row["ci_lo"] <= 0.0 <= primary_row["ci_hi"]

    print(f"\n=== GENERATION NULL: {'PASS' if zero_included else 'FAIL'} ===")
    print(f"  primary_mean null: {primary_row['point']:+.3f} pp "
          f"[{primary_row['ci_lo']:+.3f}, {primary_row['ci_hi']:+.3f}]  "
          f"zero {'included' if zero_included else 'EXCLUDED'}")
    print(f"  new floor (largest |CI endpoint|), primary_mean = {floor[PRIMARY]:.3f} pp")

    import csv
    p = Path(OUT_TSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {p}")

    with open(OUT_JSON, "w") as fh:
        json.dump({"n_blocks": "83 (full shared1000_blockset, no BD intersection -- see "
                                "run_generation_noise_shared1000.sh header)",
                   "n_subjects": len(SUBJECTS), "gate": "PASS" if zero_included else "FAIL",
                   "primary_mean_null": {"point": primary_row["point"],
                                          "ci_lo": primary_row["ci_lo"],
                                          "ci_hi": primary_row["ci_hi"]},
                   "generation_floor": {k: float(v) for k, v in floor.items()}}, fh, indent=2)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
