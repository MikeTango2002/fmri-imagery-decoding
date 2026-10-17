#!/usr/bin/env bash
# Brain-Diffuser through the benchmark harness on the shared1000 block set, then the three readings of
# compare_trisect_vs_bd_shared1000.py (Sections 6.4, 7.1 and 8.4). No generation GPU is needed.
#
# Inputs:  Brain-Diffuser's reconstructions (benchmark A), TRISECT's cells from run_oracle_shared1000.sh
# Outputs: bd_s1k_subj0N.tsv and bd_s1k425_subj0N.tsv, the three readings, bd_vs_published_check.json
#
# Steps: (1) prepare_bd_shared1000_blocks.py: match Brain-Diffuser's rows to the blocks, check on pixels;
# (2) score it on the blocks, paired with TRISECT; (3) score it against the 425-native ground truth;
# (4) the three readings; (5) the absolute values of step 3 against the published ones.
#
# Usage (from the project root):
#   setsid nohup bash src/run_bd_shared1000_blocks.sh > log_bd_shared1000_blocks.log 2>&1 < /dev/null &
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$PROJECT_ROOT/.uv/bin:$PATH"
export OPENBLAS_CORETYPE=SkylakeX
source .venv_kneeland/bin/activate

SUBJ=(01 02 05 07)
G6=data/results/shared1000_comparison
DEV="${G6_DEVICE:-cuda}"

echo "=== step 1: index, mapping check, symlinks, 425 GT ==="
python src/prepare_bd_shared1000_blocks.py

BLOCKS=$(python - <<'PY'
import json
print(" ".join(str(b) for b in json.load(open("data/results/shared1000_comparison/bd_block_index.json"))["blocks_kept"]))
PY
)
echo "kept blocks: $BLOCKS"

echo "=== step 2: BD through the harness, blockset GT (224->425) -- the paired axis ==="
for s in "${SUBJ[@]}"; do
    python src/evaluate_block_set.py \
        --recon-root "data/bd_s1k_blockset/subj${s}" \
        --proxy-dir data/shared1000_blockset \
        --label "bd_s1k_subj${s}" --gen-rep 1 --device "$DEV" \
        --blocks $BLOCKS \
        --out "${G6}/bd_s1k_subj${s}.tsv"
done

echo "=== step 3: BD through the harness, 425-native GT -- the comparison with the published values ==="
for s in "${SUBJ[@]}"; do
    python src/evaluate_block_set.py \
        --recon-root "data/bd_s1k_blockset/subj${s}" \
        --proxy-dir data/shared1000_blockset_gt425 \
        --label "bd_s1k425_subj${s}" --gen-rep 1 --device "$DEV" \
        --blocks $BLOCKS \
        --out "${G6}/bd_s1k425_subj${s}.tsv"
done

echo "=== step 4: the three pre-registered readings ==="
python src/compare_trisect_vs_bd_shared1000.py

echo "=== step 5: absolutes on the 425 GT against the published values, per subject and pooled ==="
python - <<'PY'
import csv, json, numpy as np
from pathlib import Path
G6 = Path("data/results/shared1000_comparison")
M = ["PixCorr", "SSIM", "EffNet-B", "SwAV"]
pub = {"PixCorr": 0.273, "SSIM": 0.365, "EffNet-B": 0.728, "SwAV": 0.421}   # published values (Section 4.4.3)
tol = {"PixCorr": 0.04, "SSIM": 0.04, "EffNet-B": 0.01, "SwAV": 0.01}       # tolerance of the comparison
out, per_subj = {}, {}
for s in ("01", "02", "05", "07"):
    rows = list(csv.DictReader(open(G6 / f"bd_s1k425_subj{s}.tsv"), delimiter="\t"))
    per_subj[s] = {m: float(np.mean([float(r[m]) for r in rows])) for m in M}
for m in M:
    v = float(np.mean([per_subj[s][m] for s in per_subj]))
    d = v - pub[m]
    out[m] = {"measured": v, "published": pub[m], "delta": d,
              "within_tolerance": bool(abs(d) <= tol[m]), "tolerance": tol[m]}
    print(f"  {m:<10} {v:.4f}  published {pub[m]:.3f}  delta {d:+.4f}  "
          f"{'OK' if abs(d) <= tol[m] else 'OUT'}")
json.dump({"per_subject": per_subj, "pooled_vs_published": out},
          open(G6 / "bd_vs_published_check.json", "w"), indent=2)
print(f"wrote {G6/'bd_vs_published_check.json'}")
PY

echo "=== done ==="
