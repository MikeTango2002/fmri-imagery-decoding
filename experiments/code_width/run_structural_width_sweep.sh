#!/usr/bin/env bash
# Sweep over the width of z_struct (Section 5.2): for each d_out, refit the PCA of z_struct, rebuild
# targets.h5, retrain the adapter and run the adapter-vs-true-context check.
#
# Question: does the weak structural channel (35.66% explained variance at the default d=32) improve
# with more components, or is 32 the information ceiling of pooled DINOv2? A failed step skips to the
# next d; one bad run does not stop the sweep. GPU needed; launch with setsid nohup to survive a
# disconnect.
#
# Usage:
#   bash run_structural_width_sweep.sh             # d_struct = 64 96 (default)
#   bash run_structural_width_sweep.sh 64 96 128   # or an explicit list
set -uo pipefail
cd "$(dirname "$0")"

if [ "$#" -eq 0 ]; then
    D_VALUES=(64 96)
else
    D_VALUES=("$@")
fi

source .venv_targets/bin/activate

LOG_DIR="logs/struct_sweep"
mkdir -p "$LOG_DIR"

for D in "${D_VALUES[@]}"; do
    echo "=== d_struct=$D : 1/4 fit_pca_zstruct ==="
    python src/fit_pca_zstruct.py --d-out "$D" > "$LOG_DIR/d${D}_1_fit_pca.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "d_struct=$D: fit_pca_zstruct FAILED (see $LOG_DIR/d${D}_1_fit_pca.log), skipping to next d"
        continue
    fi

    echo "=== d_struct=$D : 2/4 assemble_targets ==="
    python src/assemble_targets.py --struct-d "$D" > "$LOG_DIR/d${D}_2_extract_targets.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "d_struct=$D: assemble_targets FAILED (see $LOG_DIR/d${D}_2_extract_targets.log), skipping to next d"
        continue
    fi

    echo "=== d_struct=$D : 3/4 train_adapter (40 epochs) ==="
    python src/train_adapter.py \
        --targets "data/targets/targets_structd${D}.h5" \
        --out "data/targets/a_psi_checkpoint_structd${D}.pt" \
        --epochs 40 \
        > "$LOG_DIR/d${D}_3_train_a_psi.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "d_struct=$D: train_adapter FAILED (see $LOG_DIR/d${D}_3_train_a_psi.log), skipping to next d"
        continue
    fi

    echo "=== d_struct=$D : 4/4 sample_adapter_vs_true_context ==="
    python src/sample_adapter_vs_true_context.py \
        --targets "data/targets/targets_structd${D}.h5" \
        --ckpt "data/targets/a_psi_checkpoint_structd${D}.pt" \
        --out-dir "data/gate0_structd${D}" \
        --n-images 6 --seed 0 \
        > "$LOG_DIR/d${D}_4_gate.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "d_struct=$D: sample_adapter_vs_true_context FAILED (see $LOG_DIR/d${D}_4_gate.log)"
        continue
    fi

    echo "=== d_struct=$D : done, see data/gate0_structd${D}/ ==="
done

echo "Sweep finished."
