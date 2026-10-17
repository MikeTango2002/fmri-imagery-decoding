#!/usr/bin/env bash
# Sweep over the width of z_sem (Section 5.2): for each d_sem, rebuild targets.h5 from the fitted
# pca_zsem/d<D> basis, retrain the adapter and run the adapter-vs-true-context check.
#
# Question: does the adapter image cosine, 0.670 at d_sem=128, rise with more components?
# The bases for all widths come from one run of pca_width_zsem.py; no PCA is refitted here.
# 128 is not in the default list: assemble_targets.py writes targets.h5 for the default width (no
# suffix), and that run is the baseline (data/targets/a_psi_checkpoint.pt). A failed step skips to
# the next d. GPU needed; launch with setsid nohup to survive a disconnect.
#
# Usage:
#   bash run_semantic_width_sweep.sh              # d_sem = 256 512 (default)
#   bash run_semantic_width_sweep.sh 256 512 768  # or an explicit list
set -uo pipefail
cd "$(dirname "$0")"

if [ "$#" -eq 0 ]; then
    D_VALUES=(256 512)
else
    D_VALUES=("$@")
fi

source .venv_targets/bin/activate

LOG_DIR="logs/sem_sweep"
mkdir -p "$LOG_DIR"

for D in "${D_VALUES[@]}"; do
    if [ ! -f "data/targets/pca_zsem/d${D}/pca_basis.npz" ]; then
        echo "d_sem=$D: no PCA basis at data/targets/pca_zsem/d${D}/ -- run pca_width_zsem.py first (it fits all of {32,64,128,256,512,768} in one shot), skipping"
        continue
    fi

    echo "=== d_sem=$D : 1/3 assemble_targets ==="
    python src/assemble_targets.py --sem-d "$D" > "$LOG_DIR/d${D}_1_extract_targets.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "d_sem=$D: assemble_targets FAILED (see $LOG_DIR/d${D}_1_extract_targets.log), skipping to next d"
        continue
    fi

    echo "=== d_sem=$D : 2/3 train_adapter (40 epochs) ==="
    python src/train_adapter.py \
        --targets "data/targets/targets_semd${D}.h5" \
        --out "data/targets/a_psi_checkpoint_semd${D}.pt" \
        --epochs 40 \
        > "$LOG_DIR/d${D}_2_train_a_psi.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "d_sem=$D: train_adapter FAILED (see $LOG_DIR/d${D}_2_train_a_psi.log), skipping to next d"
        continue
    fi

    echo "=== d_sem=$D : 3/3 sample_adapter_vs_true_context ==="
    python src/sample_adapter_vs_true_context.py \
        --targets "data/targets/targets_semd${D}.h5" \
        --ckpt "data/targets/a_psi_checkpoint_semd${D}.pt" \
        --out-dir "data/gate0_semd${D}" \
        --n-images 6 --seed 0 \
        > "$LOG_DIR/d${D}_3_gate.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "d_sem=$D: sample_adapter_vs_true_context FAILED (see $LOG_DIR/d${D}_3_gate.log)"
        continue
    fi

    echo "=== d_sem=$D : done, see data/gate0_semd${D}/ ==="
done

echo "Sweep finished."
