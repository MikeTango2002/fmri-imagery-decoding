#!/bin/bash
# Step 2 of the swap with Brain-Diffuser (Section 8.4): the Kneeland image-metric harness on the three arms.
#
# Inputs:  the reconstructions of run_bd_swap.sh, data/shared1000_blockset_gt425, bd_block_index.json
# Outputs: one TSV per (arm, subject) in data/results/shared1000_comparison/swap/
#
# evaluate_block_set.py is reused as it is: it only needs a manifest.json and a blocks/blockNN_gt.pt folder,
# which the 425-pixel ground truth of prepare_bd_shared1000_blocks.py already is.
#
# Usage (from the project root, after run_bd_swap.sh, PROJECT_ROOT exported):
#   bash src/run_bd_swap_eval.sh
set -e
cd "$(dirname "$0")/.."
# PROJECT_ROOT: the folder that holds .uv/ (the uv binary); export it before launching.
export PATH="$PROJECT_ROOT/.uv/bin:$PATH"
export OPENBLAS_CORETYPE=SkylakeX
source .venv_kneeland/bin/activate
# The torch build of .venv_kneeland is CPU-only (as in run_bd_shared1000_blocks.sh):
# evaluate_block_set.py with --device cuda stops with "Torch not compiled with CUDA enabled".

BLOCKS=$(python3 -c "import json; print(' '.join(str(b) for b in json.load(open('data/results/shared1000_comparison/bd_block_index.json'))['blocks_kept']))")
OUT_DIR=data/results/shared1000_comparison/swap
mkdir -p "$OUT_DIR"

for arm in BT TB BB; do
  for subj in 01 02 05 07; do
    echo "=== $arm subj$subj ==="
    python src/evaluate_block_set.py \
      --recon-root data/nsdcore_recons_b3/S1K_${arm}_subj${subj}_s0.85_m0.6 \
      --proxy-dir data/shared1000_blockset_gt425 \
      --blocks $BLOCKS \
      --label S1K_${arm}_subj${subj} --gen-rep 1 --device cpu \
      --out $OUT_DIR/S1K_${arm}_subj${subj}.tsv
  done
done

echo "swap eval done."
