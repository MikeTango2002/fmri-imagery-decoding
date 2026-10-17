#!/bin/bash
# The 2x2 swap with Brain-Diffuser (Section 8.4): generation of the BT, TB and BB arms.
#
# Inputs:  the 65 kept blocks (prepare_bd_shared1000_blocks.py), Brain-Diffuser's predicted latents
# Outputs: data/nsdcore_recons_b3/S1K_{BT,TB,BB}_subj0N_s0.85_m0.6/
#
# Gen-rep 1, with the production constants of TRISECT (s=0.85 m=0.6, the same as the TT control
# oracle_budget/S1K_T3_subj0N.tsv). BT takes the context from Brain-Diffuser, TB its z_low, BB both.
# One GPU, sequential: one job per GPU, not DDP.
#
# Usage (from the project root, PROJECT_ROOT exported):
#   bash src/run_bd_swap.sh
set -e
cd "$(dirname "$0")/.."
# PROJECT_ROOT: the folder that holds .uv/ (the uv binary); export it before launching.
export PATH="$PROJECT_ROOT/.uv/bin:$PATH"
export OPENBLAS_CORETYPE=SkylakeX
export CUDA_VISIBLE_DEVICES=0
source .venv_targets/bin/activate

COMMON="--dataset nsdcore --gen-rep 1 --harness-blocks --strength 0.85 --mixing 0.6 --seed 0"

for subj in 1 2 5 7; do
  echo "=== BT subj0${subj} ==="
  python src/reconstruct.py $COMMON --subj $subj --bd-context \
    --out-dir data/nsdcore_recons_b3/S1K_BT_subj0${subj}_s0.85_m0.6
done

for subj in 1 2 5 7; do
  echo "=== TB subj0${subj} ==="
  python src/reconstruct.py $COMMON --subj $subj --bd-zlow \
    --out-dir data/nsdcore_recons_b3/S1K_TB_subj0${subj}_s0.85_m0.6
done

for subj in 1 2 5 7; do
  echo "=== BB subj0${subj} ==="
  python src/reconstruct.py $COMMON --subj $subj --bd-context --bd-zlow \
    --out-dir data/nsdcore_recons_b3/S1K_BB_subj0${subj}_s0.85_m0.6
done

echo "swap generation done."
