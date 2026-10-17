#!/bin/bash
# Score the reconstructions of the decodable basis (k=64) with the Kneeland harness (Section 8.1.3).
#
# Inputs:  data/nsdcore_recons_b3/S1K_I3K64_subj0N_s0.85_m0.6 (reconstruct.py, one replica, on the 65 blocks)
# Outputs: data/results/low_level_factor/decodable_basis/S1K_I3K64_subj0N.tsv
#
# Those reconstruct.py calls (--pred-dir at the output of decodable_basis_predict_shared1000.py) were typed
# on the command line and are not saved as a script (see README). Same 65 blocks as the Brain-Diffuser
# comparison, as in run_bd_swap_eval.sh. Run it after the four subjects are done.
set -e
cd "$(dirname "$0")/.."
# PROJECT_ROOT: the folder that holds .uv/ (the uv binary); export it before launching.
export PATH="$PROJECT_ROOT/.uv/bin:$PATH"
export OPENBLAS_CORETYPE=SkylakeX
source .venv_kneeland/bin/activate
# The torch build of .venv_kneeland is CPU-only (same as in run_bd_swap_eval.sh).

BLOCKS=$(python3 -c "import json; print(' '.join(str(b) for b in json.load(open('data/results/shared1000_comparison/bd_block_index.json'))['blocks_kept']))")
OUT_DIR=data/results/low_level_factor/decodable_basis
mkdir -p "$OUT_DIR"

for subj in 01 02 05 07; do
  echo "=== subj$subj ==="
  python src/evaluate_block_set.py \
    --recon-root data/nsdcore_recons_b3/S1K_I3K64_subj${subj}_s0.85_m0.6 \
    --proxy-dir data/shared1000_blockset_gt425 \
    --blocks $BLOCKS \
    --label S1K_I3K64_subj${subj} --gen-rep 1 --device cpu \
    --out $OUT_DIR/S1K_I3K64_subj${subj}.tsv
done

echo "decodable-basis eval done."
