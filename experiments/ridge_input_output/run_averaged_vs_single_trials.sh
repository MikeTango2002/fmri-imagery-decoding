#!/usr/bin/env bash
# Averaged versus single-trial ridge: fit the two arms and run the comparison. CPU only.
#
# Outputs: ridge_weights_wide (control, z_low only), ridge_weights_avg (treatment, four branches), the comparison TSV
#
# Steps: (0) control, single trial, wide alpha grid, GroupKFold (plain leave-one-out would leak the other
# repetitions of an image, which share its target row, into the validation fold); (1) treatment, betas averaged
# over the repetitions of each image before z-scoring, same grid; (2) compare_averaged_vs_single_trials.py.
# The semantic branches are not refit for the control: their alpha sits well inside the default grid; they
# are read from the final fit in ridge_weights/. Some CPUs need OPENBLAS_CORETYPE=SkylakeX (warning below).
#
# Usage (project root, targets environment active):
#   bash src/run_averaged_vs_single_trials.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${OPENBLAS_CORETYPE:-}" != "SkylakeX" ]; then
    echo "WARNING: OPENBLAS_CORETYPE is not SkylakeX -- export it first if this is the" >&2
    echo "thesis server (Sapphire Rapids), or the ridge solve can be silently wrong." >&2
fi

SUBJ="1 2 5 7"
WIDE_DIR=data/betas_trisect/ridge_weights_wide
AVG_DIR=data/betas_trisect/ridge_weights_avg
GATE_OUT=data/results/ridge_input_output/averaged_vs_single_trials.tsv

echo "=== step 0: control -- z_low, single trial, wide grid, group-cv ==="
python src/fit_ridge.py --subj $SUBJ --branches z_low \
    --alphas-logspace 1 7 25 --group-cv \
    --out-dir "$WIDE_DIR"

echo "=== step 1: treatment -- all 4 branches, averaged input, wide grid ==="
python src/fit_ridge.py --subj $SUBJ --average-reps \
    --alphas-logspace 1 7 25 \
    --out-dir "$AVG_DIR"

echo "=== step 2: the gate ==="
python src/compare_averaged_vs_single_trials.py --out "$GATE_OUT"
