#!/usr/bin/env bash
# CPU steps of the shared1000 oracle budget: the selection split for the strength and mixing of T3
# (Section 8.2). It can run while the budget sweeps own both GPUs.
#
# Outputs: ridge weights refit without the dev split (subject 01 only: the constants are chosen once),
#          their predictions on the dev trials, a dev block set in the 12-stimulus format of the harness
#
# Steps: 1) fit_ridge.py --exclude-dev; 2) predict_ridge_shared1000.py --split dev; 3)
# make_shared1000_blockset.py with --pred-dir and --out-dir. The constants must not be chosen on
# shared1000; they are selected on the per-subject dev hold-out (derive_nsdcore_dev_mask.py).
#
# Usage: setsid nohup ./src/run_oracle_shared1000_prep.sh > run_oracle_shared1000_prep.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # project root (the folder that holds data/)

# PROJECT_ROOT: the folder that holds .uv/ (the uv binary); export it before launching.
export PATH="$PROJECT_ROOT/.uv/bin:$PATH"
export OPENBLAS_CORETYPE=SkylakeX
# The GPU sweep needs few cores, but its CPU evaluations later do: leave half the machine.
export OPENBLAS_NUM_THREADS=48
export OMP_NUM_THREADS=48

TARGETS_PY=.venv_targets/bin/python
SUBJ=1
DEVHELD_DIR=data/betas_trisect/ridge_weights_devheld
DEV_PRED_DIR=data/betas_trisect/nsdcore_predictions_dev
DEV_BLOCKSET=data/dev_blockset_subj01

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== oracle shared1000 prep (dev selection split, subj0${SUBJ}): start ==="

# --- stage 1: ridge refit excluding dev (the expensive one, ~30 min/branch measured) ---
if [ ! -f "$DEVHELD_DIR/subj0${SUBJ}/summary.json" ]; then
  log "stage 1: refitting the ridge with --exclude-dev for subj0${SUBJ} (4 branches)"
  "$TARGETS_PY" src/fit_ridge.py --subj "$SUBJ" --exclude-dev \
    --out-dir "$DEVHELD_DIR" > log_d1_8_ridge_devheld_subj0${SUBJ}.log 2>&1
  log "stage 1: done"
else
  log "stage 1: $DEVHELD_DIR/subj0${SUBJ}/summary.json already present, skipping"
fi

# --- stage 2: reapply those weights to the dev trials ---
if [ ! -f "$DEV_PRED_DIR/subj0${SUBJ}/img73k_ids.npy" ]; then
  log "stage 2: predicting the dev split from the devheld weights"
  "$TARGETS_PY" src/predict_ridge_shared1000.py --subj "$SUBJ" --split dev \
    --weights-dir "$DEVHELD_DIR" --out-dir "$DEV_PRED_DIR"
else
  log "stage 2: $DEV_PRED_DIR/subj0${SUBJ} already present, skipping"
fi

# --- stage 3: dev blockset, same 12-stimulus format as the shared1000 one ---
if [ ! -f "$DEV_BLOCKSET/manifest.json" ]; then
  log "stage 3: building the dev blockset"
  "$TARGETS_PY" src/make_shared1000_blockset.py --subj "$SUBJ" \
    --pred-dir "$DEV_PRED_DIR" --out-dir "$DEV_BLOCKSET"
else
  log "stage 3: $DEV_BLOCKSET/manifest.json already present, skipping"
fi

log "=== oracle shared1000 prep: done -- run_oracle_shared1000.sh can now select the T3 constants on dev ==="
