#!/usr/bin/env bash
# Voxel-side arms of the ridge input: CPU-only fits scored on shared1000.
#
# Arm 1: per-voxel reliability weighting. Arm 2: restriction to the voxels of a `streams` ROI (higher_ventral plus
# the 6 individual ROIs; the nsdgeneral control is reused from data/betas_trisect/ridge_weights/subj0N/summary.json).
#
# Each fit is BLAS-parallel: a test on subject 1 (ROI arm, z_low only, 7 configurations) took 4m37s of wall time on
# 128 cores.
#
# Usage: setsid nohup bash src/run_voxel_reliability_and_roi.sh > log_voxel_reliability_roi.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # project root

PY=.venv_targets/bin/python

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== streams atlas -> nsdgeneral voxel order ==="
"$PY" src/map_streams_rois.py > log_h4_preflight.log 2>&1
log "preflight done"

log "=== arm 1: reliability weighting, all subjects, all branches ==="
"$PY" src/voxel_reliability_and_roi.py --arm reliability --subj 1 2 5 7 \
    --out-dir data/betas_trisect/h4 > log_h4_reliability.log 2>&1
log "arm 1 (reliability) done"

log "=== arm 2: ROI structure, all subjects, all branches ==="
"$PY" src/voxel_reliability_and_roi.py --arm roi --subj 1 2 5 7 \
    --out-dir data/betas_trisect/h4 > log_h4_roi.log 2>&1
log "arm 2 (roi) done"

log "=== CPU-only fits complete ==="
