#!/usr/bin/env bash
# Generate TRISECT's shared1000 arm with a diffusion seed per (replica, subject, stimulus) in reconstruct.py
# and with averaging before re-standardizing in predict_ridge_shared1000.py, so that the generation null
# can be measured (Sections 6.4 and 7.1).
#
# Outputs: S1K_T3_h0_* images and shared1000_comparison/S1K_T3_h0_subj0N.tsv
#
# It runs all 83 blocks, not the 65 shared with Brain-Diffuser: the null compares TRISECT with itself and needs no
# Brain-Diffuser data, and 83 is a superset of the 65, which gives more power.
# A single reconstruct.py process saturates the GPU (about 144 s per block and subject, about 13 h for 4 subjects
# x 83 blocks x 4 replicas); subjects run in series. Resumable: reconstruct.py skips PNGs already on disk.
#
# Usage (from the project root):
#   setsid nohup bash src/run_generation_noise_shared1000.sh > log_generation_noise.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # project root (the folder that holds data/)

TARGETS_PY=.venv_targets/bin/python
KNEELAND_PY=.venv_kneeland/bin/python

BLOCKSET=data/shared1000_blockset
RECON_ROOT=data/nsdcore_recons_b3
SWEEP_DIR=data/results/shared1000_comparison/setting_sweep
GEN_REP=4
STRENGTH=0.85
MIXING=0.6            # S1K_T3's own winner combo (oracle_budget/S1K_T3_subj0N_setting_choice.json), unchanged
                       # by the fixes, which change the normalization and the seed, not the generator constants
SUBJECTS=(1 2 5 7)

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== regenerating S1K_T3 with the fixed seed and re-standardization order ==="

# --- stage 0: refresh the 4 subjects' shared1000 predictions (fast, CPU, minutes) ---
# All four subjects are done here, so that this script is a complete, repeatable entry point on its own.
for s in "${SUBJECTS[@]}"; do
  log "stage 0: predict_ridge_shared1000.py --subj $s"
  "$TARGETS_PY" src/predict_ridge_shared1000.py --subj "$s" \
    > "log_h0_predict_subj0${s}.log" 2>&1
done
log "stage 0: done"

# --- stage 1: reconstruct, one subject at a time (single GPU, already saturated, see header) ---
# Retry wrapper: the WSL2 GPU passthrough (dxgkrnl) can be flaky
# (`dxgkio_make_resident: Ioctl failed: -12`, an ENOMEM in the driver, which reaches the program as
# `torch.AcceleratorError: CUDA error: unknown error`). It is unrelated to this code: nvidia-smi
# shows a healthy GPU with no process right after each such crash. A fresh process reliably gets a
# working CUDA context again, and reconstruct.py already skips every (block, stimulus, replica) PNG
# that is on disk; running the same command again resumes with no lost work beyond the image
# being generated.
# `timeout` guards against the OTHER known WSL2 failure: a silent HANG (process alive, 0% GPU
# utilisation, no exception) instead of a crash. 6 h is generous (steady state is about 3.3 h per
# subject) but far from "stuck forever".
MAX_RETRIES=5
for s in "${SUBJECTS[@]}"; do
  label="S1K_T3_h0_subj0${s}_s${STRENGTH}_m${MIXING}"
  outdir="$RECON_ROOT/${label}"
  if [ -f "$outdir/.recon_done" ]; then
    log "stage 1: $label already done, skipping"
    continue
  fi
  attempt=1
  while [ $attempt -le $MAX_RETRIES ]; do
    log "stage 1: reconstructing $label (83 blocks x 12 stim x $GEN_REP reps, attempt $attempt/$MAX_RETRIES)"
    if timeout 6h "$TARGETS_PY" src/reconstruct.py \
        --dataset nsdcore --harness-blocks --blockset-dir "$BLOCKSET" --subj "$s" \
        --gen-rep "$GEN_REP" --strength "$STRENGTH" --mixing "$MIXING" \
        --out-dir "$outdir" >> "log_h0_recon_subj0${s}.log" 2>&1; then
      touch "$outdir/.recon_done"
      log "stage 1: $label done"
      break
    fi
    log "stage 1: $label attempt $attempt FAILED (see log_h0_recon_subj0${s}.log -- likely the "
    log "  known WSL2/dxgkrnl CUDA flakiness, not a code bug) -- retrying after 10s"
    attempt=$((attempt + 1))
    sleep 10
  done
  if [ ! -f "$outdir/.recon_done" ]; then
    log "stage 1: $label FAILED after $MAX_RETRIES attempts -- giving up, see log_h0_recon_subj0${s}.log"
    exit 1
  fi
done
log "stage 1: all reconstructions done"

# --- stage 2: eval (CPU, .venv_kneeland) + relabel into S1K_T3_h0_subj0N.tsv ---
for s in "${SUBJECTS[@]}"; do
  arm="S1K_T3_h0_subj0${s}"
  label="${arm}_s${STRENGTH}_m${MIXING}"
  sweep_out="$SWEEP_DIR/${label}.tsv"
  final_out="data/results/shared1000_comparison/${arm}.tsv"
  if [ -f "$final_out" ]; then
    log "stage 2: $final_out already present, skipping"
    continue
  fi
  log "stage 2: evaluating $label"
  "$KNEELAND_PY" src/evaluate_block_set.py \
      --recon-root "$RECON_ROOT/${label}" --proxy-dir "$BLOCKSET" \
      --label "$label" --gen-rep "$GEN_REP" --device cpu \
      --out "$sweep_out" > "log_h0_eval_subj0${s}.log" 2>&1
  "$KNEELAND_PY" src/pick_oracle_constants.py --arm "$arm" --sweep-dir "$SWEEP_DIR" \
      --out "$final_out" --winner-json "data/results/shared1000_comparison/${arm}_setting_choice.json" \
      >> "log_h0_eval_subj0${s}.log" 2>&1
  log "stage 2: wrote $final_out"
done
log "stage 2: all evals done"

# --- stage 3: the generation null itself; does it now include zero on primary_mean? ---
log "stage 3: generation null (replicas {0,1} vs {2,3})"
"$KNEELAND_PY" src/generation_noise.py

log "=== done ==="
