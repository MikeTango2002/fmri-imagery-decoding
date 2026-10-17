#!/usr/bin/env bash
# End-to-end reconstructions of two ridge variants on the 83 blocks of shared1000: the per-dimension output gain
# (gamma=1, arm H2G1) and the reliability-weighted voxels (arm H4REL).
#
# Outputs: S1K_H2G1_subj0N.tsv and S1K_H4REL_subj0N.tsv under data/results/ridge_input_output/
#
# No control is regenerated: both arms are paired with the S1K_T3_h0_subj0N arm left by
# run_generation_noise_shared1000.sh (the gain with gamma=0 is bit-identical to the final re-standardization).
# The full 83-block set is used, as there: both comparisons are TRISECT against itself; the 65-block
# intersection with Brain-Diffuser is not needed. About 144 s/block, 13 h per arm for 4 subjects on one GPU.
# Resumable: reconstruct.py skips every PNG already on disk.
#
# Usage: setsid nohup bash src/run_gain_and_reliability_reconstructions.sh > log_gain_reliability.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # project root

TARGETS_PY=.venv_targets/bin/python
KNEELAND_PY=.venv_kneeland/bin/python

BLOCKSET=data/shared1000_blockset
RECON_ROOT=data/nsdcore_recons_b3
SWEEP_DIR=data/results/ridge_input_output/setting_sweep
GEN_REP=4
STRENGTH=0.85
MIXING=0.6            # same setting as S1K_T3(_h0): only the input predictions differ
SUBJECTS=(1 2 5 7)
MAX_RETRIES=5

log() { echo "[$(date +%H:%M:%S)] $*"; }

reconstruct_arm() {
  local tag="$1" pred_dir="$2"
  for s in "${SUBJECTS[@]}"; do
    local label="${tag}_subj0${s}_s${STRENGTH}_m${MIXING}"
    local outdir="$RECON_ROOT/${label}"
    if [ -f "$outdir/.recon_done" ]; then
      log "$tag: $label already done, skipping"
      continue
    fi
    local attempt=1
    while [ $attempt -le $MAX_RETRIES ]; do
      log "$tag: reconstructing $label (83 blocks x 12 stim x $GEN_REP reps, attempt $attempt/$MAX_RETRIES)"
      if timeout 6h "$TARGETS_PY" src/reconstruct.py \
          --dataset nsdcore --harness-blocks --blockset-dir "$BLOCKSET" --subj "$s" \
          --pred-dir "$pred_dir" \
          --gen-rep "$GEN_REP" --strength "$STRENGTH" --mixing "$MIXING" \
          --out-dir "$outdir" >> "log_${tag,,}_recon_subj0${s}.log" 2>&1; then
        touch "$outdir/.recon_done"
        log "$tag: $label done"
        break
      fi
      log "$tag: $label attempt $attempt FAILED (see log_${tag,,}_recon_subj0${s}.log -- likely "
      log "  the known WSL2/dxgkrnl CUDA flakiness documented in run_generation_noise_shared1000.sh, not a code bug) -- retrying after 10s"
      attempt=$((attempt + 1))
      sleep 10
    done
    if [ ! -f "$outdir/.recon_done" ]; then
      log "$tag: $label FAILED after $MAX_RETRIES attempts -- giving up"
      exit 1
    fi
  done
}

eval_arm() {
  local tag="$1"
  for s in "${SUBJECTS[@]}"; do
    local arm="${tag}_subj0${s}"
    local label="${arm}_s${STRENGTH}_m${MIXING}"
    local sweep_out="$SWEEP_DIR/${label}.tsv"
    local final_out="data/results/ridge_input_output/${arm}.tsv"
    if [ -f "$final_out" ]; then
      log "$tag eval: $final_out already present, skipping"
      continue
    fi
    log "$tag eval: evaluating $label"
    "$KNEELAND_PY" src/evaluate_block_set.py \
        --recon-root "$RECON_ROOT/${label}" --proxy-dir "$BLOCKSET" \
        --label "$label" --gen-rep "$GEN_REP" --device cpu \
        --out "$sweep_out" > "log_${tag,,}_eval_subj0${s}.log" 2>&1
    "$KNEELAND_PY" src/pick_oracle_constants.py --arm "$arm" --sweep-dir "$SWEEP_DIR" \
        --out "$final_out" --winner-json "data/results/ridge_input_output/${arm}_setting_choice.json" \
        >> "log_${tag,,}_eval_subj0${s}.log" 2>&1
    log "$tag eval: wrote $final_out"
  done
}

log "=== per-dimension gain: gamma=1 (a_k propto r_k^1), shared1000, 83 blocks x 4 subjects ==="
reconstruct_arm S1K_H2G1 data/betas_trisect/nsdcore_predictions_h2g1
eval_arm S1K_H2G1
log "per-dimension gain: end-to-end gate"
"$KNEELAND_PY" src/compare_dimension_gain.py

log "=== reliability weighting: reliability-weighted X, shared1000, 83 blocks x 4 subjects ==="
reconstruct_arm S1K_H4REL data/betas_trisect/nsdcore_predictions_h4rel
eval_arm S1K_H4REL
log "reliability weighting: end-to-end gate"
"$KNEELAND_PY" src/compare_voxel_reliability_weighting.py

log "=== run_gain_and_reliability_reconstructions.sh done ==="
