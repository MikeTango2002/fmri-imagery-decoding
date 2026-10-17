#!/usr/bin/env bash
# MLP encoder (delta_ctr=0.2, seed 0) against the ridge, end to end through the harness (Section 8.1.1).
#
# Inputs:  ridge and MLP predictions (predict_ridge_*.py, predict_mlp.py), the shared1000 block set
# Outputs: data/results/decoder_alternatives/mlp/{ridge_imagery,mlp_imagery,mlp_s1k_subj0N}.tsv
#
# Both arms use the same generator constants, strength 0.85 and mixing 0.6 (the winner of the oracle
# development sweep). Three legs: ridge on imagery, MLP on imagery, MLP on shared1000 (the expensive
# one); the ridge on shared1000 is oracle_budget/S1K_T3_subj0N.tsv, made with the same constants. The
# imagery legs of this script are not the ones of Table 8.2: run_mlp_vs_ridge_imagery_trials.sh makes them.
#
# Usage: setsid nohup ./src/run_mlp_vs_ridge.sh > run_mlp_vs_ridge.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # project root (the folder that holds data/)

TARGETS_PY=.venv_targets/bin/python
KNEELAND_PY=.venv_kneeland/bin/python

STRENGTH=0.85
MIXING=0.6
SUBJECTS=(1 2 5 7)
RECON_ROOT=data/d2_3_recons
EVAL_DIR=data/results/decoder_alternatives/mlp
GT18=data/nsddata_stimuli/stimuli/imagery_stimuli_18.pt
BLOCKSET=data/shared1000_blockset
S1K_GEN_REP=4

log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$EVAL_DIR" "$RECON_ROOT"

# --- preflight ---
for f in "$GT18" "$BLOCKSET/manifest.json" data/mlp_encoder_checkpoints/delta_ctr_0.2/model_best.pt \
         data/betas_trisect/imagery_predictions/subj01/z_low_pred.npy \
         data/betas_trisect/mlp_predictions_imagery/subj01/z_low_pred.npy \
         data/betas_trisect/mlp_predictions_shared1000/subj01/z_low_pred.npy; do
  [ -f "$f" ] || { log "PREFLIGHT FAILED: missing $f"; exit 1; }
done
log "preflight OK"

# ============================================================================
# Leg 1+2: real NSD-Imagery, ridge and MLP, same constants
# ============================================================================
imagery_leg() {
  local label="$1" pred_dir="$2"
  local outdir="$RECON_ROOT/${label}"
  if [ ! -f "$outdir/.recon_done" ]; then
    for subj in "${SUBJECTS[@]}"; do
      log "  $label: reconstructing subj0${subj}"
      "$TARGETS_PY" src/reconstruct.py --dataset imagery --subj "$subj" \
        --gen-rep 10 --strength "$STRENGTH" --mixing "$MIXING" --pred-dir "$pred_dir" \
        --out-dir "$outdir" > "log_d2_3_recon_${label}_subj0${subj}.log" 2>&1
    done
    touch "$outdir/.recon_done"
  else
    log "  $label: reconstructions already done, skipping"
  fi

  local out="$EVAL_DIR/${label}.tsv"
  if [ ! -f "$out" ]; then
    log "  $label: evaluating (final_evaluations_imagery.py)"
    "$KNEELAND_PY" ../mindeye_II/src/final_evaluations_imagery.py --model template \
      --path_template 'subj{subj:02d}/{cond}/sample{sample:03d}/{stim:05d}.png' \
      --recon_root "$outdir" --gt "$GT18" --subj "${SUBJECTS[@]}" --mode vision imagery \
      --gen_rep 10 --device cpu --label "$label" --out "$out" \
      > "log_d2_3_eval_${label}.log" 2>&1
  else
    log "  $label: eval already done, skipping"
  fi
}

log "=== leg 1: ridge-imagery @ s=$STRENGTH m=$MIXING ==="
imagery_leg "ridge_imagery" "data/betas_trisect/imagery_predictions"
log "=== leg 2: mlp-imagery @ s=$STRENGTH m=$MIXING ==="
imagery_leg "mlp_imagery" "data/betas_trisect/mlp_predictions_imagery"

# ============================================================================
# Leg 3: shared1000, MLP only (the ridge reuses oracle_budget/S1K_T3_subj0N.tsv, same constants)
# ============================================================================
log "=== leg 3: mlp-shared1000 @ s=$STRENGTH m=$MIXING (gen_rep=$S1K_GEN_REP) ==="
for subj in "${SUBJECTS[@]}"; do
  label="mlp_s1k_subj0${subj}"
  outdir="$RECON_ROOT/${label}"
  if [ -f "$outdir/.recon_done" ]; then
    log "  $label: reconstructions already done, skipping"
  else
    log "  $label: reconstructing"
    "$TARGETS_PY" src/reconstruct.py --dataset nsdcore --harness-blocks \
      --blockset-dir "$BLOCKSET" --subj "$subj" --gen-rep "$S1K_GEN_REP" \
      --strength "$STRENGTH" --mixing "$MIXING" \
      --pred-dir data/betas_trisect/mlp_predictions_shared1000 --out-dir "$outdir" \
      > "log_d2_3_recon_${label}.log" 2>&1
    touch "$outdir/.recon_done"
  fi

  out="$EVAL_DIR/${label}.tsv"
  if [ -f "$out" ]; then
    log "  $label: eval already done, skipping"
  else
    log "  $label: evaluating (evaluate_block_set.py)"
    "$KNEELAND_PY" src/evaluate_block_set.py --recon-root "$outdir" --proxy-dir "$BLOCKSET" \
      --label "$label" --gen-rep "$S1K_GEN_REP" --device cpu --out "$out" \
      > "log_d2_3_eval_${label}.log" 2>&1
  fi
done

log "=== all legs done -- src/compare_mlp_vs_ridge.py reads $EVAL_DIR/*.tsv + oracle_budget/S1K_T3_subj0N.tsv ==="
