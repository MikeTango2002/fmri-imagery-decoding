#!/usr/bin/env bash
# Generate and score the hybrid arm (Section 8.1.1): z_sem and z_struct from the MLP, z_low from the ridge.
#
# Inputs:  hybrid prediction folders from build_hybrid_predictions.py (imagery, shared1000, and the control
#          on shared1000 with z_low from the MLP and the rest from the ridge)
# Outputs: data/results/decoder_alternatives/hybrid/*.tsv, reconstructions in data/g4_0_recons
#
# No retraining. Same generator constants as run_mlp_vs_ridge.sh (strength 0.85, mixing 0.6) and the same
# S1K_GEN_REP=4; the shared1000 cells line up with oracle_budget/S1K_T3_subj0N.tsv without rerunning
# it. All three legs run (compute, not analysis); compare_hybrid_vs_ridge.py prints vision first.
#
# Usage: setsid nohup ./src/run_hybrid_vs_ridge.sh > run_hybrid_vs_ridge.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # project root (the folder that holds data/)

export OPENBLAS_CORETYPE=SkylakeX
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TARGETS_PY=.venv_targets/bin/python
KNEELAND_PY=.venv_kneeland/bin/python

STRENGTH=0.85
MIXING=0.6
SUBJECTS=(1 2 5 7)
RECON_ROOT=data/g4_0_recons
EVAL_DIR=data/results/decoder_alternatives/hybrid
GT18=data/nsddata_stimuli/stimuli/imagery_stimuli_18.pt
BLOCKSET=data/shared1000_blockset
S1K_GEN_REP=4

log() { echo "[$(date +%F' '%H:%M:%S)] $*"; }
mkdir -p "$EVAL_DIR" "$RECON_ROOT"

# --- preflight ---
for f in "$GT18" "$BLOCKSET/manifest.json" \
         data/betas_trisect/g4_0_hybrid_imagery/subj01/z_low_pred.npy \
         data/betas_trisect/g4_0_hybrid_shared1000/subj01/z_low_pred.npy \
         data/betas_trisect/g4_0_ctrl_shared1000/subj01/z_low_pred.npy \
         data/results/oracle_budget/S1K_T3_subj01.tsv; do
  [ -f "$f" ] || { log "PREFLIGHT FAILED: missing $f"; exit 1; }
done
log "preflight OK (GPU $CUDA_VISIBLE_DEVICES)"

# ============================================================================
# Leg 1: real NSD-Imagery, hybrid arm (reported, not selected)
# ============================================================================
imagery_leg() {
  local label="$1" pred_dir="$2"
  local outdir="$RECON_ROOT/${label}"
  if [ ! -f "$outdir/.recon_done" ]; then
    for subj in "${SUBJECTS[@]}"; do
      log "  $label: reconstructing subj0${subj}"
      "$TARGETS_PY" src/reconstruct.py --dataset imagery --subj "$subj" \
        --gen-rep 10 --strength "$STRENGTH" --mixing "$MIXING" --pred-dir "$pred_dir" \
        --out-dir "$outdir" > "log_g4_0_recon_${label}_subj0${subj}.log" 2>&1
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
      > "log_g4_0_eval_${label}.log" 2>&1
  else
    log "  $label: eval already done, skipping"
  fi
}

log "=== leg 1: hybrid-imagery (sem=MLP, low=ridge) @ s=$STRENGTH m=$MIXING ==="
imagery_leg "hybrid_imagery" "data/betas_trisect/g4_0_hybrid_imagery"

# ============================================================================
# Legs 2+3: shared1000, hybrid (selection) + symmetric control (vision-only)
# ============================================================================
s1k_leg() {
  local label="$1" pred_dir="$2"
  local outdir="$RECON_ROOT/${label}"
  if [ -f "$outdir/.recon_done" ]; then
    log "  $label: reconstructions already done, skipping"
  else
    log "  $label: reconstructing (harness-blocks, gen_rep=$S1K_GEN_REP)"
    for subj in "${SUBJECTS[@]}"; do
      "$TARGETS_PY" src/reconstruct.py --dataset nsdcore --harness-blocks \
        --blockset-dir "$BLOCKSET" --subj "$subj" --gen-rep "$S1K_GEN_REP" \
        --strength "$STRENGTH" --mixing "$MIXING" --pred-dir "$pred_dir" \
        --out-dir "$outdir/subj0${subj}" > "log_g4_0_recon_${label}_subj0${subj}.log" 2>&1
    done
    touch "$outdir/.recon_done"
  fi

  for subj in "${SUBJECTS[@]}"; do
    local out="$EVAL_DIR/${label}_subj0${subj}.tsv"
    if [ -f "$out" ]; then
      log "  $label subj0${subj}: eval already done, skipping"
    else
      log "  $label subj0${subj}: evaluating (evaluate_block_set.py)"
      "$KNEELAND_PY" src/evaluate_block_set.py --recon-root "$outdir/subj0${subj}" \
        --proxy-dir "$BLOCKSET" --label "${label}_subj0${subj}" --gen-rep "$S1K_GEN_REP" \
        --device cpu --out "$out" > "log_g4_0_eval_${label}_subj0${subj}.log" 2>&1
    fi
  done
}

log "=== leg 2: hybrid-shared1000 (selection) @ s=$STRENGTH m=$MIXING ==="
s1k_leg "hybrid_s1k" "data/betas_trisect/g4_0_hybrid_shared1000"

log "=== leg 3: ctrl-shared1000 (symmetric control) @ s=$STRENGTH m=$MIXING ==="
s1k_leg "ctrl_s1k" "data/betas_trisect/g4_0_ctrl_shared1000"

log "=== all legs done -- src/compare_hybrid_vs_ridge.py reads $EVAL_DIR/*.tsv + oracle_budget/S1K_T3_subj0N.tsv + decoder_alternatives/mlp/ridge_imagery_trial.tsv ==="
