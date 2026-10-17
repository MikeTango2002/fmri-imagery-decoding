#!/usr/bin/env bash
# Uncompressed low-level target (bfull) against the compressed one, on NSD-Imagery (Section 8.1.3).
#
# Inputs:  the weights and predictions of fit_ridge_uncompressed_zlow.py, the production predictions
# Outputs: data/results/low_level_factor/uncompressed_zlow/: bfull_imagery.tsv, d1024refit_imagery.tsv,
#          comparison_summary.tsv
#
# Steps: ridge fit (CPU, one eigendecomposition per stage for both branches), predict, assemble merged
# prediction folders, reconstruct (GPU), evaluation, compare_uncompressed_zlow.py. The generator constants
# are those of the final model (DIFF_STRENGTH_DEFAULT=0.75, MIX_DEFAULT=0.4) and are not overridden; the
# only change from the final model is the z_low ridge target and alpha procedure.
#
# Usage: setsid nohup ./run_uncompressed_zlow.sh > log_uncompressed_zlow.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # the project root, the folder that holds data/

export OPENBLAS_CORETYPE=SkylakeX          # some CPUs miscompute large matmuls without it
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TARGETS_PY=.venv_targets/bin/python
KNEELAND_PY=.venv_kneeland/bin/python

SUBJECTS=(1 2 5 7)
GEN_REP=10
GT18=data/nsddata_stimuli/stimuli/imagery_stimuli_18.pt

WEIGHTS_D1024=data/betas_trisect/ridge_weights_g2_1_d1024refit
WEIGHTS_BFULL=data/betas_trisect/ridge_weights_g2_1_bfull
PRED_D1024=data/betas_trisect/imagery_predictions_g2_1_d1024refit
PRED_BFULL=data/betas_trisect/imagery_predictions_g2_1_bfull
PROD_PRED=data/betas_trisect/imagery_predictions   # untouched production: z_sem_img/z_sem_txt/z_struct
RECON_ROOT=data/imagery_recons_g2_1
EVAL_DIR=data/results/low_level_factor/uncompressed_zlow
V0_SUBJ_TSVS=data/results/low_level_factor/scalar_restandardization/v0repro_subj
V0_ALL_TSV=data/results/low_level_factor/scalar_restandardization/v0repro_all.tsv

log() { echo "[$(date +%F' '%H:%M:%S)] $*"; }
mkdir -p "$EVAL_DIR" "$RECON_ROOT"

# ---------------------------------------------------------------- 0. preflight
for f in "$GT18" data/targets/a_psi_checkpoint.pt data/targets/pca_zlow/pca_basis.npz \
         data/targets/vdvae_latents_31l.npy data/targets/vdvae_latents_31l_layer_dims.json \
         data/targets/pca_zlow/z_low.npy \
         data/targets/nsdcore_dev_mask_subj01.npy data/targets/nsdcore_dev_mask_subj07.npy \
         "$PROD_PRED/subj01/z_sem_img_pred.npy" "$PROD_PRED/subj07/z_struct_pred.npy" \
         data/betas_trisect/nsd_imagery/subj01/betas_averaged.npy; do
  [ -f "$f" ] || { log "PREFLIGHT FAILED: missing $f"; exit 1; }
done
log "preflight OK (GPU $CUDA_VISIBLE_DEVICES)"

# ------------------------------------- 1. ridge fit: shared eigendecomposition, both branches
if [ -f "$WEIGHTS_BFULL/subj07/z_low_weights.pkl" ] && [ -f "$WEIGHTS_D1024/subj07/z_low_weights.pkl" ]; then
  log "step 1: ridge weights already on disk for all subjects, skipping"
else
  log "=== step 1: fit_ridge_uncompressed_zlow.py (CPU, ~1h/subject x 4, sequential) ==="
  "$TARGETS_PY" src/fit_ridge_uncompressed_zlow.py --subj "${SUBJECTS[@]}" \
    --out-dir-d1024 "$WEIGHTS_D1024" --out-dir-bfull "$WEIGHTS_BFULL"
fi

# ------------------------------------- 2. predict z_low, both arms (CPU, seconds)
predict_arm() {
  local weights_dir="$1" pred_dir="$2" label="$3"
  if [ -f "$pred_dir/subj07/z_low_pred.npy" ]; then
    log "  $label: z_low predictions already on disk, skipping"
    return 0
  fi
  log "  $label: predicting z_low"
  "$TARGETS_PY" src/predict_ridge_imagery.py --subj "${SUBJECTS[@]}" --branches z_low \
    --weights-dir "$weights_dir" --out-dir "$pred_dir"
}
log "=== step 2: predict z_low (both arms) ==="
predict_arm "$WEIGHTS_D1024" "$PRED_D1024" "d1024-refit"
predict_arm "$WEIGHTS_BFULL" "$PRED_BFULL" "b-full"

# ------------------------------------- 3. assemble: symlink the 3 untouched branches in
assemble_arm() {
  local pred_dir="$1" label="$2"
  for subj in "${SUBJECTS[@]}"; do
    local d
    d=$(printf "%s/subj%02d" "$pred_dir" "$subj")
    for b in z_sem_img z_sem_txt z_struct; do
      [ -e "$d/${b}_pred.npy" ] || ln -s "$(realpath "$PROD_PRED/subj$(printf '%02d' "$subj")/${b}_pred.npy")" "$d/${b}_pred.npy"
    done
  done
  log "  $label: assembled (z_low own, other 3 branches symlinked from production)"
}
log "=== step 3: assemble merged prediction dirs ==="
assemble_arm "$PRED_D1024" "d1024-refit"
assemble_arm "$PRED_BFULL" "b-full"

# ------------------------------------- 4. reconstruct, both arms, parallel on GPU 0
recon_leg() {
  local label="$1" pred_dir="$2" zlow_mode="$3"
  local outdir="$RECON_ROOT/${label}"
  if [ -f "$outdir/.recon_done" ]; then
    log "  $label: reconstructions already done, skipping"
    return 0
  fi
  for subj in "${SUBJECTS[@]}"; do
    log "  $label: reconstructing subj0${subj} (--zlow-mode $zlow_mode)"
    "$TARGETS_PY" src/reconstruct.py --dataset imagery --subj "$subj" \
      --gen-rep "$GEN_REP" --pred-dir "$pred_dir" --out-dir "$outdir" \
      --zlow-mode "$zlow_mode" > "log_g2_1_recon_${label}_subj0${subj}.log" 2>&1
  done
  touch "$outdir/.recon_done"
  log "  $label: reconstructions done"
}

log "=== step 4: reconstructions (both arms in parallel on GPU $CUDA_VISIBLE_DEVICES) ==="
recon_leg "d1024refit" "$PRED_D1024" "vdvae" &
PID_D1024=$!
recon_leg "bfull" "$PRED_BFULL" "raw" &
PID_BFULL=$!
FAILED=0
wait $PID_D1024 || { log "d1024-refit recon leg FAILED"; FAILED=1; }
wait $PID_BFULL || { log "b-full recon leg FAILED"; FAILED=1; }
[ "$FAILED" -eq 0 ] || { log "aborting before eval"; exit 1; }

# ------------------------------------- 5. eval, both arms, parallel on CPU
eval_leg() {
  local label="$1"
  local out="$EVAL_DIR/${label}_imagery.tsv"
  if [ -f "$out" ]; then
    log "  $label: eval already done, skipping"
    return 0
  fi
  log "  $label: evaluating (final_evaluations_imagery.py)"
  "$KNEELAND_PY" ../mindeye_II/src/final_evaluations_imagery.py --model template \
    --path_template 'subj{subj:02d}/{cond}/sample{sample:03d}/{stim:05d}.png' \
    --recon_root "$RECON_ROOT/${label}" --gt "$GT18" --subj "${SUBJECTS[@]}" \
    --mode vision imagery --gen_rep "$GEN_REP" --device cpu --label "$label" --out "$out" \
    > "log_g2_1_eval_${label}.log" 2>&1
}

log "=== step 5: eval (both arms in parallel, CPU) ==="
eval_leg "d1024refit" &
PID_D1024=$!
eval_leg "bfull" &
PID_BFULL=$!
wait $PID_D1024 || { log "d1024-refit eval leg FAILED"; FAILED=1; }
wait $PID_BFULL || { log "b-full eval leg FAILED"; FAILED=1; }
[ "$FAILED" -eq 0 ] || exit 1

# ------------------------------------- 6. concat v0's per-subject TSVs, for the informational row
if [ ! -f "$V0_ALL_TSV" ]; then
  log "=== step 6: concatenating v0's per-subject TSVs for the informational vs-v0 row ==="
  first=1
  : > "$V0_ALL_TSV"
  for subj in "${SUBJECTS[@]}"; do
    f=$(printf "%s%02d.tsv" "$V0_SUBJ_TSVS" "$subj")
    if [ ! -f "$f" ]; then
      log "  WARNING: missing $f -- vs-v0 row will be skipped"
      rm -f "$V0_ALL_TSV"
      break
    fi
    if [ "$first" -eq 1 ]; then
      cat "$f" >> "$V0_ALL_TSV"
      first=0
    else
      tail -n +2 "$f" >> "$V0_ALL_TSV"
    fi
  done
fi

# ------------------------------------- 7. the pre-registered gate
log "=== step 7: gate (bootstrap_ci_imagery.py machinery, stimulus-cluster CI, n=12) ==="
GATE_ARGS=(--bfull "$EVAL_DIR/bfull_imagery.tsv" --d1024refit "$EVAL_DIR/d1024refit_imagery.tsv" \
  --out "$EVAL_DIR/comparison_summary.tsv")
[ -f "$V0_ALL_TSV" ] || GATE_ARGS+=(--v0 "")
[ -f data/results/baselines/brain_diffuser.tsv ] || GATE_ARGS+=(--bd "")
set +e
"$TARGETS_PY" src/compare_uncompressed_zlow.py "${GATE_ARGS[@]}"
GATE_STATUS=$?
set -e

log "=== done -- gate exit status $GATE_STATUS (0=PASS, 1=FAIL). See $EVAL_DIR/comparison_summary.tsv ==="
exit 0
