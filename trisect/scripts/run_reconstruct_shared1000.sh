#!/usr/bin/env bash
# Generate TRISECT's shared1000 arm (S1K_T3) at the generator setting of Brain-Diffuser, strength 0.75 /
# mixing 0.4, so that the paired comparison with Brain-Diffuser (Section 6.4) runs at matched generation
# constants.
#
# Stages: 0 predictions (CPU), 1 reconstruction (two lanes of two subjects, one per GPU, ~5-7 h),
# 2 evaluation (CPU, .venv_kneeland), 3 comparison with Brain-Diffuser.
# Everything goes to *_p3 names, next to the 0.85/0.6 arm of the oracle budget.
# Stage 0 writes its own prediction dir, with the repetitions averaged before the re-standardization
# (`predict_ridge_shared1000.py` must be that version).
# Resumable: every stage skips its finished output.
#
# Usage (from the trisect/ folder):
#   setsid nohup bash scripts/run_reconstruct_shared1000.sh > log_p3.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # trisect/

export OPENBLAS_CORETYPE=SkylakeX   # Sapphire Rapids CPUs miscompute without it

TARGETS_PY=.venv_targets/bin/python
KNEELAND_PY=.venv_kneeland/bin/python

BLOCKSET=data/shared1000_blockset
RECON_ROOT=data/nsdcore_recons_b3
SWEEP_DIR=data/results/trisect/shared1000/setting_sweep
PRED_DIR=data/betas_trisect/nsdcore_predictions_h0
GEN_REP=4
STRENGTH=0.75         # the paper's operating point, the whole point of this run
MIXING=0.4            # ditto; Brain-Diffuser's own constants, which make the comparison paired
SUBJECTS=(1 2 5 7)
LANE0=(1 2)           # GPU 0
LANE1=(5 7)           # GPU 1
MAX_RETRIES=3

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== S1K_T3 at strength=$STRENGTH mixing=$MIXING (paired with Brain-Diffuser) ==="

# --- stage 0: shared1000 predictions with the averaging-first order, into their own dir (CPU, minutes) ---
for s in "${SUBJECTS[@]}"; do
  if [ -d "$PRED_DIR/subj0${s}" ]; then
    log "stage 0: $PRED_DIR/subj0${s} present, skipping"
    continue
  fi
  log "stage 0: predict_ridge_shared1000.py --subj $s --out-dir $PRED_DIR"
  "$TARGETS_PY" src/predict_ridge_shared1000.py --subj "$s" --out-dir "$PRED_DIR" \
    > "log_p3_predict_subj0${s}.log" 2>&1
done
log "stage 0: done"

# --- stage 1: reconstruct, two subjects per GPU, the two lanes in parallel ---
reconstruct_subject() {
  local s="$1" gpu="$2"
  local label="S1K_T3_p3_subj0${s}_s${STRENGTH}_m${MIXING}"
  local outdir="$RECON_ROOT/${label}"
  if [ -f "$outdir/.recon_done" ]; then
    log "stage 1 [gpu$gpu]: $label already done, skipping"
    return 0
  fi
  local attempt=1
  while [ $attempt -le $MAX_RETRIES ]; do
    log "stage 1 [gpu$gpu]: $label (83 blocks x 12 stim x $GEN_REP reps, attempt $attempt/$MAX_RETRIES)"
    if CUDA_VISIBLE_DEVICES="$gpu" timeout 10h "$TARGETS_PY" src/reconstruct.py \
        --dataset nsdcore --harness-blocks --blockset-dir "$BLOCKSET" --subj "$s" \
        --pred-dir "$PRED_DIR" --gen-rep "$GEN_REP" \
        --strength "$STRENGTH" --mixing "$MIXING" \
        --out-dir "$outdir" >> "log_p3_recon_subj0${s}.log" 2>&1; then
      touch "$outdir/.recon_done"
      log "stage 1 [gpu$gpu]: $label done"
      return 0
    fi
    log "stage 1 [gpu$gpu]: $label attempt $attempt FAILED (see log_p3_recon_subj0${s}.log) -- retrying in 10s"
    attempt=$((attempt + 1))
    sleep 10
  done
  log "stage 1 [gpu$gpu]: $label FAILED after $MAX_RETRIES attempts"
  return 1
}

run_lane() {
  local gpu="$1"; shift
  for s in "$@"; do
    reconstruct_subject "$s" "$gpu" || return 1
  done
}

run_lane 0 "${LANE0[@]}" &
pid0=$!
run_lane 1 "${LANE1[@]}" &
pid1=$!
fail=0
wait $pid0 || fail=1
wait $pid1 || fail=1
if [ $fail -ne 0 ]; then
  log "stage 1: at least one lane failed -- stopping before the eval so nothing partial is scored"
  exit 1
fi
log "stage 1: all reconstructions done"

# --- stage 2: eval (CPU, .venv_kneeland) + relabel into S1K_T3_p3_subj0N.tsv ---
for s in "${SUBJECTS[@]}"; do
  arm="S1K_T3_p3_subj0${s}"
  label="${arm}_s${STRENGTH}_m${MIXING}"
  sweep_out="$SWEEP_DIR/${label}.tsv"
  final_out="data/results/trisect/shared1000/${arm}.tsv"
  if [ -f "$final_out" ]; then
    log "stage 2: $final_out already present, skipping"
    continue
  fi
  log "stage 2: evaluating $label"
  "$KNEELAND_PY" src/evaluate_block_set.py \
      --recon-root "$RECON_ROOT/${label}" --proxy-dir "$BLOCKSET" \
      --label "$label" --gen-rep "$GEN_REP" --device cpu \
      --out "$sweep_out" > "log_p3_eval_subj0${s}.log" 2>&1
  "$KNEELAND_PY" src/pick_oracle_constants.py --arm "$arm" --sweep-dir "$SWEEP_DIR" \
      --out "$final_out" --winner-json "data/results/trisect/shared1000/${arm}_setting_choice.json" \
      >> "log_p3_eval_subj0${s}.log" 2>&1
  log "stage 2: wrote $final_out"
done
log "stage 2: all evals done"

# --- stage 3: comparison with Brain-Diffuser at paired constants, on the 65 BD-intersected bridge blocks ---
log "stage 3: comparison, TRISECT(0.75/0.4) vs Brain-Diffuser(0.75/0.4)"
"$KNEELAND_PY" src/compare_trisect_vs_bd_shared1000.py \
    --trisect-prefix trisect/shared1000/S1K_T3_p3_subj \
    --trisect-label "TRISECT T3 s=$STRENGTH m=$MIXING (paired)" \
    --out data/results/shared1000_comparison/trisect_vs_bd_S1K_T3_p3.tsv \
    --out-null data/results/shared1000_comparison/generation_null_S1K_T3_p3.tsv \
    --out-json data/results/shared1000_comparison/trisect_vs_bd_S1K_T3_p3_summary.json

log "=== done ==="
