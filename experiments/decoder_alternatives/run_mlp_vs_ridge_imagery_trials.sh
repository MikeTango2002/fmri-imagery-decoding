#!/usr/bin/env bash
# The two imagery legs of the MLP-vs-ridge comparison, repeated with --trial-level (Section 8.1.1).
#
# Inputs:  trial-level ridge predictions (made in step 1) and MLP predictions (mlp_predictions_imagery_trial)
# Outputs: data/results/decoder_alternatives/mlp/{ridge,mlp}_imagery_trial.tsv
#
# Why: the MLP is trained on single trials (std ~1), and repetition-averaged betas (std ~0.30) shrink its
# input, which alone lowers cos(z_sem) from 0.38 to 0.21 (see README). --trial-level averages the
# predictions instead of the betas. The ridge is run again as control: it is affine, and step 2 checks that
# its predictions did not move. The shared1000 leg is not affected and is not rerun. One GPU is enough.
#
# Usage: setsid nohup ./src/run_mlp_vs_ridge_imagery_trials.sh > trials.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # project root (the folder that holds data/)

export OPENBLAS_CORETYPE=SkylakeX          # some CPUs miscompute large matmuls without it
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TARGETS_PY=.venv_targets/bin/python
KNEELAND_PY=.venv_kneeland/bin/python

STRENGTH=0.85
MIXING=0.6
SUBJECTS=(1 2 5 7)
GEN_REP=10
RECON_ROOT=data/d2_3_recons
EVAL_DIR=data/results/decoder_alternatives/mlp
GT18=data/nsddata_stimuli/stimuli/imagery_stimuli_18.pt
RIDGE_PRED=data/betas_trisect/imagery_predictions_trial
MLP_PRED=data/betas_trisect/mlp_predictions_imagery_trial

log() { echo "[$(date +%F' '%H:%M:%S)] $*"; }
mkdir -p "$EVAL_DIR" "$RECON_ROOT"

# ---------------------------------------------------------------- 0. preflight
for f in "$GT18" data/targets/a_psi_checkpoint.pt data/targets/pca_zlow/pca_basis.npz \
         data/betas_trisect/ridge_weights/subj01/z_low_weights.pkl \
         data/betas_trisect/nsd_imagery/subj01/betas_flat.npy \
         "$MLP_PRED/subj01/z_low_pred.npy" "$MLP_PRED/subj07/z_low_pred.npy"; do
  [ -f "$f" ] || { log "PREFLIGHT FAILED: missing $f"; exit 1; }
done
log "preflight OK (GPU $CUDA_VISIBLE_DEVICES)"

# ------------------------------------- 1. ridge predictions, trial-level (CPU, seconds)
# The MLP predictions are produced on the machine that holds the 1.2 GB checkpoint and copied in.
if [ ! -f "$RIDGE_PRED/subj07/z_low_pred.npy" ]; then
  log "=== step 1: ridge trial-level predictions ==="
  "$TARGETS_PY" src/predict_ridge_imagery.py --subj "${SUBJECTS[@]}" --trial-level \
    --out-dir "$RIDGE_PRED"
else
  log "step 1: ridge trial-level predictions already on disk, skipping"
fi

# ------------------------------------- 2. the control: the ridge must NOT have moved
log "=== step 2: affine-invariance control (ridge trial-level vs production) ==="
"$TARGETS_PY" - <<'PYEOF'
import sys
import numpy as np
from pathlib import Path

BRANCHES = ("z_sem_img", "z_sem_txt", "z_struct", "z_low")
old_root, new_root = Path("data/betas_trisect/imagery_predictions"), Path("data/betas_trisect/imagery_predictions_trial")
worst = 0.0
for subj in (1, 2, 5, 7):
    for b in BRANCHES:
        o = np.load(old_root / f"subj{subj:02d}" / f"{b}_pred.npy").astype(np.float64)
        n = np.load(new_root / f"subj{subj:02d}" / f"{b}_pred.npy").astype(np.float64)
        assert o.shape == n.shape, f"subj{subj} {b}: {o.shape} vs {n.shape}"
        rel = np.abs(o - n).max() / max(np.abs(o).max(), 1e-12)
        worst = max(worst, rel)
        print(f"  subj{subj:02d} {b:<10} max relative diff {rel:.3e}")
print(f"\n  worst relative diff across all 16 arrays: {worst:.3e}")
if worst > 1e-4:
    sys.exit("CONTROL FAILED: the ridge moved -- averaging is supposed to commute with an "
             "affine decoder, so --trial-level (or average_by_label's row order) is wrong. "
             "Do NOT interpret the MLP arm until this is understood.")
print("  CONTROL PASSED: the ridge is invariant to the switch, as an affine decoder must be.")
PYEOF

# ------------------------------------- 3. reconstructions, both arms, parallel on one GPU
recon_leg() {
  local label="$1" pred_dir="$2"
  local outdir="$RECON_ROOT/${label}"
  if [ -f "$outdir/.recon_done" ]; then
    log "  $label: reconstructions already done, skipping"
    return 0
  fi
  for subj in "${SUBJECTS[@]}"; do
    log "  $label: reconstructing subj0${subj}"
    "$TARGETS_PY" src/reconstruct.py --dataset imagery --subj "$subj" \
      --gen-rep "$GEN_REP" --strength "$STRENGTH" --mixing "$MIXING" --pred-dir "$pred_dir" \
      --out-dir "$outdir" > "log_d2_3_fix_recon_${label}_subj0${subj}.log" 2>&1
  done
  touch "$outdir/.recon_done"
  log "  $label: reconstructions done"
}

log "=== step 3: reconstructions (both arms in parallel on GPU $CUDA_VISIBLE_DEVICES) ==="
recon_leg "ridge_imagery_trial" "$RIDGE_PRED" &
PID_RIDGE=$!
recon_leg "mlp_imagery_trial"   "$MLP_PRED" &
PID_MLP=$!
FAILED=0
wait $PID_RIDGE || { log "RIDGE recon leg FAILED"; FAILED=1; }
wait $PID_MLP   || { log "MLP recon leg FAILED";   FAILED=1; }
[ "$FAILED" -eq 0 ] || { log "aborting before eval"; exit 1; }

# ------------------------------------- 4. eval, both arms, parallel on CPU
eval_leg() {
  local label="$1"
  local out="$EVAL_DIR/${label}.tsv"
  if [ -f "$out" ]; then
    log "  $label: eval already done, skipping"
    return 0
  fi
  log "  $label: evaluating (final_evaluations_imagery.py)"
  "$KNEELAND_PY" ../mindeye_II/src/final_evaluations_imagery.py --model template \
    --path_template 'subj{subj:02d}/{cond}/sample{sample:03d}/{stim:05d}.png' \
    --recon_root "$RECON_ROOT/${label}" --gt "$GT18" --subj "${SUBJECTS[@]}" \
    --mode vision imagery --gen_rep "$GEN_REP" --device cpu --label "$label" --out "$out" \
    > "log_d2_3_fix_eval_${label}.log" 2>&1
}

log "=== step 4: eval (both arms in parallel, CPU) ==="
eval_leg "ridge_imagery_trial" &
PID_RIDGE=$!
eval_leg "mlp_imagery_trial" &
PID_MLP=$!
wait $PID_RIDGE || { log "RIDGE eval leg FAILED"; FAILED=1; }
wait $PID_MLP   || { log "MLP eval leg FAILED";   FAILED=1; }
[ "$FAILED" -eq 0 ] || exit 1

log "=== done -- $EVAL_DIR/{ridge,mlp}_imagery_trial.tsv ready for src/compare_mlp_vs_ridge.py ==="
