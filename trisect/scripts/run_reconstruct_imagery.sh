#!/usr/bin/env bash
# Reconstruct the NSD-Imagery arm of the final model (the row reported in Chapter 7).
#
# Model: ridge on z_sem / z_struct / z_low, the adapter trained on images only, frozen VD dual-guided
# generation at strength 0.75 / mixing 0.4 (Brain-Diffuser's constants), 10 replicas per stimulus.
# Reads data/betas_trisect/imagery_predictions (`predict_ridge_imagery.py`); writes data/imagery_recons_b3_h0/.
# Each (stimulus, replica) has its own diffusion seed, `diffusion_seed()` in reconstruct.py; no two
# stimuli share the initial noise.
#
# Two lanes of two subjects, one lane pinned per GPU (CUDA_VISIBLE_DEVICES), not DDP: about 30-60
# minutes on two GPUs. Resumable: reconstruct.py skips every PNG already on disk.
#
# Usage (from the trisect/ folder):  bash scripts/run_reconstruct_imagery.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export OPENBLAS_CORETYPE=SkylakeX

TARGETS_PY=.venv_targets/bin/python
OUT_ROOT=data/imagery_recons_b3_h0
GEN_REP=10
LANE0=(1 2)
LANE1=(5 7)
MAX_RETRIES=3

log() { echo "[$(date +%H:%M:%S)] [reconstruct-imagery] $*"; }

reconstruct_subject() {
  local s="$1" gpu="$2"
  if [ -f "$OUT_ROOT/.recon_done_subj0${s}" ]; then
    log "gpu$gpu: subj0$s already done, skipping"
    return 0
  fi
  local attempt=1
  while [ $attempt -le $MAX_RETRIES ]; do
    log "gpu$gpu: subj0$s (5 conditions x 6 stimuli x $GEN_REP reps, attempt $attempt/$MAX_RETRIES)"
    if CUDA_VISIBLE_DEVICES="$gpu" timeout 4h "$TARGETS_PY" src/reconstruct.py \
        --subj "$s" --gen-rep "$GEN_REP" --out-dir "$OUT_ROOT" \
        >> "log_p5_recon_subj0${s}.log" 2>&1; then
      touch "$OUT_ROOT/.recon_done_subj0${s}"
      log "gpu$gpu: subj0$s done"
      return 0
    fi
    log "gpu$gpu: subj0$s attempt $attempt FAILED (see log_p5_recon_subj0${s}.log) -- retrying in 10s"
    attempt=$((attempt + 1))
    sleep 10
  done
  log "gpu$gpu: subj0$s FAILED after $MAX_RETRIES attempts"
  return 1
}

run_lane() {
  local gpu="$1"; shift
  for s in "$@"; do reconstruct_subject "$s" "$gpu" || return 1; done
}

log "=== reconstructing the NSD-Imagery arm, s=0.75 m=0.4 (defaults) ==="
mkdir -p "$OUT_ROOT"
run_lane 0 "${LANE0[@]}" & pid0=$!
run_lane 1 "${LANE1[@]}" & pid1=$!
fail=0
wait $pid0 || fail=1
wait $pid1 || fail=1
[ $fail -eq 0 ] || { log "a lane failed -- not marking the arm complete"; exit 1; }

n=$(find "$OUT_ROOT" -name '*.png' | wc -l)
log "=== done: $n PNG under $OUT_ROOT (expected 1200) ==="
[ "$n" -eq 1200 ] || log "WARNING: expected 1200 PNG, found $n"
