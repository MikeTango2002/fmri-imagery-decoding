#!/bin/bash
# Pixel low-level target on NSD-Imagery (Section 8.1.3): predictions, reconstruction, 8-metric evaluation.
#
# Inputs:  ridge_weights_check_pixel16 (ridge on the pixel target, see README), production weights otherwise
# Outputs: imagery_predictions_pixel16, imagery_recons_b3_pixel16, data/results/low_level_factor/pixel_target/
#
# Per subject (1 5 7 2, GPU 1): 1) predict_ridge_imagery.py (z_low from the pixel weights, the rest from the
# production ones); 2) reconstruct.py --zlow-mode pixel --pixel-res 16 (30 rows x 10 repetitions); 3)
# final_evaluations_imagery.py --model template, the harness of the production run. A subject with a .done
# sentinel is skipped; finished steps are skipped and failed ones retried.
#
# Usage (setsid nohup so that it survives a disconnect):
#   setsid nohup ./run_pixel_low_target_imagery.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"   # absolute, so that the cd of the eval step into ../mindeye_II/src cannot throw off relative paths
export OPENBLAS_CORETYPE=SkylakeX   # some CPUs miscompute large matmuls without it

SUBJECTS=(1 5 7 2)
GPU=1
RIDGE_PIXEL_DIR="data/betas_trisect/ridge_weights_check_pixel16"
PRED_PIXEL_DIR="data/betas_trisect/imagery_predictions_pixel16"
RECON_PIXEL_DIR="data/imagery_recons_b3_pixel16"
RESULTS_DIR="data/results/low_level_factor/pixel_target"
GT_STIMULI="${PROJECT_ROOT}/../mindeye_II/src/imagery_dataset/nsddata_stimuli/stimuli/imagery_stimuli_18.pt"
LOG_DIR="logs/pixel16_imagery_b4_server_gpu1"
MASTER_LOG="$LOG_DIR/PIXEL16_IMAGERY_B4_DRIVER.log"
MAX_RETRIES=3
mkdir -p "$LOG_DIR"

log_master() { echo "$*" >> "$MASTER_LOG"; }

run_py_step() {
    local name="$1" venv="$2" gpu_pin="$3" workdir="$4"; shift 4
    local logfile="$LOG_DIR/log_${name}.log"
    log_master "=== [$name] START $(date -Iseconds) ==="
    log_master "cmd: (cd $workdir && CUDA_VISIBLE_DEVICES=$gpu_pin python $*)"
    local t0 t1 pid exit_code
    t0=$(date +%s)
    ( cd "$workdir" && export CUDA_VISIBLE_DEVICES="$gpu_pin"; source "$venv" && python "$@" ) > "$logfile" 2>&1 &
    pid=$!
    log_master "pid=$pid log=$logfile"
    wait "$pid"
    exit_code=$?
    t1=$(date +%s)
    log_master "=== [$name] END $(date -Iseconds) exit=$exit_code duration=$((t1 - t0))s ==="
    log_master ""
    return $exit_code
}

run_py_step_retry() {
    local name="$1"; shift
    local attempt=1 exit_code=1
    while [ $attempt -le $MAX_RETRIES ]; do
        if [ $attempt -gt 1 ]; then
            log_master "[$name] retry attempt $attempt/$MAX_RETRIES after failure, sleeping 30s"
            sleep 30
        fi
        run_py_step "${name}_try${attempt}" "$@"
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            break
        fi
        attempt=$((attempt + 1))
    done
    return $exit_code
}

ensure_predictions() {
    local subj="$1" pad="$2"
    local pdir="${PRED_PIXEL_DIR}/subj${pad}"
    if [ -f "${pdir}/z_low_pred.npy" ] && [ -f "${pdir}/z_sem_img_pred.npy" ] \
        && [ -f "${pdir}/z_struct_pred.npy" ]; then
        log_master "=== [predict_pixel16_imagery_subj${pad}] SKIP already-done $(date -Iseconds) ==="
        log_master ""
        return 0
    fi
    run_py_step "predict_pixel16_imagery_subj${pad}_sem_struct" .venv_targets/bin/activate "" . \
        src/predict_ridge_imagery.py --subj "$subj" \
        --branches z_sem_img z_sem_txt z_struct --out-dir "$PRED_PIXEL_DIR" \
        || return 1
    run_py_step "predict_pixel16_imagery_subj${pad}_zlow" .venv_targets/bin/activate "" . \
        src/predict_ridge_imagery.py --subj "$subj" --branches z_low \
        --weights-dir "$RIDGE_PIXEL_DIR" --out-dir "$PRED_PIXEL_DIR"
}

ensure_generation() {
    local subj="$1" pad="$2"
    local n_png
    n_png=$(find "${RECON_PIXEL_DIR}/subj${pad}" -iname "*.png" 2>/dev/null | wc -l)
    if [ "$n_png" -ge 300 ]; then
        log_master "=== [gen_pixel16_imagery_subj${pad}] SKIP already-done ($n_png PNGs) $(date -Iseconds) ==="
        log_master ""
        return 0
    fi
    run_py_step_retry "gen_pixel16_imagery_subj${pad}" .venv_targets/bin/activate "$GPU" . \
        src/reconstruct.py --subj "$subj" --gen-rep 10 \
        --zlow-mode pixel --pixel-res 16 --pred-dir "$PRED_PIXEL_DIR" \
        --out-dir "$RECON_PIXEL_DIR"
}

ensure_eval() {
    local subj="$1" pad="$2"
    local out_abs="${RESULTS_DIR}/pixel16_subj${pad}.tsv"
    local out_rel="${PROJECT_ROOT}/${out_abs}"
    if [ -f "$out_abs" ] && [ "$(wc -l < "$out_abs")" -ge 2 ]; then
        log_master "=== [eval_pixel16_imagery_subj${pad}] SKIP already-done ($out_abs) $(date -Iseconds) ==="
        log_master ""
        return 0
    fi
    run_py_step_retry "eval_pixel16_imagery_subj${pad}" "${PROJECT_ROOT}/.venv_kneeland/bin/activate" "" ../mindeye_II/src \
        final_evaluations_imagery.py \
        --model template --path_template 'subj{subj02}/{cond}/sample{sample03}/{stim05}.png' \
        --recon_root "${PROJECT_ROOT}/${RECON_PIXEL_DIR}" \
        --gt "$GT_STIMULI" \
        --subj "$subj" --mode vision imagery --gen_rep 10 --device cpu \
        --out "$out_rel" \
        || return 1
    log_master "--- [eval_pixel16_imagery_subj${pad}] tail of log ---"
    tail -n 10 "$LOG_DIR/log_eval_pixel16_imagery_subj${pad}_try1.log" 2>/dev/null >> "$MASTER_LOG" || true
    log_master ""
}

log_master "############################################################"
log_master "pixel16 NSD-Imagery benchmark B chain: subj 01/05/07, GPU $GPU"
log_master "Started: $(date -Iseconds)  Host: $(hostname)"
log_master "############################################################"
log_master ""

overall_exit=0
for subj in "${SUBJECTS[@]}"; do
    pad=$(printf '%02d' "$subj")
    log_master ">>> subj${pad}: chain start $(date -Iseconds) <<<"
    if ! ensure_predictions "$subj" "$pad"; then
        log_master ">>> subj${pad}: PREDICTIONS FAILED, continuing to next $(date -Iseconds) <<<"
        overall_exit=1; continue
    fi
    if ! ensure_generation "$subj" "$pad"; then
        log_master ">>> subj${pad}: GENERATION FAILED after $MAX_RETRIES attempts, continuing to next $(date -Iseconds) <<<"
        overall_exit=1; continue
    fi
    if ! ensure_eval "$subj" "$pad"; then
        log_master ">>> subj${pad}: EVAL FAILED after $MAX_RETRIES attempts, continuing to next $(date -Iseconds) <<<"
        overall_exit=1; continue
    fi
    mkdir -p "$RESULTS_DIR"
    touch "${RESULTS_DIR}/.done_subj${pad}_pixel16"
    log_master ">>> subj${pad}: chain COMPLETE $(date -Iseconds) <<<"
    log_master ""
done

log_master "############################################################"
log_master "All subjects processed. Finished: $(date -Iseconds)  overall_exit=$overall_exit"
log_master "############################################################"
exit $overall_exit
