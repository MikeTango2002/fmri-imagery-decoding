#!/bin/bash
# Re-standardization ablation per branch on NSD-Imagery (Section 8.1.3), GPU 0.
#
# Inputs:  the production ridge weights, adapter checkpoint and harness
# Outputs: data/results/low_level_factor/scalar_restandardization/{zlowscalar,v0repro,zlowraw}_*.tsv
#
# restandardize() is worth +6.5 pp of branch two-way on z_sem_txt but -6.0 pp on z_low, which is also the
# channel behind "imagery simple"; all-or-nothing is not obviously right. Three arms, everything else
# bit-identical (weights, adapter, strength and mixing defaults, --gen-rep 10, --seed 0, 4 subjects):
# zlowscalar (one mean and scale for the whole z_low branch), v0repro (production: all four branches) and
# zlowraw (z_low not restandardized). The production arm is regenerated, not reused, so that the paired
# bootstrap compares per-image metrics produced by the same code state and seed.
#
# Usage:
#   setsid nohup ./run_scalar_restandardization.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
export OPENBLAS_CORETYPE=SkylakeX   # some CPUs miscompute large matmuls without it

SUBJECTS=(1 2 5 7)
GPU=0
RESULTS_DIR="data/results/low_level_factor/scalar_restandardization"
GT_STIMULI="${PROJECT_ROOT}/../mindeye_II/src/imagery_dataset/nsddata_stimuli/stimuli/imagery_stimuli_18.pt"
LOG_DIR="logs/e6_restd_perbranch_gpu0"
MASTER_LOG="$LOG_DIR/E6_RESTD_PERBRANCH_DRIVER.log"
MAX_RETRIES=3
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

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
        [ $exit_code -eq 0 ] && break
        attempt=$((attempt + 1))
    done
    return $exit_code
}

# --- one arm end to end -------------------------------------------------------------------
# $1 arm tag, $2 prediction dir, $3 recon dir, $4... extra flags for predict_ridge_imagery.py
run_arm() {
    local arm="$1" pred_dir="$2" recon_dir="$3"; shift 3
    local extra=("$@")

    log_master "##### ARM ${arm} start $(date -Iseconds) #####"

    # 1. predictions (CPU only, pure numpy matmul, seconds)
    if [ -f "${pred_dir}/subj07/z_low_pred.npy" ]; then
        log_master "=== [predict_${arm}] SKIP already-done ==="
    else
        run_py_step_retry "predict_${arm}" .venv_targets/bin/activate "" . \
            src/predict_ridge_imagery.py --subj "${SUBJECTS[@]}" \
            --out-dir "$pred_dir" "${extra[@]}" || return 1
    fi

    # 2. reconstruction + 3. harness, per subject
    for subj in "${SUBJECTS[@]}"; do
        local pad n_png out_abs
        pad=$(printf '%02d' "$subj")
        log_master ">>> ${arm}/subj${pad}: chain start $(date -Iseconds) <<<"

        n_png=$(find "${recon_dir}/subj${pad}" -iname "*.png" 2>/dev/null | wc -l)
        if [ "$n_png" -ge 300 ]; then
            log_master "=== [gen_${arm}_subj${pad}] SKIP already-done ($n_png PNGs) ==="
        elif ! run_py_step_retry "gen_${arm}_subj${pad}" .venv_targets/bin/activate "$GPU" . \
                src/reconstruct.py --subj "$subj" --gen-rep 10 \
                --pred-dir "$pred_dir" --out-dir "$recon_dir"; then
            log_master ">>> ${arm}/subj${pad}: GENERATION FAILED, continuing <<<"
            continue
        fi

        out_abs="${RESULTS_DIR}/${arm}_subj${pad}.tsv"
        if [ -f "$out_abs" ] && [ "$(wc -l < "$out_abs")" -ge 2 ]; then
            log_master "=== [eval_${arm}_subj${pad}] SKIP already-done ==="
            continue
        fi
        if ! run_py_step_retry "eval_${arm}_subj${pad}" "${PROJECT_ROOT}/.venv_kneeland/bin/activate" "" ../mindeye_II/src \
                final_evaluations_imagery.py \
                --model template --path_template 'subj{subj02}/{cond}/sample{sample03}/{stim05}.png' \
                --recon_root "${PROJECT_ROOT}/${recon_dir}" \
                --gt "$GT_STIMULI" \
                --subj "$subj" --mode vision imagery --gen_rep 10 --device cpu \
                --out "${PROJECT_ROOT}/${out_abs}"; then
            log_master ">>> ${arm}/subj${pad}: EVAL FAILED, continuing <<<"
            continue
        fi
        log_master "--- [eval_${arm}_subj${pad}] tail ---"
        tail -n 12 "$LOG_DIR/log_eval_${arm}_subj${pad}_try1.log" 2>/dev/null >> "$MASTER_LOG" || true
        log_master ">>> ${arm}/subj${pad}: chain COMPLETE $(date -Iseconds) <<<"
        log_master ""
    done
    log_master "##### ARM ${arm} done $(date -Iseconds) #####"
    log_master ""
}

log_master "############################################################"
log_master "Per-branch re-standardization ablation (NSD-Imagery), GPU $GPU"
log_master "Started: $(date -Iseconds)  Host: $(hostname)"
log_master "############################################################"
log_master ""

# Order matters if the run is cut short. The scalar arm comes first: a test showed that z_low
# leaves the ridge at std 0.024 against a target scale of ~1; the "raw" arm would hand the VDVAE
# decoder a near-zero latent and lose for a trivial scale reason, not for the reason this ablation
# is about. The scalar arm keeps the scale (std 0.641) and drops only the per-dimension
# equalization; it is the arm that tests the mechanism. Then the paired control, then the
# raw arm as a bound.
run_arm zlowscalar data/betas_trisect/imagery_predictions_zlowscalar data/imagery_recons_b3_zlowscalar \
    --restandardize-scalar-branches z_low
run_arm v0repro data/betas_trisect/imagery_predictions data/imagery_recons_b3_v0repro
run_arm zlowraw data/betas_trisect/imagery_predictions_zlowraw data/imagery_recons_b3_zlowraw \
    --no-restandardize-branches z_low

log_master "############################################################"
log_master "ALL DONE $(date -Iseconds)"
log_master "############################################################"
