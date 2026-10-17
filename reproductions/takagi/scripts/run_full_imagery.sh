#!/usr/bin/env bash
# Full NSD-Imagery run of Takagi (Kneeland et al. 2025) on all 4 subjects: six restartable stages.
#
# The shared1000 run is run_full_shared1000.sh; this script refuses to start without the weights it produces.
# Only stage 3 needs .venv_blip; each venv is activated in the child shell, and activate .venv before launching.
#
# Stages (compute):
#   1 prepare betas    prepare_nsd_imagery_takagi.py                     CPU, minutes
#   2 decode features  decode_imagery_features.py                        CPU+RAM, ~17 GB npz per subject, sequential
#   3 captions         decode_captions_imagery.py                        GPU, minutes  [.venv_blip]
#   4 reconstruction   diffusion_decoding_imagery.py --n_iter 10         2 GPUs, ~2-3 h
#   5 metrics          eval_extract_features + evaluate_reconstruction   2 GPUs, ~2 h
#   6 aggregation      aggregate_imagery_metrics.py                      CPU, seconds
#
# Usage (from the takagi/ root):
#   bash run_full_imagery.sh [--from 4 | --only-eval | --force]    # --only-eval = --from 5; --force redoes done stages

set -uo pipefail
export OPENBLAS_CORETYPE=SkylakeX

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SUBJECTS=(subj01 subj02 subj05 subj07)
# Vision first on purpose: it is the control, and a broken chain shows up there first.
CONDITIONS=(visioncomplex visionsimple imagerycomplex imagerysimple imageryconcepts)
ROIS_BLIP="early_ventral_midventral_midlateral_lateral_parietal"
NSTIM=6
NREP=10
GPU0_SUBJ=(subj01 subj05)
GPU1_SUBJ=(subj02 subj07)

FROM=1
FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)      FROM="$2"; shift 2 ;;
        --only-eval) FROM=5; shift ;;
        --force)     FORCE=1; shift ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
done

VENV="$ROOT/.venv"
VENV_BLIP="$ROOT/.venv_blip"

die()   { echo; echo "FATAL: $*"; exit 1; }
stage() { echo; echo "=== Stage $1: $2 ==="; }
skip()  { echo "  already complete, skipping (use --force to redo it)"; }

# Wait for background children, then report whether any exited non-zero.
wait_all() {
    local pids=("$@") rc=0 p
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    return $rc
}

# ---- stage 0: prerequisites (never skipped) ------------------------------
stage 0 "prerequisites"
[[ -x "$VENV/bin/python" ]] || die ".venv is missing: bash setup_venv.sh"
for S in "${SUBJECTS[@]}"; do
    [[ -s "decoded/$S/${S}_early_weights_init_latent.npz" ]] \
        || die "missing decoded/$S/${S}_early_weights_init_latent.npz (written by ridge.py --save_weights)"
    [[ -s "decoded/$S/${S}_${ROIS_BLIP}_weights_blip.npz" ]] \
        || die "missing the blip weights of $S (written by ridge.py --save_weights)"
    [[ -s "nsd/nsddata_betas/ppdata/$S/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz" ]] \
        || die "missing the NSD-Imagery betas of $S (download them first)"
done
[[ -s "nsd/nsddata_stimuli/stimuli/imagery_stimuli_18.pt" ]] \
    || die "missing imagery_stimuli_18.pt"
[[ -s "codes/diffusion_sd1/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt" ]] \
    || die "missing the Stable Diffusion v1.4 checkpoint (sd-v1-4.ckpt)"
"$VENV/bin/python" -c "import skimage,sys; sys.exit(0 if skimage.__version__.startswith('0.19') else 1)" \
    || die "scikit-image != 0.19.x: uv pip install 'scikit-image==0.19.3' (the SSIM call of the metrics uses multichannel=, removed in 0.23)"
echo "  ok."

# ---- completeness probes -------------------------------------------------
betas_done() {
    local S=$1 C R
    for C in "${CONDITIONS[@]}"; do
        for R in early ventral midventral midlateral lateral parietal; do
            [[ -s "mrifeat_imagery/$S/${S}_${R}_${C}_betas_ave.npy" ]] || return 1
        done
    done
}
feats_done() {
    local S=$1 C
    for C in "${CONDITIONS[@]}"; do
        [[ -s "decoded_imagery/$S/${S}_early_scores_init_latent_${C}.npy" ]] || return 1
        [[ -s "decoded_imagery/$S/${S}_${ROIS_BLIP}_scores_blip_${C}.npy" ]] || return 1
    done
}
captions_done() {
    local S=$1 C F
    for C in "${CONDITIONS[@]}"; do
        F="decoded_imagery/$S/captions/captions_brain_${C}.csv"
        [[ -s "$F" ]] || return 1
        [[ $(wc -l < "$F") -ge $NSTIM ]] || return 1
    done
}
recon_done() {
    local S=$1 C N
    for C in "${CONDITIONS[@]}"; do
        N=$(find "decoded/image-text-imagery/$S/$C/samples" -maxdepth 1 \
            -name '*_[0-9][0-9][0-9].png' ! -name '*_org.png' 2>/dev/null | wc -l)
        [[ $N -ge $((NSTIM * NREP)) ]] || return 1
    done
}

# ---- stage 1: prepare betas ---------------------------------------------
if [[ $FROM -le 1 ]]; then
    stage 1 "prepare the betas (prepare_nsd_imagery_takagi.py)"
    NEED=()
    for S in "${SUBJECTS[@]}"; do
        if [[ $FORCE -eq 1 ]] || ! betas_done "$S"; then NEED+=("$S"); fi
    done
    if [[ ${#NEED[@]} -eq 0 ]]; then
        skip
    else
        echo "  subjects: ${NEED[*]}  (log: prepare_imagery.log)"
        ( source "$VENV/bin/activate"
          export OPENBLAS_CORETYPE=SkylakeX
          cd "$ROOT/codes/utils" || exit 1
          python prepare_nsd_imagery_takagi.py \
              --subject "${NEED[@]}" \
              --data_root ../../nsd --out_dir ../../mrifeat_imagery \
              --weights_dir ../../decoded ) 2>&1 | tee "$ROOT/prepare_imagery.log"
        for S in "${NEED[@]}"; do
            betas_done "$S" || die "stage 1 failed on $S: read prepare_imagery.log"
        done
        echo "  ok."
    fi
fi

# ---- stage 2: decode features -------------------------------------------
# Sequential on purpose: the blip weight file is ~17 GB per subject and two of them
# in RAM at once is how this stage gets OOM-killed.
if [[ $FROM -le 2 ]]; then
    stage 2 "inference from the weights (decode_imagery_features.py)"
    NEED=()
    for S in "${SUBJECTS[@]}"; do
        if [[ $FORCE -eq 1 ]] || ! feats_done "$S"; then NEED+=("$S"); fi
    done
    if [[ ${#NEED[@]} -eq 0 ]]; then
        skip
    else
        echo "  subjects: ${NEED[*]}  (sequential, log: decode_imagery.log)"
        ( source "$VENV/bin/activate"
          export OPENBLAS_CORETYPE=SkylakeX
          cd "$ROOT/codes/utils" || exit 1
          for S in "${NEED[@]}"; do
              for T in init_latent blip; do
                  python decode_imagery_features.py --subject "$S" --target "$T" \
                      --imagery_dir ../../mrifeat_imagery --weights_dir ../../decoded \
                      --out_dir ../../decoded_imagery \
                    || echo "decode $S $T FAILED (exit $?)"
              done
          done ) 2>&1 | tee "$ROOT/decode_imagery.log"
        for S in "${NEED[@]}"; do
            feats_done "$S" || die "stage 2 failed on $S: read decode_imagery.log"
        done
        echo "  ok."
    fi
fi

# ---- stage 3: captions (.venv_blip) --------------------------------------
if [[ $FROM -le 3 ]]; then
    stage 3 "brain captions (decode_captions_imagery.py, venv .venv_blip)"
    NEED=()
    for S in "${SUBJECTS[@]}"; do
        if [[ $FORCE -eq 1 ]] || ! captions_done "$S"; then NEED+=("$S"); fi
    done
    if [[ ${#NEED[@]} -eq 0 ]]; then
        skip
    else
        [[ -x "$VENV_BLIP/bin/python" ]] || die ".venv_blip is missing: bash setup_venv_blip.sh"
        echo "  subjects: ${NEED[*]}  (log: decode_captions_imagery.log)"
        ( source "$VENV_BLIP/bin/activate"
          export OPENBLAS_CORETYPE=SkylakeX
          cd "$ROOT/codes/caption/BLIP" || exit 1
          for S in "${NEED[@]}"; do
              for C in "${CONDITIONS[@]}"; do
                  CUDA_VISIBLE_DEVICES=0 python decode_captions_imagery.py \
                      --subject "$S" --condition "$C" \
                    || echo "captions $S $C FAILED (exit $?)"
              done
          done ) 2>&1 | tee "$ROOT/decode_captions_imagery.log"
        for S in "${NEED[@]}"; do
            captions_done "$S" || die "stage 3 failed on $S: read decode_captions_imagery.log"
        done
        echo
        echo "  --- read the captions before spending 3 h of GPU ---"
        for S in "${NEED[@]}"; do
            echo "  == $S imagerycomplex"
            head -6 "decoded_imagery/$S/captions/captions_brain_imagerycomplex.csv" | sed 's/^/     /'
        done
    fi
fi

# ---- stage 4: reconstruction --------------------------------------------
if [[ $FROM -le 4 ]]; then
    stage 4 "reconstruction (diffusion_decoding_imagery.py --n_iter $NREP)"
    NEED0=(); NEED1=()
    for S in "${GPU0_SUBJ[@]}"; do
        if [[ $FORCE -eq 1 ]] || ! recon_done "$S"; then NEED0+=("$S"); fi
    done
    for S in "${GPU1_SUBJ[@]}"; do
        if [[ $FORCE -eq 1 ]] || ! recon_done "$S"; then NEED1+=("$S"); fi
    done

    PIDS=()
    # $0 = takagi root, $1 = n_iter, then the conditions up to the '--' marker,
    # then the subjects. Positional passing avoids quoting the arrays into a string.
    run_recon() {
        local G=$1; shift
        local SS=("$@")
        [[ ${#SS[@]} -eq 0 ]] && return 0
        local LOGFILE="$ROOT/recon_imagery_gpu${G}.log"
        echo "  GPU $G: ${SS[*]}  (log: recon_imagery_gpu${G}.log)"
        CUDA_VISIBLE_DEVICES="$G" nohup bash -c '
            source "$0/.venv/bin/activate"
            export OPENBLAS_CORETYPE=SkylakeX
            cd "$0/codes/diffusion_sd1" || exit 1
            NR=$1; shift
            CONDS=(); while [[ "$1" != "--" ]]; do CONDS+=("$1"); shift; done; shift
            for S in "$@"; do
                for C in "${CONDS[@]}"; do
                    python diffusion_decoding_imagery.py \
                        --subject "$S" --condition "$C" --method text \
                        --n_iter "$NR" --gpu 0 --skip_existing \
                        --imagery_dir ../../decoded_imagery \
                        --out_dir ../../decoded/image-text-imagery \
                      || echo "FAILED recon $S $C (exit $?)"
                done
                echo "recon imagery $S done"
            done
        ' "$ROOT" "$NREP" "${CONDITIONS[@]}" -- "${SS[@]}" > "$LOGFILE" 2>&1 &
        PIDS+=("$!")
    }

    if [[ ${#NEED0[@]} -eq 0 && ${#NEED1[@]} -eq 0 ]]; then
        skip
    else
        run_recon 0 "${NEED0[@]:+${NEED0[@]}}"
        run_recon 1 "${NEED1[@]:+${NEED1[@]}}"
        echo "  waiting for the generation (PIDs: ${PIDS[*]})..."
        wait_all "${PIDS[@]}" || echo "  WARNING: at least one job exited with an error"
        for S in "${SUBJECTS[@]}"; do
            recon_done "$S" || die "stage 4 incomplete on $S: grep FAILED recon_imagery_gpu*.log"
        done
        TOT=$(find decoded/image-text-imagery -name '*_[0-9][0-9][0-9].png' ! -name '*_org.png' | wc -l)
        echo "  ok: $TOT PNGs (expected $(( ${#SUBJECTS[@]} * ${#CONDITIONS[@]} * NSTIM * NREP )))"
    fi
fi

# ---- stage 5: metrics ----------------------------------------------------
# imageryconcepts is absent here on purpose: no visual ground truth; it enters no
# image-based metric. It is generated only for the human-rater figures.
if [[ $FROM -le 5 ]]; then
    stage 5 "metrics (eval_extract_features + evaluate_reconstruction)"
    echo "  ground-truth features (12 stimuli, computed once)"
    ( source "$VENV/bin/activate"
      export OPENBLAS_CORETYPE=SkylakeX
      cd "$ROOT/codes/utils" || exit 1
      CUDA_VISIBLE_DEVICES=0 python eval_extract_features_imagery.py --gt ) \
        || die "ground-truth extraction failed"

    PIDS=()
    run_metrics() {
        local G=$1; shift
        local SS=("$@")
        [[ ${#SS[@]} -eq 0 ]] && return 0
        local LOGFILE="$ROOT/metrics_imagery_gpu${G}.log"
        echo "  GPU $G: ${SS[*]}  (log: metrics_imagery_gpu${G}.log)"
        CUDA_VISIBLE_DEVICES="$G" nohup bash -c '
            source "$0/.venv/bin/activate"
            export OPENBLAS_CORETYPE=SkylakeX
            cd "$0/codes/utils" || exit 1
            NR=$1; shift
            for S in "$@"; do
                for M in vision imagery; do
                    for N in $(seq 0 $(( NR - 1 ))); do
                        python eval_extract_features_imagery.py \
                            --subject "$S" --modality "$M" --sample "$N" \
                          && python evaluate_reconstruction_imagery.py \
                            --subject "$S" --modality "$M" --sample "$N" \
                          || echo "FAILED metrics $S $M $N"
                    done
                done
                echo "metrics $S done"
            done
        ' "$ROOT" "$NREP" "${SS[@]}" > "$LOGFILE" 2>&1 &
        PIDS+=("$!")
    }

    run_metrics 0 "${GPU0_SUBJ[@]}"
    run_metrics 1 "${GPU1_SUBJ[@]}"
    echo "  waiting for the metrics (PIDs: ${PIDS[*]})..."
    wait_all "${PIDS[@]}" || echo "  WARNING: at least one job exited with an error"

    NTSV=$(find results/imagery_metrics -name 'per_image_subj*_s*.tsv' 2>/dev/null | wc -l)
    NEXP=$(( ${#SUBJECTS[@]} * 2 * NREP ))
    echo "  per-image TSV: $NTSV/$NEXP"
    [[ $NTSV -ge $NEXP ]] || echo "  WARNING: missing cells, grep FAILED metrics_imagery_gpu*.log"
fi

# ---- stage 6: aggregation ------------------------------------------------
if [[ $FROM -le 6 ]]; then
    stage 6 "aggregation"
    mkdir -p "$ROOT/results/imagery_metrics"
    ( source "$VENV/bin/activate"
      export OPENBLAS_CORETYPE=SkylakeX
      cd "$ROOT/codes/utils" || exit 1
      python aggregate_imagery_metrics.py ) \
      | tee "$ROOT/results/imagery_metrics/summary.md"
fi

echo
echo "=== DONE (benchmark B) ==="
echo "  Summary:    results/imagery_metrics/summary.md"
echo "  Tables:     results/imagery_metrics/summary_overall.tsv (rows modality x stim_type)"
echo "  A WARNING 'cell ... has N samples' is a half-empty cell, not a detail."
echo "  Copy the numbers somewhere safe BEFORE deleting any intermediate file."
