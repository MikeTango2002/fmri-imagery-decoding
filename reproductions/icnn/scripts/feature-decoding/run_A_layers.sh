#!/usr/bin/env bash
# Benchmark A, stage 1: one VGG19 layer at a time, 3 workers per step.
# Run it from icnn/feature-decoding/, with the environment active, after the Deeprecon data are in place:
#     bash run_A_layers.sh
# One log per worker is written to ../training_logs/ (create the folder first).
# Layers run cheapest-first so the 24 GB conv1_* disk peak lands as late as possible.
set -uo pipefail
export OPENBLAS_CORETYPE=SkylakeX
CFG=config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml
DEC=data/feature_decoders/ImageNetTraining/deeprecon_fmriprep_pyfastl2lir_alpha100_allunits/caffe/VGG19
LOGDIR=../training_logs

# Runs one step of the pipeline as 3 workers and reports failure of any of them.
# `wait` with no arguments always returns 0; the PIDs have to be waited on
# one by one: otherwise the `break` below never fires and the run ends with a
# silently incomplete evaluation.db.
# Each worker gets its own log: three of them interleaved on one stdout is unreadable,
# and the per-worker file is what tells you which (layer, subject) a failure belongs to.
run_step() {
    local SCRIPT=$1 L=$2
    local STEP=${SCRIPT%%_*}          # train | predict
    local PIDS=() P W FAIL=0
    for W in 1 2 3; do
        python "$SCRIPT" $CFG -o "decoder.features.layers=[$L]" "decoded_feature.features.layers=[$L]" \
            > "$LOGDIR/${STEP}_${L}_w${W}.log" 2>&1 &
        PIDS+=($!)
        sleep 5   # stagger the starts: two workers must not read the sqlite lock at once
    done
    for P in "${PIDS[@]}"; do wait "$P" || FAIL=1; done
    return $FAIL
}

for L in fc8 fc7 fc6 conv5_4 conv5_3 conv5_2 conv5_1 conv4_4 conv4_3 conv4_2 conv4_1 \
         conv3_4 conv3_3 conv3_2 conv3_1 conv2_2 conv2_1 conv1_2 conv1_1; do
    echo "=== $L  $(date +%H:%M:%S) ==="
    run_step train_decoder_fastl2lir.py   "$L" || { echo "train $L FAILED";   break; }
    run_step predict_feature_fastl2lir.py "$L" || { echo "predict $L FAILED"; break; }
    find $DEC/$L -name 'W.mat' -delete    # y_mean/y_norm stay: evaluation.py still works
    df -h --output=avail / | tail -1      # the free space must not go down
done
