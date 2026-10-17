#!/usr/bin/env bash
# Memory and time profile of Train.py at several micro-batches, in one driver.
#
# Runs Train.py --profile_memory once per micro-batch and fits the two lines that decide the rest: MiB per sample
# and seconds per sample. Each run is ~20 iterations plus startup; a sweep takes minutes. An OOM is a result
# here, not a failure: it is recorded and the sweep continues. setsid, not nohup alone: accelerate launch goes
# through torch.distributed elastic, whose agent reinstalls a SIGHUP handler over nohup's SIG_IGN.
#
# Usage (from src/, venv active, alone on the cards):
#   setsid nohup bash profile_memory.sh --batches "14 21 28" \
#     > ../train_logs/profiling/profile_sweep.log 2>&1 < /dev/null &
#   bash profile_memory.sh --batches "35 49 63" --ckpt prior          # checkpointed prior: does the paper's batch fit?
#   bash profile_memory.sh --batches 63 --ckpt prior --multi-subject  # stage 1
#   bash profile_memory.sh --batches "21 35" --acc-cfg <a config with CPU offload>

set -uo pipefail

BATCHES="14 21 28"
CKPT=""
ITERS=20
MULTI=0
SUBJ=1
HIDDEN_DIM=4096
GAS=1
TAG=""
DRY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batches)        BATCHES="$2"; shift 2 ;;
        --ckpt)           CKPT="$2"; shift 2 ;;
        --iters)          ITERS="$2"; shift 2 ;;
        --multi-subject)  MULTI=1; shift ;;
        --subj)           SUBJ="$2"; shift 2 ;;
        --hidden-dim)     HIDDEN_DIM="$2"; shift 2 ;;
        --grad-accum)     GAS="$2"; shift 2 ;;
        --acc-cfg)        ACC_CFG="$2"; shift 2 ;;
        --tag)            TAG="$2"; shift 2 ;;
        --dry-run)        DRY=1; shift ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(cd .. && pwd)"

export NUM_GPUS=2
export ACC_CFG="${ACC_CFG:-$ROOT/accelerate_2gpu_zero2.yaml}"
export DATA=mindeyev2_dataset
export FCACHE=$DATA/frozen_cache.hdf5
PROF="$ROOT/train_logs/profiling"
mkdir -p "$PROF"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Name the sweep after what makes it different; two sweeps never overwrite each other's JSON.
if [[ -z "$TAG" ]]; then
    if [[ $MULTI -eq 1 ]]; then TAG=multi; else TAG=single; fi
    if [[ -n "$CKPT" ]]; then TAG="${TAG}_ckpt${CKPT//,/-}"; else TAG="${TAG}_ckptnone"; fi
    case "$ACC_CFG" in *offload*) TAG="${TAG}_offload" ;; esac
    if [[ "$GAS" -ne 1 ]]; then TAG="${TAG}_gas$GAS"; fi
fi

die() { echo "FATAL: $*" >&2; exit 1; }

# ---- preflight ------------------------------------------------------------
[[ -n "${VIRTUAL_ENV:-}" ]] || die "no venv active. source ../.venv/bin/activate first."
[[ -f "$ACC_CFG" ]] || die "accelerate config not found: $ACC_CFG"
[[ -f "$FCACHE" ]]  || die "frozen cache not found: $FCACHE (build it first with precompute_frozen.py)"
DS_JSON=$(sed -n 's/^ *deepspeed_config_file: *//p' "$ACC_CFG" | tr -d '"'"'"'')
if [[ -n "$DS_JSON" ]]; then
    [[ -f "$DS_JSON" ]] || die "$ACC_CFG points at '$DS_JSON', not found relative to $(pwd)"
fi
if grep -q 'offload_optimizer_device: *cpu' "$ACC_CFG" && [[ -z "$(command -v nvcc)" ]]; then
    die "$(basename "$ACC_CFG") offloads via the accelerate plugin -> DeepSpeedCPUAdam -> needs
   nvcc. Use an offload config that keeps Train.py's torch.optim.AdamW (zero_force_ds_cpu_optimizer: false)."
fi
bash apply_upstream_patches.sh --check > /dev/null 2>&1 \
    || die "upstream patches missing. Run: bash apply_upstream_patches.sh"
OTHERS=$(pgrep -f 'Train\.py .*--model_name=(prof_|verify_)' | tr '\n' ' ')
[[ -z "${OTHERS// }" ]] || die "another profiling/verify run is on the GPUs (pids: $OTHERS).
   These are not parallelisable: their peaks would be each other's."
if [[ $MULTI -eq 1 ]]; then
    for B in $BATCHES; do
        (( B % 7 == 0 )) || die "stage-1 batch $B is not a multiple of 7; Train.py would
   silently train on $(( (B/7)*7 )) samples per GPU."
    done
fi

echo "  batches: $BATCHES | ckpt: ${CKPT:-none} | iters: $ITERS | gas: $GAS"
echo "  stage: $([[ $MULTI -eq 1 ]] && echo 'multi-subject (stage 1)' || echo 'single-subject (stage 2)') | hidden_dim: $HIDDEN_DIM"
echo "  accelerate config: $(basename "$ACC_CFG")"
echo "  tag: $TAG -> $PROF/prof_${TAG}_b<N>_rank0.json"
echo

# ---- the sweep ------------------------------------------------------------
for B in $BATCHES; do
    NAME="prof_${TAG}_b${B}"
    LOG="$PROF/${NAME}.log"
    STAGE=(--no-multi_subject --subj=$SUBJ)
    [[ $MULTI -eq 1 ]] && STAGE=(--multi_subject --subj=$SUBJ)

    ARGS=(
        --model_name=$NAME
        --data_path=$DATA --cache_dir=$DATA
        "${STAGE[@]}" --batch_size=$B --num_sessions=40
        --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150
        --use_prior --prior_scale=30 --clip_scale=1
        --blurry_recon --blur_scale=.5 --no-use_image_aug
        --n_blocks=4 --hidden_dim=$HIDDEN_DIM
        --frozen_cache=$FCACHE --pixcorr_stride=10
        --grad_accum_steps=$GAS
        --profile_memory=$ITERS
        --ckpt_interval=999 --no-ckpt_saving --no-wandb_log
    )
    [[ -n "$CKPT" ]] && ARGS+=(--grad_checkpointing=$CKPT)

    echo "=== batch $B  (log: train_logs/profiling/${NAME}.log) ==="
    date '+    start %F %T'
    if [[ $DRY -eq 1 ]]; then
        printf '    would run: GLOBAL_BATCH_SIZE=%s accelerate launch --config_file=%s Train.py %s\n\n' \
            "$((B*NUM_GPUS))" "$ACC_CFG" "${ARGS[*]}"
        continue
    fi

    PORT=$(( RANDOM % 8000 + 11000 ))
    GLOBAL_BATCH_SIZE=$((B*NUM_GPUS)) accelerate launch \
        --config_file="$ACC_CFG" --main_process_port=$PORT Train.py "${ARGS[@]}" > "$LOG" 2>&1
    RC=$?
    if [[ $RC -ne 0 ]]; then
        if grep -q "out of memory" "$LOG"; then
            echo "    OOM at batch $B -- that is a data point, not a crash. Continuing."
            grep -m1 -o "Tried to allocate [^;]*" "$LOG" | sed 's/^/      /'
            grep -m1 -o "GPU 0 has a total capacity of [^;]*" "$LOG" | sed 's/^/      /'
        else
            echo "    FAILED (exit $RC), and not on memory. First traceback:"
            awk '/Traceback \(most recent call last\)/{f=1} f{print; n++} n>25{exit}' "$LOG" \
                | sed 's/^/      /'
        fi
    else
        grep -h "\[mem-profile\] $NAME" "$LOG" | sed 's/^/    /'
    fi
    date '+    done  %F %T'
    echo
done

[[ $DRY -eq 1 ]] && exit 0

# ---- the fit --------------------------------------------------------------
echo "=== fit: memory and time per sample ==="
python fit_memory_scaling.py --out "$PROF/memory_scaling.md" "$PROF"/prof_"${TAG}"_b*_rank0.json
echo
echo "All the configurations measured so far, together:"
python fit_memory_scaling.py "$PROF"/prof_*_rank0.json > "$PROF/memory_scaling_all.md" \
    && echo "  -> train_logs/profiling/memory_scaling_all.md"
