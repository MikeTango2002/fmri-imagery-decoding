#!/usr/bin/env bash
# MindEye2 training: stage 1 (multi-subject pretraining) and stage 2 (single-subject finetuning), per subject.
#
# Why interleaved: peak disk (four pretrainings then four finetunings keep 8 checkpoints alive, going subject by
# subject keeps 2) and feedback (the first finetuned last.pth lands after ~2.6 days instead of ~10.3; the
# inference chain can be validated on one subject while the others queue). The recipe is unchanged. Resumable per
# subject (a stage whose last.pth exists is skipped), not mid-run: Train.py has no --resume_from_ckpt.
# setsid, not nohup: accelerate launch with DeepSpeed goes through torch.distributed elastic, whose agent installs
# its own SIGHUP handler over nohup's SIG_IGN; a hangup ended a run once ("SignalException ... got signal: 1").
#
# Usage (from src/, venv active; days of GPU):
#   setsid nohup bash run_training.sh --prune-pretrain > ../train_logs/logs/run_training.log 2>&1 < /dev/null &
#   bash run_training.sh --subjects "1 5" --dry-run    # a subset; --dry-run prints the commands, launches nothing
# --prune-pretrain drops multisubject_subj0X once subject X is finetuned.

set -uo pipefail

SUBJECTS="1 2 5 7"
PRUNE=0
DRY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --subjects)       SUBJECTS="$2"; shift 2 ;;
        --prune-pretrain) PRUNE=1; shift ;;
        --dry-run)        DRY=1; shift ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(cd .. && pwd)"

export NUM_GPUS=2
# No CPU offload. Gradient checkpointing on the prior alone gets batch 63 per GPU into VRAM:
# measured in the real stage-1 config (multi-subject, gas=4), peak reserved 44194 MiB against
# ~47370 usable, +3176 MiB of headroom. The offload configurations of the working copy were never
# needed, and if VRAM were ever short again, the cheaper knob is DeepSpeed's reduce_bucket_size,
# not the offload.
#
# gas4, not the plain zero2 config: 2 GPUs x 63 x 4 accumulation steps = 504 = the paper's
# optimizer batch, and 59 optimizer updates per epoch = exactly what the authors' 8-GPU run did
# (3750//63). Stage 2 rides the same config: 625//4 = 156 updates per epoch, again exactly the
# paper's, with an optimizer batch of 24 x 2 x 4 = 192. Both stages MUST pass --grad_accum_steps=4
# to match it: Train.py asserts the two against each other.
export ACC_CFG="${ACC_CFG:-$ROOT/accelerate_2gpu_zero2_gas4.yaml}"
# This setting is load-bearing without the offload: the config OOMs at batch 24, asking for
# a 6 GB contiguous block with ~17 GB sitting in "reserved but unallocated" fragments.
# Allocator-only setting: it changes no arithmetic.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DATA=mindeyev2_dataset
export FCACHE=$DATA/frozen_cache.hdf5
LOGS="$ROOT/train_logs/logs"
mkdir -p "$LOGS"

# Shared across both stages: the hyperparameters of the paper.
COMMON=(
    --data_path=$DATA --cache_dir=$DATA
    --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150
    --use_prior --prior_scale=30 --clip_scale=1
    --blurry_recon --blur_scale=.5 --no-use_image_aug
    --n_blocks=4 --hidden_dim=4096 --num_sessions=40
    --frozen_cache=$FCACHE
    # Both stages accumulate 4 micro-steps per optimizer update. It is not a memory device: it
    # is what makes the optimizer batch (504 / 192) and the length of the OneCycleLR horizon
    # (8850 / 23400 updates) equal the paper's on 2 GPUs instead of 8. It also has to agree with
    # $ACC_CFG or Train.py aborts.
    --grad_accum_steps=4
    --pixcorr_stride=10 --ckpt_slim
    --ckpt_interval=10 --ckpt_saving --no-wandb_log
)

# ---- preflight ------------------------------------------------------------
die() { echo "FATAL: $*" >&2; exit 1; }

[[ -n "${VIRTUAL_ENV:-}" ]] || die "no venv active. source ../.venv/bin/activate first."
[[ -f "$ACC_CFG" ]] || die "accelerate config not found: $ACC_CFG"
# Train.py asserts --grad_accum_steps against the resolved DeepSpeed config, but only after
# loading the model and the 73k images, minutes. Catch the mismatch here instead.
CFG_GAS=$(sed -n 's/^[[:space:]]*gradient_accumulation_steps:[[:space:]]*\([0-9]\+\).*/\1/p' "$ACC_CFG")
[[ "${CFG_GAS:-1}" == "4" ]] || die "$(basename "$ACC_CFG") has gradient_accumulation_steps=${CFG_GAS:-1},
   but COMMON passes --grad_accum_steps=4. Use accelerate_2gpu_zero2_gas4.yaml, or change both."
[[ -f "$FCACHE" ]]  || die "frozen cache not found: $FCACHE (build it first with precompute_frozen.py)"
bash apply_upstream_patches.sh --check > /dev/null 2>&1 \
    || die "upstream patches missing. Run: bash apply_upstream_patches.sh"

# The cache refuses to open if a part it declares is incomplete; this is a real check and
# not a file-exists test.
python - "$FCACHE" <<'PY' || die "frozen cache is not usable"
import sys
from frozen_cache import FrozenCache
c = FrozenCache(sys.argv[1])
print(f"  frozen cache OK: {c}")
PY

AVAIL_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "  disk available: ${AVAIL_GB} GB"
# ~5 GB per slim checkpoint at 4096, 2 alive per subject, plus room for the run in flight.
[[ ${AVAIL_GB} -ge 30 ]] || die "under 30 GB free; a single checkpoint pair will not fit"

echo "  subjects: $SUBJECTS | prune pretrain: $PRUNE | dry run: $DRY"
# Recorded in the log because an NCCL hang is diagnosed exactly by which of these was set.
echo "  NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-unset} NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-unset}"
echo

# ---- helpers --------------------------------------------------------------
run_stage() {
    # $1 = log file, rest = accelerate args
    local logfile="$1"; shift
    if [[ $DRY -eq 1 ]]; then
        printf '    would run: accelerate launch %s\n' "$*"
        return 0
    fi
    # A fresh port per stage: a previous run that died can leave the old one in TIME_WAIT.
    local port=$(( RANDOM % 8000 + 11000 ))
    GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE accelerate launch \
        --config_file="$ACC_CFG" --main_process_port=$port "$@" > "$logfile" 2>&1
}

# ---- the loop -------------------------------------------------------------
for S in $SUBJECTS; do
    PRE="multisubject_subj0${S}_4096hid"
    FIN="finetuned_subj0${S}_40sess_4096hid"

    echo "=== subject 0$S ==="
    date '+    start %F %T'

    # --- stage 1: multi-subject pretraining (all subjects except S) ---
    if [[ -f "$ROOT/train_logs/$PRE/last.pth" ]]; then
        echo "    [1/2] $PRE already present, skipping"
    else
        echo "    [1/2] pretraining $PRE  (log: train_logs/logs/pretrain_subj0${S}.log)"
        # --batch_size is PER PROCESS: the argparse value overwrites the one derived from
        # GLOBAL_BATCH_SIZE, and Train.py divides it again by the number of subjects (7 in
        # stage 1). The paper's 63 is therefore also per GPU ("9 samples per each of 7
        # subjects" is exactly that division), which is why it can be matched here on 2 cards.
        #
        # This per-GPU number is the ONLY batch that matters for fidelity. The contrastive losses
        # (utils.mixco_nce and utils.soft_clip_loss) use in-batch negatives
        # and the loop has no cross-device gather; the paper's 8-GPU run had 63 negatives per
        # process, never 504. Aggregate batch is an optimizer-side quantity only.
        #
        # 63 is the paper's own number and it is measured to fit, not extrapolated:
        # multi-subject with gas=4 and --grad_checkpointing=prior, peak reserved 44194 MiB with
        # +3176 MiB to spare, 2.704 s/it. A smaller per-GPU batch would be a deviation on the
        # only quantity the contrastive loss cares about.
        #
        # Upstream contradicts itself on this number: the paper's
        # text says 63 ("9 samples per each of 7 subjects"), README.md:75, the exact command
        # behind the released checkpoints; says 42, and accel.slurm:21 ships 21 with a comment
        # claiming the paper used 42. The paper's text is followed: it is the published recipe, and
        # it is the strictest of the three.
        export BATCH_SIZE=63
        # Train.py does an integer division by the 7 pretraining subjects: a value that is not a
        # multiple of 7 silently drops the remainder (24 -> 3 per subject -> 21 actually used).
        # Only the multi-subject stage is affected; stage 2 has len(subj_list) == 1.
        (( BATCH_SIZE % 7 == 0 )) || die "stage-1 BATCH_SIZE=$BATCH_SIZE is not a multiple of 7;
   Train.py would silently train on $(( (BATCH_SIZE / 7) * 7 )) samples per GPU."
        export GLOBAL_BATCH_SIZE=$(( BATCH_SIZE * NUM_GPUS ))    # 126; x4 accumulation = 504
        # Checkpointing the 6 prior layers is what makes 63 fit; it costs ~+26% on the s/it and
        # produces bit-identical gradients (tests/test_grad_checkpointing.py, all rows at
        # 0.000e+00). Stage 2 uses it too (see below).
        run_stage "$LOGS/pretrain_subj0${S}.log" Train.py \
            --model_name=$PRE --multi_subject --subj=$S --batch_size=$BATCH_SIZE \
            --grad_checkpointing=prior "${COMMON[@]}"
        if [[ $DRY -eq 0 && ! -f "$ROOT/train_logs/$PRE/last.pth" ]]; then
            echo "    FAILED: no last.pth for $PRE -- see the log. Stopping."; exit 1
        fi
    fi

    # --- stage 2: single-subject finetuning, initialised from stage 1 ---
    if [[ -f "$ROOT/train_logs/$FIN/last.pth" ]]; then
        echo "    [2/2] $FIN already present, skipping"
    else
        echo "    [2/2] finetuning $FIN  (log: train_logs/logs/finetune_subj0${S}.log)"
        # 24 per GPU is the authors' own stage-2 value (README.md:81), unchanged. What changed is
        # the accumulation: with gas=4 the optimizer sees 24 x 2 x 4 = 192 per update and the
        # epoch is 156 updates, both exactly the paper's 8-GPU numbers. At gas=1 they would be
        # 48 and 625, i.e. a 4x smaller optimizer batch over a 4x longer LR horizon.
        #
        # --grad_checkpointing=prior is needed here too: measured with gas=4 (profile_memory.sh),
        # batch 24 without checkpointing peaks at 46574 MiB reserved; only +796 MiB under the
        # 47370 MiB ceiling, below the ~1 GB margin treated here as an OOM, not a pass. With
        # --grad_checkpointing=prior the same batch peaks at 38354 MiB, +9016 MiB of margin, for
        # +7.1% s/it (1.166 -> 1.249). Bit-identical gradients (tests/test_grad_checkpointing.py),
        # so this is instrumentation, not a recipe deviation.
        export BATCH_SIZE=24
        export GLOBAL_BATCH_SIZE=$(( BATCH_SIZE * NUM_GPUS ))    # 48; x4 accumulation = 192
        run_stage "$LOGS/finetune_subj0${S}.log" Train.py \
            --model_name=$FIN --no-multi_subject --subj=$S --batch_size=$BATCH_SIZE \
            --grad_checkpointing=prior \
            --multisubject_ckpt="$ROOT/train_logs/$PRE" "${COMMON[@]}"
        if [[ $DRY -eq 0 && ! -f "$ROOT/train_logs/$FIN/last.pth" ]]; then
            echo "    FAILED: no last.pth for $FIN -- see the log. Stopping."; exit 1
        fi
    fi

    # --- prune stage 1 ---
    # Only once the log proves that the finetuning actually consumed it. Without that line in the
    # log the finetuning silently trained from random init, and deleting the pretrain
    # checkpoint would destroy the only evidence and ~28 h of GPU.
    if [[ $PRUNE -eq 1 && $DRY -eq 0 && -d "$ROOT/train_logs/$PRE" ]]; then
        if grep -q -- "---loading .*/$PRE/last.pth ckpt---" "$LOGS/finetune_subj0${S}.log" 2>/dev/null; then
            SZ=$(du -sh "$ROOT/train_logs/$PRE" | cut -f1)
            rm -rf "$ROOT/train_logs/$PRE"
            echo "    pruned $PRE ($SZ freed)"
        else
            echo "    NOT pruning $PRE: no '---loading .../last.pth ckpt---' in the finetune log."
            echo "    Check test/fwd_pct_correct at epoch 0 -- it must be well above 1/300."
        fi
    fi

    date '+    done  %F %T'
    echo
done

echo "=== training done ==="
ls -lh "$ROOT"/train_logs/finetuned_subj0*_40sess_4096hid/last.pth 2>/dev/null
echo
echo "Next: bash run_inference_eval.sh"
echo "The frozen cache is dead weight from here on -- inference never opens it."
