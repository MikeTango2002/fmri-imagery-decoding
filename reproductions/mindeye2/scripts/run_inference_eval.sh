#!/usr/bin/env bash
# MindEye2 inference + shared1000 evaluation (reconstructions and metrics), two subjects at a time.
#
# None of the inference scripts uses accelerate; each is a single-GPU job: GPU 0 takes subjects 1 and 5, GPU 1
# takes 2 and 7, which halves the time. Everything runs from src/ because enhanced_recon_inference.py has its
# dataset paths hardcoded relative to cwd. Idempotent: a subject whose output exists is skipped; a crashed run
# can be relaunched.
#
# Usage (from src/, venv active):
#   nohup bash run_inference_eval.sh > ../train_logs/logs/run_inference_eval.log 2>&1 &
#   bash run_inference_eval.sh --subjects "1 5"
#   bash run_inference_eval.sh --skip-eval      # reconstructions only, stop before final_evaluations.py
#   bash run_inference_eval.sh --dry-run

set -uo pipefail

SUBJECTS="1 2 5 7"
SKIP_EVAL=0
DRY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --subjects)  SUBJECTS="$2"; shift 2 ;;
        --skip-eval) SKIP_EVAL=1; shift ;;
        --dry-run)   DRY=1; shift ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(cd .. && pwd)"
LOGS="$ROOT/train_logs/logs"
mkdir -p "$LOGS" evals tables

DATA=mindeyev2_dataset

die() { echo "FATAL: $*" >&2; exit 1; }

# ---- preflight ------------------------------------------------------------
[[ -n "${VIRTUAL_ENV:-}" ]] || die "no venv active. source ../.venv/bin/activate first."
bash apply_upstream_patches.sh --check > /dev/null 2>&1 \
    || die "upstream patches missing. Run: bash apply_upstream_patches.sh"

for S in $SUBJECTS; do
    [[ -f "$ROOT/train_logs/finetuned_subj0${S}_40sess_4096hid/last.pth" ]] \
        || die "no finetuned checkpoint for subject 0$S -- run run_training.sh first"
done
echo "  subjects: $SUBJECTS | skip eval: $SKIP_EVAL | dry run: $DRY"
echo

# ---- one subject, start to finish, on one GPU -----------------------------
one_subject() {
    local G=$1 S=$2
    local M="finetuned_subj0${S}_40sess_4096hid"
    local OUT="evals/$M"

    echo "[gpu$G subj0$S] start $(date '+%F %T')"

    # Base reconstructions. Also writes _all_blurryrecons.pt / _all_predcaptions.pt /
    # _all_clipvoxels.pt, all four of which the refinement below then loads.
    if [[ -f "$OUT/${M}_all_recons.pt" ]]; then
        echo "[gpu$G subj0$S] reconstructions already done, skipping"
    else
        echo "[gpu$G subj0$S] reconstructing: recon_inference.py"
        [[ $DRY -eq 1 ]] || CUDA_VISIBLE_DEVICES=$G python recon_inference.py \
            --model_name=$M --subj=$S --data_path=$DATA --cache_dir=$DATA \
            --hidden_dim=4096 --n_blocks=4 --new_test \
            || { echo "[gpu$G subj0$S] reconstruction FAILED"; return 1; }
    fi

    # SDXL (ZavyChromaXL) refinement. This is the artefact both benchmarks are scored on.
    if [[ -f "$OUT/${M}_all_enhancedrecons.pt" ]]; then
        echo "[gpu$G subj0$S] refinement already done, skipping"
    else
        echo "[gpu$G subj0$S] refining: enhanced_recon_inference.py"
        [[ $DRY -eq 1 ]] || CUDA_VISIBLE_DEVICES=$G python enhanced_recon_inference.py \
            --model_name=$M --subj=$S \
            || { echo "[gpu$G subj0$S] refinement FAILED"; return 1; }
    fi

    # Benchmark A metrics.
    if [[ $SKIP_EVAL -eq 1 ]]; then
        echo "[gpu$G subj0$S] eval skipped by request"
    elif [[ -f "tables/${M}_all_enhancedrecons.csv" ]]; then
        echo "[gpu$G subj0$S] eval already done, skipping"
    else
        echo "[gpu$G subj0$S] evaluating: final_evaluations.py"
        # Expected to end in FileNotFoundError on the UMAP tail (left unpatched on purpose, see
        # apply_upstream_patches.sh); both CSVs are already written by then, and the exit status
        # is not the success criterion.
        [[ $DRY -eq 1 ]] || CUDA_VISIBLE_DEVICES=$G python final_evaluations.py \
            --model_name=$M --subj=$S --data_path=$DATA --cache_dir=$DATA \
            --all_recons_path=evals/${M}/${M}_all_enhancedrecons.pt
        if [[ $DRY -eq 0 && ! -f "tables/${M}_all_enhancedrecons.csv" ]]; then
            echo "[gpu$G subj0$S] evaluation FAILED: no CSV produced"; return 1
        fi
    fi

    echo "[gpu$G subj0$S] done $(date '+%F %T')"
}

# ---- split the subjects across the two cards ------------------------------
lane() {
    local G=$1; shift
    for S in "$@"; do one_subject "$G" "$S"; done
}

G0=(); G1=(); i=0
for S in $SUBJECTS; do
    (( i % 2 == 0 )) && G0+=("$S") || G1+=("$S")
    i=$(( i + 1 ))
done

PIDS=()
if [[ ${#G0[@]} -gt 0 ]]; then
    echo "GPU 0: ${G0[*]}  (log: train_logs/logs/infer_gpu0.log)"
    lane 0 "${G0[@]}" > "$LOGS/infer_gpu0.log" 2>&1 &
    PIDS+=($!)
fi
if [[ ${#G1[@]} -gt 0 ]]; then
    echo "GPU 1: ${G1[*]}  (log: train_logs/logs/infer_gpu1.log)"
    lane 1 "${G1[@]}" > "$LOGS/infer_gpu1.log" 2>&1 &
    PIDS+=($!)
fi

FAIL=0
for P in "${PIDS[@]}"; do wait "$P" || FAIL=1; done

echo
echo "=== inference + eval done (failures: $FAIL) ==="
ls -lh evals/finetuned_subj0*_40sess_4096hid/*_all_enhancedrecons.pt 2>/dev/null
wc -l tables/finetuned_subj0*_40sess_4096hid_all_enhancedrecons.csv 2>/dev/null
echo
echo "Next: collect the CSVs in tables/ and compare them with the values of the"
echo "      MindEye2 paper (Table 7)."
