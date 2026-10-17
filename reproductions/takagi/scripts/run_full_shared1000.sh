#!/usr/bin/env bash
# Full shared1000 run of Takagi: identification (the original metric) and the bridge to the common harness.
# 982 images x 4 subjects x 5 samples, for both methods (cvpr, text). The NSD-Imagery run is run_full_imagery.sh.
#
# Phases, on 2 GPUs in parallel (GPU 0 = subj01, subj05; GPU 1 = subj02, subj07), each waiting for the previous:
#   1 generate the 982 PNGs   2 img2feat_decoded + identification   3 bridge features + evaluation   4 aggregate
# Phase 2 takes one (subject, method) at a time per GPU and deletes the feature cache once identification.py has
# consumed it (~14 GB each; keeping all eight needs ~112 GB), and runs the 7 features of a combination in parallel
# (serially ~4.5 h). Each writes its own CSV part, because identification.py rewrites identification.csv wholesale.
# --n-iter defaults to 5, upstream's value and not Kneeland's 10: a declared deviation of the bridge. 10 costs
# ~13 h and barely helps (between-sample sd 0.0146 on SSIM against a 0.032 discrepancy). --skip_existing keys on
# {idx:05}_{n_iter-1:03}.png; raising it redoes samples 0-4, bit-identical: a free self-check.
#
# Usage (from the takagi/ root, .venv active; OPENBLAS_CORETYPE is set automatically):
#   bash run_full_shared1000.sh [--only-eval | --allow-partial | --keep-features | --n-iter 10]

set -uo pipefail
export OPENBLAS_CORETYPE=SkylakeX

ONLY_EVAL=0
ALLOW_PARTIAL=0
KEEP_FEATURES=0
NREP=5
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only-eval)     ONLY_EVAL=1; shift ;;
        --allow-partial) ALLOW_PARTIAL=1; shift ;;
        --keep-features) KEEP_FEATURES=1; shift ;;
        --n-iter)        NREP="$2"; shift 2 ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
done
[[ "$NREP" =~ ^[0-9]+$ && $NREP -ge 1 ]] || { echo "--n-iter must be a positive integer"; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SUBJECTS=(subj01 subj02 subj05 subj07)
NFULL=982
# NREP comes from --n-iter above (default 5); it drives the completeness checks, the
# --n_iter passed to diffusion_decoding.py, and --expected_samples on the bridge.

# ---- helpers ------------------------------------------------------------
pngs_complete() {
    # $1 = method, $2 = subject -> 0 if all 982*5 sample PNGs are present
    local SP="decoded/image-$1/$2/samples"
    [[ -d "$SP" ]] || return 1
    local N
    N=$(find "$SP" -maxdepth 1 -name '*_[0-9][0-9][0-9].png' ! -name '*_org.png' | wc -l)
    [[ $N -ge $(( NFULL * NREP )) ]]
}

wait_for_logs() {
    # wait until all background pids in $@ have exited
    local pids=("$@")
    local all_done
    while true; do
        all_done=1
        for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && { all_done=0; break; }; done
        [[ $all_done -eq 1 ]] && break
        sleep 30
    done
}

# ---- phase 1: generate PNGs ---------------------------------------------
if [[ $ONLY_EVAL -eq 0 ]]; then
    echo "=== Phase 1: generating PNGs (diffusion_decoding.py) ==="

    PIDS_GEN=()
    for M in cvpr text; do
        # check if any subject is still incomplete
        NEED_SUBJ=()
        for S in "${SUBJECTS[@]}"; do
            pngs_complete "$M" "$S" || NEED_SUBJ+=("$S")
        done
        [[ ${#NEED_SUBJ[@]} -eq 0 ]] && { echo "  method=$M: all subjects complete, skipping"; continue; }

        # split subjects across 2 GPUs
        G0_SUBJ=(); G1_SUBJ=()
        for i in "${!NEED_SUBJ[@]}"; do
            [[ $(( i % 2 )) -eq 0 ]] && G0_SUBJ+=("${NEED_SUBJ[$i]}") || G1_SUBJ+=("${NEED_SUBJ[$i]}")
        done

        # Launches one background job and appends its PID to PIDS_GEN.
        # It must not print the PID on stdout: capturing it with $(...) would also
        # capture the log line, and the resulting "pid" would never be waited on.
        run_recon() {
            local G=$1 MM=$2; shift 2
            local SS=("$@")
            [[ ${#SS[@]} -eq 0 ]] && return 0
            local LOGFILE="recon_${MM}_full_gpu${G}.log"
            echo "  method=$MM GPU=$G: ${SS[*]}  (log: $LOGFILE)"
            CUDA_VISIBLE_DEVICES=$G nohup bash -c '
                source "$0/.venv/bin/activate"
                export OPENBLAS_CORETYPE=SkylakeX
                cd "$0/codes/diffusion_sd1" || exit 1
                NF=$1; NR=$2; MM=$3; shift 3
                for S in "$@"; do
                    python diffusion_decoding.py \
                        --imgidx 0 --imgidx_end "$NF" --skip_existing \
                        --n_iter "$NR" \
                        --gpu 0 --subject "$S" --method "$MM" \
                      && echo "recon $MM $S done" \
                      || echo "recon $MM $S FAILED (exit $?)"
                done
            ' "$ROOT" "$NFULL" "$NREP" "$MM" "${SS[@]}" > "$ROOT/$LOGFILE" 2>&1 &
            PIDS_GEN+=("$!")
        }

        [[ ${#G0_SUBJ[@]} -gt 0 ]] && run_recon 0 "$M" "${G0_SUBJ[@]}"
        [[ ${#G1_SUBJ[@]} -gt 0 ]] && run_recon 1 "$M" "${G1_SUBJ[@]}"
    done

    if [[ ${#PIDS_GEN[@]} -gt 0 ]]; then
        echo "  waiting for PNG generation (PIDs: ${PIDS_GEN[*]})..."
        wait_for_logs "${PIDS_GEN[@]}"
        echo "  generation done."
    fi

    # quick count check
    echo
    SHORT=()
    for M in cvpr text; do
        for S in "${SUBJECTS[@]}"; do
            SP="decoded/image-$M/$S/samples"
            N=$(find "$SP" -maxdepth 1 -name '*_[0-9][0-9][0-9].png' ! -name '*_org.png' 2>/dev/null | wc -l)
            echo "  $M/$S: $N samples (expected $((NFULL*NREP)))"
            [[ $N -ge $(( NFULL * NREP )) ]] || SHORT+=("$M/$S:$N")
        done
    done

    # Stop here rather than evaluate a half-generated set. Downstream nothing notices:
    # identification.py dies on the first missing .npy (loud, harmless), but the bridge
    # reads however many _org.png happen to be there and produces a plausible-looking
    # PixCorr over 128 images instead of 982; a wrong number that looks like a right one.
    # The usual causes, in order: subjNN_ventral_scores_c.npy missing (kills every cvpr
    # subject instantly), a job killed by hand to free VRAM
    # (exit 143 in recon_*_full_gpu*.log), CUDA OOM.
    if [[ ${#SHORT[@]} -gt 0 && $ALLOW_PARTIAL -eq 0 ]]; then
        echo
        echo "FATAL: ${#SHORT[@]} combination(s) short of $(( NFULL * NREP )) samples:"
        printf '         %s\n' "${SHORT[@]}"
        echo "       Look at recon_*_full_gpu*.log for the reason, fix it, and re-launch:"
        echo "       --skip_existing resumes and regenerates nothing already complete."
        echo "       Pass --allow-partial only if you know why a combination is short."
        exit 1
    fi
fi

# ---- phase 2: benchmark A (img2feat_decoded + identification) -----------
echo
echo "=== Phase 2: benchmark A (img2feat_decoded + identification) ==="

cd "$ROOT/codes/utils"
mkdir -p "$ROOT/results/parts"

PIDS_FEAT=()
for G_IDX in 0 1; do
    SUBJ_SLICE=()
    for i in "${!SUBJECTS[@]}"; do
        [[ $(( i % 2 )) -eq $G_IDX ]] && SUBJ_SLICE+=("${SUBJECTS[$i]}")
    done
    [[ ${#SUBJ_SLICE[@]} -eq 0 ]] && continue

    LOGFILE="$ROOT/benchA_gpu${G_IDX}.log"
    echo "  GPU $G_IDX: ${SUBJ_SLICE[*]}  (log: benchA_gpu${G_IDX}.log)"
    CUDA_VISIBLE_DEVICES=$G_IDX nohup bash -c '
        # bash -c assigns the first argument to $0, not $1.
        ROOT=$0; NFULL=$1; KEEP=$2; shift 2
        source "$ROOT/.venv/bin/activate"
        export OPENBLAS_CORETYPE=SkylakeX
        cd "$ROOT/codes/utils" || exit 1

        FEATS=(inception alexnet5 alexnet12 alexnet18 clip clip_h6 clip_h12)
        CSV="$ROOT/results/identification.csv"
        PARTS="$ROOT/results/parts"

        for S in "$@"; do
            for M in cvpr text; do
                FEATDIR="$ROOT/identification/$M/$S"

                # Per-feature resume: a row already in the merged CSV, or in a part file
                # from an interrupted run, is a finished feature. Costs one grep and can
                # save two hours.
                TODO=()
                for F in "${FEATS[@]}"; do
                    grep -q "^$S,$M,$F,$NFULL," "$CSV" 2>/dev/null && continue
                    grep -q "^$S,$M,$F,$NFULL," "$PARTS/ident_${M}_${S}_${F}.csv" 2>/dev/null && continue
                    TODO+=("$F")
                done
                if [ ${#TODO[@]} -eq 0 ]; then
                    echo "identification $M $S: 7/7 already done, skipping"
                    [ "$KEEP" -eq 0 ] && rm -rf "$FEATDIR"
                    continue
                fi

                # Extract only if the cache is not already complete: 7 .npy per PNG.
                NPNG=$(find "$ROOT/decoded/image-$M/$S/samples" -maxdepth 1 -name "*.png" | wc -l)
                NNPY=$(find "$FEATDIR" -maxdepth 1 -name "*.npy" 2>/dev/null | wc -l)
                if [ "$NNPY" -ge $(( NPNG * 7 )) ] && [ "$NPNG" -gt 0 ]; then
                    echo "img2feat $M $S: cache complete ($NNPY .npy), skipping"
                else
                    python img2feat_decoded.py --gpu 0 --subject "$S" --method "$M" \
                      && echo "img2feat $M $S done" \
                      || { echo "img2feat $M $S FAILED"; continue; }
                fi

                # The 7 features in parallel, one CSV part each (identification.py rewrites
                # its --out wholesale; they must not share a file). Peak RAM is dominated
                # by clip_h6/clip_h12: 982 x 6 x 1.05 MB ~ 6 GB apiece.
                echo "identification $M $S: launching ${#TODO[@]} feature(s) in parallel: ${TODO[*]}"
                IPIDS=()
                for F in "${TODO[@]}"; do
                    python identification.py --usefeat "$F" --subject "$S" --method "$M" \
                        --nimage "$NFULL" --out "$PARTS/ident_${M}_${S}_${F}.csv" \
                        > "$PARTS/ident_${M}_${S}_${F}.log" 2>&1 &
                    IPIDS+=($!)
                done
                IFAIL=0
                for P in "${IPIDS[@]}"; do wait "$P" || IFAIL=1; done

                if [ "$IFAIL" -eq 0 ]; then
                    echo "identification $M $S done"
                    # Only now is the 14 GB cache disposable: every feature that needed it
                    # has produced its row.
                    [ "$KEEP" -eq 0 ] && { rm -rf "$FEATDIR"; echo "freed $FEATDIR"; }
                else
                    echo "identification $M $S FAILED -- keeping $FEATDIR for the retry"
                fi
            done
        done
    ' "$ROOT" "$NFULL" "$KEEP_FEATURES" "${SUBJ_SLICE[@]}" > "$LOGFILE" 2>&1 &
    PIDS_FEAT+=($!)
done

echo "  waiting for feature extraction + identification (PIDs: ${PIDS_FEAT[*]})..."
wait_for_logs "${PIDS_FEAT[@]}"
echo "  phase 2 lanes done."

# Merge the CSV parts back into results/identification.csv. Same row key as
# identification.py (subject, method, usefeat, nimage): a part overwrites its own row and
# nothing else; the rows earned by earlier runs survive.
echo "  merging identification CSV parts..."
python - "$ROOT" <<'PY'
import csv, glob, os, shutil, sys

root = sys.argv[1]
results = os.path.join(root, 'results')
parts = os.path.join(results, 'parts')
main = os.path.join(results, 'identification.csv')

FIELDS = ['subject', 'method', 'usefeat', 'nimage', 'acc', 'acc_sem',
          'acc_per_image', 'timestamp']
FEAT_ORDER = ['inception', 'alexnet5', 'alexnet12', 'alexnet18',
              'clip', 'clip_h6', 'clip_h12']

rows = {}
def key(r):
    return (r.get('subject'), r.get('method'), r.get('usefeat'), r.get('nimage'))

if os.path.exists(main):
    with open(main, newline='') as f:
        for r in csv.DictReader(f):
            rows[key(r)] = r

added = 0
for p in sorted(glob.glob(os.path.join(parts, 'ident_*.csv'))):
    with open(p, newline='') as f:
        for r in csv.DictReader(f):
            rows[key(r)] = r
            added += 1

# The per-image .npy landed next to the part CSV; identification.py stores only its
# basename in acc_per_image; moving it up keeps the reference valid.
for npy in glob.glob(os.path.join(parts, 'acc_*.npy')):
    shutil.move(npy, os.path.join(results, os.path.basename(npy)))

def sortkey(r):
    feat = r.get('usefeat', '')
    rank = FEAT_ORDER.index(feat) if feat in FEAT_ORDER else len(FEAT_ORDER)
    return (r.get('method', ''), r.get('subject', ''), rank, feat)

with open(main, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, restval='', extrasaction='ignore')
    w.writeheader()
    w.writerows(sorted(rows.values(), key=sortkey))

print('  merged {} part row(s); identification.csv now has {} rows'.format(added, len(rows)))
PY

echo "  benchmark A results in $ROOT/results/identification.csv"
column -s, -t "$ROOT/results/identification.csv" 2>/dev/null | head -40

# ---- phase 3: bridge to the evaluation harness --------------------------
echo
echo "=== Phase 3: shared1000 bridge (eval_extract_features + evaluate_reconstruction) ==="

cd "$ROOT/codes/utils"

# GT features (one per subject, small)
for S in "${SUBJECTS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python eval_extract_features_imagery.py \
        --bridge --gt --subject "$S" --method text \
      && echo "  GT features $S done" || echo "  GT features $S FAILED"
done

# Sample features + evaluation on 2 GPUs
PIDS_EVAL=()
for G_IDX in 0 1; do
    SUBJ_SLICE=()
    for i in "${!SUBJECTS[@]}"; do
        [[ $(( i % 2 )) -eq $G_IDX ]] && SUBJ_SLICE+=("${SUBJECTS[$i]}")
    done
    [[ ${#SUBJ_SLICE[@]} -eq 0 ]] && continue

    LOGFILE="$ROOT/ponte_gpu${G_IDX}.log"
    echo "  GPU $G_IDX: ${SUBJ_SLICE[*]}  (log: ponte_gpu${G_IDX}.log)"
    CUDA_VISIBLE_DEVICES=$G_IDX nohup bash -c '
        # bash -c assigns the first argument to $0, not $1.
        ROOT=$0; NR=$1; KEEP=$2; shift 2
        source "$ROOT/.venv/bin/activate"
        export OPENBLAS_CORETYPE=SkylakeX
        cd "$ROOT/codes/utils" || exit 1
        for S in "$@"; do
            for N in $(seq 0 $(( NR - 1 ))); do
                if python eval_extract_features_imagery.py \
                        --bridge --subject "$S" --method text --sample "$N" \
                   && python evaluate_reconstruction_imagery.py \
                        --bridge --subject "$S" --method text --sample "$N"; then
                    # ~560 MB per sample dir at 982 images (alexnet layer 2 is 90% of it);
                    # the per-image TSV that evaluate_reconstruction just wrote is the
                    # artefact that matters, and re-extraction is minutes of GPU.
                    [ "$KEEP" -eq 0 ] && rm -rf "$ROOT/eval_features_imagery/bridge/${S}_text_s${N}"
                else
                    echo "FAILED bridge $S sample $N"
                fi
            done
            echo "bridge $S done"
        done
    ' "$ROOT" "$NREP" "$KEEP_FEATURES" "${SUBJ_SLICE[@]}" > "$LOGFILE" 2>&1 &
    PIDS_EVAL+=($!)
done

echo "  waiting for bridge evaluation (PIDs: ${PIDS_EVAL[*]})..."
wait_for_logs "${PIDS_EVAL[@]}"
echo "  bridge evaluation done."

# ---- phase 4: aggregate -------------------------------------------------
echo
echo "=== Phase 4: aggregate the bridge ==="
mkdir -p "$ROOT/results"
cd "$ROOT/codes/utils"
python aggregate_imagery_metrics.py \
    --bridge --expected_samples "$NREP" \
    --subjects subj01 subj02 subj05 subj07 \
  | tee "$ROOT/results/ponte_shared1000_summary.md"

echo
echo "=== DONE ==="
echo "  Bridge summary:  $ROOT/results/ponte_shared1000_summary.md"
echo "  Benchmark A CSV: $ROOT/results/identification.csv"
echo "  Copy the numbers somewhere safe before deleting any intermediate file."
