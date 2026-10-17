#!/usr/bin/env bash
# The four oracle arms on shared1000, which has brain data and is outside every fit (Section 8.2).
# It gives the third budget term, T2 - T3, which the proxy set cannot: no subject saw a proxy image.
#
# Inputs:  83 blocks of 12 (996 images, make_shared1000_blockset.py), the winner JSONs of run_oracle_t1_t2.sh,
#          data/dev_blockset_subj01 from run_oracle_shared1000_prep.sh
# Outputs: S1K_T0, S1K_T1, S1K_T2 (1 run each, no voxels), S1K_T3 (4 runs, one per subject), gen_rep=4,
#          scored by evaluate_block_set.py with the same 11-distractor pool as the proxy arms
#
# shared1000 is the measurement set; no constant is chosen on it: T0-T2 reuse their best proxy setting,
# T3 is selected on the dev hold-out (same 3x3 grid, 25 blocks, gen_rep=2). Without the dev split it stops.
#
# Usage: setsid nohup ./src/run_oracle_shared1000.sh > run_oracle_shared1000.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # project root (the folder that holds data/)

# PROJECT_ROOT: the folder that holds .uv/ (the uv binary); export it before launching.
export PATH="$PROJECT_ROOT/.uv/bin:$PATH"
export OPENBLAS_CORETYPE=SkylakeX

TARGETS_PY=.venv_targets/bin/python
KNEELAND_PY=.venv_kneeland/bin/python

BLOCKSET=data/shared1000_blockset
DEV_BLOCKSET=data/dev_blockset_subj01
DEV_PRED_DIR=data/betas_trisect/nsdcore_predictions_dev
SWEEP_DIR=data/results/oracle_budget/setting_sweep      # one sweep directory for every arm, as for the proxy sweeps
RECON_ROOT=data/nsdcore_recons_b3
GEN_REP=4
STRENGTHS=(0.65 0.75 0.85)
MIXINGS=(0.2 0.4 0.6)
DEV_BLOCKS=25          # dev selection: 25 blocks of 12 = 300 images
DEV_GEN_REP=2
SUBJECTS=(1 2 5 7)
GPUS=(1)               # GPU1 only: GPU0 was in use by another job
S1K_CTX=data/targets/a_psi_true_context_shared1000.h5  # S1K_T0 needs this, see stage 1.5

log() { echo "[$(date +%H:%M:%S)] $*"; }

arm_flags() {
  case "$1" in
    S1K_T0) echo "--oracle-targets ctx --zlow-raw --a-psi-context $S1K_CTX" ;;
    S1K_T1) echo "--oracle-targets all --zlow-raw" ;;
    S1K_T2) echo "--oracle-targets all" ;;
    S1K_T3) echo "" ;;                                  # nothing oracle: the full pipeline
    *)      echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

# winner_of <winner-json> -> "<strength> <mixing>"
winner_of() {
  "$KNEELAND_PY" -c "
import json, sys
w = json.load(open(sys.argv[1]))['winner']
print(w['strength'], w['mixing'])" "$1"
}

log "=== oracle arms on shared1000: start ==="

# --- stage 0: preflight ---
for f in "$BLOCKSET/manifest.json" "$DEV_BLOCKSET/manifest.json" \
         data/results/oracle_budget/T0_setting_choice.json \
         data/results/oracle_budget/T1_setting_choice.json \
         data/results/oracle_budget/T2_setting_choice.json \
         data/targets/targets.h5 data/targets/a_psi_true_context.h5 \
         data/targets/vdvae_latents_31l.npy data/targets/pca_zlow/pca_basis.npz; do
  [ -f "$f" ] || { log "PREFLIGHT FAILED: missing $f"; exit 1; }
done
for s in "${SUBJECTS[@]}"; do
  [ -f "data/betas_trisect/nsdcore_predictions/subj0${s}/img73k_ids.npy" ] \
    || { log "PREFLIGHT FAILED: no shared1000 predictions for subj0${s}"; exit 1; }
done
log "stage 0: preflight OK"

# --- stage 1: the constants of T3, selected on dev (never on shared1000) ---
if [ ! -f data/results/oracle_budget/devT3_setting_choice.json ]; then
  log "stage 1: dev sweep for the constants of T3 ($DEV_BLOCKS blocks, gen_rep=$DEV_GEN_REP)"
  combos=()
  for s in "${STRENGTHS[@]}"; do for m in "${MIXINGS[@]}"; do combos+=("$s $m"); done; done
  blocks=$(seq 0 $((DEV_BLOCKS - 1)) | tr '\n' ' ')
  i=0
  while [ $i -lt ${#combos[@]} ]; do
    pids=()
    for gpu in "${GPUS[@]}"; do
      if [ $i -ge ${#combos[@]} ]; then break; fi
      read -r s m <<< "${combos[$i]}"
      outdir="$RECON_ROOT/devT3_s${s}_m${m}"
      if [ ! -f "$outdir/.recon_done" ]; then
        log "  dev T3 s=$s m=$m on GPU$gpu"
        # shellcheck disable=SC2086
        ( CUDA_VISIBLE_DEVICES=$gpu "$TARGETS_PY" src/reconstruct.py \
            --dataset nsdcore --harness-blocks --blockset-dir "$DEV_BLOCKSET" \
            --pred-dir "$DEV_PRED_DIR" --subj 1 \
            --limit-rows $((DEV_BLOCKS * 12)) --gen-rep "$DEV_GEN_REP" \
            --strength "$s" --mixing "$m" --out-dir "$outdir" \
            > "log_d1_8_recon_devT3_s${s}_m${m}.log" 2>&1 \
          && touch "$outdir/.recon_done" ) &
        pids+=($!)
      fi
      i=$((i+1))
    done
    for pid in "${pids[@]:-}"; do [ -n "$pid" ] && wait "$pid"; done
  done
  for s in "${STRENGTHS[@]}"; do
    for m in "${MIXINGS[@]}"; do
      label="devT3_s${s}_m${m}"
      [ -f "$SWEEP_DIR/${label}.tsv" ] && continue
      log "  evaluating $label"
      # shellcheck disable=SC2086
      "$KNEELAND_PY" src/evaluate_block_set.py --recon-root "$RECON_ROOT/${label}" \
        --proxy-dir "$DEV_BLOCKSET" --label "$label" --gen-rep "$DEV_GEN_REP" \
        --blocks $blocks --device cpu --out "$SWEEP_DIR/${label}.tsv" \
        > "log_d1_8_eval_${label}.log" 2>&1
    done
  done
  "$KNEELAND_PY" src/pick_oracle_constants.py --arm devT3 --sweep-dir "$SWEEP_DIR" \
    --out data/results/oracle_budget/devT3.tsv \
    --winner-json data/results/oracle_budget/devT3_setting_choice.json
else
  log "stage 1: devT3_setting_choice.json already present, skipping the dev sweep"
fi

read -r T0S T0M < <(winner_of data/results/oracle_budget/T0_setting_choice.json)
read -r T1S T1M < <(winner_of data/results/oracle_budget/T1_setting_choice.json)
read -r T2S T2M < <(winner_of data/results/oracle_budget/T2_setting_choice.json)
read -r T3S T3M < <(winner_of data/results/oracle_budget/devT3_setting_choice.json)
log "constants: T0 $T0S/$T0M  T1 $T1S/$T1M  T2 $T2S/$T2M (proxy sweeps)  T3 $T3S/$T3M (dev sweep)"

# --- stage 1.5: the true VD context of S1K_T0, missing from a_psi_true_context.h5 (that file
# deliberately excludes shared1000, see the docstring of extract_shared1000_vd_context.py) ---
if [ ! -f "$S1K_CTX" ]; then
  log "stage 1.5: extracting shared1000 VD true context for S1K_T0 -> $S1K_CTX"
  "$TARGETS_PY" src/extract_shared1000_vd_context.py --out "$S1K_CTX" \
    > log_d1_8_extract_shared1000_ctx.log 2>&1
else
  log "stage 1.5: $S1K_CTX already present, skipping"
fi

# --- stage 2: the 7 measurement runs, 2 at a time across the 2 GPUs ---
# "arm subj strength mixing"; T0/T1/T2 have no voxels, and one run stands for every subject and
# --subj 1 only supplies the shared1000 id order (identical for all four subjects, asserted by
# make_shared1000_blockset.py).
queue=("S1K_T0 1 $T0S $T0M" "S1K_T1 1 $T1S $T1M" "S1K_T2 1 $T2S $T2M")
for s in "${SUBJECTS[@]}"; do queue+=("S1K_T3 $s $T3S $T3M"); done

labels=()
i=0
while [ $i -lt ${#queue[@]} ]; do
  pids=()
  for gpu in "${GPUS[@]}"; do
    if [ $i -ge ${#queue[@]} ]; then break; fi
    read -r arm subj s m <<< "${queue[$i]}"
    if [ "$arm" = "S1K_T3" ]; then label="${arm}_subj0${subj}_s${s}_m${m}"; else label="${arm}_s${s}_m${m}"; fi
    labels+=("$label")
    outdir="$RECON_ROOT/${label}"
    if [ -f "$outdir/.recon_done" ]; then
      log "  $label already done, skipping"
      i=$((i+1)); continue
    fi
    log "  launching $label on GPU$gpu"
    # shellcheck disable=SC2046,SC2086
    ( CUDA_VISIBLE_DEVICES=$gpu "$TARGETS_PY" src/reconstruct.py \
        --dataset nsdcore --harness-blocks --blockset-dir "$BLOCKSET" --subj "$subj" \
        $(arm_flags "$arm") --gen-rep "$GEN_REP" --strength "$s" --mixing "$m" \
        --out-dir "$outdir" > "log_d1_8_recon_${label}.log" 2>&1 \
      && touch "$outdir/.recon_done" ) &
    pids+=($!)
    i=$((i+1))
  done
  for pid in "${pids[@]:-}"; do [ -n "$pid" ] && wait "$pid"; done
done
log "stage 2: all reconstructions done"

# --- stage 3: eval, two at a time (CPU-bound, the machine has the cores) ---
pending=()
for label in "${labels[@]}"; do
  out="$SWEEP_DIR/${label}.tsv"
  if [ -f "$out" ]; then log "stage 3: $out already present, skipping"; continue; fi
  log "stage 3: evaluating $label"
  ( "$KNEELAND_PY" src/evaluate_block_set.py --recon-root "$RECON_ROOT/${label}" \
      --proxy-dir "$BLOCKSET" --label "$label" --gen-rep "$GEN_REP" \
      --device cpu --out "$out" > "log_d1_8_eval_${label}.log" 2>&1 ) &
  pending+=($!)
  if [ ${#pending[@]} -ge 2 ]; then
    for pid in "${pending[@]}"; do wait "$pid"; done
    pending=()
  fi
done
for pid in "${pending[@]:-}"; do [ -n "$pid" ] && wait "$pid"; done
log "stage 3: all evals done"

# --- stage 4: one oracle_budget/<arm>.tsv per arm, relabelled, a 1-combo "sweep" per arm ---
for arm in S1K_T0 S1K_T1 S1K_T2; do
  "$KNEELAND_PY" src/pick_oracle_constants.py --arm "$arm" --sweep-dir "$SWEEP_DIR" \
    --out "data/results/oracle_budget/${arm}.tsv" \
    --winner-json "data/results/oracle_budget/${arm}_setting_choice.json"
done
for s in "${SUBJECTS[@]}"; do
  arm="S1K_T3_subj0${s}"
  "$KNEELAND_PY" src/pick_oracle_constants.py --arm "$arm" --sweep-dir "$SWEEP_DIR" \
    --out "data/results/oracle_budget/${arm}.tsv" \
    --winner-json "data/results/oracle_budget/${arm}_setting_choice.json"
done

log "=== done -- oracle_budget.py reads oracle_budget/S1K_*.tsv (the subject lives in the model column, "
log "    NOT in the subj column, which holds the block index) ==="
