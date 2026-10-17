#!/usr/bin/env bash
# T2 arm on shared1000 at the Brain-Diffuser operating point, strength 0.75 and mixing 0.4 (Section 8.2).
# The T3 arm at these constants is already on disk (trisect/shared1000/S1K_T3_p3_subj0N.tsv); this
# is the only arm needed to read T2 - T3 at matched constants.
#
# Outputs (new *_p4b names; the T2 arm at 0.85 / 0.6 stays where it is):
#   images        data/nsdcore_recons_b3/S1K_T2_p4b_s0.75_m0.4/
#   per-arm TSV   data/results/oracle_budget/setting_sweep/S1K_T2_p4b_s0.75_m0.4.tsv
#   arm TSV       data/results/oracle_budget/S1K_T2_p4b.tsv
#
# T2 uses no voxels (--oracle-targets all); it is one run, not four; --subj 1 only gives the
# shared1000 id order, identical across subjects. About 1-2 h on one GPU (83 blocks x 12 x 4 repetitions).
#
# Usage (project root):
#   setsid nohup bash src/run_oracle_t2_shared1000_paper_constants.sh > run_oracle_t2_paper_constants.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # project root (the folder that holds data/)

export OPENBLAS_CORETYPE=SkylakeX   # some CPUs miscompute large matmuls without it

TARGETS_PY=.venv_targets/bin/python
KNEELAND_PY=.venv_kneeland/bin/python

BLOCKSET=data/shared1000_blockset
RECON_ROOT=data/nsdcore_recons_b3
SWEEP_DIR=data/results/oracle_budget/setting_sweep
GEN_REP=4
STRENGTH=0.75         # matches S1K_T3_p3 and the constants of Brain-Diffuser itself
MIXING=0.4
GPU=0

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== S1K_T2 at strength=$STRENGTH mixing=$MIXING (matched with S1K_T3_p3) ==="

for f in "$BLOCKSET/manifest.json" \
         "data/results/trisect/shared1000/S1K_T3_p3_subj01.tsv" \
         "data/results/trisect/shared1000/S1K_T3_p3_subj02.tsv" \
         "data/results/trisect/shared1000/S1K_T3_p3_subj05.tsv" \
         "data/results/trisect/shared1000/S1K_T3_p3_subj07.tsv" \
         data/targets/targets.h5 data/targets/pca_zlow/pca_basis.npz; do
  [ -f "$f" ] || { log "PREFLIGHT FAILED: missing $f"; exit 1; }
done
log "preflight OK"

label="S1K_T2_p4b_s${STRENGTH}_m${MIXING}"
outdir="$RECON_ROOT/${label}"

if [ -f "$outdir/.recon_done" ]; then
  log "stage 1: $label already done, skipping"
else
  log "stage 1: $label (83 blocks x 12 stim x $GEN_REP reps, GPU$GPU)"
  CUDA_VISIBLE_DEVICES=$GPU "$TARGETS_PY" src/reconstruct.py \
      --dataset nsdcore --harness-blocks --blockset-dir "$BLOCKSET" --subj 1 \
      --oracle-targets all --gen-rep "$GEN_REP" \
      --strength "$STRENGTH" --mixing "$MIXING" \
      --out-dir "$outdir" > "log_p4b_recon.log" 2>&1
  touch "$outdir/.recon_done"
  log "stage 1: done"
fi

final_out="data/results/oracle_budget/S1K_T2_p4b.tsv"
if [ -f "$final_out" ]; then
  log "stage 2: $final_out already present, skipping"
else
  log "stage 2: evaluating $label"
  "$KNEELAND_PY" src/evaluate_block_set.py \
      --recon-root "$outdir" --proxy-dir "$BLOCKSET" \
      --label "$label" --gen-rep "$GEN_REP" --device cpu \
      --out "$SWEEP_DIR/${label}.tsv" > "log_p4b_eval.log" 2>&1
  "$KNEELAND_PY" src/pick_oracle_constants.py --arm "S1K_T2_p4b" --sweep-dir "$SWEEP_DIR" \
      --out "$final_out" --winner-json "data/results/oracle_budget/S1K_T2_p4b_setting_choice.json" \
      >> "log_p4b_eval.log" 2>&1
  log "stage 2: wrote $final_out"
fi

log "stage 3: T2-T3 at matched constants (s=0.75 m=0.4)"
"$KNEELAND_PY" src/oracle_budget_fixed_constants.py \
    --out data/results/oracle_budget/budget_shared1000_T2_T3_matched_setting.tsv

log "=== done -- the number to report is in data/results/oracle_budget/budget_shared1000_T2_T3_matched_setting.tsv ==="
