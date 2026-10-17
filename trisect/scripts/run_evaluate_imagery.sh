#!/usr/bin/env bash
# Score the NSD-Imagery arm of the final model with the Kneeland harness, like the five baselines.
# Runs after run_reconstruct_imagery.sh, on CPU (--device cpu, ~30 min on 9 cores); it can share
# the machine with GPU jobs. Venv: mindeye_II/.venv, the one that scored all five baseline rows.
#
# Outputs, under data/results/ (the model column keeps the run label trisect_b3_h0):
#   trisect/trisect.tsv                         8 image metrics, vision + imagery, 4 subjects
#   trisect/trisect_chance.tsv                  the same with --roll_gt (measured chance)
#   brain_correlation/trisect_brain_corr.tsv    GNet brain correlation (Early / Higher / VisCtx)
#   brain_correlation/trisect_conceptual_brain_corr.tsv   GNet on imageryconcepts, the only metric
#                                               defined there; last stage, its failure affects nothing else
# GNet needs gnet_multisubject.pt, brain_region_masks.hdf5 and the canonical betas (BETAS below).
#
# Usage (from the trisect/ folder):  bash scripts/run_evaluate_imagery.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PROJ="$PWD"

SRC="$PROJ/../mindeye_II/src"          # absolute: stages run after a cd into $SRC
PY="$PROJ/../mindeye_II/.venv/bin/python"
RECON_ROOT="$PROJ/data/imagery_recons_b3_h0"
TEMPLATE='subj{subj02}/{cond}/sample{sample03}/{stim05}.png'
GT="$PROJ/data/nsddata_stimuli/stimuli/imagery_stimuli_18.pt"
STIM_META="$PROJ/../mindeye_II/src/imagery_dataset/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3"
# The canonical betas all five baseline rows were scored against: <root>/subj0N/nsd_imagery.pt.
# data/reference/preprocessed_data/ symlinks into this same tree but names the dirs subjectN,
# which gnet_brain_correlation.py does not accept; the original root is used.
BETAS="$PROJ/../mindeye_I/dataset/preprocessed"
KOUT="$PROJ/data/results/trisect"
GOUT="$PROJ/data/results/brain_correlation"

log() { echo "[$(date +%H:%M:%S)] [evaluate-imagery] $*"; }
mkdir -p "$KOUT" "$GOUT"

cd "$SRC"

# --- 1. the eight image metrics, vision + imagery, 4 subjects ---
if [ -f "$KOUT/trisect.tsv" ]; then
  log "trisect.tsv present, skipping"
else
  log "image metrics (8 columns, vision+imagery, 4 subjects)"
  "$PY" final_evaluations_imagery.py --model template --label trisect_b3_h0 \
      --path_template "$TEMPLATE" --recon_root "$RECON_ROOT" --gt "$GT" \
      --subj 1 2 5 7 --mode vision imagery --gen_rep 10 --device cpu \
      --out "$KOUT/trisect.tsv"
  log "wrote trisect.tsv"
fi

# --- 2. measured chance, the same run with the ground truth rolled inside each block ---
if [ -f "$KOUT/trisect_chance.tsv" ]; then
  log "trisect_chance.tsv present, skipping"
else
  log "measured chance (--roll_gt)"
  "$PY" final_evaluations_imagery.py --model template --label trisect_b3_h0_chance \
      --path_template "$TEMPLATE" --recon_root "$RECON_ROOT" --gt "$GT" \
      --subj 1 2 5 7 --mode vision imagery --gen_rep 10 --roll_gt --device cpu \
      --out "$KOUT/trisect_chance.tsv"
  log "wrote trisect_chance.tsv"
fi

# --- 3. GNet brain correlation, the four default conditions ---
if [ -f "$GOUT/trisect_brain_corr.tsv" ]; then
  log "trisect_brain_corr.tsv present, skipping"
else
  log "GNet brain correlation (imagery/vision x simple/complex)"
  "$PY" gnet_brain_correlation.py compute --model template --label trisect_b3_h0 \
      --path_template "$TEMPLATE" --recon_root "$RECON_ROOT" \
      --betas_root "$BETAS" --stim_meta "$STIM_META" \
      --gnet mindeyev2_dataset/gnet_multisubject.pt \
      --masks mindeyev2_dataset/brain_region_masks.hdf5 \
      --subj 1 2 5 7 --n_samples 10 --device cpu \
      --out "$GOUT/trisect_brain_corr.tsv"
  log "wrote trisect_brain_corr.tsv"
fi

# --- 4. bonus: GNet on the conceptual stimuli, the only metric defined there ---
if [ -f "$GOUT/trisect_conceptual_brain_corr.tsv" ]; then
  log "trisect_conceptual_brain_corr.tsv present, skipping"
else
  log "GNet on imageryconcepts -- failure here does not affect stages 1-3"
  "$PY" gnet_brain_correlation.py compute --model template --label trisect_b3_h0_concepts \
      --path_template "$TEMPLATE" --recon_root "$RECON_ROOT" \
      --betas_root "$BETAS" --stim_meta "$STIM_META" \
      --gnet mindeyev2_dataset/gnet_multisubject.pt \
      --masks mindeyev2_dataset/brain_region_masks.hdf5 \
      --subj 1 2 5 7 --conditions imageryconcepts --n_samples 10 --device cpu \
      --out "$GOUT/trisect_conceptual_brain_corr.tsv" || log "stage 4 FAILED -- see above"
fi

log "=== evaluation done ==="
