#!/usr/bin/env bash
# Structural-factor ablation on the proxy set (Section 8.1.4, Table 8.4): does z_struct (from DINOv2) matter end to end?
#
# Inputs:  the proxy set with true factors; the baseline is oracle_budget/T2.tsv, not re-run
# Outputs: structural_factor/T2_z_struct_zero.tsv and T2_z_struct_shuffle.tsv (and their reconstructions)
#
#   z_struct_zero     z_struct := 0 in the whitened PCA space (the mean): how much does z_struct contribute
#   z_struct_shuffle  z_struct permuted across the stimuli of each block, fixed seed: does the adapter use it,
#                     and is it misled by a wrong one
#
# Both at the winning combination of the oracle budget (strength 0.85, mixing 0.6, gen_rep 4, the seeds of the
# baseline). One GPU (index 1), sequential, not the 2-GPU queue of run_oracle_t1_t2.sh.
#
# Usage: setsid nohup bash run_structural_ablation.sh > log.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # project folder, the one that holds src/ and data/

# PROJECT_ROOT: folder that holds the shared .uv/ (the uv binary is not on the default PATH)
export PATH="${PROJECT_ROOT}/.uv/bin:$PATH"
export OPENBLAS_CORETYPE=SkylakeX

TARGETS_PY=.venv_targets/bin/python
KNEELAND_PY=.venv_kneeland/bin/python

STRENGTH=0.85
MIXING=0.6
GEN_REP=4
SWEEP_DIR=data/results
BASELINE="$SWEEP_DIR/oracle_budget/T2.tsv"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== structural ablation (z_struct_zero, z_struct_shuffle @ T2): start ==="

# --- stage 0: preflight ---
for f in data/targets/targets_proxyset.h5 data/targets/a_psi_checkpoint.pt \
         data/targets/vdvae_latents_proxy_raw.npy data/targets/pca_zlow/pca_basis.npz \
         data/proxy_set/manifest.json "$BASELINE"; do
  [ -f "$f" ] || { log "PREFLIGHT FAILED: missing $f"; exit 1; }
done
log "stage 0: preflight OK"

# --- stage 1: smoke test both arms before committing to the paid run ---
for arm in z_struct_zero z_struct_shuffle; do
  smoke="data/proxy_recons_b3/_smoke_${arm}"
  if [ ! -d "$smoke" ]; then
    log "stage 1: $arm smoke test (block 0, 2 rows, 1 rep)"
    CUDA_VISIBLE_DEVICES=1 "$TARGETS_PY" src/reconstruct.py --dataset proxy \
      --oracle-targets all --strength "$STRENGTH" --mixing "$MIXING" \
      --ablate-factor "$arm" --proxy-blocks 0 --gen-rep 1 --limit-rows 2 --out-dir "$smoke"
    log "stage 1: $arm smoke OK ($(find "$smoke" -name '*.png' | wc -l) images)"
  else
    log "stage 1: $smoke already present, skipping"
  fi
done

# --- stage 2: the two full arms, sequential on GPU1 ---
for arm in z_struct_zero z_struct_shuffle; do
  outdir="data/proxy_recons_b3/T2_${arm}"
  if [ -f "$outdir/.recon_done" ]; then
    log "stage 2: $arm already done, skipping"
    continue
  fi
  log "stage 2: launching $arm on GPU1 -> $outdir"
  CUDA_VISIBLE_DEVICES=1 "$TARGETS_PY" src/reconstruct.py --dataset proxy \
    --oracle-targets all --strength "$STRENGTH" --mixing "$MIXING" --gen-rep "$GEN_REP" \
    --ablate-factor "$arm" --out-dir "$outdir" \
    > "log_d1_9_recon_${arm}.log" 2>&1 && touch "$outdir/.recon_done"
  log "stage 2: $arm reconstruction done"
done

# --- stage 3: eval each arm (CPU, kneeland venv), same wrapper as the other oracle arms ---
declare -A OUT_TSV=( [z_struct_zero]="$SWEEP_DIR/structural_factor/T2_z_struct_zero.tsv"
                      [z_struct_shuffle]="$SWEEP_DIR/structural_factor/T2_z_struct_shuffle.tsv" )
for arm in z_struct_zero z_struct_shuffle; do
  out="${OUT_TSV[$arm]}"
  if [ -f "$out" ]; then
    log "stage 3: $out already present, skipping"
    continue
  fi
  log "stage 3: evaluating $arm -> $out"
  "$KNEELAND_PY" src/evaluate_block_set.py \
    --recon-root "data/proxy_recons_b3/T2_${arm}" --label "T2_${arm}" --gen-rep "$GEN_REP" \
    --device cpu --out "$out" > "log_d1_9_eval_${arm}.log" 2>&1
done
log "stage 3: all evals done"

# --- stage 4: paired CI, primary_mean on proxy-simple/complex separately, vs oracle_budget/T2.tsv ---
log "stage 4: bootstrap CI -> $SWEEP_DIR/structural_factor/ablation_bootstrap_ci.tsv"
"$KNEELAND_PY" src/analyze_structural_ablation.py \
  --baseline "$BASELINE" \
  --zero "${OUT_TSV[z_struct_zero]}" \
  --shuffle "${OUT_TSV[z_struct_shuffle]}" \
  --out "$SWEEP_DIR/structural_factor/ablation_bootstrap_ci.tsv" > log_d1_9_analyze.log 2>&1
log "stage 4: done"

log "=== structural ablation: done ==="
