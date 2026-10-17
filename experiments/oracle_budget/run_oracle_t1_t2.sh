#!/usr/bin/env bash
# T1 and T2 arms of the oracle budget (Section 8.2) on the proxy set, with the same sweep, seeds and
# evaluation as T0 (run_oracle_t0.sh): the arms are comparable only if nothing else moves.
#
#   T1 = adapter(z_sem, z_struct) from the true proxy targets + raw z_low     -> cost of PCA bases and adapter
#   T2 = same context + PCA d=1024 z_low                                      -> cost of the z_low compression
#   T1nol2 = T1 without the final L2 on z_sem, at T1's winning setting only (adapter not retrained:
#            a sensitivity check, not the L2-free pipeline)
#
# T3 is not here: it needs ridge predictions (run_oracle_shared1000.sh). One queue of (arm, strength,
# mixing) settings keeps both GPUs busy; 18 settings take about 6-7 h. Resumable.
#
# Usage (project root):
#   setsid nohup bash src/run_oracle_t1_t2.sh > log_oracle_t1_t2.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."   # project root: the folder that holds data/ and src/

# PROJECT_ROOT: the folder that holds .uv/ (the uv binary); export it before launching.
export PATH="$PROJECT_ROOT/.uv/bin:$PATH"
export OPENBLAS_CORETYPE=SkylakeX

TARGETS_PY=.venv_targets/bin/python
KNEELAND_PY=.venv_kneeland/bin/python

STRENGTHS=(0.65 0.75 0.85)
MIXINGS=(0.2 0.4 0.6)
GEN_REP=4
SWEEP_DIR=data/results/oracle_budget/setting_sweep
NOL2_TARGETS=data/targets/targets_proxyset_nol2.h5

log() { echo "[$(date +%H:%M:%S)] $*"; }

# T1 and T2 differ by exactly one flag on the reconstruction side.
arm_flags() {
  case "$1" in
    T1)     echo "--oracle-targets all --zlow-raw" ;;
    T2)     echo "--oracle-targets all" ;;
    T1nol2) echo "--oracle-targets all --zlow-raw --proxy-targets $NOL2_TARGETS" ;;
    *)      echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

log "=== oracle T1, T2: start ==="

# --- stage 0: preflight: every input that T0 did not need ---
for f in data/targets/targets_proxyset.h5 data/targets/a_psi_checkpoint.pt \
         data/targets/vdvae_latents_proxy_raw.npy data/targets/pca_zlow/pca_basis.npz \
         data/proxy_set/manifest.json; do
  [ -f "$f" ] || { log "PREFLIGHT FAILED: missing $f"; exit 1; }
done
log "stage 0: preflight OK"

# --- stage 1: smoke test both arms before committing to the sweep ---
for arm in T1 T2; do
  smoke="data/proxy_recons_b3/_smoke_${arm}"
  if [ ! -d "$smoke" ]; then
    log "stage 1: $arm smoke test (block 0, 2 rows, 1 rep)"
    # shellcheck disable=SC2046
    "$TARGETS_PY" src/reconstruct.py --dataset proxy $(arm_flags "$arm") \
      --proxy-blocks 0 --gen-rep 1 --limit-rows 2 --out-dir "$smoke"
    log "stage 1: $arm smoke OK ($(find "$smoke" -name '*.png' | wc -l) images)"
  else
    log "stage 1: $smoke already present, skipping"
  fi
done

# --- stage 2: the shared sweep queue, 2 combos at a time across the 2 GPUs ---
queue=()
for arm in T1 T2; do
  for s in "${STRENGTHS[@]}"; do
    for m in "${MIXINGS[@]}"; do
      queue+=("$arm $s $m")
    done
  done
done
log "stage 2: ${#queue[@]} (arm,strength,mixing) combos, gen_rep=$GEN_REP, 2 GPUs in parallel"

i=0
while [ $i -lt ${#queue[@]} ]; do
  pids=()
  for gpu in 0 1; do
    if [ $i -ge ${#queue[@]} ]; then break; fi
    read -r arm s m <<< "${queue[$i]}"
    outdir="data/proxy_recons_b3/${arm}_s${s}_m${m}"
    if [ -f "$outdir/.recon_done" ]; then
      log "  $arm strength=$s mixing=$m already done, skipping"
      i=$((i+1))
      continue
    fi
    log "  launching $arm strength=$s mixing=$m on GPU$gpu -> $outdir"
    # shellcheck disable=SC2046
    ( CUDA_VISIBLE_DEVICES=$gpu "$TARGETS_PY" src/reconstruct.py \
        --dataset proxy $(arm_flags "$arm") --gen-rep "$GEN_REP" \
        --strength "$s" --mixing "$m" --out-dir "$outdir" \
        > "log_d1_45_recon_${arm}_s${s}_m${m}.log" 2>&1 \
      && touch "$outdir/.recon_done" ) &
    pids+=($!)
    i=$((i+1))
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done
log "stage 2: sweep reconstructions done"

# --- stage 3: eval each combo (CPU, kneeland venv), same wrapper as T0 ---
mkdir -p "$SWEEP_DIR"
for arm in T1 T2; do
  for s in "${STRENGTHS[@]}"; do
    for m in "${MIXINGS[@]}"; do
      label="${arm}_s${s}_m${m}"
      out="$SWEEP_DIR/${label}.tsv"
      if [ -f "$out" ]; then
        log "stage 3: $out already present, skipping"
        continue
      fi
      log "stage 3: evaluating $label"
      "$KNEELAND_PY" src/evaluate_block_set.py \
        --recon-root "data/proxy_recons_b3/${label}" --label "$label" --gen-rep "$GEN_REP" \
        --device cpu --out "$out" > "log_d1_45_eval_${label}.log" 2>&1
    done
  done
done
log "stage 3: all evals done"

# --- stage 4: winner per arm, same criterion as T0 ---
for arm in T1 T2; do
  log "stage 4: selecting winning (strength,mixing) for $arm"
  "$KNEELAND_PY" src/pick_oracle_constants.py --arm "$arm" \
    --sweep-dir "$SWEEP_DIR" \
    --out "data/results/oracle_budget/${arm}.tsv" \
    --winner-json "data/results/oracle_budget/${arm}_setting_choice.json"
done

# --- stage 5 (extra): the L2 arm, at T1's winning setting only ---
if [ ! -f "$NOL2_TARGETS" ]; then
  log "stage 5a: extracting L2-free proxy targets -> $NOL2_TARGETS"
  "$TARGETS_PY" src/extract_proxy_targets.py --no-zsem-l2 --out "$NOL2_TARGETS"
else
  log "stage 5a: $NOL2_TARGETS already present, skipping"
fi

read -r WS WM < <("$KNEELAND_PY" - <<'PYEOF'
import json
w = json.load(open("data/results/oracle_budget/T1_setting_choice.json"))["winner"]
print(w["strength"], w["mixing"])
PYEOF
)
label="T1nol2_s${WS}_m${WM}"
outdir="data/proxy_recons_b3/${label}"
log "stage 5b: T1nol2 at T1's winner strength=$WS mixing=$WM"
if [ ! -f "$outdir/.recon_done" ]; then
  # shellcheck disable=SC2046
  "$TARGETS_PY" src/reconstruct.py --dataset proxy $(arm_flags T1nol2) \
    --gen-rep "$GEN_REP" --strength "$WS" --mixing "$WM" --out-dir "$outdir" \
    > "log_d1_45_recon_${label}.log" 2>&1 && touch "$outdir/.recon_done"
else
  log "stage 5b: $outdir already done, skipping"
fi
if [ ! -f "$SWEEP_DIR/${label}.tsv" ]; then
  log "stage 5c: evaluating $label"
  "$KNEELAND_PY" src/evaluate_block_set.py \
    --recon-root "$outdir" --label "$label" --gen-rep "$GEN_REP" \
    --device cpu --out "$SWEEP_DIR/${label}.tsv" > "log_d1_45_eval_${label}.log" 2>&1
fi
"$KNEELAND_PY" src/pick_oracle_constants.py --arm T1nol2 \
  --sweep-dir "$SWEEP_DIR" \
  --out data/results/oracle_budget/T1nol2.tsv \
  --winner-json data/results/oracle_budget/T1nol2_setting_choice.json

log "=== oracle T1, T2, T1nol2: done ==="
