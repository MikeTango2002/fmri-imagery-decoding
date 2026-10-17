#!/usr/bin/env bash
# T0 arm of the oracle budget (Section 8.2): true Versatile Diffusion context + raw z_low (no PCA),
# on the proxy set, with a strength x mixing sweep (3 x 3 settings).
#
# Outputs: the sweep reconstructions, one evaluation TSV per setting, T0.tsv and T0_setting_choice.json
#
# Stages: captions, true context, raw VDVAE latents, two smoke tests, the sweep (two settings at a time,
# one per GPU), the evaluation of each setting, the choice of the winner (pick_oracle_constants.py).
# Resumable: every stage skips its own finished output.
#
# Usage (project root, both GPUs free):
#   setsid nohup bash src/run_oracle_t0.sh > log_oracle_t0.log 2>&1 < /dev/null &
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

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== oracle T0: start ==="

# --- stage 1: captions (proxy-simple templated, proxy-complex via BLIP-2 flan-t5-xxl) ---
if [ ! -f data/proxy_set/captions.npy ]; then
  log "stage 1: captions.npy missing -- generating"
  "$TARGETS_PY" -c "import sentencepiece" 2>/dev/null \
    || uv pip install --python "$TARGETS_PY" sentencepiece
  "$TARGETS_PY" src/caption_proxy_set.py
else
  log "stage 1: data/proxy_set/captions.npy already present, skipping"
fi

# --- stage 2: true VD context (T0's context arm) ---
if [ ! -f data/targets/a_psi_true_context_proxyset.h5 ]; then
  log "stage 2: extracting true VD context for the proxy set"
  "$TARGETS_PY" src/extract_proxy_vd_context.py
else
  log "stage 2: a_psi_true_context_proxyset.h5 already present, skipping"
fi

# --- stage 3: raw (91168-dim, no PCA) VDVAE latents (T0's init arm) ---
if [ ! -f data/targets/vdvae_latents_proxy_raw.npy ]; then
  log "stage 3: extracting raw VDVAE latents for the proxy set"
  "$TARGETS_PY" src/extract_proxy_vdvae_latents.py
else
  log "stage 3: vdvae_latents_proxy_raw.npy already present, skipping"
fi

# --- stage 4: two smoke tests before committing to the sweep ---
if [ ! -d data/proxy_recons_b3/_smoke ]; then
  log "stage 4a: reconstruction smoke test (block 0, 2 rows, 1 rep)"
  "$TARGETS_PY" src/reconstruct.py --dataset proxy \
    --oracle-targets ctx --zlow-raw --proxy-blocks 0 --gen-rep 1 --limit-rows 2 \
    --out-dir data/proxy_recons_b3/_smoke
  n=$(find data/proxy_recons_b3/_smoke -name '*.png' | wc -l)
  log "stage 4a: smoke test OK ($n images)"
else
  log "stage 4a: data/proxy_recons_b3/_smoke already present, skipping"
fi

if [ ! -f data/results/oracle_budget/setting_sweep/_smoke_gt_check.tsv ]; then
  log "stage 4b: evaluate_block_set.py wrapper smoke test (recon == ground truth, block 0 only)"
  "$TARGETS_PY" - <<'PYEOF'
from pathlib import Path
import torch

gt = torch.load("data/proxy_set/blocks/block00_gt.pt")  # (18,3,425,425) in [0,1]
out = Path("data/proxy_recons_b3/_smoke_gt_check")
for cond, base in (("imagerysimple", 0), ("imagerycomplex", 6)):
    d = out / "subj00" / cond / "sample000"
    d.mkdir(parents=True, exist_ok=True)
    for stim in range(6):
        arr = (gt[base + stim].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
        from PIL import Image
        Image.fromarray(arr).save(d / f"{stim:05d}.png")
print("wrote smoke-check reconstructions (== ground truth) under", out)
PYEOF
  "$KNEELAND_PY" src/evaluate_block_set.py \
    --recon-root data/proxy_recons_b3/_smoke_gt_check --label smoke_gt_check \
    --gen-rep 1 --blocks 0 --device cpu \
    --out data/results/oracle_budget/setting_sweep/_smoke_gt_check.tsv
  log "stage 4b: expect PixCorr=SSIM=1.000, AlexNet/InceptionV3/CLIP=0.9091 (10/11) above -- "
  log "  check the log of this script before trusting the sweep that follows"
else
  log "stage 4b: _smoke_gt_check.tsv already present, skipping"
fi

# --- stage 5: strength x mixing sweep, 2 combos in parallel across the 2 free GPUs ---
combos=()
for s in "${STRENGTHS[@]}"; do
  for m in "${MIXINGS[@]}"; do
    combos+=("$s $m")
  done
done
log "stage 5: ${#combos[@]} (strength,mixing) combos, gen_rep=$GEN_REP, 2 GPUs in parallel"

i=0
while [ $i -lt ${#combos[@]} ]; do
  pids=()
  for gpu in 0 1; do
    if [ $i -ge ${#combos[@]} ]; then break; fi
    read -r s m <<< "${combos[$i]}"
    outdir="data/proxy_recons_b3/T0_s${s}_m${m}"
    if [ -f "$outdir/.recon_done" ]; then
      log "  strength=$s mixing=$m already done, skipping"
      i=$((i+1))
      continue
    fi
    log "  launching strength=$s mixing=$m on GPU$gpu -> $outdir"
    ( CUDA_VISIBLE_DEVICES=$gpu "$TARGETS_PY" src/reconstruct.py \
        --dataset proxy --oracle-targets ctx --zlow-raw --gen-rep "$GEN_REP" \
        --strength "$s" --mixing "$m" --out-dir "$outdir" \
        > "log_d1_3_recon_s${s}_m${m}.log" 2>&1 \
      && touch "$outdir/.recon_done" ) &
    pids+=($!)
    i=$((i+1))
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done
log "stage 5: sweep reconstructions done"

# --- stage 6: eval each combo (CPU, mindeye_II/kneeland venv) ---
mkdir -p data/results/oracle_budget/setting_sweep
for s in "${STRENGTHS[@]}"; do
  for m in "${MIXINGS[@]}"; do
    label="T0_s${s}_m${m}"
    out="data/results/oracle_budget/setting_sweep/${label}.tsv"
    if [ -f "$out" ]; then
      log "stage 6: $out already present, skipping"
      continue
    fi
    log "stage 6: evaluating $label"
    "$KNEELAND_PY" src/evaluate_block_set.py \
      --recon-root "data/proxy_recons_b3/${label}" --label "$label" --gen-rep "$GEN_REP" \
      --device cpu --out "$out" > "log_d1_3_eval_${label}.log" 2>&1
  done
done
log "stage 6: all evals done"

# --- stage 7: pick the winning setting, write oracle_budget/T0.tsv ---
log "stage 7: selecting winning (strength,mixing) and writing oracle_budget/T0.tsv"
"$KNEELAND_PY" src/pick_oracle_constants.py \
  --sweep-dir data/results/oracle_budget/setting_sweep \
  --out data/results/oracle_budget/T0.tsv \
  --winner-json data/results/oracle_budget/T0_setting_choice.json

log "=== oracle T0: done ==="
