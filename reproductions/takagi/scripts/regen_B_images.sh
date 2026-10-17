#!/usr/bin/env bash
# Regenerate the 1200 NSD-Imagery reconstructions of Takagi, for the qualitative figure.
#
# The images were deleted after the metrics were computed (disk space), and the metrics are not recomputed. Same
# code, weights and seed as stages 1, 2 and 4 of run_full_imagery.sh. Differences, all on purpose: one GPU, four
# subjects in sequence; no stage 3 (the captions of the original run are on disk and .venv_blip no longer exists);
# no stage 5, which would overwrite eval_features_imagery/, the source of the published numbers (the identity check
# writes to eval_features_imagery_regen/); logs get a regen_ prefix, and the original logs stay untouched.
#
# Usage (from the takagi/ root; .venv is not needed in the calling shell):
#   nohup bash regen_B_images.sh > regen_B_images.log 2>&1 &

set -uo pipefail
export OPENBLAS_CORETYPE=SkylakeX

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SUBJECTS=(subj01 subj02 subj05 subj07)
CONDITIONS=(visioncomplex visionsimple imagerycomplex imagerysimple imageryconcepts)
NREP=10
GPU=0
CHECK_SAMPLES=(0 9)

source "$ROOT/.venv/bin/activate"

echo "=== $(date '+%F %T') stage 1: betas ==="
( cd codes/utils && python prepare_nsd_imagery_takagi.py \
      --subject "${SUBJECTS[@]}" \
      --data_root ../../nsd --out_dir ../../mrifeat_imagery \
      --weights_dir ../../decoded ) > regen_prepare_imagery.log 2>&1 \
    || { echo "stage 1 FAILED, see regen_prepare_imagery.log"; exit 1; }

echo "=== $(date '+%F %T') stage 2: scores from betas x weights ==="
( cd codes/utils
  for S in "${SUBJECTS[@]}"; do
      for T in init_latent blip; do
          python decode_imagery_features.py --subject "$S" --target "$T" \
              --imagery_dir ../../mrifeat_imagery --weights_dir ../../decoded \
              --out_dir ../../decoded_imagery || exit 1
      done
  done ) > regen_decode_imagery.log 2>&1 \
    || { echo "stage 2 FAILED, see regen_decode_imagery.log"; exit 1; }

echo "=== $(date '+%F %T') stage 4: reconstruction on GPU $GPU ==="
( cd codes/diffusion_sd1
  for S in "${SUBJECTS[@]}"; do
      for C in "${CONDITIONS[@]}"; do
          CUDA_VISIBLE_DEVICES=$GPU python diffusion_decoding_imagery.py \
              --subject "$S" --condition "$C" --method text \
              --n_iter "$NREP" --gpu 0 --skip_existing \
              --imagery_dir ../../decoded_imagery \
              --out_dir ../../decoded/image-text-imagery \
            || echo "FAILED recon $S $C"
      done
      echo "recon imagery $S done"
  done ) > regen_recon_imagery_gpu${GPU}.log 2>&1
TOT=$(find decoded/image-text-imagery -name '*_[0-9][0-9][0-9].png' ! -name '*_org.png' | wc -l)
echo "  PNG: $TOT (expected $(( ${#SUBJECTS[@]} * ${#CONDITIONS[@]} * 6 * NREP )))"

echo "=== $(date '+%F %T') identity check: features of samples ${CHECK_SAMPLES[*]} ==="
( cd codes/utils
  for S in "${SUBJECTS[@]}"; do
      for M in vision imagery; do
          for N in "${CHECK_SAMPLES[@]}"; do
              CUDA_VISIBLE_DEVICES=$GPU python eval_extract_features_imagery.py \
                  --subject "$S" --modality "$M" --sample "$N" \
                  --feat_dir ../../eval_features_imagery_regen \
                || echo "FAILED features $S $M $N"
          done
      done
  done ) > regen_features.log 2>&1

python - <<'EOF'
import glob, os, numpy as np
new = sorted(glob.glob('eval_features_imagery_regen/subj*/*/*.npy'))
worst = 0.0
for f in new:
    old = f.replace('eval_features_imagery_regen', 'eval_features_imagery')
    a, b = np.load(old).astype(np.float64), np.load(f).astype(np.float64)
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    rel = np.abs(a - b).max() / np.abs(a).max()
    r = min(np.corrcoef(x, y)[0, 1] for x, y in zip(a, b))
    worst = max(worst, rel)
    print(f'{f}: max|diff|/max|old| = {rel:.2e}, min per-image r = {r:.6f}')
print(f'{len(new)} feature files compared, worst relative difference {worst:.2e}')
EOF

echo "=== $(date '+%F %T') DONE ==="
