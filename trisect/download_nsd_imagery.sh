#!/usr/bin/env bash
# Download the NSD-Imagery raw data (subjects 1, 2, 5, 7) that `prepare_betas.py` needs.
#
# Inputs:  HF reesekneeland/umn-imagery (public): nsdimagery_stimuli.pkl3, imagery_stimuli_18.pt and the
#          authors' preprocessed tensors nsd_imagery{,_unnormalized}.pt, the reference for the
#          cross-check of `prepare_nsd_imagery.py`; public NSD S3 bucket: betas_nsdimagery.nii.gz, nsdgeneral.nii.gz
# Outputs: ./data/ (~1.7 GB). Override the root with DATA_ROOT=...
#
# Adapted from the MindEye1 reproduction's `download_nsd_imagery.sh`: plain HTTPS instead of the aws cli,
# because the public S3 objects need no signing.
#
# Usage:
#   bash download_nsd_imagery.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/data}"
HF_BASE="https://huggingface.co/datasets/reesekneeland/umn-imagery/resolve/main"
S3_BASE="https://natural-scenes-dataset.s3.amazonaws.com"

echo "dest: $DATA_ROOT"
mkdir -p "$DATA_ROOT/nsddata_stimuli/stimuli"

echo "-> trial metadata + target stimuli (HuggingFace)"
curl -fSL --retry 5 --retry-delay 5 --speed-limit 2048 --speed-time 30 -C - -o "$DATA_ROOT/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3" \
  "$HF_BASE/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3"
curl -fSL --retry 5 --retry-delay 5 --speed-limit 2048 --speed-time 30 -C - -o "$DATA_ROOT/nsddata_stimuli/stimuli/imagery_stimuli_18.pt" \
  "$HF_BASE/nsddata_stimuli/stimuli/imagery_stimuli_18.pt"

echo "-> authors' preprocessed tensors (reference for prepare_nsd_imagery.py's cross-check)"
for SUBJ in 1 2 5 7; do
  mkdir -p "$DATA_ROOT/reference/preprocessed_data/subject${SUBJ}"
  for F in nsd_imagery.pt nsd_imagery_unnormalized.pt; do
    curl -fSL --retry 5 --retry-delay 5 --speed-limit 2048 --speed-time 30 -C - -o "$DATA_ROOT/reference/preprocessed_data/subject${SUBJ}/$F" \
      "$HF_BASE/preprocessed_data/subject${SUBJ}/$F"
  done
done

for SUBJ in 01 02 05 07; do
  echo "-> imagery betas + ROI mask, subj${SUBJ}"
  mkdir -p "$DATA_ROOT/nsddata_betas/ppdata/subj${SUBJ}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR"
  mkdir -p "$DATA_ROOT/nsddata/ppdata/subj${SUBJ}/func1pt8mm/roi"
  curl -fSL --retry 5 --retry-delay 5 --speed-limit 2048 --speed-time 30 -C - -o "$DATA_ROOT/nsddata_betas/ppdata/subj${SUBJ}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz" \
    "$S3_BASE/nsddata_betas/ppdata/subj${SUBJ}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz"
  curl -fSL --retry 5 --retry-delay 5 --speed-limit 2048 --speed-time 30 -C - -o "$DATA_ROOT/nsddata/ppdata/subj${SUBJ}/func1pt8mm/roi/nsdgeneral.nii.gz" \
    "$S3_BASE/nsddata/ppdata/subj${SUBJ}/func1pt8mm/roi/nsdgeneral.nii.gz"
done

echo "done"
