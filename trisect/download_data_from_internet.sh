#!/usr/bin/env bash
# Download the NSD-derived files TRISECT needs from their original sources (same recipe as the MindEye2
# reproduction).
#
# Inputs:  nsd_expdesign.mat (public NSD S3 bucket); coco_images_224_float16.hdf5, COCO_73k_subj_indices.hdf5
#          and, with --betas, betas_all_subj0{1,2,5,7}_fp32_renorm.hdf5 (HF pscotti/mindeyev2, gated);
#          COCO_73k_annots_curated.npy (HF pscotti/naturalscenesdataset, gated)
# Outputs: ./data/{mindeyev2_dataset,nsddata,annots}/ (~21.4 GB, ~6.4 GB more with --betas)
#
# Needs the venv active (huggingface_hub), curl, and a Hugging Face token in ../HF_thesis_token.txt
# (override with TOKEN_FILE=...). The token is never part of the repository.
#
# Usage:
#   bash download_data_from_internet.sh --betas
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DEST:-$SCRIPT_DIR/data}"
TOKEN_FILE="${TOKEN_FILE:-$SCRIPT_DIR/../HF_thesis_token.txt}"

if [[ ! -f "$TOKEN_FILE" ]]; then
  echo "HF token not found at $TOKEN_FILE (needed for the gated pscotti/mindeyev2 and pscotti/naturalscenesdataset repos)" >&2
  exit 1
fi
export HF_TOKEN
HF_TOKEN="$(cat "$TOKEN_FILE")"

mkdir -p "$DEST/mindeyev2_dataset" "$DEST/nsddata/experiments/nsd" "$DEST/annots"

echo "dest: $DEST"

# trial -> 73k image index (subjectim + masterordering), rebuilt and cross-checked by
# prepare_nsdcore_index.py against COCO_73k_subj_indices.hdf5 below. Public bucket: a plain HTTPS GET
# on the virtual-hosted URL returns 200, no --no-sign-request needed.
curl -fSL -o "$DEST/nsddata/experiments/nsd/nsd_expdesign.mat" \
  "https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/nsd/nsd_expdesign.mat"

# 5 human COCO captions per image (the human source of z_sem^txt)
curl -fSL -H "Authorization: Bearer $HF_TOKEN" \
  -o "$DEST/annots/COCO_73k_annots_curated.npy" \
  "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/COCO_73k_annots_curated.npy"

# 73k NSD images at 224x224 fp16 (CLIP/DINOv2 native resolution, ~21 GB) + the trial index,
# both from the gated pscotti/mindeyev2 dataset, mirrors the MindEye2 reproduction's
# download_mindeyev2.py, restricted to just the files TRISECT needs.
FILES="COCO_73k_subj_indices.hdf5 coco_images_224_float16.hdf5"
if [[ "${1:-}" == "--betas" ]]; then
  for N in 01 02 05 07; do
    FILES="$FILES betas_all_subj${N}_fp32_renorm.hdf5"
  done
fi

python3 - "$DEST/mindeyev2_dataset" $FILES <<'PY'
import sys
from huggingface_hub import hf_hub_download

data_path, *files = sys.argv[1:]
for i, f in enumerate(files, 1):
    print(f"[{i}/{len(files)}] {f}")
    hf_hub_download("pscotti/mindeyev2", filename=f, repo_type="dataset",
                     revision="main", local_dir=data_path)
PY

echo "done"
