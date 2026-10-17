#!/bin/bash
# Download the NSD-Imagery files that MindEye1 needs, for subjects 1, 2, 5, 7 (public since 2025-09-01).
#
# Inputs:  the official NSD S3 bucket (aws cli) and the benchmark authors' HuggingFace dataset (curl)
# Outputs: ../dataset/: imagery betas, nsdgeneral mask, trial metadata, the 18 target stimuli and the authors'
#          preprocessed tensors, which turn the cross-check of prepare_nsd_imagery.py into a real assert
#
# Usage (from src/):  bash download_nsd_imagery.sh

set -u

DATA_ROOT="../dataset"

# The trial metadata (nsdimagery_stimuli.pkl3) and the 18 target stimuli
# (imagery_stimuli_18.pt) are NOT part of the official NSD S3 bucket: they are
# derived files published by the benchmark authors on HuggingFace.
HF_BASE="https://huggingface.co/datasets/reesekneeland/umn-imagery/resolve/main"

# Check if aws cli is installed
if ! command -v aws &> /dev/null
then
    echo "ERROR: aws cli was not found."
    echo "Install aws cli using: python -m pip install awscli"
    exit 1
fi

if ! command -v curl &> /dev/null
then
    echo "ERROR: curl was not found."
    exit 1
fi

echo "Starting download of the NSD-Imagery dataset (subjects 1, 2, 5, 7)..."

# 1. Download trial metadata + target stimuli from the authors' HuggingFace repo
echo "-> Downloading trial metadata and target stimuli (HuggingFace)..."
mkdir -p "$DATA_ROOT/nsddata_stimuli/stimuli"

# exps / cues / image_map: maps each of the 720 trials to run type and target stimulus
curl -L --fail --retry 3 -o "$DATA_ROOT/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3" \
    "$HF_BASE/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3"

# Ground truth stimuli, float32 tensor of shape (18, 3, 425, 425)
curl -L --fail --retry 3 -o "$DATA_ROOT/nsddata_stimuli/stimuli/imagery_stimuli_18.pt" \
    "$HF_BASE/nsddata_stimuli/stimuli/imagery_stimuli_18.pt"

# 1b. The authors' own preprocessed tensors: these are exactly what prepare_nsd_imagery.py
# produces; they turn its cross-check into a real assert instead of a skipped branch.
# Without them the cross-check cannot run, and a wrong mask or a wrong normalization
# convention passes unnoticed. They are required.
echo "-> Downloading the authors' preprocessed tensors (reference for the cross-check)..."
for SUBJ in 1 2 5 7; do
    mkdir -p "$DATA_ROOT/reference/preprocessed_data/subject${SUBJ}"
    for F in nsd_imagery.pt nsd_imagery_unnormalized.pt; do
        curl -L --fail --retry 3 \
            -o "$DATA_ROOT/reference/preprocessed_data/subject${SUBJ}/$F" \
            "$HF_BASE/preprocessed_data/subject${SUBJ}/$F"
    done
done

# 2. Download data (imagery betas and ROI mask) for the subjects
for SUBJ in 1 2 5 7; do
    echo "-> Downloading data for subj0${SUBJ}..."

    # Download imagery betas (version 3, volumetric 1.8 mm)
    aws s3 cp --no-sign-request \
        s3://natural-scenes-dataset/nsddata_betas/ppdata/subj0${SUBJ}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz \
        $DATA_ROOT/nsddata_betas/ppdata/subj0${SUBJ}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz

    # Download ROI mask (nsdgeneral 1.8 mm)
    aws s3 cp --no-sign-request \
        s3://natural-scenes-dataset/nsddata/ppdata/subj0${SUBJ}/func1pt8mm/roi/nsdgeneral.nii.gz \
        $DATA_ROOT/nsddata/ppdata/subj0${SUBJ}/func1pt8mm/roi/nsdgeneral.nii.gz
done

echo "Download complete! All files have been saved in the $DATA_ROOT directory"
