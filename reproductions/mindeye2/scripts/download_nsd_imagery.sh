#!/usr/bin/env bash
# Populate imagery_dataset/ (next to this script) with the NSD-Imagery files MindEye2 needs. Subjects 1, 2, 5, 7.
#
# Reuses the dataset folder of the MindEye1 working copy (../../mindeye_I/dataset) instead of
# re-downloading: the raw NSD betas / ROI masks and the umn-imagery HuggingFace metadata are the
# same bytes for every model of the thesis (nothing here is MindEye1-specific), and that copy
# was already cross-checked against the authors' own preprocessed tensors by the MindEye1
# prepare_nsd_imagery.py. Reusing it means this script needs no AWS credentials when that folder
# already has the four subjects. Falls back to a real download for anything not found there (a
# subject it has not fetched, or a checkout without the MindEye1 copy next to this one).
#
# Run from inside src/:
#   bash download_nsd_imagery.sh

set -u

cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_ROOT="imagery_dataset"
MINDEYE_I_DATASET="$(cd ../../mindeye_I/dataset 2>/dev/null && pwd)"
HF_BASE="https://huggingface.co/datasets/reesekneeland/umn-imagery/resolve/main"
SUBJECTS="1 2 5 7"

if [[ -n "$MINDEYE_I_DATASET" ]]; then
    echo "Reusing raw NSD-Imagery data already on disk at $MINDEYE_I_DATASET"
else
    echo "No mindeye_I/dataset found next to this repo -- everything will be downloaded fresh."
    command -v aws >/dev/null || { echo "ERROR: aws cli not found. Install with: python -m pip install awscli"; exit 1; }
fi
command -v curl >/dev/null || { echo "ERROR: curl not found."; exit 1; }

# $1 = path relative to both DATA_ROOT and MINDEYE_I_DATASET; the rest is the fetch command
# to run (only) if mindeye_I does not already have that file.
link_or_fetch() {
    local REL="$1"; shift
    local DEST="$DATA_ROOT/$REL"
    mkdir -p "$(dirname "$DEST")"
    if [[ -n "$MINDEYE_I_DATASET" && -f "$MINDEYE_I_DATASET/$REL" ]]; then
        ln -sf "$MINDEYE_I_DATASET/$REL" "$DEST"
        echo "  reused mindeye_I: $REL"
    else
        echo "  downloading: $REL"
        "$@" || { echo "FAILED: $REL"; exit 1; }
    fi
}

# 1. Trial metadata + target stimuli (not part of the official NSD S3 bucket, derived
# files published by the benchmark authors on HuggingFace).
link_or_fetch "nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3" \
    curl -L --fail --retry 3 -o "$DATA_ROOT/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3" \
        "$HF_BASE/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3"

link_or_fetch "nsddata_stimuli/stimuli/imagery_stimuli_18.pt" \
    curl -L --fail --retry 3 -o "$DATA_ROOT/nsddata_stimuli/stimuli/imagery_stimuli_18.pt" \
        "$HF_BASE/nsddata_stimuli/stimuli/imagery_stimuli_18.pt"

# 2. Per-subject raw data (NSD S3 bucket) and the authors' preprocessed reference tensors
# (HuggingFace) used by prepare_nsd_imagery.py's cross-check.
for SUBJ in $SUBJECTS; do
    echo "-> subj0${SUBJ}"

    link_or_fetch "nsddata/ppdata/subj0${SUBJ}/func1pt8mm/roi/nsdgeneral.nii.gz" \
        aws s3 cp --no-sign-request \
            "s3://natural-scenes-dataset/nsddata/ppdata/subj0${SUBJ}/func1pt8mm/roi/nsdgeneral.nii.gz" \
            "$DATA_ROOT/nsddata/ppdata/subj0${SUBJ}/func1pt8mm/roi/nsdgeneral.nii.gz"

    link_or_fetch "nsddata_betas/ppdata/subj0${SUBJ}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz" \
        aws s3 cp --no-sign-request \
            "s3://natural-scenes-dataset/nsddata_betas/ppdata/subj0${SUBJ}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz" \
            "$DATA_ROOT/nsddata_betas/ppdata/subj0${SUBJ}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz"

    for F in nsd_imagery.pt nsd_imagery_unnormalized.pt; do
        link_or_fetch "reference/preprocessed_data/subject${SUBJ}/$F" \
            curl -L --fail --retry 3 \
                -o "$DATA_ROOT/reference/preprocessed_data/subject${SUBJ}/$F" \
                "$HF_BASE/preprocessed_data/subject${SUBJ}/$F"
    done
done

echo
echo "Done. Data staged in $DATA_ROOT (symlinked from mindeye_I/dataset where available)."
