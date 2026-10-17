#!/usr/bin/env bash
# Link the NSD / NSD-Imagery files that the other reproductions already downloaded into the
# layout the iCNN benchmark-B scripts expect.
#
# Nothing is copied: betas_nsdimagery.nii.gz is 1.6 GB per subject and nsd_stimuli.hdf5 is
# 39.5 GB and is shared by symlink with the MindEye1 and Takagi folders; it is stored once.
# Symlinks only, idempotent.
#
# Run from anywhere, with PROJECT_ROOT set to the folder that holds mindeye_I/, takagi/ and icnn/:
#   bash src/link_existing_nsd_data.sh
# then check for dead links:
#   find feature-decoding/data -xtype l
set -euo pipefail

ROOT="${ROOT:-$PROJECT_ROOT}"
DEST="${DEST:-$ROOT/icnn/feature-decoding/data}"

MINDEYE="$ROOT/mindeye_I/dataset"
TAKAGI="$ROOT/takagi/nsd"

link() {  # link <target> <linkname>; refuses to create a dead link
  local target="$1" name="$2"
  if [ ! -e "$target" ]; then
    echo "MISSING: $target  (nothing linked to $name)" >&2
    MISSING=$((MISSING + 1))
    return
  fi
  mkdir -p "$(dirname "$name")"
  ln -sfn "$target" "$name"
  echo "  $name -> $target"
}

MISSING=0
echo "dest: $DEST"

for N in 01 02 05 07; do
  link "$MINDEYE/nsddata_betas/ppdata/subj$N/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz" \
       "$DEST/nsd/nsddata_betas/ppdata/subj$N/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz"
  link "$MINDEYE/nsddata/ppdata/subj$N/func1pt8mm/roi/nsdgeneral.nii.gz" \
       "$DEST/nsd/nsddata/ppdata/subj$N/func1pt8mm/roi/nsdgeneral.nii.gz"
done

# NSD experiment design: subjectim + masterordering, i.e. the trial -> 73k image pairing
# that prepare_nsdcore_icnn.py rebuilds. The copy of Takagi sits next to the stimuli it indexes.
link "$TAKAGI/nsddata/experiments/nsd/nsd_expdesign.mat" \
     "$DEST/nsd/nsddata/experiments/nsd/nsd_expdesign.mat"

# 73k NSD images: the target features are recomputed from these on every run; the
# link must survive any cleanup in takagi/.
link "$TAKAGI/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5" \
     "$DEST/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"

# NSD-Imagery trial metadata + the 18 target stimuli, in the HuggingFace layout
# (reesekneeland/umn-imagery), which is what prepare_nsd_imagery_icnn.py looks for.
link "$MINDEYE/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3" \
     "$DEST/nsd_imagery/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3"
link "$MINDEYE/nsddata_stimuli/stimuli/imagery_stimuli_18.pt" \
     "$DEST/nsd_imagery/nsddata_stimuli/stimuli/imagery_stimuli_18.pt"

# Reference tensors for the cross-check in prepare_nsd_imagery_icnn.py. These are MindEye1's own
# regenerated copies, which match the authors' HuggingFace tensors to max|diff| 1.03e-05
# (prepare_nsd_imagery.py of the MindEye1 reproduction, docstring of normalize_imagery_betas).
# NOT mindeye_I/dataset/reference/preprocessed_data/: those files are 142-byte pointers.
for N in 1 2 5 7; do
  link "$MINDEYE/preprocessed/subj0$N/nsd_imagery.pt" \
       "$DEST/nsd_imagery/preprocessed_data/subject$N/nsd_imagery.pt"
  link "$MINDEYE/preprocessed/subj0$N/nsd_imagery_unnormalized.pt" \
       "$DEST/nsd_imagery/preprocessed_data/subject$N/nsd_imagery_unnormalized.pt"
done

echo
if [ "$MISSING" -gt 0 ]; then
  echo "$MISSING target(s) missing: download only those (the commands are in the README of this folder)" >&2
  exit 1
fi
echo "all targets present; checking for dead links under $DEST"
find "$DEST" -xtype l -print | sed 's/^/  DEAD: /'
echo "done"
