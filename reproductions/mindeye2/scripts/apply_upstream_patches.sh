#!/usr/bin/env bash
# Idempotent fixes that the nbconvert mirrors of the upstream notebooks need before anything runs.
#
# `jupyter nbconvert --to script` wipes them; re-run after every conversion. Order of a rebuild: nbconvert ->
# `git apply patches/*.patch` -> this script (the Train.py patch is made against the plain nbconvert output).
#   (0) IPython magics left in all four mirrors -> SyntaxError at parse time
#   (a) `plotting` never defined in non-interactive runs -> NameError
#   (b) refiner checkpoint pinned to the authors' /weka cluster path
#   (c) brain_region_masks.hdf5 opened relative to cwd
#   (d) the three ground-truth .pt opened relative to cwd
#   (f) Accelerator(dispatch_batches=False): without it the two ranks deadlock
# Not patched on purpose: (e), the UMAP tail of final_evaluations.py, which raises FileNotFoundError after both
# CSVs are written; patching it would touch code that produced no number.
#
# Usage (from src/):  bash apply_upstream_patches.sh [--check]    # --check only verifies, exit 1 if one is missing

set -uo pipefail

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

cd "$(dirname "${BASH_SOURCE[0]}")"
SCRIPTS=(Train.py recon_inference.py enhanced_recon_inference.py final_evaluations.py)
FAIL=0

note() { printf '  %-6s %s\n' "$1" "$2"; }

# ---- (0) IPython magics ---------------------------------------------------
# nbconvert copies `%load_ext autoreload` / `%autoreload 2` verbatim. They sit inside an
# `if utils.is_interactive():` block, but Python parses the whole file before running any of
# it; `python Train.py` dies on the magic without ever reaching main(). Comment them out.
echo "(0) IPython magics"
if [[ $CHECK_ONLY -eq 0 ]]; then
    sed -i 's/^\( *\)%\(load_ext autoreload\|autoreload 2\)/\1# %\2/' "${SCRIPTS[@]}"
fi
for f in "${SCRIPTS[@]}"; do
    if grep -qE '^ *%' "$f"; then note "MISS" "$f still has a bare magic"; FAIL=1
    else note "ok" "$f"; fi
done

# ---- (a) plotting ---------------------------------------------------------
echo "(a) plotting defined in non-interactive runs"
if [[ $CHECK_ONLY -eq 0 ]]; then
    sed -i "s/^if utils.is_interactive(): plotting=True$/plotting = utils.is_interactive()/" \
        recon_inference.py enhanced_recon_inference.py
fi
for f in recon_inference.py enhanced_recon_inference.py; do
    if grep -q "^plotting = utils.is_interactive()" "$f"; then note "ok" "$f"
    else note "MISS" "$f"; FAIL=1; fi
done

# ---- (b) refiner checkpoint path -----------------------------------------
echo "(b) ZavyChromaXL path repointed at the local dataset"
if [[ $CHECK_ONLY -eq 0 ]]; then
    sed -i "s#^base_ckpt_path = '/weka/.*\$#base_ckpt_path = os.path.abspath('mindeyev2_dataset/zavychromaxl_v30.safetensors')#" \
        enhanced_recon_inference.py
fi
if grep -q "^base_ckpt_path = os.path.abspath('mindeyev2_dataset/zavychromaxl_v30.safetensors')" enhanced_recon_inference.py
then note "ok" "enhanced_recon_inference.py"
else note "MISS" "enhanced_recon_inference.py"; FAIL=1; fi

# ---- (c) brain_region_masks.hdf5 -----------------------------------------
echo "(c) brain_region_masks.hdf5 reachable from cwd"
if [[ $CHECK_ONLY -eq 0 ]]; then
    ln -sfn "$PWD/mindeyev2_dataset/brain_region_masks.hdf5" brain_region_masks.hdf5
fi
if [[ -e brain_region_masks.hdf5 ]]; then note "ok" "brain_region_masks.hdf5"
else note "MISS" "brain_region_masks.hdf5"; FAIL=1; fi

# ---- (d) ground-truth .pt -------------------------------------------------
# Link the individual files, never the directory: src/evals/ is also where
# recon_inference.py writes its own output.
echo "(d) evals/*.pt reachable from cwd"
mkdir -p evals
for F in all_images.pt all_captions.pt all_git_generated_captions.pt; do
    [[ $CHECK_ONLY -eq 0 ]] && ln -sfn "$PWD/mindeyev2_dataset/evals/$F" "evals/$F"
    if [[ -e "evals/$F" ]]; then note "ok" "evals/$F"
    else note "MISS" "evals/$F"; FAIL=1; fi
done

# ---- (f) dispatch_batches=False -------------------------------------------
# The webdataset train_dl is an IterableDataset, for which accelerate 0.24 defaults to
# dispatch_batches=True: rank 0 alone pulls batch_size*num_processes samples and scatters a
# different slice to each rank. The pre-load loop then does
#     if len(image0) != len(image_idx): continue   # hdf5 cant handle duplicate indexing
# on that per-rank slice; a batch with duplicate COCO indices is skipped on one rank and
# kept on the other. The `iter` counters drift apart, whichever rank reaches
# num_iterations_per_epoch-1 first leaves the loop and starts training, and the other blocks
# forever in broadcast_object_list waiting for a batch that will never be sent. NCCL matches
# collectives by order; rank 0's gradient ALLREDUCE meets rank 1's BROADCAST and both
# hang until the 30-minute watchdog aborts the job.
#
# dispatch_batches=False restores what the dataset was built for: `resampled=True` plus an
# identity nodesplitter means every rank is meant to draw its own independent stream. No
# collective is left in the data path; a diverging skip is harmless. Batch size per rank
# and iterations per epoch are unchanged; this is a deadlock fix, not a recipe change.
echo "(f) Accelerator(dispatch_batches=False)"
if [[ $CHECK_ONLY -eq 0 ]]; then
    sed -i 's/^accelerator = Accelerator(split_batches=False, mixed_precision="fp16")$/accelerator = Accelerator(split_batches=False, mixed_precision="fp16", dispatch_batches=False)/' \
        Train.py
fi
if grep -q '^accelerator = Accelerator(split_batches=False, mixed_precision="fp16", dispatch_batches=False)' Train.py
then note "ok" "Train.py"
else note "MISS" "Train.py -- ranks will deadlock at the first backward"; FAIL=1; fi

# ---- the added instrumentation still in place? ------------------------------
# nbconvert would silently drop these too, and a run without them is a different recipe.
echo "(*) local instrumentation markers"
for M in frozen-cache pixcorr-stride ckpt-slim grad-ckpt grad-accum mem-profile; do
    N=$(grep -c "$M" Train.py)
    if [[ $N -gt 0 ]]; then note "ok" "$M ($N sites)"
    else note "MISS" "$M absent from Train.py -- regenerated by nbconvert?"; FAIL=1; fi
done
# grad-ckpt is the only one that also lives in models.py (the wrapped blocks); without those
# the flag would parse and then do nothing at all, which is the worst of the failure modes.
N=$(grep -c "grad-ckpt" models.py)
if [[ $N -gt 0 ]]; then note "ok" "grad-ckpt in models.py ($N sites)"
else note "MISS" "grad-ckpt absent from models.py -- --grad_checkpointing would be a no-op"; FAIL=1; fi

echo
if [[ $FAIL -eq 0 ]]; then echo "All patches in place."; else echo "SOME PATCHES MISSING (see MISS above)"; fi
exit $FAIL
