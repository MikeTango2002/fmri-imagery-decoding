#!/usr/bin/env bash
# MindEye1 on NSD-Imagery, all stages in one script: reconstructions, per-image metrics, aggregation, GNet.
#
# The betas enter the reconstructions, not the metrics; a change in the betas restarts the chain from the
# reconstructions (scoring the old .pt files again gives the same numbers). Stages 2-5 start only if stage 1
# produced all 200 tensors; a crash does not yield averages over a subset. The betas are z-scored with
# ddof=1, as in the tensors of the benchmark authors. Stage 2 is an optional comparison with a run made on
# betas z-scored with ddof=0 (suffix ddof0), when that run is on disk.
#
# Usage (PROJECT_ROOT exported; from the src/ folder of MindEye1, detached):
#   nohup bash ../run_benchmark_B.sh > ../train_logs/logs/rerun_B.log 2>&1 &
# GPUS="0 1" (default) runs one process per card, subjects split 1+2 / 5+7; GPUS=0 runs all four on one card in
# twice the time. Neither moves a number. After a crash in stage 1, RESUME=1 keeps the tensors already in
# imagery/; without it the preflight refuses to start, on purpose.

set -uo pipefail

RROOT="${RROOT:-$PROJECT_ROOT}"
VD_DIR="${VD_DIR:-$RROOT/mindeye_I/train_logs/models/versatile-diffusion}"
LOGS="$RROOT/mindeye_I/train_logs/logs"
SRC="$RROOT/mindeye_I/src"
STAMP="$(date +%Y%m%d_%H%M%S)"
GPUS="${GPUS:-0 1}"
GPU_FIRST="${GPUS%% *}"          # the card the cheap CPU-bound-ish stages 3 and 5 use

export PATH="$RROOT/.uv/bin:$PATH"
export OPENBLAS_CORETYPE=SkylakeX      # some CPUs miscompute large matmuls without it

banner() { echo; echo "================================================================"; \
           echo "== $1   [$(date '+%F %T')]"; echo "================================================================"; }
die()    { echo; echo "ABORT: $1" >&2; exit 1; }

cd "$SRC" || die "cannot cd into $SRC"
mkdir -p "$LOGS"

# ----------------------------------------------------------------- preflight --
banner "0. preflight"

# Everything the five stages need from outside this directory, checked here rather than at the
# stage that needs it: a missing SwAV would otherwise surface at stage 3, after the night of
# GPU that stage 1 costs. Fatal for what cannot be re-downloaded or costs 15 GB; a warning for
# what torch.hub and clip fetch by themselves on first use.
[[ -d "$VD_DIR" && -f "$VD_DIR/model_index.json" ]] \
  || die "VD_DIR is not a Versatile Diffusion cache: $VD_DIR
  It is NOT downloaded automatically, on purpose: an empty --vd_cache_dir used to start a
  15 GB download inside src/ that then died on the scheduler. Re-fetch it with
    python -c \"from huggingface_hub import snapshot_download; snapshot_download('shi-labs/versatile-diffusion', local_dir='\$VD_DIR')\"
  (the layout must be flat, model_index.json at the top)."

# The checkpoints are the one thing here that cannot be downloaded again: they are the
# NSD-core training.
# The exact filenames matter, not just the directories. Reconstructions_imagery_batch.py:192
# opens 'epoch120.pth' by that literal name, and the prior is read from 'last.pth'
# (Reconstructions_imagery.py:301) in a different directory. Anything else named epoch120
# is a different file. The batch driver does refuse to start without the autoencoder, but
# finding out here costs seconds instead of a model load.
for S in 01 02 05 07; do
  P="../train_logs/prior_257_final_subj${S}_bimixco_softclip_byol/last.pth"
  A="../train_logs/models/autoencoder_subj${S}_4x_locont_no_reconst/epoch120.pth"
  [[ -s "$P" ]] || die "missing $P
  This one cannot be downloaded: it is the NSD-core training. Stop here."
  [[ -s "$A" ]] || die "missing $A
  The name is literal: not epoch_120.pt, not last.pth. Without it upstream would silently
  fall back to img2img_strength=1, i.e. generate WITHOUT the low-level pipeline -- a
  different method from the one Kneeland et al. benchmarked, with nothing in the output
  saying so. The batch driver refuses instead; so does this preflight."
done
echo "  checkpoints: prior/last.pth + autoencoder/epoch120.pth for all 4 subjects"

[[ -d "$HOME/.cache/clip" ]] || echo "  note: ~/.cache/clip is empty, CLIP ViT-L/14 (~1.7 GB) will be downloaded on first use"
ls "$HOME"/.cache/torch/hub/checkpoints/*swav* >/dev/null 2>&1 \
  || echo "  note: SwAV is not in torch.hub, stage 3 will download it"

# Stage 5 only. A missing GNet must not cost the whole run: warn now, skip the stage later.
GNET="$RROOT/mindeye_II/src/mindeyev2_dataset/gnet_multisubject.pt"
MASKS="$RROOT/mindeye_II/src/mindeyev2_dataset/brain_region_masks.hdf5"
HAVE_GNET=1
[[ -s "$GNET" && -s "$MASKS" ]] || { HAVE_GNET=0
  echo "  WARNING: GNet or brain_region_masks.hdf5 missing -> stage 5 will be skipped"
  echo "           (~1.3 GB; stages 1-4 do not need it)"; }

for S in 1 2 5 7; do
  F="../dataset/preprocessed/subj0$S/nsd_imagery.pt"
  [[ -s "$F" ]] || die "missing $F -- run prepare_nsd_imagery.py first"
done
echo "  betas: 4 subjects present under ../dataset/preprocessed"
echo "  (regenerated $(date -r ../dataset/preprocessed/subj01/nsd_imagery.pt '+%F %T'))"

# The betas must be the ddof=1 ones. If preprocessed/ is older than preprocessed_ddof0/, the
# directories were not swapped and this would silently run on the ddof=0 tensors.
if [[ -d ../dataset/preprocessed_ddof0 ]]; then
  if [[ ../dataset/preprocessed/subj01/nsd_imagery.pt -ot ../dataset/preprocessed_ddof0/subj01/nsd_imagery.pt ]]; then
    die "dataset/preprocessed/ is OLDER than dataset/preprocessed_ddof0/: the regenerated
  tensors are not the ones in place. Check which directory prepare_nsd_imagery.py wrote to."
  fi
  echo "  old tensors kept aside in ../dataset/preprocessed_ddof0 (good)"
fi

# --skip_existing would skip everything if the .pt of a previous run were still in imagery/, and the run
# would end in seconds having done nothing while looking successful.
#
# RESUME=1 is the one legitimate exception: a previous invocation of THIS script died partway
# through stage 1 (OOM, someone else's job filling the card, a reboot) and imagery/ holds the
# partial output of the NEW betas. Then skipping what is there is exactly right. It is opt-in
# because the script cannot tell a half-finished new run from a complete old one by looking at
# the files: only you know which one is on disk. Before setting it, check the integrity of
# what survived (a .pt truncated by a crash is 'existing' as far as --skip_existing knows):
#   python -c "import glob,torch,sys; [torch.load(p,map_location='cpu') for p in sorted(glob.glob('imagery/recons_subj0*_seed*.pt'))]"
NOLD=$(ls imagery/recons_subj0*_seed*.pt 2>/dev/null | wc -l)
if [[ $NOLD -gt 0 && "${RESUME:-0}" != "1" ]]; then
  die "imagery/ already holds $NOLD reconstruction tensors. --skip_existing would skip them
  all and stages 2-5 would score the OLD run. Move them aside first:
      mv imagery imagery_ddof0_$STAMP && mkdir imagery
  Unless these ARE the new run, interrupted partway: then re-launch with RESUME=1."
fi
[[ $NOLD -gt 0 ]] && echo "  RESUME=1: $NOLD/200 tensors already present, they will be skipped"
mkdir -p imagery

echo "  free disk:"; df -h "$RROOT" | tail -1 | sed 's/^/    /'
echo "  GPUs requested: $GPUS"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader | sed 's/^/    gpu /'

# A reconstruction process with --recons_per_sample=16 peaks at ~30 GB.
# 34000 leaves ~4 GB of headroom on a 47 GB card. Sharing the card with someone else's job is
# fine for correctness, independent processes, different files, but it both halves the
# speed and is what actually kills the run; it is a loud warning, not a footnote.
for G in $GPUS; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G")
  [[ ${FREE:-0} -ge 34000 ]] || die "gpu $G has only ${FREE} MiB free, ~34000 are needed
  (a process with --recons_per_sample=16 peaks at ~30 GB). Wait until the other user frees the
  card, or use only the free card: GPUS=<n> RESUME=1 bash ../run_benchmark_B.sh"
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$G" | wc -l)
  [[ $BUSY -eq 0 ]] || echo "    WARNING: gpu $G already has $BUSY foreign process(es). The run time
    doubles and, if those processes grow, this run dies of OOM."
done

# Fragmentation was named explicitly by the OOM message. Costs nothing when memory is ample.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ../.venv/bin/activate || die "cannot activate mindeye_I venv"
python -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())" \
  || die "the mindeye_I venv is broken"

# ------------------------------------------------------- 1. reconstructions --
banner "1. reconstructions -- 200 .pt, 5 conditions x 10 seeds x 4 subjects"

# One process per requested GPU, never DDP: the training scripts multiply epochs by the process
# count, which would change the recipe. With both cards the subjects split 1+2 / 5+7; with one card that card
# takes all four, sequentially, for twice the wall time and the same output.
declare -A SPLIT
if [[ "$GPUS" == "0 1" ]]; then
  SPLIT[0]="1 2"; SPLIT[1]="5 7"
else
  SPLIT[$GPU_FIRST]="1 2 5 7"
fi

PIDS=()
for G in $GPUS; do
  env CUDA_VISIBLE_DEVICES="$G" python Reconstructions_imagery_batch.py \
    --subj ${SPLIT[$G]} --seeds 0 1 2 3 4 5 6 7 8 9 \
    --vd_cache_dir="$VD_DIR" --imagery_dir=../dataset/preprocessed --out_dir=imagery \
    --img2img_strength=.85 --recons_per_sample=16 --skip_existing \
    > "$LOGS/imagery_recons_gpu${G}_$STAMP.log" 2>&1 &
  PIDS+=($!)
  echo "  gpu$G  subj ${SPLIT[$G]}  PID ${PIDS[-1]}  ->  $LOGS/imagery_recons_gpu${G}_$STAMP.log"
done

for P in "${PIDS[@]}"; do wait "$P"; echo "  PID $P exit $?"; done

NPT=$(ls imagery/recons_subj0*_seed*.pt 2>/dev/null | wc -l)
NFAIL=$(cat "$LOGS"/imagery_recons_gpu*_$STAMP.log 2>/dev/null | grep -c FAILED 2>/dev/null | paste -sd+ | bc)
echo "  tensors written: $NPT / 200   FAILED lines: ${NFAIL:-0}"
[[ $NPT -eq 200 ]] || die "expected 200 tensors, found $NPT. The averages downstream would be
  computed over a subset without saying so. Read the two logs and relaunch: --skip_existing
  makes the same command restartable."

# --------------------------------------- 2. ddof=0 run against ddof=1 run --
banner "2. old run (ddof=0) vs new run -- how far the betas moved the images"

# Same seeds, same code, betas differing by a single 1.01058 = 1/sqrt(47/48) factor (the ratio of
# the ddof=0 and ddof=1 standard deviations of a 48-trial run). Optional; skipped if
# imagery_ddof0/ is not on disk.
if [[ -d imagery_ddof0 ]] && [[ $(ls imagery_ddof0/recons_subj0*_seed*.pt 2>/dev/null | wc -l) -gt 0 ]]; then
  python - <<'PY' 2>&1 | tee "$LOGS/imagery_ddof0_vs_new_$STAMP.log"
import glob, os, torch

rows = []
for new in sorted(glob.glob("imagery/recons_subj0*_seed*.pt")):
    old = os.path.join("imagery_ddof0", os.path.basename(new))
    if not os.path.exists(old):
        continue
    a = torch.load(new, map_location="cpu").float()
    b = torch.load(old, map_location="cpu").float()
    if a.shape != b.shape:
        print(f"  shape mismatch on {os.path.basename(new)}: {tuple(a.shape)} vs {tuple(b.shape)}")
        continue
    # Pixel correlation per stimulus, then averaged: the same quantity PixCorr reports;
    # the number is directly readable against the metric it would perturb.
    n = a.shape[0]
    cors = []
    for i in range(n):
        x, y = a[i].flatten(), b[i].flatten()
        x = x - x.mean(); y = y - y.mean()
        cors.append(float((x @ y) / (x.norm() * y.norm() + 1e-12)))
    rows.append((os.path.basename(new), float((a - b).abs().max()),
                 float((a - b).abs().mean()), sum(cors) / len(cors)))

if not rows:
    print("  no comparable pair")
else:
    mean_r = sum(r[3] for r in rows) / len(rows)
    worst = min(rows, key=lambda r: r[3])
    print(f"  pairs compared : {len(rows)}")
    print(f"  mean PixCorr old-vs-new : {mean_r:.4f}")
    print(f"  worst : {worst[0]}  r={worst[3]:.4f}  max|diff|={worst[1]:.1f}/255")
    print(f"  mean max|diff| : {sum(r[2] for r in rows) / len(rows):.3f}/255")
    print()
    print("  Reading: the sampling of Versatile Diffusion is stochastic but reseeded per")
    print("  (condition, seed), so with the same betas the two runs would be identical. All the")
    print("  difference here comes from the 1.01058 factor on the betas, carried through the")
    print("  prior. An r close to 1 means the images are the same; a low r means the deviation")
    print("  was NOT negligible and has to be reported as such.")
PY
else
  echo "  imagery_ddof0/ not found: comparison skipped (optional)."
  echo "  To do it, put the old tensors in this folder:"
  echo "      $RROOT/mindeye_I/src/imagery_ddof0/"
fi

# ------------------------------------------------------------ 3. metrics --
banner "3. per-image metrics -- 80 TSV"

# TSVs of a previous invocation would be overwritten in place; they are moved aside instead.
if [[ -d imagery/metrics ]] && [[ $(ls imagery/metrics/*.tsv 2>/dev/null | wc -l) -gt 0 ]]; then
  mv imagery/metrics "imagery/metrics_ddof0_$STAMP"
  echo "  old TSVs moved to imagery/metrics_ddof0_$STAMP"
fi

env CUDA_VISIBLE_DEVICES="$GPU_FIRST" python Reconstruction_Metrics_imagery_batch.py \
  --subj 1 2 5 7 --modality vision imagery --seeds 0 1 2 3 4 5 6 7 8 9 \
  --recon_dir=imagery --imagery_dir=../dataset/preprocessed --out_dir=imagery/metrics \
  --img2img_strength=0.85 --recons_per_sample=16 --skip_existing \
  > "$LOGS/imagery_metrics_$STAMP.log" 2>&1
echo "  exit code: $?"

NTSV=$(ls imagery/metrics/per_image_*.tsv 2>/dev/null | wc -l)
echo "  TSVs written: $NTSV / 80   (no imageryconcepts: there is no visual ground truth)"
grep FAILED "$LOGS/imagery_metrics_$STAMP.log" | head
[[ $NTSV -eq 80 ]] || die "expected 80 TSVs, found $NTSV -- see $LOGS/imagery_metrics_$STAMP.log"

# ---------------------------------------------------------- 4. aggregation --
banner "4. aggregation -- the tables to transcribe"

python aggregate_imagery_metrics.py --metrics_dir=imagery/metrics \
  2>&1 | tee "$LOGS/imagery_summary_$STAMP.log"

# ------------------------------------------------------ 5. brain correlation --
banner "5. brain correlation GNet -- Early / Higher / VisCtx"

# Different repo, different venv: the MindEye2 repository owns gnet_multisubject.pt and models.py.
# The betas are the model-independent canonical source: the ddof=1 ones prepared for this model.
if [[ $HAVE_GNET -eq 0 ]]; then
  echo "  SKIPPED: GNet missing (reported by the preflight). Stages 1-4 are complete and"
  echo "  their numbers are valid; this stage can be run again on its own once GNet has"
  echo "  been downloaded."
  banner "END (without stage 5)"
  exit 0
fi

deactivate 2>/dev/null || true
cd "$RROOT/mindeye_II/src" || die "cannot cd into mindeye_II/src"
source ../.venv/bin/activate || die "cannot activate mindeye_II venv"

OUT="$RROOT/results/brain_correlation"
mkdir -p "$OUT"
[[ -f "$OUT/mindeye_I_brain_corr.tsv" ]] && \
  mv "$OUT/mindeye_I_brain_corr.tsv" "$OUT/mindeye_I_brain_corr_ddof0_$STAMP.tsv"

BETAS=../../mindeye_I/dataset/preprocessed
META=../../mindeye_I/dataset/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3
STIM=../../mindeye_I/dataset/nsddata_stimuli/stimuli/imagery_stimuli_18.pt
CONDS="imagerysimple imagerycomplex visionsimple visioncomplex imageryconcepts"

CUDA_VISIBLE_DEVICES="$GPU_FIRST" python gnet_brain_correlation.py compute \
    --model mindeye_I --recon_root ../../mindeye_I/src/imagery \
    --betas_root $BETAS --stim_meta $META --gt_ceiling $STIM \
    --conditions $CONDS \
    --out "$OUT/mindeye_I_brain_corr.tsv" 2>&1 | tee "$OUT/mindeye_I_$STAMP.log"

banner "6. summary table (new MindEye1 + unchanged Brain-Diffuser)"
python gnet_brain_correlation.py report "$OUT"/*_brain_corr.tsv 2>&1 | tee "$OUT/report_$STAMP.log"

banner "END"
cat <<EOF
  To read, in this order:
    $LOGS/imagery_ddof0_vs_new_$STAMP.log   how far the betas moved the images
    $LOGS/imagery_summary_$STAMP.log        the 8 image metrics
    $OUT/report_$STAMP.log                  Early/Higher/VisCtx

  Failure criterion: AlexNet(2) and AlexNet(5) on imagery must be ABOVE 55% (published
  59.6% / 61.0%). Values around 50% are the signature of an error in the beta
  preparation, not a finding.
EOF
