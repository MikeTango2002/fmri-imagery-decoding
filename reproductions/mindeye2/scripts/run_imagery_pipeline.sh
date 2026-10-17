#!/usr/bin/env bash
# End-to-end NSD-Imagery pipeline for MindEye2: steps 2-6 chained (step 1 is download_nsd_imagery.sh).
#
# Each step starts only once the previous one has exited 0 and its output is on disk; it can run unattended.
# It assumes step 1 ran and the finetuned checkpoints of every subject in --subj are on disk. Step 4 splits its
# work as (subject, mode) pairs across the two GPUs; even --subj "1" uses both cards. Step 3, a manual look at
# 12 reconstructions, is an automated gate: the mean PixCorr over the 12 vision stimuli must clear 0.02 (chance is
# ~0, the published target 0.161) or the script stops before the full matrix. It only catches a broken pipeline
# (wrong mask, misaligned stimuli, a checkpoint that failed to load); still look at ../evals_imagery_smoke.png.
# The smoke test runs on the first subject of --subj, vision mode, gen_rep=1.
#
# Usage (from src/, venv active):
#   nohup bash run_imagery_pipeline.sh --subj "1" > ../train_logs/logs/imagery_pipeline.log 2>&1 &
#   bash run_imagery_pipeline.sh                  # subjects 1 2 5 7 (all four trained)

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(cd .. && pwd)"
LOGS="$ROOT/train_logs/logs"
mkdir -p "$LOGS" ../../results/baselines ../../results/brain_correlation

SUBJECTS="1 2 5 7"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --subj) SUBJECTS="$2"; shift 2 ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
done
read -r SMOKE_SUBJ _ <<< "$SUBJECTS"

die() { echo; echo "FATAL: $*" >&2; exit 1; }

echo "=== imagery pipeline start $(date '+%F %T')  (subjects: $SUBJECTS) ==="

# ---- preflight --------------------------------------------------------------
[[ -n "${VIRTUAL_ENV:-}" ]] || die "no venv active. source ../.venv/bin/activate first."
for S in $SUBJECTS; do
    [[ -f "$ROOT/train_logs/finetuned_subj0${S}_40sess_4096hid/last.pth" ]] \
        || die "no finetuned checkpoint for subject 0$S -- run run_training.sh for it first"
done
[[ -d imagery_dataset/nsddata_stimuli ]] || die "imagery_dataset missing -- run download_nsd_imagery.sh first"

# ---- step 2: prepare betas ---------------------------------------------------
echo; echo "--- step 2: prepare_nsd_imagery.py  $(date '+%F %T') ---"
python prepare_nsd_imagery.py --subj $SUBJECTS \
    --data_root imagery_dataset --out_dir imagery_dataset/preprocessed_data \
    --reference_dir imagery_dataset/reference --betas_hdf5_dir mindeyev2_dataset \
    2>&1 | tee "$LOGS/prepare_imagery.log"
grep -q "all checks passed" "$LOGS/prepare_imagery.log" \
    || die "prepare_nsd_imagery.py did not print 'all checks passed' -- see $LOGS/prepare_imagery.log"

# ---- step 3: smoke test + automated sanity gate ------------------------------
echo; echo "--- step 3: smoke test (subj0${SMOKE_SUBJ}, vision, gen_rep=1)  $(date '+%F %T') ---"
export SMOKE_MODEL="finetuned_subj0${SMOKE_SUBJ}_40sess_4096hid"

CUDA_VISIBLE_DEVICES=0 python recon_inference_imagery.py \
    --model_name=$SMOKE_MODEL --subj=$SMOKE_SUBJ --mode=vision --gen_rep=1 \
    --data_path=mindeyev2_dataset --cache_dir=mindeyev2_dataset \
    --imagery_dir=imagery_dataset --hidden_dim=4096 --n_blocks=4 --blurry_recon \
    2>&1 | tee "$LOGS/smoke_recon_subj0${SMOKE_SUBJ}_vision.log" \
    || die "recon_inference_imagery.py smoke test failed -- see $LOGS/smoke_recon_subj0${SMOKE_SUBJ}_vision.log"

CUDA_VISIBLE_DEVICES=0 python enhanced_recon_inference_imagery.py \
    --model_name=$SMOKE_MODEL --subj=$SMOKE_SUBJ --mode=vision --gen_rep=1 \
    2>&1 | tee "$LOGS/smoke_enhanced_subj0${SMOKE_SUBJ}_vision.log" \
    || die "enhanced_recon_inference_imagery.py smoke test failed -- see $LOGS/smoke_enhanced_subj0${SMOKE_SUBJ}_vision.log"

echo "  automated PixCorr sanity gate (vision target ~0.161, chance ~0)..."
python - <<'PY' | tee "$LOGS/smoke_gate.log"
import os, sys, torch
M = os.environ["SMOKE_MODEL"]
r = torch.load(f"evals/{M}/{M}_all_enhancedrecons_vision.pt")[:12, 0].float()   # (12,3,256,256)
g = torch.load("imagery_dataset/preprocessed_data/gt_12.pt").float()            # (12,3,425,425)
g = torch.nn.functional.interpolate(g, size=r.shape[-2:], mode="bilinear", align_corners=False)
r_flat, g_flat = r.flatten(1), g.flatten(1)
r_c = r_flat - r_flat.mean(1, keepdim=True)
g_c = g_flat - g_flat.mean(1, keepdim=True)
corr = (r_c * g_c).sum(1) / (r_c.norm(dim=1) * g_c.norm(dim=1) + 1e-8)
mean_corr = float(corr.mean())
print(f"mean PixCorr over 12 vision stimuli: {mean_corr:.4f}")
THRESHOLD = 0.02   # well below the 0.161 target, well above what a broken pipeline gives
if mean_corr < THRESHOLD:
    print(f"GATE FAILED: {mean_corr:.4f} < {THRESHOLD}")
    sys.exit(1)
print("GATE PASSED")
PY
[[ ${PIPESTATUS[0]} -eq 0 ]] \
    || die "smoke test sanity gate failed -- pipeline looks broken, aborting before the full matrix. See $LOGS/smoke_gate.log"

python - <<'PY'
import os, torch
from torchvision.utils import save_image
M = os.environ["SMOKE_MODEL"]
r = torch.load(f"evals/{M}/{M}_all_enhancedrecons_vision.pt")
g = torch.load("imagery_dataset/preprocessed_data/gt_18.pt")
save_image(torch.cat([torch.nn.functional.interpolate(g[:12], 256), r[:12, 0]]),
           "../evals_imagery_smoke.png", nrow=12)
print("wrote ../evals_imagery_smoke.png")
PY

# ---- step 4: full matrix, requested subjects x 2 modes x 10 samples ----------
echo; echo "--- step 4: full matrix  $(date '+%F %T') ---"
run_one () {   # $1 = subj  $2 = mode  $3 = gpu
    S=$1; MODE=$2; GPU=$3; M="finetuned_subj0${S}_40sess_4096hid"
    CUDA_VISIBLE_DEVICES=$GPU python recon_inference_imagery.py \
        --model_name=$M --subj=$S --mode=$MODE --gen_rep=10 \
        --data_path=mindeyev2_dataset --cache_dir=mindeyev2_dataset \
        --imagery_dir=imagery_dataset --hidden_dim=4096 --n_blocks=4 --blurry_recon \
        || { echo "FAILED recon subj0${S} ${MODE}"; return; }
    CUDA_VISIBLE_DEVICES=$GPU python enhanced_recon_inference_imagery.py \
        --model_name=$M --subj=$S --mode=$MODE --gen_rep=10 \
        || echo "FAILED enhanced subj0${S} ${MODE}"
}
run_lane () {   # $1 = gpu, then a list of "subj:mode" pairs
    GPU=$1; shift
    for PAIR in "$@"; do
        S="${PAIR%%:*}"; MODE="${PAIR##*:}"
        run_one "$S" "$MODE" "$GPU"
    done
}

# Build the work list as subj:mode pairs and round-robin it across the two GPUs; even a
# single subject (2 pairs: vision, imagery) uses both cards instead of leaving one idle.
WORK=()
for S in $SUBJECTS; do
    WORK+=("$S:vision" "$S:imagery")
done
G0=(); G1=(); i=0
for W in "${WORK[@]}"; do
    (( i % 2 == 0 )) && G0+=("$W") || G1+=("$W")
    i=$(( i + 1 ))
done

PIDS=()
if [[ ${#G0[@]} -gt 0 ]]; then
    echo "  gpu0: ${G0[*]}"
    nohup bash -c "$(declare -f run_one); $(declare -f run_lane); run_lane 0 ${G0[*]}" \
        > "$LOGS/imagery_recons_gpu0.log" 2>&1 &
    PIDS+=($!)
fi
if [[ ${#G1[@]} -gt 0 ]]; then
    echo "  gpu1: ${G1[*]}"
    nohup bash -c "$(declare -f run_one); $(declare -f run_lane); run_lane 1 ${G1[*]}" \
        > "$LOGS/imagery_recons_gpu1.log" 2>&1 &
    PIDS+=($!)
fi
echo "  PIDs: ${PIDS[*]} -- waiting..."
for P in "${PIDS[@]}"; do wait "$P"; done

N_FAILED=$(grep -c FAILED "$LOGS/imagery_recons_gpu0.log" "$LOGS/imagery_recons_gpu1.log" 2>/dev/null \
    | awk -F: '{s+=$2} END{print s+0}')
[[ $N_FAILED -eq 0 ]] || die "$N_FAILED subject/mode run(s) FAILED in step 4 -- see $LOGS/imagery_recons_gpu*.log"

N_EXPECTED=$(( $(echo $SUBJECTS | wc -w) * 2 ))
N_TENSORS=$(ls evals/*/*_all_enhancedrecons_{vision,imagery}.pt 2>/dev/null | wc -l)
[[ $N_TENSORS -ge $N_EXPECTED ]] \
    || die "expected >= $N_EXPECTED enhancedrecons tensors after step 4, found $N_TENSORS"

# ---- step 5: metrics ----------------------------------------------------------
echo; echo "--- step 5: metrics  $(date '+%F %T') ---"
export OPENBLAS_CORETYPE=SkylakeX      # some CPUs miscompute large matmuls without it

python final_evaluations_imagery.py \
    --model mindeye_II \
    --recon_root evals \
    --recons_pt 'finetuned_subj{subj02}_40sess_4096hid/finetuned_subj{subj02}_40sess_4096hid_all_enhancedrecons_{mode}.pt' \
    --gt ../../mindeye_I/dataset/nsddata_stimuli/stimuli/imagery_stimuli_18.pt \
    --subj $SUBJECTS --mode vision imagery --gen_rep 10 \
    --out ../../results/baselines/mindeye_II.tsv \
    2>&1 | tee "$LOGS/imagery_metrics.log"
[[ -s ../../results/baselines/mindeye_II.tsv ]] \
    || die "final_evaluations_imagery.py produced no ../../results/baselines/mindeye_II.tsv -- see $LOGS/imagery_metrics.log"

python gnet_brain_correlation.py compute \
    --model mindeye_II \
    --recon_root evals \
    --betas_root ../../mindeye_I/dataset/preprocessed \
    --stim_meta imagery_dataset/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3 \
    --gnet mindeyev2_dataset/gnet_multisubject.pt \
    --masks mindeyev2_dataset/brain_region_masks.hdf5 \
    --subj $SUBJECTS --n_samples 10 \
    --out ../../results/brain_correlation/mindeye_II_brain_corr.tsv \
    2>&1 | tee "$LOGS/imagery_gnet.log"
[[ -s ../../results/brain_correlation/mindeye_II_brain_corr.tsv ]] \
    || die "gnet_brain_correlation.py produced no ../../results/brain_correlation/mindeye_II_brain_corr.tsv -- see $LOGS/imagery_gnet.log"

# ---- step 6: aggregate ---------------------------------------------------------
echo; echo "--- step 6: aggregate  $(date '+%F %T') ---"
python aggregate_imagery_metrics.py \
    --image_tsv ../../results/baselines/mindeye_II.tsv \
    --gnet_tsv  ../../results/brain_correlation/mindeye_II_brain_corr.tsv \
    --expected_reps 10 --out_dir ../../results/baselines/agg \
    2>&1 | tee "$LOGS/imagery_summary.log"

echo
echo "=== imagery pipeline done $(date '+%F %T')  (subjects: $SUBJECTS) ==="
echo "Summary: $LOGS/imagery_summary.log"
if [[ "$SUBJECTS" != "1 2 5 7" ]]; then
    echo "NOTE: this run only covers subjects [$SUBJECTS] -- Kneeland's tables average over"
    echo "      1,2,5,7. Re-run with --subj \"<remaining>\" once the rest are trained, then"
    echo "      re-run step 5+6 (or this whole script) over all four before averaging."
fi
echo "Next: compare against the published values (Section 4.5.2 of the thesis; chance = 40.91%, not 50%),"
echo "      reading the tables in ../../results/baselines/agg; the per-image metrics are"
echo "      in ../../results/baselines/mindeye_II.tsv."
