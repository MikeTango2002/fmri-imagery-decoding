#!/bin/bash
# setup_venv.sh: builds the Python environment (venv, no conda) of the iCNN / Shen et al. (2019) path.
# It is the environment for the Kamitani Lab Python-3 port `feature-decoding/`:
#   stage 1 = DNN feature decoding (fMRI -> VGG19 features), ridge regression on CPU.
# Prereq: `uv` must already be on PATH.
# Run from the icnn/ folder:   . setup_venv.sh   (source it so the venv stays active)
#
# NOTE on the GPU. Stage 1 is pure linear algebra (FastL2LiR / sklearn Ridge) and runs on
# CPU+RAM only: the 2 TB of RAM matter here, the GPUs do not. torch is installed anyway
# because stage 2 (iCNN image reconstruction, `bdpy.recon.torch.icnn` + VGG19 + the deep
# generator) is GPU work and shares this same bdpy install. Ada (sm_89) needs a CUDA 11.8+
# build, which is why the cu118 index is explicit below.
#
# NOTE on the ORIGINAL repo. `KamitaniLab/DeepImageReconstruction` (Python 2.7 + the 2016
# Caffe fork `dosovits/caffe-fr-chairs`) cannot be built for sm_89 at all; this port is the
# supported replacement.

set -e

# --- repo root = folder containing this script (icnn/), clone lives in feature-decoding/ ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
FD_DIR="$REPO_ROOT/feature-decoding"

command -v uv >/dev/null 2>&1 || { echo "ERROR: 'uv' not found on PATH. Install it first."; return 1 2>/dev/null || exit 1; }
[ -d "$FD_DIR" ] || { echo "ERROR: feature-decoding/ clone not found at $FD_DIR"; return 1 2>/dev/null || exit 1; }

# --- uv local isolation (per-user sandbox: cache + python builds under the repo) ---
export UV_CACHE_DIR="$REPO_ROOT/.uv_cache"
export UV_PYTHON_INSTALL_DIR="$REPO_ROOT/.uv_python"

# --- Python 3.10: bdpy declares `requires_python <3.12`, and the repo pins numpy<2.0.0.
#     The system Python 3.12 on the server is therefore NOT usable here. ---
uv python install 3.10.13
uv venv --python 3.10.13 "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# --- keep caches inside the repo (nothing under $HOME/.cache). Written into the venv's
#     activate script so it also applies to future SSH sessions and `nohup`-ed jobs.
#     TORCH_HOME matters only for stage 2 if any torchvision weight is ever pulled. ---
CACHE_DIR="$REPO_ROOT/cache"
mkdir -p "$CACHE_DIR/torch" "$CACHE_DIR/xdg"
if ! grep -q "ICNN_CACHE_BLOCK" "$VENV_DIR/bin/activate"; then
  cat >> "$VENV_DIR/bin/activate" <<EOF

# --- ICNN_CACHE_BLOCK (added by setup_venv.sh): caches stay inside the repo ---
export TORCH_HOME="$CACHE_DIR/torch"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
EOF
fi
export TORCH_HOME="$CACHE_DIR/torch"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"

# --- torch FIRST, from the explicit cu118 index; nothing pulls a default build later.
#     Same pin as the other reproductions (2.0.1+cu118); all of them use one torch version.
#     Only stage 2 (iCNN reconstruction) uses it; stage 1 never touches the GPU. ---
uv pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# --- numpy MUST stay on the 1.x ABI: requirements.txt pins numpy<2.0.0 AND torch 2.0.1 is
#     compiled against NumPy 1.x. 1.26.4 is the last 1.x with cp310 wheels. ---
uv pip install numpy==1.26.4

# --- feature-decoding stack (requirements.txt of the clone) ---
# bdpy = BData format + ModelTraining/ModelTest + evals.metrics + (stage 2) recon/dl.torch.
# fastl2lir = the authors' fast ridge with per-voxel selection (`num: 500`).
uv pip install "bdpy>=0.24" fastl2lir hydra-core omegaconf pyyaml tqdm matplotlib

# --- extras the scripts import but requirements.txt does NOT list ---
# scikit-learn: train_decoder_sklearn_ridge.py (the variant the NSD-Imagery authors used, Appendix A.6).
# hdf5storage: evaluation.py loads y_mean/y_norm with it. pandas: imported by evaluation.py.
# pillow/scipy/h5py: stage 2 (image I/O, .mat features, BData backend).
# nibabel: benchmark B only: src/prepare_nsd_imagery_icnn.py reads the NSD NIfTI betas and
# the nsdgeneral mask. Nothing in benchmark A needs it.
# scikit-image: benchmark B only: rgb2gray and the SSIM of the NSD-Imagery harness (Gaussian
# window, sigma=1.5, use_sample_covariance=False), so that SSIM is not written by hand: a uniform
# window reads about 0.09 low, which could be mistaken for a reproduction gap. The metrics
# themselves are computed by the common harness in evaluation/, not by a script of this folder.
uv pip install scikit-learn hdf5storage pandas h5py scipy pillow nibabel scikit-image

# --- re-pin numpy: some of the above may have resolved a 2.x on top of it ---
uv pip install numpy==1.26.4

# --- sanity check: numpy ABI, torch build + GPUs, and every import the 3 scripts need ---
python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())"
# numpy must print 1.26.4: if a 2.x slipped back in, re-run `uv pip install numpy==1.26.4`.
python - <<'PY' && echo "bdpy/fastl2lir/hydra import OK" || echo "WARN: some imports failed — check the log above"
import bdpy
from bdpy.dataform import Features, DecodedFeatures
from bdpy.ml import ModelTraining, ModelTest
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification
from fastl2lir import FastL2LiR
from sklearn.linear_model import Ridge
import hdf5storage, hydra, nibabel
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
print('bdpy', bdpy.__version__ if hasattr(bdpy, '__version__') else '(no __version__)')
PY
# Stage 2 only: bdpy.recon needs torch. Not fatal for stage 1 if it fails.
python -c "from bdpy.recon.torch.icnn import reconstruct; from bdpy.dl.torch.models import VGG19, AlexNetGenerator, layer_map" 2>/dev/null \
  && echo "bdpy.recon (stage 2) import OK" || echo "NOTE: bdpy.recon not importable — only needed for the reconstruction stage"

echo ""
echo "Done. venv at: $VENV_DIR"
echo "Isolated uv cache at: $UV_CACHE_DIR"
echo "Activate in future sessions with:  source \"$VENV_DIR/bin/activate\""
echo ""
echo "The scripts use RELATIVE paths (./data/...): run them from $FD_DIR"
echo "Next: download the data (about 70 GB, about 135 GB at the peak while unzipping):"
echo "  cd \"$FD_DIR/data\" && python download.py fmri_deeprecon_fmriprep_vc"
