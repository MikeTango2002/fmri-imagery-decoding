#!/bin/bash
# setup_venv.sh: venv-based replacement for `conda env create -f environment.yml` (no conda).
# Recreates the Brain-Diffuser environment using `uv` + a repo-local .venv.
# Prereq: `uv` must already be on PATH.
#
# GPU target: RTX 6000 Ada (sm_89). The original environment.yml pins pytorch 1.12.1+cu113,
# which has no kernels for sm_89 ("no kernel image available"). The only hardware-forced
# deviation is therefore torch 2.0.1 built for CUDA 11.8 (cu118, the floor with cu118 wheels).
#
# Usage (from the brain_diffuser/ root):   . setup_venv.sh   (source it so the venv stays active)

set -e

# --- repo root = folder containing this script ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
VENV_DIR="$REPO_ROOT/.venv"

command -v uv >/dev/null 2>&1 || { echo "ERROR: 'uv' not found on PATH. Install it first."; return 1 2>/dev/null || exit 1; }

# --- local uv isolation (per-user sandboxing) ---
export UV_CACHE_DIR="$REPO_ROOT/.uv_cache"
export UV_PYTHON_INSTALL_DIR="$REPO_ROOT/.uv_python"

# --- Python 3.10 (original was 3.8.13; 3.10 has wheels for the whole stack and stays compatible) ---
uv python install 3.10.11
uv venv --python 3.10.11 "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# --- torch FIRST, from the explicit cu118 index (the default wheel is cu117 and lacks Ada) ---
uv pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1 torchvision==0.15.2

# --- scientific stack (Ridge regressions = sklearn/numpy on CPU, the rest for features/recon) ---
uv pip install numpy==1.23.4 scipy==1.9.3 pandas==1.5.1 scikit-learn==1.1.3
# numpy is repeated here on purpose: it keeps the resolver from bumping it while
# solving the pins below (opencv/matplotlib only require numpy>=1.21).
# Two pins from environment.yml are raised because they predate Python 3.10 and ship no
# cp310 wheel; uv would build them from source (opencv pulls numpy==1.19.3 as a build
# requirement and that build fails without MKL/BLAS on the server):
#   opencv-python 4.5.1.48 -> 4.8.1.78   (cv2 is only imported by versatile_diffusion's
#                                         evaluator/data_factory, outside the pipeline)
#   matplotlib    3.4.2    -> 3.5.3      (first release with cp310 wheels; only used by
#                                         versatile_diffusion/lib/cfg_helper.py)
uv pip install numpy==1.23.4 h5py==3.7.0 nibabel==4.0.2 \
  matplotlib==3.5.3 opencv-python==4.8.1.78 scikit-image==0.19.3

# --- versatile_diffusion / vdvae lib dependencies (feature extraction + reconstruction) ---
uv pip install transformers==4.24.0 tokenizers
# pyyaml raised from environment.yml's 5.4.1: no cp310 wheel and its sdist no longer builds
# against Cython 3 ("AttributeError: cython_sources"). Safe here because every yaml.load() in
# versatile_diffusion/lib/cfg_helper.py already passes Loader=yaml.FullLoader, required by 6.x.
uv pip install einops==0.3.0 omegaconf==2.1.1 pyyaml==6.0.1 easydict==1.9 kornia==0.6.8 ftfy regex
uv pip install pytorch-lightning==1.9.5   # required by versatile_diffusion's LDM configs

# --- evaluation metrics (eval_extract_features / evaluate_reconstruction) ---
uv pip install lpips==0.1.3 torchmetrics==0.7.3 timm
# OpenAI CLIP for the CLIP metric (the PyPI "clip==1.0" from environment.yml is an unrelated project).
# The requirement must be named `clip`, not `openai-clip`: OpenAI's setup.py declares name="clip"
# and recent uv aborts on the mismatch ("metadata name `clip` does not match given name"). This is
# still the real CLIP from git. --no-deps keeps its unpinned `torch` from touching the cu118 build;
# its actual runtime deps (ftfy, regex, tqdm) are already installed above.
uv pip install "clip @ git+https://github.com/openai/CLIP.git" --no-deps

# --- taming-transformers: import required by versatile_diffusion's KL autoencoder configs ---
uv pip install "taming-transformers @ git+https://github.com/CompVis/taming-transformers.git"

# --- awscli to download NSD from the S3 bucket (data/download_nsddata.py) ---
uv pip install awscli

uv pip install ipython

# --- sanity check: torch build + both GPUs visible ---
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())"

echo ""
echo "Done. venv at: $VENV_DIR"
echo "Isolated uv cache at: $UV_CACHE_DIR"
echo "Activate in future sessions with:  source \"$VENV_DIR/bin/activate\""
