#!/bin/bash
# Build the venv (.venv_targets) for target extraction, PCA fitting and reconstruction.
#
# One venv serves every step: Versatile Diffusion runs through the deprecated `diffusers` pipeline.
# torch must be a cu128 build, because the default PyPI wheel has no kernels for compute capability
# 12.0 (RTX 5060 Laptop, Blackwell). requirements-lock.txt is the frozen list of the environment that
# produced the thesis numbers.
#
# Usage (from the trisect/ folder):   . setup_venv.sh   (source it so the venv stays active)

set -e

# --- repo root = folder containing this script ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
VENV_DIR="$REPO_ROOT/.venv_targets"

# --- Python 3.12 (system Python; no uv/pyenv needed locally) ---
/usr/bin/python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# --- torch FIRST, from the explicit cu128 index (default PyPI wheel lacks sm_120 kernels) ---
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# --- smoke test 1: torch sees the GPU and can run a matmul on it ---
python -c "
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
x = torch.randn(4096, 4096, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('matmul ok on', torch.cuda.get_device_name(0))
"

# --- rest of the stack: VD (diffusers), CLIP/DINOv2 targets, PCA, HDF5 I/O ---
# nibabel: reads the NSD-Imagery .nii.gz volumes (prepare_betas.py --stage imagery, reused verbatim
# from the iCNN reproduction's prepare_nsd_imagery_icnn.py); no GPU/torch coupling.
pip install diffusers==0.39.0 transformers==5.15.0 open_clip_torch==3.3.0 \
  h5py==3.16.0 scikit-learn==1.9.0 scipy==1.18.0 pillow==12.3.0 accelerate==1.14.0 nibabel==5.3.2

# py-spy: sampling profiler to peek into long-running background jobs (e.g. the RidgeCV fits of
# fit_ridge.py) without instrumenting them. It needs `sudo env "PATH=$PATH" py-spy dump --pid ...`
# because ptrace_scope=1 blocks same-user attach.
pip install py-spy==0.4.2

# --- torchvision fix: plain `pip install` above (via transformers/diffusers deps) pulls the
# vanilla PyPI torchvision wheel, which is NOT linked against the cu128 torch build installed
# above -> `RuntimeError: operator torchvision::nms does not exist` at import time. Force the
# matching +cu128 build from the same index.
pip install --force-reinstall --no-deps torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# --- smoke test 2: the deprecated VD pipeline imports and is compatible with this transformers ---
python -c "
from diffusers.pipelines.deprecated.versatile_diffusion import VersatileDiffusionDualGuidedPipeline
print('import ok')
"

# DINOv2 is NOT installed here: it downloads on first use via
#   torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

echo ""
echo "Done. venv at: $VENV_DIR"
echo "Activate in future sessions with:  source \"$VENV_DIR/bin/activate\""
