#!/bin/bash
# setup_venv.sh: venv-based replacement for the conda setup.sh of MindEye1 (the server has no conda).
# Builds the pinned MindEye1 environment using `uv` and a project-local .venv.
# Prereqs: `uv` must already be on PATH.
# GPU target: RTX 6000 Ada (sm_89) -> torch 2.0.1 built for CUDA 11.8 (cu118) is what this script asks for.
# NOTE: the torch line below never took effect on the server. The environment that produced the
# thesis numbers holds torch 2.13.0 (CUDA 13 wheels); it is frozen in requirements-lock.txt.
# Run from inside src/:   . setup_venv.sh   (source it so the venv stays active)

set -e

# --- locate repo root (parent of this src/ dir) so .venv lives at mindeye_I/.venv ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_ROOT/.venv"

command -v uv >/dev/null 2>&1 || { echo "ERROR: 'uv' not found on PATH. Install it first."; return 1 2>/dev/null || exit 1; }

# --- uv local isolation: the uv cache and the Python build stay inside the repository folder ---
export UV_CACHE_DIR="$REPO_ROOT/.uv_cache"
export UV_PYTHON_INSTALL_DIR="$REPO_ROOT/.uv_python"

# --- Python 3.10.8 (torch 2.0.1 has no wheels for 3.12+); uv fetches a standalone build ---
uv python install 3.10.8
uv venv --python 3.10.8 "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# --- torch FIRST, from the cu118 index; nothing pulls the default cu117 build ---
uv pip install torch==2.0.1 torchvision==0.15.2

# --- everything else (mirrors the original conda setup.sh, all via pip) ---
uv pip install numpy matplotlib tqdm scikit-image jupyterlab
uv pip install accelerate
uv pip install clip-retrieval webdataset pandas ftfy regex kornia umap-learn
# OpenAI CLIP from git (provides clip.load). The requirement must be named `clip`, as the repo's
# setup.py declares: recent uv rejects `openai-clip @ git+...` for the name mismatch. This is still
# OpenAI's CLIP, not the unrelated PyPI package named "clip", which must not be installed.
# Keep this line after every other package that ships `clip/` (see "Stack" in the README).
uv pip install "clip @ git+https://github.com/openai/CLIP.git"
uv pip install dalle2-pytorch
uv pip install diffusers==0.14.0
uv pip install info-nce-pytorch==0.1.0
uv pip install pytorch-msssim
uv pip install awscli
uv pip install nibabel scipy

# --- pinned dependencies for MindEye1 compatibility (early 2023 snapshot) ---
uv pip install huggingface_hub==0.13.4 transformers==4.27.4 sentence-transformers==2.2.2

# --- sanity check: torch build + both GPUs visible ---
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())"

echo ""
echo "Done. venv at: $VENV_DIR"
echo "Local uv cache isolated at: $UV_CACHE_DIR"
echo "Activate in future sessions with:  source \"$VENV_DIR/bin/activate\""