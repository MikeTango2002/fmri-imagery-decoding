#!/bin/bash
# setup_venv.sh: venv-based replacement for the conda setup.sh of MindEye2 (the server has no conda).
# Builds the pinned MindEye2 environment using `uv` and a project-local .venv.
# Prereqs: `uv` must already be on PATH.
# GPU target: RTX 6000 Ada (sm_89).
#   torch 2.1.0's default PyPI wheel is the cu121 build, which already supports
#   sm_89; (unlike MindEye1's torch 2.0.1 = cu117) no custom index is needed here.
# Run from inside src/:   . setup_venv.sh   (source it so the venv stays active)

set -e

# --- locate repo root (parent of this src/ dir) so .venv lives at mindeye_II/.venv ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_ROOT/.venv"

command -v uv >/dev/null 2>&1 || { echo "ERROR: 'uv' not found on PATH. Install it first."; return 1 2>/dev/null || exit 1; }

# --- uv local isolation: the uv cache and the Python build stay inside the repository folder ---
export UV_CACHE_DIR="$REPO_ROOT/.uv_cache"
export UV_PYTHON_INSTALL_DIR="$REPO_ROOT/.uv_python"

# --- Python 3.11 (upstream setup.sh uses python3.11; server system Python is 3.12) ---
uv python install 3.11
uv venv --python 3.11 "$VENV_DIR"
source "$VENV_DIR/bin/activate"

uv pip install --upgrade pip

# --- exact dependency set from upstream setup.sh (same pins that fix the recipe) ---
# torch/torchvision pulled in here from the default index (cu121 wheels, Ada-ready).
uv pip install \
  numpy matplotlib==3.8.2 jupyter jupyterlab_nvdashboard jupyterlab tqdm \
  scikit-image==0.22.0 accelerate==0.24.1 webdataset==0.2.73 pandas==2.2.0 \
  einops ftfy regex kornia==0.7.1 h5py==3.10.0 open_clip_torch==2.24.0 \
  torchvision==0.16.0 torch==2.1.0 transformers==4.37.2 xformers==0.0.22.post7 \
  torchmetrics==1.3.0.post0 diffusers==0.23.0 deepspeed==0.13.1 wandb \
  omegaconf==2.3.0 pytorch-lightning==2.0.1 sentence-transformers==2.5.1 \
  evaluate==0.4.1 nltk==3.8.1 rouge_score==0.1.2 umap-learn==0.5.5 "pyarrow<14"
# NOTE: upstream setup.sh pins `umap==0.1.1`, but that PyPI package is an unrelated stub and
# does NOT provide the `umap.UMAP` used in final_evaluations.py (that comes from `umap-learn`).
# Upstream only "worked" because umap-learn was already in their base env; under uv it fails to
# resolve. Corrected to umap-learn, which supplies the `import umap` the eval code actually needs.
# pyarrow<14: evaluate==0.4.1 pulls in a `datasets` version whose _ArrayXDExtensionType still
# subclasses pa.PyExtensionType, removed in pyarrow 14. Left unpinned, uv resolves the newest
# pyarrow and final_evaluations.py's `import evaluate` dies with
# "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", after 7a/7b already
# succeeded; it looks like a late-pipeline failure but is really just this transitive pin gap.

# CLIP from OpenAI's git repo (provides clip.load); --no-deps mirrors upstream so it doesn't
# pull an incompatible torch. NOTE: the requirement must be named `clip` (not `openai-clip`):
# the repo's setup.py declares name="clip", and recent uv rejects the name mismatch with
# "metadata name `clip` does not match given name `openai-clip`". This is STILL OpenAI's real
# CLIP (from git), NOT the unrelated PyPI package also named "clip"; that one must not be used.
uv pip install "clip @ git+https://github.com/openai/CLIP.git" --no-deps
# dalle2-pytorch supplies DiffusionPrior/CausalTransformer for models.py's PriorNetwork.
# Its torch requirement has no upper bound, and uv re-resolves the whole environment: installed
# on its own it silently upgraded torch to 2.13, which breaks deepspeed 0.13.1 at import
# (torch >=2.2 removed torch.distributed.elastic.agent.server.api.log). Repeating the pins here
# forces the resolver to pick a dalle2-pytorch compatible with torch 2.1.0 instead.
uv pip install dalle2-pytorch torch==2.1.0 torchvision==0.16.0 xformers==0.0.22.post7

# --- extras needed for this workflow (not in upstream setup.sh) ---
# nbconvert: converts the .ipynb notebooks to the .py mirrors that run under nohup;
# huggingface_hub: used by download_mindeyev2.py (transformers already pulls it, pinned for safety).
uv pip install nbconvert "huggingface_hub<0.26"
uv pip install "setuptools==69.5.1"
uv pip install nibabel

# --- sanity check: exact pins, both GPUs visible, and every import the pipeline needs ---
# Asserts rather than prints: a drifted torch only surfaces days later, as an ImportError
# inside `accelerate launch`, after the dataset and the frozen cache are already built.
python - <<'PY'
import torch, torchvision, deepspeed, diffusers, transformers, xformers
from dalle2_pytorch import DiffusionPrior  # noqa: F401
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())
print('ds', deepspeed.__version__, '| diffusers', diffusers.__version__, '| tf', transformers.__version__)
assert torch.__version__.startswith('2.1.0'), f'torch drifted to {torch.__version__} (expected 2.1.0)'
assert torchvision.__version__.startswith('0.16.0'), f'torchvision drifted to {torchvision.__version__}'
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, 'expected 2 visible GPUs'
print('env OK')
PY

echo ""
echo "Done. venv at: $VENV_DIR"
echo "Local uv cache isolated at: $UV_CACHE_DIR"
echo "Activate in future sessions with:  source \"$VENV_DIR/bin/activate\""
echo ""
echo "NOTE: do NOT run 'accelerate config' -- the wizard's default gradient_clipping=1.0 would"
echo "      add clipping Train.py does not have. Pass the config file shipped in the repo root:"
echo "      accelerate launch --config_file=../accelerate_2gpu_zero2_gas4.yaml ..."
