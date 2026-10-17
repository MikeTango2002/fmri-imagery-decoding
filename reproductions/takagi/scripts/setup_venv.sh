#!/bin/bash
# setup_venv.sh: venv-based setup (no conda) for the Takagi & Nishimoto pipeline.
# Builds the MAIN environment used by the core CVPR reconstruction path:
#   ridge regression (himalaya) + Stable Diffusion v1.4 (CompVis `ldm`) + nsd_access.
# Prereqs: `uv` must already be on PATH (it is installed once, outside this folder).
# GPU target: RTX 6000 Ada (sm_89) -> torch built for CUDA 11.8 (cu118).
# Run from the takagi/ repo root:   . setup_venv.sh   (source it so the venv stays active)
#
# NOTE: unlike MindEye there is no single environment that runs everything. The BLIP (text),
# GAN/iCNN and depth pipelines pin MUTUALLY INCOMPATIBLE `transformers` versions (BLIP
# 4.15.0, SD1 4.19.2, depth needs the newest). BLIP gets its own venv (setup_venv_blip.sh).
# This script only sets up the core path (ridge + SD1): feature extraction, the ridge fits
# of the CVPR branch, image generation and the identification metric.

set -e

# --- locate repo root (dir holding this script) so .venv lives at takagi/.venv ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
SD1_DIR="$REPO_ROOT/codes/diffusion_sd1/stable-diffusion"

command -v uv >/dev/null 2>&1 || { echo "ERROR: 'uv' not found on PATH. Install it first."; return 1 2>/dev/null || exit 1; }

# --- uv local isolation (per-user sandbox: cache + python builds under the repo) ---
export UV_CACHE_DIR="$REPO_ROOT/.uv_cache"
export UV_PYTHON_INSTALL_DIR="$REPO_ROOT/.uv_python"

# --- Python 3.10: the CompVis ldm stack targets 3.8, but 3.10 works and has wheels for
#     torch 2.0.1+cu118 (needed for Ada; 3.12 is too new for these pins). ---
uv python install 3.10.13
uv venv --python 3.10.13 "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# --- keep EVERY model/dataset cache inside the repo (nothing under $HOME/.cache) ---
# Written into the venv's activate script so it applies to every future shell session and to
# `nohup`-ed jobs, with nothing to re-export by hand. Covers: HuggingFace (HF_HOME /
# TRANSFORMERS_CACHE), torchvision's pretrained weights for the identification metric, which
# img2feat_decoded.py pulls with `pretrained=True` (TORCH_HOME), and anything else that follows
# the XDG spec. Marker keeps the append idempotent if the script is re-run over an existing venv.
CACHE_DIR="$REPO_ROOT/cache"
mkdir -p "$CACHE_DIR/hf" "$CACHE_DIR/torch" "$CACHE_DIR/xdg"
if ! grep -q "TAKAGI_CACHE_BLOCK" "$VENV_DIR/bin/activate"; then
  cat >> "$VENV_DIR/bin/activate" <<EOF

# --- TAKAGI_CACHE_BLOCK (added by setup_venv.sh): caches stay inside the repo ---
export HF_HOME="$CACHE_DIR/hf"
export TRANSFORMERS_CACHE="$CACHE_DIR/hf/transformers"
export TORCH_HOME="$CACHE_DIR/torch"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
EOF
fi
export HF_HOME="$CACHE_DIR/hf"
export TRANSFORMERS_CACHE="$CACHE_DIR/hf/transformers"
export TORCH_HOME="$CACHE_DIR/torch"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"

# --- torch FIRST, from the cu118 index; nothing pulls a default (cu117/sm-limited) build.
#     IMPORTANT: the upstream environment.yaml pins torch==1.11.0, which has NO sm_89 kernels
#     (fails on Ada with "no kernel image available"). The environment deliberately uses 2.0.1+cu118 instead
#     and rely on the newer pytorch-lightning below to stay compatible with the ldm code.
#     The explicit --index-url is REQUIRED: without it PyPI ships the default cu117 build
#     (torch 2.0.1 + nvidia-*-cu11==11.7.*), which also fails on Ada. ---
uv pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# --- numpy MUST stay on the 1.x ABI: torch 2.0.1 / torchvision 0.15.2 are compiled against
#     NumPy 1.x; a numpy 2.x resolved later by himalaya/pandas/opencv breaks every import
#     with "Failed to initialize NumPy: _ARRAY_API not found" (and torch loses numpy interop).
#     1.26.4 is the last 1.x release with cp310 wheels. Installed right after torch so all the
#     packages below resolve against it. ---
uv pip install numpy==1.26.4

# --- CompVis / ldm inference stack (adapted pins for torch 2.0 + Ada) ---
# pytorch-lightning bumped from the upstream 1.4.2 (which breaks on torch 2.x) to a 1.9.x that
# still exposes the LightningModule API the ldm code imports.
uv pip install pytorch-lightning==1.9.5 torchmetrics==0.11.4
uv pip install omegaconf==2.1.1 einops==0.6.1 kornia==0.6.9
uv pip install transformers==4.19.2          # SD1 text encoder (FrozenCLIPEmbedder)
# NOTE: transformers 4.19.2 can no longer DOWNLOAD from huggingface.co (the hub now answers with a
# relative Location header, which this version hands to requests unchanged -> MissingSchema).
# Downgrading huggingface_hub does NOT help: 4.19.2 does the HTTP itself. So the CLIP weights are
# fetched with wget below instead of bumping transformers.

# --- CLIP ViT-L/14: needed by SD1's FrozenCLIPEmbedder (ldm/modules/encoders/modules.py:141,
#     CLIPTokenizer + CLIPTextModel) and by the identification metric (img2feat_decoded.py:67-69,
#     CLIPModel + CLIPProcessor). The text weights are NOT inside sd-v1-4.ckpt; this ~1.7 GB
#     download is mandatory for the CVPR path. wget follows the hub's relative redirect correctly.
#     Both call sites hardcode the repo id "openai/clip-vit-large-patch14"; transformers treats an
#     identifier as a LOCAL path when a directory of that name exists relative to the cwd; the
#     two symlinks below make from_pretrained read from the repo; no upstream code patch, no
#     network at run time. ---
CLIP_DIR="$CACHE_DIR/hf/openai/clip-vit-large-patch14"
CLIP_BASE="https://huggingface.co/openai/clip-vit-large-patch14/resolve/main"
mkdir -p "$CLIP_DIR"
for F in config.json preprocessor_config.json tokenizer_config.json special_tokens_map.json \
         vocab.json merges.txt tokenizer.json pytorch_model.bin; do
  if [ ! -s "$CLIP_DIR/$F" ]; then
    echo "downloading CLIP ViT-L/14: $F"
    # .part + mv so an interrupted download never leaves a truncated file that looks complete
    wget -q --show-progress -O "$CLIP_DIR/$F.part" "$CLIP_BASE/$F"
    mv "$CLIP_DIR/$F.part" "$CLIP_DIR/$F"
  fi
done
# cwd of the scripts that load it: codes/utils (feature extraction, identification) and
# codes/diffusion_sd1 (generation)
ln -sfn "$CACHE_DIR/hf/openai" "$REPO_ROOT/codes/utils/openai"
ln -sfn "$CACHE_DIR/hf/openai" "$REPO_ROOT/codes/diffusion_sd1/openai"
# albumentations 0.4.3's setup.py does `import pkg_resources` at build time, which recent
# setuptools (>=81) no longer ships -> the isolated build dies with ModuleNotFoundError.
# Provide an old-enough setuptools in the env and build this one wheel WITHOUT isolation.
# (albumentations is only used by ldm's training/degradation code, not the CVPR inference
#  path, but the upstream pin is kept so the env matches.)
uv pip install "setuptools<81" wheel
uv pip install --no-build-isolation albumentations==0.4.3
uv pip install opencv-python imageio imageio-ffmpeg
uv pip install invisible-watermark test-tube

# taming-transformers + OpenAI CLIP are imported by the ldm modules (VQ/attention, cond stages).
# taming: NOT the CompVis git repo (`git+https://github.com/CompVis/taming-transformers.git`) --
# it has no `taming/__init__.py` and its setup.py uses find_packages(); the install produces
# metadata with ZERO packages and `ldm/models/autoencoder.py` dies with "No module named 'taming'"
# (same packaging bug as ldm below, but here there is no local tree to point a .pth at).
# taming-transformers-rom1504 is the same CompVis code republished with working packaging.
# --no-deps is REQUIRED: its install_requires lists unpinned torch/pytorch-lightning/omegaconf and
# would drag in a non-cu118 torch, which fails on Ada (sm_89). Every real dep is already installed.
uv pip install --no-deps taming-transformers-rom1504
uv pip install "clip @ git+https://github.com/openai/CLIP.git"   # this one packages fine (has __init__.py)

# install the local ldm package (provides `import ldm...`) in editable mode
uv pip install -e "$SD1_DIR"

# --- the editable install above is NOT enough to get `import ldm` working. The upstream tree has
#     no `ldm/__init__.py` (nor ldm/models/, ldm/modules/), and setup.py uses find_packages(),
#     which skips directories without __init__.py -> the `latent-diffusion` distribution installs
#     ZERO packages. Upstream gets away with it because their scripts run from the
#     stable-diffusion/ root, where `ldm` resolves as an implicit namespace package via cwd; the scripts here
#     run from codes/utils and codes/diffusion_sd1 instead. So drop a .pth in
#     site-packages: site reads it at interpreter startup, making `ldm` importable from any cwd
#     (also for `nohup`-ed jobs), with no PYTHONPATH to re-export each new shell. ---
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$SD1_DIR" > "$SITE_PACKAGES/ldm_sd1.pth"
echo "wrote $SITE_PACKAGES/ldm_sd1.pth -> $SD1_DIR"

# --- ridge regression + NSD access + feature extraction (requirements.txt of this repo) ---
uv pip install himalaya scikit-learn
uv pip install nibabel h5py pandas scipy pycocotools matplotlib tqdm IPython
uv pip install timm fairscale                # timm/fairscale are listed upstream; harmless here
uv pip install awscli                        # installs the `aws` CLI inside the venv (no root needed)

# --- sanity check: torch build + both GPUs visible, and ldm importable ---
python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())"
# numpy must print 1.26.4: if a 2.x slipped back in, re-run `uv pip install numpy==1.26.4`.
python -c "import ldm.util, ldm.models.diffusion.ddim, ldm.models.autoencoder; import himalaya; import nsd_access" 2>/dev/null && echo "ldm/taming/himalaya/nsd_access import OK" || echo "WARN: some core imports failed — check the log above"
# CLIP must load from the local dir (run from codes/utils, where the feature scripts run).
(cd "$REPO_ROOT/codes/utils" && python -c "
from transformers import CLIPTokenizer, CLIPTextModel
CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
" 2>/dev/null) && echo "CLIP ViT-L/14 loads from $CLIP_DIR — OK" || echo "WARN: CLIP failed to load — check the download above"

echo ""
echo "Done. MAIN venv (ridge + SD1 CVPR path) at: $VENV_DIR"
echo "Local uv cache isolated at: $UV_CACHE_DIR"
echo "Activate in future sessions with:  source \"$VENV_DIR/bin/activate\""
echo ""
echo "The other pipelines need SEPARATE venvs (conflicting deps). Each has its own script"
echo "— run from the takagi/ repo root, do NOT source into the main env:"
echo "  - BLIP text decoding:  bash setup_venv_blip.sh    (-> .venv_blip, Python 3.9)"
echo "  - depth (SD v2.0):     not evaluated, script not included"
echo "  - GAN/iCNN (bdpy):     not evaluated, script not included"

