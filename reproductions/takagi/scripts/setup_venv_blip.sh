#!/bin/bash
# setup_venv_blip.sh: BLIP text-decoding venv for the Takagi & Nishimoto pipeline. It is the
# environment of the "+Decoded Text" branch: BLIP features, their ridge fit and the captions.
# Builds a SEPARATE environment (.venv_blip); do NOT source it into the main env.
# Prereqs: `uv` must already be on PATH (it is installed once, outside this folder).
# GPU target: RTX 6000 Ada (sm_89) -> torch built for CUDA 11.8 (cu118).
# Run from the takagi/ repo root:   bash setup_venv_blip.sh   (or `. setup_venv_blip.sh`)
#
# WHY Python 3.9 (not 3.10 like the main venv): BLIP pins transformers==4.15.0, which needs
# tokenizers==0.10.3. That old tokenizers has NO prebuilt wheel for cp310; on 3.10 uv tries
# to build it from source and fails with "can't find Rust compiler". cp39 wheels DO exist;
# 3.9 avoids the Rust toolchain entirely (and is closer to the upstream 3.8 target anyway).

set -e

# --- locate repo root (dir holding this script) so the venv lives at takagi/.venv_blip ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv_blip"

command -v uv >/dev/null 2>&1 || { echo "ERROR: 'uv' not found on PATH. Install it first."; return 1 2>/dev/null || exit 1; }

# --- uv local isolation (per-user sandbox: cache + python builds under the repo) ---
export UV_CACHE_DIR="$REPO_ROOT/.uv_cache"
export UV_PYTHON_INSTALL_DIR="$REPO_ROOT/.uv_python"

uv python install 3.9
uv venv --python 3.9 "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# --- torch FIRST from the cu118 index (the default PyPI wheel is cu117 and fails on Ada) ---
uv pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# --- numpy 1.x, same reason as setup_venv.sh: torch 2.0.1 / torchvision 0.15.2 are compiled
#     against the NumPy 1.x C ABI. Left unpinned, the packages below drag in numpy 2.x (2.0.2 is
#     the last release with cp39 wheels) and every torch import dies with
#     "Failed to initialize NumPy: _ARRAY_API not found". 1.26.4 is the last 1.x with cp39
#     wheels. Installed right after torch so everything below resolves against it. ---
uv pip install numpy==1.26.4

# --- BLIP stack (pinned by upstream Takagi caption path) ---
uv pip install timm==0.4.12 transformers==4.15.0 fairscale==0.4.4 pycocoevalcap
# Same list as setup_venv.sh: these are the deps of the vendored nsd_access/nsda.py, which the
# BLIP scripts import. matplotlib and IPython look gratuitous for a feature-extraction job but
# nsda.py imports them at module level (pyplot/image at line 10-11, `from IPython import embed`
# at line 18); leaving them out makes `import nsd_access` fail outright.
uv pip install nibabel h5py pandas scipy pycocotools matplotlib tqdm IPython

# --- ridge stack: the BLIP branch does not stop at feature extraction, it also runs
#     codes/utils/ridge.py (--target blip) inside THIS venv, and that imports himalaya +
#     sklearn. Same line as in setup_venv.sh. numpy is re-pinned right
#     after because scikit-learn resolves numpy 2.x on cp39 and would break every torch import.
uv pip install himalaya scikit-learn
uv pip install numpy==1.26.4

# --- sanity check: torch cu118 build + GPUs visible, and BLIP deps importable ---
python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())"
# numpy must print 1.26.4: if a 2.x slipped back in, re-run `uv pip install numpy==1.26.4`.
python -c "import transformers, timm, fairscale" 2>/dev/null && echo "transformers/timm/fairscale import OK" || echo "WARN: some BLIP imports failed — check the log above"
python -c "import himalaya, sklearn" 2>/dev/null && echo "himalaya/sklearn import OK (ridge.py --target blip)" || echo "WARN: himalaya/sklearn missing — ridge.py will die at import"
# nsd_access lives in codes/utils and is reached via a CWD-relative sys.path.append("../../utils/"),
# so the BLIP scripts only import it when run from codes/caption/BLIP. Check it the same way.
(cd "$REPO_ROOT/codes/caption/BLIP" && python -c "
import sys; sys.path.append('../../utils/')
import nsd_access; print('nsd_access import OK (from codes/caption/BLIP)')") \
  || echo "WARN: nsd_access not importable — run the BLIP scripts from codes/caption/BLIP"

echo ""
echo "Done. BLIP venv at: $VENV_DIR"
echo "Activate with:  source \"$VENV_DIR/bin/activate\""
