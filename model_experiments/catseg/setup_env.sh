#!/usr/bin/env bash
# Creates ./venv with detectron2 + CAT-Seg's own pinned stack (verified against
# https://github.com/KU-CVLAB/CAT-Seg's requirements.txt/INSTALL.md), and vendors the real
# CAT-Seg repo into ./CAT-Seg.
#
# detectron2 has no prebuilt wheel for most CUDA/torch combos and must be built from
# source (the `git+https://...` install below) -- this needs a working nvcc matching your
# torch build's CUDA version on the cluster; expect this step to take several minutes.
# CAT-Seg's own INSTALL.md's tested combo is Python 3.8 + torch 1.13.1 + CUDA 11.7; adjust
# the torch install line below to match your cluster's actual CUDA version (`nvidia-smi`)
# if different -- see https://pytorch.org/get-started/previous-versions/.
set -e

# ---------------------------------------------------------------------------
# ASU SOL notes (learned the hard way on the sclip env -- see sclip/setup_env_sol.sh):
#  * `python3 -m venv` on SOL picks up the SYSTEM python 3.6, and `pip install torch`
#    then resolves to a max of 1.10.2. Create the interpreter with mamba instead.
#  * CAT-Seg's own INSTALL.md is tested on Python 3.8; several pins below
#    (pillow==8.2.0, opencv-python==4.5.1.48) have no wheels for 3.11+ and will try
#    to build from source. Stick to 3.8.
#  * `conda activate` frequently no-ops on SOL -- always call the absolute
#    interpreter path ($ENVDIR/bin/python), never a bare `python`.
#  * Keep pip's cache off the home quota; 44G of ~/.cache filled it last time.
# ---------------------------------------------------------------------------
if [ -z "${HOME:-}" ] || [ ! -d "${HOME:-}" ]; then
    HOME=$(getent passwd "$(id -un)" | cut -d: -f6); export HOME
    echo "HOME was unset -- recovered as $HOME"
fi
ENVDIR=${ENVDIR:-$(pwd)/catseg_venv}          # must match PY= in alpha_catseg.sbatch / benchmark_catseg.sbatch
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/$USER/pip_cache}
mkdir -p "$PIP_CACHE_DIR"

# Idempotent: if the allocation times out mid-build, just rerun this script. mamba
# refuses to create over an existing prefix, and pip skips already-satisfied packages,
# so a rerun resumes rather than starting over.
if [ -x "$ENVDIR/bin/python" ]; then
    echo "reusing existing env at $ENVDIR ($($ENVDIR/bin/python --version 2>&1))"
elif command -v mamba >/dev/null 2>&1; then
    echo "creating $ENVDIR with mamba (python 3.8 -- CAT-Seg's tested version)"
    mamba create -y -p "$ENVDIR" python=3.8
else
    echo "mamba not found -- falling back to venv. CHECK the python version below is 3.8+,"
    echo "otherwise torch will silently resolve to an ancient release."
    python3 -m venv "$ENVDIR"
fi

# SOL's /lib64/libstdc++.so.6 lacks CXXABI_1.3.15, which conda-installed libs need.
# Outside the create branch so a rerun still applies it.
command -v mamba >/dev/null 2>&1 && mamba install -y -p "$ENVDIR" -c conda-forge libstdcxx-ng libgcc-ng

PY="$ENVDIR/bin/python"
$PY --version
$PY -m pip install --upgrade pip
pip() { $PY -m pip "$@"; }      # so every pip below hits THIS env, not mamba base

# Adjust the --index-url / torch version to match your cluster's CUDA (nvidia-smi). This
# default targets CUDA 11.7, CAT-Seg's own tested combo.
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt

# ---------------------------------------------------------------------------
# detectron2: no prebuilt wheel for most combos -- builds from source.
#
# It compiles CUDA kernels, so nvcc's MAJOR version must match the CUDA that
# torch was built against (torch only warns on a minor mismatch, but raises on a
# major one). SOL's default module is CUDA 12.x while CAT-Seg's stack is cu117,
# which fails with:
#     The detected CUDA version (12.9) mismatches the version that was used to
#     compile PyTorch (11.7).
# Check that up front rather than 10 minutes into a compile.
# ---------------------------------------------------------------------------
TORCH_CUDA=$($PY -c "import torch;print(torch.version.cuda or '')")
NVCC_CUDA=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
echo "torch built against CUDA: ${TORCH_CUDA:-none};  nvcc on PATH: ${NVCC_CUDA:-none}"
if [ -z "$NVCC_CUDA" ]; then
    echo "ABORT: no nvcc on PATH. module load a CUDA matching torch (${TORCH_CUDA})." >&2
    exit 1
fi
if [ "${TORCH_CUDA%%.*}" != "${NVCC_CUDA%%.*}" ]; then
    echo "ABORT: CUDA major mismatch (torch ${TORCH_CUDA} vs nvcc ${NVCC_CUDA})." >&2
    echo "  Fix EITHER side, then re-run this script:" >&2
    echo "   a) module avail cuda   &&   module load a ${TORCH_CUDA%%.*}.x build   (preferred)" >&2
    echo "   b) reinstall torch for CUDA ${NVCC_CUDA%%.*}.x and update the pip line above" >&2
    exit 1
fi

export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
# Build kernels only for the GPUs we actually run on (A100 = sm_80). Without this
# nvcc targets every architecture it knows and the build takes ~3x longer.
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
echo "CUDA_HOME=$CUDA_HOME  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# @v0.6: detectron2's main branch tracks recent torch and breaks against 1.13.1.
# --no-build-isolation: build against THIS env's torch, not a fresh one pip would
# otherwise fetch into a sandbox.
pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@v0.6"

if [ ! -d "CAT-Seg" ]; then
    git clone https://github.com/KU-CVLAB/CAT-Seg.git
fi

# $PY, not python3: the system interpreter would download into the wrong site-packages
# and leave the env without wordnet. NLTK_DATA keeps the corpus off the home quota.
# $HOME is not always set inside `srun --pty bash` on SOL. Unguarded, the default
# below collapses to "/workspace/OVS/nltk_data" and mkdir dies with
# "cannot create directory '/workspace': Permission denied".
if [ -z "${HOME:-}" ] || [ ! -d "${HOME:-}" ]; then
    HOME=$(getent passwd "$(id -un)" | cut -d: -f6)
    export HOME
    echo "HOME was unset -- recovered as $HOME"
fi
export NLTK_DATA=${NLTK_DATA:-$HOME/workspace/OVS/nltk_data}
mkdir -p "$NLTK_DATA" || { echo "ABORT: cannot create NLTK_DATA=$NLTK_DATA" >&2; exit 1; }
$PY -c "import nltk; nltk.download('wordnet', download_dir='$NLTK_DATA'); nltk.download('omw-1.4', download_dir='$NLTK_DATA')"

echo "done."
echo "IMPORTANT -- before running anything by hand, in every new shell:"
echo "  export LD_LIBRARY_PATH=$ENVDIR/lib:\$LD_LIBRARY_PATH"
echo "  (SOL's /lib64/libstdc++.so.6 lacks CXXABI_1.3.15 that this env's libicu needs.)"
echo "  The sbatch files already set this."
echo
echo "Use the ABSOLUTE interpreter path, not \`conda activate\`:"
echo "  $ENVDIR/bin/python"
echo "This is the path alpha_catseg.sbatch / benchmark_catseg.sbatch expect as PY=."
echo
echo "Verify:"
echo "  $ENVDIR/bin/python -c \"import torch,detectron2;print(torch.__version__,torch.cuda.is_available(),detectron2.__version__)\""
echo
echo "Next: download a checkpoint from CAT-Seg's model zoo (see CAT-Seg/README.md), e.g."
echo "  CAT-Seg (L), ViT-L/14@336px: https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_large.pth"
echo "  CAT-Seg (B), ViT-B/16:       https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_base.pth"
echo "then run the verification in this folder's README.md before trusting a real run."
