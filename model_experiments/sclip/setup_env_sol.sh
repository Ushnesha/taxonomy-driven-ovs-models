#!/usr/bin/env bash
# SOL-specific build of ./sclip_venv for the SCLIP experiments.
#
# Differences from setup_env.sh:
#   1. Creates the env with an EXPLICIT Python 3.10 (via mamba) instead of bare
#      `python3 -m venv`, which on SOL compute nodes resolves to the system
#      Python 3.6 -- where the newest available torch is 1.10.2, so the
#      `torch==2.1.2` pin fails with "No matching distribution found".
#      3.10 (not 3.11) because mmcv 2.0.1 has no cp311 prebuilt wheel.
#   2. Never uses `conda activate` / `source activate` -- on SOL that silently
#      falls back to mamba base. Every command uses the absolute interpreter.
#
# The result still lives at ./sclip_venv/bin/python, so alpha_sclip.sbatch needs
# no changes.
#
# Run on a GPU node:
#   salloc -p general -q public --gres=gpu:1 -c 8 --mem=32G -t 0-02:00:00
#   cd ~/workspace/OVS/taxonomy-driven-ovs-models/model_experiments/sclip
#   bash setup_env_sol.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The env lives at ./sclip_venv, same as setup_env.sh and what alpha_sclip.sbatch
# expects. Override only if you need it elsewhere (e.g. a tight home quota):
#   SCLIP_ENV_DIR=/scratch/$USER/OVS/envs/sclip bash setup_env_sol.sh
# (a non-default path gets symlinked back to ./sclip_venv, so the sbatch still works)
ENVDIR="${SCLIP_ENV_DIR:-$HERE/sclip_venv}"
PY="$ENVDIR/bin/python"

if [ ! -x "$PY" ]; then
    echo "==> creating Python 3.10 env at $ENVDIR"
    mkdir -p "$(dirname "$ENVDIR")"
    module load mamba/latest 2>/dev/null || true
    mamba create -y -p "$ENVDIR" python=3.10
fi

# SOL's RHEL nodes ship an old /lib64/libstdc++.so.6 (CXXABI up to ~1.3.11). The conda
# env's libicui18n.so.78 -- pulled in by python/sqlite3 -- needs CXXABI_1.3.15, and the
# dynamic loader prefers the system lib, giving:
#   ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found
# Fix: make sure the env carries its own modern libstdc++, and put it first on the
# library search path. LD_LIBRARY_PATH must also be set in alpha_sclip.sbatch.
echo "==> ensuring modern libstdc++ inside the env"
mamba install -y -p "$ENVDIR" -c conda-forge libstdcxx-ng libgcc-ng
export LD_LIBRARY_PATH="$ENVDIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# symlink ./sclip_venv -> $ENVDIR so the sbatch path works unchanged
if [ "$ENVDIR" != "$HERE/sclip_venv" ]; then
    rm -rf "$HERE/sclip_venv"
    ln -sfn "$ENVDIR" "$HERE/sclip_venv"
    echo "==> linked $HERE/sclip_venv -> $ENVDIR"
fi

# pip's download cache is throwaway and runs to several GB -- keep it off the home
# quota (this is what caused the earlier "No space left on device"). Override with
# PIP_CACHE_DIR=... if you want it elsewhere.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/OVS/pip_cache}"
mkdir -p "$PIP_CACHE_DIR" 2>/dev/null || unset PIP_CACHE_DIR

echo "==> using $PY"
"$PY" --version

"$PY" -m pip install --upgrade pip
"$PY" -m pip install "numpy<2" "setuptools<81" wheel

# torch 2.0.1 + cu118, NOT the 2.1.2 in setup_env.sh. Reason: mmcv 2.0.1 (which
# mmsegmentation 1.1.1 requires, <2.1.0) only has prebuilt wheels under
#   https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/
# Every cu121 / torch2.1 index carries mmcv 2.1.0+ only, so `mim install mmcv==2.0.1`
# there falls through to a source build that fails. Moving torch is the least risky
# change -- it keeps the mmcv/mmseg pins the vendored SCLIP/ code was verified against.
echo "==> torch 2.0.1 (CUDA 11.8 build)"
"$PY" -m pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118
"$PY" -m pip install "numpy<2"

# opencv-python 5.x hard-requires numpy>=2, which mmcv 2.0.1's ABI cannot use.
# Pin it BEFORE mmcv pulls it in as a dependency.
"$PY" -m pip install "opencv-python<5" "numpy<2"

echo "==> openmim / mmengine / mmcv / mmsegmentation"
"$PY" -m pip install openmim
"$ENVDIR/bin/mim" install mmengine==0.10.7
"$PY" -m pip install "numpy<2"

# Explicit wheel index -- do not let mim guess it.
MMCV_INDEX=https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
"$PY" -m pip install mmcv==2.0.1 -f "$MMCV_INDEX" --only-binary=mmcv
"$PY" -m pip install "numpy<2" "opencv-python<5"

# NOTE: this pulls in `openxlab`, whose stale requests~=2.28.2 / tqdm~=4.65.0 pins
# make pip print "dependency conflicts" warnings. Harmless -- openxlab is only
# OpenMMLab's checkpoint-download CLI, and SCLIP loads CLIP weights through its own
# vendored SCLIP/clip/clip.py instead. Do NOT downgrade requests/tqdm to satisfy it;
# transformers/datasets need the newer ones.
"$PY" -m pip install mmsegmentation==1.1.1
"$PY" -m pip install "numpy<2" "opencv-python<5"

echo "==> SCLIP + experiment deps"
"$PY" -m pip install ftfy regex "yapf==0.40.1" pycocotools requests nltk
"$PY" -m pip install "transformers==4.40.0" "sentence-transformers==2.7.0" \
    ipython datasets sentencepiece
"$PY" -m pip install "numpy<2"
"$PY" -m pip install pandas matplotlib          # for analyze_alpha.py

NLTK_DATA="${NLTK_DATA:-$HOME/workspace/OVS/nltk_data}" \
    "$PY" -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

[ -d "$HERE/SCLIP" ] || git clone https://github.com/wangf3014/SCLIP.git "$HERE/SCLIP"

echo
echo "==> verifying"
"$PY" - <<'PYEOF'
import sys, torch, numpy, cv2
print("python :", sys.version.split()[0], "|", sys.executable)
print("opencv :", cv2.__version__)
print("torch  :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")
print("numpy  :", numpy.__version__)
import mmcv, mmengine, mmseg
print("mmcv   :", mmcv.__version__, "| mmengine:", mmengine.__version__, "| mmseg:", mmseg.__version__)
print("OK")
PYEOF
echo
echo "done. Interpreter: $PY"
