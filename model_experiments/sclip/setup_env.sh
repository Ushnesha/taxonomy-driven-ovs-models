#!/usr/bin/env bash
# Creates ./sclip_venv with the exact pinned stack sclip_blend_experiment.py
# was verified against, and clones the real SCLIP repo into ./SCLIP.
#
# On Linux/CUDA (cluster), `mim install mmcv==2.0.1` should find a prebuilt
# wheel and just work. On macOS, no prebuilt mmcv wheel exists and it builds
# from source (~5 min); sclip_blend_experiment.py's mmcv.ops stub covers a
# broken compiled-extension case observed on macOS arm64. numpy is re-pinned
# below because several installs above pull in numpy>=2, which mmcv 2.0.1's
# ABI does not support.
set -e

python3 -m venv sclip_venv
source sclip_venv/bin/activate
pip install --upgrade pip
pip install "numpy<2" "setuptools<81" wheel

pip install torch==2.1.2 torchvision==0.16.2
pip install "numpy<2"

pip install openmim
mim install mmengine==0.10.7
pip install "numpy<2"

mim install mmcv==2.0.1 || pip install --no-build-isolation mmcv==2.0.1
pip install "numpy<2"

pip install mmsegmentation==1.1.1
pip install "numpy<2"

pip install ftfy regex "yapf==0.40.1" pycocotools requests nltk
python3 -c "import nltk; nltk.download('wordnet')"

# Needed to run this folder's expanded_benchmark_helpers.py/benchmark_data.py/
# approaches.py inside this venv -- newer transformers requires torch>=2.5,
# which conflicts with the mmcv 2.0.1 pin above, so these are pinned to
# versions contemporaneous with torch 2.1.2.
pip install "transformers==4.40.0" "sentence-transformers==2.7.0" ipython datasets sentencepiece
pip install "numpy<2"

if [ ! -d "SCLIP" ]; then
    git clone https://github.com/wangf3014/SCLIP.git
fi

echo "done. activate with: source sclip_venv/bin/activate"
