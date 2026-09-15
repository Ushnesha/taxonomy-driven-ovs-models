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

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Adjust the --index-url / torch version to match your cluster's CUDA (nvidia-smi). This
# default targets CUDA 11.7, CAT-Seg's own tested combo.
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt

# detectron2: no prebuilt wheel for most combos -- builds from source (several minutes).
pip install "git+https://github.com/facebookresearch/detectron2.git"

if [ ! -d "CAT-Seg" ]; then
    git clone https://github.com/KU-CVLAB/CAT-Seg.git
fi

python3 -c "import nltk; nltk.download('wordnet')"

echo "done. activate with: source venv/bin/activate"
echo
echo "Next: download a checkpoint from CAT-Seg's model zoo (see CAT-Seg/README.md), e.g."
echo "  CAT-Seg (L), ViT-L/14@336px: https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_large.pth"
echo "  CAT-Seg (B), ViT-B/16:       https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_base.pth"
echo "then run the verification in this folder's README.md before trusting a real run."
