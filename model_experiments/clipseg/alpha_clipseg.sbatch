#!/bin/bash
# COCO alpha sweep for CLIPSeg -- runs model_experiments/clipseg/alpha_value_experiment.py
# at full scale (every eligible COCO category, 300 positive images each).
#
# Submit from this directory:
#     cd ~/workspace/OVS/taxonomy-driven-ovs-models/model_experiments/clipseg
#     sbatch alpha_clipseg.sbatch
#
#SBATCH --job-name=alpha_clipseg
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

# ---------- paths ----------
PY=/home/sschaud5/.conda/envs/tasa/bin/python          # absolute interpreter -- do NOT rely on `conda activate` on SOL
REPO=$HOME/workspace/OVS/taxonomy-driven-ovs-models
EXPDIR=$REPO/model_experiments/clipseg
COCO_DIR=$HOME/workspace/OVS/coco_data
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$HOME/workspace/OVS/results/alpha_clipseg_${STAMP}

# ---------- caches (keep multi-GB downloads off the small home quota) ----------
export HF_HOME=$HOME/workspace/OVS/hf_cache
export TORCH_HOME=$HOME/workspace/OVS/torch_cache
export NLTK_DATA=$HOME/workspace/OVS/nltk_data
export PYTHONNOUSERSITE=1                               # ignore ~/.local, use only the tasa env
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

mkdir -p "$OUT" "$HF_HOME" "$TORCH_HOME" "$NLTK_DATA" "$COCO_DIR"
cd "$EXPDIR"
echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID"
echo "out=$OUT"

# ---------- sanity: correct interpreter, deps present, wordnet available ----------
$PY - <<'EOF'
import sys, torch, transformers, pycocotools, nltk, numpy, PIL
print("python :", sys.executable)
print("prefix :", sys.prefix)
print("torch  :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
print("deps ok")
EOF

# ---------- run ----------
# --n-images 300 : plenty for a smooth alpha curve (this is calibration, not a headline
#                  number). Change to -1 for the exhaustive every-positive-image run.
$PY alpha_value_experiment.py \
    --coco-dir "$COCO_DIR" \
    --grid both \
    --n-categories -1 \
    --n-images 300 \
    --out-dir "$OUT"

echo "DONE -> $OUT"
ls -la "$OUT"
