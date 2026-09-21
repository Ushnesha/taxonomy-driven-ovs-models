#!/bin/bash
# COCO alpha sweep for GroupViT -- runs model_experiments/groupvit/alpha_value_experiment.py
# at full scale (every eligible COCO category, 350 positive images each), matching the
# CLIPSeg and SCLIP sweeps so the three models' alphas are derived the same way.
#
# USAGE
#   cd ~/workspace/OVS/taxonomy-driven-ovs-models/model_experiments/groupvit
#   sbatch alpha_groupvit.sbatch                 # coarse + fine in one job
#   sbatch alpha_groupvit.sbatch coarse          # coarse only (recommended first)
#   sbatch alpha_groupvit.sbatch fine <OUT_DIR>  # resume / run the fine grid separately
#   sbatch alpha_groupvit.sbatch coarse "" 20    # arg3 = --n-images (dry run)
#
# Resumable: checkpointed on (grid, category, img_ref). Resubmit the SAME grid with the
# SAME OUT_DIR to continue. summarize() runs after EACH grid, so a job killed during
# "both" still leaves a usable coarse summary.
#
#SBATCH --job-name=alpha_groupvit
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

GRID=${1:-coarse}
PY=/home/sschaud5/.conda/envs/tasa/bin/python   # absolute path -- `conda activate` silently falls back to mamba base on SOL
REPO=$HOME/workspace/OVS/taxonomy-driven-ovs-models
EXPDIR=$REPO/model_experiments/groupvit
COCO_DIR=$HOME/workspace/OVS/coco_data
OUT=${2:-$HOME/workspace/OVS/results/alpha_groupvit_$(date +%Y%m%d_%H%M%S)}
N_IMAGES=${3:-350}                               # 350 = same as clipseg/sclip, for cross-model consistency

export HF_HOME=$HOME/workspace/OVS/hf_cache
export TORCH_HOME=$HOME/workspace/OVS/torch_cache
export NLTK_DATA=$HOME/workspace/OVS/nltk_data
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

mkdir -p "$OUT" "$HF_HOME" "$TORCH_HOME" "$NLTK_DATA" "$COCO_DIR"
cd "$EXPDIR"
echo "=================================================="
echo "node   = $SLURMD_NODENAME   job = $SLURM_JOB_ID"
echo "grid   = $GRID   n_images = $N_IMAGES"
echo "OUT    = $OUT"
echo "  (pass this OUT as arg 2 to resume)"
echo "=================================================="

# ---------- sanity: right interpreter, GPU present, wordnet available ----------
$PY - <<'PYEOF'
import sys, torch, transformers, pycocotools, nltk, numpy, PIL
print("python      :", sys.executable)
print("torch       :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")
print("transformers:", transformers.__version__)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
if not torch.cuda.is_available():
    sys.exit("ABORT: no CUDA device visible -- a CPU run would take weeks.")
print("deps ok")
PYEOF

$PY alpha_value_experiment.py \
    --coco-dir "$COCO_DIR" \
    --grid "$GRID" \
    --n-categories -1 \
    --n-images "$N_IMAGES" \
    --out-dir "$OUT"

echo "DONE ($GRID) -> $OUT"
ls -la "$OUT"
