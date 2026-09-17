#!/bin/bash
# COCO alpha sweep for SCLIP -- model_experiments/sclip/alpha_value_experiment.py
#
# USAGE
#
#   sbatch alpha_sclip.sbatch both        # coarse then fine, one job (default)
#
#   # guarded dry run -- all 37 categories, 2 images each, ~15 min:
#   sbatch --time=0-01:00:00 alpha_sclip.sbatch coarse /tmp/sclip_thdfix 2
#
# Or split them, passing stage 1's OUT dir to stage 2 so it appends:
#   sbatch alpha_sclip.sbatch coarse
#   sbatch alpha_sclip.sbatch fine /home/sschaud5/workspace/OVS/results/alpha_sclip_20260910_2200
#
# Cost split: coarse is 88 embeddings/image, fine is 248 -- so coarse is ~26% of a
# "both" run and fine ~74%. The script summarizes after EACH grid, so even a "both"
# job that hits the time limit leaves a usable coarse summary on disk.
#
# If a job hits the time limit, just resubmit the SAME command -- the detail CSV is
# checkpointed on (grid, category, img_ref) and resumes where it stopped.
#
#SBATCH --job-name=alpha_sclip
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

GRID=${1:-both}                            # arg1: coarse | fine | both
N_IMAGES=${3:-350}                         # arg3: images per category (use a small
                                           #       value for a guarded dry run)
REPO=$HOME/workspace/OVS/taxonomy-driven-ovs-models
EXPDIR=$REPO/model_experiments/sclip
PY=$EXPDIR/sclip_venv/bin/python             # SCLIP's OWN venv -- NOT the tasa conda env
COCO_DIR=$HOME/workspace/OVS/coco_data       # shared with the clipseg run, already populated
OUT=${2:-$HOME/workspace/OVS/results/alpha_sclip_$(date +%Y%m%d_%H%M%S)}

export HF_HOME=$HOME/workspace/OVS/hf_cache
export TORCH_HOME=$HOME/workspace/OVS/torch_cache
export NLTK_DATA=$HOME/workspace/OVS/nltk_data
export PYTHONNOUSERSITE=1
# SOL's /lib64/libstdc++.so.6 is too old for the conda env's libicu (CXXABI_1.3.15).
# Put the env's own lib dir first or `import sqlite3` (and thus nltk) fails.
export LD_LIBRARY_PATH="$EXPDIR/sclip_venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

mkdir -p "$OUT" "$HF_HOME" "$TORCH_HOME" "$NLTK_DATA"
cd "$EXPDIR"
echo "=================================================="
echo "node   = $SLURMD_NODENAME"
echo "job    = $SLURM_JOB_ID"
echo "grid   = $GRID"
echo "n-imgs = $N_IMAGES"
echo "OUT    = $OUT"
echo "  (pass this OUT as arg 2 for the next stage)"
echo "=================================================="

if [ ! -x "$PY" ]; then
    echo "ERROR: $PY not found. Build it first ON A GPU NODE:"
    echo "  salloc -p general -q public --gres=gpu:1 -c 8 --mem=32G -t 0-02:00:00"
    echo "  cd $EXPDIR && bash setup_env.sh"
    exit 1
fi

# ---------- sanity: mmcv/mmseg is the fragile part of the SCLIP stack ----------
$PY - <<'PYEOF'
import sys, torch, numpy, nltk
print("python :", sys.executable)
print("prefix :", sys.prefix)
print("torch  :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")
print("numpy  :", numpy.__version__, "(must be <2 for mmcv 2.0.1)")
import mmcv, mmengine, mmseg
print("mmcv   :", mmcv.__version__, "| mmengine:", mmengine.__version__, "| mmseg:", mmseg.__version__)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
if not torch.cuda.is_available():
    sys.exit("ABORT: no CUDA device visible. SCLIP on CPU would take weeks -- check that "
             "--gres=gpu:1 was honored and that this torch build sees the GPU.")
print("deps ok")
PYEOF

# ---------- run ----------
# 350 images/category, matching the CLIPSeg run for cross-model comparability.
$PY alpha_value_experiment.py \
    --coco-dir "$COCO_DIR" \
    --grid "$GRID" \
    --n-categories -1 \
    --n-images "$N_IMAGES" \
    --out-dir "$OUT"

echo "DONE ($GRID) -> $OUT"
ls -la "$OUT"
