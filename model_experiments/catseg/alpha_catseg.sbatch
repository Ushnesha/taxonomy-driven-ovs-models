#!/bin/bash
# COCO alpha sweep for CAT-Seg -- runs model_experiments/catseg/alpha_value_experiment.py
# at full scale (every eligible COCO category, 350 positive images each), matching the
# clipseg / sclip / groupvit sweeps so all four models' alphas are derived identically.
#
# USAGE
#   cd ~/workspace/OVS/taxonomy-driven-ovs-models/model_experiments/catseg
#   sbatch alpha_catseg.sbatch coarse                 # recommended first
#   sbatch alpha_catseg.sbatch coarse <OUT_DIR>       # resume an interrupted run
#   sbatch alpha_catseg.sbatch coarse "" 20           # arg3 = --n-images (dry run)
#
# Resumable: checkpointed on (grid, category, img_ref). Resubmit the SAME grid with the
# SAME OUT_DIR to continue. summarize() runs after EACH grid.
#
# CAT-Seg is a JOINT-SOFTMAX model like SCLIP: one forward pass per (variant, approach)
# rather than one batched call, so expect it to be far slower than CLIPSeg. Start coarse.
#
#SBATCH --job-name=alpha_catseg
#SBATCH --partition=public      # NOT 'general': as of the May 2026 SOL maintenance,
                                # 'general' holds only privately-owned nodes and
                                # 'public' is the partition for public ones.
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00      # CAT-Seg cannot batch: ~75 forward passes per image
                               # (variants x alphas x weighted/unweighted) vs CLIPSeg's
                               # one. Full coarse sweep is ~40-130 GPU-hours, so expect
                               # 1-3 resubmits even at the 2-day maximum. Resume with
                               # the SAME grid and OUT_DIR -- checkpointed on
                               # (grid, category, img_ref), so nothing is recomputed.
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

# $HOME is not always set inside `srun --pty bash` on SOL, and every path below
# depends on it -- unguarded, BENCHMARK_DIR becomes "/workspace/OVS/benchmark" and the
# run dies with FileNotFoundError on word_sets_v2.json. Recover it from passwd.
if [ -z "${HOME:-}" ] || [ ! -d "${HOME:-}" ]; then
    HOME=$(getent passwd "$(id -un)" | cut -d: -f6); export HOME
    echo "HOME was unset -- recovered as $HOME"
fi

GRID=${1:-coarse}
REPO=$HOME/workspace/OVS/taxonomy-driven-ovs-models
EXPDIR=$REPO/model_experiments/catseg
PY=$EXPDIR/catseg_venv/bin/python          # absolute path -- `conda activate` silently falls back to mamba base on SOL
COCO_DIR=$HOME/workspace/OVS/coco_data
OUT=${2:-$HOME/workspace/OVS/results/alpha_catseg_$(date +%Y%m%d_%H%M%S)}
N_IMAGES=${3:-350}
FINE_RANGE=${4:-}            # e.g. "0.75,0.85,0.02" -- ONLY meaningful with grid=fine

# Override from the command line without editing this file, e.g.
#   CATSEG_WEIGHTS=/scratch/$USER/model_base.pth \
#   CATSEG_CONFIG=$PWD/CAT-Seg/configs/vitb_384.yaml \
#   sbatch alpha_catseg.sbatch ...
CONFIG=${CATSEG_CONFIG:-$EXPDIR/CAT-Seg/configs/vitl_336.yaml}
WEIGHTS=${CATSEG_WEIGHTS:-$HOME/workspace/OVS/checkpoints/catseg_model_large.pth}

export HF_HOME=$HOME/workspace/OVS/hf_cache
export TORCH_HOME=$HOME/workspace/OVS/torch_cache
export NLTK_DATA=$HOME/workspace/OVS/nltk_data
export PYTHONNOUSERSITE=1
# SOL's /lib64/libstdc++.so.6 predates CXXABI_1.3.15, which the env's libicu
# needs -- without this every import dies with:
#   ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found
export LD_LIBRARY_PATH="$EXPDIR/catseg_venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

mkdir -p "$OUT" "$HF_HOME" "$TORCH_HOME" "$NLTK_DATA" "$COCO_DIR"
cd "$EXPDIR"
echo "=================================================="
echo "node   = $SLURMD_NODENAME   job = $SLURM_JOB_ID"
echo "grid   = $GRID   n_images = $N_IMAGES   threshold = 0.0 (argmax)"
echo "config = $CONFIG"
echo "weights= $WEIGHTS"
echo "OUT    = $OUT"
echo "  (pass this OUT as arg 2 to resume)"
echo "=================================================="

# ---- fail before the queue time is wasted, not after ----
[ -f "$CONFIG" ]  || { echo "ABORT: config not found: $CONFIG"; exit 1; }
[ -f "$WEIGHTS" ] || { echo "ABORT: weights not found: $WEIGHTS"; exit 1; }

$PY - <<'PYEOF'
import sys, torch
print("python :", sys.executable)
print("torch  :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")
import detectron2; print("detectron2:", detectron2.__version__)
import nltk
try: nltk.data.find("corpora/wordnet")
except LookupError: nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
if not torch.cuda.is_available():
    sys.exit("ABORT: no CUDA device visible.")
print("deps ok")
PYEOF

ARGS=(--coco-dir "$COCO_DIR" --config "$CONFIG" --weights "$WEIGHTS"
      --grid "$GRID" --n-categories -1 --n-images "$N_IMAGES" --out-dir "$OUT")
[ -n "$FINE_RANGE" ] && ARGS+=(--fine-range "$FINE_RANGE")

$PY alpha_value_experiment.py "${ARGS[@]}"

echo "DONE ($GRID) -> $OUT"
ls -la "$OUT"
