#!/bin/bash
# Prove (or disprove) the SCLIP prompt-convention fix before re-running a 3-day job.
#
# Runs check_native_iou.py: SCLIP's own un-injected prediction vs templated vs bare-word
# injection, on the same benchmark images. ~15 minutes, so it fits htc comfortably and
# needs no interactive session (which keeps dying with the SSH connection).
#
# USAGE
#   sbatch verify_sclip.sbatch                # 8 categories x 2 images
#   sbatch verify_sclip.sbatch 12 3           # arg1 = n_categories, arg2 = n_images
#
# READ THE VERDICT AT THE BOTTOM OF THE LOG. Only re-run the full positive/negative
# jobs if it says CONFIRMED.
#
#SBATCH --job-name=verify_sclip
#SBATCH --partition=htc          # short job; htc backfills far better than public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G                # GPU jobs have a 24000 MB floor on SOL
#SBATCH --time=0-01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

N_CATEGORIES=${1:-8}
N_IMAGES=${2:-2}
THRESHOLDS=${3:-0,0.005,0.01,0.02,0.05,0.1}

# $HOME can be unset in some SOL shells; every path below depends on it.
if [ -z "${HOME:-}" ] || [ ! -d "${HOME:-}" ]; then
    HOME=$(getent passwd "$(id -un)" | cut -d: -f6); export HOME
    echo "HOME was unset -- recovered as $HOME"
fi

REPO=$HOME/workspace/OVS/taxonomy-driven-ovs-models
EXPDIR=$REPO/model_experiments/sclip
PY=$EXPDIR/sclip_venv/bin/python

export BENCHMARK_DIR=$HOME/workspace/OVS/benchmark
export HF_HOME=$HOME/workspace/OVS/hf_cache
export NLTK_DATA=$HOME/workspace/OVS/nltk_data
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
# Reduce allocator fragmentation: the [num_queries, H, W] logit map is ~5.4 GiB on a
# 2200x1650 ADE20K image with the 326-class benchmark vocabulary, and large blocks
# fragment badly on a 19.5 GiB card.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}
# SOL's /lib64/libstdc++.so.6 predates CXXABI_1.3.15 that this env's libicu needs
export LD_LIBRARY_PATH="$EXPDIR/sclip_venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$EXPDIR"
echo "=================================================="
echo "node   = $SLURMD_NODENAME   job = $SLURM_JOB_ID"
echo "n_categories=$N_CATEGORIES  n_images=$N_IMAGES"
echo "=================================================="

[ -x "$PY" ] || { echo "ABORT: no interpreter at $PY (is the env named .conda instead?)"; exit 1; }
[ -d "$BENCHMARK_DIR" ] || { echo "ABORT: BENCHMARK_DIR missing: $BENCHMARK_DIR"; exit 1; }

echo
echo "approaches.py embed() convention (MUST be desc=False for this test to be the fix):"
grep -n "^def embed" approaches.py

$PY - <<'PYEOF'
import sys, torch
print("python :", sys.executable)
print("torch  :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")
if not torch.cuda.is_available():
    # SCLIP loads OpenAI CLIP in fp16 and CPU Half kernels are incomplete -- this would
    # fail with 'not implemented for Half' rather than merely run slowly.
    sys.exit("ABORT: no CUDA device. SCLIP is fp16; a CPU run will not work.")
free, total = torch.cuda.mem_get_info()
print(f"gpu mem: {total/2**30:.1f} GiB total, {free/2**30:.1f} GiB free")
if total/2**30 < 16:
    print("WARNING: under 16 GiB -- if this OOMs, resubmit with --gres=gpu:a100:1 or --n-images 1.")
print("deps ok")
PYEOF

$PY check_native_iou.py --n-categories "$N_CATEGORIES" --n-images "$N_IMAGES" \
    --thresholds "$THRESHOLDS"

echo
echo "DONE -- read the verdict above. Re-run positive/negative ONLY on 'CONFIRMED'."
