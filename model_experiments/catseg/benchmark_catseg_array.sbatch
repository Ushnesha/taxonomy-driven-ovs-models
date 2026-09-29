#!/bin/bash
# SHARDED CAT-Seg benchmark run: a SLURM job array, one GPU per shard.
#
# CAT-Seg cannot batch -- its joint softmax takes one embedding per forward pass, so a
# single image costs ~20 passes (4 variants x 5 approaches). Sequentially that is days:
# 25 hours reached only category "cake". Sharding the CATEGORY list across array tasks
# is the only way to use more than one GPU, since the work is embarrassingly parallel
# across categories and each shard writes its own CSV.
#
# USAGE
#   sbatch --array=0-7 benchmark_catseg_array.sbatch positive
#   sbatch --array=0-7 benchmark_catseg_array.sbatch negative "" 30
#   # inherit a previous sequential run so its finished categories are not redone:
#   SEEN=/path/to/positive_set_experiment_catseg.csv \
#     sbatch --array=0-7 benchmark_catseg_array.sbatch positive
#
# All tasks in one array share an OUT dir named after SLURM_ARRAY_JOB_ID, so they land
# together and merge_shards.py can find them:
#   python3 merge_shards.py <OUT_DIR>
#
# Resumable per shard: resubmit the SAME array with the SAME OUT dir (arg 2) and each
# task skips what its own CSV already has.
#
#SBATCH --job-name=bench_catseg_arr
# 'general' holds only privately-owned nodes since the May 2026 SOL maintenance;
# 'public' is the partition for publicly available ones.
#SBATCH --partition=public
#SBATCH --qos=public
# PER TASK, not per array: --array=0-7 therefore occupies 8 GPUs at once.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out

set -euo pipefail

# SOL can reject a submission with "Invalid account or account/partition
# combination specified" when no account is named. Find yours with:
#   sacctmgr show assoc user=$USER format=account,partition,qos%40
# then submit with:  ACCOUNT=<name> sbatch --array=0-7 ...
# (an empty ACCOUNT leaves SLURM to use your default, which is usually right)

SET=${1:?usage: sbatch --array=0-7 benchmark_catseg_array.sbatch <positive|negative> [out_dir] [limit_images]}

# $HOME can be unset in some SOL shells; every path below depends on it.
if [ -z "${HOME:-}" ] || [ ! -d "${HOME:-}" ]; then
    HOME=$(getent passwd "$(id -un)" | cut -d: -f6); export HOME
    echo "HOME was unset -- recovered as $HOME"
fi

: "${SLURM_ARRAY_TASK_ID:?this script must be submitted with --array, e.g. --array=0-7}"
SHARD_I=$SLURM_ARRAY_TASK_ID
SHARD_N=$(( ${SLURM_ARRAY_TASK_MAX:-0} + 1 ))

ALPHA=${ALPHA:-0.80}            # CAT-Seg's own sweep value (coarse 0.80, fine 0.80)
REPO=$HOME/workspace/OVS/taxonomy-driven-ovs-models
EXPDIR=$REPO/model_experiments/catseg
PY=$EXPDIR/catseg_venv/bin/python
OUT=${2:-$HOME/workspace/OVS/results/${SET}_catseg_array_${SLURM_ARRAY_JOB_ID}}
LIMIT_IMAGES=${3:-}

CONFIG=${CATSEG_CONFIG:-$EXPDIR/CAT-Seg/configs/vitl_336.yaml}
WEIGHTS=${CATSEG_WEIGHTS:-/scratch/$USER/catseg_ckpt/catseg_model_large.pth}

export BENCHMARK_DIR=$HOME/workspace/OVS/benchmark
export HF_HOME=$HOME/workspace/OVS/hf_cache
export TORCH_HOME=$HOME/workspace/OVS/torch_cache
export NLTK_DATA=$HOME/workspace/OVS/nltk_data
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}
export LD_LIBRARY_PATH="$EXPDIR/catseg_venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

mkdir -p "$OUT"
cd "$EXPDIR"
echo "=================================================="
echo "node   = $SLURMD_NODENAME   array = ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "shard  = $SHARD_I / $SHARD_N"
echo "set    = $SET   alpha = $ALPHA (weighted)  threshold = 0.0 (argmax)"
echo "OUT    = $OUT"
echo "  (pass this OUT as arg 2 to resume; merge with merge_shards.py \$OUT)"
echo "=================================================="

if [ "$SHARD_N" -le 1 ]; then
    echo "ABORT: SLURM_ARRAY_TASK_MAX is $SLURM_ARRAY_TASK_MAX -- submit a RANGE (--array=0-7),"
    echo "  not a single index, or every category lands in one shard."
    exit 1
fi

[ -d "$BENCHMARK_DIR" ] || { echo "ABORT: BENCHMARK_DIR missing: $BENCHMARK_DIR"; exit 1; }
for f in word_sets_v2.json positive_set_v2.json negative_set_v2.json img_metadata.pkl; do
    [ -f "$BENCHMARK_DIR/$f" ] || { echo "ABORT: missing $BENCHMARK_DIR/$f"; exit 1; }
done
[ -f "$CONFIG" ]  || { echo "ABORT: config not found: $CONFIG";  exit 1; }
[ -f "$WEIGHTS" ] || { echo "ABORT: weights not found: $WEIGHTS"; exit 1; }
[ -x "$PY" ]      || { echo "ABORT: no interpreter at $PY"; exit 1; }

$PY - <<'PYEOF'
import sys, torch
print("torch  :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")
if not torch.cuda.is_available():
    sys.exit("ABORT: no CUDA device visible.")
print("deps ok")
PYEOF

if curl -sf --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama: REACHABLE -- llm_descriptor will use the LLM"
else
    echo "Ollama: NOT reachable -- llm_descriptor FALLS BACK to WordNet glosses."
fi

ARGS=(--alpha "$ALPHA" --weighted --config "$CONFIG" --weights "$WEIGHTS"
      --out-dir "$OUT" --category-shard "${SHARD_I}/${SHARD_N}")
[ -n "$LIMIT_IMAGES" ] && ARGS+=(--limit-images "$LIMIT_IMAGES")
[ -n "${SEEN:-}" ]     && ARGS+=(--also-seen-from "$SEEN")

$PY "${SET}_set_experiment.py" "${ARGS[@]}"

echo "DONE shard $SHARD_I/$SHARD_N -> $OUT"
ls -la "$OUT"
echo
echo "When ALL shards finish:  $PY merge_shards.py $OUT"
