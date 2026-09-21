#!/bin/bash
# Benchmark experiments for CAT-Seg -- 5 approaches x 4 variants over the 326-category
# LVIS/ADE20K/PascalVOC benchmark.
#
# USAGE
#   sbatch benchmark_catseg.sbatch positive                 # IoU on positive images
#   sbatch benchmark_catseg.sbatch negative "" 30           # FPR, 30 images/category
#   sbatch benchmark_catseg.sbatch positive <OUT_DIR>       # resume an interrupted run
#
# Resumable: checkpointed on (category, img_id). Resubmit the SAME command with the SAME
# OUT_DIR to continue. summarize()/meta.finish() run only at the end, so a job that hits
# the time limit leaves a complete-so-far detail CSV but no summary.
#
# On resume, pass the SAME arg3 (--limit-images) as the original run: the sample is seeded
# per (seed, category), so a different cap reshuffles the draw and mixes two image sets
# into one CSV.
#
#SBATCH --job-name=bench_catseg
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

SET=${1:?usage: sbatch benchmark_catseg.sbatch <positive|negative> [out_dir] [limit_images]}
ALPHA=${ALPHA:-0.69}        # OVERRIDE ME once alpha_catseg.sbatch has run: CAT-Seg's own
                            # sweep value, not CLIPSeg's. Exported so you can do
                            #   ALPHA=0.8 sbatch benchmark_catseg.sbatch positive
REPO=$HOME/workspace/OVS/taxonomy-driven-ovs-models
EXPDIR=$REPO/model_experiments/catseg
PY=$EXPDIR/catseg_venv/bin/python
OUT=${2:-$HOME/workspace/OVS/results/${SET}_catseg_$(date +%Y%m%d_%H%M%S)}
LIMIT_IMAGES=${3:-}

# Override from the command line without editing this file, e.g.
#   CATSEG_WEIGHTS=/scratch/$USER/model_base.pth \
#   CATSEG_CONFIG=$PWD/CAT-Seg/configs/vitb_384.yaml \
#   sbatch benchmark_catseg.sbatch ...
CONFIG=${CATSEG_CONFIG:-$EXPDIR/CAT-Seg/configs/vitl_336.yaml}
WEIGHTS=${CATSEG_WEIGHTS:-$HOME/workspace/OVS/checkpoints/catseg_model_large.pth}

export BENCHMARK_DIR=$HOME/workspace/OVS/benchmark
export HF_HOME=$HOME/workspace/OVS/hf_cache
export TORCH_HOME=$HOME/workspace/OVS/torch_cache
export NLTK_DATA=$HOME/workspace/OVS/nltk_data
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p "$OUT" "$HF_HOME" "$TORCH_HOME" "$NLTK_DATA"
cd "$EXPDIR"
echo "=================================================="
echo "node   = $SLURMD_NODENAME   job = $SLURM_JOB_ID"
echo "set    = $SET   alpha = $ALPHA (weighted)   threshold = 0.0 (argmax)"
echo "OUT    = $OUT"
echo "  (pass this OUT as arg 2 to resume, with the SAME arg3)"
echo "=================================================="

for f in word_sets_v2.json positive_set_v2.json negative_set_v2.json img_metadata.pkl; do
    [ -f "$BENCHMARK_DIR/$f" ] || { echo "ABORT: missing $BENCHMARK_DIR/$f"; exit 1; }
done
[ -f "$CONFIG" ]  || { echo "ABORT: config not found: $CONFIG"; exit 1; }
[ -f "$WEIGHTS" ] || { echo "ABORT: weights not found: $WEIGHTS"; exit 1; }
echo "benchmark files + checkpoint OK"

$PY - <<'PYEOF'
import sys, torch, detectron2, nltk
print("python    :", sys.executable)
print("torch     :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")
print("detectron2:", detectron2.__version__)
try: nltk.data.find("corpora/wordnet")
except LookupError: nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
if not torch.cuda.is_available():
    sys.exit("ABORT: no CUDA device visible.")
print("deps ok")
PYEOF

if curl -sf --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama: REACHABLE -- llm_descriptor will use the LLM"
else
    echo "Ollama: NOT reachable -- llm_descriptor FALLS BACK to WordNet glosses."
    echo "        Valid run, but report that approach as 'gloss', not 'LLM'."
fi

ARGS=(--alpha "$ALPHA" --weighted --config "$CONFIG" --weights "$WEIGHTS" --out-dir "$OUT")
[ -n "$LIMIT_IMAGES" ] && ARGS+=(--limit-images "$LIMIT_IMAGES")

$PY "${SET}_set_experiment.py" "${ARGS[@]}"

echo "DONE ($SET) -> $OUT"
ls -la "$OUT"
